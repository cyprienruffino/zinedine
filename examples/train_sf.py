"""Self-play with Sample Factory's asynchronous PPO (APPO), for multi-day runs.

Runs in its own venv (Sample Factory pins numpy 1.26 / gymnasium 0.29):
    uv venv .venv-sf --python 3.12 && VIRTUAL_ENV=.venv-sf uv pip install torch sample-factory -e .
    .venv-sf/bin/python examples/train_sf.py --experiment sf1 --num_workers 16 --num_policies 2

Why: SB3's lockstep vec env waits for the slowest of its games every step (resets and CPU jitter), which
cost ~4x of the machine's simulation throughput. Sample Factory collects asynchronously (workers double
buffer their envs, the learner trains while they play), so games never wait for each other.

Self-play is native: each game is a SoccerDuelEnv (both teams learning agents) and with --num_policies N
the policies of the population are mixed in every game (`pbt_mix_policies_in_one_env`), so they play
themselves and each other. Evaluate exported players offline (examples/elo.py with the sf checkpoints).
Any Sample Factory flag works on the command line; the defaults below are this project's.
"""

import functools
import os
import sys
from pathlib import Path

# One BLAS/OpenMP thread per process, set before numpy/torch load so spawned workers inherit it. Frozen
# snapshot opponents (phase 3) run numpy matrix products in every rollout worker, and with the default
# (one thread per core in each of ~20 processes) the machine thrashed at a load average of 145.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import torch  # noqa: E402

# Sample Factory 2.1.1 loads its own checkpoints with torch.load's default, which PyTorch >= 2.6 changed
# to weights_only=True; that refuses the non-tensor values in the checkpoint, Sample Factory swallows the
# error and silently starts from scratch on every resume. Module level, so the spawned learner processes
# (which re-import this script) get the patch too.
torch.load = functools.partial(torch.load, weights_only=False)

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args  # noqa: E402
from sample_factory.utils.utils import str2bool  # noqa: E402
from sample_factory.envs.env_utils import register_env  # noqa: E402
from sample_factory.algo.learning import learner_worker  # noqa: E402
from sample_factory.train import run_rl  # noqa: E402

# Sample Factory pins every learner to 1 torch thread (learner_worker calls init_torch_runtime(cfg) with its
# default max_num_threads=1). With a single policy, that one thread does all the GRU backprop and caps the run
# at ~2.5-3k samples/s whatever the worker count. Give learners --learner_threads instead.
_init_torch_runtime = learner_worker.init_torch_runtime
learner_worker.init_torch_runtime = lambda cfg, max_num_threads=1: _init_torch_runtime(
    cfg, max_num_threads=getattr(cfg, "learner_threads", max_num_threads))

# On resume, Sample Factory takes the learning rate from the checkpoint (curr_lr) and ignores --learning_rate
# (only PBT can change it). Every warm-started run since sf3 kept sf1/sf2's 3e-4 instead of the 1e-4 we asked
# for. With a constant schedule, the command line wins.
from sample_factory.algo.learning.learner import Learner  # noqa: E402
_load_state = Learner._load_state


def _load_state_cfg_lr(self, checkpoint_dict, load_progress=True):
    _load_state(self, checkpoint_dict, load_progress)
    if self.cfg.lr_schedule == "constant" and self.curr_lr != self.cfg.learning_rate:
        print(f"learning rate from checkpoint {self.curr_lr} replaced by --learning_rate {self.cfg.learning_rate}")
        self.curr_lr = self.cfg.learning_rate
        self._apply_lr(self.curr_lr)


Learner._load_state = _load_state_cfg_lr

from soccer_rl.duel import SoccerDuelEnv, SoccerVsPoolEnv  # noqa: E402

ENV = "soccer_duel"


def steps_done_per_env(cfg) -> int:
    """How far a resumed run already is, per env: each policy's latest checkpoint name ends with the
    samples it trained on; two samples (one per team) make one env step, spread over all envs."""
    exp = Path(cfg.train_dir) / cfg.experiment
    samples = 0
    for pdir in exp.glob("checkpoint_p*"):
        names = [int(c.stem.split("_")[-1]) for c in pdir.glob("checkpoint_*.pth")]
        samples += max(names, default=0)
    return samples // 2 // (cfg.num_workers * cfg.num_envs_per_worker)


def make_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    if cfg.opponent_pool:
        # Phase 3: one learner per game against frozen snapshots; envs alternate sides.
        env_id = getattr(env_config, "env_id", 0) if env_config is not None else 0
        return SoccerVsPoolEnv(cfg.opponent_pool.split(","), side=("p1", "p2")[env_id % 2],
                               recent_frac=cfg.opponent_recent_frac, seed=env_id,
                               obs="v2", history=cfg.history, frame_skip=cfg.frame_skip, episode="goal",
                               max_steps=cfg.max_episode_steps, draw_penalty=cfg.draw_penalty,
                               shaping=cfg.shaping, time_feature=cfg.time_feature,
                               modifiers=cfg.modifiers, modifier_feature=cfg.modifier_feature)
    return SoccerDuelEnv(obs="v2", history=cfg.history, frame_skip=cfg.frame_skip, episode="goal",
                         max_steps=cfg.max_episode_steps, draw_penalty=cfg.draw_penalty,
                         switch_cost=cfg.switch_cost, shaping=cfg.shaping, anneal_steps=cfg.anneal_env_steps,
                         anneal_start=steps_done_per_env(cfg), time_feature=cfg.time_feature,
                         modifiers=cfg.modifiers, modifier_feature=cfg.modifier_feature)


def add_env_args(parser):
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--max_episode_steps", type=int, default=2000)
    parser.add_argument("--draw_penalty", type=float, default=0.3)
    parser.add_argument("--switch_cost", type=float, default=0.0, help="reward cost per new press (anti-mashing)")
    parser.add_argument("--shaping", type=float, default=0.0)
    parser.add_argument("--modifiers", type=str, default="none", choices=["none", "random"],
                        help="match modifiers: classic only, or rolled per episode with the game's own probabilities")
    parser.add_argument("--modifier_feature", type=str2bool, default=False,
                        help="append the episode's modifiers to the observation (7 values per frame)")
    parser.add_argument("--learner_threads", type=int, default=4,
                        help="torch threads per learner process (Sample Factory's default is 1)")
    parser.add_argument("--opponent_pool", type=str, default="",
                        help="comma-separated snapshot dirs: play frozen past opponents instead of self-play duels")
    parser.add_argument("--opponent_recent_frac", type=float, default=0.5,
                        help="draw opponents from this newest share of the pool (0.5 = newest half)")
    parser.add_argument("--time_feature", type=str2bool, default=False,
                        help="point clock in the observation (steps since kickoff), so agents see the draw penalty coming")
    parser.add_argument("--anneal_env_steps", type=int, default=0,
                        help="fade switch cost and shaping to 0 over this many steps of each env (0 = keep)")


def parse(argv):
    if "--env" not in argv:
        argv = ["--env", ENV] + argv  # Sample Factory requires --env before defaults apply
    parser, _ = parse_sf_args(argv=argv)
    add_env_args(parser)
    parser.set_defaults(
        env=ENV, train_dir="runs/sf", device="cpu",
        num_workers=16, num_envs_per_worker=2, worker_num_splits=2,  # double buffering: no idle workers
        rollout=128, batch_size=8192, num_batches_per_epoch=1, num_epochs=4,
        gamma=0.999, gae_lambda=0.98, learning_rate=3e-4, exploration_loss_coeff=0.003,
        ppo_clip_ratio=0.2, max_grad_norm=0.5, value_loss_coeff=0.5,
        use_rnn=False, encoder_mlp_layers=[256, 256], nonlinearity="tanh",
        normalize_input=False,  # obs v2 is already normalized with measured statistics
        num_policies=2, with_pbt=False, pbt_mix_policies_in_one_env=True,
        train_for_env_steps=10_000_000_000, save_every_sec=600, keep_checkpoints=5,
        decorrelate_envs_on_one_worker=True, heartbeat_interval=60, with_wandb=False,
    )
    return parse_full_cfg(parser, argv)


def main():
    register_env(ENV, make_env)
    cfg = parse(sys.argv[1:])
    return run_rl(cfg)


if __name__ == "__main__":
    sys.exit(main())
