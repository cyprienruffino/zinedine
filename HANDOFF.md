# Handoff: Soccer Physics RL tooling

Status as of 2026-11-02 morning, after the first overnight training session on the new machine
(Intel Core Ultra 7 366H, 16 cores, 30 GB). `README.md` covers usage, architecture, the wire protocol
and gotchas; this file covers state, measurements, findings and what's next.

## Goal and preferences

Cyprien wants RL tooling for *Soccer Physics* (Unity 2022.1.7f1, Mono) via BepInEx, then a strong
agent.

- **The policy is single-button** (`two_buttons=False`, `Discrete(2)`).
- **Keep scope tight.** Propose before adding big features. Cyprien delegated the training
  curriculum overnight ("I trust you").
- **No sound.** Launched games must be silent (now enforced, see below).

## What exists

| Path | What |
|---|---|
| `SoccerPhysics/` | Game with BepInEx 5.4.23.6 |
| `plugin/SoccerRL/` | Plugin source (`Bridge.cs`, `Patches.cs`, `Introspect.cs`, `Plugin.cs`); build with `scripts/build_plugin.sh` |
| `soccer_rl/` | `client.py`, `launcher.py` (`GameProcess`), `env.py` (`SoccerPhysicsEnv`), `selfplay.py` (snapshots, PFSP, press prior) |
| `soccer_rl/obs_stats_v2.npz` | Feature mean/std used by `obs="v2"`; regenerate with `scripts/obs_stats.py` |
| `examples/train_ppo.py` | PPO vs built-in AI / idle / random; flags for every hyperparameter, `--press-prior`, `--press-cost` + `--anneal`, `--obs`, `--history` |
| `examples/train_selfplay.py` | Self-play PPO with a PFSP snapshot pool, plus `--builtin-envs` and `--idle-envs` as yardsticks |
| `examples/evaluate.py` | Parallel evaluation vs `--opponent builtin/idle/random`, ± margin |
| `examples/duel.py` | Head-to-head between two models (`path[:obs[:history]]`), alternating sides |
| `runs/night/` | Overnight runs: `<name>_logs/progress.csv`, `<name>_ckpt/`, `<name>_pool/`; `status.py` summarises them, `launch.sh` starts a detached run |
| `runs/night/final/` | Final checkpoints of the best self-play runs, `eval.txt`, `duel.txt` |

Not a git repo yet; nothing is committed.

## Changes made in this session

- **Single reload per episode**: `auto_reload` setting; goal-mode envs turn off the game's own
  post-goal reload (it was immediately overwritten by `reset()`). 58 → 32 scene loads per 30 episodes.
- **Silence**: the plugin mutes from `Awake`, and the launcher makes PulseAudio/ALSA unreachable
  (`sound=False` default), so games open no audio stream at all.
- **Hidden jump timers exposed**: `jumpLimiter` (10-tick cooldown after a jump) and `jump2delay`
  (second boost after 3 ticks) are sent as player fields 5–6 (protocol change: 6 fields per player).
- **`obs="v2"`**: v1 + the timers, normalized with measured mean/std (leg heights were scaled ~10×
  too small; the ball never rotates, so its spin and cos(angle) are constants).
- **`history=n`**: frame stacking inside the env; policy opponents get their own stacked view.
- **Opponent hooks**: env calls `opponent.reset()` each episode; `info["opponent_id"]`;
  `set_shaping` / `set_press_cost` for annealing through SB3 wrappers (`set_attr` stops at Monitor).

## Measurements (new machine)

| Measurement | Result |
|---|---|
| Raw throughput | 1 game 5,988 ticks/s; 16 games 28,416 ticks/s total (7,104 env-steps/s at frame_skip 4) |
| SB3 `SubprocVecEnv`, 16 envs, no learning | 2,792 steps/s: lockstep means every step waits for whichever game is reloading |
| PPO, 16 envs | ~2,200 steps/s (torch threads don't matter); 32 envs barely better |
| 3–4 runs × 16 envs side by side | ~1,000–1,100 steps/s each (~4,500 total): the way to use the CPU |

## Key findings

1. **Never pressing beats the built-in AI**: idle wins ~49%, loses ~13% (goal diff +0.36–0.40 per
   episode). An unpressed player stays upright like a wall; one press tilts the players 25–47° for
   seconds. Pressing on just 0.5% of steps drops it to 37% / 57%. Random wins 36% / loses 52%.
   **The built-in AI is a broken benchmark**: no heuristic or PPO run beat idle against it.
2. **PPO is bimodal here**: starting near idle (`--press-prior 0.0005`) collapses to pure idle;
   anything above ~0.2% drifts to pressing on ~40% of steps. A per-press reward cost (even 0.01 per
   step) didn't stop the drift.
3. **Long horizon needed**: idle wins arrive late (median 509 steps at frame_skip 2), so use
   gamma 0.999 and GAE λ 0.98.
4. **Self-play with an idle yardstick works but plateaus**: starting from an active PPO policy,
   self-play went from about −0.3 to +0.09 goals per episode against a standing opponent within
   ~3M steps, then flattened. A lower learning rate (1e-4) didn't help.

## Final results (400 episodes each, argmax actions unless noted)

| Policy | vs built-in AI won / lost | vs idle won / lost |
|---|---|---|
| Idle | 49% / 13% | — |
| `sp3_from_r10` (self-play, 12.8M steps) | 39.5% / 35.5% | 41.2% / 32.5% |
| `sp7_lr1e4` (sp3 continued at lr 1e-4) | 37.2% / 43.5% | 43.0% / 34.3% |
| `sp6_from_r10_diverse` | 33.2% / 43.5% | 38.2% / 35.0% |

Head-to-head (300 episodes per pair, `runs/night/final/duel.txt`): all three are indistinguishable
(48.9–50.7% of decided games), so the self-play runs converged to similar strength.

## Day 2 (2026-11-02): research and preparation (no training during the day)

Cyprien's direction: **self-play is the way**; **no macro actions** (keep full per-step control, fix
behaviour through rewards and exploration instead); **draw penalty** ("the goal is to win, not to draw");
question the architecture and algorithm instead of assuming them.

Diagnosis: per-step independent sampling explores with scattered taps, the worst behaviour in this game.
Literature notes:
- Correlated (pink) action noise beats white noise for exploration (Eberhard et al., ICLR 2023). For
  PPO, noise between white and pink (β≈0.5) worked best (Hollenstein et al., AAAI 2024). Both papers
  cover Gaussian policies only; for our Bernoulli policy, sampling by inverse CDF from a correlated
  uniform keeps each step's marginal exact.
- Bansal et al. 2018 (competitive self-play): dense rewards linearly annealed to 0 over 10–15% of
  training beat agents that kept them; opponents drawn from the recent half of history worked best.
- Slime Volleyball Gym: PPO self-play beat directly trained PPO head-to-head yet scored worse against
  the fixed baseline. Fixed yardsticks mislead, so use Elo (as Unity ML-Agents does).

New code:
- `soccer_rl/noise.py`: `CorrelatedPolicy` (SB3 policy, `--noise-beta`), `CorrelatedUniform`.
  Verified: exact press rates; at β=1 and P=0.05, half as many separate presses, twice as long.
  Learns CartPole and survives save/load.
- Env: `draw_penalty` (any goalless end, timeout or ball out, so kicking the ball out can't dodge it),
  `time_feature` (episode clock in the obs). **Fix**: ball-out and visible timeouts are now terminal;
  SB3 used to bootstrap past them as if play continued.
- Self-play: `--noise-beta` (opponents sample with the same noise), `--draw-penalty`,
  `--time-feature`, press-rate logging. `--init` now copies weights into a freshly configured model,
  so sampling and hyperparameters can change.
- `examples/elo.py`: round-robin between snapshots from one or more pools (plus idle and random) and a
  Bradley-Terry fit, anchored at idle = 1000.

## Night 2 plan (`runs/tonight.sh`, runs under `runs/n2/`)

| Run | Start | Noise β | Other |
|---|---|---|---|
| A_pink | sp3 | 1 | draw penalty 0.3 |
| B_red | sp3 | 2 | draw penalty 0.3 |
| C_pink_scratch | scratch | 1 | draw penalty 0.3, shaping 0.02 fading over 10M |
| D_white | sp3 | 0 | draw penalty 0.3 (control for the noise) |

Decision rules, applied without waiting for Cyprien:
- **Every ~2 h**: `examples/elo.py runs/n2/*_pool runs/night/sp3_from_r10_pool --max-players 4
  --games 12 --procs 8`.
- **Replace the 2 lowest runs** (by the Elo of their newest snapshot) with variants of the best: a
  larger rollout (`--n-steps 2048`), draw penalty 0.5, lr 1e-4, or a bigger network from scratch
  (`--net 512,512`).
- **Watch `press_rate`** for collapse (→ 0) or saturation.
- **Stop all training in the morning** before Cyprien's workday.

## Night 2 results (2026-11-02 12:30 → 2026-11-03 08:00)

Findings, in order:
1. **PPO with obs v2 was thrashing**: approx_kl 0.13–0.16 per update (healthy ~0.01); every v2 run had
   it, the v1 runs didn't. Fixed with `--target-kl 0.02` (KL now ~0.025–0.03). Always check KL first.
2. **The plain pipeline learns, slowly**: PPO against a fixed random opponent went 40% → 54% won in 2M
   steps, then plateaued.
3. **Standing still is the best scripted strategy against *every* opponent tried**: against random,
   idle wins 55.7% / loses 29.3%, better than PPO (54% / 37%) and every heuristic. Likely mechanism:
   two regimes. A player who never pressed is an upright wall (low, heavy centre of mass); after one
   press they're tilted for seconds and pressing is the only way to move. So pressing begets pressing,
   and every PPO run drifts to ~40–50% presses, even from a 1% prior.
4. **Correlated (pink/red) noise**: correct marginals, longer press runs, a small edge at best
   (`E_pink` slightly above `F_white` in Elo, within noise).
5. **All PPO variants hit the same ceiling** (~52% won / 30% lost vs idle, Elo ~1100 with idle =
   1000): frame_skip 4, a 512×512 network, near-idle priors, bigger batches with lower lr. A 1,200-game
   test confirms the skill is real but small: `E_pink` after 33M steps rated 1054 vs 971 for its
   untrained net (+83 Elo, ~62% expected score).
6. **ES self-play** (`examples/train_es.py`, antithetic pairs on common random numbers, centred ranks,
   argmax play): from scratch it stalled in draws; from a PPO snapshot it improved its score against
   idle (+0.19 → +0.33 per game), but in Elo it stays within noise of PPO (~1080–1120).

Final ranking (60 games per pair, 300 games per player, noise ~±20; `runs/final2/elo_final.txt`):
es4_bigpop 1091, PPO E_pink 1090, es5_sigma05 1085, es1_from_E 1059, random 1022, idle 1000. PPO and ES
are tied; the best agents are ~+90 Elo over idle (~63% expected score).

Tooling added: `--target-kl`, `env.set_side()`, `examples/train_es.py` (workers relaunch dead games),
`elo.py --argmax-pools`, `runs/n2/memguard.sh` (stops a run when RAM is low).

Gotchas: `TaskStop` on a shell that launched `setsid nohup` runs kills those runs too, so launch them
from a command that exits immediately. `pkill -f <pattern>` inside a command whose own text contains
the pattern kills that command; use `[t]rain_...` patterns or a script file.

## Day 3 (2026-11-03): human check

Cyprien ("terrible at this game") beat the best ES agent 5–3, the same score as against the scripted
pulse rhythm. His remark: **the agents mash the button and rarely hold long enough for full jumps**.
Measured: 68–76% of their presses last one decision (2 ticks), too short for the second jump boost
(3+ ticks held); only 3–8% last 4+ decisions. Mashing is what PPO and ES start from (50% presses
sampled independently) and they never unlearn it.

Prepared for night 3 (`runs/n4.sh`, nothing running yet): an annealed **switch cost** (reward cost per
new press, an action-rate penalty; `--switch-cost`, `--switch-anneal`) plus red/pink exploration
noise, from scratch. Logs now include `rollout/tap_share` and `rollout/hold_mean`. `elo.py` adds
`pulse` as a reference player (holds 6, releases 6), and `watch.py --snapshot X.npz [--argmax]` plays
self-play snapshots.

## Day 3: switch to Sample Factory for multi-day runs

Cyprien: training sessions were far too short for a game with long delays between rewards; **count run
times in days**. Profiling showed SB3's lockstep wastes most of the machine. One env steps in 1.3 ms, a
16-env lockstep step takes ~5 ms (each step waits for the slowest game; resets 24 ms add ~20%). Raw
capacity is ~14k env-steps/s at frame_skip 2; three SB3 runs used ~3.1k samples/s together.

**Sample Factory (APPO, asynchronous)** in its own venv (`.venv-sf`, Python 3.12; it pins numpy 1.26 /
gymnasium 0.29):
- `soccer_rl/duel.py`: `SoccerDuelEnv`, both teams are learning agents. Zero-sum goals, per-agent
  switch cost, draw penalty for both, per-env annealing, win/tap/hold statistics per episode.
- `examples/train_sf.py`: population self-play (`--num_policies N`, policies mixed in every game),
  double-buffered workers, CPU device.
- `scripts/export_sf.py`: checkpoints → numpy snapshots for elo/watch/SB3 (verified against SF's
  forward pass, max difference 3e-7).
- `runs/sf1.sh`: the long run (4 policies, 10 workers × 2 games, ~7k samples/s) under a
  **supervisor** (`runs/sf/sf1.supervise.sh`, log `runs/sf/sf1.supervisor.log`). It stops training
  below 3.5 GB of free RAM and resumes from checkpoints above 9 GB, so the run yields to Cyprien during
  the day and comes back on its own. It also keeps the exporter running (pools every 30 min,
  `runs/sf/sf1/pool_p*`). Rerunning `runs/sf1.sh` is safe (it won't start a second supervisor).
- Day-1 adjustments: switch cost 0.005 → 0 at 16:20 (it caused hold-forever play: mean holds of
  75–500 decisions, draws up to 60%), draw penalty 0.3 → 0.6. Draws fell to 7–25% within an hour, but
  mashing came back (taps 54–63%). Whether mashing is actually weak is open: `pulse` (no mashing)
  rates *below* the mashing agents (966 vs 1048).
- Watch stats with `.venv-sf/bin/python scripts/sf_stats.py runs/sf/sf1` (win, draw, taps, holds
  per policy).
- Reward scheme of sf1, per agent: goal ±1 (zero-sum), draw −0.6 for both (timeout after 2,000 steps
  or ball out), shaping 0.02 × ball progress fading over each env's first 4M steps, switch cost 0.
  The shaping fade used to restart with every supervisor restart (each env counted its own steps
  from 0), so it never fully faded before day 2. **Fixed** (day 2): `train_sf.py` recovers per-env
  progress from the checkpoint sample counts (`steps_done_per_env`), so the fade resumes. Applies to
  sf1 from its next restart; it's already past the fade, so shaping stays 0.
- Mashing: Cyprien agrees not to prevent it if it wins (the agents mash at ~50–80% taps with no loss
  of Elo).
- **Queued for after sf1** (Cyprien: no hurry): a run with a lower draw penalty (e.g. 0.3), since 0.6
  may push too much risk-taking.
- Day 3: Cyprien watched p0 vs p2 (`watch.py --snapshot ... --opponent-snapshot ...`, p0 won 5–3) and
  saw two modes: "maintain" near their own goal, "mash" near the opponent's. Measured on p0 (30 games):
  near its own goal it holds the button 87–89% of the time with 19–29-step holds; near the opponent's
  goal it presses 52–59% with 2–3-step holds (22–23 new presses per 100 steps). Midfield stalemates
  happen when both sides are in maintain mode; the ball drifting on its own breaks them.
- **sf2 (GRU) started day 3, 16:06.** After 6 h of training (109M samples) its best policy matched
  sf1's best; after 12 h (216M) it was level with sf1's best (1158 vs 1155 within one round) with ~10×
  fewer samples per policy, and still climbing. Day 4: sf2 has ~6 GB of fixed RAM overhead (4 GRU
  learners ~0.9 GB each plus inference workers), so its supervisor (`runs/sf2.sh`) only starts it when
  ≥ 4 workers fit above the 3.5 GB floor (~11 GB free). During Cyprien's workday it mostly stays
  paused. Progress checks use checkpoint sample counts: Sample Factory's log counter restarts each
  session.
- **Bug found day 4 (fixed): Sample Factory never resumed.** PyTorch ≥ 2.6 loads with
  `weights_only=True` by default, which refuses SF 2.1.1's checkpoints; SF catches the error and logs
  "Did not load from checkpoint, starting from scratch!". Every supervisor restart of sf1 (47 failed
  loads) and sf2 therefore started from zero, and the new low-counter checkpoints were deleted right
  away by `keep_checkpoints` (sorted by name). Results are still valid per session: the longest sf1
  session ran from day 2 00:21 to day 3 15:46, and sf2's 6 h/12 h results came from one session (day 3
  17:51 → day 4 09:49). sf2's sessions from 09:50 to 20:10 were all from scratch and are lost. **Fix**:
  `train_sf.py` patches `torch.load(weights_only=False)` at module level (the spawned learners
  re-import it); verified on a copy: all 4 policies load (36.9M / 67.9M / 68.6M / 56.1M samples).
  After any restart, check the log for "Loaded experiment state", not "Did not load".
- Day 6: sf2 sat idle for 2 days (its supervisor waited for RAM that never freed up), so it was cut to
  2 policies (p0, and p2's weights in slot p1; old p1/p3 in `runs/sf/sf2/archive_p1_p3/`) with a
  14-worker cap. A watchdog (background check) now wakes me after 30 min of downtime or a failed
  resume.
- **sf3 (day 6, 22:52): sf2 plus a point clock.** Cyprien saw both agents sit in "maintain mode" for a
  whole point. The env's `time_feature` is now a point clock (steps since kickoff / 2000, restarting
  after each goal, so it also works in full matches); the duel env shares it with P2's view. sf3 was
  warm-started from sf2's latest checkpoints (p0 52.3M, p1 88.7M) with zero weights on the clock
  inputs (`scripts/expand_sf_obs.py`, which also expands the Adam moments): identical play at start
  (max logit difference 1e-6). It then has to learn to use the clock. `runs/sf3.sh`.
- `watch.py` match mode: a step landing in the reload after a goal no longer crashes the env (the last
  complete frame is reused).
- `elo.py` can mix players with and without the clock (`--time-feature`; it strips the clock columns
  for 780-input players). Reference players for sf3 rounds: `runs/sf/refs/` (sf2 p2 at 63M, sf1 p0,
  PPO E).
- sf3 progress (one uninterrupted session since day 6, 22:52): after 6 h no clear change; after 12 h
  (322M samples, day 7, 08:53) **sf3 p1 at 181M = 1174, best ever**: +72 over its warm start and ~+50
  over sf2's best in the same round. sf3 p0 at 138M = 1164. Draws are still 11–18% of training
  games, so it's not yet clear the gain comes from the clock rather than from more training.
- Day 7, 20:00 decision round (24 games per pair, about ±17 per player, `runs/sf/elo_sf3_decision.txt`):
  sf3 p1 1073 → 1106 (89M → 277M), p0 1103 → 1070 (52M → 224M), sf1 p0 1088, PPO E 1040, sf2 p2 1030.
  The 1174 at 12 h was mostly noise. sf3 was flat for ~a day and draws didn't fall (~20%), so the clock
  alone didn't break stalemates. **Change: learning rate 3e-4 → 1e-4** (same experiment, resumed at
  283M / 228M samples; SF applies the new constant rate even though the checkpoint stores the old one).
- The "lower draw penalty" variant is on hold: it would make stalling cheaper, the opposite of the
  stalemate goal.
- **Curriculum** (agreed with Cyprien, day 7): (1) consolidate at lr 1e-4; (2) stalemate pressure, with
  the point timeout shortened stepwise 2000 → 1500 → 1000 steps; (3) a frozen-snapshot opponent pool
  against forgetting (harder in SF: frozen opponents' moves would be recorded as the learner's, so it
  needs a separate setup); (4) random game variants. Each phase starts from the previous one's best
  snapshot, picked in a 24-games-per-pair round.
- Phase 1 result: flat. Rounds at 01:30 and 06:41 on day 8 put every sf3 snapshot within 1086–1129 (sf1
  p0 ~1095, sf2 p2 ~1075, PPO E ~1020–1070), with draws at ~19–20%. **Phase 2 started day 8, 06:42**:
  `--max_episode_steps 1500` (60 s per point) on sf3's latest checkpoints (378M / 318M samples).
- Phase 2 at 1500 steps: draws unchanged (p0 20.7% → 19.3%, p1 23.7% → 23.7% over 10 h) while points got
  shorter (538 → 431 and 636 → 492 steps); Elo held (latest 1084–1091, `runs/sf/elo_sf3_ph2.txt`). In
  120 headless games, ~60% of draws were timeouts and ~40% ball-outs. Cyprien: keep timeouts and
  ball-outs at the same penalty (a cheaper ball-out would invite kicking the ball out on purpose).
  **Day 8, 17:22: timeout 1500 → 1000** (40 s), resumed at 402M / 475M samples.
- Phase 2 at 1000 steps: draws rose first (p0 19.6% → 25%, p1 23.5% → 28%), then the agents adapted
  (19.9% / 22.4% after ~21 h): no more draws despite 20 s less per point, so they score faster. Elo held
  (p1 1139 at 703M). By day 10 sf3 had plateaued: the latest snapshots were back at their warm-start
  level (p1 1094, p0 1115, `runs/sf/elo_sf3_d10.txt`).
- **Phase 3: sf4 (day 10, 10:32)**: the same 2 GRU policies (from sf3's latest, 711M / 804M samples)
  play frozen snapshots in single-agent games (`SoccerVsPoolEnv` in `soccer_rl/duel.py`,
  `--opponent_pool` in `train_sf.py`). Opponents come from the newest half of sf3's pools plus sf4's
  own exports; `SnapshotOpponent` takes several dirs and has `recent_frac`. `runs/sf4.sh`.
  **Gotcha:** the numpy opponents made OpenBLAS start one thread per core in every worker (load
  average 145, ~800 samples/s). `train_sf.py` now sets OMP/OPENBLAS/MKL threads to 1, giving
  ~3,100 samples/s (one sample per game step).
- Day 10 evening: the laptop's other workloads left too little RAM for 2 GRU learners,
  so **sf4 runs 1 learning policy** (p1's line in slot p0; the frozen pool provides the diversity;
  old p0 in `runs/sf/sf4/archive_p0/`). It starts at ≥ 6.4 GB free, with ≥ 2 workers. The watchdog
  only counts downtime when there's RAM for the run.
- **Day 11, 08:40: first gain since sf3's plateau.** sf4 at 899M samples = 1150, +46 over itself at
  849M (1104) and +35 over sf3's best (1115) in the same round (`runs/sf/elo_sf4_2.txt`, 24 games
  per pair, ±22). Needs confirmation in the next round.
- **Not confirmed** (day 12, `runs/sf/elo_sf4_3.txt`): the same 899M snapshot rated 1061, and the latest
  (950M) 1090 vs 1087 at the switch. The +46 was a ~2σ fluke. Lesson: confirm any single +2σ result
  before reporting it as progress. Classic-mode strength has been flat for ~a week across phases 1–3
  (best agents ~60–100 above idle).
- **Phase 4: sf5 (day 12, 14:17)**: sf4 at 1.0B samples, plus random match modifiers rolled per episode
  with the game's own probabilities and visible in the observation (`modifier_feature`: 7 values per
  frame after the clock; frame 196 → 203, obs 812). Warm-started with zero weights on the new inputs
  (max logit difference 4e-6). Headless matches keep all 25 bodies (no observation change).
  `SnapshotOpponent` and `elo.py` trim each frame for older snapshots (784/780 inputs); `elo.py` has
  `--modifiers random --modifier-feature` to rate players on random variants. `runs/sf5.sh`.
- Day 13 throughput fixes: Sample Factory pins each learner to 1 torch thread (patched: `--learner_threads`,
  default 4), `--num_epochs 2`, and back to 2 policies (two learners in parallel). Supervisors now also
  *grow* a running run when RAM frees up (restart once ≥ 4 more workers fit).
- sf5 results (days 13–15): **first confirmed classic gain.** Over three 30-games-per-pair rounds the
  later p1 snapshots beat sf5's start by ~+30–50 and sf3's best by ~+30–60 (best: p1 at 1.22B samples,
  1131 classic). On random variants, the whole frozen-opponent line (sf4/sf5, including its start)
  rates ~30–40 above sf3/sf1, but the modifier inputs add nothing measurable yet.
- Throughput by setup: MLP duels (sf1) ~8k samples/s; GRU duels (sf2/sf3) ~5k; GRU vs frozen opponents
  (sf4/sf5) ~3.4k (1 sample per game step, plus a numpy opponent per worker); much lower when RAM-limited.
- **sf6 (day 15, 01:24): back to self-play duels** from sf5's latest checkpoints (1.28B per policy),
  same obs and settings: **~7,000–7,400 samples/s** on 14 workers (CPU-bound, load ~32). Experiment:
  does it match sf5's gains at equal training time? If newer snapshots start losing to older ones,
  bring frozen opponents back. `runs/sf6.sh`.
- Day 15, sf6 after 12 h of duels: round-to-round noise (snapshots swinging ±30–40 between rounds) made
  the broad rating rounds useless for decisions, so I switched to **600-game head-to-heads**
  (`runs/sf/h2h_sf6_*.txt`, ±14). Result: sf6 latest (1.435B), sf6's start (= sf5 final) and sf5's
  champion (p1 at 1.22B) are **equal within 10 Elo**, classic and random variants. Both duels and
  frozen opponents have plateaued at this level.
- `runs/sf7.sh`: frozen opponents from sf5's final weights, with sf6's snapshots and the exploiter's
  (`runs/sf/sf8_exploit/pool_p0`) added to the pool. **Started day 18, 22:00** (resumed at 1.279B/1.280B).
- **sf8_exploit (day 15, 16:42):** league-style exploiter. One policy trained only against the frozen
  champion (`runs/sf/champion_pool/`: sf6 p0 at 1.435B), warm-started from sf6 p0's latest. Its
  training win rate against the champion measures the champion's exploitable headroom.
  `runs/sf8_exploit.sh`.
- Exploiter after ~9 h: win 35% → 39%, loss 37% → 35% (draws ~27%). The champion has only small
  exploitable weaknesses. Sanity check: the champion against itself through `SoccerVsPoolEnv` comes out
  roughly even (37% / 32% over 100 games; P2 does somewhat better), so the frozen-opponent env isn't
  biased. **Gotcha:** in Sample Factory's summaries, a tag matching "loss" can be a training-loss scalar,
  not the game loss rate; use 1 − win − draw.
- **Learning-rate bug (found day 19):** on resume, Sample Factory restores the learning rate from the
  checkpoint and ignores `--learning_rate`. sf1/sf2 trained at 3e-4, so every warm-started run since (sf3–sf8)
  ran at 3e-4, not the 1e-4 in its script. `examples/train_sf.py` now forces the command-line value; check
  `train/lr` in TensorBoard after a restart. sf7 runs at a real 1e-4 from 1.339B/1.338B (day 19, 14:30).
- sf7's supervisor thrashed (70 stop/starts overnight, ~2k samples/s instead of ~3.4k): with the larger
  opponent pool, a worker takes ~0.75 GB, not 0.45. `runs/sf7.sh` now sizes at 700 MB per worker.
- **Exploiter verdict (day 18, after ~29 h, 1.556B):** win 37%, loss 36%, draw 27%; the edge never grew
  (9 h: 39/33, 27 h: 41/40). The champion has no exploitable weakness this method can find. Stopped;
  its snapshots now feed sf7's pool for diversity.
- Supervisor restarts can fail with "Components take too long to start" (28 games plus learners booting at
  once on a loaded machine); the supervisor retries and the next attempt succeeds.
- A 1-policy run (exploiter) is single-learner-bound: ~1,500–4,000 samples/s.
- Queue after sf2 (GRU only, for a clean comparison): add `time_feature` (episode clock), so agents
  can see the draw penalty coming and break stalemates on purpose.
- Day 2, 17:00 Elo (3 snapshots per policy): p2 1082 → 1082 (71M → 168M samples), p0 1058 → 1058,
  p1 1043 → 1027, only p3 still climbing (1027 → 1076). PPO E 1064, built-in AI 1034 (it swings ±50
  between rounds), idle 1000. Three of four policies are flat since ~50–150M samples.
- **Prepared, not started: `runs/sf2.sh`**, sf1 plus a GRU core (256 units, recurrence 64). GRU support
  is end to end: `scripts/export_sf.py` exports it, `NumpyPolicy` runs it (matches Sample Factory over a
  20-step sequence to 1e-7), and `elo.py`, `watch.py` and `SnapshotOpponent` reset the hidden state each
  episode. Decision rule: if sf1's top is still ~1090 on the morning of day 3, stop sf1 and start sf2.
- Day 2, 23:25 Elo: **sf1 is still improving.** Within the round: p0 1075 → 1103 (99M → 207M
  samples), p2 1048 → 1077, while PPO E rated 993 (best sf1 policy +110, ~65% expected score). At 10:55
  the gap was ~0. Ratings drift ±50–90 *between* rounds (PPO E: 1080 → 993 with no change to it), so
  compare players within one round only. **Revised rule**: keep sf1 while its lead over PPO E and
  over its own earlier snapshots keeps growing; switch to sf2 after a full day without growth.

The SB3 path still works (`.venv`); `soccer_rl.noise.CorrelatedPolicy` now imports SB3 lazily.

## Later ideas

0. **The open question: is the ceiling the game or the methods?** A human baseline would settle it:
   Cyprien playing the best agent with `examples/watch.py --human`. If he wins easily, the methods are
   missing skill that exists. Ideas for that case: a recurrent policy to track momentum, behaviour
   cloning from his games, or a league with exploiters (AlphaStar-style).

1. **Throughput**: an async vec env (step whichever games are ready) would recover the ~60% lost to
   lockstep reloads.
2. **Credit assignment**: frame_skip 4, larger batches, a distributional critic.
3. **Bigger networks**: residual MLPs with LayerNorm if the plain MLP saturates.
4. **Random modifiers** (rain, ball types, ...) with the modifier settings added to the obs.

## Gotchas that cost time (details in README)

- **Sandbox**: the game can't run inside Claude Code's sandbox (steam-run's bwrap can't nest, and
  `file` / the game die with SIGSYS); game-launching commands need the sandbox disabled.
- **zsh doesn't word-split `$var`**: multi-word flags in a variable arrive as one argument (use
  `${=var}`).
- **SubprocVecEnv scripts need `if __name__ == "__main__":`** (forkserver start method).
- **`uv` cache**: in the sandbox set `UV_CACHE_DIR` to a temp dir (`~/.cache/uv` is read-only).
- **TMPDIR differs inside and outside the sandbox**; use the scratchpad path for files shared between both.
- **Windowed game freezes after one tick** without `runInBackground` (the plugin forces it on).
- **Logs disappear under `steam-run`** (private `/tmp`); the launcher captures stdout instead.
- **The built-in AI only drives P2**, so `opponent="builtin"` requires `side="p1"`.
