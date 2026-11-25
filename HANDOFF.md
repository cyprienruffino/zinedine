# Handoff: Soccer Physics RL

Training was stopped on **2026-11-25 at 15:00 (Paris)** so it can resume on another machine.
`README.md` covers setup, usage, architecture, the wire protocol and game gotchas. This file covers
the current state, how to resume, the training setup, results, findings and what's next, followed by
the full history.

Day numbers in the history are dates in November 2026: day 1 = 2026-11-01, day 24 = 2026-11-24.

## 1. Current state

- **Latest run: sf7**, stopped cleanly at 15:00 with final checkpoints at **p0 2.271B / p1 2.302B samples**
  (`checkpoint_000820178_2270625792.pth`, `checkpoint_000827768_2301714432.pth`). Nothing is running.
- **Champion: sf7 p0 at 2.108B samples**, `runs/sf/champion_pool/champion_sf7_p0_2108M.npz`.
  It beat the previous champion (sf6 p0 at 1.435B) by +26 to +34 Elo in three 12,600-game
  head-to-heads (§4).
- **Why sf7 improved:** a learning-rate bug was fixed on day 19. Every warm-started run since sf3
  had trained at 3e-4 instead of the 1e-4 in its script (§5.1). This is the first confirmed gain
  since sf5.
- **Final check (day 25, 12,600 games, classic):** sf7 p0 at 2.244B = 1121, the champion = 1118,
  sf7 p1 at 2.273B = 1117, the old sf6 champion = 1090. sf7 has **flattened** since 2.108B (+3, within
  ±14). Per §6, the next step is lowering the learning rate to 3e-5.
- **Code:** everything is pushed to https://github.com/cyprienruffino/zinedine (`main`).
- **Training data:** `runs/` is git-ignored. To resume, the checkpoints and opponent pools are in
  the GitHub release **sf7-final-2026-11-25**.

## 2. How to resume on a new machine

1. Clone the repo and do the README's **Setup** section: game copy in `./SoccerPhysics/`,
   `scripts/install_bepinex.sh`, `scripts/build_plugin.sh`, and both venvs (`.venv` for SB3/elo,
   `.venv-sf` for Sample Factory). Versions known to work in `.venv-sf`: Python 3.12.13,
   sample-factory 2.1.1, torch 2.15.0+cpu, numpy 1.26.4, gymnasium 0.29.1.
2. Check the game runs: `.venv/bin/python examples/quickstart.py`. On a non-NixOS machine,
   `SOCCER_RL_WRAPPER=""` disables the steam-run wrapper. Inside Claude Code's sandbox the game
   can't start, so run game commands with the sandbox disabled.
3. Download the release assets and extract them at the repo root. Archive paths start with `runs/sf/`:
   ```bash
   gh release download sf7-final-2026-11-25 --repo cyprienruffino/zinedine --dir /tmp/zin
   for f in /tmp/zin/*.tar.gz; do tar xzf "$f"; done
   ```
   That restores the following:
   - `runs/sf/sf7/{config.json,checkpoint_p0,checkpoint_p1}`: the latest checkpoint of each
     policy.
   - Every opponent-pool directory sf7 samples from: `runs/sf/sf{4,5,6,7}/pool_p{0,1}` and
     `runs/sf/sf8_exploit/pool_p0`.
   - `runs/sf/champion_pool/` and the head-to-head sets `runs/sf/h2h_sf6`, `runs/sf/h2h_sf7` and
     `runs/sf/h2h_sf7b`.
4. **Resume sf7.** Pick one of two ways:
   - **Shared laptop:** `bash runs/sf7.sh` starts the RAM-aware supervisor (§3.5).
   - **Dedicated machine:** run the trainer directly, with workers ≈ (free RAM − 7 GB) / 0.75 GB,
     max ~1 per core:
     ```bash
     setsid nohup .venv-sf/bin/python examples/train_sf.py --experiment sf7 --num_workers 14 \
       --num_envs_per_worker 2 --num_policies 2 --switch_cost 0 --shaping 0.02 \
       --anneal_env_steps 4000000 --draw_penalty 0.6 --use_rnn True --rnn_type gru --rnn_size 256 \
       --recurrence 64 --time_feature True --learning_rate 0.0001 --max_episode_steps 1000 \
       --num_epochs 2 --modifiers random --modifier_feature True \
       --opponent_pool runs/sf/sf4/pool_p0,runs/sf/sf4/pool_p1,runs/sf/sf5/pool_p0,runs/sf/sf5/pool_p1,runs/sf/sf6/pool_p0,runs/sf/sf6/pool_p1,runs/sf/sf7/pool_p0,runs/sf/sf7/pool_p1,runs/sf/sf8_exploit/pool_p0 \
       >> runs/sf/sf7.log 2>&1 < /dev/null &
     setsid nohup .venv-sf/bin/python scripts/export_sf.py runs/sf/sf7 --every 1800 >> runs/sf/sf7.export.log 2>&1 < /dev/null &
     ```
     The exporter must run too: it writes sf7's own snapshots into `pool_p*`, which sf7 also plays
     against. This is sf7's exact configuration so far (lr 1e-4). For the recommended next step
     (§6.1), use `--learning_rate 0.00003`, here or in `runs/sf7.sh`.
5. **Verify the resume** within the first minutes. Both checks have failed silently before:
   - `grep "Loaded experiment state" runs/sf/sf7/sf_log.txt | tail -2` must show ~2.27B / 2.30B
     samples per policy. "Did not load" means it started from scratch (§5.2).
   - TensorBoard `train/lr` must be 1e-4 (`runs/sf/sf7/.summary/{0,1}`). 3e-4 means the
     learning-rate patch isn't applied (§5.1).
6. **Monitor progress:**
   - Training stats: `.venv-sf/bin/python scripts/sf_stats.py runs/sf/sf7 --points 4` (win, draw,
     tap share per policy). The game loss rate is 1 − win − draw, not the "loss" tag.
   - Decisions: use 600-games-per-pair head-to-heads (§3.6), never broad rounds.

## 3. Training setup (sf7)

### 3.1 Algorithm and infrastructure
- Sample Factory 2.1.1 **APPO** (asynchronous PPO), CPU only. Settings:
  - 2 learning policies, each with its own learner (`--learner_threads 4`);
  - workers × 2 games each (`--num_envs_per_worker 2`);
  - rollout 128, batch 8192, 1 batch per epoch, 2 epochs;
  - PPO clip 0.2, value clip 1.0, value loss coefficient 0.5;
  - entropy coefficient 0.003, max gradient norm 0.5;
  - Adam (eps 1e-6), constant learning rate **1e-4**;
  - gamma 0.999, GAE λ 0.98;
  - returns normalized, no V-trace, `max_policy_lag` 1000.
- **Frozen-opponent mode** (`SoccerVsPoolEnv`): each game is one learner against a `SnapshotOpponent`.
  - The opponent is a NumPy copy of a snapshot, drawn from the `--opponent_pool` directories, with
    `recent_frac` 0.5: the newest half of each directory's history.
  - The opponent changes every episode and keeps its own GRU state and stacked view.
  - Learners play both sides; observations are mirrored, so one policy plays either side.
- **Duel mode** (`SoccerDuelEnv`, used by sf1–sf3 and sf6): both teams are learners.
- Patches in `examples/train_sf.py`. All are module-level, because the spawned learners re-import
  the script:
  - `torch.load(weights_only=False)`: without it, every resume silently restarts from scratch (§5.2);
  - the learner thread count;
  - OMP/OPENBLAS/MKL threads set to 1;
  - **`--learning_rate` wins over the checkpoint's stored rate** (§5.1).

### 3.2 Network (~670k weights)
- Input: 812 = 4 stacked frames × 203.
- Encoder: MLP 256-256, tanh.
- Core: GRU 256, trained on sequences of 64 steps (`--recurrence 64`).
- Heads: a 2-logit policy (single button: press or release) and a value head. Actor and critic
  share weights.
- Orthogonal initialization.
- `scripts/export_sf.py` exports the policy to `.npz` (`w0, b0, w1, b1, gru_*, w_out, b_out`).
  `soccer_rl.selfplay.NumpyPolicy` runs it; it matches Sample Factory to 1e-7.

### 3.3 Observation (per frame, 203 values)
- `obs="v2"`: egocentric and mirrored for P2. Body states are normalized with
  `soccer_rl/obs_stats_v2.npz`.
- It includes the hidden jump timers `jumpLimiter` and `jump2delay` for each player.
- Point clock (`time_feature`): steps since kickoff / horizon, reset after each goal.
- 7 match-modifier values (`modifier_feature`).
- `history=4` frames.
- `frame_skip=2`: the game runs 50 physics ticks/s, so that's 25 decisions per second. The point
  timeout `max_episode_steps=1000` is 40 s.
- Older snapshots that lack the clock (780 inputs) or the modifiers (784 inputs) are trimmed
  automatically by `SnapshotOpponent` and `elo.py`.

### 3.4 Reward (per agent, zero-sum goals)
- Goal: ±1.
- **Draw: −0.6** for any goalless end, both timeout and ball out.
  - Timeout and ball out must cost the same, otherwise agents kick the ball out on purpose
    (Cyprien's call).
  - Kickoff-to-timeout counts as one episode.
- Shaping: 0.02 × ball progress, faded linearly to 0 over each env's first 4M steps. The fade
  resumes from checkpoint sample counts (`steps_done_per_env`), so in sf7 it's long gone.
- No switch cost (anti-mashing penalty) and no macro actions: both are Cyprien's call. Mashing is
  fine if it wins.
- Match modifiers are rolled per episode with the game's own probabilities (`--modifiers random`).

### 3.5 Supervisor (`runs/sfN.sh` → `runs/sf/sfN.supervise.sh`)
Built for Cyprien's laptop (16 cores, 30 GB, shared with his work). It acts on free RAM:

| Free RAM | Action |
|---|---|
| < 3.5 GB | Kill the training process group |
| > 7.9 GB | Start with (free − 7 GB) / 700 MB workers, max 14 |
| ≥ 4 more workers fit | Grow a running run (SIGINT, so a checkpoint is saved first) |

It also keeps the exporter running. The log is `runs/sf/sfN.supervisor.log`. Rerunning
`runs/sfN.sh` is safe: it won't start a second supervisor.

To stop it cleanly:
1. Kill the supervisor (`*.supervise.sh`).
2. Kill the exporter.
3. SIGINT the trainer and wait up to ~3 min: it writes a final checkpoint.
4. Kill the trainer's process group, which takes the games with it.

RAM per sf7 worker is about 0.75 GB (games + NumPy opponents). Fixed overhead is about 3 GB
(learners + inference). With 14 workers sf7 runs at ~3.4–4k samples/s.

### 3.6 Evaluation
- `examples/elo.py <dirs of .npz> --games 600 --procs 3 --time-feature --modifier-feature
  --modifiers none|random [--seed N]`.
- It runs a round-robin of all `.npz` files in the given directories plus `idle`, `random` and `pulse`,
  then a Bradley-Terry fit anchored at **idle = 1000**.
- 600 games per pair gives about ±14 per player. A 4-player set takes ~1 h 20 min on 3 processes.
- The broad rounds of the early days (24–60 games per pair) swung ±30–90 between rounds and misled
  several decisions.
- Rule: confirm any single +2σ result before calling it progress. sf7's gain was confirmed three
  times.
- Watch two bots: `examples/watch.py --snapshot A.npz --opponent-snapshot B.npz [--argmax]`. To play
  against a bot yourself, use `--human`.

## 4. Results

### Strength over time (Elo, idle = 1000)

| Agent | Elo | Source |
|---|---|---|
| Built-in AI | below idle | idle beats it 49% / 13% |
| SB3 PPO / ES best (days 1–3) | ~1090 | `runs/final2/elo_final.txt`, 60 games per pair |
| sf3 / sf5 / sf6 plateau (days 7–18) | ~1090–1100 | `runs/sf/h2h_sf6_*.txt` |
| sf7 p0 at 1.341B (right after the learning-rate fix) | 1082–1092 | `runs/sf/h2h_sf7_*.txt` |
| sf7 p1 at 2.135B | 1107–1116 | same |
| **sf7 p0 at 2.108B (champion)** | **1125–1128** | same |
| sf7 p0 at 2.244B / p1 at 2.273B (day 25) | 1121 / 1117 (champion 1118, sf6 1090) | `runs/sf/h2h_sf7b_none.txt` |

The three day-24 head-to-heads had 12,600 games each and 7 players. All used the classic variant
except one on random modifiers.

| Player | Classic, seed 0 | Classic, seed 7 | Random variants |
|---|---|---|---|
| sf7 p0, 2.108B | 1125 | 1128 | 1125 |
| sf7 p1, 2.135B | 1107 | 1115 | 1116 |
| champion sf6 p0, 1.435B | 1092 | 1094 | 1099 |
| sf7 p0, 1.341B | 1082 | 1090 | 1092 |
| random / pulse | 938 / 938 | 939 / 929 | 970 / 961 |

Human check (day 3): Cyprien, by his own words "terrible at this game", beat the best ES agent 5–3.
There's no human baseline against the current agents yet.

### Run lineage
`sf1` (MLP duels) → `sf2` (GRU) → `sf3` (+clock, shorter points) → `sf4` (frozen opponents) →
`sf5` (+modifiers) → `sf6` (back to duels) → `sf7` (frozen opponents, larger pool, real lr 1e-4).
`sf8_exploit` is a side branch from sf6 p0.

## 5. Bugs and findings that matter

1. **Learning rate on resume (day 19).** Sample Factory restores `curr_lr` from the checkpoint, and
   only PBT can change it afterwards. sf1/sf2 saved 3e-4, so sf3–sf8 all ran at 3e-4 despite
   `--learning_rate 0.0001`. Fixed in `train_sf.py`.
   - The gradient norm sat at the 0.5 clip limit on every update, so steps were as large as
     allowed.
   - Training at the real 1e-4 produced the first gain in about 10 days. This is consistent with
     arXiv 2603.06009: PPO plateaus when the step size is too large for the gradient noise, and a
     smaller step or a larger batch is the fix.
   - The day-7 note "SF applies the new constant rate" was wrong.
2. **Resume silently restarted from scratch (day 4).** torch ≥ 2.6 defaults to `weights_only=True`,
   which rejects SF's checkpoints; SF then logs "Did not load from checkpoint, starting from
   scratch!" and carries on. Fixed by the `torch.load` patch. Always grep for "Loaded experiment
   state".
3. **Idle is the real baseline.** Never pressing wins 49% / loses 13% against the built-in AI. An
   unpressed player stays upright like a wall, and one press tilts them for seconds. The built-in AI
   is useless as a benchmark.
4. **Agents mash:** 68–76% of presses last one decision. Cyprien accepts this if it wins. They show
   two modes: "maintain" (hold) near their own goal and "mash" near the opponent's.
5. **The exploiter found no weakness** (sf8_exploit, ~29 h against the frozen sf6 champion: 37% won,
   36% lost). It was warm-started from the champion itself, so this is weak evidence (§6).
6. **The frozen-opponent env is unbiased:** the champion against itself comes out ~even, with P2
   slightly better.
7. **Modifier inputs** add nothing measurable yet. The frozen-opponent line rates ~30–40 higher on
   random variants mainly because it trained on them.
8. **Diagnostics on sf7** (day 19, still at 3e-4):
   - policy entropy 0.078 nats (the maximum is 0.69), so it's almost deterministic per step;
   - KL 0.016 per update, 1.7% of samples clipped;
   - policy lag ~0.9 versions;
   - `valids_fraction` 0.906: ~9% of samples are dropped as off-policy at policy switches.

## 6. Next steps (ranked)

From the day-19 literature review (sources below):

1. **Resume sf7 at 3e-5** (`--learning_rate 0.00003`; the patch applies it on resume). sf7 gained
   ~+35 Elo between 1.34B and 2.11B at 1e-4, then flattened by 2.24B. If 3e-5 also stalls, raise the
   batch instead (`--num_batches_per_epoch 4` or `--rollout 256`). Same principle: less step noise
   per update. Judge after ≥ 300M samples per policy with a 600-games-per-pair head-to-head against
   `champion_sf7_p0_2108M.npz`.
2. **Exploiter from scratch, or from a much older snapshot, with a higher entropy coefficient**
   (0.01) against the new champion.
   - AlphaStar reset its exploiters to a non-main initialization to get a different play style.
   - Our exploiter inherited the champion's near-deterministic play.
   - It beating the champion > 55% would prove there's headroom left.
3. **Bigger network, only after 1–2.** A plain MLP gets worse when widened (SimBa, ICLR 2025;
   CHAIN, arXiv 2409.04792). If tried:
   - use SimBa-style residual blocks + LayerNorm + running observation normalization;
   - a wider critic, which matters more than the actor;
   - a lower learning rate, trained from scratch;
   - the expected obstacle is RAM/CPU on the laptop.
4. **Plasticity loss** (Dohare et al., Nature 2024): the sf4→sf7 lineage has trained the same
   weights for 2B+ samples in a changing self-play setting. Possible checks: count saturated tanh
   units on real observations, or try shrink-and-perturb. Caveat: runs from scratch plateaued at the
   same level too.
5. **Human baseline against the current champion** (`watch.py --human`). It would settle whether
   the ceiling is the game or the methods. Cyprien prefers not to play more (he's "terrible"), so
   ask before relying on it.

Literature checked:
- Bansal 2018 (competitive self-play, shaping annealed to 0, opponent sampling): matches what we do.
- Liu et al. 2019 (MuJoCo soccer, population-based training).
- AlphaStar 2019 and TStarBot-X (exploiter resets).
- SimBa / SimbaV2 (scaling networks in RL) and CHAIN (PPO degrades when widened).
- Dohare 2024 and the 2024 plasticity survey (arXiv 2411.04832).
- arXiv 2603.06009 (PPO stagnation; fix with a smaller step or more samples per update).

## 7. Repo map

| Path | What |
|---|---|
| `plugin/SoccerRL/` | BepInEx plugin (C#, Harmony): lockstep TCP bridge, introspection, mute |
| `soccer_rl/env.py` | `SoccerPhysicsEnv`: obs v1/v2, history, clock, modifiers, shaping, draw penalty |
| `soccer_rl/duel.py` | `SoccerDuelEnv` (two learners), `SoccerVsPoolEnv` (learner vs frozen snapshot) |
| `soccer_rl/selfplay.py` | `NumpyPolicy` (MLP/GRU), `SnapshotOpponent`, PFSP pool helpers |
| `soccer_rl/client.py`, `launcher.py` | Wire protocol client, game process launcher (silent, per-instance config) |
| `soccer_rl/noise.py` | Correlated (pink/red) exploration noise for SB3 |
| `examples/train_sf.py` | Sample Factory training (duels or `--opponent_pool`), with the patches in §3.1 |
| `examples/elo.py` | Round-robin + Bradley-Terry ratings (`--builtin`, `--modifiers`, `--argmax-pools`) |
| `examples/watch.py` | Watch bot vs bot, bot vs built-in AI, or play against a bot (`--human`) |
| `examples/train_ppo.py`, `train_selfplay.py`, `train_es.py` | Early SB3 PPO / self-play / ES pipelines (`.venv`) |
| `scripts/export_sf.py` | SF checkpoints → `.npz` pools (`--every 1800` keeps running) |
| `scripts/expand_sf_obs.py` | Warm-start surgery: add zero-weighted inputs, Adam moments included |
| `scripts/sf_stats.py` | Per-policy win/draw/tap stats over time from SF summaries |
| `scripts/install_bepinex.sh`, `build_plugin.sh` | Setup |
| `runs/sf1.sh` … `runs/sf8_exploit.sh` | Launch scripts with their supervisors; the header comment says what each run is |

Never commit the game, its DLLs, decompiled code or decompile scripts; the repo is public.
Commit as `ruffino.cyprien@protonmail.com`.

## 8. Gotchas

- **Sandbox:** the game can't run inside Claude Code's sandbox (steam-run's bwrap can't nest, SIGSYS).
  Disable the sandbox for game commands, `gh` and `git push`.
- **git push:** plain `git push` fails because no credential helper is configured. Use
  `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin main`. Stage
  explicit paths: the sandbox mounts device files in the repo root that break `git add -A`.
- **pkill self-matching:** `pkill -f <pattern>` inside a command whose text contains the pattern
  kills that command. Use `[t]rain_sf`-style patterns.
- **`TaskStop` on a shell that launched `setsid nohup` runs kills them too:** launch from a command
  that exits immediately.
- **Claude Code kills background watchers under memory pressure** (day 20). Sf7 then trained for
  4 days unwatched; re-arm watchers after any such notice.
- **zsh doesn't word-split `$var`:** use explicit arguments or `${=var}`.
- **TMPDIR differs inside and outside the sandbox.**
- **OpenBLAS thread storm** with NumPy opponents (load 145): thread counts must be 1. `train_sf.py`
  sets them.
- **SF summary tag "loss"** can be a training-loss scalar; the game loss rate is 1 − win − draw.
- **SF's log counter restarts each session;** use checkpoint sample counts for progress.
- **Restarts can fail** with "Components take too long to start" on a loaded machine; the supervisor
  retries.
- **`keep_checkpoints` 5** sorts by name. That's fine now resumes work, but it deleted fresh
  checkpoints back when resumes silently restarted from zero.
- **Windowed game freezes** without `runInBackground` (the plugin forces it on). Logs vanish under
  `steam-run` (private `/tmp`), so the launcher captures stdout instead.
- **The built-in AI only drives P2.**
- **SubprocVecEnv scripts need `if __name__ == "__main__":`**, and in the sandbox `uv` needs
  `UV_CACHE_DIR` set to a temp dir.

---

## History

### Day 1–2 (2026-11-01/02): tooling, first training
New machine: Intel Core Ultra 7 366H, 16 cores, 30 GB.

Changes:
- One reload per episode.
- Silence enforced: the plugin mutes in `Awake`, and the launcher makes audio unreachable.
- Hidden jump timers exposed: `jumpLimiter` (10-tick cooldown) and `jump2delay` (second boost after
  3 ticks).
- `obs="v2"`, normalized. Leg heights had been scaled ~10× too small, and ball spin is constant.
- `history=n` frame stacking.
- Opponent hooks and annealing setters.

Throughput:
- 1 game runs 5,988 ticks/s; 16 games run 28,416 ticks/s.
- SB3 `SubprocVecEnv` lockstep with 16 envs: 2,792 steps/s; PPO ~2,200.
- 3–4 runs side by side: ~4,500 steps/s in total.

Findings:
- Idle beats the built-in AI. PPO is bimodal: it either collapses to idle or drifts to ~40% presses.
- A long horizon is needed (gamma 0.999, λ 0.98).
- Self-play plateaued around +0.09 goals per episode against idle.
- Final results: `sp3_from_r10` 39.5% won / 35.5% lost against the built-in AI. All self-play runs
  were equal head-to-head.

### Day 2: research (no training during Cyprien's workday)
Cyprien's direction:
- self-play;
- no macro actions;
- a draw penalty ("the goal is to win, not to draw");
- question the architecture and algorithm.

Literature:
- Correlated (pink) action noise (Eberhard ICLR 2023; Hollenstein AAAI 2024, β≈0.5 for PPO).
- Bansal 2018: shaping annealed to 0 over 10–15% of training; opponents drawn from the recent half
  of history.
- Slime Volleyball: fixed yardsticks mislead, so use Elo.

Added:
- `CorrelatedPolicy`;
- `draw_penalty`, covering any goalless end;
- `time_feature`;
- terminal ball-outs and timeouts (SB3 used to bootstrap past them);
- `elo.py`.

### Night 2 (runs/n2): noise and KL
- PPO with obs v2 was thrashing: approx_kl 0.13–0.16 per update. Fixed with `--target-kl 0.02`.
- Standing still is the best scripted strategy against every opponent. Pressing begets pressing:
  once tilted, pressing is the only way to move.
- Pink/red noise gave a small edge at best.
- Every PPO variant hit the same ceiling: frame_skip 4, a 512×512 network, near-idle priors, bigger
  batches. The best agents were ~+90 Elo over idle.
- ES self-play tied PPO (es4_bigpop 1091, PPO E_pink 1090).
- Added `runs/n2/memguard.sh`.

### Day 3: human check, switch to Sample Factory
- Cyprien beat the best ES agent 5–3; agents mash (68–76% one-decision taps).
- Cyprien: **count runs in days**. SB3's lockstep wasted most of the machine (one env steps in
  1.3 ms, a 16-env lockstep step takes ~5 ms).
- Sample Factory APPO in `.venv-sf`: `SoccerDuelEnv`, `train_sf.py`, `export_sf.py`, `runs/sf1.sh`
  with a supervisor.
- **sf1** settings: 4 MLP policies, 10 workers × 2 games, ~7k samples/s. Its reward: goal ±1, draw
  −0.6 (timeout after 2,000 steps or ball out), shaping 0.02 fading over 4M steps.
- Day-1 adjustments: a switch cost of 0.005 made agents hold forever, so it went to 0. Draw penalty
  0.3 → 0.6. The shaping fade now resumes from checkpoints (`steps_done_per_env`).
- Day 2 rounds: sf1 kept improving (p0 1075 → 1103, PPO E 993). Ratings drift ±50–90 between rounds,
  so compare only within a round.
- Cyprien watched p0 vs p2 and saw the maintain/mash modes. Measured: near its own goal p0 holds 87–89%
  of the time; near the opponent's it presses 52–59% with 2–3-step holds. Midfield stalemates
  happen when both sides are in maintain mode.
- **sf2 (GRU 256, recurrence 64)** started day 3, 16:06. It matched sf1's best with ~10× fewer
  samples per policy. GRU support is end to end (export, NumpyPolicy, elo/watch reset the hidden
  state).

### Day 4–6: the resume bug, sf3
- Day 4: SF had never resumed (torch `weights_only`). sf1's 47 restarts and several sf2 sessions
  started from zero. Fixed. sf2 has ~6 GB of fixed RAM overhead with 4 GRU learners.
- Day 6: sf2 sat idle for 2 days (its supervisor waited for RAM), so it was cut to 2 policies.
  Watchdogs were added.
- **sf3** (day 6, 22:52): sf2 plus a point clock, which restarts after goals. Warm-started with zero
  weights on the new inputs (`expand_sf_obs.py`). `watch.py` match mode was fixed for reloads.

### Day 7–10: sf3 curriculum
- Day 7 decision round: the 1174 seen at 12 h was mostly noise, and sf3 was flat with ~20% draws.
  The learning rate was changed 3e-4 → 1e-4. **Per §5.1 it never actually applied.**
- Curriculum agreed with Cyprien:
  1. consolidate;
  2. stalemate pressure (point timeout 2000 → 1500 → 1000);
  3. frozen-opponent pool;
  4. random variants.
- At 1500 steps, draws stayed the same while points got shorter. ~60% of draws were timeouts and
  ~40% ball-outs, and both keep the same penalty.
- At 1000 steps (day 8, 17:22), agents adapted and scored faster. Elo held, then plateaued by day 10.
- The "lower draw penalty" variant was dropped: it makes stalling cheaper.

### Day 10–15: frozen opponents (sf4, sf5), throughput
- **sf4** (day 10): GRU policies against frozen snapshots (`SoccerVsPoolEnv`, `--opponent_pool`,
  `recent_frac`). OpenBLAS thread storm fixed. RAM limits forced 1 learning policy.
- Day 11: a +46 at 899M was **not confirmed** on day 12 (a ~2σ fluke). Classic strength had been
  flat for a week.
- **sf5** (day 12): random match modifiers, visible in the observation (frame 196 → 203). Added the
  learner-thread patch, 2 epochs and 2 policies, and supervisors that grow a running run.
- sf5 results: the first confirmed classic gain, about +30–50 over its start (p1 at 1.22B: 1131).
  The frozen-opponent line rates ~30–40 higher on random variants.
- Throughput by setup:
  - MLP duels ~8k samples/s;
  - GRU duels ~5–7k;
  - frozen opponents ~3.4k;
  - a single policy is learner-bound at 1.5–4k.
- Env optimizations: `_obs_v2` vectorized and the client buffer reused. Env CPU per duel step fell
  0.75 → 0.55 ms, and sf6 went ~7.2k → ~8k samples/s.

### Day 15–18: sf6 duels, exploiter
- **sf6** (day 15, 01:24): back to duels from sf5's final weights, at ~7,000–8,000 samples/s.
- 600-game head-to-heads replaced broad rounds. sf6 latest (1.435B), sf6's start and sf5's champion
  were equal within 10 Elo, on classic and random variants.
- **sf8_exploit** (day 15, 16:42): one policy trained only against the frozen sf6 p0 champion,
  warm-started from it.
  - After 9 h it won 39% and lost 33%; after 29 h (1.556B) it won 37% and lost 36%.
  - Stopped on day 18. Its snapshots feed sf7's pool.
- Cyprien paused training overnight on days 17–18 for his own CPU task.

### Day 18–25: sf7, the learning-rate fix
- **sf7** (day 18, 22:00): frozen opponents from sf5's final weights (1.279B / 1.280B). The pool is
  sf4–sf7 plus the exploiter.
- The supervisor thrashed overnight (70 stop/starts) because a worker takes ~0.75 GB with this pool.
  Now sized at 700 MB per worker.
- Day 19: literature review and the learning-rate bug. sf7 has run at a real 1e-4 since
  1.339B / 1.338B (day 19, 14:30).
- Day 20: memory pressure from other processes (a `tsc` build was OOM-killed) caused a few stops,
  all resumed cleanly. The watchdog was killed by Claude Code and not re-armed until day 24; training
  ran fine meanwhile.
- Day 24: three head-to-heads confirmed the gain (§4). The champion was promoted to sf7 p0 at
  2.108B.
- Day 25: final head-to-head (§1). Stopped at 15:00 for the move to another machine.
