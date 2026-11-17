#!/usr/bin/env bash
# sf5 (day 12, curriculum phase 4): sf4 (1 GRU policy, clock, 1000-step points, lr 1e-4, frozen-opponent pool)
# plus random match modifiers rolled per episode with the game's own probabilities (rain, splits, mini/big
# goals, ball types, headless), visible in the observation (7 values per frame). Warm start: sf4 at 1.0B
# samples with zero weights on the modifier inputs (identical play at start, max logit diff 4e-6). Goal:
# play every variant of the real game, not only the classic setup. Strength in classic mode has plateaued.
# Day 13, 15:20: throughput fixes. Sample Factory pins learners to 1 torch thread, so a single policy was
# learner-bound at ~2.5-3k samples/s: --learner_threads 4 (default in train_sf.py), --num_epochs 2, and back
# to 2 policies (p1 = copy of p0 at 1.09B samples) so two learners train in parallel.
#
# This script starts a supervisor that keeps the run alive for days while leaving RAM to Cyprien:
#   - stops training when available RAM drops below 3.5 GB (kills the process group, so memory is freed)
#   - restarts it (resuming from the latest checkpoints) once more than 8 GB is available, with as many workers
#     (≤ 10) as fit above the 3.5 GB floor, so it doesn't stop/start in a loop
#   - keeps the exporter (Elo pools every 30 min) running
# Log: runs/sf/sf1.supervisor.log. Stop everything: kill the supervisor, then the training process group.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=sf5
mkdir -p runs/sf

cat > runs/sf/$EXP.supervise.sh <<SUP
#!/usr/bin/env bash
cd "$(pwd)"
train_pid() { ps -eo pid,args | awk '/[t]rain_sf\.py --experiment $EXP / && !/zsh|bash/ {print \$1}' | head -1; }
start() {  # \$1 = number of workers
  (setsid nohup .venv-sf/bin/python examples/train_sf.py --experiment $EXP --num_workers \$1 --num_envs_per_worker 2 \\
    --num_policies 2 --switch_cost 0 --shaping 0.02 --anneal_env_steps 4000000 --draw_penalty 0.6 --use_rnn True --rnn_type gru --rnn_size 256 --recurrence 64 --time_feature True --learning_rate 0.0001 --max_episode_steps 1000 --num_epochs 2 --modifiers random --modifier_feature True --opponent_pool runs/sf/sf3/pool_p0,runs/sf/sf3/pool_p1,runs/sf/sf4/pool_p0,runs/sf/sf4/pool_p1,runs/sf/sf5/pool_p0,runs/sf/sf5/pool_p1 \\
    >> runs/sf/$EXP.log 2>&1 < /dev/null &)
}
while true; do
  if ! ps -eo args | grep -q "[e]xport_sf.py runs/sf/$EXP "; then
    (setsid nohup .venv-sf/bin/python scripts/export_sf.py runs/sf/$EXP --every 1800 >> runs/sf/$EXP.export.log 2>&1 < /dev/null &)
  fi
  avail=\$(awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo)
  pid=\$(train_pid)
  if [[ -n "\$pid" ]] && (( avail < 3500 )); then
    kill -- -"\$(ps -o pgid= -p "\$pid" | tr -d ' ')"
    echo "\$(date '+%F %H:%M') stopped: \${avail} MB available"
    sleep 60
  elif [[ -n "\$pid" ]]; then
    # Grow a running run when RAM frees up (it only ever shrank before): restart it bigger once at least 4 more
    # workers fit with 1 GB to spare above the floor. Resumes from checkpoints; costs about a minute.
    cur=\$(ps -o args= -p "\$pid" | grep -o -- '--num_workers [0-9]*' | awk '{print \$2}')
    extra=\$(( (avail - 3500 - 1000) / 450 ))
    if [[ -n "\$cur" ]] && (( extra >= 4 && cur < 14 )); then
      kill -INT "\$pid"; for _ in \$(seq 20); do kill -0 "\$pid" 2>/dev/null || break; sleep 3; done
      kill -- -"\$(ps -o pgid= -p "\$pid" | tr -d ' ')" 2>/dev/null
      echo "\$(date '+%F %H:%M') upsizing from \$cur workers: \${avail} MB available"
      sleep 15
      continue
    fi
  elif [[ -z "\$pid" ]] && (( avail > 7900 )); then
    # Measured: ~1.8 GB for the learners + ~0.55 GB per worker (2 games). Pick a size that stays above the
    # 3.5 GB floor with ~0.7 GB to spare, otherwise it gets stopped again right away (stop/start thrashing).
    # sf2 (GRU, 2 policies since day 6) has ~3 GB of fixed overhead (learners ~0.9 GB each, inference workers
    # ~0.55 GB each) plus ~0.45 GB per worker. Only start when >= 4 workers fit above the 3.5 GB floor:
    # smaller runs barely train and just thrash between stop and start.
    workers=\$(( (avail - 3500 - 3000 - 500) / 450 )); (( workers > 14 )) && workers=14
    if (( workers < 2 )); then sleep 30; continue; fi  # small runs are slow, but better than idle
    start \$workers
    echo "\$(date '+%F %H:%M') started with \$workers workers: \${avail} MB available"
    sleep 120
  fi
  sleep 30
done
SUP
chmod +x runs/sf/$EXP.supervise.sh
# (pgrep, not `ps | grep -q`: with pipefail, grep -q's early exit makes the pipeline fail)
if pgrep -f "^bash runs/sf/$EXP.supervise.sh" > /dev/null; then echo "supervisor already running"; exit 0; fi
(setsid nohup runs/sf/$EXP.supervise.sh >> runs/sf/$EXP.supervisor.log 2>&1 < /dev/null &)
echo "supervisor started (log runs/sf/$EXP.supervisor.log)"
