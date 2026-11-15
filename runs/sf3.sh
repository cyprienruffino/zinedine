#!/usr/bin/env bash
# sf3 (day 6): sf2 (GRU, 2 policies) plus a point clock in the observation (steps since kickoff / 2000), so
# agents can see the draw penalty coming and break midfield stalemates (Cyprien saw both agents sit in
# "maintain mode" for a whole point). Warm-started from sf2's latest checkpoints with the clock inputs'
# weights at zero (scripts/expand_sf_obs.py): identical behaviour at start (verified, max logit diff 1e-6).
# Day 7, 20:00: flat for ~a day (p1 +33, p0 -33 in a 24-games-per-pair round), draws ~20%: learning rate
# 3e-4 -> 1e-4 to consolidate instead of drifting (same experiment, resumes from its checkpoints).
# Day 8, 06:45, curriculum phase 2: still flat at 1e-4 (all snapshots 1093-1112 in two 24-games-per-pair
# rounds), draws ~19%. Point timeout 2000 -> 1500 steps (80 -> 60 s); the clock rescales with it.
# Day 8, 17:25: draws unchanged at 1500 (19-24%), Elo held (latest 1084-1091): timeout 1500 -> 1000 (40 s).
#
# This script starts a supervisor that keeps the run alive for days while leaving RAM to Cyprien:
#   - stops training when available RAM drops below 3.5 GB (kills the process group, so memory is freed)
#   - restarts it (resuming from the latest checkpoints) once more than 8 GB is available, with as many workers
#     (≤ 10) as fit above the 3.5 GB floor, so it doesn't stop/start in a loop
#   - keeps the exporter (Elo pools every 30 min) running
# Log: runs/sf/sf1.supervisor.log. Stop everything: kill the supervisor, then the training process group.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=sf3
mkdir -p runs/sf

cat > runs/sf/$EXP.supervise.sh <<SUP
#!/usr/bin/env bash
cd "$(pwd)"
train_pid() { ps -eo pid,args | awk '/[t]rain_sf\.py --experiment $EXP / && !/zsh|bash/ {print \$1}' | head -1; }
start() {  # \$1 = number of workers
  (setsid nohup .venv-sf/bin/python examples/train_sf.py --experiment $EXP --num_workers \$1 --num_envs_per_worker 2 \\
    --num_policies 2 --switch_cost 0 --shaping 0.02 --anneal_env_steps 4000000 --draw_penalty 0.6 --use_rnn True --rnn_type gru --rnn_size 256 --recurrence 64 --time_feature True --learning_rate 0.0001 --max_episode_steps 1000 \\
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
  elif [[ -z "\$pid" ]] && (( avail > 8800 )); then
    # Measured: ~1.8 GB for the learners + ~0.55 GB per worker (2 games). Pick a size that stays above the
    # 3.5 GB floor with ~0.7 GB to spare, otherwise it gets stopped again right away (stop/start thrashing).
    # sf2 (GRU, 2 policies since day 6) has ~3 GB of fixed overhead (learners ~0.9 GB each, inference workers
    # ~0.55 GB each) plus ~0.45 GB per worker. Only start when >= 4 workers fit above the 3.5 GB floor:
    # smaller runs barely train and just thrash between stop and start.
    workers=\$(( (avail - 3500 - 3000 - 500) / 450 )); (( workers > 14 )) && workers=14
    if (( workers < 4 )); then sleep 30; continue; fi
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
