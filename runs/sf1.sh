#!/usr/bin/env bash
# Long Sample Factory self-play run (days): population of 4 policies, both teams learning in every game.
#   shaping 0.02 fading over ~12 h (4M steps per env). Day 1 16:20: switch cost 0.005 -> 0 (it caused
#   hold-forever play) and draw penalty 0.3 -> 0.6 (draws had climbed to 39-60%: a stalling equilibrium)
#   obs v2 + 4-frame history; frame_skip 2; 10 workers x 2 games
#
# This script starts a supervisor that keeps the run alive for days while leaving RAM to Cyprien:
#   - stops training when available RAM drops below 3.5 GB (kills the process group, so memory is freed)
#   - restarts it (resuming from the latest checkpoints) once more than 8 GB is available, with as many workers
#     (≤ 10) as fit above the 3.5 GB floor, so it doesn't stop/start in a loop
#   - keeps the exporter (Elo pools every 30 min) running
# Log: runs/sf/sf1.supervisor.log. Stop everything: kill the supervisor, then the training process group.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=sf1
mkdir -p runs/sf

cat > runs/sf/$EXP.supervise.sh <<SUP
#!/usr/bin/env bash
cd "$(pwd)"
train_pid() { ps -eo pid,args | awk '/[t]rain_sf\.py --experiment $EXP / && !/zsh|bash/ {print \$1}' | head -1; }
start() {  # \$1 = number of workers
  (setsid nohup .venv-sf/bin/python examples/train_sf.py --experiment $EXP --num_workers \$1 --num_envs_per_worker 2 \\
    --num_policies 4 --switch_cost 0 --shaping 0.02 --anneal_env_steps 4000000 --draw_penalty 0.6 \\
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
  elif [[ -z "\$pid" ]] && (( avail > 8000 )); then
    # Measured: ~1.8 GB for the learners + ~0.55 GB per worker (2 games). Pick a size that stays above the
    # 3.5 GB floor with ~0.7 GB to spare, otherwise it gets stopped again right away (stop/start thrashing).
    workers=\$(( (avail - 6000) / 550 )); (( workers > 10 )) && workers=10; (( workers < 2 )) && workers=2
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
