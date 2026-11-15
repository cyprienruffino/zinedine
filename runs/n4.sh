#!/usr/bin/env bash
# Night 3: stop the button mashing. Cyprien played the best agents and saw them mash the button. Measured:
# 68-76% of their presses last one decision (2 ticks), too short for the second jump boost (3+ ticks held).
# Fix without restricting actions: red/pink exploration noise (presses come in runs) plus an annealed switch
# cost (reward cost per new press, an action-rate penalty). All from scratch, PPO self-play, target-kl 0.02.
#   P  red noise,  switch cost 0.005 (≈ -0.3/episode for mashing), fading over 15M steps
#   Q  pink noise, switch cost 0.005, fading over 15M steps
#   R  red noise,  switch cost 0.015 (strong), fading over 15M steps
#   S  red noise,  no switch cost: control
# Watch rollout/tap_share and rollout/hold_mean in the logs; rate with examples/elo.py (now includes pulse).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/n4
sed 's#runs/night/#runs/n4/#g' runs/night/status.py > runs/n4/status.py
WANT=${*:-P Q R S}
run() {
  local name=$1; shift
  [[ " $WANT " == *" ${name%%_*} "* ]] || return 0
  (setsid nohup .venv/bin/python examples/train_selfplay.py --out "runs/n4/$name" \
    --obs v2 --history 4 --frame-skip 2 --max-steps 2000 --gamma 0.999 --gae-lambda 0.98 \
    --ent 0.003 --builtin-envs 2 --idle-envs 3 --draw-penalty 0.3 --target-kl 0.02 \
    --shaping 0.02 --shaping-anneal 10000000 --steps 80000000 "$@" \
    > "runs/n4/$name.log" 2>&1 < /dev/null &)
  echo "started $name"
}
run P_red_sw005  --noise-beta 2 --switch-cost 0.005 --switch-anneal 15000000
run Q_pink_sw005 --noise-beta 1 --switch-cost 0.005 --switch-anneal 15000000
run R_red_sw015  --noise-beta 2 --switch-cost 0.015 --switch-anneal 15000000
run S_red_ctrl   --noise-beta 2
