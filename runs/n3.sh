#!/usr/bin/env bash
# Night 2b (2026-11-02): same experiment as runs/tonight.sh, after finding that the v2-obs runs made
# PPO updates ~10x too large (approx_kl 0.13-0.16 instead of ~0.01): every run now uses --target-kl 0.02,
# shaping 0.02 fading over 10M steps (Bansal et al.), draw penalty 0.3.
#   E  pink noise (beta 1), from scratch      F  plain sampling (beta 0), from scratch: control
#   G  red noise (beta 2), from scratch       H  pink noise, warm start from sp3
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/n3
cp runs/night/status.py runs/n3/status.py
sed -i 's#runs/night/#runs/n3/#g' runs/n3/status.py
INIT=runs/night/final/sp3_from_r10.zip

# Optional arguments pick runs by letter (e.g. `runs/tonight.sh A B D`); default: all four.
WANT=${*:-E F G H}
run() {
  local name=$1; shift
  [[ " $WANT " == *" ${name%%_*} "* ]] || return 0
  setsid nohup .venv/bin/python examples/train_selfplay.py --out "runs/n3/$name" \
    --obs v2 --history 4 --frame-skip 2 --max-steps 2000 --gamma 0.999 --gae-lambda 0.98 \
    --ent 0.003 --builtin-envs 2 --idle-envs 3 --draw-penalty 0.3 --target-kl 0.02 --shaping 0.02 --shaping-anneal 10000000 --steps 80000000 "$@" \
    > "runs/n3/$name.log" 2>&1 < /dev/null &
  echo "$name pid $!"
}

run E_pink   --noise-beta 1
run F_white  --noise-beta 0
run G_red    --noise-beta 2
run H_pink_sp3 --init "$INIT" --noise-beta 1
