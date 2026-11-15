#!/usr/bin/env bash
# Night 2 (2026-11-02): self-play with correlated exploration and a draw penalty.
# Starts 4 detached runs (16 envs each) under runs/n2/; check with: .venv/bin/python runs/n2/status.py
#
#   A  sp3 warm start, pink noise (beta 1), draw penalty 0.3
#   B  sp3 warm start, red noise (beta 2), draw penalty 0.3
#   C  from scratch,   pink noise (beta 1), draw penalty 0.3, shaping 0.02 fading over 10M steps
#   D  sp3 warm start, plain sampling (beta 0), draw penalty 0.3: control for the noise
#
# All runs share obs v2 + 4-frame history, so their snapshot pools can be compared with examples/elo.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/n2
cp runs/night/status.py runs/n2/status.py
sed -i 's#runs/night/#runs/n2/#g' runs/n2/status.py
INIT=runs/night/final/sp3_from_r10.zip

# Optional arguments pick runs by letter (e.g. `runs/tonight.sh A B D`); default: all four.
WANT=${*:-A B C D}
run() {
  local name=$1; shift
  [[ " $WANT " == *" ${name%%_*} "* ]] || return 0
  setsid nohup .venv/bin/python examples/train_selfplay.py --out "runs/n2/$name" \
    --obs v2 --history 4 --frame-skip 2 --max-steps 2000 --gamma 0.999 --gae-lambda 0.98 \
    --ent 0.003 --builtin-envs 2 --idle-envs 3 --draw-penalty 0.3 --steps 80000000 "$@" \
    > "runs/n2/$name.log" 2>&1 < /dev/null &
  echo "$name pid $!"
}

run A_pink  --init "$INIT" --noise-beta 1 --shaping 0
run B_red   --init "$INIT" --noise-beta 2 --shaping 0
run C_pink_scratch --noise-beta 1 --shaping 0.02 --shaping-anneal 10000000
run D_white --init "$INIT" --noise-beta 0 --shaping 0
