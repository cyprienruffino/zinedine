#!/usr/bin/env bash
# Morning wrap-up (2026-11-03): stop all training, then rate the finalists with many games per pair.
set -uo pipefail
cd "$(dirname "$0")/.."
for pid in $(ps -eo pid,args | awk '/[t]rain_(es|selfplay|ppo)\.py/ && !/zsh|bash/ {print $1}'); do
  kill -- -"$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null
done
sleep 10
echo "trainers left: $(ps -eo args | grep -c '[t]rain_')  games left: $(ps -eo args | grep -c '[S]occerPhysics.x86_64')"

mkdir -p runs/final2/es_pool runs/final2/ppo_pool
cp runs/es/skill_pool/E_latest.npz runs/final2/ppo_pool/ppo_E.npz
for r in es1_from_E es4_bigpop es5_sigma05; do
  cp "$(ls -t runs/es/${r}_pool/*.npz | head -1)" "runs/final2/es_pool/$r.npz"
done
ls runs/final2/*_pool
.venv/bin/python examples/elo.py runs/final2/ppo_pool runs/final2/es_pool --argmax-pools runs/final2/es_pool \
  --max-players 3 --games 60 --procs 8 2>&1 | grep -v Warn | tee runs/final2/elo_final.txt
