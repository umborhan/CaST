#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:-help}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$HOME/Desktop/cvpr2027}"
OLD="${OLD:-$PROJECT/cast_paper_rgbd_B20_v15}"
TRAIN_OUT="${TRAIN_OUT:-$PROJECT/cast_b20_seed0_priority_train_3shifts}"
EVAL_OUT="${EVAL_OUT:-$PROJECT/cast_b20_seed0_priority_eval_3shifts}"
PORT="${PORT:-2200}"
CARLA_SH="${CARLA_SH:-$HOME/carla/CarlaUE4.sh}"
PYTHON="${PYTHON:-python}"
CARLA="$CARLA_SH -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$PORT"
# Choose C first for fastest preliminary combined-shift results, then L and A.
# Comma-separated domain shifts: C, L, A or A,L,C.
SUITES="${SUITES:-C}"
[[ ",$SUITES," != *,,* ]] || { echo "Empty suite in SUITES=$SUITES" >&2; exit 2; }
export CAST_PRIORITY_SUITES="$SUITES"
IFS=',' read -r -a SUITE_LIST <<< "$SUITES"
RUNS=()
SEEN_SUITES=" "
for SUITE in "${SUITE_LIST[@]}"; do
  [[ "$SUITE" == A || "$SUITE" == L || "$SUITE" == C ]] || { echo "Invalid suite '$SUITE'; use C,L,A" >&2; exit 2; }
  [[ "$SEEN_SUITES" != *" $SUITE "* ]] || { echo "Repeated suite '$SUITE'" >&2; exit 2; }
  SEEN_SUITES="$SEEN_SUITES$SUITE "
  RUNS+=(
    "${SUITE}__cast__B20__s0__sampler-cast"
    "${SUITE}__finetune__B20__s0"
    "${SUITE}__uniform__B20__s0"
    "${SUITE}__openset_entropy__B20__s0"
  )
done
COMMON=(
 --profile robust --port "$PORT" --tm-port "${TM_PORT:-8000}" --perception rgbd
 --route-length 200 --budget 20
 --source-towns Town10HD_Opt,Town03_Opt
 --layout-towns Town01_Opt,Town02_Opt,Town04_Opt,Town05_Opt
 --lateral-control controller --light-source v2i --light-token-version 2
 --critic-target lcb --critic-disagreement-penalty 0.3 --align-metric action
 --late-brake-prob 0.5 --source-rl-max 0.25 --target-rl-max 0.25
 --takeover-loss-weight 1.0 --takeover-penalty=-5.0
 --hazard-aug-prob 0.25 --hazard-aug-version 1
 --target-state-bc-weight 2.0 --target-state-bc-threshold 0.35
 --adapt-min-weather-episodes 1
 --eval-routes-per-town 20 --scripted-routes-per-town 20
)
error(){ echo "ERROR: $*" >&2; exit 2; }
check_old(){
  [[ -d "$OLD/runs" ]] || error "Existing B20 runs missing: $OLD/runs"
  [[ -s "$OLD/run_config.json" ]] || error "Existing B20 training configuration missing"
  local x
  for x in source/DONE source/source_agent.pt source/source_snapshot.npz source/causal_dataset.npz source/null_bank.pt \
           perception/DONE perception/segnet.pt; do
    [[ -s "$OLD/$x" ]] || error "Missing frozen artifact: $OLD/$x"
  done
  for x in "${RUNS[@]}"; do
    [[ -s "$OLD/runs/$x/ckpt/state.json" || -f "$OLD/runs/$x/ADAPT_DONE" ]] || \
      error "Missing original B20 checkpoint for $x; cannot reuse it"
  done
}
check_paths(){
 [[ "$(realpath -m "$OLD")" != "$(realpath -m "$TRAIN_OUT")" ]] || error "TRAIN_OUT equals OLD"
 [[ "$(realpath -m "$OLD")" != "$(realpath -m "$EVAL_OUT")" ]] || error "EVAL_OUT equals OLD"
 [[ "$(realpath -m "$TRAIN_OUT")" != "$(realpath -m "$EVAL_OUT")" ]] || error "TRAIN_OUT equals EVAL_OUT"
}
link_expected(){
  local src="$1" dst="$2"
  [[ -e "$src" ]] || error "Source missing: $src"
  if [[ -e "$dst" || -L "$dst" ]]; then
    [[ -L "$dst" ]] || error "$dst exists but is not a symlink; refusing to overwrite"
    [[ "$(realpath -e "$dst")" == "$(realpath -e "$src")" ]] || error "$dst points somewhere unexpected"
  else
    ln -s "$(realpath -e "$src")" "$dst"
  fi
}
init_base(){
  local dest="$1"
  mkdir -p "$dest/runs" "$dest/results" "$dest/command_logs"
  link_expected "$OLD/source" "$dest/source"
  link_expected "$OLD/perception" "$dest/perception"
  if [[ ! -f "$dest/run_config.json" ]]; then
    cp -p "$OLD/run_config.json" "$dest/run_config.json"
  fi
}
init_train(){
  check_paths;check_old
  init_base "$TRAIN_OUT"
  local x
  for x in "${RUNS[@]}";do
    link_expected "$OLD/runs/$x" "$TRAIN_OUT/runs/$x"
  done
  "$PYTHON" -m py_compile "$HERE/car.py" "$HERE/car_eval.py"
  echo "[READY] Training reuses original B20 s0 checkpoints in: $OLD/runs"
  echo "[READY] Source and RGB-D perception are loaded without retraining."
  echo "[PLAN] ${#RUNS[@]} B20 runs for suite(s) $SUITES × one seed; keep existing partial episodes."
}
init_eval(){
  check_paths; check_old
  init_base "$EVAL_OUT"
  local x f d
  for x in "${RUNS[@]}"; do
    [[ -f "$OLD/runs/$x/ADAPT_DONE" && -s "$OLD/runs/$x/agent.pt" ]] || \
      error "Training incomplete for $x; complete 'train' first"
    d="$EVAL_OUT/runs/$x"; mkdir -p "$d"
    for f in ADAPT_DONE agent.pt growth.pt adapt.json; do
      if [[ -e "$OLD/runs/$x/$f" ]];then link_expected "$OLD/runs/$x/$f" "$d/$f";fi
    done
  done
  echo "[READY] Evaluations write ONLY to $EVAL_OUT (original B20 eval files remain untouched)."
}
status(){
  "$PYTHON" - "$OLD" "$EVAL_OUT" "${RUNS[@]}" <<'PY_STATUS'
from pathlib import Path
import json, sys
old, evaluation = map(Path, sys.argv[1:3])
runids = sys.argv[3:]
total=0
finished=0
for name in runids:
    folder=old/'runs'/name
    p=folder/'ckpt'/'state.json'
    try: ep=int(json.loads(p.read_text())['ep'])
    except (OSError,ValueError,KeyError,TypeError): ep=0
    done=(folder/'ADAPT_DONE').exists()
    if done: ep=max(ep,20)
    total+=ep
    finished+=int(done)
    suite=name.split('__')[0]
    towns=('Town10HD','Town03') if suite == 'A' else ('Town01','Town02','Town04','Town05')
    eval_count=sum((evaluation/'runs'/name/f'eval_{suite}__{town}.json').is_file() for town in towns)
    print(f'{name:43} train={ep:>2}/20 done={done} eval_towns={eval_count}/{len(towns)}')
print(f'TOTAL TRAINED: {total}/{20*len(runids)} episodes, {finished}/{len(runids)} complete')
expected = sum(20*(2 if n.startswith('A__') else 4) for n in runids)
print(f'EVALUATION: 20 held-out routes per town per method, {expected} total')
PY_STATUS
}
case "$PHASE" in
  init) init_train;;
  status) check_old; status;;
  train)
    [[ -d "$TRAIN_OUT/runs" ]] || error "First run: bash run_b20_s0.sh init"
    init_train
    mkdir -p "$TRAIN_OUT/command_logs"
    echo "[TRAIN] suites=$SUITES, B20, seed 0; selected runs resume original checkpoints"
    "$PYTHON" -u "$HERE/car.py" "${COMMON[@]}" --out-dir "$TRAIN_OUT" --stages table1 --carla-cmd "$CARLA" \
      2>&1 | tee -a "$TRAIN_OUT/command_logs/priority_train.log"
    ;;
  eval)
    init_eval
    echo "[EVAL] suite(s)=$SUITES, 20 held-out routes per town per method; no safety shield"
    "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py" \
      "${COMMON[@]}" --out-dir "$EVAL_OUT" --town "${EVAL_TOWN:-all}" --eval-stages table1 --eval-routes-per-town "${EVAL_ROUTES_PER_TOWN:-20}" --carla-cmd "$CARLA" \
      2>&1 | tee -a "$EVAL_OUT/command_logs/priority_eval.log"
    ;;
  aggregate)
    init_eval
    "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py" \
      "${COMMON[@]}" --out-dir "$EVAL_OUT" --aggregate --eval-stages table1 \
      2>&1 | tee -a "$EVAL_OUT/command_logs/aggregate.log"
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage: bash run_b20_s0.sh {init|status|train|eval|aggregate}
Resume selected ORIGINAL B20 seed-0 runs. Set SUITES=C (default), L, A, or A,L,C.
Do not run original core_train and priority train/eval on the same port simultaneously.
Do not start eval until all selected runs say done=True and training exited.
Source / RGB-D perception are loaded, NOT retrained.
Train checkpoint directories are symlinked to the existing original B20 directories;
training updates only selected checkpoints. Other original B20 seeds are unaffected.
Evaluation uses a distinct output tree to avoid changing original eval artifacts.
No B200 proxy: proxy-dependent metrics are not produced; DS/SR/RC are available after evaluation.
HELP
    ;;
  *) error "Unknown phase '$PHASE'";;
esac
