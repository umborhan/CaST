#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:-help}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$HOME/Desktop/cvpr2027}"
OLD="${OLD:-$PROJECT/cast_paper_rgbd_B20_v15}"
TRAIN_OUT="${TRAIN_OUT:-$PROJECT/cast_b20_seed0_priority_train}"
EVAL_OUT="${EVAL_OUT:-$PROJECT/cast_b20_seed0_priority_eval}"
PORT="${PORT:-2200}"
CARLA_SH="${CARLA_SH:-$HOME/carla/CarlaUE4.sh}"
PYTHON="${PYTHON:-python}"
CARLA="$CARLA_SH -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$PORT"
RUNS=(
 C__cast__B20__s0__sampler-cast
 C__finetune__B20__s0
 C__uniform__B20__s0
 C__openset_entropy__B20__s0
)
COMMON=(
 --profile robust --port "$PORT" --perception rgbd
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
  echo "[READY] Training reuses original B20 C/s0 checkpoints in: $OLD/runs"
  echo "[READY] Source and RGB-D perception are loaded without retraining."
  echo "[PLAN] Four B20 C-shift runs × one seed, keep existing partial episodes."
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
    total+=ep
    finished+=int(done)
    eval_count=sum((evaluation/'runs'/name/f'eval_C__Town{i:02d}.json').is_file() for i in (1,2,4,5))
    print(f'{name:43} train={ep:>2}/20 done={done} eval_towns={eval_count}/4')
print(f'TOTAL TRAINED: {total}/80 episodes, {finished}/4 complete')
print('EVALUATION: 20 held-out routes per town per method, 320 total')
PY_STATUS
}
case "$PHASE" in
  init) init_train;;
  status) check_old; status;;
  train)
    [[ -d "$TRAIN_OUT/runs" ]] || error "First run: bash run_b20_s0.sh init"
    init_train
    mkdir -p "$TRAIN_OUT/command_logs"
    echo '[TRAIN] C-shift, B20, seed 0; four selected runs resume from their original checkpoint files'
    "$PYTHON" -u "$HERE/car.py" "${COMMON[@]}" --out-dir "$TRAIN_OUT" --stages table1 --carla-cmd "$CARLA" \
      2>&1 | tee -a "$TRAIN_OUT/command_logs/priority_train.log"
    ;;
  eval)
    init_eval
    echo '[EVAL] Four frozen policies, 4 towns, 20 held-out routes per town; no safety shield'
    "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py" \
      "${COMMON[@]}" --out-dir "$EVAL_OUT" --town all --eval-stages table1 --carla-cmd "$CARLA" \
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
Resume four selected ORIGINAL B20 C-shift seed-0 runs in the existing project.
Do not run original core_train and priority train/eval on the same port simultaneously.
Do not start eval until status says all four done and the train process exited.
Source / RGB-D perception are loaded, NOT retrained.
Train checkpoint directories are symlinked to the existing original B20 directories;
training updates those four checkpoints only. Other original B20 seeds are unaffected.
Evaluation uses a distinct output tree to avoid changing original eval artifacts.
No B200 proxy: Fig 3 embedding-gap metric is omitted, but DS/SR/RC are valid.
HELP
    ;;
  *) error "Unknown phase '$PHASE'";;
esac
