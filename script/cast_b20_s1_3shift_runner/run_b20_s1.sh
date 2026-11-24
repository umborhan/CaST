#!/usr/bin/env bash
# Independent second target-stage seed for CaST B20. Does not modify seed-0 checkpoints.
set -Eeuo pipefail
PHASE="${1:-help}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$HOME/Desktop/cvpr2027}"
OLD="${OLD:-$PROJECT/cast_paper_rgbd_B20_v15}"
TRAIN_OUT="${TRAIN_OUT:-$PROJECT/cast_b20_seed1_priority_train_3shifts}"
EVAL_OUT="${EVAL_OUT:-$PROJECT/cast_b20_seed1_priority_eval_3shifts}"
PYTHON="${PYTHON:-/home/umb/miniconda3/envs/safe/bin/python}"
CARLA_SH="${CARLA_SH:-/home/umb/carla/CarlaUE4.sh}"
SUITES="${SUITES:-A}"
EVAL_TOWN="${EVAL_TOWN:-all}"
PORT="${PORT:-2200}"
TM_PORT="${TM_PORT:-8000}"
SEED=1
CARLA="$CARLA_SH -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$PORT"

error(){ echo "[ERROR] $*" >&2; exit 2; }
[[ -f "$HERE/.cast_seed1_isolated" ]] || error "First prepare the isolated runner with prepare_seed1.py; refusing to use seed-0 car.py"
[[ -f "$HERE/car.py" && -f "$HERE/car_eval.py" ]] || error "Missing copied car.py / car_eval.py"
[[ -x "$PYTHON" ]] || error "Python missing: $PYTHON"
[[ -f "$CARLA_SH" ]] || error "CARLA launcher missing: $CARLA_SH"
[[ "$PORT" =~ ^[0-9]+$ && "$TM_PORT" =~ ^[0-9]+$ ]] || error "PORT and TM_PORT must be integers"
(( TM_PORT > 0 && TM_PORT <= 65535 && TM_PORT != PORT && TM_PORT != PORT + 1 && TM_PORT != PORT + 2 )) || error "Traffic Manager port conflicts with CARLA ports"
case "$(realpath -m "$TRAIN_OUT")" in "$(realpath -m "$OLD")"|"$(realpath -m "$EVAL_OUT")") error "Training output collision";; esac
[[ "$(realpath -m "$EVAL_OUT")" != "$(realpath -m "$OLD")" ]] || error "Evaluation output collision"
[[ "$TRAIN_OUT" == *seed1* || "$TRAIN_OUT" == *s1* ]] || error "TRAIN_OUT must have seed1 or s1 in its path (protect seed 0)"
[[ "$EVAL_OUT" == *seed1* || "$EVAL_OUT" == *s1* ]] || error "EVAL_OUT must have seed1 or s1 in its path (protect seed 0)"

IFS=',' read -r -a SUITE_LIST <<< "$SUITES"
((${#SUITE_LIST[@]} > 0)) || error "SUITES is empty"
declare -A SEEN=()
RUNS=()
for s in "${SUITE_LIST[@]}"; do
  [[ "$s" == A || "$s" == L || "$s" == C ]] || error "Invalid suite '$s'; use A,L,C"
  [[ -z "${SEEN[$s]:-}" ]] || error "Duplicate suite $s"
  SEEN[$s]=1
  RUNS+=("${s}__cast__B20__s1__sampler-cast" "${s}__finetune__B20__s1" "${s}__uniform__B20__s1" "${s}__openset_entropy__B20__s1")
done
export CAST_PRIORITY_SUITES="$SUITES"

# Maintain identical training options to B20/s0; seed and output directory are the only target-stage changes.
COMMON=(
  --profile robust --seed "$SEED" --port "$PORT" --tm-port "$TM_PORT" --perception rgbd
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
  --scripted-routes-per-town 20
)

require_old(){
  [[ -s "$OLD/run_config.json" ]] || error "Missing source training configuration: $OLD/run_config.json"
  local p
  for p in source/DONE source/source_agent.pt source/source_snapshot.npz source/causal_dataset.npz source/null_bank.pt perception/DONE perception/segnet.pt; do
    [[ -s "$OLD/$p" ]] || error "Missing frozen artifact $OLD/$p"
  done
}
link_exact(){
  local from="$1" to="$2"
  [[ -e "$from" ]] || error "Source missing: $from"
  if [[ -e "$to" || -L "$to" ]]; then
    [[ -L "$to" ]] || error "Refusing to overwrite an existing non-symlink: $to"
    [[ "$(realpath -e "$to")" == "$(realpath -e "$from")" ]] || error "Existing symlink does not point to expected source: $to"
  else
    ln -s "$(realpath -e "$from")" "$to"
  fi
}
base_dir(){
  local out="$1" config_from="$2"
  mkdir -p "$out/runs" "$out/results" "$out/command_logs"
  link_exact "$OLD/source" "$out/source"
  link_exact "$OLD/perception" "$out/perception"
  if [[ ! -e "$out/run_config.json" ]]; then
    cp -p "$config_from/run_config.json" "$out/run_config.json"
  fi
}
init_train(){
  require_old
  base_dir "$TRAIN_OUT" "$OLD"
  local id
  for id in "${RUNS[@]}"; do
    [[ ! -L "$TRAIN_OUT/runs/$id" ]] || error "Refusing a symlinked training run: $id"
    # Always start from a truly new s1 checkpoint, or resume only the same s1 output folder.
    if [[ -e "$TRAIN_OUT/runs/$id" ]]; then
      [[ -d "$TRAIN_OUT/runs/$id" ]] || error "Unexpected run path: $id"
    else
      mkdir -p "$TRAIN_OUT/runs/$id"
    fi
  done
  "$PYTHON" -m py_compile "$HERE/car.py" "$HERE/car_eval.py"
  echo "[READY] seed=1 | suites=$SUITES | B20 | source/perception frozen from $OLD"
  echo "[READY] train=$TRAIN_OUT (new independent s1 directories; seed-0 checkpoints untouched)"
}
init_eval(){
  require_old
  [[ -f "$TRAIN_OUT/run_config.json" ]] || error "Missing seed-1 training directory; train first"
  base_dir "$EVAL_OUT" "$TRAIN_OUT"
  local id f d
  for id in "${RUNS[@]}"; do
    [[ -f "$TRAIN_OUT/runs/$id/ADAPT_DONE" && -s "$TRAIN_OUT/runs/$id/agent.pt" ]] || error "Training not finished: $id"
    d="$EVAL_OUT/runs/$id"; mkdir -p "$d"
    for f in ADAPT_DONE agent.pt growth.pt adapt.json; do
      if [[ -e "$TRAIN_OUT/runs/$id/$f" ]]; then link_exact "$TRAIN_OUT/runs/$id/$f" "$d/$f"; fi
    done
  done
  echo "[READY] Seed-1 frozen checkpoints: $TRAIN_OUT"
  echo "[READY] Evaluation results: $EVAL_OUT"
}
expected_n(){
  [[ "$SUITES" == L ]] && echo 10 || echo 20
}
case "$PHASE" in
  init) init_train ;;
  train)
    init_train
    echo "[TRAIN] seed=1 suites=$SUITES — B20 independent target adaptation"
    "$PYTHON" -u "$HERE/car.py" "${COMMON[@]}" --eval-routes-per-town 20 \
      --out-dir "$TRAIN_OUT" --stages table1 --carla-cmd "$CARLA" \
      2>&1 | tee -a "$TRAIN_OUT/command_logs/seed1_train.log"
    ;;
  eval)
    ((${#SUITE_LIST[@]} == 1)) || error "Evaluate one suite at a time so L=10 and A/C=20 routes per town"
    case "$SUITES" in
      A) N=20; [[ "$EVAL_TOWN" == all || "$EVAL_TOWN" == Town03_Opt || "$EVAL_TOWN" == Town10HD_Opt ]] || error "Invalid A town: $EVAL_TOWN";;
      L|C) [[ "$EVAL_TOWN" != all ]] || error "Specify EVAL_TOWN=Town01_Opt, Town04_Opt or Town05_Opt (exclude Town02)";
           [[ "$EVAL_TOWN" == Town01_Opt || "$EVAL_TOWN" == Town04_Opt || "$EVAL_TOWN" == Town05_Opt ]] || error "Invalid $SUITES town: $EVAL_TOWN";
           [[ "$SUITES" == L ]] && N=10 || N=20;;
    esac
    init_eval
    echo "[EVAL] seed=1 suite=$SUITES town=$EVAL_TOWN routes/town=$N tm_port=$TM_PORT"
    "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py" "${COMMON[@]}" \
      --eval-routes-per-town "$N" --out-dir "$EVAL_OUT" --town "$EVAL_TOWN" \
      --eval-stages table1 --carla-cmd "$CARLA" \
      2>&1 | tee -a "$EVAL_OUT/command_logs/seed1_${SUITES}_${EVAL_TOWN}.log"
    ;;
  aggregate)
    ((${#SUITE_LIST[@]} == 1)) || error "Aggregate one suite at a time"
    init_eval
    N="$(expected_n)"
    "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py" "${COMMON[@]}" \
      --eval-routes-per-town "$N" --out-dir "$EVAL_OUT" --aggregate --eval-stages table1 \
      2>&1 | tee -a "$EVAL_OUT/command_logs/seed1_${SUITES}_aggregate.log"
    ;;
  status)
    "$PYTHON" "$HERE/audit_seed1.py" --train-out "$TRAIN_OUT" --eval-out "$EVAL_OUT" \
      --suites "$SUITES" --status
    ;;
  audit)
    "$PYTHON" "$HERE/audit_seed1.py" --train-out "$TRAIN_OUT" --eval-out "$EVAL_OUT" --suites "$SUITES"
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage: SUITES=A,L,C bash run_b20_s1.sh {init|train|status|audit}
       SUITES=A bash run_b20_s1.sh eval                # A: Town03 and Town10HD, 20 routes each
       SUITES=L EVAL_TOWN=Town01_Opt bash run_b20_s1.sh eval   # repeat for Town04, Town05; 10 routes/town
       SUITES=C EVAL_TOWN=Town01_Opt bash run_b20_s1.sh eval   # repeat for Town04, Town05; 20 routes/town
       SUITES=A bash run_b20_s1.sh aggregate
Original source/perception reused from OLD=~/Desktop/cvpr2027/cast_paper_rgbd_B20_v15.
New seed-1 checkpoints and evaluation results are isolated from s0; source/perception are frozen.
Never start two CARLA runs on the same RPC/TM ports simultaneously.
HELP
    ;;
  *) error "Unknown phase $PHASE";;
esac
