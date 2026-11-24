#!/usr/bin/env bash
set -euo pipefail
# Verified against the provided CaST training/evaluation scripts on 2026-11-11.
# Executes on Ubuntu with CARLA 0.9.15, CUDA/PyTorch and the 'safe' conda environment.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-help}"
PROJECT="${PROJECT:-$HOME/Desktop/cvpr2027}"
OUT="${OUT:-$PROJECT/cast_paper_rgbd_B20_v15}"
PORT="${PORT:-2200}"
CARLA_SH="${CARLA_SH:-$HOME/carla/CarlaUE4.sh}"
CARLA="$CARLA_SH -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$PORT"
PYTHON="${PYTHON:-python}"
mkdir -p "$PROJECT" "$OUT" "$OUT/command_logs"
COMMON=(
 --profile robust
 --out-dir "$OUT"
 --port "$PORT"
 --perception rgbd
 --route-length 200
 --budget 20
 --source-towns Town10HD_Opt,Town03_Opt
 --layout-towns Town01_Opt,Town02_Opt,Town04_Opt,Town05_Opt
 --lateral-control controller
 --light-source v2i
 --light-token-version 2
 --critic-target lcb
 --critic-disagreement-penalty 0.3
 --align-metric action
 --late-brake-prob 0.5
 --source-rl-max 0.25
 --target-rl-max 0.25
 --takeover-loss-weight 1.0
 --takeover-penalty=-5.0
 --hazard-aug-prob 0.25
 --hazard-aug-version 1
 --target-state-bc-weight 2.0
 --target-state-bc-threshold 0.35
 --adapt-min-weather-episodes 1
 --eval-routes-per-town 20
 --scripted-routes-per-town 20
)
helpmsg() {
 cat <<'EOF'
Usage: bash run_cast_b20.sh PHASE
  check         verify local code and CARLA executable; CPU learning smoke-test
  source        train frozen perception front end and source policy
  reuse_source  OPTIONAL: copy compatible source/perception from PREV_OUT (no B50 target runs)
  core_train    train target proxy + 9 Table 1 methods on A,L,C (5 target-stage seeds)
  core_eval     evaluate source, expert and Table 1 on all towns, unshielded
  extra_train   train shortcut comparisons, budget curves, ablations, sensitivity
  extra_eval    evaluate remaining figures/tables on all towns, unshielded
  aggregate     merge all completed result files into paper CSVs
  plots         generate publication-ready vector plots from measured paper CSVs
  shield_A      OPTIONAL shield-assisted source vs adapted CaST (A checkpoint only)
  shield_C      OPTIONAL shield-assisted source vs adapted CaST (C checkpoint only)
  all           run source, core_train, core_eval, extra_train, extra_eval, aggregate sequentially

Environment overrides: PROJECT OUT PORT CARLA_SH PYTHON.
CAUTION: all is a VERY large training/evaluation experiment collection.
EOF
}
train() {
 local phase="$1" stages="$2"
 echo "[RUN] $phase: $stages";
 "$PYTHON" -u "$HERE/car.py" "${COMMON[@]}" --stages "$stages" --carla-cmd "$CARLA"    2>&1 | tee -a "$OUT/command_logs/${phase}.log"
}
eval_paper() {
 local phase="$1" stages="$2"
 echo "[RUN] $phase: $stages";
 "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py"    "${COMMON[@]}" --town all --eval-stages "$stages" --carla-cmd "$CARLA"    2>&1 | tee -a "$OUT/command_logs/${phase}.log"
}
reuse_source() {
  local prev="${PREV_OUT:-}"
  if [[ -z "$prev" ]]; then
    echo 'ERROR: set PREV_OUT=/absolute/path/to/your/old/training/output first' >&2
    return 2
  fi
  prev="$(realpath -e "$prev")"
  if [[ "$prev" == "$(realpath -m "$OUT")" ]]; then
    echo 'ERROR: PREV_OUT and OUT must be different directories' >&2
    return 2
  fi
  if [[ -e "$OUT/source" || -e "$OUT/perception" || -e "$OUT/run_config.json" ]]; then
    echo "ERROR: destination already contains training artifacts: $OUT" >&2
    echo 'Use a fresh OUT or review its artifacts before proceeding; nothing was overwritten.' >&2
    return 2
  fi
  local f
  for f in run_config.json source/DONE source/source_agent.pt source/causal_dataset.npz source/null_bank.pt source/source_snapshot.npz perception/DONE perception/segnet.pt; do
    if [[ ! -s "$prev/$f" ]]; then
      echo "ERROR: source artifact missing or empty: $prev/$f" >&2
      return 2
    fi
  done
  # Check the recorded training revision, source-stage value options, training towns,
  # perception modality, and original source route length before copying anything.
  "$PYTHON" - "$HERE/car.py" "$prev" "${COMMON[@]}" <<'PY_REUSE'
import importlib.util, json, math, pathlib, sys
path, prev, *flags = sys.argv[1:]
sp = importlib.util.spec_from_file_location('cast_reuse_check', path)
mod = importlib.util.module_from_spec(sp)
sys.modules[sp.name] = mod
sp.loader.exec_module(mod)
old = json.load(open(pathlib.Path(prev) / 'run_config.json'))
if old.get('training_revision') != mod.TRAINING_REVISION:
    raise SystemExit('ERROR: prior source used an incompatible training revision; run the source phase instead.')
sys.argv = [path] + flags
args = mod.parse_args()
P = dict(mod.PROFILES[args.profile])
P['budget'] = args.budget
cfg = mod.make_cfg(args, P)
try:
    mod.validate_value_options(old.get('value_training_options'), cfg, source_only=True)
except ValueError as exc:
    raise SystemExit('ERROR: source value-option mismatch. Do not mix training configurations.\n' + str(exc))
old_cfg = old.get('cfg', {})
old_perception = old_cfg.get('perception_mode')
if old_perception != cfg.perception_mode:
    raise SystemExit(f'ERROR: perception mode mismatch: old={old_perception!r}, new={cfg.perception_mode!r}')
old_len = old_cfg.get('route_target_length_m')
if old_len is None or not math.isclose(float(old_len), float(cfg.route_target_length_m), abs_tol=0.1):
    raise SystemExit(f'ERROR: source training route-length mismatch: old={old_len}, new={cfg.route_target_length_m}.')
old_args = old.get('args', {})
def norm_towns(value):
    return tuple(x.strip().removesuffix('_Opt').casefold() for x in str(value).split(',') if x.strip())
if not old_args.get('source_towns') or norm_towns(old_args['source_towns']) != norm_towns(args.source_towns):
    raise SystemExit(f'ERROR: source training towns mismatch: old={old_args.get("source_towns")!r}, new={args.source_towns!r}')
print('[REUSE] Prior source protocol/options, towns, route length and perception mode match.')
PY_REUSE
  mkdir -p "$OUT/source" "$OUT/perception"
  for f in source/source_agent.pt source/causal_dataset.npz source/null_bank.pt source/source_snapshot.npz perception/segnet.pt; do
    cp -a "$prev/$f" "$OUT/$f"
  done
  if [[ -f "$prev/source/source_stats.json" ]]; then
    cp -a "$prev/source/source_stats.json" "$OUT/source/source_stats.json"
  fi
  cp -a "$prev/run_config.json" "$OUT/run_config.json"
  # DONE markers are last: incomplete transfers never appear completed.
  cp -a "$prev/perception/DONE" "$OUT/perception/DONE"
  cp -a "$prev/source/DONE" "$OUT/source/DONE"
  echo "[REUSE] Source and RGB-D perception copied from: $prev"
  echo "[REUSE] Destination: $OUT"
  echo '[NEXT] bash run_cast_b20.sh core_train'
}

eval_shield() {
 local suite="$1" weather="$2"
 echo "[RUN] shield-assisted diagnostic train-suite=$suite weather=$weather";
 "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py"    --profile robust --out-dir "$OUT" --port "$PORT" --perception rgbd    --route-length 200 --budget 20    --source-towns Town10HD_Opt,Town03_Opt    --layout-towns Town01_Opt,Town02_Opt,Town04_Opt,Town05_Opt    --lateral-control controller --light-source v2i --light-token-version 2    --critic-target lcb --critic-disagreement-penalty 0.3 --align-metric action    --late-brake-prob 0.5 --source-rl-max 0.25 --target-rl-max 0.25    --takeover-loss-weight 1.0 --takeover-penalty=-5.0    --hazard-aug-prob 0.25 --hazard-aug-version 1    --target-state-bc-weight 2.0 --target-state-bc-threshold 0.35    --adapt-min-weather-episodes 1    --town all --train-suite "$suite" --policies source,adapted    --weather-set "$weather" --episodes 20 --route-seed 70000    --eval-tag "B20_${suite}_${weather}" --use-safety-shield --carla-cmd "$CARLA"    2>&1 | tee -a "$OUT/command_logs/shield_${suite}.log"
}
case "$PHASE" in
 help|-h|--help) helpmsg ;;
 check)
  "$PYTHON" -m py_compile "$HERE/car.py" "$HERE/car_eval.py"
  if [[ ! -f "$CARLA_SH" ]]; then echo "ERROR: CARLA launcher missing: $CARLA_SH" >&2; exit 2; fi
  "$PYTHON" - "$HERE/car.py" <<'PY_CHECK'
import ast, sys
p=sys.argv[1]
s=ast.parse(open(p).read())
for name in ('build_suites','plan','aggregate','worker_main'):
 assert any(isinstance(x,ast.FunctionDef) and x.name==name for x in s.body), name
print('[CHECK] source syntax and experiment stages found')
PY_CHECK
  "$PYTHON" "$HERE/car.py" --learning-smoke-test
  echo "[CHECK] passed. Output: $OUT";;
 source) train source 'perception,source';;
 reuse_source) reuse_source;;
 core_train) train core_train 'proxy,table1';;
 core_eval) eval_paper core_eval 'source_eval,expert,table1';;
 extra_train) train extra_train 'table2,fig5,ablation,sensitivity';;
 extra_eval) eval_paper extra_eval 'table2,fig5,ablation,sensitivity';;
 plots)
  "$PYTHON" -u "$HERE/plot_paper_csv.py" --results "$OUT/results" --output "$OUT/paper_figures" \
    2>&1 | tee -a "$OUT/command_logs/plots.log";;
 aggregate)
  "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py"     "${COMMON[@]}" --aggregate --eval-stages all     2>&1 | tee -a "$OUT/command_logs/aggregate.log";;
 shield_A) eval_shield A adverse;;
 shield_C) eval_shield C adverse;;
 all)
  train source 'perception,source'
  train core_train 'proxy,table1'
  eval_paper core_eval 'source_eval,expert,table1'
  train extra_train 'table2,fig5,ablation,sensitivity'
  eval_paper extra_eval 'table2,fig5,ablation,sensitivity'
  "$PYTHON" -u "$HERE/car_eval.py" --car-module "$HERE/car.py"     "${COMMON[@]}" --aggregate --eval-stages all     2>&1 | tee -a "$OUT/command_logs/aggregate.log";;
 *) echo "Unknown phase: $PHASE" >&2; helpmsg; exit 2;;
esac
