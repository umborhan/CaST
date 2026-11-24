#!/usr/bin/env bash
# One-seed, matched B20 shortcut and ablation studies for the CaST CVPR draft.
# Uses an isolated output directory and an isolated copy of the algorithm sources.
set -euo pipefail

PHASE="${1:-help}"
PROJECT="${PROJECT:-$HOME/Desktop/cvpr2027}"
RUNNER="${RUNNER:-$PROJECT/cast_b20_seed0_runner}"
ORIGINAL="${ORIGINAL:-$PROJECT/cast_paper_rgbd_B20_v15}"
OUT="${OUT:-$PROJECT/cast_b20_tables_s0}"
CARLA_SH="${CARLA_SH:-$HOME/carla/CarlaUE4.sh}"
PORT="${PORT:-2200}"
TM_PORT="${TM_PORT:-8000}"
PYTHON="${PYTHON:-python}"
CODE="$OUT/code"
CARLA="$CARLA_SH -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$PORT"

err(){ echo "ERROR: $*" >&2; exit 2; }
link_once(){
  local src="$1" dst="$2"
  [[ -e "$src" ]] || err "Missing source artifact: $src"
  if [[ -L "$dst" ]]; then
     [[ "$(realpath -e "$dst")" == "$(realpath -e "$src")" ]] || err "$dst points to another location"
  elif [[ -e "$dst" ]]; then
     err "$dst exists and is not a symlink; refusing to change it"
  else
     ln -s "$(realpath -e "$src")" "$dst"
  fi
}
copy_checkpoint(){
  local id="$1" src="$ORIGINAL/runs/$1" dst="$OUT/runs/$1" f
  [[ -f "$src/ADAPT_DONE" && -s "$src/agent.pt" ]] || err "Completed B20 checkpoint missing: $src"
  mkdir -p "$dst"
  for f in ADAPT_DONE agent.pt growth.pt adapt.json; do
    if [[ -f "$src/$f" ]]; then
      [[ -e "$dst/$f" ]] || cp -p "$src/$f" "$dst/$f"
    fi
  done
}
init(){
  [[ -f "$RUNNER/car.py" && -f "$RUNNER/car_eval.py" ]] || err "Priority runner not found: $RUNNER"
  [[ -f "$ORIGINAL/source/DONE" && -f "$ORIGINAL/perception/DONE" ]] || err "Frozen source/perception missing: $ORIGINAL"
  mkdir -p "$OUT" "$CODE" "$OUT/runs" "$OUT/results" "$OUT/command_logs"
  link_once "$ORIGINAL/source" "$OUT/source"
  link_once "$ORIGINAL/perception" "$OUT/perception"
  [[ -s "$ORIGINAL/run_config.json" ]] || err "Original training protocol record missing: $ORIGINAL/run_config.json"
  [[ -s "$OUT/run_config.json" ]] || cp -p "$ORIGINAL/run_config.json" "$OUT/run_config.json"
  if [[ ! -f "$CODE/car.py" ]]; then
    cp "$RUNNER/car.py" "$CODE/car.py"
    "$PYTHON" - "$CODE/car.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = 'PROFILES["robust"]["seeds_table1"] = 1'
if needle not in s:
    raise SystemExit('Could not locate robust profile one-seed anchor; inspect source version')
s = s.replace(needle, needle + '\nPROFILES["robust"]["seeds"] = 1  # targeted table tests, s0 only', 1)
# Only run the rows needed for the two user-facing tables. Keep labels explicit
# when a proxy variant is not identical to a strict no-intervention ablation.
start = s.index('ABLATIONS: List[Tuple[str, Dict[str, object]]] = [')
end = s.index('SENSITIVITY: List[Tuple', start)
s = s[:start] + '''ABLATIONS: List[Tuple[str, Dict[str, object]]] = [
    ("Without coverage gate", {"gate": False}),
    ("Without epistemic channel", {"epistemic": False}),
    ("Attention replacing causal effect", {"relevance": "attention"}),
    ("Without invariance", {"invariance": False}),
    ("Uniform sampling", {"sampler": "uniform"}),
    ("Full CaST", {}),
]
''' + s[end:]
start = s.index('SHORTCUT_ROWS: List[Tuple[str, str, Dict[str, object]]] = [')
end = s.index('SAMPLERS =', start)
s = s[:start] + '''SHORTCUT_ROWS: List[Tuple[str, str, Dict[str, object]]] = [
    ("Uniform transfer", "uniform", {}),
    ("Causal baseline (CausalVAD-style)", "deconf", {}),
    ("CaST", "cast", {}),
]
''' + s[end:]
p.write_text(s)
print('One-seed plan patched: 3 shortcut methods; 5 ablations + full CaST.')
PY
  fi
  if [[ ! -f "$CODE/car_eval.py" ]]; then cp "$RUNNER/car_eval.py" "$CODE/car_eval.py"; fi
  "$PYTHON" -m py_compile "$CODE/car.py" "$CODE/car_eval.py"
  # Copy already completed checkpoints to skip retraining full CaST and uniform.
  # Evaluation results remain separate from the original B20 runs.
  copy_checkpoint 'C__cast__B20__s0__sampler-cast'
  copy_checkpoint 'C__uniform__B20__s0'
  echo "[READY] One-seed B20 shortcut/ablation code: $CODE"
  echo "[READY] Output isolated at: $OUT"
  echo "[READY] Reused frozen RGB-D and source; copied complete CaST/uniform checkpoints"
  echo "[NOTE] Source directory is linked; model weights are reused, and optional variant-specific source caches may be added."
}
COMMON=(
  --profile robust --port "$PORT" --tm-port "$TM_PORT" --perception rgbd
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

case "$PHASE" in
  init)
    init
    ;;
  train)
    init
    echo '[START] Training missing seed-0 B20 shortcut/ablation variants. Do not run another CARLA job on this port.'
    "$PYTHON" -u "$CODE/car.py" "${COMMON[@]}" --out-dir "$OUT" --stages table2,ablation --carla-cmd "$CARLA" \
      2>&1 | tee -a "$OUT/command_logs/table_training.log"
    ;;
  eval)
    init
    echo '[START] Paired C-shift, barrel and queue evaluation; no safety shield.'
    "$PYTHON" -u "$CODE/car_eval.py" --car-module "$CODE/car.py" "${COMMON[@]}" \
      --out-dir "$OUT" --town all --eval-stages table2,ablation --carla-cmd "$CARLA" \
      2>&1 | tee -a "$OUT/command_logs/table_eval.log"
    ;;
  aggregate)
    init
    "$PYTHON" -u "$CODE/car_eval.py" --car-module "$CODE/car.py" "${COMMON[@]}" \
      --out-dir "$OUT" --aggregate --eval-stages table2,ablation \
      2>&1 | tee -a "$OUT/command_logs/table_aggregate.log"
    ;;
  status)
    [[ -d "$OUT/runs" ]] || err "Run 'init' first"
    "$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
rs=list((root/'runs').glob('*'))
done=sum((r/'ADAPT_DONE').is_file() for r in rs if r.is_dir())
print(f'Completed adaptation run folders: {done}/{len([r for r in rs if r.is_dir()])}')
for name in ('table2_shortcut.csv','table5_ablation.csv'):
 p=root/'results'/name
 print(f'{name}: {"available" if p.is_file() else "not yet generated"}')
PY
    ;;
  audit)
    "$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import csv,sys,math
root=Path(sys.argv[1])/'results'
for name in ('table2_shortcut.csv','table5_ablation.csv'):
 p=root/name
 print('\n'+name)
 if not p.exists():
  print('  Not generated');continue
 with p.open() as f:rows=list(csv.DictReader(f))
 for row in rows:
  if name.startswith('table2'):
   counts=[int(float(row.get(k,'0') or '0')) for k in ('n_barrel_events','n_red_events','n_queue_events')]
   print(' ',row.get('method'), 'events(barrel,red,queue)=',counts,
         'False=',row.get('false_stop_pct'),'Miss=',row.get('missed_stop_pct'),
         'Prec=',row.get('precision_pct'),'Rec=',row.get('recall_pct'),
         'COMPLETE' if all(counts) else 'INCOMPLETE')
  else:
   print(' ',row.get('variant'), 'DS=',row.get('DS'),'Coll/km=',row.get('coll_per_km'),
         'AUROC=',row.get('auroc_lowest_bin'))
PY
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: bash run_cast_tables_s0.sh {init|train|eval|aggregate|status|audit}
One-seed B20 continuation for controlled shortcut and component ablation tables.
Run sequentially; do NOT run in parallel with other experiments using CARLA port 2200.
Source/perception artifacts are reused; full CaST and uniform checkpoints copied.
Additional variants require new B20 training; evaluation can take substantial time.
Causal precision and recall require matched barrel/queue trials, not standard route metrics.
Ablation labels are 'Without epistemic channel' and 'Attention replacing causal effect';
these are NOT strictly identical to 'without uncertainty floor'/'without intervention'.
EOF
    ;;
  *) err "Unknown phase: $PHASE";;
esac
