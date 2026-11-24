#!/usr/bin/env bash
# Single-seed B20 ablations, one held-out evaluation town; reuse original Full CaST.
set -euo pipefail
PHASE="${1:-help}"
PROJECT="${PROJECT:-$HOME/Desktop/cvpr2027}"
RUNNER="${RUNNER:-$PROJECT/cast_b20_seed0_runner}"
ORIGINAL="${ORIGINAL:-$PROJECT/cast_paper_rgbd_B20_v15}"
PRIORITY_TRAIN="${PRIORITY_TRAIN:-$PROJECT/cast_b20_seed0_priority_train}"
PRIORITY_EVAL="${PRIORITY_EVAL:-$PROJECT/cast_b20_seed0_priority_eval}"
OUT="${OUT:-$PROJECT/cast_b20_priority_ablation_town04_s0}"
TOWN="${TOWN:-Town04_Opt}"
TOWN_BASE="${TOWN%_Opt}"
CARLA_SH="${CARLA_SH:-$HOME/carla/CarlaUE4.sh}"
PORT="${PORT:-2200}"
TM_PORT="${TM_PORT:-8000}"
PYTHON="${PYTHON:-python}"
CODE="$OUT/code"
CARLA="$CARLA_SH -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=$PORT"
CAST_ID="C__cast__B20__s0__sampler-cast"

err(){ echo "[ERROR] $*" >&2; exit 2; }
link_or_check(){
  local src="$1" dst="$2"
  [[ -e "$src" ]] || err "Missing file: $src"
  if [[ -L "$dst" ]]; then
    [[ "$(realpath -e "$dst")" == "$(realpath -e "$src")" ]] || err "Existing $dst points elsewhere"
  elif [[ -e "$dst" ]]; then
    err "Existing $dst is not a symlink; refusing overwrite"
  else
    ln -s "$(realpath -e "$src")" "$dst"
  fi
}

init(){
  [[ "$TOWN_BASE" =~ ^Town(01|02|04|05)$ ]] || err 'TOWN must be Town01_Opt, Town02_Opt, Town04_Opt or Town05_Opt'
  [[ -f "$RUNNER/car.py" && -f "$RUNNER/car_eval.py" ]] || err "Missing original runner code: $RUNNER"
  [[ -f "$ORIGINAL/source/DONE" && -f "$ORIGINAL/perception/DONE" ]] || err "Source/perception not complete in $ORIGINAL"
  [[ -f "$PRIORITY_TRAIN/runs/$CAST_ID/ADAPT_DONE" ]] || err "Full CaST B20 checkpoint missing in $PRIORITY_TRAIN"
  [[ -s "$PRIORITY_EVAL/runs/$CAST_ID/eval_C__${TOWN_BASE}.json" ]] || err "Full CaST $TOWN_BASE result missing in $PRIORITY_EVAL"
  mkdir -p "$CODE" "$OUT/runs/$CAST_ID" "$OUT/results" "$OUT/command_logs"
  link_or_check "$ORIGINAL/source" "$OUT/source"
  link_or_check "$ORIGINAL/perception" "$OUT/perception"
  if [[ ! -e "$OUT/run_config.json" ]]; then
    [[ -s "$ORIGINAL/run_config.json" ]] || err "Missing $ORIGINAL/run_config.json"
    cp -p "$ORIGINAL/run_config.json" "$OUT/run_config.json"
  fi
  if [[ ! -e "$CODE/car.py" ]]; then
    cp -p "$RUNNER/car.py" "$CODE/car.py"
    "$PYTHON" - "$CODE/car.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
anchor='PROFILES["robust"]["seeds_table1"] = 1'
assert anchor in s, 'One-seed anchor missing; runner version changed'
s=s.replace(anchor, anchor+'\nPROFILES["robust"]["seeds"] = 1  # one-town ablation, seed 0 only', 1)
start=s.index('ABLATIONS: List[Tuple[str, Dict[str, object]]] = [')
end=s.index('SENSITIVITY: List[Tuple', start)
s=s[:start]+'''ABLATIONS: List[Tuple[str, Dict[str, object]]] = [
    ("Without coverage gate", {"gate": False}),
    ("Attention replacing causal effect", {"relevance": "attention"}),
    ("Uniform sampling (no CaST sampler)", {"sampler": "uniform"}),
    ("Full CaST (reused)", {}),
]
'''+s[end:]
old='out.append(cast_spec("C", seed, B, cfg, eval_suites=("C", "barrel"), **kw))'
assert s.count(old)==1, 'Expected ablation eval suite anchor not found'
s=s.replace(old,'out.append(cast_spec("C", seed, B, cfg, eval_suites=("C",), **kw))',1)
p.write_text(s)
print('[READY] Patched: three key ablations + full CaST; no barrel/queue tests; seed 0')
PY
  fi
  if [[ ! -e "$CODE/car_eval.py" ]]; then cp -p "$RUNNER/car_eval.py" "$CODE/car_eval.py"; fi
  # The optional B200 proxy is not needed to measure DS/SR/RC/collisions.
  # If the evaluator still blocks C-shift without it, disable only that block.
  "$PYTHON" - "$CODE/car_eval.py" <<'PY_PROXY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); s = p.read_text()
old = '''            if es == "C" and spec.is_cast() and proxy is None:
                # Fig. 3's embedding gap needs the proxy run; evaluating now would store an empty gap column.
                skipped.append(f"{spec.run_id()} (suite C waits for the proxy run)")
                continue'''
if old in s:
    replacement = '''            if es == "C" and spec.is_cast() and proxy is None:
                print(f"[EVAL] {spec.run_id()}: B200 proxy unavailable; optional embedding-gap metric omitted")'''
    backup = p.with_suffix('.py.before_b20_proxy_fix')
    if not backup.exists(): backup.write_text(s)
    p.write_text(s.replace(old, replacement, 1))
    print('[FIX] Optional B200 proxy no longer blocks the C-shift driving evaluation')
PY_PROXY
  "$PYTHON" -m py_compile "$CODE/car.py" "$CODE/car_eval.py"
  local f
  for f in ADAPT_DONE agent.pt growth.pt adapt.json; do
    if [[ -e "$PRIORITY_TRAIN/runs/$CAST_ID/$f" ]]; then
      link_or_check "$PRIORITY_TRAIN/runs/$CAST_ID/$f" "$OUT/runs/$CAST_ID/$f"
    fi
  done
  # This existing held-out result is the SAME frozen full CaST model on the SAME town.
  link_or_check "$PRIORITY_EVAL/runs/$CAST_ID/eval_C__${TOWN_BASE}.json" \
                "$OUT/runs/$CAST_ID/eval_C__${TOWN_BASE}.json"
  echo "[READY] Existing Full CaST checkpoint + $TOWN_BASE eval reused (no retrain / reevaluation)."
  echo "[PLAN] 3 NEW B20 variants × 20 training episodes = 60 adaptation episodes."
  echo "[PLAN] 3 NEW variants × 20 held-out routes in $TOWN_BASE = 60 evaluations."
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
  init) init ;;
  train)
    init
    echo '[TRAIN] Training only the three priority B20 ablation variants.'
    "$PYTHON" -u "$CODE/car.py" "${COMMON[@]}" --out-dir "$OUT" \
      --stages ablation --carla-cmd "$CARLA" 2>&1 | tee -a "$OUT/command_logs/ablation_train.log"
    ;;
  eval)
    init
    "$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import importlib.util,sys
p=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('ca',p/'code'/'car.py')
# Do not import CARLA/torch here: simply check checkpoint completion markers.
expected = [
    'C__cast__B20__s0__sampler-cast',
    'C__cast__B20__s0__sampler-cast_gate-False',
    'C__cast__B20__s0__sampler-cast_relevance-attention',
    'C__cast__B20__s0',  # uniform sampler: default sampler has no suffix
]
missing = [name for name in expected if not (p/'runs'/name/'ADAPT_DONE').is_file()]
print(f'[CHECK] Completed ablation checkpoints: {len(expected)-len(missing)}/{len(expected)}')
if missing:
    raise SystemExit('Missing completed checkpoints: ' + ', '.join(missing))
PY
    echo "[EVAL] Only $TOWN_BASE (no other evaluation towns or shortcut tests)."
    "$PYTHON" -u "$CODE/car_eval.py" --car-module "$CODE/car.py" \
      "${COMMON[@]}" --out-dir "$OUT" --town "$TOWN" \
      --eval-stages ablation --carla-cmd "$CARLA" \
      2>&1 | tee -a "$OUT/command_logs/ablation_eval_${TOWN_BASE}.log"
    ;;
  aggregate)
    init
    "$PYTHON" "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/aggregate_one_town.py" \
       --out "$OUT" --town "$TOWN_BASE" --expect 20
    ;;
  status)
    [[ -d "$OUT/runs" ]] || err 'Run init first'
    "$PYTHON" - "$OUT" "$TOWN_BASE" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1])/'runs';town=sys.argv[2]
expected = [
 'C__cast__B20__s0__sampler-cast',
 'C__cast__B20__s0__sampler-cast_gate-False',
 'C__cast__B20__s0__sampler-cast_relevance-attention',
 'C__cast__B20__s0',
]
for name in expected:
 p=r/name
 if not p.is_dir(): continue
 done=(p/'ADAPT_DONE').is_file()
 st=p/'ckpt'/'state.json'
 try: ep=json.loads(st.read_text()).get('ep',0)
 except (OSError,ValueError,KeyError): ep=20 if done else 0
 ev=p/f'eval_C__{town}.json'
 partial=p/f'eval_C__{town}.partial.json'
 q=ev if ev.exists() else partial
 try: n=len(json.loads(q.read_text())['routes']) if q.exists() else 0
 except (OSError,ValueError,KeyError):n=0
 print(f'{p.name:62} train={ep}/20 done={done} eval={n}/20')
PY
    ;;
  help|-h|--help)
    cat <<'HELP'
Run: bash run_priority_ablation_s0.sh {init|status|train|eval|aggregate}
Default TOWN=Town04_Opt; override as: TOWN=Town05_Opt bash ... eval
Requires completed priority B20 CaST source/training/checkpoint and Town result.
Only three key variants train (60 total B20 adaptation episodes); only
one town is evaluated (60 NEW held-out routes). Original CaST results are reused.
Other towns are still included in adaptation to match original CaST training.
DO NOT run another CARLA job on the same port simultaneously.
'Attention replacing causal effect' is a proxy, not complete intervention removal.
HELP
    ;;
  *) err "Unknown phase: $PHASE";;
esac
