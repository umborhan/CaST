#!/usr/bin/env bash
# Run frozen suite-A/B20/s0 held-out evaluation without modifying run_b20_s0.sh.
# Usage: bash evaluate_A_B20_s0.sh
set -Eeuo pipefail

PROJECT="${PROJECT:-$HOME/Desktop/cvpr2027}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_DIR="${RUNNER_DIR:-$HERE}"
ORIGINAL="$RUNNER_DIR/run_b20_s0.sh"

export SUITES=A
export PORT=2300
export TM_PORT=8100
export PYTHON="${PYTHON:-/home/umb/miniconda3/envs/safe/bin/python}"
export TRAIN_OUT="${TRAIN_OUT:-$PROJECT/cast_b20_seed0_priority_train_A_parallel}"
export EVAL_OUT="${EVAL_OUT:-$PROJECT/cast_b20_seed0_priority_eval_A_parallel}"
export CARLA_SH="${CARLA_SH:-/home/umb/carla/CarlaUE4.sh}"

[[ -f "$ORIGINAL" ]] || { echo "[ERROR] Missing runner $ORIGINAL; copy this script into cast_b20_s0_3shift_runner/" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "[ERROR] Missing executable Python: $PYTHON" >&2; exit 1; }
[[ -f "$CARLA_SH" ]] || { echo "[ERROR] CARLA launcher missing: $CARLA_SH" >&2; exit 1; }
[[ -f "$RUNNER_DIR/car.py" && -f "$RUNNER_DIR/car_eval.py" ]] || {
    echo "[ERROR] This runner directory also needs car.py and car_eval.py" >&2; exit 1;
}

# Copy the existing runner to a temporary sibling. This intentionally does not
# modify the user's original run_b20_s0.sh or touch frozen checkpoints.
# Append TM_PORT to the COMMON Bash array just before the phase switch.
# This avoids relying on any particular order or whitespace of CLI flags.
# Appending last also overrides an earlier --tm-port if one was already present.
TEMP_RUNNER="$(mktemp "$RUNNER_DIR/.run_b20_s0_eval_A_XXXXXX.sh")"
trap 'rm -f "$TEMP_RUNNER"' EXIT
"$PYTHON" - "$ORIGINAL" "$TEMP_RUNNER" <<'PY'
from pathlib import Path
import re
import sys

source, destination = map(Path, sys.argv[1:])
s = source.read_text(encoding="utf-8")
if not re.search(r"\bCOMMON\s*=\s*\(", s):
    raise SystemExit(
        "[ERROR] run_b20_s0.sh has no COMMON=(...) array; "
        "the evaluation launcher cannot safely inherit its settings."
    )
# Typical runner form: case "$PHASE" in
switch = re.search(r"(?m)^[ \t]*case[ \t]+[^\n]*\bin[ \t]*(?:#.*)?$", s)
if switch is None or 'PHASE' not in switch.group(0):
    raise SystemExit(
        "[ERROR] Could not identify the runner phase switch ('case ...PHASE... in'). "
        "Original runner was not modified; share run_b20_s0.sh for a direct patch."
    )
# A standalone statement before the phase switch works with multi-line and
# single-line COMMON arrays, and keeps the original file intact.
patch = (
    '\n# Added ONLY to this temporary evaluation runner (not the original).\n'
    'COMMON+=(--tm-port "$TM_PORT")\n'
    'echo "[TM CHECK] Evaluation CLI now uses --tm-port $TM_PORT"\n'
)
s = s[:switch.start()] + patch + s[switch.start():]
destination.write_text(s, encoding="utf-8")
print("[READY] Temporary runner created; Traffic Manager port appended to COMMON.")
PY
bash -n "$TEMP_RUNNER"

echo "[CONFIG] Original runner left unchanged: $ORIGINAL"

# Verify files for every adapted method, not merely training log lines.
"$PYTHON" - "$PROJECT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])/'cast_paper_rgbd_B20_v15'/'runs'
ids = ['A__cast__B20__s0__sampler-cast', 'A__finetune__B20__s0',
       'A__uniform__B20__s0', 'A__openset_entropy__B20__s0']
for name in ids:
    folder = root/name
    missing = [f for f in ('ADAPT_DONE','agent.pt') if not (folder/f).is_file()]
    if missing:
        raise SystemExit(f'[NOT READY] {name}: missing {", ".join(missing)}')
    print('[READY]', name)
PY

echo "[CONFIG] suite=A | CARLA RPC=$PORT | TM=$TM_PORT | 20 routes/town | seed 0"
echo "[CONFIG] training checkpoints: $PROJECT/cast_paper_rgbd_B20_v15/runs"
echo "[CONFIG] evaluation output:  $EVAL_OUT"
echo "[STATUS BEFORE]"
bash "$TEMP_RUNNER" status

echo "[EVAL] Starting 4 methods × 2 towns × 20 held-out routes = 160 route evaluations"
bash "$TEMP_RUNNER" eval

echo "[AGGREGATE] Rebuild suite-A summaries from completed JSON files"
bash "$TEMP_RUNNER" aggregate

echo "[STATUS AFTER]"
bash "$TEMP_RUNNER" status

echo "[AUDIT] Checking held-out JSONs if audit_A_B20_s0.py is available"
AUDIT="$RUNNER_DIR/audit_A_B20_s0.py"
if [[ -f "$AUDIT" ]]; then
    "$PYTHON" "$AUDIT" --out "$EVAL_OUT"
else
    echo "[TIP] Put audit_A_B20_s0.py beside this launcher, then run it separately."
fi
echo "[DONE] Evaluation outputs: $EVAL_OUT"
