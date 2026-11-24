# Transfer What Is Caused and Covered: Policy Transfer for Safe End-to-End Driving

**CaST — Causal-Support Transfer** · CARLA 0.9.15 · PyTorch · Closed-loop policy adaptation

![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)

CaST studies a specific failure mode in transferring an end-to-end driving policy: **a source behavior may be useful in one interaction and misleading in another, even within the same target scene**. Instead of treating the entire target observation as equally transferable, CaST estimates (i) whether an entity affects a safe driving decision and (ii) whether its interaction is represented in source experience. Those estimates influence alignment, uncertainty, and selection of adaptation episodes.

This repository provides the CARLA implementation, frozen-policy evaluation code, selected B20 checkpoints, route-level summaries, ablation files, and illustrative figures. The README emphasizes the **implementation and experiment interface**, rather than reproducing the paper's derivations.

> **Research status.** Results are descriptive and protocol-specific. The complete seed-0 evaluations and the currently archived seed-1 subset must not be interpreted as a fully replicated multi-seed study. The checkpoint bundle contains *adapted actors*, not necessarily all source-perception artifacts required to re-create or evaluate them from an empty directory. This is simulator research code, **not a road-ready driving system**.

## 1. Method at a glance

**Framework diagram:** [CaST system overview (PDF)](figures/cast-framework.pdf).

The driving state contains an ego token `x₀`, detected entity tokens `xᵢ`, validity mask `mᵢ`, and a pooled scene representation `b(s)`. The policy operates over their relational representation. A separate frozen source projection, `W_S`, gives a stable reference for support queries during target adaptation.

| Signal | Meaning in the implementation | What uses it |
|:--|:--|:--|
| `ĉᵢ` | Predicted effect of an intervention on entity `i`, supervised by simulator probes | Alignment, invariance, target scenario scoring |
| `ρᵢ` | Rank-based coverage of the entity interaction in a frozen source feature bank | Alignment gate `βᵢ` |
| `ρ̃ᵢ` | Coverage after accepted target-bank additions | Uncertainty floor `ℓ̃ᵢ` |
| `ρ₀`, `ρ̃₀` | Analogous scene-level coverage, including contexts where entities are missed | Scene uncertainty |
| `ū(s)` | Fused uncertainty from motion, coverage, and value disagreement | SAC entropy and cautious value estimation |
| `U(s)` | Unresolved causal-support mass | CaST target-template sampler |

The intended decisions are **align** when a relevant interaction is covered, **retain uncertainty** when relevant but uncovered, and **reduce reliance** when an interaction has low estimated causal effect. These are *training signals*, not formal guarantees of route safety or causal identifiability.

**Decision-map diagram:** [Causal support map (PDF)](figures/support_map.pdf).

## 2. Repository map

```text
CaST/
├── code/
│   ├── car.py                # environment, perception, policy, training, adaptation, aggregation
│   └── car_eval.py           # frozen-checkpoint route evaluation and CSV export
├── checkpoints/
│   ├── CHECKPOINT_INDEX.csv  # seed, suite, original paths, hashes
│   ├── seed0/{A,L,C}/.../agent.pt
│   └── seed1/{A,L,C}/.../agent.pt
├── script/                  # archived experiment launchers; see SCRIPT_INDEX.txt
├── results/
│   ├── paper_csv/           # selected paper-facing tables and provenance
│   ├── CaST_supplement_snapshot/
│   └── cast_b20_*/          # source experiment CSV exports, organized by run family
├── figures/                 # method / supplementary visuals
├── graphs/                  # reported experiment plots
├── screenshot/              # CARLA images
├── video/                   # CARLA recordings (Git LFS)
└── LICENSE
```

**Start here:** `code/car.py` for the algorithm, `code/car_eval.py` for evaluation, `results/paper_csv/10_result_provenance.csv` for result provenance, and `checkpoints/CHECKPOINT_INDEX.csv` for archived weights. The result folders include historical, incomplete, and summary-only CSVs; a filename's presence is **not** a completion certificate.

## 3. Execution flow — from cameras to the adapted actor

```text
CARLA world / route / traffic / adverse weather
       │
       ├── six surrounding RGB views (and depth in the rgbd configuration)
       │         │
       │         └── frozen detector / segmentation + token tracking
       │                 ├── ego scalars x₀
       │                 ├── per-entity features xᵢ and valid mask
       │                 └── pooled scene context b(s)
       │
       └── relational actor
                 ├── attention wᵢ and embedding hᵢ
                 ├── motion head → displacement + predicted variance
                 ├── causal head → ĉᵢ
                 └── SAC policy → action a

Source artifacts (frozen)          Target adaptation (trainable)
  W_S, source policy π_S            actor / critics / auxiliary heads
  interaction bank Z_S         →    βᵢ, ℓ̃ᵢ, ū, U(s)
  scene bank B_S                     selective alignment + invariance
  null intervention bank            B20 scenario sampling and replay updates
                                        │
                                        ▼
                                  adapted agent.pt
                                        │
                                        ▼
                             car_eval.py (no learning)
```

The released code supports multiple observation interfaces via `--perception`: `camera` (six cameras and the frozen BEV detector), `rgbd` (RGB/depth cameras and frozen segmentation/tracking), and `gt` (a noisy privileged-token diagnostic). **Do not combine results from these interfaces as if they were the same sensing protocol.** The B20 experiments described here use the camera-derived RGB-D research pipeline; reproduce the precise saved configuration from the run metadata rather than relying on command defaults.

### 3.1 Observation, detection, and tracking

`CameraRig` receives synchronized multi-view sensor data. `BEVDetector` transforms image evidence into spatial detections; optional RGB-D components `SegNet`, `rgbd_detect`, and `TokenTracker` construct and maintain object identities across frames. `CaSTCarlaEnv` then combines detected objects with ego motion and route information into the tensors consumed by the actor.

Code navigation:

```python
# code/car.py: key entry points (names, not a drop-in execution script)
CameraRig                     # camera callbacks and frame retrieval
BEVDetector                   # camera-to-BEV detections
SegNet, rgbd_detect           # RGB-D segmentation / clustering alternative
TokenTracker                  # object association and short occlusions
CaSTCarlaEnv._rgbd_entity_features
CaSTCarlaEnv._tokens_from     # entity/scene features
RelationalEncoder             # attention + interaction embedding
CaSTActor                     # action distribution and motion head
```

The model observes **camera-derived** tokens in the camera modes. Simulator actor states also exist in the code for teacher supervision, intervention construction, and auditing. Such privileged information must not be confused with ordinary inference inputs. `--light-source map_camera` associates a relevant traffic light using map geometry but reads the light color from camera evidence; `v2i` accesses broadcast state and is a different, assisted configuration.

### 3.2 Relational representation and policy

`RelationalEncoder` embeds the per-object features and computes masked attention. Schematically:

```python
h_i = W_theta(x_i)
logit_i = relevance(x_0, x_i) - λ_p * ||p_i||² - λ_d * log(σ_i² + ε)
w_i = masked_softmax(logit_i, valid=m_i)
s = concat(sum_i(w_i * h_i), x_0)
```

The actor uses a bounded squashed-Gaussian action distribution, and an ensemble of critics estimates action values. `CaSTActor.motion_head` additionally predicts each entity's relative displacement and variance. The motion variance is an aleatoric channel; critic disagreement is a separate epistemic channel. The exact action interpretation depends on the saved `--lateral-control` setting. Under `controller`, the route tracker supplies steering while the learned policy determines longitudinal behavior. Under `policy`, steering is learned. **Evaluation enforces compatibility with the checkpoint**, so do not switch between the two modes when reporting a result.

### 3.3 Source coverage and the alignment gate

`CoverageModel` queries frozen-source k-nearest-neighbor distances. The rank of a target distance relative to held-out source distances defines `ρ`, rather than a hand-selected Euclidean cutoff. Source entity features and whole-scene features have separate banks.

The implemented gate is compact:

```python
# code/car.py: gate_and_floor()
beta = clamp(rho / tau_rho - 1, 0, 1)
ell  = clamp(1 - rho / tau_rho, 0, 1)

# CaSTAgent: thresholds derive from a run specification
τ_c   = 1 - q
τ_rho = q
```

`βᵢ` weights source alignment; `ℓ̃ᵢ` is computed from updated support for the uncertainty channel. Importantly, **the frozen source ranks and the growing target bank play different roles**. Adding target observations should not rewrite the original source reference distribution.

The run's `q` is consequential: with `q=0.05`, for example, the coverage gate saturates at `ρ ≥ 0.10`. This is an implementation property, not evidence that support ranks are calibrated probabilities of correct driving.

### 3.4 Physical interventions and causal supervision

The causal estimator is not an attention score. Training-state probes alter one scene element while attempting to hold the remaining scenario comparable. `expert_action`, `rollout_margin`, `apply_op`, and `entity_effects` evaluate the change in a privileged decision and a short-horizon safety margin. `null_probe_effects` and `NullBank` provide reference effects for distance-conditioned ranking. `Labeler` schedules intervention work; `CausalDataset` stores labels; `CausalHead` predicts `ĉᵢ` during routine execution.

```python
# Conceptual dataflow, not an exact function signature
factual  = (expert_action(scene), rollout_margin(scene))
changed  = apply_op(scene, entity=i, intervention=omega)
probe    = (expert_action(changed), rollout_margin(changed))
raw_i    = action_gap(factual, probe) + safety_margin_gap(factual, probe)
c_i      = null_bank.rank(raw_i, distance_bin_i)
loss_E   = mean((causal_head(x_0, x_i) - c_i)**2)
```

Token removal also appears as a differentiable proxy in the invariance objective. **Removing an input token is not equivalent to a physical simulator intervention**; the code distinguishes those operations. The repository's summary CSVs do not by themselves establish held-out causal-label accuracy.

### 3.5 Updating the policy

`CaSTAgent.update` orchestrates the SAC losses, actor regularization, causal relevance, and uncertainty. The policy term is based on a cautious value estimate, while transfer terms can preserve or suppress particular source relationships.

```python
# Conceptual decomposition; coefficients and branches are run-dependent
L_total = L_SAC + λ_A * L_align + λ_I * L_invariance
                + λ_M * L_motion + λ_E * L_effect
                + supervised_imitation_terms
```

Several **important implementation choices** are configurable:

| Option | Code behavior | Why it matters |
|:--|:--|:--|
| `--align-metric action` | Squared difference in *squashed action space* | Default experimental transfer distance; not policy KL |
| `--align-metric kl` | Gaussian policy divergence | Alternate / earlier alignment formulation |
| `--uncertainty-fusion mean` | Averages active uncertainty contributions | Does not guarantee monotonicity or calibration |
| `--uncertainty-fusion sum` | Additive fusion | Can saturate; a separate protocol choice |
| `--critic-target lcb` | Mean target value with disagreement penalty | Changes the value-learning target |
| `--target-state-bc-weight` | Enables optional privileged imitation on selected target states | Assisted setting; report separately when enabled |
| `--use-safety-shield` | Optional evaluation-time shield | **Not** the unshielded B20 paper protocol |

The manuscript's mathematical exposition is an abstract method description. For an exact replication, the **run configuration and these implementation branches take precedence**; do not claim that an `action`-distance experiment directly evaluated a KL objective.

### 3.6 Choosing target episodes

`TemplateSampler` scores condition templates using the selected sampling rule. For CaST, the score comes from mean unresolved causal mass, `U(s)`; for other variants, sampling may be uniform or uncertainty-based. Scores are updated after episodes, and high-scoring templates can generate local variants in traffic load, weather, and degradation.

```python
# Simplified version of TemplateSampler.sample / update
score[m] = running_mean(observed_U_for_template_m)
prob[m]  = (1 - eps) * softmax(score / temperature)[m] + eps / M
chosen   = sample(prob, constrained_by_current_town_and_weather)
```

To limit CARLA town switches, `run_adaptation` collects episodes in town blocks. A separate balancing option can allocate initial visits across weather conditions. The fixed target budget `B` counts **completed adaptation episodes**, not failed simulator startups or interrupted rollouts.

**Coverage-bank acceptance in the released code:** the acceptance condition uses safety statistics of the **just-collected adaptation episode** (`collision` and `min_phi`), rather than an independently executed validation episode. The README states this explicitly because it affects interpretation of the algorithm's *validated growth* terminology.

## 4. Code tour: where to start reading

The numbers below are approximate line positions in the uploaded 2026-11-24 version of `code/car.py`; search for the names rather than depending on offsets.

| Component / callable | Approx. line | Purpose |
|:--|--:|:--|
| `Config` | 573 | Environment, model, optimizer, simulator and protocol defaults |
| `CaSTCarlaEnv` | 5667 | Conditions, observations, route execution, events and safety measurements |
| `CameraRig`, `BEVDetector`, `SegNet`, `TokenTracker` | 4834–5464 | Sensor-to-entity processing |
| `entity_effects`, `null_probe_effects`, `NullBank` | 7251–7304 | Physical probes and reference-effect distributions |
| `RelationalEncoder`, `CaSTActor`, `CaSTCritic`, `CausalHead` | 7491–7762 | Policy and auxiliary networks |
| `CoverageModel`, `gate_and_floor` | 7818–7866 | Rank support, source/target bank queries and gates |
| `CaSTReplay`, `RunSpec` | 7876–8045 | Training transitions and experiment identity |
| `CaSTAgent` | 8119 | Learning, acting, uncertainty, transfer losses and checkpoint loading |
| `build_suites` | 8806 | Definition of source, appearance, layout and combined shifts |
| `train_episode` | 9632 | A single episode, supervision, rewards and actor/critic updates |
| `run_rgbd_perception_stage`, `run_source_stage` | 10323–10597 | Perception and source-policy stages |
| `SourceArtifacts`, `make_run_agent` | 10739–10862 | Frozen source models, feature banks and run initialization |
| `TemplateSampler`, `run_adaptation` | 10912–10981 | Budgeted target sampling, checkpoint/resume, bank growth |
| `evaluate_suite` | 11239 | Deterministic held-out route evaluation |
| `plan`, `aggregate` | 11551, 11731 | Experimental sweep construction and reporting |
| `supervisor_main` | 12376 | CARLA process recovery and map-switch orchestration |

### `code/car_eval.py`: two evaluation interfaces

`car_eval.py` dynamically imports its sibling `car.py` and explicitly checks that the necessary functions exist. Keep **both files from the same revision**.

| Entry point | Role |
|:--|:--|
| `load_car_module` | Imports the matching training module |
| `load_agent`, `freeze` | Loads finished adaptation weights and disables gradients |
| `evaluate_town` | Legacy suite/town evaluation, resumable JSON records |
| `pilot_routes`, `drive_pilot_episode` | Pilot-style fixed-seed route evaluation |
| `evaluate_pilot_town` | Source/adapted pilot comparison |
| `aggregate_main` | Rebuilds summaries from completed evaluations |
| `supervisor_main` | Runs towns with simulator restarts / isolation |

The evaluator has **two output protocols**. The pilot interface (the evaluator's default) writes tagged route/episode files. The archived B20 paper launchers use the **legacy paper-stage interface** with per-town `eval_<suite>__<town>.json` and derived CSVs. Changing `--eval-stages`, the route seed, or the experiment tag changes what is being measured.

## 5. Experimental protocol and released evidence

The paper studies three controlled transfer conditions. All four methods receive the same nominal **B20 adaptation budget** in the matched comparisons: SAC fine-tuning, Uniform alignment, Open-set entropy, and **CaST**.

| Suite | Domain change | Seed-0 matched held-out evaluation |
|:--|:--|:--|
| `A` — appearance | Source towns with adverse weather / visual conditions | Town03 + Town10HD; 20 routes per method/town, 160 method-route records |
| `L` — layout | Changed road layout, source-like weather | Town01 + Town04 + Town05; 10 routes per method/town, 120 records |
| `C` — combined | Changed town layout with adverse visual conditions | Town01 + Town04 + Town05; 20 routes per method/town, 240 records |
| `C` — four-town extension | Includes Town02 in addition to the three-town subset | 20 routes per method/town, 320 records |

**The four-town C evaluation overlaps with the three-town subset.** It is not an independent replication. Seed-1 Town03 appearance-shift results are archived separately; incomplete seed-1 towns are not pooled into a completed A/L/C comparison. The six stored CaST adaptation checkpoints do not imply that all their frozen policy evaluations are finished.

The primary metrics are driving score (`DS`), success rate (`SR`), route completion (`RC`), infraction score (`IS`), collisions per driven kilometer, minimum time-to-collision (`TTC`, capped at 10 s for reported means), and cross-track error (`CTE`). Route success and driving score measure different properties; a high success percentage does not automatically imply best DS or collision performance.

### Selected seed-0 measurements

| Suite / subset | SAC fine-tuning DS / SR | Uniform DS / SR | Open-set DS / SR | CaST DS / SR |
|:--|--:|--:|--:|--:|
| A — 2 towns, N=40/method | 95.74 / 92.50% | 91.00 / 85.00% | 90.76 / 80.00% | **88.09 / 60.00%** |
| L — 3 towns, N=30/method | 95.84 / 73.33% | 95.87 / 80.00% | 93.30 / 86.67% | **92.36 / 76.67%** |
| C — 3 towns, N=60/method | 93.60 / 81.67% | 94.02 / 85.00% | 92.17 / 86.67% | **90.79 / 86.67%** |
| C — 4 towns, N=80/method | 94.23 / 82.50% | 92.33 / 77.50% | 91.71 / 83.75% | **90.20 / 83.75%** |

These values are reported for transparency, **not as claims that CaST dominates the baselines**. CaST is competitive on certain route-success and safety indicators, while some baselines achieve higher mean driving scores. See the route-level exports for failure cases and matched comparisons.

For the main-paper CSV views, inspect:

```bash
ls results/paper_csv/
head -n 8 results/paper_csv/02_main_transfer_A_L_C.csv
head -n 8 results/paper_csv/10_result_provenance.csv
head -n 8 results/CaST_supplement_snapshot/00_file_status.csv
```

The Town04 ablation CSV is an **aggregate-only** source in the released collection; direct intervention-label precision/recall, learning-curve claims, and complete across-seed inference require additional raw evidence.

## 6. Setup and verification

**Platform used:** Ubuntu + CARLA **0.9.15**, a Python environment with PyTorch/NumPy and the CARLA Python API; a GPU is recommended for camera perception. Exact wheel compatibility depends on the local CARLA distribution. The repository does **not** include a pinned, tested `requirements.txt` or a bundled simulator installation. Install these separately and check the local environment before running costly experiments.

```bash
# From the repository root
cd ~/Desktop/CaST_publish

# Replace with your own environment path if different
PYTHON=/home/umb/miniconda3/envs/safe/bin/python

# Confirm CLI and run the model-only CPU checks (does not start CARLA)
"$PYTHON" code/car.py --help
"$PYTHON" code/car_eval.py --help
"$PYTHON" code/car.py --learning-smoke-test --cpu
```

The repository places source code under `code/`, unlike the original experiment directories where the scripts and `car.py` lived together. **Run the launchers in an environment whose relative paths have been checked**. A copied `.sh` file may still refer to the original B20 folder layout and is not a portable one-command reproduction script.

### Checkpoint inventory

The archive has six CaST target adaptation checkpoints:

```text
checkpoints/seed0/A/A__cast__B20__s0__sampler-cast/agent.pt
checkpoints/seed0/L/L__cast__B20__s0__sampler-cast/agent.pt
checkpoints/seed0/C/C__cast__B20__s0__sampler-cast/agent.pt
checkpoints/seed1/A/A__cast__B20__s1__sampler-cast/agent.pt
checkpoints/seed1/L/L__cast__B20__s1__sampler-cast/agent.pt
checkpoints/seed1/C/C__cast__B20__s1__sampler-cast/agent.pt
```

Weights are tracked through Git LFS. After cloning, use `git lfs pull` before loading a checkpoint. Each `.pt` is a **state dictionary bundle**, not a complete standalone evaluation environment; `SourceArtifacts` also expects compatible source actor, perception weights, banks and calibration metadata.

```bash
# Confirm LFS objects were fetched and verify saved hashes
git lfs pull
git lfs ls-files
python - <<'PY'
import csv, hashlib
from pathlib import Path
manifest = Path('checkpoints/CHECKPOINT_INDEX.csv')
with manifest.open(newline='') as f:
    for row in csv.DictReader(f):
        print(row.get('seed'), row.get('suite'), row.get('sha256', '')[:16])
PY
```

The manifest stores the original training paths for provenance; paths on another machine will differ. Run **frozen evaluations only on completed, protocol-compatible checkpoints**. The evaluator checks training settings such as steering/light input, and new configuration values should not be silently substituted for the saved ones.

## 7. Running a new experiment

For a **new run**, use the checked-in training interface with a **fresh output directory** and explicit ports. The example below describes CLI structure; it is **not** a command that reproduces the released B20 tables without their source/perception stages and matching config.

```bash
PYTHON=/home/umb/miniconda3/envs/safe/bin/python
CARLA="$HOME/carla/CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=2200"

"$PYTHON" code/car.py \
  --profile paper \
  --stages perception,source,pilot \
  --perception rgbd \
  --seed 0 --budget 20 \
  --route-length 200 \
  --port 2200 --tm-port 8000 \
  --out-dir "$HOME/Desktop/cast_new_experiment" \
  --carla-cmd "$CARLA"
```

For the **released 3-shift B20 protocol**, the original runners are archived under `script/`. Their source and evaluation destinations were independent directories, and the evaluator was invoked once per selected suite/town. Before using a runner, open it and verify `PROJECT`, `OLD`, `TRAIN_OUT`, `EVAL_OUT`, `PYTHON`, `CARLA_SH`, `PORT`, `TM_PORT`, `SUITES`, and `EVAL_TOWN`. A single `agent.pt` cannot replace an initialized experiment directory.

**Safe workflow:** validate the local CARLA maps and sensor settings → produce or restore source artifacts → initialize independent adaptation run directories → train the four methods to their B20 markers → evaluate the planned held-out routes without gradients → audit per-town JSON keys and route counts → regenerate CSVs. Do not write to the archived experiment directories when trying a new hyperparameter.

## 8. Reading the evaluation files

The legacy evaluator writes route JSON files at positions like:

```text
<eval-out>/runs/<run-id>/eval_A__Town03.json
<eval-out>/runs/<run-id>/eval_L__Town01.json
<eval-out>/runs/<run-id>/eval_C__Town05.json
```

Each route record contains its route identifier and driving condition as well as score and safety fields. A **matched comparison** requires the same `(route, conditions)` keys across all four methods for a given suite/town. Do not compare unpaired runs by assuming the same row number is the same route.

```python
# Minimal audit of one already exported town (supply actual paths)
import json
from pathlib import Path

files = [Path(p) for p in [
    '/path/to/cast/eval_A__Town03.json',
    '/path/to/finetune/eval_A__Town03.json',
    '/path/to/uniform/eval_A__Town03.json',
    '/path/to/openset/eval_A__Town03.json',
]]
keys = []
for file in files:
    rows = json.loads(file.read_text())['routes']
    k = {(str(r['route']), json.dumps(r['conditions'], sort_keys=True)) for r in rows}
    assert len(k) == len(rows), f'duplicate route key: {file}'
    keys.append(k)
assert all(k == keys[0] for k in keys[1:]), 'route/condition mismatch'
print(f'MATCHED: {len(keys[0])} routes per method')
```

For every score, report the **suite, town set, seed, budget, route count, shield setting, and observation interface**. Report missing and partial files as such. The number of evaluation routes is not the number of independent training seeds.

## 9. Figures and demonstrations

| Asset | Description |
|:--|:--|
| `figures/cast-framework.pdf` | System overview: perception → support → interventions → adaptation |
| `figures/support_map.pdf` | Interaction-level transfer decision map |
| `figures/supp_ablation.pdf` | Town04 component ablation visualization |
| `figures/supp_coverage_bins.pdf` | Coverage-bin failure diagnostic |
| `graphs/validation_*.pdf` | Driving-score, weather, safety and ablation summaries |
| `screenshot/`, `video/` | Example CARLA execution outputs, not substitutes for route metrics |

<a href="video/1.mp4">Video 1</a> · <a href="video/2.mp4">Video 2</a> · <a href="video/3.mp4">Video 3</a>

## 10. Reproducibility boundaries and limitations

The codebase includes experimental branches beyond the evaluated paper configuration. Specific limitations matter when using it as a reference implementation:

- **Partial evaluations:** archived seed-1 results do not establish complete two-seed A/L/C statistics. Existing seed-1 CaST checkpoint files alone do not mean four-method matched evaluations are complete.
- **Source dependencies:** the repository checkpoint subset does not include every detector, frozen-source, replay, and support-bank artifact. Full offline reproduction from the six `.pt` files alone is not supported.
- **Implementation versus idealized training order:** bank acceptance currently examines the adaptation episode's measured safety, not a second, independently driven validation rollout.
- **Alignment and fusion:** the current `action` metric and `mean` fusion differ from an exclusively KL-based alignment or additive uncertainty description. Log these choices explicitly.
- **Causal evidence:** simulator interventions define training signals, but reported aggregate driving outcomes alone do not validate causal-head accuracy or shortcut removal.
- **Perception and assisted settings:** map-assisted traffic-light association, `v2i`, ground-truth tokens, target expert labels and shielded evaluation alter the observation/supervision protocol; distinguish them from unassisted camera-based results.
- **Statistical scope:** route-level uncertainty estimates are correlated within a policy run; do not present them as between-seed confidence intervals.

We welcome comparisons that isolate these components under the same adaptation budget and **identical held-out route/condition keys**.

## 11. Citation and use

If you use this repository, cite the associated paper once its public bibliographic record is available. A provisional BibTeX entry can be adapted after acceptance/publication:

```bibtex
@misc{cast_policy_transfer,
  title  = {Transfer What Is Caused and Covered: Policy Transfer for Safe End-to-End Driving},
  note   = {Research manuscript and associated implementation},
  url    = {https://github.com/umborhan/CaST}
}
```

This entry deliberately does **not** claim an acceptance venue, DOI, or publication year. For use and redistribution conditions, see [`LICENSE`](LICENSE). CARLA and third-party packages retain their respective licenses.
