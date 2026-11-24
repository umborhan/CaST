#!/usr/bin/env python3
"""Frozen CaST evaluation: 20 episodes PER TOWN, Town10HD first.

Complete evaluator update v6 (2026-10-28):
  - Pair with the corrected car.py: stop-line crossing metrics, not post-line occupancy.
  - Existing compatible weights can be evaluated without loading optimizer state.
  - Reports penalty-free and intervention-free goals alongside ordinary goal success.
  - Saves signal event IDs/positions, collision details, and the final 3 s of actions.
  - --route-length 200 is forwarded to the matching car.py.
  - The pilot loop permits the environment's traffic/near-goal timeout extensions
    (2400 base steps, at most 4000 steps with the current 200 m configuration).
  - --allow-map-load resolves installed Town/_Opt aliases, loads a different map
    through car.py's isolated child process, then verifies the loaded world.
  - Model/perception artifacts are checked before starting or switching CARLA.
  - Use a new --eval-tag after updating; saved result hashes prevent mixing versions.
  - Completed episodes are preserved; repeated native TM startup crashes stop early.

Default mode evaluates the finished suite-A pilot (budget 10, seed 0) on the
same weather distribution in every town. No adaptation or causal-head fitting
is called. Put this file beside the car.py used to train the checkpoint.

    python car_eval.py --profile paper --out-dir ./cast_results_rgbd5 \
        --perception rgbd --port 2200 --town Town10HD --episodes 20 --budget 10

Then repeat with --town Town01 --allow-map-load, followed by Town02,
Town03, Town04, Town05. Alternatively use --town all --carla-cmd with your
CARLA launch command. --towns supplies an explicit ordered list; Town10HD is
always first for --town all. Installed _Opt aliases are resolved automatically.

--policies source,adapted compares both checkpoints on the same route seeds
(20 episodes EACH per town). Default: adapted only. --weather-set adverse
balances fog_dense, rain_heavy, night, glare: five episodes each at N=20.
Use --weather-set source for clear/cloudy weather, or mixed for both sets.

Results: <out-dir>/evaluation/<eval-tag>/<adapted-run>/<weather>_N20_R70000/
    <town>/<policy>.json     episode checkpoint, saved after EVERY episode
    <town>/episodes.csv      episode metrics, reasons, seeds, route hashes
    summary.csv             one row per town/policy, including completion
Rerunning resumes completed episodes. Changed checkpoints/settings cannot
silently reuse results; use a new --eval-tag for a separate experiment.
The existing environment traffic-jam watchdog remains enabled; NPC removals
and safety interventions are reported. No teacher controls the ego vehicle.

The original paper evaluator remains available with --eval-stages all (or a
paper stage list). In that LEGACY mode --eval-routes is a per-suite total;
--episodes, --policies, --weather-set, --train-suite and --eval-tag do not apply.

Legacy paper evaluation (companion of car.py, which only trains):

One CARLA town per process.  The script loads the frozen checkpoints written by car.py (the
source policy, the privileged expert, and every adapted target-stage run), drives the evaluation
routes of every suite that lie in the requested town with deterministic actions, performs no
gradient updates, and stores the results per run, suite, and town:

    cast_results/runs/<run>/eval_<suite>__<town>.json   per-town result (resumable, crash safe)
    cast_results/runs/<run>/eval_<suite>.json           merged once every town of the suite is done
    cast_results/results/eval_<town>.csv                one row per route driven in that town
    cast_results/results/*.csv                          paper tables (--aggregate, or after --town all)

Usage
  Start CARLA in the town yourself and evaluate it:
      ./CarlaUE4.sh /Game/Carla/Maps/Town02 -quality-level=Low -nosound -carla-rpc-port=2200
      python car_eval.py --town Town02 --port 2200
  Or let the script start a fresh CARLA server per town and run all towns one after another:
      python car_eval.py --town all --port 2200 --carla-cmd "$HOME/carla/CarlaUE4.sh -quality-level=Low -nosound -carla-rpc-port=2200"
  Paper CSV files from whatever is finished:
      python car_eval.py --aggregate

Every option of car.py (--profile, --out-dir, --eval-routes, --source-towns, --layout-towns, ...) is
accepted and must match the training run.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import dataclasses
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def load_car_module(path_text: str):
    """Load car.py as a module without running its main function."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        sibling = Path(__file__).resolve().parent / path
        path = sibling if sibling.exists() else Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"car.py not found: {path}; put car_eval.py beside car.py or pass --car-module")
    spec = importlib.util.spec_from_file_location("cast_car", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("setup_run", "evaluate_suite", "plan", "base_spec", "make_run_agent", "run_adaptation",
                 "aggregate", "write_source_eval", "load_detector", "restart_carla_server", "EVAL_STAGES"):
        if not hasattr(module, name):
            raise ImportError(f"{path} is missing {name}; use the car.py that ships with this car_eval.py")
    return module


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(description="CaST evaluation: one CARLA town per process, frozen checkpoints",
                                epilog="All other options are passed to car.py's argument parser.")
    p.add_argument("--car-module", default="car.py")
    p.add_argument("--town", default="Town10HD", help="one town, default Town10HD; 'all' requires --carla-cmd")
    p.add_argument("--eval-stages", default="pilot",
                   help="pilot (default): frozen checkpoint in each town; or legacy paper stages / 'all'")
    p.add_argument("--episodes", type=int, default=20, help="pilot: completed episodes PER TOWN PER POLICY")
    p.add_argument("--towns", default="", help="pilot --town all: comma-separated towns; Town10HD goes first")
    p.add_argument("--policies", default="adapted", help="pilot: adapted, source, or source,adapted")
    p.add_argument("--train-suite", choices=("A", "L", "C"), default="A", help="suite of the saved CaST checkpoint")
    p.add_argument("--weather-set", choices=("adverse", "source", "mixed"), default="adverse")
    p.add_argument("--route-seed", type=int, default=70000, help="held-out route seed namespace")
    p.add_argument("--eval-tag", default="town20", help="separate experiment folder (letters, digits, _ or -)")
    p.add_argument("--aggregate", action="store_true", help="rebuild CSV summaries from saved evaluation results")
    p.add_argument("--eval-worker", action="store_true", help=argparse.SUPPRESS)
    own, rest = p.parse_known_args()
    if own.episodes <= 0 or own.episodes > 100000:
        p.error("--episodes must be between 1 and 100000")
    if not 0 <= own.route_seed < 2**31:
        p.error("--route-seed must be in [0, 2**31)")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", own.eval_tag):
        p.error("--eval-tag may contain only letters, digits, _ and -")
    own.policies = ",".join(dict.fromkeys(x.strip() for x in own.policies.split(",") if x.strip()))
    if not own.policies or any(x not in ("source", "adapted") for x in own.policies.split(",")):
        p.error("--policies must be adapted, source, or source,adapted")
    if not own.aggregate and not own.town:
        p.error("--town is required (or use --aggregate)")
    return own, rest


def car_args(car: Any, rest: Sequence[str]) -> argparse.Namespace:
    saved = sys.argv
    try:
        sys.argv = ["car.py"] + [a for a in rest if a != "--worker"]
        return car.parse_args()
    finally:
        sys.argv = saved


def tb(car: Any, town: str) -> str:
    name = str(town).replace("\\", "/").rstrip("/").split("/")[-1]
    return re.sub(r"_Opt$", "", name, flags=re.IGNORECASE)


# ----------------------------------------------------------------------------------------------
# Which runs and suites to evaluate
# ----------------------------------------------------------------------------------------------
def eval_specs(car: Any, stages: Sequence[str], P: Dict[str, Any], cfg: Any) -> List[Any]:
    specs: Dict[str, Any] = {}
    for st in stages:
        if st == "source_eval":
            got = [car.base_spec("source_only", s, 0, 0, cfg, eval_suites=(s,)) for s in ("source", "A", "L", "C")]
        else:
            got = car.plan(st, P, cfg)
        for sp in got:
            rid = sp.run_id()
            if rid in specs:
                merged = tuple(dict.fromkeys(tuple(specs[rid].eval_suites) + tuple(sp.eval_suites)))
                specs[rid] = dataclasses.replace(specs[rid], eval_suites=merged)
            else:
                specs[rid] = sp
    return list(specs.values())


def suite_towns(car: Any, suite: Any) -> List[str]:
    return list(dict.fromkeys(tb(car, r.town) for r in suite.eval_routes))


def town_file(run_dir: str, suite: str, town: str) -> str:
    return os.path.join(run_dir, f"eval_{suite}__{town}.json")


# ----------------------------------------------------------------------------------------------
# Evaluation of one town
# ----------------------------------------------------------------------------------------------
def remap(res: Dict[str, Any], idx: Sequence[int]) -> Dict[str, Any]:
    """Route indices of a town subset -> indices of the full suite."""
    for r in res["routes"]:
        r["route"] = int(idx[int(r["route"])])
    res["steps"]["route"] = [int(idx[int(x)]) for x in res["steps"]["route"]]
    for row in res["ident"]:
        row["route"] = int(idx[int(row["route"])])
    return res


def load_agent(car: Any, spec: Any, C: Any, run_cfg: Any):
    if spec.method == "expert":
        return None, True
    if spec.method == "source_only":
        return car.make_run_agent(spec, C.art, run_cfg, C.device), True
    run_dir = C.run_dir(spec)
    if not os.path.exists(os.path.join(run_dir, "ADAPT_DONE")):
        return None, False
    # run_adaptation only loads the finished checkpoint when ADAPT_DONE exists (no training here).
    return car.run_adaptation(spec, C.art, C.suites, C.mgr, C.args, run_cfg, C.device, run_dir), True


def freeze(agent: Any) -> None:
    if agent is None:
        return
    for name in ("actor", "critics", "target_critics", "causal_head", "src_actor", "W_S"):
        mod = getattr(agent, name, None)
        if mod is not None and hasattr(mod, "parameters"):
            mod.eval()
            for prm in mod.parameters():
                prm.requires_grad_(False)


def evaluate_town(car: Any, C: Any, P: Dict[str, Any], stages: Sequence[str], town: str) -> None:
    import copy as _copy
    cfg = C.cfg
    town = tb(car, town)
    specs = eval_specs(car, stages, P, cfg)
    skipped: List[str] = []
    for spec in specs:
        run_dir = C.run_dir(spec)
        todo = []
        for es in spec.eval_suites:
            suite = C.suites[es]
            idx = [i for i, r in enumerate(suite.eval_routes) if tb(car, r.town) == town]
            if not idx:
                continue
            if os.path.exists(os.path.join(run_dir, f"eval_{es}.json")) or os.path.exists(town_file(run_dir, es, town)):
                continue
            todo.append((es, suite, idx))
        if not todo:
            continue
        run_cfg = _copy.copy(cfg)
        run_cfg.knn_k = int(spec.k)
        agent, ok = load_agent(car, spec, C, run_cfg)
        if not ok:
            skipped.append(spec.run_id())
            continue
        freeze(agent)
        for es, suite, idx in todo:
            proxy = C.proxy_W() if (es == "C" and spec.is_cast()) else None
            if es == "C" and spec.is_cast() and proxy is None:
                # Fig. 3's embedding gap needs the proxy run; evaluating now would store an empty gap column.
                skipped.append(f"{spec.run_id()} (suite C waits for the proxy run)")
                continue
            print(f"[EVAL] {spec.run_id()} suite={es} town={town} routes={len(idx)}")
            sub = dataclasses.replace(suite, eval_routes=[suite.eval_routes[i] for i in idx])
            partial = os.path.join(run_dir, f"eval_{es}__{town}.partial.json")
            res = car.evaluate_suite(agent, C.ectx, sub, C.mgr, run_cfg, scripted=es in ("barrel", "queue"),
                                     proxy_W=proxy, nulls=C.art.nulls if (agent is not None and spec.is_cast()) else None,
                                     partial_path=partial)
            res = remap(res, idx)
            out = town_file(run_dir, es, town)
            with open(out + ".tmp", "w") as f:
                json.dump(res, f)
            os.replace(out + ".tmp", out)
            if os.path.exists(partial):
                os.remove(partial)
    if skipped:
        print(f"[EVAL] {len(skipped)} run/suite evaluations are not ready yet and were skipped, e.g. {skipped[0]}")
    write_town_csv(car, C, specs, town)


def write_town_csv(car: Any, C: Any, specs: Sequence[Any], town: str) -> None:
    rows = []
    for spec in specs:
        run_dir = os.path.join(C.out, "runs", spec.run_id())
        for es in spec.eval_suites:
            path = town_file(run_dir, es, town)
            if not os.path.exists(path):
                continue
            for r in json.load(open(path))["routes"]:
                rows.append([spec.run_id(), es, town, r["route"], r["conditions"].split("|")[1], r["DS"], r["RC"], r["IS"],
                             r["success"], r["collisions"], r["red_lights"], r["km"], r["min_ttc"], r["cte"]])
    if rows:
        os.makedirs(os.path.join(C.out, "results"), exist_ok=True)
        car.write_csv(os.path.join(C.out, "results", f"eval_{town}.csv"),
                      ["run", "suite", "town", "route", "weather", "DS", "RC", "IS", "success", "collisions",
                       "red_lights", "km", "min_ttc_s", "cte_m"], rows)


# ----------------------------------------------------------------------------------------------
# Frozen pilot: the SAME checkpoint on 20 new routes in EACH town
# ----------------------------------------------------------------------------------------------
RUNTIME_CONFIG_FIELDS = ("ego_half_length", "ego_half_width")


def stable_config(cfg: Any) -> Dict[str, Any]:
    """Config for result metadata without fields car.py overwrites at run time: the ego size is set from the
    spawned vehicle, so a policy evaluated after another one in the same process recorded the measured size
    and its results could not be resumed in a process that skipped the first policy."""
    return {k: v for k, v in dataclasses.asdict(cfg).items() if k not in RUNTIME_CONFIG_FIELDS}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pilot_setup(car: Any, cargs: argparse.Namespace):
    # Do not call setup_run: it reconstructs paper suites and can overwrite
    # results/suites_used.csv. This mode does not need source replay or banks.
    P = dict(car.PROFILES[cargs.profile])
    for key in ("budget", "source_steps", "eval_routes"):
        if int(getattr(cargs, key, 0)) > 0:
            P[key] = int(getattr(cargs, key))
    cfg = car.make_cfg(cargs, P)
    # 'policy' preserves TRAINING observation normalization (notably NPC density).
    # Learning is disabled by the evaluation code, not by changing these features.
    cfg.mode_name = "policy"
    cfg.use_training_curriculum = False
    # The safety shield stays off unless --use-safety-shield is passed; shielded results go to <tag>_shield
    # and every activation is counted (safety_interventions). Earlier versions forced it off here, so the
    # flag was silently ignored.
    cfg.use_safety_shield = bool(getattr(cargs, "use_safety_shield", False))
    cfg.red_light_assist = False
    if cfg.no_rendering_mode and cfg.use_cameras:
        raise ValueError("RGB/RGB-D evaluation needs rendering; remove --no-rendering")
    device = car.torch.device("cpu" if cargs.cpu or not car.torch.cuda.is_available() else "cuda")
    return P, cfg, device


def pilot_spec(car: Any, own: argparse.Namespace, cfg: Any, P: Dict[str, Any]):
    return car.base_spec("cast", own.train_suite, int(cfg.seed), int(P["budget"]), cfg)


def pilot_root(car: Any, own: argparse.Namespace, cfg: Any, P: Dict[str, Any]) -> Path:
    tag = own.eval_tag
    if bool(getattr(cfg, "eval_light_override", False)):
        tag = f"{tag}_light-{cfg.light_source}"   # diagnostic runs never mix with paper results
    if bool(getattr(cfg, "use_safety_shield", False)):
        tag = f"{tag}_shield"                      # shield-assisted results are reported separately
    return (Path(cfg.out_dir) / "evaluation" / tag / pilot_spec(car, own, cfg, P).run_id()
            / f"{own.weather_set}_N{own.episodes}_R{own.route_seed}")


def checkpoint_path(car: Any, own: argparse.Namespace, cfg: Any, P: Dict[str, Any], policy: str) -> Path:
    if policy == "source":
        folder = Path(cfg.out_dir) / "source"
        marker, filename = "DONE", "source_agent.pt"
    else:
        folder = Path(cfg.out_dir) / "runs" / pilot_spec(car, own, cfg, P).run_id()
        marker, filename = "ADAPT_DONE", "agent.pt"
    if not (folder / marker).is_file() or not (folder / filename).is_file():
        raise FileNotFoundError(f"Finished {policy} checkpoint missing: {folder / filename} (and {marker}). "
                                "Use the training --out-dir, --budget, --seed and --train-suite.")
    return folder / filename


def check_pilot_artifacts(car: Any, own: argparse.Namespace, cfg: Any, P: Dict[str, Any]) -> None:
    """Report missing files before any CARLA startup or map mutation."""
    for policy in own.policies.split(","):
        checkpoint_path(car, own, cfg, P, policy)
    if cfg.perception_mode in ("camera", "rgbd"):
        name = "segnet.pt" if cfg.perception_mode == "rgbd" else "detector.pt"
        path = Path(cfg.out_dir) / "perception" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Frozen perception checkpoint missing or empty: {path}. "
                                    "Use the original training directory and its trained detector.")


def load_frozen_policy(car: Any, own: argparse.Namespace, cfg: Any, P: Dict[str, Any], device: Any,
                       policy: str, path: Path):
    run_cfg = copy.copy(cfg)
    car.match_token_encoder(run_cfg, str(path))
    sp = (pilot_spec(car, own, run_cfg, P) if policy == "adapted" else
          car.base_spec("source_only", "source", int(cfg.seed), 0, run_cfg))
    agent = car.CaSTAgent(run_cfg, device, sp)
    if hasattr(car, "TRAINING_REVISION"):
        agent.load(str(path), strict_actor=True, evaluation_only=True)
        print(f"[EVAL] checkpoint training protocol={agent.loaded_training_revision}; "
              f"metric={car.EVALUATION_METRIC_REVISION}")
    else:
        agent.load(str(path), strict_actor=True)
        print("[EVAL WARNING] legacy car.py: crossing-counter correction is unavailable")
    freeze(agent)
    # No make_run_agent / SourceArtifacts.head: those may fit a missing causal
    # head during loading. Driving with the saved actor needs no coverage banks.
    return agent


def ordered_towns(car: Any, own: argparse.Namespace, cargs: argparse.Namespace) -> List[str]:
    if own.town != "all":
        return [own.town]
    requested = (own.towns.split(",") if own.towns else
                 cargs.source_towns.split(",") + cargs.layout_towns.split(","))
    result, seen = ["Town10HD"], {"town10hd"}
    for town in requested:
        town = town.strip()
        key = tb(car, town).casefold()
        if town and key not in seen:
            seen.add(key)
            result.append(town)
    return result


def installed_maps(car: Any, cargs: argparse.Namespace) -> List[str]:
    client = car.carla.Client(cargs.host, int(cargs.port))
    client.set_timeout(15.0)
    # Preserve exact names (_Opt included); available_towns() strips that suffix.
    return [str(p) for p in client.get_available_maps()
            if re.fullmatch(r"Town\d+(?:HD)?(?:_Opt)?", str(p).split("/")[-1], re.IGNORECASE)]


def resolve_eval_towns(car: Any, requested: Sequence[str], available: Sequence[str],
                       strict: bool = True) -> List[str]:
    result = []
    for town in requested:
        matches = [p for p in available if tb(car, p).casefold() == tb(car, town).casefold()]
        if not matches:
            if strict or tb(car, town).casefold() == "town10hd":
                raise ValueError(f"Evaluation town {town} is not installed. Available: " + ", ".join(available))
            print(f"[EVAL] skip uninstalled town {town}")
            continue
        # Prefer the optimized map where both variants exist. Always use a name
        # actually returned by CARLA instead of guessing that _Opt is installed.
        pick = sorted(matches, key=lambda x: (not x.lower().endswith("_opt"), x))[0]
        result.append(pick.split("/")[-1])
    return result


def ensure_eval_town(car: Any, cargs: argparse.Namespace, requested: str) -> str:
    """Return the actual loaded map name, changing worlds only when authorized."""
    current = str(car.current_server_town(cargs.host, int(cargs.port)))
    print(f"[MAP] endpoint={cargs.host}:{cargs.port} current={current} requested={requested}", flush=True)
    if tb(car, current).casefold() == tb(car, requested).casefold():
        return current.replace("\\", "/").rstrip("/").split("/")[-1]
    if not bool(getattr(cargs, "allow_map_load", False)):
        raise ValueError(f"CARLA at {cargs.host}:{cargs.port} is in {current}, requested {requested}. "
                         "Pass --allow-map-load to load it through the API, or --carla-cmd to restart in that map.")
    target = resolve_eval_towns(car, [requested], installed_maps(car, cargs), strict=True)[0]
    loader = getattr(car, "load_town_in_child", None)
    if loader is None:
        raise ImportError("This evaluator needs the updated car.py with load_town_in_child for --allow-map-load.")
    print(f"[MAP] loading {target} through the CARLA API in a separate process; this can take several minutes.",
          flush=True)
    if not loader(cargs, target):
        raise RuntimeError(f"Could not load {target} at {cargs.host}:{cargs.port}. "
                           "Saved evaluation episodes are retained; restart CARLA and rerun.")
    current = str(car.current_server_town(cargs.host, int(cargs.port)))
    print(f"[MAP] after load: current={current} requested={target}", flush=True)
    if tb(car, current).casefold() != tb(car, target).casefold():
        raise RuntimeError(f"Map load did not select {target}: CARLA reports {current}. Evaluation stopped.")
    return current.replace("\\", "/").rstrip("/").split("/")[-1]


def pilot_step_limit(cfg: Any) -> int:
    """Outer guard only. The environment still grants credits and decides termination."""
    base = int(cfg.max_episode_steps)
    if base <= 0:
        raise ValueError("max_episode_steps must be positive")
    return (base + max(0, int(getattr(cfg, "blocked_timeout_extension_steps", 0)))
            + max(0, int(getattr(cfg, "near_goal_timeout_extension_steps", 0))))


def pilot_routes(car: Any, own: argparse.Namespace, cfg: Any, town: str) -> List[Any]:
    weathers = (list(car.ADVERSE_WEATHERS) if own.weather_set == "adverse" else
                list(car.SOURCE_WEATHERS) if own.weather_set == "source" else
                list(car.SOURCE_WEATHERS) + list(car.ADVERSE_WEATHERS))
    # Stable across processes, town order, selected policies and restarts.
    offset = zlib.crc32(tb(car, town).lower().encode("utf-8"))
    seed0 = (int(own.route_seed) + offset) % (2**31 - 100001)
    rng = random.Random(seed0)
    weather_order = [weathers[i % len(weathers)] for i in range(own.episodes)]
    rng.shuffle(weather_order)
    source_towns = {tb(car, t).casefold() for t in getattr(own, "source_towns", "").split(",")}
    return [car.Conditions(town, weather, rng.randint(cfg.train_npc_min, cfg.train_npc_max),
                           rng.randint(cfg.train_walker_min, cfg.train_walker_max),
                           in_source_support=(tb(car, town).casefold() in source_towns
                                              and weather in car.SOURCE_WEATHERS),
                           route_seed=seed0 + i)
            for i, weather in enumerate(weather_order)]


class EvaluationSimulatorError(RuntimeError):
    """A simulator attempt failed; it must not consume an evaluation episode."""


def drive_pilot_episode(car: Any, mgr: Any, agent: Any, cfg: Any, cond: Any, index: int) -> Dict[str, Any]:
    seed = int(cond.route_seed)
    car.set_global_seed(seed)
    try:
        env = mgr.get(cond.town)
        env.set_detector(mgr.detector, mgr.device)
        env.set_conditions(cond)
        env.tm.set_random_device_seed(seed)
        env.world.set_pedestrians_seed(seed)
        # Bypass EnvManager.reset's seed+7919 retry: the same route must be retried.
        obs, info = env.reset(seed=seed)
    except car.TownSwitch:
        raise
    except Exception as exc:
        raise EvaluationSimulatorError(f"reset: {exc}") from exc
    xy = [[round(float(x), 3), round(float(y), 3)] for x, y in env.route_xy]
    route_hash = hashlib.sha256(json.dumps(xy).encode()).hexdigest()
    total_reward, cte_sum, min_ttc, dropped, steps = 0.0, 0.0, 999.0, 0, 0
    trace = deque(maxlen=max(1, int(round(3.0 / max(float(getattr(cfg, "dt", 0.05)), 1e-6)))))
    # Light diagnostics (observation only): what colour the policy saw against the true state, and what it
    # saw while standing still at a green light or on a free road (the "stuck"/"blocked" failures).
    light_seen: Dict[str, Dict[str, int]] = {}
    still_saw: Dict[str, int] = {}
    red_saw: List[Dict[str, Any]] = []
    n_red = 0
    term = trunc = False
    with car.torch.inference_mode():
        for _ in range(pilot_step_limit(cfg)):
            action = agent.act(obs, deterministic=True)
            if not np.isfinite(action).all():
                raise ValueError("Frozen policy produced a non-finite action; evaluation stopped")
            # Observation-only diagnostic: never call a teacher or modify policy control.
            perceived_lead = None
            saw_light, saw_light_d = (car.perceived_light(obs, cfg) if hasattr(car, "perceived_light")
                                      else ("n/a", -1.0))
            if hasattr(car, "lead_token"):
                d, closing, kind, slot, variance = car.lead_token(obs, cfg)
                if np.isfinite(d):
                    perceived_lead = {"distance_m": float(d), "closing_ms": car.safe_float(closing),
                                      "kind": str(kind), "variance": car.safe_float(variance)}
            try:
                obs, reward, term, trunc, info = env.step(action)
            except Exception as exc:
                raise EvaluationSimulatorError(f"step: {exc}") from exc
            steps += 1
            total_reward += car.safe_float(reward)
            cte_sum += abs(car.safe_float(info.get("dL", 0.0)))
            min_ttc = min(min_ttc, car.safe_float(info.get("time_to_conflict", 999.0), 999.0))
            dropped += int(bool(info.get("frame_dropped", False)))
            true_light = str(info.get("tl_state", "None"))
            row_ = light_seen.setdefault(true_light, {})
            row_[saw_light] = row_.get(saw_light, 0) + 1
            lead_close = perceived_lead is not None and perceived_lead["distance_m"] < 10.0
            if (car.safe_float(info.get("speed_ms", 0.0)) < 0.3 and true_light not in ("Red", "Yellow")
                    and not lead_close):
                still_saw[saw_light] = still_saw.get(saw_light, 0) + 1
            reds = sum(k == "red_light" for k, _ in env.infractions)
            if reds > n_red:
                red_saw.append({"step": steps, "saw": saw_light, "saw_distance_m": round(saw_light_d, 1),
                                "speed_ms": round(car.safe_float(info.get("speed_ms", 0.0)), 2)})
                n_red = reds
            trace.append({"step": steps, "speed_ms": car.safe_float(info.get("speed_ms", 0.0)),
                          "command": [float(x) for x in action],
                          "applied": [float(x) for x in info.get("applied_action", action)],
                          "perceived_lead": perceived_lead,
                          "perceived_light": saw_light, "perceived_light_m": round(saw_light_d, 1),
                          "traffic_light": str(info.get("tl_state", "None")),
                          "stop_line_distance_m": car.safe_float(info.get("tl_dist", -1.0)),
                          "route_completion_pct": car.safe_float(info.get("route_completion_pct", 0.0)),
                          "cte_m": car.safe_float(info.get("dL", 0.0))})
            if term or trunc:
                break
    if not (term or trunc):
        info = dict(info, term_reason="eval_step_limit", success=False)
        trunc = True
    infr = list(env.infractions)
    rc = car.safe_float(info.get("route_completion_pct", 0.0))
    is_score = float(np.prod([car.IS_PENALTY.get(k, 1.0) for k, _ in infr])) if infr else 1.0
    reason = str(info.get("term_reason", "terminated" if term else "timeout"))
    details = ""
    if reason in ("stuck", "blocked") and hasattr(env, "describe_wait"):
        details = str(env.describe_wait())
    success = int(bool(info.get("success", False)))
    interventions = int(getattr(env, "safety_interventions", 0))
    jam_removed = int(info.get("jam_removed", 0))
    penalty_free = int(bool(success) and not infr)
    return {"episode": index + 1, "route_seed": seed, "route_hash": route_hash,
            "conditions": cond.key(), "weather": cond.weather, "npc": cond.npc, "walkers": cond.walkers,
            "route_length_m": float(env.route_total_len_m), "RC": rc, "IS": is_score, "DS": rc * is_score,
            "success": success, "end_reason": reason, "details": details,
            "penalty_free_success": penalty_free,
            "unassisted_success": int(bool(success) and jam_removed == 0 and interventions == 0),
            "clean_unassisted_success": int(bool(penalty_free) and jam_removed == 0 and interventions == 0),
            "collisions": sum(k.startswith("collision") for k, _ in infr),
            "red_lights": sum(k == "red_light" for k, _ in infr),
            "km": car.safe_float(info.get("distance_driven_m", 0.0)) / 1000.0,
            "min_ttc_s": min_ttc, "cte_m": cte_sum / max(steps, 1), "steps": steps,
            "env_return_proxy": total_reward, "terminated": bool(term), "truncated": bool(trunc),
            "frame_drops": dropped, "jam_removed": jam_removed,
            "safety_interventions": interventions,
            "collision_kind": str(info.get("collision_kind", "")),
            "collision_where": str(info.get("collision_where", "")),
            "infraction_events": [{"kind": str(k), "distance_m": float(d)} for k, d in infr],
            "red_light_events": list(getattr(env, "red_light_events", [])),
            "light_seen_vs_true": light_seen, "still_on_free_road_saw": still_saw, "red_violation_saw": red_saw,
            "final_trace": list(trace)}


def light_summary(row: Dict[str, Any]) -> str:
    """One line: colour seen against the true light, steps standing still on a free road by what the policy
    saw, and what it saw when it crossed on red."""
    def share(d: Dict[str, int]) -> str:
        n = max(sum(d.values()), 1)
        return "/".join(f"{k}:{100 * v / n:.0f}%" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))
    seen = row.get("light_seen_vs_true", {})
    parts = [f"{t}->[{share(d)}]" for t, d in seen.items() if t in ("Red", "Yellow", "Green")]
    still = row.get("still_on_free_road_saw", {})
    dt = 0.05
    still_s = f"still_free={sum(still.values()) * dt:.0f}s[{share(still)}]" if still else "still_free=0s"
    reds = ",".join(f"{r['saw']}@{r['saw_distance_m']:.0f}m" for r in row.get("red_violation_saw", []))
    return "light: " + " ".join(parts) + f" | {still_s}" + (f" | ran_red_seeing={reds}" if reds else "")


def write_dict_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def summarize_pilot(root: Path) -> None:
    summaries = []
    for folder in sorted(root.glob("Town*"), key=lambda p: (p.name != "Town10HD", p.name)):
        episode_rows, by_policy = [], {}
        for path in sorted(folder.glob("*.json")):
            with path.open() as f:
                result = json.load(f)
            if "episodes" not in result:
                continue
            meta, rows = result["metadata"], result["episodes"]
            # Aggregation can read older result files. Derived fields describe
            # their recorded counters; they do not retroactively repair events.
            for row in rows:
                row.setdefault("penalty_free_success", int(bool(row["success"])
                    and row.get("collisions", 0) == 0 and row.get("red_lights", 0) == 0
                    and float(row.get("IS", 1.0)) >= 1.0 - 1e-12))
                row.setdefault("unassisted_success", int(bool(row["success"])
                    and row.get("jam_removed", 0) == 0 and row.get("safety_interventions", 0) == 0))
                row.setdefault("clean_unassisted_success", int(bool(row["penalty_free_success"])
                    and bool(row["unassisted_success"])))
            policy = meta["policy"]
            by_policy[policy] = rows
            episode_rows.extend(dict(policy=policy, **r) for r in rows)
            if not rows:
                continue
            n, km = len(rows), sum(r["km"] for r in rows)
            summaries.append({"town": folder.name, "policy": policy, "completed": n,
                              "metric_revision": meta.get("metric_revision", "legacy"),
                              "requested": meta["episodes_requested"], "complete": result["complete"],
                              "successes": sum(r["success"] for r in rows),
                              "success_rate_pct": 100.0 * sum(r["success"] for r in rows) / n,
                              "penalty_free_success_rate_pct": 100.0 * sum(r["penalty_free_success"] for r in rows) / n,
                              "unassisted_success_rate_pct": 100.0 * sum(r["unassisted_success"] for r in rows) / n,
                              "clean_unassisted_success_rate_pct": 100.0 * sum(r["clean_unassisted_success"] for r in rows) / n,
                              "mean_RC": sum(r["RC"] for r in rows) / n,
                              "mean_DS": sum(r["DS"] for r in rows) / n,
                              "collisions": sum(r["collisions"] for r in rows),
                              "collisions_per_km": sum(r["collisions"] for r in rows) / max(km, 1e-6),
                              "red_lights": sum(r["red_lights"] for r in rows),
                              "stuck": sum(r["end_reason"] in ("stuck", "blocked") for r in rows),
                              "off_route": sum(r["end_reason"] in ("off_route", "offroad") for r in rows),
                              "jam_removed": sum(r["jam_removed"] for r in rows),
                              "safety_interventions": sum(r["safety_interventions"] for r in rows)})
        write_dict_csv(folder / "episodes.csv", episode_rows)
        if "source" in by_policy and "adapted" in by_policy:
            source = {r["route_seed"]: r for r in by_policy["source"]}
            mismatched = [r["episode"] for r in by_policy["adapted"] if r["route_seed"] in source
                          and r["route_hash"] != source[r["route_seed"]]["route_hash"]]
            if mismatched:
                print(f"[EVAL WARNING] {folder.name}: actual routes differ between policies at episodes "
                      f"{mismatched}; these results are NOT a paired route comparison.")
    write_dict_csv(root / "summary.csv", summaries)
    if summaries:
        print(f"[EVAL] summary: {root / 'summary.csv'}")


def evaluate_pilot_town(car: Any, own: argparse.Namespace, cargs: argparse.Namespace,
                        P: Dict[str, Any], cfg: Any, device: Any) -> int:
    check_pilot_artifacts(car, own, cfg, P)
    own.town = ensure_eval_town(car, cargs, own.town)
    town = tb(car, own.town)
    own.source_towns = cargs.source_towns
    routes = pilot_routes(car, own, cfg, own.town)
    root = pilot_root(car, own, cfg, P)
    mgr = car.EnvManager(cargs, cfg)
    mgr.device = device
    try:
        detector_sha = None
        if cfg.perception_mode in ("camera", "rgbd"):
            mgr.detector = car.load_detector(cfg.out_dir, cfg, device)
            if mgr.detector is None:
                raise FileNotFoundError("Frozen perception checkpoint missing in --out-dir")
            name = "segnet.pt" if cfg.perception_mode == "rgbd" else "detector.pt"
            detector_sha = digest_file(Path(cfg.out_dir) / "perception" / name)
        for policy in own.policies.split(","):
            path = checkpoint_path(car, own, cfg, P, policy)
            # Canonical JSON handles tuple fields consistently after a restart.
            meta = json.loads(json.dumps({"schema": 3, "policy": policy, "episodes_requested": own.episodes,
                    "metric_revision": getattr(car, "EVALUATION_METRIC_REVISION", "legacy"),
                    "clean_success_definition": "goal, no tracked infractions, no watchdog removals or safety interventions",
                    "checkpoint": str(path.resolve()), "checkpoint_sha256": digest_file(path),
                    "car_sha256": digest_file(Path(car.__file__)), "evaluator_sha256": digest_file(Path(__file__)),
                    "detector_sha256": detector_sha, "config": stable_config(cfg),
                    "safety_shield": bool(getattr(cfg, "use_safety_shield", False)),
                    "routes": [dataclasses.asdict(c) for c in routes], "deterministic": True,
                    "timeout_policy": "environment with traffic/near-goal extensions",
                    "outer_step_limit": pilot_step_limit(cfg),
                    "traffic_watchdog": "enabled; removals reported", "teacher_control": False}))
            result_path = root / town / f"{policy}.json"
            result = {"metadata": meta, "episodes": [], "complete": False}
            if result_path.exists():
                with result_path.open() as f:
                    result = json.load(f)
                if result.get("metadata") != meta:
                    raise ValueError(f"Settings/checkpoint changed for {result_path}. Use a new --eval-tag.")
                rows = result["episodes"]
                if len(rows) > own.episodes or any(r["episode"] != i + 1 or r["route_seed"] != routes[i].route_seed
                                                   for i, r in enumerate(rows)):
                    raise ValueError(f"Invalid episode checkpoint: {result_path}")
            done = len(result["episodes"])
            print(f"[EVAL] {town} {policy}: {done}/{own.episodes} saved | checkpoint={path}")
            if done == own.episodes:
                continue
            agent = load_frozen_policy(car, own, cfg, P, device, policy, path)
            # Start a new environment instance and reseed every route for each policy.
            mgr.close()
            for index in range(done, own.episodes):
                for attempt in range(3):
                    try:
                        row = drive_pilot_episode(car, mgr, agent, cfg, routes[index], index)
                        break
                    except EvaluationSimulatorError as exc:
                        down = mgr.env is not None and car.server_unresponsive(mgr.env, exc.__cause__ or exc)
                        mgr.close()
                        print(f"[EVAL] episode {index + 1} simulator failure ({attempt + 1}/3): {exc}")
                        if down or attempt == 2:
                            raise car.SimulatorDown(str(exc)) from exc
                result["episodes"].append(row)
                result["complete"] = len(result["episodes"]) == own.episodes
                atomic_json(result_path, result)
                summarize_pilot(root)
                print(f"[EVAL {town} {policy}] ep={index + 1}/{own.episodes} weather={row['weather']} "
                      f"RC={row['RC']:.1f}% DS={row['DS']:.1f} success={row['success']} "
                      f"coll={row['collisions']} red={row['red_lights']} end={row['end_reason']} "
                      f"unjammed={row['jam_removed']} clean={row['clean_unassisted_success']}")
                if row['collision_where']:
                    print(f"    collision: {row['collision_kind']} {row['collision_where']}")
                if row['details']:
                    print(f"    wait: {row['details']}")
                if not row['success'] or row['red_lights']:
                    print("    " + light_summary(row))
            del agent
        summarize_pilot(root)
    except car.SimulatorDown as exc:
        print(f"[EVAL] simulator unavailable: {exc}. Saved episodes are kept; rerun this command after restart.")
        return car.SIM_DOWN_EXIT
    finally:
        mgr.close()
    return 0


def pilot_worker_command(own: argparse.Namespace, rest: Sequence[str], town: str) -> List[str]:
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--eval-worker", "--eval-stages", "pilot",
            "--car-module", own.car_module, "--town", town, "--episodes", str(own.episodes),
            "--policies", own.policies, "--train-suite", own.train_suite, "--weather-set", own.weather_set,
            "--route-seed", str(own.route_seed), "--eval-tag", own.eval_tag] + list(rest)


def pilot_main(car: Any, own: argparse.Namespace, rest: List[str], cargs: argparse.Namespace) -> int:
    P, cfg, device = pilot_setup(car, cargs)
    root = pilot_root(car, own, cfg, P)
    if own.aggregate:
        summarize_pilot(root)
        return 0
    if int(cargs.eval_routes) > 0:
        raise ValueError("Pilot mode counts episodes PER TOWN. Use --episodes 20 instead of --eval-routes.")
    check_pilot_artifacts(car, own, cfg, P)
    if int(cargs.max_restarts) < 0 or int(getattr(cargs, "max_startup_crashes", 2)) < 1:
        raise ValueError("--max-restarts must be nonnegative and --max-startup-crashes positive")
    if own.eval_worker:
        return evaluate_pilot_town(car, own, cargs, P, cfg, device)
    if own.town == "all" and not cargs.carla_cmd:
        raise ValueError("--town all requires --carla-cmd. With a manually started server, run --town Town10HD, "
                         "then --town Town01, --town Town02, etc., after starting each map.")
    requested = ordered_towns(car, own, cargs)
    try:
        available = installed_maps(car, cargs)
    except RuntimeError:
        if not cargs.carla_cmd:
            raise
        # Bootstrap only when no server is reachable; subsequent names come from
        # CARLA's map inventory. The source default is the installed Town10HD_Opt.
        first = cargs.source_towns.split(",")[0] if tb(car, requested[0]) == "Town10HD" else requested[0]
        car.restart_carla_server(cargs, first)
        available = installed_maps(car, cargs)
    towns = resolve_eval_towns(car, requested, available, strict=bool(own.towns) or own.town != "all")
    print(f"[EVAL] sequence: {' -> '.join(tb(car, t) for t in towns)} | {own.episodes} episodes per town/policy "
          f"| policies={own.policies} | weather={own.weather_set}")
    print(f"[EVAL] route target={cfg.route_target_length_m:g} m | base limit={cfg.max_episode_steps} steps "
          f"| outer guard={pilot_step_limit(cfg)} steps; environment controls timeout extensions")
    if not cargs.carla_cmd:
        own.town = towns[0]
        return evaluate_pilot_town(car, own, cargs, P, cfg, device)
    startup_path = Path(cfg.out_dir) / "runtime_startup.json"
    for town in towns:
        car.restart_carla_server(cargs, town)
        restarts, startup_crashes = 0, 0
        while True:
            # Only this worker's marker may classify a native startup crash.
            try:
                startup_path.unlink()
            except FileNotFoundError:
                pass
            try:
                rc = subprocess.call(pilot_worker_command(own, rest, town))
            except KeyboardInterrupt:
                print("[EVAL] interrupted; saved episodes are retained.")
                return 130
            if rc == 0:
                break
            if rc in (-2, -15, 130, 143):
                return 130
            # Configuration/checkpoint/programming failures are not restartable.
            if rc > 0 and rc != car.SIM_DOWN_EXIT:
                print(f"[EVAL] worker exited {rc}; fix the reported error before retrying.")
                return rc
            try:
                startup = json.loads(startup_path.read_text())
            except (OSError, ValueError):
                startup = {}
            if rc < 0 and startup.get("phase") == "traffic_manager_start":
                startup_crashes += 1
                if startup_crashes >= int(getattr(cargs, "max_startup_crashes", 2)):
                    print(f"[EVAL] repeated native Traffic Manager startup failures in {town}; stopped. "
                          f"Inspect {startup_path} and logs/carla_server.log. Saved episodes are retained.")
                    return 1
            else:
                startup_crashes = 0
            restarts += 1
            if restarts > int(cargs.max_restarts):
                print("[EVAL] restart limit reached; finished episodes are saved.")
                return 1
            print(f"[EVAL] {town} worker exited {rc}; restart {restarts}/{cargs.max_restarts}")
            car.restart_carla_server(cargs, town)
    summarize_pilot(root)
    return 0


# ----------------------------------------------------------------------------------------------
# Merge per-town results and write the paper tables
# ----------------------------------------------------------------------------------------------
def merge_results(car: Any, C: Any, P: Dict[str, Any], stages: Sequence[str]) -> Tuple[int, int]:
    done = missing = 0
    for spec in eval_specs(car, stages, P, C.cfg):
        run_dir = os.path.join(C.out, "runs", spec.run_id())
        for es in spec.eval_suites:
            final = os.path.join(run_dir, f"eval_{es}.json")
            if os.path.exists(final):
                done += 1
                continue
            parts = [town_file(run_dir, es, t) for t in suite_towns(car, C.suites[es])]
            if not all(os.path.exists(p) for p in parts):
                missing += 1
                continue
            loaded = [json.load(open(p)) for p in parts]
            merged: Dict[str, Any] = {"routes": sorted([r for d in loaded for r in d["routes"]], key=lambda r: r["route"]),
                                      "steps": {k: [x for d in loaded for x in d["steps"][k]] for k in loaded[0]["steps"]}}
            for key in ("ident", "acc_pairs", "gap", "share_uncovered", "reliance_distractor", "head_ms_list"):
                merged[key] = [x for d in loaded for x in d.get(key, [])]
            merged["head_ms"] = float(np.mean(merged["head_ms_list"])) if merged["head_ms_list"] else float("nan")
            with open(final + ".tmp", "w") as f:
                json.dump(merged, f)
            os.replace(final + ".tmp", final)
            done += 1
    return done, missing


def used_towns(cargs: argparse.Namespace) -> List[str]:
    """Towns the training run actually built its suites from (results/suites_used.csv, written by car.py).
    Falling back to the requested list would re-index the routes if a town was missing on the server."""
    path = os.path.join(cargs.out_dir, "results", "suites_used.csv")
    if os.path.exists(path):
        with open(path) as f:
            got = [r["town"] for r in csv.DictReader(f) if r.get("status") == "used"]
        if got:
            return list(dict.fromkeys(got))
    return [t for t in cargs.source_towns.split(",") + cargs.layout_towns.split(",") if t]


def aggregate_main(car: Any, cargs: argparse.Namespace, stages: Sequence[str]) -> None:
    towns = used_towns(cargs)
    P, cfg, device, out, suites, mgr, C = car.setup_run(cargs, avail=towns)
    done, missing = merge_results(car, C, P, stages)
    print(f"[AGGREGATE] {done} run/suite results complete, {missing} still need towns")
    car.write_source_eval(C, P)
    if os.path.exists(os.path.join(out, "source", "DONE")):
        car.aggregate(C, P)


# ----------------------------------------------------------------------------------------------
# One town in this process
# ----------------------------------------------------------------------------------------------
def worker_main(car: Any, own: argparse.Namespace, cargs: argparse.Namespace, stages: Sequence[str]) -> int:
    P, cfg, device, out, suites, mgr, C = car.setup_run(cargs)
    current = tb(car, car.current_server_town(cargs.host, cargs.port))
    town = tb(car, own.town)
    if current != town:
        print(f"[EVAL] CARLA is running {current}, not {town}. Start CARLA in {town}, for example\n"
              f"    ./CarlaUE4.sh {car.map_path(own.town)} -quality-level=Low -nosound -carla-rpc-port={cargs.port}\n"
              f"  or pass --carla-cmd so car_eval.py starts it.")
        return 3
    if cfg.perception_mode in ("camera", "rgbd"):
        mgr.detector = car.load_detector(out, cfg, device)
        if mgr.detector is None:
            raise SystemExit("the frozen detector is missing: run car.py first")
    if not os.path.exists(os.path.join(out, "source", "DONE")):
        raise SystemExit("the source stage is not finished: run car.py first")
    print(f"[EVAL] evaluation only | town={town} | stages={list(stages)} | device={device} | deterministic, no gradients")
    try:
        evaluate_town(car, C, P, stages, town)
        done, missing = merge_results(car, C, P, stages)
        print(f"[EVAL] {town} finished; {done} run/suite results complete, {missing} still need other towns")
    except car.SimulatorDown as e:
        if mgr.env is not None:
            mgr.env._server_down = True
        print(f"[EVAL] CARLA stopped responding ({e}). Restart CARLA in {town} and run the same command again; "
              f"finished routes are kept.")
        return car.SIM_DOWN_EXIT
    finally:
        mgr.close()
    return 0


def needed_towns(car: Any, cargs: argparse.Namespace, stages: Sequence[str]) -> List[str]:
    towns = used_towns(cargs)
    P, cfg, device, out, suites, mgr, C = car.setup_run(cargs, avail=towns)
    names: Dict[str, str] = {}
    for spec in eval_specs(car, stages, P, cfg):
        for es in spec.eval_suites:
            for r in suites[es].eval_routes:
                names.setdefault(tb(car, r.town), r.town)
    return list(names.values())


def supervisor_main(car: Any, own: argparse.Namespace, rest: List[str], cargs: argparse.Namespace,
                    stages: Sequence[str]) -> int:
    """With --carla-cmd: a fresh CARLA server per town, the town evaluated in a child process, and the
    server restarted in the same town after a crash (the child resumes from its partial files)."""
    towns = needed_towns(car, cargs, stages) if own.town == "all" else [own.town]
    print(f"[EVAL] towns: {', '.join(towns)}")
    for town in towns:
        restarts = 0
        car.restart_carla_server(cargs, town)
        while True:
            cmd = [sys.executable, os.path.abspath(__file__), "--town", town, "--eval-stages", own.eval_stages,
                   "--car-module", own.car_module, "--eval-worker"] + rest
            rc = subprocess.call(cmd)
            if rc == 0:
                break
            restarts += 1
            print(f"[EVAL] worker for {town} exited with code {rc}; restart {restarts}/{cargs.max_restarts}")
            if restarts >= int(cargs.max_restarts):
                raise SystemExit("too many restarts")
            time.sleep(5.0)
            car.restart_carla_server(cargs, town)
    aggregate_main(car, cargs, stages)
    return 0


def main() -> int:
    import faulthandler
    faulthandler.enable()
    own, rest = parse_args()
    car = load_car_module(own.car_module)
    cargs = car_args(car, rest)
    if own.eval_stages == "pilot":
        return pilot_main(car, own, rest, cargs)
    stages = car.EVAL_STAGES if own.eval_stages == "all" else tuple(s.strip() for s in own.eval_stages.split(","))
    bad = [s for s in stages if s not in car.EVAL_STAGES]
    if bad:
        raise SystemExit(f"unknown evaluation stage(s) {bad}")
    if own.aggregate:
        aggregate_main(car, cargs, stages)
        return 0
    if own.eval_worker or not cargs.carla_cmd:
        if own.town == "all":
            raise SystemExit("--town all needs --carla-cmd (each town runs in a fresh CARLA server); with CARLA "
                             "started by hand, evaluate one town at a time: --town Town10HD_Opt, --town Town01, ...")
        car.wait_for_carla_ready(cargs.host, cargs.port, timeout_s=120.0)
        return worker_main(car, own, cargs, stages)
    return supervisor_main(car, own, rest, cargs, stages)


if __name__ == "__main__":
    raise SystemExit(main())
