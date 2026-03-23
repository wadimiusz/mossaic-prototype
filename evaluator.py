"""Evaluator for MoSSAIC DistGraph evolution — multi-task transformers.

Scores candidates across multiple task families (modular addition, greater-than,
Dyck languages, DFA simulation). The same `build_dist` program must handle all of them.
"""

import glob
import json
import os
import random
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch

from distgraph import DistGraphError, execute as execute_distgraph

EVAL_TIMEOUT = 120
LOG_PROB_CLAMP_MIN = -1e6
LOG_PROB_CLAMP_MAX = 1e6
FLOP_PENALTY_NATS = 0.01
PERF_EVENT = "fp_flops_retired.all"
E_CORE = "16"
MODELS_PER_EVAL = 7  # sample across task families
MIN_ACCURACY = 0.95  # filter out models that didn't train well

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = {
    "modular_addition": os.path.join(_DIR, "modular_addition_models"),
    "greater_than": os.path.join(_DIR, "greater_than_models"),
    "dyck": os.path.join(_DIR, "dyck_models"),
    "dfa": os.path.join(_DIR, "dfa_models"),
    "induction": os.path.join(_DIR, "induction_models"),
}
CACHE_DIR = os.path.join(_DIR, "model_cache")

# ---- Data loading (cached) ----

_cached_models = None
_cached_baselines = None


def get_models():
    """Load all trained models across all task families. Cached once per process."""
    global _cached_models
    if _cached_models is not None:
        return _cached_models

    os.makedirs(CACHE_DIR, exist_ok=True)
    models = []
    task_counts = {}

    for task_name, model_dir in MODEL_DIRS.items():
        if not os.path.exists(model_dir):
            continue
        model_files = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
        count = 0
        for path in model_files:
            name = os.path.splitext(os.path.basename(path))[0]
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

            config = ckpt["config"]
            weights_np = ckpt["weights_np"]
            residuals = torch.tensor(ckpt["residuals"], dtype=torch.float64)

            # Check if model trained well enough
            # if "labels" in ckpt:
            # labels = ckpt["labels"]
            # For models with a classification task, check implicit accuracy
            # by seeing if residuals cluster by label
            # We trust the training script's accuracy check — filter by checking
            # if the checkpoint has reasonable residuals
            n_samples = residuals.shape[0]
            if n_samples < 4:
                continue

            # Serialize for subprocess
            npz_path = os.path.join(CACHE_DIR, f"{name}_weights.npz")
            config_path = os.path.join(CACHE_DIR, f"{name}_config.json")
            if not os.path.exists(npz_path):
                np.savez(npz_path, **weights_np)
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    json.dump(config, f)

            models.append(
                {
                    "name": name,
                    "task": config.get("task", "modular_addition"),
                    "config": config,
                    "residuals": residuals,
                    "npz_path": npz_path,
                    "config_path": config_path,
                }
            )
            count += 1
        task_counts[task_name] = count

    _cached_models = models
    print(f"  Loaded {len(models)} models: {task_counts}")
    return models


def get_baselines():
    """Compute diagonal Gaussian baseline log-prob for each model."""
    global _cached_baselines
    if _cached_baselines is not None:
        return _cached_baselines

    models = get_models()
    _cached_baselines = {}
    for m in models:
        residuals = m["residuals"]
        mu = residuals.mean(dim=0)
        std = residuals.std(dim=0).clamp(min=1e-6)
        baseline_dist = torch.distributions.Independent(
            torch.distributions.Normal(mu, std), 1
        )
        _cached_baselines[m["name"]] = float(baseline_dist.log_prob(residuals).mean())

    print(
        f"  Baselines: min={min(_cached_baselines.values()):.1f}, max={max(_cached_baselines.values()):.1f}"
    )
    return _cached_baselines


# ---- Candidate execution ----


def make_wrapper_script(program_path, models_info, results_path):
    return f"""\
import json
import numpy as np

def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {{k: _to_serializable(v) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

spec = __import__('importlib.util').util.spec_from_file_location("program", {program_path!r})
program = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(program)

models_info = {json.dumps(models_info)}

results = []
for mi in models_info:
    weights = dict(np.load(mi["npz_path"]))
    with open(mi["config_path"]) as f:
        config = json.load(f)
    config.pop("task", None)
    try:
        dist_dict = program.build_dist(weights, config)
        results.append({{"ok": True, "dist": _to_serializable(dist_dict)}})
    except Exception as e:
        results.append({{"ok": False, "error": str(e)}})

with open({results_path!r}, 'w') as f:
    json.dump(results, f)
"""


def _check_perf_paranoid():
    with open("/proc/sys/kernel/perf_event_paranoid") as f:
        level = int(f.read().strip())
    if level > 2:
        raise RuntimeError(
            f"perf_event_paranoid is {level} but must be <= 2.\n"
            f"  echo 'kernel.perf_event_paranoid = 2' | sudo tee /etc/sysctl.d/99-perf.conf\n"
            f"  sudo sysctl -p /etc/sysctl.d/99-perf.conf"
        )


def parse_perf_flops(stderr):
    for line in stderr.splitlines():
        if "fp_flops_retired" in line:
            token = line.strip().split()[0].replace(",", "")
            if token == "<not":
                return None
            return int(token)
    return None


_cached_baseline_flops = None


def get_baseline_flops():
    global _cached_baseline_flops
    if _cached_baseline_flops is None:
        models = get_models()
        subset = models[:3]
        models_info = [
            {"npz_path": m["npz_path"], "config_path": m["config_path"]} for m in subset
        ]
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".py") as noop_f,
            tempfile.NamedTemporaryFile(suffix=".json") as noop_results_f,
        ):
            noop_f.write(
                "import numpy as np, json\n"
                "def build_dist(w, c):\n"
                '    d = c["d_model"]\n'
                '    return {"format":"mossaic.distgraph","version":1,"root":"y",\n'
                '      "nodes":[{"id":"m","kind":"const","value":[0.0]*d},\n'
                '       {"id":"s","kind":"const","value":[0.0]*d},\n'
                '       {"id":"sc","kind":"tensor_op","op":"softplus","args":[{"$ref":"s"}],"kwargs":{}},\n'
                '       {"id":"b","kind":"call","callable":"torch.distributions.Normal",\n'
                '        "returns":"distribution","kwargs":{"loc":{"$ref":"m"},"scale":{"$ref":"sc"}}},\n'
                '       {"id":"y","kind":"call","callable":"torch.distributions.Independent",\n'
                '        "returns":"distribution","kwargs":{"base_distribution":{"$ref":"b"},\n'
                '        "reinterpreted_batch_ndims":1}}]}\n'
            )
            noop_f.flush()
            _, _cached_baseline_flops = _run_candidate(
                noop_f.name, models_info, noop_results_f.name
            )
    return _cached_baseline_flops


def _run_candidate(program_path, models_info, results_path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as wrapper_f:
        wrapper_f.write(make_wrapper_script(program_path, models_info, results_path))
        wrapper_f.flush()
        result = subprocess.run(
            [
                "perf",
                "stat",
                "-e",
                PERF_EVENT,
                "--",
                "taskset",
                "-c",
                E_CORE,
                sys.executable,
                wrapper_f.name,
            ],
            capture_output=True,
            text=True,
            timeout=EVAL_TIMEOUT,
        )
        flops = parse_perf_flops(result.stderr)
        return result, flops


def score_distgraph(dist_dict, y_samples, expected_d):
    dist = execute_distgraph(dist_dict, expected_event_shape=(expected_d,))
    log_probs = dist.log_prob(y_samples)
    log_probs = torch.clamp(log_probs, min=LOG_PROB_CLAMP_MIN, max=LOG_PROB_CLAMP_MAX)
    valid = torch.isfinite(log_probs)
    if valid.sum() == 0:
        return float("-inf")
    return float(log_probs[valid].mean())


# ---- Main entry point ----


def evaluate(program_path):
    try:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {program_path}")
        print("=" * 60)
        with open(program_path) as f:
            print(f.read())
        print("=" * 60)

        _check_perf_paranoid()
        all_models = get_models()
        baselines = get_baselines()
        d_model = all_models[0]["config"]["d_model"]
        baseline_flops = get_baseline_flops()

        # Sample across task families — ensure at least 1 per family if possible
        by_task = {}
        for m in all_models:
            by_task.setdefault(m["task"], []).append(m)

        subset = []
        tasks_available = list(by_task.keys())
        random.shuffle(tasks_available)

        # Pick 1-2 from each task family
        for task in tasks_available:
            pool = by_task[task]
            n_pick = min(2, len(pool))
            subset.extend(random.sample(pool, n_pick))

        # Fill remaining slots randomly
        remaining = [m for m in all_models if m not in subset]
        n_extra = max(0, MODELS_PER_EVAL - len(subset))
        if n_extra > 0 and remaining:
            subset.extend(random.sample(remaining, min(n_extra, len(remaining))))

        models_info = [
            {"npz_path": m["npz_path"], "config_path": m["config_path"]} for m in subset
        ]

        with tempfile.NamedTemporaryFile(suffix=".json") as results_f:
            result, flops = _run_candidate(program_path, models_info, results_f.name)

            if result.returncode != 0:
                with open(program_path) as f:
                    code = f.read()
                return {
                    "combined_score": -1e9,
                    "error": f"Candidate crashed:\n{result.stderr}\n\n--- Candidate code ---\n{code}",
                }

            with open(results_f.name) as f:
                raw_results = json.load(f)

        per_model = {}
        per_task_improvements = {}
        errors = []
        for model, entry in zip(subset, raw_results):
            name = model["name"]
            task = model["task"]
            if not entry.get("ok"):
                per_model[name] = -1e9
                errors.append(f"{name}: {entry.get('error', 'unknown')}")
                continue
            try:
                raw = score_distgraph(entry["dist"], model["residuals"], d_model)
                improvement = raw - baselines[name]
                per_model[name] = improvement
                per_task_improvements.setdefault(task, []).append(improvement)
            except (DistGraphError, Exception) as e:
                per_model[name] = -1e9
                errors.append(f"{name}: {e}")

        improvements = [v for v in per_model.values() if v > -1e8]
        mean_improvement = (
            sum(improvements) / len(improvements) if improvements else -1e9
        )

        # Per-task summary
        task_summary = {}
        for task, imps in per_task_improvements.items():
            task_summary[task] = sum(imps) / len(imps)

        # FLOP penalty
        if flops is not None and baseline_flops is not None:
            candidate_flops = max(flops - baseline_flops, 0)
            flop_penalty_nats = FLOP_PENALTY_NATS * candidate_flops / 1e6
        else:
            candidate_flops = None
            flop_penalty_nats = 0.0

        return {
            "combined_score": mean_improvement - flop_penalty_nats,
            "mean_improvement": mean_improvement,
            "flop_penalty_nats": flop_penalty_nats,
            "flops_candidate": candidate_flops,
            "per_task": task_summary,
            "per_model": {k: round(v, 2) for k, v in per_model.items()},
            "models_evaluated": [m["name"] for m in subset],
            "tasks_evaluated": list(set(m["task"] for m in subset)),
            "n_models": len(subset),
            "n_failed": len(errors),
            "errors": errors[:5] if errors else [],
        }

    except subprocess.TimeoutExpired:
        return {
            "combined_score": -1e9,
            "error": f"Candidate timed out after {EVAL_TIMEOUT}s",
        }
    except Exception:
        return {
            "combined_score": -1e9,
            "error": traceback.format_exc(),
        }
