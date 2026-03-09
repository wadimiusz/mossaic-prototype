"""Evaluator for MoSSAIC DistGraph evolution.

Scores candidates by mean log-probability of the true network output distribution
under the candidate's declared distribution (DistGraph v1).
"""

import json
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch
from scipy.special import erf, expit

from distgraph import DistGraphError, execute as execute_distgraph

EVAL_TIMEOUT = 120  # seconds; candidate killed and scored 0 if exceeded
N_SCORING_SAMPLES = 1_000_000
LOG_PROB_CLAMP_MIN = -1e6
LOG_PROB_CLAMP_MAX = 1e6
FLOP_PENALTY_NATS = 10  # nats of penalty per unit of flops_ratio
PERF_EVENT = "fp_flops_retired.all"
E_CORE = "16"

# --- Activation functions (mirrored in initial_program.py) ---


def relu(x):
    return np.maximum(x, 0)


def tanh_act(x):
    return np.tanh(x)


def sigmoid(x):
    return expit(x)


def gelu(x):
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def softplus(x):
    return np.log1p(np.exp(x))


def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1.0)


def silu(x):
    return x * expit(x)


ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh_act,
    "sigmoid": sigmoid,
    "gelu": gelu,
    "leaky_relu": leaky_relu,
    "softplus": softplus,
    "elu": elu,
    "silu": silu,
}

# --- Architecture configs ---
# (n_layers, d_in, d_hidden, d_out, activation)
ARCHITECTURES = [
    # 2-layer networks
    (2, 8, 16, 8, "relu"),
    (2, 4, 8, 4, "tanh"),
    (2, 16, 32, 8, "sigmoid"),
    (2, 8, 16, 4, "gelu"),
    # 3-layer networks
    (3, 8, 16, 8, "leaky_relu"),
    (3, 4, 8, 4, "softplus"),
    (3, 16, 16, 8, "elu"),
    (3, 8, 12, 4, "silu"),
    # 4-layer networks
    (4, 8, 16, 8, "relu"),
    (4, 4, 8, 4, "tanh"),
    (4, 8, 12, 8, "gelu"),
    (4, 16, 16, 8, "sigmoid"),
    # 5-layer networks
    (5, 8, 16, 8, "leaky_relu"),
    (5, 4, 8, 4, "elu"),
    (5, 8, 12, 4, "softplus"),
    (5, 16, 16, 8, "silu"),
    # 6-layer networks
    (6, 8, 16, 8, "relu"),
    (6, 4, 8, 4, "tanh"),
    (6, 8, 12, 8, "gelu"),
    (6, 16, 16, 4, "sigmoid"),
]

_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_NPZ_PATH = os.path.join(_DIR, "tasks_data.npz")
TASKS_MANIFEST_PATH = os.path.join(_DIR, "tasks_manifest.json")


def forward_pass(layers, activation_fn, x):
    """Run a batch of inputs through a network.

    x: (n_samples, d_in)
    Returns: (n_samples, d_out)
    """
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = activation_fn(h)
    return h


# ---- Task generation and caching ----


def generate_tasks(rng):
    """Generate task suite: 20 architectures with random weights and input covariance."""
    tasks = []
    for n_layers, d_in, d_hidden, d_out, activation_name in ARCHITECTURES:
        layers = []
        for i in range(n_layers):
            if i == 0:
                W = rng.standard_normal((d_hidden, d_in))
                b = rng.standard_normal(d_hidden)
            elif i == n_layers - 1:
                W = rng.standard_normal((d_out, d_hidden))
                b = rng.standard_normal(d_out)
            else:
                W = rng.standard_normal((d_hidden, d_hidden))
                b = rng.standard_normal(d_hidden)
            layers.append((W, b))

        A = rng.standard_normal((d_in, d_in))
        input_cov = A @ A.T / d_in + 0.1 * np.eye(d_in)

        tasks.append(
            {
                "layers": layers,
                "activation": activation_name,
                "input_cov": input_cov,
                "d_out": d_out,
            }
        )
    return tasks


def generate_scoring_samples(tasks, rng, n_samples=N_SCORING_SAMPLES):
    """Generate y = f(x) samples for scoring, one array per task."""
    samples = []
    for task in tasks:
        activation_fn = ACTIVATIONS[task["activation"]]
        d_in = task["layers"][0][0].shape[1]
        L = np.linalg.cholesky(task["input_cov"])
        z = rng.standard_normal((n_samples, d_in))
        x = z @ L.T
        y = forward_pass(task["layers"], activation_fn, x)
        samples.append(y)
    return samples


def save_tasks(tasks, npz_path, manifest_path):
    """Serialize task weights/metadata to disk (for subprocess access)."""
    arrays = {}
    manifest = []
    for i, task in enumerate(tasks):
        for j, (W, b) in enumerate(task["layers"]):
            arrays[f"W_{i}_{j}"] = W
            arrays[f"b_{i}_{j}"] = b
        arrays[f"input_cov_{i}"] = task["input_cov"]
        manifest.append(
            {
                "n_layers": len(task["layers"]),
                "activation": task["activation"],
                "d_out": task["d_out"],
            }
        )
    np.savez(npz_path, **arrays)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


def load_tasks(npz_path, manifest_path):
    """Load tasks from disk."""
    data = np.load(npz_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    tasks = []
    for i, meta in enumerate(manifest):
        layers = []
        for j in range(meta["n_layers"]):
            layers.append((data[f"W_{i}_{j}"], data[f"b_{i}_{j}"]))
        tasks.append(
            {
                "layers": layers,
                "activation": meta["activation"],
                "input_cov": data[f"input_cov_{i}"],
                "d_out": meta["d_out"],
            }
        )
    return tasks


# ---- Caches ----

_cached_tasks = None
_cached_scoring_samples = None  # list of torch tensors
_cached_task_files = None


def get_tasks():
    global _cached_tasks
    if _cached_tasks is None:
        if os.path.exists(TASKS_NPZ_PATH) and os.path.exists(TASKS_MANIFEST_PATH):
            _cached_tasks = load_tasks(TASKS_NPZ_PATH, TASKS_MANIFEST_PATH)
        else:
            rng = np.random.default_rng(seed=42)
            _cached_tasks = generate_tasks(rng)
            save_tasks(_cached_tasks, TASKS_NPZ_PATH, TASKS_MANIFEST_PATH)
    return _cached_tasks


def get_task_files():
    global _cached_task_files
    if _cached_task_files is None:
        get_tasks()
        _cached_task_files = (TASKS_NPZ_PATH, TASKS_MANIFEST_PATH)
    return _cached_task_files


def get_scoring_samples():
    global _cached_scoring_samples
    if _cached_scoring_samples is None:
        tasks = get_tasks()
        rng = np.random.default_rng(seed=123)
        np_samples = generate_scoring_samples(tasks, rng)
        _cached_scoring_samples = [
            torch.tensor(y, dtype=torch.float64) for y in np_samples
        ]
    return _cached_scoring_samples


# ---- FLOP counting ----


def _check_perf_paranoid():
    """Fail fast if perf_event_paranoid is too restrictive for FLOP counting."""
    with open("/proc/sys/kernel/perf_event_paranoid") as f:
        level = int(f.read().strip())
    if level > 2:
        raise RuntimeError(
            f"perf_event_paranoid is {level} but must be ≤ 2 for FLOP counting.\n"
            f"Apply permanently with:\n"
            f"  echo 'kernel.perf_event_paranoid = 2' | sudo tee /etc/sysctl.d/99-perf.conf\n"
            f"  sudo sysctl -p /etc/sysctl.d/99-perf.conf\n"
            f"Or temporarily for this boot:\n"
            f"  echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid"
        )


def parse_perf_flops(stderr):
    """Parse FLOP count from perf stat stderr output."""
    for line in stderr.splitlines():
        if "fp_flops_retired" in line:
            token = line.strip().split()[0].replace(",", "")
            if token == "<not":
                return None
            return int(token)
    return None


def compute_forward_pass_flops(layers):
    """Approximate FLOP cost of one forward pass through the network."""
    flops = 0
    for i, (W, b) in enumerate(layers):
        d_out, d_in = W.shape
        flops += 2 * d_in * d_out + d_out
        if i < len(layers) - 1:
            flops += d_out
    return flops


_cached_baseline_flops = None


def get_baseline_flops():
    """Measure baseline FLOP overhead (no-op candidate), cached once per process."""
    global _cached_baseline_flops
    if _cached_baseline_flops is None:
        npz_path, manifest_path = get_task_files()
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".py") as noop_f,
            tempfile.NamedTemporaryFile(suffix=".json") as noop_results_f,
        ):
            # No-op: returns a trivial DistGraph for each task
            noop_f.write(
                "import json, numpy as np\n"
                "def build_dist(layers, activation, input_cov):\n"
                "    d = layers[-1][0].shape[0]\n"
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
                noop_f.name, npz_path, manifest_path, noop_results_f.name
            )
    return _cached_baseline_flops


# ---- Candidate execution ----


def make_wrapper_script(program_path, npz_path, manifest_path, results_path):
    """Generate script that calls build_dist for each task and writes JSON results."""
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

data = np.load({npz_path!r})
with open({manifest_path!r}) as f:
    manifest = json.load(f)

results = []
for i, meta in enumerate(manifest):
    layers = []
    for j in range(meta["n_layers"]):
        W = data[f"W_{{i}}_{{j}}"]
        b = data[f"b_{{i}}_{{j}}"]
        layers.append((W, b))
    activation = meta["activation"]
    input_cov = data[f"input_cov_{{i}}"]
    try:
        dist_dict = program.build_dist(layers, activation, input_cov)
        results.append({{"ok": True, "dist": _to_serializable(dist_dict)}})
    except Exception as e:
        results.append({{"ok": False, "error": str(e)}})

with open({results_path!r}, 'w') as f:
    json.dump(results, f)
"""


def _run_candidate(program_path, npz_path, manifest_path, results_path):
    """Run candidate via perf stat and return (result, flops)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as wrapper_f:
        wrapper_f.write(
            make_wrapper_script(program_path, npz_path, manifest_path, results_path)
        )
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


def score_distgraph(dist_dict, y_samples, expected_d_out):
    """Execute a DistGraph and return mean log-prob on samples."""
    dist = execute_distgraph(dist_dict, expected_event_shape=(expected_d_out,))
    log_probs = dist.log_prob(y_samples)
    log_probs = torch.clamp(log_probs, min=LOG_PROB_CLAMP_MIN, max=LOG_PROB_CLAMP_MAX)
    valid = torch.isfinite(log_probs)
    if valid.sum() == 0:
        return float("-inf")
    return float(log_probs[valid].mean())


# ---- Main entry point ----


def evaluate_before_wrapper(program_path):
    _check_perf_paranoid()

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {program_path}")
    print("=" * 60)
    with open(program_path) as f:
        print(f.read())
    print("=" * 60)

    tasks = get_tasks()
    scoring_samples = get_scoring_samples()
    npz_path, manifest_path = get_task_files()
    baseline_flops = get_baseline_flops()

    with tempfile.NamedTemporaryFile(suffix=".json") as results_f:
        result, flops = _run_candidate(
            program_path, npz_path, manifest_path, results_f.name
        )

        if result.returncode != 0:
            return {
                "combined_score": -1e9,
                "error": f"Candidate crashed:\n{result.stderr}",
            }

        with open(results_f.name) as f:
            raw_results = json.load(f)

    # Score each task's DistGraph
    task_scores = []
    task_errors = []
    for i, (entry, y_samples, task) in enumerate(
        zip(raw_results, scoring_samples, tasks)
    ):
        if not entry.get("ok"):
            task_scores.append(-1e9)
            task_errors.append(entry.get("error", "unknown error"))
            continue
        try:
            score = score_distgraph(entry["dist"], y_samples, task["d_out"])
            task_scores.append(score)
            task_errors.append(None)
        except (DistGraphError, Exception) as e:
            task_scores.append(-1e9)
            task_errors.append(str(e))

    mean_log_prob = sum(task_scores) / len(task_scores)

    # FLOP penalty (additive, in nats)
    forward_pass_flops = sum(compute_forward_pass_flops(t["layers"]) for t in tasks)
    if flops is not None and baseline_flops is not None:
        candidate_flops = max(flops - baseline_flops, 0)
        flops_ratio = candidate_flops / forward_pass_flops
        flop_penalty_nats = FLOP_PENALTY_NATS * flops_ratio
    else:
        candidate_flops = None
        flops_ratio = None
        flop_penalty_nats = 0.0

    errors = [e for e in task_errors if e is not None]

    return {
        "combined_score": mean_log_prob - flop_penalty_nats,
        "mean_log_prob": mean_log_prob,
        "flop_penalty_nats": flop_penalty_nats,
        "flops_raw": flops,
        "flops_baseline": baseline_flops,
        "flops_candidate": candidate_flops,
        "flops_ratio": flops_ratio,
        "per_task_scores": task_scores,
        "n_tasks": len(tasks),
        "n_failed_tasks": len(errors),
        "errors": errors[:5] if errors else [],
    }


def evaluate(program_path):
    """Evaluate a candidate program. Called by OpenEvolve."""
    try:
        return evaluate_before_wrapper(program_path)
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
