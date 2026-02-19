import json
import math
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np
from scipy.special import erf, expit

PERF_EVENT = "fp_flops_retired.all"
E_CORE = "16"
FLOP_PENALTY_K = 100  # flops_ratio at which penalty = 50%
MC_N_SAMPLES = 10_000_000  # samples for MC ground truth

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

TARGET_LOG_PROBS = [-0.5, -1, -2, -3]

_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_NPZ_PATH = os.path.join(_DIR, "tasks_data.npz")
TASKS_MANIFEST_PATH = os.path.join(_DIR, "tasks_manifest.json")


def forward_pass(layers, activation_fn, x):
    """Run a batch of inputs through a network.

    x: (n_samples, d_in)
    activation_fn: callable applied between layers
    Returns: (n_samples, d_out)
    """
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = activation_fn(h)
    return h


def compute_forward_pass_flops(layers):
    """Compute FLOP cost of one forward pass through the network."""
    flops = 0
    for i, (W, b) in enumerate(layers):
        d_out, d_in = W.shape
        flops += 2 * d_in * d_out + d_out  # matmul + bias
        if i < len(layers) - 1:
            flops += d_out  # activation (approximated as 1 op per element)
    return flops


def generate_tasks(rng):
    """Generate task suite: variable-depth networks with diverse activations."""
    tasks = []

    for n_layers, d_in, d_hidden, d_out, activation_name in ARCHITECTURES:
        activation_fn = ACTIVATIONS[activation_name]

        # Random weights
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

        # Random input covariance: Σ = A A^T / d_in + 0.1 I
        A = rng.standard_normal((d_in, d_in))
        input_cov = A @ A.T / d_in + 0.1 * np.eye(d_in)

        # Random direction
        a = rng.standard_normal(d_out)

        # MC ground truth: sample x ~ N(0, Σ) via Cholesky
        L = np.linalg.cholesky(input_cov)
        z = rng.standard_normal((MC_N_SAMPLES, d_in))
        x = z @ L.T  # x ~ N(0, Σ)
        y = forward_pass(layers, activation_fn, x)
        scores = y @ a

        for target_log_p in TARGET_LOG_PROBS:
            target_p = 10.0**target_log_p
            t = float(np.quantile(scores, 1 - target_p))

            tasks.append(
                {
                    "layers": layers,
                    "activation": activation_name,
                    "a": a,
                    "t": t,
                    "input_cov": input_cov,
                    "ground_truth": target_p,
                    "forward_pass_flops": compute_forward_pass_flops(layers),
                }
            )

    return tasks


_cached_tasks = None
_cached_baseline_flops = None
_cached_task_files = None


def get_tasks():
    """Get the task suite. Loads from disk if pre-computed, otherwise generates and saves."""
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
    """Get serialized task file paths. Ensures tasks are generated/saved first."""
    global _cached_task_files
    if _cached_task_files is None:
        get_tasks()  # ensures files exist at TASKS_NPZ_PATH / TASKS_MANIFEST_PATH
        _cached_task_files = (TASKS_NPZ_PATH, TASKS_MANIFEST_PATH)
    return _cached_task_files


def get_baseline_flops():
    """Get baseline FLOP overhead, measuring only once per process."""
    global _cached_baseline_flops
    if _cached_baseline_flops is None:
        npz_path, manifest_path = get_task_files()
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".py") as noop_f,
            tempfile.NamedTemporaryFile(suffix=".json") as noop_results_f,
        ):
            noop_f.write(
                "def estimate(layers, activation, a, t, input_cov):\n    return 0.0\n"
            )
            noop_f.flush()
            _, _cached_baseline_flops = _run_candidate(
                noop_f.name, npz_path, manifest_path, noop_results_f.name
            )
    return _cached_baseline_flops


def mc_ground_truth(layers, activation_name, a, t, input_cov, n_samples, rng):
    """Estimate P(a^T y > t) by Monte Carlo sampling with correlated inputs."""
    activation_fn = ACTIVATIONS[activation_name]
    d_in = layers[0][0].shape[1]
    L = np.linalg.cholesky(input_cov)
    z = rng.standard_normal((n_samples, d_in))
    x = z @ L.T
    y = forward_pass(layers, activation_fn, x)
    scores = y @ a
    return float(np.mean(scores > t))


def parse_perf_flops(stderr):
    """Parse FLOP count from perf stat stderr output."""
    for line in stderr.splitlines():
        if "fp_flops_retired" in line:
            token = line.strip().split()[0].replace(",", "")
            if token == "<not":
                return None
            return int(token)
    return None


def save_tasks(tasks, npz_path, manifest_path):
    """Serialize tasks to .npz + JSON manifest."""
    arrays = {}
    manifest = []

    for i, task in enumerate(tasks):
        for j, (W, b) in enumerate(task["layers"]):
            arrays[f"W_{i}_{j}"] = W
            arrays[f"b_{i}_{j}"] = b
        arrays[f"a_{i}"] = task["a"]
        arrays[f"t_{i}"] = np.array([task["t"]])
        arrays[f"input_cov_{i}"] = task["input_cov"]

        manifest.append(
            {
                "n_layers": len(task["layers"]),
                "activation": task["activation"],
                "ground_truth": task["ground_truth"],
                "forward_pass_flops": task["forward_pass_flops"],
            }
        )

    np.savez(npz_path, **arrays)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


def load_tasks(npz_path, manifest_path):
    """Load tasks from .npz + JSON manifest."""
    data = np.load(npz_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    tasks = []
    for i, meta in enumerate(manifest):
        layers = []
        for j in range(meta["n_layers"]):
            W = data[f"W_{i}_{j}"]
            b = data[f"b_{i}_{j}"]
            layers.append((W, b))
        tasks.append(
            {
                "layers": layers,
                "activation": meta["activation"],
                "a": data[f"a_{i}"],
                "t": float(data[f"t_{i}"][0]),
                "input_cov": data[f"input_cov_{i}"],
                "ground_truth": meta["ground_truth"],
                "forward_pass_flops": meta["forward_pass_flops"],
            }
        )
    return tasks


def make_wrapper_script(program_path, npz_path, manifest_path, results_path):
    """Generate the Python script that runs inside perf stat."""
    return f"""\
import json, numpy as np

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
    a = data[f"a_{{i}}"]
    t = float(data[f"t_{{i}}"][0])
    input_cov = data[f"input_cov_{{i}}"]
    p_hat = program.estimate(layers, activation, a, t, input_cov)
    results.append(float(p_hat))

with open({results_path!r}, 'w') as f:
    json.dump(results, f)
"""


def _run_candidate(program_path, npz_path, manifest_path, results_path):
    """Run a candidate program via perf stat and return (perf_result, flops)."""
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
            timeout=120,
        )
        flops = parse_perf_flops(result.stderr)
        return result, flops


def evaluate_before_wrapper(program_path):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {program_path}")
    print("=" * 60)
    with open(program_path) as f:
        print(f.read())
    print("=" * 60)

    tasks = get_tasks()
    npz_path, manifest_path = get_task_files()
    baseline_flops = get_baseline_flops()

    with tempfile.NamedTemporaryFile(suffix=".json") as results_f:
        result, flops = _run_candidate(
            program_path, npz_path, manifest_path, results_f.name
        )

        if result.returncode != 0:
            return {
                "combined_score": 0.0,
                "error": f"Candidate crashed:\n{result.stderr}",
            }

        with open(results_f.name) as f:
            p_hats = json.load(f)

    log_errors = []
    for p_hat, task in zip(p_hats, tasks):
        p_star = task["ground_truth"]
        p_hat = max(float(p_hat), 1e-15)
        p_star = max(p_star, 1e-15)
        log_errors.append(abs(math.log(p_hat) - math.log(p_star)))

    mean_log_error = sum(log_errors) / len(log_errors)
    accuracy_score = 1.0 / (1.0 + mean_log_error)

    # FLOP penalty: subtract baseline, normalize by forward-pass cost
    forward_pass_flops = sum(task["forward_pass_flops"] for task in tasks)
    if flops is not None and baseline_flops is not None:
        candidate_flops = max(flops - baseline_flops, 0)
        flops_ratio = candidate_flops / forward_pass_flops
        flop_penalty = 1.0 / (1.0 + flops_ratio / FLOP_PENALTY_K)
    else:
        candidate_flops = None
        flops_ratio = None
        flop_penalty = 1.0

    return {
        "accuracy_score": accuracy_score,
        "mean_log_error": mean_log_error,
        "flops_raw": flops,
        "flops_baseline": baseline_flops,
        "flops_candidate": candidate_flops,
        "forward_pass_flops": forward_pass_flops,
        "flops_ratio": flops_ratio,
        "flop_penalty": flop_penalty,
        "combined_score": accuracy_score * flop_penalty,
        "n_tasks": len(tasks),
    }


def evaluate(program_path):
    try:
        return evaluate_before_wrapper(program_path)
    except Exception:
        return {
            "combined_score": 0.0,
            "error": traceback.format_exc(),
        }
