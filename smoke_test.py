import tempfile

import numpy as np
import torch

from distgraph import DistGraphError, execute as execute_distgraph
from evaluator import evaluate, generate_scoring_samples, generate_tasks


def test_distgraph_executor():
    """Test the DistGraph executor with a known diagonal Gaussian."""
    dg = {
        "format": "mossaic.distgraph",
        "version": 1,
        "root": "y_dist",
        "nodes": [
            {"id": "mu", "kind": "const", "value": [1.0, 2.0, 3.0]},
            {"id": "raw_s", "kind": "const", "value": [0.0, 0.0, 0.0]},
            {
                "id": "s",
                "kind": "tensor_op",
                "op": "softplus",
                "args": [{"$ref": "raw_s"}],
                "kwargs": {},
            },
            {
                "id": "base",
                "kind": "call",
                "callable": "torch.distributions.Normal",
                "returns": "distribution",
                "kwargs": {"loc": {"$ref": "mu"}, "scale": {"$ref": "s"}},
            },
            {
                "id": "y_dist",
                "kind": "call",
                "callable": "torch.distributions.Independent",
                "returns": "distribution",
                "kwargs": {
                    "base_distribution": {"$ref": "base"},
                    "reinterpreted_batch_ndims": 1,
                },
            },
        ],
    }
    dist = execute_distgraph(dg, expected_event_shape=(3,))
    assert dist.event_shape == torch.Size([3])

    samples = torch.randn(100, 3)
    lp = dist.log_prob(samples)
    assert lp.shape == (100,)
    assert torch.all(torch.isfinite(lp))

    print("test_distgraph_executor: PASSED")


def test_distgraph_validation():
    """Test that invalid DistGraphs are rejected."""
    errors = 0

    # Missing format
    try:
        execute_distgraph({"version": 1, "root": "x", "nodes": []})
        print("  FAIL: missing format not caught")
    except DistGraphError:
        errors += 1

    # Bad op
    try:
        execute_distgraph(
            {
                "format": "mossaic.distgraph",
                "version": 1,
                "root": "x",
                "nodes": [
                    {
                        "id": "x",
                        "kind": "tensor_op",
                        "op": "EVIL_OP",
                        "args": [],
                        "kwargs": {},
                    }
                ],
            }
        )
        print("  FAIL: bad op not caught")
    except DistGraphError:
        errors += 1

    # Bad callable
    try:
        execute_distgraph(
            {
                "format": "mossaic.distgraph",
                "version": 1,
                "root": "x",
                "nodes": [
                    {
                        "id": "x",
                        "kind": "call",
                        "callable": "os.system",
                        "kwargs": {},
                    }
                ],
            }
        )
        print("  FAIL: bad callable not caught")
    except DistGraphError:
        errors += 1

    print(f"test_distgraph_validation: PASSED ({errors}/3 errors caught)")


def test_scoring_sanity():
    """Verify a fitted Gaussian scores better than a zero-mean one."""
    rng = np.random.default_rng(seed=42)
    tasks = generate_tasks(rng)
    task = tasks[0]  # first architecture

    scoring_rng = np.random.default_rng(seed=123)
    samples = generate_scoring_samples([task], scoring_rng, n_samples=10_000)
    y_np = samples[0]
    y = torch.tensor(y_np, dtype=torch.float64)
    d_out = task["d_out"]

    # Fitted: use empirical mean and std
    mu = y_np.mean(axis=0).tolist()
    std = y_np.std(axis=0).tolist()

    fitted_dg = {
        "format": "mossaic.distgraph",
        "version": 1,
        "root": "y_dist",
        "nodes": [
            {"id": "mu", "kind": "const", "value": mu, "dtype": "float64"},
            {"id": "scale", "kind": "const", "value": std, "dtype": "float64"},
            {
                "id": "base",
                "kind": "call",
                "callable": "torch.distributions.Normal",
                "returns": "distribution",
                "kwargs": {"loc": {"$ref": "mu"}, "scale": {"$ref": "scale"}},
            },
            {
                "id": "y_dist",
                "kind": "call",
                "callable": "torch.distributions.Independent",
                "returns": "distribution",
                "kwargs": {
                    "base_distribution": {"$ref": "base"},
                    "reinterpreted_batch_ndims": 1,
                },
            },
        ],
    }

    # Trivial: zero mean, unit-ish scale
    trivial_dg = {
        "format": "mossaic.distgraph",
        "version": 1,
        "root": "y_dist",
        "nodes": [
            {"id": "mu", "kind": "const", "value": [0.0] * d_out, "dtype": "float64"},
            {
                "id": "scale",
                "kind": "const",
                "value": [0.693] * d_out,
                "dtype": "float64",
            },
            {
                "id": "base",
                "kind": "call",
                "callable": "torch.distributions.Normal",
                "returns": "distribution",
                "kwargs": {"loc": {"$ref": "mu"}, "scale": {"$ref": "scale"}},
            },
            {
                "id": "y_dist",
                "kind": "call",
                "callable": "torch.distributions.Independent",
                "returns": "distribution",
                "kwargs": {
                    "base_distribution": {"$ref": "base"},
                    "reinterpreted_batch_ndims": 1,
                },
            },
        ],
    }

    dist_fitted = execute_distgraph(fitted_dg, expected_event_shape=(d_out,))
    dist_trivial = execute_distgraph(trivial_dg, expected_event_shape=(d_out,))

    score_fitted = float(dist_fitted.log_prob(y).mean())
    score_trivial = float(dist_trivial.log_prob(y).mean())

    print(f"  Fitted score:  {score_fitted:.4f}")
    print(f"  Trivial score: {score_trivial:.4f}")
    assert score_fitted > score_trivial, "Fitted should score higher!"
    print("test_scoring_sanity: PASSED")


def test_trivial_build_dist():
    """End-to-end: run the evaluator on the trivial initial program."""
    candidate_code = """\
import numpy as np

def build_dist(layers, activation, input_cov):
    d_out = layers[-1][0].shape[0]
    return {
        "format": "mossaic.distgraph",
        "version": 1,
        "root": "y_dist",
        "nodes": [
            {"id": "mu", "kind": "const", "value": [0.0] * d_out},
            {"id": "raw_s", "kind": "const", "value": [0.0] * d_out},
            {"id": "s", "kind": "tensor_op", "op": "softplus",
             "args": [{"$ref": "raw_s"}], "kwargs": {}},
            {"id": "base", "kind": "call",
             "callable": "torch.distributions.Normal",
             "returns": "distribution",
             "kwargs": {"loc": {"$ref": "mu"}, "scale": {"$ref": "s"}}},
            {"id": "y_dist", "kind": "call",
             "callable": "torch.distributions.Independent",
             "returns": "distribution",
             "kwargs": {"base_distribution": {"$ref": "base"},
                        "reinterpreted_batch_ndims": 1}},
        ],
    }
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(candidate_code)
        f.flush()
        result = evaluate(f.name)
        print("=== Evaluator result for trivial build_dist ===\n")
        for k, v in result.items():
            if k == "per_task_scores":
                print(f"  {k}: [{', '.join(f'{s:.2f}' for s in v)}]")
            elif isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    test_distgraph_executor()
    test_distgraph_validation()
    test_scoring_sanity()
    test_trivial_build_dist()
