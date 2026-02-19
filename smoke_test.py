import tempfile

import numpy as np

from evaluator import (
    evaluate,
    generate_tasks,
    mc_ground_truth,
)


def cross_check_mc():
    """Verify MC ground truth is stable (two seeds should agree)."""
    rng = np.random.default_rng(seed=42)
    tasks = generate_tasks(rng)

    print("=== Cross-check: MC seed A vs MC seed B ===\n")
    print(
        f"{'arch':>20}  {'act':>12}  {'target p':>12}  {'MC-A (1e6)':>12}  {'MC-B (1e6)':>12}  {'ratio':>8}"
    )
    print("-" * 86)

    mc_rng_a = np.random.default_rng(seed=100)
    mc_rng_b = np.random.default_rng(seed=200)
    for task in tasks:
        p_a = mc_ground_truth(
            task["layers"],
            task["activation"],
            task["a"],
            task["t"],
            task["input_cov"],
            n_samples=1_000_000,
            rng=mc_rng_a,
        )
        p_b = mc_ground_truth(
            task["layers"],
            task["activation"],
            task["a"],
            task["t"],
            task["input_cov"],
            n_samples=1_000_000,
            rng=mc_rng_b,
        )

        n_layers = len(task["layers"])
        W0 = task["layers"][0][0]
        W_last = task["layers"][-1][0]
        arch = f"{n_layers}L {W0.shape[1]}-{W0.shape[0]}-...-{W_last.shape[0]}"
        ratio = p_b / p_a if p_a > 0 else float("inf")
        print(
            f"{arch:>20}  {task['activation']:>12}  {task['ground_truth']:>12.2e}  {p_a:>12.2e}  {p_b:>12.2e}  {ratio:>8.3f}"
        )

    print(f"\nTotal tasks: {len(tasks)}")
    print()


def test_trivial_estimator():
    """Run the evaluator on a trivial 'always return 0.5' candidate."""
    candidate_code = """\
def estimate(layers, activation, a, t, input_cov):
    return 0.5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(candidate_code)
        f.flush()
        result = evaluate(f.name)
        print("=== Evaluator result for trivial estimator (always 0.5) ===\n")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    cross_check_mc()
    test_trivial_estimator()
