# OpenEvolve for MoSSAIC

## Introduction

This is my contribution to the MoSSAIC project of AISC 2026. MoSSAIC deals with the problem of substrate independence for interpretability techniques. In this project, we use OpenEvolve to select a faithful interpretation to an architecture in a reproducible way.

Here's an informal description from the project's google doc:

---

Existing interpretability techniques are pretty substrate-dependent (maybe not all, but a lot of them are). This sucks because 1) what if there is an architecture change and our interpretability techniques are moot? 2) what if an adversarial superintelligence purposefully obfuscates itself?

This is because interpretability techniques are divinely (or arguably diabolically) inspired, and not created as a result of any methodical process.

There are two ways we can help:

- Make a more general, substrate-independent technique,
- Make a method for creating interpretability techniques that we could reuse when we run into a new architecture.

One way to explore the latter angle of attack is an algorithm by Google called AlphaEvolve. The important thing about AlphaEvolve is that it is a program for optimizing programs. Its potential remains underexplored IMHO, as such setup is very general in nature.

The reason this is relevant to creating interpretability techniques in a methodical, substrate-independent way is that we can treat them as code to be optimized. If we have some way to measure if this-and-this explanation is both faithful and insightful (and most importantly, how faithful and insightful), we can try to optimize them.

The choice of an evaluator is a big challenge. But one option is to borrow some traits of ARC’s setup described in “outperforming sampling”. That is, if we manage to create code that, for some input distribution, predicts the probability of a specific output, WITHOUT resorting to monte-carlo-like sampling tricks, at least as well as sampling would, then we have an interpretation that’s both faithful and insightful (well, somewhat…)

E. g.: we take a transformer WITHOUT the embedding layer or the unembedding layer (i.e. we provide the input embeddings directly, and estimate the result on the output vector, without putting them through an LM head to then use argmax or any other generation strategies).

(I have a hard time thinking about this, if it makes more sense like this than trying to fuck around with actual logits.)

Then we pose a question like “what is the probability of output in this-and-this region of output space?”. Then we sample N vectors with dimensionality D and run them through the transformer, and obtain a ratio of outputs that are indeed within the specified region.

Since we know the value N and hence the precision obtained by the sampling method, we can judge when the sample-free method’s output is at least not worse than sampling. Importantly, we would also know how bad a method is, i.e. there would be directionality to the signals of sampling.

## Results

Slides from the AISC 2026 demo day: [AlphaEvolving sampling](https://wadimiusz.me/mossaic_slides.pdf).

The score is the mean log-probability that the evolved `build_dist` assigns to held-out residual-stream samples, measured against a diagonal-Gaussian baseline. In each setting, one program has to work for all models.

| Setting | Models | Best result | Strategy the search found |
| --- | --- | --- | --- |
| Modular addition | 39 (13 primes × 3 seeds) | +385 nats | Forward pass × per-class low-rank Gaussian × Fourier-aware variance |
| Five task families, task label hidden: modular addition, greater-than, Dyck, DFA, induction | 90 | +24 nats on average | Generic forward pass on random inputs, k-means, diagonal-Gaussian mixture |

Lessons:

- On random MLPs with Gaussian input, the search finds moment propagation, a generic method.
- On a sentiment transformer, it finds a mixture of low-rank Gaussians plus FFN-SVD directions. Whether that counts as understanding the model is an open question.
- Generic methods fail where the geometry is unusual: modular addition needs Fourier features on a torus, and a Gaussian mixture in 64 dimensions models a torus poorly.
- Interpretability may not scale neatly with predictive power. The main open question I took from the project is how the two are connected.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) for dependency management
- Python 3.13 (managed automatically by uv)
- Linux with `perf` and `taskset` available (`linux-tools-common` / `linux-tools-generic`)
- An Intel hybrid CPU with E-cores (atom cores) — required for `fp_flops_retired.all` hardware counter

### Install dependencies

```bash
uv sync
```

If you plan to contribute, also install the dev dependencies and register the pre-commit hooks (runs ruff on every commit):

```bash
uv sync --dev && uv run pre-commit install
```

### Configure API keys

Create a `.env` file in the project root (see `.env.example`):

```
OPENAI_API_KEY=...  # OpenRouter key (sk-or-...) or OpenAI key (sk-...)
```

To switch between OpenRouter and direct OpenAI, change `api_base` in `config.yaml` — no `.env` changes needed.

### Hardware: enable perf counters

FLOP counting uses hardware performance counters, which require `perf_event_paranoid ≤ 2`:

```bash
# Temporary (until next reboot):
sudo sysctl kernel.perf_event_paranoid=2

# Permanent:
echo 'kernel.perf_event_paranoid=2' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

> **Note:** The evaluator pins execution to CPU core 16 (`E_CORE = "16"` in `evaluator.py`). Adjust this constant if your E-cores are on different indices — FLOP counting only works on atom/E-cores, not P-cores.

---

## Running

### Sanity check

Runs DistGraph executor tests and scores a trivial estimator across 20 architectures:

```bash
uv run python smoke_test.py
```

### Run evolution

```bash
uv run python main.py
```

Output is written to `openevolve_output/`.

## Note

This has been written with huge help from claude code. So:

- kudos to Claude Opus 4.6 (if it's a moral agent — and I posit we don't know otherwise, — it should basically be treated as a co-author),
- but also, watch out for any silliness I could have missed. I test things myself manually, of course, but with vide-coding it's always good to be paranoid.

Thanks also to Claude, who helped tidy up this README in October 2026 and pulled the Results section out of my slides.
