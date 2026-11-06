# MinSteer

Code for **MinSteer: Minimal-Pair Steering via Two-Stage Cached Continuation** (NeurIPS 2026).

## Installation

Use Linux, Python 3.10 and a CUDA GPU with enough memory for an 8B model. Obtain access to the required models on Hugging Face.

```bash
git clone https://github.com/Liumx2020/MinSteer.git
cd MinSteer
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Run an example

Select an idle GPU, then choose a task. Each command prepares inputs, extracts a direction, builds the vector and evaluates it.

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/smoke_test.py sentiment --num-eval 10
python scripts/smoke_test.py politeness --num-eval 3
python scripts/smoke_test.py toxicity --num-eval 3
```

Outputs are saved under `runs/smoke_TASK/`, including `summary.json`. Use `--output-dir runs/new_name` for another run; existing output directories are rejected. Models download on first use.

The commands have been tested on an A100 40GB with PyTorch 2.10.0+cu128 and Transformers 4.57.6. These small examples check the pipeline; they do not reproduce full-dataset paper scores. Sentiment uses the original plain-rewrite protocol, while politeness and toxicity use their matched comparison protocols.

## Paper experiments

See [setup](docs/SETUP.md), [running the experiments](docs/REPRODUCE.md), and the [paper-to-code map](docs/COVERAGE.md).

For original-protocol sentiment scaling:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/prepare_data.py sentiment
python scripts/run.py sentiment-50-extract
python scripts/run.py sentiment-50-average
python scripts/run.py sentiment-main-eval \
  --n_values 1 5 10 50 --alpha_values 1.5 \
  --vector_base_dir runs/sentiment_scaling/vectors \
  --output_file runs/sentiment_scaling/results.csv
```

The [ordered prompt pool](prompts/sentiment_pool50.json) contains five original templates and 45 context variants. The reference positive rates at alpha=1.5 are 84.6/74.4/74.8/77.2% for n=1/5/10/50. Historical RepE uses a separate evaluation protocol; centered PCA at n=1 is reported as N/A.

The additional **balanced n=5** check uses selection seeds 42–46, one prompt per scenario: **77.4 ± 2.7%** (mean ± sample standard deviation across five subsets). The [subset records](configs/sentiment_n5_subsets.json), [aggregate results](results/sentiment_n5_subsets.json) and [running instructions](docs/REPRODUCE.md#balanced-n5-subsets) are included. This is separate from the original 74.4% chef-prefix n=5 result.

After the 50-prompt extraction above, run all five subsets with:

```bash
python experiments/sentiment_scaling/subsets.py verify
python experiments/sentiment_scaling/subsets.py build
python experiments/sentiment_scaling/subsets.py evaluate
python experiments/sentiment_scaling/subsets.py summarize
```

Repeated extraction at **n=1/5/50** uses seeds **42–44** and the original class-text CAA adaptation. The [current multi-seed statistics](results/sentiment_multiseed.json) and [commands](docs/REPRODUCE.md#common-protocol-extraction-variance) are included. To recompute the bundled means and sample standard deviations without a GPU:

```bash
python experiments/extraction_variance/summarize.py --verify-reference
```

| Directory | Contents |
|---|---|
| `sentiment/`, `politeness/`, `toxicity/` | Original task implementations |
| `experiments/` | Scaling, extraction variance and transcript/readout controls |
| `baselines/` | Matched politeness, GCAV and native Causal-Steer adapters |
| `analysis/` | Register scoring and toxicity scaling analysis |
| `prompts/`, `configs/` | Authored templates, fixed input order and model versions |
| `results/` | Paper reference aggregates |
| `tests/` | CPU checks for prompts and steering logic |

Individual human annotations, raw generations, residuals, model weights and datasets are not distributed. Experimental Japanese and Chinese prompt text is preserved. The project code is available under the [MIT License](LICENSE). See [third-party attribution](THIRD_PARTY.md) for upstream terms.
