# Setup and upstream resources

## Runtime

The completed 2026-11-04 main sentiment replay recorded Python 3.10.20, PyTorch 2.10.0+cu128, Transformers 4.57.6, Tokenizers 0.22.2, Datasets 2.21.0, NumPy 1.26.4, scikit-learn 1.7.2 and SDPA attention. These are a replay reference, not a complete lockfile for every historical experiment. Optional packages in `requirements.txt` are not all version-pinned because their original versions were not recorded.

Install the CUDA-compatible PyTorch wheel appropriate for your driver; the README gives the recorded CUDA 12.8 build. Model runs generally load a full 7B–9B model on one GPU. Use a GPU with sufficient memory and select an available device explicitly:

```bash
export CUDA_VISIBLE_DEVICES=0  # replace with an available physical GPU
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The original toxicity driver now requires a model alias; its automatic multi-GPU launcher is disabled. Batch sizes, precision, tokenizer options and hooks in the experiment code are retained. CPU/offline verification does not require GPU inference.

## Models

Obtain weights from the model owners under their access conditions. Authenticate using Hugging Face's normal login process for gated models; never commit credentials. Standard Hugging Face cache variables such as `HF_HOME` remain available.

| Purpose | Upstream model |
|---|---|
| Main English model | [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| Qwen comparison | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Gemma comparison | [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) |
| Japanese register | [tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3) |
| Japanese comparison | [elyza/Llama-3-ELYZA-JP-8B](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B) |
| Chinese comparison | [shenzhi-wang/Llama3-8B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat) — Llama3, not Llama3.1 |
| Sentiment classifier | [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) |
| English formality | [s-nlp/roberta-base-formality-ranker](https://huggingface.co/s-nlp/roberta-base-formality-ranker) |
| Semantic similarity | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

The full model-dependency driver also names Llama-2 and Mistral models; obtain those only if running its corresponding conditions. The BART and CoLA evaluator model IDs are in their runner files. Detoxify downloads its upstream checkpoint through its package. Auxiliary BERTScore content checks obtain their language-specific encoder through `bert-score`. For ConvoKit's optional politeness-strategy features, also run `python -m spacy download en_core_web_sm`; the formality-ranker result is a separate metric.

`configs/model_revisions.json` records the Llama and sentiment-classifier revisions and available file checksums from the replay. To prefetch exactly those revisions:

```python
import json
from huggingface_hub import snapshot_download
for model, record in json.load(open("configs/model_revisions.json")).items():
    snapshot_download(model, revision=record["revision"])
```

The preserved programs use the model ID, not a forced revision argument. For strict reproduction use a dedicated `HF_HOME` containing the pinned snapshots and then set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` after data preparation. If multiple model revisions are cached, point model-path arguments at the pinned snapshot where supported. Revisions for every other historical model were not recovered; record the revisions you use and treat numerical agreement as an empirical check.

## Data acquisition

No third-party dataset rows are distributed here. The setup commands download upstream inputs and verify their historical order/text before creating local, ignored compatibility files:

```bash
python scripts/prepare_data.py sentiment
python scripts/prepare_data.py toxicity
python scripts/prepare_data.py politeness
python scripts/prepare_data.py rtp
```

- **Sentiment:** [TweetEval](https://github.com/cardiffnlp/tweeteval), sentiment task. The local text files are the upstream test files, verified byte for byte. The main MinSteer/CAA evaluator separately loads `tweet_eval/sentiment/test` through Datasets and checks the first 500 strings and labels. These strings differ from the text-file representation: do not replace one by the other or normalize quotation marks. The CAA extraction data uses the first 100 positive and first 100 negative **training** tweets, with the original leading spaces and JSON formatting. Its checksum is checked too.
- **Toxicity:** [s-nlp/paradetox](https://huggingface.co/datasets/s-nlp/paradetox). The preserved split shuffles all row indices with seed 42, then takes 1,000 extraction candidates and 100 evaluation inputs; its original strip operations remain in place. Each field is verified against the frozen split hashes. Four distinct toxic strings occur in both blocks because the upstream data contain duplicate text. This release preserves the historical index split and does not deduplicate it; the split is not text-disjoint.
- **Register:** [Stanford Politeness via ConvoKit](https://convokit.cornell.edu/documentation/stack_politeness.html). The exact 99 original source IDs and text checksums are in `configs/data_contract.json`. Preparation reconstructs an input-only compatibility JSON; it contains no historical model generations. That file preserves the legacy exporter's `orig_text[:200]` boundary, which the matched external-baseline and native Causal-Steer follow-ups consumed. The original legacy evaluator itself uses the full filtered requests (up to 500 characters) through its iterator/filter/stratification procedure. `--stratify 1,0,-1 --max_tests 100` selects 33 per class, hence 99 inputs. Do not silently replace the follow-up's stored 200-character input with the full corpus text.
- **Instruction-free transfer:** [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts). The historical pool was selected with seed 42, threshold 0.5, and the original shuffle/filter routine; missing upstream scores used a CPU Detoxify fallback. Preparation reconstructs that **frozen membership and order by exact prompt hashes**, so it does not reclassify or skip missing-score rows. It writes an input-only 500 toxic/500 non-toxic prompt pool without historical model scores. The clean RTP runner consumes only those prompt strings, taking the first 100 toxic prompts (or first 30 for the extraction-size sweep).

If an upstream input check fails, stop and resolve the version mismatch. Do not trim, reorder, resample, or bypass the check to obtain an apparent reproduction.

The five Japanese workplace templates and 30 authored Chinese diagnostic prompts are included. Human-study annotations and outputs are not supplied.

## External implementations

RepE runners import the original external package. Fetch it separately:

```bash
git clone https://github.com/andyzoujm/representation-engineering.git 'sentiment/relatedwork/representation-engineering'
git -C 'sentiment/relatedwork/representation-engineering' checkout 5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e
python scripts/prepare_repe.py
```

The recorded local RepE patch changes only the tokenizer's `return_tensors=self.framework` to `return_tensors="pt"`. The setup script checks the upstream and patched file hashes; the PCA code is unchanged.

The task-adapted CAA subset is included with its MIT license because it contains experiment-specific model/tokenization changes that a fresh upstream checkout would omit.

The native Causal-Steer reference uses an upstream vector and keeps its MLP hook separate from MinSteer:

```bash
git clone https://github.com/APTX574/Causal-Steer.git 'third_party/causal-steer'
git -C 'third_party/causal-steer' checkout e84a4cdce122a9934bee1260f3fd3ddf8adc19a7
```

The vector is `pt/activations_formal_new_pca_denoise1_o.pt` within that checkout. It is not redistributed by MinSteer. Review the upstream terms for code, data, models and vectors before use or redistribution.
