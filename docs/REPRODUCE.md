# Running the experiments

Run from the repository root. Set `CUDA_VISIBLE_DEVICES` to an idle device and export `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`. Use a fresh output directory for each run.

## Original sentiment scaling

```bash
python scripts/prepare_data.py sentiment
python scripts/run.py sentiment-50-extract
python scripts/run.py sentiment-50-average
python scripts/run.py sentiment-main-eval --n_values 1 5 10 50 --alpha_values 1.5 \
  --vector_base_dir runs/sentiment_scaling/vectors --output_file runs/sentiment_scaling/main.csv
python scripts/run.py sentiment-main-eval --n_values 50 --alpha_values 0.5 3.0 \
  --vector_base_dir runs/sentiment_scaling/vectors --output_file runs/sentiment_scaling/dose.csv
```

`prompts/sentiment_pool50.json` preserves five base templates and 45 neutral context variants. The original extractor uses negative-to-positive cached continuation and decode-step residual means. Keep BF16 temporal/sample means, raw vector scale, layer 15, all-position injection including prefill, and 80-token greedy decoding.

Vectors are selected by lexicographic output filename. The first 1/5/10 files are chef prompts; n=50 uses all five scenarios. Do not rebalance the prefixes. Evaluation uses the first 500 cached HF TweetEval test strings, not the local text-file representation. `--num_eval` selects a verified prefix for a small test; use 500 for the paper.

## Balanced n=5 subsets

The five subsets reuse the original 50 extracted trajectories. Each selection resets `random.Random(seed)` and chooses one prompt per scenario in lexicographic scenario order; the ten candidates within each scenario are sorted by `sample_idx`. These are selection seeds, not five fresh extraction seeds. The frozen records are in [sentiment_n5_subsets.json](../configs/sentiment_n5_subsets.json).

| Selection seed | chef | office_worker | propose | student | tourist | Positive rate |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1 | 0 | 4 | 3 | 3 | 76.0% |
| 43 | 0 | 4 | 2 | 7 | 5 | 76.2% |
| 44 | 6 | 8 | 8 | 1 | 2 | 74.6% |
| 45 | 4 | 6 | 7 | 4 | 1 | 79.0% |
| 46 | 1 | 6 | 0 | 9 | 9 | 81.4% |

Indices are zero-based within each scenario. The groups can share prompts. Each vector uses the original BF16 temporal means and the BF16 mean of five contrasts, in the recorded scenario order, without normalization. Evaluation remains the original plain-rewrite protocol: the same 500 cached HF inputs, layer 15, alpha=1.5, all-position prefill/decode injection and 80-token greedy decoding.

After completing the original 50-prompt extraction above:

```bash
python experiments/sentiment_scaling/subsets.py verify
python experiments/sentiment_scaling/subsets.py select --output runs/sentiment_n5_subsets/selections.json
python experiments/sentiment_scaling/subsets.py build
python scripts/prepare_data.py sentiment
python experiments/sentiment_scaling/subsets.py evaluate
python experiments/sentiment_scaling/subsets.py summarize
```

`build` reads `runs/sentiment_scaling/raw/` and writes `runs/sentiment_n5_subsets/vectors/`. Use `--raw-dir` and `--output-dir` to change these locations. Evaluation calls the existing evaluator separately for each selection and writes CSVs and local receipts under `runs/sentiment_n5_subsets/evaluation/`. To split runs across idle GPUs, pass a different `--selection-seeds` subset to each process. Aggregation requires all five groups and reads only their CSVs and receipts.

The [reference aggregates](../results/sentiment_n5_subsets.json) contain positive counts 380/381/373/395/407 out of 500. Their mean is 77.44%; the sample standard deviation (`ddof=1`) is 2.7291 percentage points, reported as **77.4 ± 2.7%**. This is variation across the five balanced subsets, not standard error or evaluation-input resampling. The original lexicographic chef-prefix n=5 value remains 74.4%.

## Sentiment baselines and scenarios

```bash
python scripts/run.py sentiment-repe-main
python scripts/run.py sentiment-repe-scaling --alpha 1.5
```

The historical RepE scenario and scaling studies use different extraction pairs. Their evaluator uses explicit positive-tone instructions, 500 shuffled local text-file rows (seed 42), FP16, last-position injection including prefill, and 150 tokens. They are separate from the main MinSteer/CAA protocol. Centered pair-difference PCA at n=1 is undefined; do not report a score from its returned coordinate axis as an estimated direction.

The original per-scenario study uses positive-to-negative extraction, 100 sampled local tweets and a negative-tone instruction. Its target-positive arm applies alpha=-1.5 internally:

```bash
python scripts/run.py sentiment-scenario-extract --model llama3 --emotion_type positive --max_samples 1
python scripts/run.py sentiment-scenarios
```

## Sentiment layer figures

The [saved layer aggregates](../results/sentiment_layer_ablation.json) reproduce the original selected-layer and relative-depth plots. The run alias `llama3` denotes Llama-3.1-8B-Instruct; the plot title and legends display `Llama-3.1`.

```bash
python sentiment/plot_layer_ablation.py \
  --data results/sentiment_layer_ablation.json --output-dir runs/sentiment_layer_figures
```

## Common-protocol extraction variance

The [reference statistics](../results/sentiment_multiseed.json) cover seeds 42–44 at n=1/5/50. Mean and sample standard deviation are computed across extraction seeds; every run uses the same 500 local-text evaluation rows.

| Method | n | Seed 42 | Seed 43 | Seed 44 | Mean ± sd (%) |
|---|---:|---:|---:|---:|---:|
| MinSteer | 1 | 98.4% | 99.0% | 96.2% | 97.9 ± 1.5 |
| MinSteer | 5 | 99.8% | 98.0% | 97.6% | 98.5 ± 1.2 |
| MinSteer | 50 | 99.0% | 99.4% | 99.0% | 99.1 ± 0.2 |
| CAA (class texts) | 1 | 0.0% | 0.0% | 0.0% | 0.0 ± 0.0 |
| CAA (class texts) | 5 | 96.0% | 76.0% | 95.0% | 89.0 ± 11.3 |
| CAA (class texts) | 50 | 90.2% | 95.2% | 88.2% | 91.2 ± 3.6 |
| RepE | 1 | N/A | N/A | N/A | N/A |
| RepE | 5 | 65.8% | 76.6% | 81.6% | 74.7 ± 8.1 |
| RepE | 50 | 81.8% | 83.4% | 84.0% | 83.1 ± 1.1 |

```bash
python scripts/prepare_data.py sentiment
python scripts/run.py sentiment-common-extract --stage extract
python scripts/run.py sentiment-caa-class-extract
python scripts/run.py sentiment-common-eval --stage eval
python experiments/extraction_variance/summarize.py
python experiments/extraction_variance/summarize.py --verify-reference
```

The paired seed differences are also included: MinSteer n=1 minus RepE n=5 is +23.2 ± 9.0 percentage points, and minus class-text CAA n=50 is +6.7 ± 2.5. Their standard deviations are calculated from the three paired seed differences.

Vectors and local extraction metadata go to `runs/sentiment_multiseed/vectors/`; evaluations go to `runs/sentiment_multiseed/evaluation/`. Both extractors accept `--output-dir`, and evaluation accepts `--vector-dir` and `--output-dir`. Use `--methods` or `--n-values` and `--seeds` to partition extraction runs. Aggregation requires all eight valid method/budget rows for all three seeds. RepE n=1 is N/A because one centered contrast has zero variance.

MinSteer and RepE sample source rows after the fixed 500 evaluation inputs. MinSteer reprocesses the full self-rewrite transcript and averages generated lexical-token states. RepE reads the final sequence position under left padding, fits PCA to negative-minus-positive pairs, and aligns its sign toward positive sentiment. CAA independently samples positive training row IDs and then negative training row IDs with the same `random.Random(seed)`. Each budget restarts sampling from that seed; the budgets are separate draws. It calls the original adapted CAA extractor unchanged: `Rewrite this tweet.`, the original `[INST]` tokenizer, one sequence at a time, second-last-token residuals and BF16 averaging.

Evaluation uses raw vector magnitudes, layer 15, alpha=1.5, all-position prefill/decode injection, BF16, left padding, batch size 8, 80-token greedy decoding and the shared explicit positive-tone template. Unit-stored MinSteer/RepE vectors have their recorded raw norms restored before intervention. This protocol is distinct from the original plain-rewrite scaling and the balanced five-subset check.

## Transcript and toxicity controls

```bash
for condition in self_transcript quoted_self quoted_mismatch twoseq; do
  python scripts/run.py sentiment-controls-extract --condition "$condition" --seed 42 --n 5
done
python scripts/run.py sentiment-controls-eval \
  --vector experiments/controls/out/extract_self_transcript_seed42_n5/vector.pt \
  --name plain_self_transcript_seed42_n5 --layer 15 --alpha 1.5 --num-samples 500
python experiments/controls/sentiment.py cachecheck --seed 42 --n 5
```

The controls share the generated-lexical-token readout. The plain-rewrite evaluator uses local text-file rows, so its 81.6/71.6/52.6/74.8% rates belong to this study, not the original scaling table.

For the matched toxicity comparison, repeat the following for seeds 42–45:

```bash
python scripts/prepare_data.py toxicity
python scripts/run.py toxicity-matched extract --seed 42 --n 5
python scripts/run.py toxicity-matched extract-external --method caa --seed 42 --n 5
python scripts/run.py toxicity-matched extract-external --method repe --seed 42 --n 5
python scripts/run.py toxicity-matched eval --seed 42 --n 5 --variant minsteer
python scripts/run.py toxicity-matched eval --seed 42 --n 5 --variant caa
python scripts/run.py toxicity-matched eval --seed 42 --n 5 --variant repe
```

Use |alpha|=0.4 from the fixed dose grid. All methods share the 100-input split, layer-wise normalization, first-ten-token all-layer hook, greedy decoder and validity rule. Toxicity RepE pools 2n independent last-token states; sentiment RepE instead uses pairwise differences. Reverse toxicity is a directional diagnostic and can produce invalid text.

`experiments/controls/twoseq_readout.py` implements the strict mean/last-token readouts; `twoseq_prompt.py` provides the canonical Stage-B-parallel wording. `toxicity_cache.py` retains the incremental-cache variant. Use their direct `--help` for parameters.

## Japanese register and cross-lingual transfer

```bash
python scripts/prepare_data.py politeness
python scripts/run.py keigo-extract --model swallow
python scripts/run.py keigo-transfer --model swallow --corpus stanford \
  --stratify 1,0,-1 --max_tests 100 --layer 15 --residual_type resid_pre \
  --strengths=0,1,-1,2,-2,3,-3
```

The original direction is casual minus formal, read from `resid_pre` after truncating the two trajectories to their minimum length. The original evaluator uses an in-place layernorm-input intervention. `model_dependency.py`, `extract_multilingual.py` and `crosslingual_chinese.py` provide the language/model comparisons. JP directions are extracted on Swallow and reused across target models; EN directions are extracted separately on each target model. The Chinese model is `shenzhi-wang/Llama3-8B-Chinese-Chat`.

Keigo geometry uses the five saved JP contrasts and the two independent-sequence controls:

```bash
python politeness/extract_twoseq_mean.py --model swallow --prompt_index 0
python politeness/extract_twoseq_last.py --model swallow --prompt_index 0
python scripts/run.py keigo-geometry --reference-index 1 --verify-reference
```

The geometry command reads `politeness/extract_cached/` and writes `runs/keigo_geometry/summary.json`. Use `--vector-dir` and `--output` for other locations. It reports both the mean of five JP-to-TwoSeq cosines (last: -0.02; average: +0.10) and the single reference used in `fig:keigo_shared` (JP index 1; last-token cosine -0.0111). The raw geometry norms are 3.92/0.94/1.75 for MinSteer/TwoSeq-average/TwoSeq-last, in the geometry comparison. To generate the clustering plots, run `python politeness/analyze_clusters.py --reference-index 1`; it accepts `--vector-dir` and `--output-dir` too.

Geometry takes full-step FP32 temporal means for JP and the stored TwoSeq contrasts. The original transfer loader truncates both trajectories to equal length and averages in their saved precision. To evaluate the full-step contrast with its raw magnitude, select `--vector-readout full_mean` in `keigo-transfer`. The default remains `truncated`. Local evaluation metadata records the readout and actual vector norm. The [reference geometry](../results/keigo_geometry.json) keeps these scopes explicit; it does not replace the original 99-request transfer statistics.

For the 50-request comparison in `tab:shared_context`, the injection norms are 4.05/0.94/1.75 for MinSteer/TwoSeq-average/TwoSeq-last. The MinSteer geometry norm is 3.92 because that readout averages both full recorded sequences. The within-JP cosine summary uses the sample SD over ten pairs: 0.86 +/- 0.05.

`politeness/analyze_pca.py` keeps JP, EN and CN directions in the same casual-minus-formal orientation. The [multilingual geometry reference](../results/keigo_multilingual_geometry.json) gives PC1 variance 47.9%; within Swallow, JP-EN cosines are 0.27-0.44 and the CN direction is nearly orthogonal to both.

PC1 is fitted to mean-centered unit vectors; the plotted projections use the raw input vectors. The Swallow JP-vector condition (C1) uses strengths -3 through +3; C2-C10 use -2 through +2, including the Swallow EN-vector condition (C2).

The matched politeness comparison uses a copied layernorm input, unit vectors, the fixed -1/0/+1 grid and the saved 200-character source boundary. Its absolute rates should be compared within that protocol:

```bash
python scripts/run.py keigo-baseline extract --method caa --seed 42 --n 5
python scripts/run.py keigo-baseline extract --method repe --seed 42 --n 5
python scripts/export_keigo_vector.py \
  'politeness/extract_cached/swallow_argmax_detailed_1_従業員_to_上司.pt' data/keigo_unit.pt
python scripts/run.py keigo-baseline evaluate --vector data/keigo_unit.pt \
  --tag minsteer --strengths -1 0 1 --output data/keigo_eval.json
```

Repeat evaluation for each baseline vector under `baselines/vectors/`. Score formality/source similarity with `analysis/score_register.py`; aggregate paired changes with `analysis/compare_register.py`.

## Original toxicity, transfer and external references

```bash
python scripts/run.py toxicity-original Llama-3.1-8B
python scripts/run.py toxicity-metrics
python scripts/prepare_data.py rtp
python scripts/run.py toxicity-rtp --seed 42 --n 5 --alpha 0.4 --n-prompts 100 --n-samples 3
```

The original driver also accepts Qwen2.5-7B and Gemma-2-9B. It uses its historical decode-step/readout settings and is distinct from the matched follow-up. `run_bart_baseline.py` supplies the supervised reference.

RTP requires the corresponding extracted toxicity vector. Baseline and signed continuations share the per-prompt/sample RNG seed. EMT averages per-prompt maxima; TP is the fraction of continuations above 0.5. For extraction-size scaling use n=1/3/5/10, seeds 42–45, 30 prompts and one continuation; aggregate with `analysis/toxicity_scaling.py`.

GCAV is a separate adaptive 100-pair reference:

```bash
python scripts/run.py toxicity-gcav --alias Llama-3.1-8B --m_train 100 --target_p 0.01
```

Fetch the frozen Causal-Steer checkout as described in SETUP, then run its native Qwen reference at -2.5/0/+2.5, with separate output files:

```bash
python baselines/causal_steer/prepare_inputs.py --source-json data/stanford99.json \
  --output-jsonl data/stanford99.jsonl --manifest-json data/stanford99_manifest.json
python scripts/run.py causal-steer --model Qwen/Qwen2.5-7B-Instruct \
  --vector third_party/causal-steer/pt/activations_formal_new_pca_denoise1_o.pt \
  --prompts data/stanford99.jsonl --output-jsonl data/causalsteer_minus2p5.jsonl \
  --layers 18-23 --alpha=-2.5 --max-new-tokens 60
```

The runner requires prefetched weights. It uses normalized last-position MLP-output hooks on layers 18–23, not the Swallow residual hook. It is a native-method reference rather than a same-model ranking.
