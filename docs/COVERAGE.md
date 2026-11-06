# Paper-to-code map

The paths below identify the maintained implementations by paper label. Reference aggregates are in `results/`. Small end-to-end examples exercise the real extraction and evaluation code; the complete paper experiment matrix has not been rerun for this release.

| Paper content | Code |
|---|---|
| `tab:sentiment_scaling`, `tab:repe_scaling_detail` | `experiments/sentiment_scaling/`, original `sentiment/` extractors, CAA and RepE |
| Balanced five-subset n=5 check, `app:repe_scaling` | `experiments/sentiment_scaling/subsets.py`, `configs/sentiment_n5_subsets.json`, `results/sentiment_n5_subsets.json` |
| Five-scenario robustness, `tab:n1_scenario_robustness` | `sentiment/extract_cached.py`, `evaluate_scenarios.py` |
| Sentiment layer/label analyses and vector geometry | `sentiment/evaluate_steering.py`, `analyze_vector_similarity.py` |
| `tab:sentiment_seed_variance` | `experiments/extraction_variance/`, `results/sentiment_multiseed.json` |
| `tab:mechanism_controls`, `tab:mechanism_seeds`, cache/replay agreement | `experiments/controls/sentiment.py`, `sentiment_evaluate.py`, `sentiment_aggregate.py` |
| Original keigo dose response and `tab:expd` | `politeness/extract_cached.py`, `evaluate_transfer.py`, `model_dependency.py` |
| Keigo geometry in `tab:shared_context`, `app:keigo`, `fig:keigo_shared` | `politeness/geometry.py`, `analyze_clusters.py`, `results/keigo_geometry.json` |
| Cross-language extraction and keigo geometry | `politeness/extract_multilingual.py`, `crosslingual_chinese.py`, `analyze_*.py` |
| Original independent-sequence controls | `politeness/extract_twoseq_*.py`, `toxicity/extract_twoseq_last.py` |
| `tab:paradetox_metrics`, original toxicity diagnostics | `toxicity/run_all_toxicity_eval.py`, `eval_paradetox_metrics.py`, `run_bart_baseline.py` |
| Matched-readout toxicity controls | `experiments/controls/toxicity.py`, `toxicity_cache.py`, `twoseq_readout.py`, `twoseq_prompt.py` |
| `tab:rtp_transfer`, `tab:tox_n_sweep` | `experiments/controls/rtp.py`, `analysis/toxicity_scaling.py` |
| `tab:tox_external` and GCAV reference | `experiments/controls/toxicity.py extract-external/eval`, `baselines/gcav.py` |
| `tab:polite_external` | `baselines/politeness.py`, `scripts/export_keigo_vector.py`, `analysis/score_register.py`, `compare_register.py` |
| `tab:causalsteer` | `baselines/causal_steer/` and separately obtained upstream vector |
| Full ordered sentiment prompt pool | `prompts/sentiment_pool50.json` |

## Protocol boundaries

- Original sentiment MinSteer/CAA uses cached HF strings, plain rewrite, BF16, layer 15, raw vectors, all-position prefill/decode injection and 80 tokens. Historical RepE uses local shuffled inputs, positive-tone instructions, FP16, a last-position hook and 150 tokens. The repeated-extraction study uses training class-text CAA and the common positive-tone protocol.
- The original five-scenario vectors use negative minus positive states; target-positive evaluation applies alpha=-1.5.
- Transcript controls and common-protocol extraction read generated lexical tokens rather than the original decode-step states.
- Sentiment/toxicity use `resid_post`; keigo uses `resid_pre`. The original in-place keigo hook and matched copied-input hook are distinct.
- Keigo geometry reports five individual JP comparison cosines and their mean, plus a separate single-JP reference (index 1) for the figure. Its full-step FP32 readout and raw norms are explicit. The original transfer default uses equal-length native-precision means; `--vector-readout full_mean` selects the geometry contrast for evaluation.
- The matched register follow-ups consume the historical 200-character source field. The original evaluator uses full filtered requests. The original ParaDetox index split contains four repeated toxic strings across extraction and evaluation blocks; it is preserved without deduplication.
- For the original ParaDetox rewrite diagnostic, STA is the non-toxic classification rate (`s-nlp/roberta_toxicity_classifier`), SIM is mean cosine similarity (`sentence-transformers/all-MiniLM-L6-v2`), and FL is the acceptability rate (`textattack/roberta-base-CoLA`). J multiplies these three corpus-level scores. This differs from the [original ParaDetox evaluator](https://github.com/s-nlp/paradetox/blob/main/evaluation_detox/metric_tools/joint_metrics.py), which averages sentence-level products and uses a different SIM encoder. The paper uses the local diagnostic definition for `tab:paradetox_metrics` and `tab:selfdebias_diag`.

## Limits

Human annotations, linking keys, study outputs and human-study analysis workflows are not included. The human-evaluation statistics therefore cannot be recomputed from this repository alone. Raw model generations, residual tensors, saved historical vectors, weights and dataset rows are also excluded.

The source does not provide a complete reconstruction of every paper figure, historical run selection or model revision. Full-dataset scores must be obtained by running the corresponding protocol; small examples are only pipeline checks. The exact original environment was not fully recorded, so the listed runtime is a working reference rather than a claim of bitwise reproducibility for every experiment.
