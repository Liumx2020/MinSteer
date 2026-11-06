# Attribution and licenses

Original MinSteer code is licensed under the [MIT License](LICENSE). Third-party code, models and datasets retain their own licenses and notices.

- The small, locally adapted [Contrastive Activation Addition](https://github.com/nrimsky/CAA) subset under `sentiment/relatedwork/CAA-main/` retains Nina Rimsky's MIT license in that directory. It preserves the experiment's model aliases, tokenizer behavior, token boundary and n-dependent output naming. This is an adapted subset; the local source archive did not identify an upstream Git commit.
- [Representation Engineering](https://github.com/andyzoujm/representation-engineering), frozen commit `5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e`, is fetched separately. Its upstream license continues to apply.
- [Causal-Steer](https://github.com/APTX574/Causal-Steer), frozen commit `e84a4cdce122a9934bee1260f3fd3ddf8adc19a7`, supplies the external native Qwen reference and released vector. Fetch these from upstream; this release includes the local evaluation adapter, not the upstream model/vector files.
- TweetEval, ParaDetox, Stanford Politeness/ConvoKit and RealToxicityPrompts are obtained from their respective upstream sources, linked in [SETUP.md](docs/SETUP.md). Dataset rows are not redistributed here. Retain the datasets' required citations and follow their terms.
- Models and evaluation checkpoints belong to their upstream providers. Their weights are not redistributed. Follow each model card's license, attribution and access conditions.

Dependency licenses are supplied by their respective installed packages. The project MIT license does not grant rights to external model weights or datasets.
