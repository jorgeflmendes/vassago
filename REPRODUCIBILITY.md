# Reproducibility

## Environment

Use Python 3.12 and the committed `uv.lock` for the reference CPU checks:

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

Training requires a separate CUDA-enabled PyTorch installation. The executed run used
PyTorch 2.13.0+cu130, CUDA 13.0 and an NVIDIA GeForce RTX 5080. The portable lockfile
intentionally resolves CPU PyTorch and must not be described as CUDA-enabled.

## Immutable protocol

`vassago benchmark prepare` materializes the MovieLens 1M comparison. Its
`protocol.json` binds SHA-256 hashes for the prepared interactions, catalog, query
files and upstream sequence input. The seed-42 protocol hash is
`677cc91ee500d12f3478b678249e8721d99ea9d1d65c81bdfeebed61bbf25f3e`.

Each user's last interaction is test and the second-last interaction is validation.
Histories contain at most the latest 200 items. Every model ranks the same 3,706-item
catalog after the same seen-item exclusion. The evaluator rejects partial rankings,
missing users, duplicates, invalid IDs, seen recommendations and mismatched hashes.

Meta HSTU and SASRec use `scripts/hstu_fair_adapter.py` at upstream commit
`6035c3f9b2512791b0983e0adc71749b2a22e7dc`. The adapter changes the dataset source
and exports rankings; model logic remains upstream.

## Model selection and final run

`configs/experiment/ml1m_contextual.yaml` fixes seed 42 and the complete training
configuration. Selection reads temporal validation only and chose backbone epoch 100
and contextual epoch 25. The selected model is then reinitialized and trained on all
permitted pre-test events.

The final run produced:

| Measurement | Value |
|---|---:|
| Total parameters | 356,418 |
| Contextual parameters | 4,162 |
| Selection, retraining and inference | 865.3 s |
| Final inference | 5.221 s |
| Export throughput | 1,156.92 queries/s |
| Peak allocated CUDA memory | 2,224,654,848 bytes |
| Test queries | 6,040 |

These are end-to-end benchmark measurements rather than online serving latency. HSTU
and SASRec took 3,323.6 and 3,267.0 seconds respectively in their official-run
environment. Their timings are useful context but are not controlled service-latency
comparisons.

The final configuration uses `contextual_joint_epochs: 10` and
`contextual_joint_backbone_lr_scale: 0.1`. Temporal validation selected joint epoch 5,
then selected a single interpolated checkpoint containing 55% of the frozen contextual
weights and 45% of the jointly adapted weights. The test queries were read only for the
final reporting pass.

Manifests record the resolved configuration, protocol and file hashes, Python, PyTorch,
CUDA, hardware, parameter counts and durations. Weights use SafeTensors. Generated
data, rankings and checkpoints are deliberately ignored by Git.

## Statistical analysis

Metrics are averaged per query. Paired bootstrap comparisons use matching seed and
query IDs. The contextual ranker leads HSTU by 0.0011110 NDCG@10, with 95% CI
[-0.0046037, 0.0069937] and p=0.7191. It exceeds SASRec by 0.0254375, with 95% CI
[0.0199528, 0.0309669] and p<0.001.

This is a single-seed exploratory result. Earlier architecture variants were inspected
on the same seed-42 test, creating adaptive-test bias at the research-series level.
Further architecture choices must use a new untouched protocol or dataset. A
publication claim also requires the predeclared seeds 42 through 46 for every compared
model, with aggregate means, sample standard deviations and confidence intervals.
