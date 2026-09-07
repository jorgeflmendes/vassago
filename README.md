# VASSAGO

<p align="center">
  <img src="assets/vassago-logo.png" alt="VASSAGO logo" width="760">
</p>

<p align="center">
  <strong>Variable-gap Attention-based Sequential Scoring, Adaptive Gating, and Ordering</strong><br>
  Temporal contextual-evidence ranking for sequential recommendation
</p>

<p align="center">
  <a href="#architecture">Architecture</a> ·
  <a href="#results">Benchmark</a> ·
  <a href="#reproduction">Reproduction</a> ·
  <a href="#scientific-scope-and-limitations">Limitations</a>
</p>

VASSAGO (Variable-gap Attention-based Sequential Scoring, Adaptive Gating, and
Ordering) is a reproducible research implementation of a compact temporal
contextual-evidence ranker for sequential recommendation.

> **Research status.** The current single-seed benchmark gives VASSAGO the highest
> observed NDCG@10 point estimate. The paired comparison with HSTU is statistically
> tied.

| | |
|---|---|
| **Task** | Sequential recommendation and ranking |
| **Dataset** | MovieLens 1M; 6,040 test queries over a 3,706-item catalog |
| **Protocol** | Fixed full-catalog evaluation with identical seen-item filtering |
| **Baselines** | Meta HSTU and Meta SASRec |
| **Model size** | 356,418 parameters; 4,162 contextual parameters |
| **Reference hardware** | NVIDIA RTX 5080; CUDA 13.0; bfloat16 inference |

## Architecture

The final model consists of:

* a two-layer causal SASRec backbone with tied item scoring;
* learned logarithmic event-gap embeddings;
* per-head query-time attention biases;
* exposure-safe causal negative sampling;
* a 4,162-parameter candidate-conditioned evidence module operating on recent states;
* validation-selected interpolation between frozen and jointly adapted checkpoints; and
* full-catalog evaluation with identical seen-item exclusion across models.

The evidence correction is candidate-specific and bounded by a learned gate.

Model selection is performed using each user's second-last interaction, while the final interaction is reserved for test reporting. Training terminates with an error if non-finite validation or test scores are encountered, preventing invalid rankings from being exported.

## Cold-start routing

The inference pipeline supports two cold-start settings.

When an ordered list of up to ten favorite movies is available, the model uses the contextual checkpoint together with a small set-centroid residual.

When favorite items are unavailable, a metadata-based expert uses explicit genres, directors, actors, languages, and year ranges, with a ridge projection into the learned collaborative item space. Because coarse profile attributes can produce large groups of tied candidates, a popularity prior is selected on the validation split to resolve these ties.

Once at least one positive behavioral interaction is available, both onboarding components are bypassed and ranking is performed exclusively by the final contextual model. Consequently, warm-user rankings are invariant to changes in the profile fields.

MovieLens 1M does not contain explicit onboarding profiles. To evaluate the profile-based route, an exploratory proxy was constructed using the three most frequent genres in each user's pre-query history while withholding all historical item IDs from the cold-start ranker. The popularity weight was selected using the second-last interaction.

On the final-interaction test split, this hybrid approach obtains NDCG@10 of `0.02104` and Recall@10 of `0.04172`, compared with `0.01453` and `0.02947`, respectively, for the training-popularity fallback. The paired NDCG@10 difference is `+0.00651`, with a bootstrap 95% CI of `[0.00366, 0.00947]` and `p=0.0006`.

This experiment provides evidence that the routing mechanism can improve over a popularity-only fallback under the proxy setting. It should not be interpreted as a substitute for evaluation using independently collected user profiles.

A second onboarding proxy represents the most recent pre-query items as ordered favorite selections. Validation selected ten favorites and a set-centroid residual weight of `0.025`.

Under this setting, the validation-selected onboarding adapter obtains NDCG@10 of `0.16979` and Recall@10 of `0.29089`. The corresponding values using the complete warm history are `0.19682` and `0.33692`.

The proxy therefore recovers a substantial fraction of the warm-history performance without modifying or retraining the warm model. This estimate is likely optimistic because real users select favorites explicitly, whereas the proxy reconstructs them from previously observed interactions.

For the matched cold-start comparison, all sequential models receive the same ten selected items, use the same complete historical seen-item filter, and are evaluated against the same final target.

| Model                |   Recall@10 |     NDCG@10 |   Recall@50 |     NDCG@50 |  Recall@200 |    NDCG@200 |
| -------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| VASSAGO cold adapter | **0.29089** | **0.16979** | **0.53974** | **0.22495** | **0.75033** | **0.25679** |
| Meta HSTU cold-10    |     0.07798 |     0.03899 |     0.25811 |     0.07774 |     0.54437 |     0.12051 |
| Meta SASRec cold-10  |     0.27632 |     0.15830 |     0.52583 |     0.21320 |     0.73245 |     0.24454 |

VASSAGO records the highest point estimate for all reported metrics in this cold-start comparison. Relative to SASRec, the paired NDCG@10 difference is `+0.01148`, with a bootstrap 95% CI of `[0.00605, 0.01716]` and `p<0.001`.

## Results

The main experiment uses MovieLens 1M with seed 42, 6,040 users, 3,706 catalog items, and top-200 full-catalog rankings.

All models use the same protocol, identified by the hash:

`677cc91ee500d12f3478b678249e8721d99ea9d1d65c81bdfeebed61bbf25f3e`

| Model               |   Recall@10 |     NDCG@10 |   Recall@50 |     NDCG@50 |  Recall@200 |    NDCG@200 |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| Contextual evidence | **0.33692** | **0.19682** | **0.58526** | **0.25198** | **0.78377** | **0.28218** |
| Meta HSTU           |     0.33328 |     0.19571 | **0.58526** |     0.25159 |     0.77152 |     0.27993 |
| Meta SASRec         |     0.29139 |     0.17138 |     0.55248 |     0.22902 |     0.74851 |     0.25889 |

Across the complete set of 30 reported metric fields—Recall, NDCG, HitRate, MRR and
MAP at cutoffs 10, 50 and 200, both overall and for ratings `>=4`—the contextual
model has the highest point estimate in 28 and ties HSTU on the remaining two fields,
Recall@50 and HitRate@50. The table shows the six primary Recall/NDCG fields.

For NDCG@10, the paired difference relative to HSTU is `+0.00111`, with a bootstrap 95% CI of `[-0.00460, 0.00699]` and `p=0.719`. The available evidence therefore does not distinguish the two models statistically on this metric.

Relative to SASRec, the NDCG@10 difference is `+0.02544`, with a 95% CI of `[0.01995, 0.03097]` and `p<0.001`.

The final run used an NVIDIA RTX 5080, PyTorch 2.13.0+cu130, and CUDA 13.0. Model selection, final training, and inference required 865.3 seconds. Peak allocated GPU memory was 2,224,654,848 bytes, and export throughput was 1,156.92 queries per second.

The complete model contains 356,418 parameters.

A separate inference-only experiment compares serving memory under matched conditions: the same RTX 5080, full-catalog scoring, batch size 64, and bfloat16 for all models. CUDA peak-memory statistics are reset after model initialization. Cold-start measurements use an effective sequence length of ten items, while warm-start measurements use the full 200-item sequence.

| Protocol        |      VASSAGO | Meta HSTU | Meta SASRec | Lowest peak |
| --------------- | -----------: | --------: | ----------: | ----------- |
| Warm, 200 items | **51.46 MB** |  95.45 MB |    51.76 MB | VASSAGO     |
| Cold, 10 items  | **49.79 MB** |  88.52 MB |    53.43 MB | VASSAGO     |

These measurements report allocated CUDA memory during inference and should not be interpreted as total training-memory requirements.

Dynamic serving tensors avoid unnecessary padding, while bounded query-block attention limits the warm-start memory footprint. Under this measurement protocol, VASSAGO has a slightly lower peak allocation than SASRec and a substantially lower peak than HSTU.

The complete measurement record is stored in:

`reports/runs/contextual-ml1m-final/seed42/inference-memory-comparison.json`

### Validation-selected checkpoint interpolation

The final training phase jointly fine-tunes the contextual module and backbone, using a reduced learning rate for the backbone.

Validation selected joint epoch 5. A subsequent validation step selected a single interpolated checkpoint containing 55% of the frozen contextual weights and 45% of the jointly adapted weights.

This procedure performs interpolation in weight space rather than ensembling at inference time. Serving therefore requires only one 356,418-parameter model.

The interpolation coefficient and evidence scale are selected using the second-last interaction and nine ranking metrics evaluated at cutoffs 10, 50, and 200. The final interaction is used only for test reporting.

## Reproduction

Use Python 3.12 and the committed lockfile for the CPU test environment:

```bash
uv sync --frozen
uv run pytest -q
```

For training, install a CUDA-enabled PyTorch build in a separate environment. The committed lockfile intentionally resolves the portable CPU build.

Prepare the shared evaluation protocol:

```bash
uv run vassago data download --release ml-1m --output data/raw

uv run vassago data build \
  --source data/raw/ml-1m \
  --output data/processed/ml1m

uv run vassago benchmark prepare \
  --data data/processed/ml1m \
  --output data/processed/ml1m-fair-v1
```

Select checkpoints using temporal validation without accessing test queries:

```bash
uv run vassago benchmark contextual \
  --config configs/experiment/ml1m_contextual.yaml \
  --data data/processed/ml1m \
  --protocol data/processed/ml1m-fair-v1 \
  --output reports/runs/contextual-validation.json \
  --validation-only
```

Train and export the selected model:

```bash
uv run vassago benchmark contextual \
  --config configs/experiment/ml1m_contextual.yaml \
  --data data/processed/ml1m \
  --protocol data/processed/ml1m-fair-v1 \
  --output reports/runs/contextual-ml1m-final/seed42/contextual-evidence.parquet
```

Meta baselines are generated with `scripts/hstu_fair_adapter.py` using upstream commit:

`6035c3f9b2512791b0983e0adc71749b2a22e7dc`

Evaluate all generated rankings using the same implementation:

```bash
uv run vassago benchmark evaluate \
  --protocol data/processed/ml1m-fair-v1 \
  --predictions \
    reports/runs/contextual-ml1m-final/seed42/contextual-evidence.parquet \
    reports/runs/contextual-ml1m-final/seed42/hstu.parquet \
    reports/runs/contextual-ml1m-final/seed42/sasrec.parquet \
  --output reports/runs/contextual-ml1m-final/seed42/comparison.json
```

For the matched CUDA inference-memory experiment, run VASSAGO with:

```text
scripts/measure_contextual_inference.py --batch-size 64 --bf16
```

and use the same:

```text
--eval-batch-size 64 --bf16
```

arguments with the Meta adapter.

Use `--history-length 10` for the cold-start condition and omit it for the full warm-history protocol.

Evaluate both cold-start onboarding routes using the frozen final checkpoint:

```bash
python scripts/train_cold_onboarding.py \
  --protocol data/processed/ml1m-fair-v1 \
  --config configs/experiment/ml1m_contextual.yaml \
  --warm-weights reports/runs/contextual-ml1m-final/seed42/contextual-evidence.safetensors \
  --output reports/runs/contextual-ml1m-final/seed42/cold-onboarding.safetensors \
  --device cuda

python scripts/evaluate_profile_cold_start.py \
  --data data/processed/ml1m \
  --protocol data/processed/ml1m-fair-v1 \
  --config configs/experiment/ml1m_contextual.yaml \
  --weights reports/runs/contextual-ml1m-final/seed42/contextual-evidence.safetensors \
  --cold-weights reports/runs/contextual-ml1m-final/seed42/cold-onboarding.safetensors \
  --output reports/runs/contextual-ml1m-final/profile-cold-start-seed42.json \
  --device cuda
```

Run manifests record configurations, protocol hashes, prediction hashes, software and hardware information, parameter counts, and execution times.

Generated datasets, rankings, and model weights are excluded from version control.

## Scientific scope and limitations

The reported results are based on a single-seed exploratory comparison.

Earlier variants developed during this research series were inspected using the same seed-42 test split. For this reason, stronger publication-level claims would require evaluation on a new untouched protocol or dataset, together with the registered multi-seed analysis.

The reported offline ranking metrics measure predictive performance under the specified experimental protocol. They do not, by themselves, establish improvements in user satisfaction, causal outcomes, fairness, robustness, or production safety.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [DATA_CARD.md](DATA_CARD.md), and [DATA_LICENSES.md](DATA_LICENSES.md) for the complete experimental protocol, data provenance, and documented limitations.

## Repository layout

```text
configs/experiment/ml1m_contextual.yaml       Final model configuration

src/vassago/contextual_ranker.py              Architecture, training, and export
src/vassago/fair_benchmark.py                 Fixed protocol and shared evaluator

scripts/hstu_fair_adapter.py                  Pinned official Meta baseline adapter
scripts/evaluate_profile_cold_start.py        Profile-proxy selection and evaluation
scripts/train_cold_onboarding.py              Validation-selected cold-adapter training
scripts/measure_contextual_inference.py       CUDA serving-memory measurement

tests/test_contextual_ranker.py               Architecture and causal-sampling tests
tests/test_fair_benchmark.py                  Protocol and evaluator contract tests
```

The VASSAGO source code is available under the [PolyForm Strict License 1.0.0](LICENSE). It permits non-commercial use, research, study and evaluation, but does not permit modification, distribution or commercial use. Licenses for the dataset and pretrained components are documented separately in `DATA_LICENSES.md`.
