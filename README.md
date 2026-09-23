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
  <a href="#acknowledgements">Acknowledgements</a> ·
  <a href="#scientific-scope-and-limitations">Limitations</a>
</p>

VASSAGO (Variable-gap Attention-based Sequential Scoring, Adaptive Gating, and
Ordering) is a reproducible research implementation of a compact temporal
contextual-evidence ranker for sequential recommendation.

> **Research status.** Protocol v3 removes the held-out interaction timestamp from every
> model input. MovieLens 1M is now a development benchmark; comparative claims are based
> only on the registered external evaluation described below.
>
> The published results below evaluate the contextual-evidence model in this source tree.

| | |
|---|---|
| **Task** | Sequential recommendation and ranking |
| **Datasets** | MovieLens 1M development; registered MovieLens 100K external test |
| **Protocol** | Fixed full-catalog evaluation with complete seen-item filtering |
| **Baselines** | Meta HSTU and Meta SASRec |
| **Model size** | Recorded per checkpoint manifest |
| **Reference hardware** | NVIDIA RTX 5080; CUDA 13.0; bfloat16 inference |

## Architecture

The final model consists of:

* a two-layer causal SASRec backbone with tied item scoring;
* learned logarithmic event-gap embeddings;
* per-head query-time attention biases using only an explicit request time;
* exposure-safe causal negative sampling;
* a candidate-conditioned evidence module operating on recent states;
* validation-selected interpolation between frozen and jointly adapted checkpoints; and
* full-catalog evaluation with identical seen-item exclusion across models.

The evidence correction is candidate-specific and bounded by a learned scale.

Model selection is performed using each user's second-last interaction, while the final interaction is reserved for test reporting. Offline evaluation uses the last observed timestamp as the request time; the held-out interaction timestamp is never passed to the model. Training terminates with an error if non-finite validation or test scores are encountered, preventing invalid rankings from being exported.

## Cold-start routing

The inference pipeline supports two cold-start settings.

When an ordered list of up to ten favorite movies is available, the model uses the contextual checkpoint together with a small set-centroid residual.

When favorite items are unavailable, a metadata-based expert uses explicit genres, directors, actors, languages, and year ranges, with a ridge projection into the learned collaborative item space. Because coarse profile attributes can produce large groups of tied candidates, a popularity prior is selected on the validation split to resolve these ties.

Once at least one positive behavioral interaction is available, both onboarding components are bypassed and ranking is performed exclusively by the final contextual model. Consequently, warm-user rankings are invariant to changes in the profile fields.

MovieLens 1M does not contain explicit onboarding profiles. To evaluate the profile-based route, an exploratory proxy was constructed using the three most frequent genres in each user's pre-query history while withholding all historical item IDs from the cold-start ranker. The popularity weight was selected using the second-last interaction.

A second onboarding proxy represents recent pre-query items as ordered favorite selections. Favorite count and set-centroid weight are selected on the independent validation split.

These proxies are exploratory and do not replace evaluation with independently collected user profiles. Their earlier results used a selection checkpoint that had already seen the validation target and are therefore withdrawn. The corrected workflow selects the adapter using a pre-validation checkpoint, then retrains it for the selected number of epochs from the final warm checkpoint.

## Results

Protocol v3 uses complete pre-query seen-item exclusion, deterministic score-tie handling, strict rank validation, and causal timestamps. Protocol v1 and v2 results are withdrawn: v1 had incomplete seen-item filtering and dependent cold-start selection, while v2 still supplied the held-out interaction time as the offline request time.

The shared evaluator reports Recall, NDCG, MRR, MAP and HitRate together with catalog coverage, long-tail coverage, average popularity, novelty and genre diversity. Recall and HitRate, and MAP and MRR, are equivalent in the one-target-per-query protocol and are not counted as separate wins.

On the MovieLens 1M development split, the causal VASSAGO run obtains Recall@10 `0.332450`, NDCG@10 `0.195728`, Recall@50 `0.591887`, NDCG@50 `0.253240`, Recall@200 `0.780132`, and NDCG@200 `0.281826`. These values are diagnostics rather than an untouched comparative result.

### External benchmark (MovieLens-100K Protocol v3)

The registered MovieLens 100K evaluation contains 943 queries over 1,682 items. All three models use five seeds (42–46), a fixed configuration, the same causal split, complete-catalog ranking, and complete pre-query seen-item exclusion under Protocol v3. VASSAGO uses the compact linear feedforward architecture (`dimension=48`, `ffn_dim=48`) with calibrated additive evidence scaling ($\lambda = 0.50$).

#### 1. Ranking Accuracy and Precision

| Model | Parameters | NDCG@10 | Recall@10 | MRR@10 | NDCG@50 | Recall@50 | NDCG@200 | Recall@200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **VASSAGO (ours)** | 123,042 | $\mathbf{0.1134 \pm 0.0037}$ | $\mathbf{0.2087 \pm 0.0043}$ | $\mathbf{0.0846 \pm 0.0040}$ | $0.1685 \pm 0.0045$ | $0.4621 \pm 0.0077$ | $\mathbf{0.2138 \pm 0.0031}$ | $0.7601 \pm 0.0104$ |
| Meta HSTU adapted | **120,900** | $0.1080 \pm 0.0027$ | $0.2038 \pm 0.0080$ | $0.0793 \pm 0.0018$ | $\mathbf{0.1690 \pm 0.0035}$ | $\mathbf{0.4821 \pm 0.0118}$ | $0.2121 \pm 0.0031$ | $\mathbf{0.7676 \pm 0.0087}$ |
| Meta SASRec adapted | 125,300 | $0.1043 \pm 0.0018$ | $0.1951 \pm 0.0073$ | $0.0767 \pm 0.0008$ | $0.1603 \pm 0.0013$ | $0.4530 \pm 0.0069$ | $0.2060 \pm 0.0010$ | $0.7546 \pm 0.0016$ |

*Notes on accuracy metrics:*
* At top-rank cutoffs ($K=10$), VASSAGO achieves higher precision than both Meta baselines across NDCG@10, Recall@10, and MRR@10. Paired permutation tests across the 5 seeds on NDCG@10 yield $p = 0.803$ versus Meta HSTU and $p = 0.290$ versus Meta SASRec (the mean difference is positive, but the 5-seed sample size does not achieve statistical significance at $\alpha = 0.05$).
* At intermediate and deep cutoffs ($K=50$ and $K=200$), Meta HSTU achieves higher recall (Recall@50 `0.4821` vs `0.4621`; Recall@200 `0.7676` vs `0.7601`) and slightly higher NDCG@50 (`0.1690` vs `0.1685`).
* In model size, Meta HSTU remains the most compact architecture (120,900 parameters); VASSAGO operates at 123,042 parameters (-1.8% fewer than Meta SASRec's 125,300).

#### 2. Beyond-Accuracy and Catalog Dynamics (@10)

| Model | Catalog Coverage ↑ | Long-Tail Coverage ↑ | Novelty (Self-Info) ↑ | Genre Diversity ↑ | Mean Popularity ↓ |
|---|---:|---:|---:|---:|---:|
| **VASSAGO (ours)** | $\mathbf{0.6813}$ | $\mathbf{0.4164}$ | $\mathbf{9.83}$ | $\mathbf{0.7701}$ | $\mathbf{157.2}$ |
| Meta HSTU adapted | $0.5422$ | $0.2811$ | $9.12$ | $0.7180$ | $198.4$ |
| Meta SASRec adapted | $0.4912$ | $0.2215$ | $8.84$ | $0.6942$ | $212.4$ |

*Observations on catalog exploration:* VASSAGO exhibits significantly higher catalog coverage and recommendation of long-tail items at cutoff 10, resulting in higher average self-information novelty and lower concentration on mainstream popular titles.

#### 3. Serving Efficiency (bfloat16, batch size 64)

| Model | Parameters | P95 Latency ↓ | Throughput (QPS) ↑ | Peak VRAM ↓ |
|---|---:|---:|---:|---:|
| **VASSAGO (ours)** | 123,042 | $\mathbf{1.82\text{ ms}}$ | $\mathbf{1,420}$ | $\mathbf{142\text{ MB}}$ |
| Meta HSTU adapted | **120,900** | $2.14\text{ ms}$ | $1,180$ | $184\text{ MB}$ |
| Meta SASRec adapted | 125,300 | $2.31\text{ ms}$ | $1,150$ | $188\text{ MB}$ |

#### 4. Objective Metric Breakdown Across Evaluated Dimensions

Rather than conflating separate metrics into an arbitrary "score", the empirical trade-offs across all 16 measured dimensions are summarized below:

| Dimension | Metric | Meta SASRec | Meta HSTU | VASSAGO (ours) | Best Result |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model Footprint** | Parameters ($\downarrow$) | 125,300 | **120,900** | 123,042 | **Meta HSTU** |
| **Top-10 Precision** | NDCG@10 ($\uparrow$) | 0.1043 | 0.1080 | **0.1134** | **VASSAGO** |
| | Recall@10 ($\uparrow$) | 0.1951 | 0.2038 | **0.2087** | **VASSAGO** |
| | MRR@10 ($\uparrow$) | 0.0767 | 0.0793 | **0.0846** | **VASSAGO** |
| **Cutoff 50 Retrieval** | NDCG@50 ($\uparrow$) | 0.1603 | **0.1690** | 0.1685 | **Meta HSTU** |
| | Recall@50 ($\uparrow$) | 0.4530 | **0.4821** | 0.4621 | **Meta HSTU** |
| **Cutoff 200 Retrieval** | NDCG@200 ($\uparrow$) | 0.2060 | 0.2121 | **0.2138** | **VASSAGO** |
| | Recall@200 ($\uparrow$) | 0.7546 | **0.7676** | 0.7601 | **Meta HSTU** |
| **Catalog & Diversity** | Catalog Coverage@10 ($\uparrow$) | 0.4912 | 0.5422 | **0.6813** | **VASSAGO** |
| | Long-Tail Coverage@10 ($\uparrow$) | 0.2215 | 0.2811 | **0.4164** | **VASSAGO** |
| | Novelty@10 ($\uparrow$) | 8.84 | 9.12 | **9.83** | **VASSAGO** |
| | Genre Diversity@10 ($\uparrow$) | 0.6942 | 0.7180 | **0.7701** | **VASSAGO** |
| | Average Popularity@10 ($\downarrow$) | 212.4 | 198.4 | **157.2** | **VASSAGO** |
| **Inference Efficiency** | P95 Latency ($\downarrow$) | 2.31 ms | 2.14 ms | **1.82 ms** | **VASSAGO** |
| | Throughput QPS ($\uparrow$) | 1,150 | 1,180 | **1,420** | **VASSAGO** |
| | Peak VRAM ($\downarrow$) | 188 MB | 184 MB | **142 MB** | **VASSAGO** |

**Empirical Summary**:
* **Top-rank ranking ($K=10$) & Catalog Exploration**: VASSAGO obtains the highest accuracy at the immediate decision boundary ($K=10$), accompanied by greater recommendation diversity and lower memory/latency overhead during full-catalog scoring.
* **Broad candidate recall ($K=50, K=200$) & Parameter footprint**: Meta HSTU remains stronger at capturing broader candidate sets deeper in the catalog (Recall@50 and Recall@200) and holds the lowest total parameter count (120,900 vs 123,042).
* Machine-readable benchmark data: [configs/experiment/ml100k_external_results.json](configs/experiment/ml100k_external_results.json) | Recipe: [configs/experiment/ml100k_external_recipe.json](configs/experiment/ml100k_external_recipe.json). Reproducible with `python scripts/evaluate_vassago_compact.py`.

### Validation-selected checkpoint interpolation

The final training phase jointly fine-tunes the contextual module and backbone, using a reduced learning rate for the backbone.

Validation selected base epoch 75, contextual epoch 10, and joint epoch 5 under the causal MovieLens 1M development protocol. A subsequent validation step selected a single interpolated checkpoint containing 65% of the frozen contextual weights and 35% of the jointly adapted weights.

This procedure performs interpolation in weight space rather than ensembling at inference time. Serving therefore requires only one 356,418-parameter model.

The interpolation coefficient and evidence scale are selected using the second-last interaction and the registered primary metric, NDCG@10. The final interaction is used only for test reporting. Historical v3 values used an earlier multi-metric scale rule and remain exploratory rather than a claim of superiority.

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
  --output data/processed/ml1m-fair-v3
```

Select checkpoints using temporal validation without accessing test queries:

```bash
uv run vassago benchmark contextual \
  --config configs/experiment/ml1m_contextual.yaml \
  --data data/processed/ml1m \
  --protocol data/processed/ml1m-fair-v3 \
  --output artifacts/runs/contextual-ml1m-v3/seed42/contextual-validation.json \
  --validation-only
```

Train and export the selected model:

```bash
uv run vassago benchmark contextual \
  --config configs/experiment/ml1m_contextual.yaml \
  --data data/processed/ml1m \
  --protocol data/processed/ml1m-fair-v3 \
  --output artifacts/runs/contextual-ml1m-v3/seed42/contextual-evidence.parquet
```

Meta baselines are generated with `scripts/hstu_fair_adapter.py` using upstream commit:

`6035c3f9b2512791b0983e0adc71749b2a22e7dc`

Evaluate all generated rankings using the same implementation:

```bash
uv run vassago benchmark evaluate \
  --protocol data/processed/ml1m-fair-v3 \
  --predictions \
    artifacts/runs/contextual-ml1m-v3/seed42/contextual-evidence.parquet \
    artifacts/runs/contextual-ml1m-v3/seed42/hstu-adapted.parquet \
    artifacts/runs/contextual-ml1m-v3/seed42/sasrec-adapted.parquet \
  --output artifacts/runs/contextual-ml1m-v3/seed42/comparison.json
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
  --protocol data/processed/ml1m-fair-v3 \
  --config configs/experiment/ml1m_contextual.yaml \
  --selection-weights artifacts/runs/contextual-ml1m-v3/seed42/contextual-validation.safetensors \
  --selection-manifest artifacts/runs/contextual-ml1m-v3/seed42/contextual-validation.json \
  --final-weights artifacts/runs/contextual-ml1m-v3/seed42/contextual-evidence.safetensors \
  --final-manifest artifacts/runs/contextual-ml1m-v3/seed42/contextual-evidence.parquet.manifest.json \
  --output artifacts/runs/contextual-ml1m-v3/seed42/cold-onboarding.safetensors \
  --device cuda

python scripts/evaluate_profile_cold_start.py \
  --data data/processed/ml1m \
  --protocol data/processed/ml1m-fair-v3 \
  --config configs/experiment/ml1m_contextual.yaml \
  --selection-weights artifacts/runs/contextual-ml1m-v3/seed42/contextual-validation.safetensors \
  --selection-manifest artifacts/runs/contextual-ml1m-v3/seed42/contextual-validation.json \
  --final-weights artifacts/runs/contextual-ml1m-v3/seed42/contextual-evidence.safetensors \
  --final-manifest artifacts/runs/contextual-ml1m-v3/seed42/contextual-evidence.parquet.manifest.json \
  --selection-cold-weights artifacts/runs/contextual-ml1m-v3/seed42/cold-onboarding-selection.safetensors \
  --final-cold-weights artifacts/runs/contextual-ml1m-v3/seed42/cold-onboarding.safetensors \
  --cold-manifest artifacts/runs/contextual-ml1m-v3/seed42/cold-onboarding.safetensors.manifest.json \
  --output artifacts/runs/contextual-ml1m-v3/profile-cold-start-seed42.json \
  --device cuda
```

Run manifests record configurations, protocol hashes, prediction hashes, software and hardware information, parameter counts, and execution times.

Generated datasets, rankings, and model weights are excluded from version control.

## Acknowledgements

VASSAGO thanks the contributors to [Recommenders](https://github.com/recommenders-team/recommenders) for their public collection of recommendation-system practices and examples.

Its separation of data preparation, modeling, offline evaluation, model selection, and operationalization informed this project's experimental workflow. In particular, it reinforced the use of explicit ranking metrics at fixed cutoffs, an isolated model-selection split, reproducible run metadata, and clear limits on what offline ranking results establish.

VASSAGO does not vendor code, trained weights, datasets, or benchmark results from Recommenders. The architecture, protocol v3 implementation, model training, and reported rankings in this repository were developed and executed independently.

## Scientific scope and limitations

MovieLens 1M is a development benchmark because its test split informed earlier architecture work. The registered MovieLens 100K external comparison fixes one configuration per model and seeds 42 through 46 before aggregate test evaluation.

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
