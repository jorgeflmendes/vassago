# Reproducibility

## Environment

Use Python 3.12 and the committed `uv.lock` for the reference CPU checks:

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src scripts tests
uv run mypy src scripts
```

Training requires a separate CUDA-enabled PyTorch installation. The executed run used
PyTorch 2.13.0+cu130, CUDA 13.0 and an NVIDIA GeForce RTX 5080. The portable lockfile
intentionally resolves CPU PyTorch and must not be described as CUDA-enabled.

## Immutable protocol

`vassago benchmark prepare` materializes protocol v3 of the comparison. Its
`protocol.json` binds SHA-256 hashes for the prepared interactions, catalog, query
files and upstream sequence input. The protocol hash is generated from those inputs.

Each user's last interaction is test and the second-last interaction is validation.
Model inputs contain at most the latest 200 items. Seen-item exclusion uses the complete
pre-query history. Every model ranks the complete catalog recorded by the protocol. The
evaluator rejects partial rankings, missing users, duplicate items or ranks, invalid IDs,
seen recommendations and mismatched hashes.

Protocol v3 never exposes the held-out interaction time as a model feature. Historical
events retain their own timestamps, while the offline query time is the timestamp of the
last observed event. The held-out timestamp is stored separately for audit only.

Meta HSTU and SASRec use `scripts/hstu_fair_adapter.py` at upstream commit
`6035c3f9b2512791b0983e0adc71749b2a22e7dc`. The adapter changes the dataset source
and exports rankings; model logic remains upstream.

The experimental workflow was informed by the public
[Recommenders](https://github.com/recommenders-team/recommenders) project, particularly
its separation of data preparation, modeling, offline evaluation, model selection, and
operationalization. VASSAGO does not vendor its code, data, weights, or results.

## Model selection

`configs/experiment/ml1m_contextual.yaml` fixes the training configuration. Warm-model
selection uses the second-last interaction and emits a checkpoint trained without that
target. Cold-start selection must use this pre-validation checkpoint. After the number
of cold-adapter epochs is fixed, the adapter is retrained from the final warm checkpoint
using all permitted pre-test events.

MovieLens 1M is a development benchmark because earlier architecture variants were
inspected on its test split. The external MovieLens 100K run is registered in
`configs/experiment/ml100k_external_benchmark.json`. VASSAGO uses the frozen recipe in
`configs/experiment/ml100k_external_recipe.json`; it does not tune epochs, interpolation,
or evidence scale on the external dataset.

Manifests record the resolved configuration, protocol and file hashes, Python, PyTorch,
CUDA, hardware, parameter counts and durations. Weights use SafeTensors. Generated
data, rankings and checkpoints are deliberately ignored by Git.

## Serving measurements

Serving measurements include the causal timestamps used by the model, complete seen-item
filtering, and top-200 selection. They run warm-up batches followed by repeated measured
passes and report throughput, p50/p95/p99 batch latency, peak allocated CUDA memory, and
peak reserved CUDA memory. HTTP, serialization, queueing, and network time are outside the
measurement, so queries per second must not be described as API requests per second.

The standalone VASSAGO measurement is reproducible, but it is not used to claim a
cross-model serving winner. The upstream Meta evaluator combines its forward path with
different metric and candidate-filtering work. Cross-model throughput requires a common
harness around identical masking and top-k operations.

## Statistical analysis

Metrics are averaged per query. Per-seed comparisons use paired bootstrap over matching
query IDs. The aggregate primary comparison uses a paired crossed multiplier bootstrap
with shared seed and user weights. Recall and HitRate, and MAP and MRR, coincide under this
one-target-per-query protocol and must not be counted as independent wins.

Protocol v1 and v2 results are withdrawn after the audit. Protocol v3 removes the
held-out timestamp from model inputs. MovieLens 1M v3 remains a development result and
cannot repair prior adaptive use of that test split. The registered external comparison
uses seeds 42 through 46, one fixed configuration per model, and reports aggregate means,
sample standard deviations, and confidence intervals only after all runs complete.
The compact committed result in `configs/experiment/ml100k_external_results.json` records
the aggregate values and SHA-256 of the full local comparison artifact.
