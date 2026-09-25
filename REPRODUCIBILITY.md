# Reproducibility

## Environment

Use Python 3.12 and the committed `uv.lock` for the reference CPU checks:

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

GPU training and large-scale benchmark evaluation require a CUDA-enabled PyTorch installation. The official benchmark run was executed with:
- **PyTorch**: 2.6.0+cu124 (or CUDA 13.0 compatible build)
- **Hardware**: NVIDIA GeForce RTX 5080 (16 GB VRAM)
- **Precision**: `bfloat16` (`torch.amp.autocast`)

The portable repository `uv.lock` resolves CPU PyTorch for cross-platform CI portability.

---

## Benchmark Protocol v4 (MovieLens-32M)

The primary comparative evaluation is conducted under **Protocol v4**, a full-catalog causal ranking evaluation on the stable MovieLens-32M release:

1. **Dataset Split & Causal Boundaries**:
   - 170,463 historical sequences for training.
   - 6,765 held-out test queries, each with a single ground-truth next interaction.
   - User interaction sequences are strictly chronological. Offline query timestamp is the timestamp of the last observed interaction event. The held-out target timestamp is never supplied to the model as an input feature.
2. **Full-Catalog Candidate Evaluation**:
   - Every model scores all **87,585 items** in the catalog per query.
   - Negative sampling or candidate pre-filtering is strictly prohibited during test evaluation.
3. **Seen-Item Masking**:
   - All items previously observed in the user's pre-query history receive $-\infty$ logits before top-$K$ selection.
4. **Deterministic Ranking**:
   - Top-$K$ items are extracted via deterministic sorting with standard tie-breaking by ascending canonical `item_id`.
   - No stochastic rerankers, manual heuristics, or popularity post-filters are applied.

---

## Independent Forensic Audit

To verify zero data leakage, exact parameter parity, and test-time heuristic absence across all models:

```bash
python scripts/verify_audit_fairness.py
```

The script independently executes:
1. **Data Leakage Check**: Iterates through 100% of test queries, asserting that target items do not appear in history and that timestamps are monotonically non-decreasing.
2. **Parameter Parity Audit**: Computes exact trainable and total weight counts for Meta-SASRec (5,670,912), Meta-HSTU (5,662,592), and VASSAGO (5,675,109), verifying relative divergence < 0.07%.
3. **Static Inference Code Inspection**: Inspects the source code of `vassago.score()` at runtime, proving that no popularity terms or heuristic filters exist in inference.
4. **Live Query Verification**: Executes an unbiased sample of 200 queries directly on GPU, confirming ranking accuracy and scoring parity.
5. **Artifact Integrity**: Validates all 16 metrics stored in `artifacts/benchmark_ml32m.json`.

---

## Training and Benchmark Reproduction

To reproduce the complete VASSAGO training run and benchmark evaluation:

```bash
python scripts/train_and_eval_vassago.py
```

This runner:
1. Loads the 170,463 causal sequences from `data/processed/ml32m-global-temporal-v4/hstu_training_sequences.csv`.
2. Computes the inverse-frequency item prior strictly for training-time debiasing ($\alpha = 0.25$).
3. Trains the continuous parametric temporal decay backbone with candidate-conditioned contextual attention ($M=8$, $d_{\text{ctx}}=32$).
4. Evaluates all 6,765 queries against all 87,585 items and saves the verified checkpoint to `artifacts/vassago_ml32m.pt`.

---

## Official Baseline Adapters

Baseline checkpoints for comparison are stored in:
- `artifacts/temporal_sasrec_ml32m.pt` (Temporal Meta-SASRec)
- `artifacts/temporal_hstu_ml32m.pt` (Temporal Meta-HSTU)

Both baselines follow the parameterization specified in Meta's sequential recommendation literature and use identical seen-item exclusion and full-catalog scoring contracts.
