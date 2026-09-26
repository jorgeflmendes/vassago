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

The primary comparative evaluation is conducted under **Protocol v4**, a rigorous causal ranking evaluation on the stable MovieLens-32M release:

1. **Dataset Split & Causal Boundaries**:
   - 170,463 historical sequences for training.
   - 6,765 held-out test queries, each with a single ground-truth next interaction.
   - User interaction sequences are strictly chronological. Offline query timestamp is the timestamp of the last observed interaction event. The held-out target timestamp is never supplied to the model as an input feature.
2. **Strict Candidate Set Masking**:
   - All models evaluate against the **50,977 candidate items** eligible prior to the global test cutoff (`candidate_item_ids`).
   - The 36,608 catalog items never observed prior to the cutoff receive $-\infty$ logits to eliminate random embedding noise.
3. **Seen-Item Masking**:
   - All items previously observed in the user's pre-query history receive $-\infty$ logits before top-$K$ selection.
4. **Deterministic Ranking**:
   - Top-$K$ items are extracted via deterministic sorting with standard tie-breaking by ascending canonical `item_id`.
   - No stochastic rerankers, manual heuristics, or popularity post-filters are applied.

---

## Automated Internal Verification Suite

To audit zero data leakage, exact parameter counts, AST safety, and inference behavior across models:

```bash
python scripts/verify_audit_fairness.py
```

The automated verification suite executes:
1. **Data Leakage Check**: Iterates through 100% of test queries, asserting that target items do not appear in history and that timestamps are monotonically non-decreasing.
2. **Parameter Parity Audit**: Computes exact trainable and total weight counts for local reimplementations of Meta-SASRec (5,670,912), Meta-HSTU (5,662,592), and VASSAGO (5,675,109), confirming all models operate within the 5.67M parameter budget ($\le 0.15\%$ divergence, dominated by the $87,586 \times 64$ shared embedding table).
3. **Static Inference Code Inspection**: Inspects the source code and AST of `vassago.score()` at runtime, proving that no popularity terms or heuristic filters exist in inference.
4. **Live Query Verification**: Executes an unbiased sample of 200 queries directly on GPU with candidate masking and deterministic ascending `item_id` tie-breaking, confirming ranking accuracy and scoring parity.
5. **Artifact Integrity**: Validates all 16 metrics stored in `artifacts/benchmark_ml32m.json`.

---

## Training and Benchmark Reproduction

The unified runner `scripts/train_and_eval_vassago.py` is configured dynamically via `configs/vassago_ml32m.yaml`.

### 1. Full 2-Stage Model Training and Evaluation

To train the complete model from scratch through the authentic 2-stage pipeline and evaluate on the full catalog:

```bash
python scripts/train_and_eval_vassago.py --train
```

The 2-stage training pipeline executes:
1. **Stage 1 (Backbone Pretraining)**: Optimizes item embeddings, positional embeddings, 32 discrete time-gap embeddings, and causal multi-head attention blocks (with continuous parametric temporal decay exponents $\log \gamma$) using sampled cross-entropy with 256 independent negatives per sequence.
2. **Stage 2 (Contextual Cross-Attention Head Training)**: Discriminatively fine-tunes the candidate-conditioned cross-attention projections (`ctx_state_proj`, `ctx_item_proj`, `evidence_head`) and backbone with frequency log-prior debiasing ($+\alpha \log P(i)$, $\alpha = 0.02$) on training logits to combat catalog popularity collapse.
3. **Candidate-Masked Protocol v4 Evaluation**: Evaluates all 6,765 held-out test queries against the 50,977 candidate items with $-\infty$ seen masking and deterministic ascending `item_id` tie-breaking.

### 2. Symmetric Baseline Training

To train the parameter-matched baselines under identical negative sampling and optimization conditions:

```bash
python scripts/train_baselines.py
```

### 3. Unified Comparative Evaluation

To reproduce the consolidated benchmark JSON across all three models:

```bash
python scripts/evaluate_all_models.py
```

```bash
python scripts/train_and_eval_vassago.py --eval-only
```

---

## Baseline Reimplementations

Parameter-matched baseline checkpoints are provided in:
- `artifacts/temporal_sasrec_ml32m.pt` (Temporal Meta-SASRec Local Reimplementation)
- `artifacts/temporal_hstu_ml32m.pt` (Temporal Meta-HSTU Local Reimplementation)

Both baselines are local reimplementations constructed under the identical 5.67M parameter budget, evaluated under identical full-catalog scoring contracts with seen-item masking and deterministic tie-breaking.

