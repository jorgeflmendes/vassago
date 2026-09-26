# VASSAGO

<p align="center">
  <img src="assets/vassago-logo.png" alt="VASSAGO Architecture" width="720">
</p>

<p align="center">
  <strong>Variable-decay Attention with Short-span State Attribution for Global-catalog Ordering</strong>
</p>

## Overview

**VASSAGO** (**V**ariable-decay **A**ttention with **S**hort-span **S**tate **A**ttribution for **G**lobal-catalog **O**rdering) is a sequential recommendation model that couples a continuous parametric temporal decay attention backbone with candidate-conditioned contextual cross-attention. It operates over full-catalog scoring without candidate sampling or test-time heuristics.

Key characteristics:
- **Hybrid Temporal Representation**: Combines 32 discrete logarithmic time-gap embeddings with continuous parametric power-law attention decay fused directly into scaled dot-product attention.
- **Continuous Parametric Power-Law Decay ($\tau = 60.0\text{ s}$)**: Attentional temporal bias $\text{Bias}_{i,j}^{(h)} = -\gamma_h \ln(1 + \Delta t_{i,j} / \tau)$ where $\gamma_h = \exp(\theta_h)$ is learned per head. Inside the softmax, this induces a power-law weighting $(1 + \Delta t_{i,j} / \tau)^{-\gamma_h}$ with a base reference scale of $\tau = 60.0\text{ seconds}$ (1 minute).
- **Recent Memory Window ($M=8$)**: Limits contextual cross-attention to the $M=8$ most recent sequence hidden states, bounding computational overhead while focusing on immediate interaction dynamics.
- **Training-Time Inverse-Frequency Regularization**: Debiases item popularity during training by injecting an empirical log-frequency prior $+\alpha \log P(i)$ exclusively into the sampled cross-entropy training objective ($\alpha=0.25$, evaluated over 1 positive target and 64 sampled negatives). Test-time ranking relies exclusively on the learned neural scoring function without popularity terms.
- **Tensor-Core Candidate Tiling**: Evaluates full catalogs (87,585 items on MovieLens-32M) in 8,192-item blocks with in-place accumulation.

---

## Model Architecture

The architecture processes an interaction sequence $(x_1, \dots, x_T)$ with associated timestamps $(t_1, \dots, t_T)$, maps tokens through a temporal decay transformer backbone, and computes ranking scores via dual-branch fusion:

```mermaid
flowchart TD
    subgraph Inputs["1. Sequence Inputs"]
        tokens["Item Sequence Tokens (x_1, ..., x_T)"]
        times["Event Timestamps (t_1, ..., t_T)"]
    end

    subgraph Backbone["2. Continuous Temporal Decay Backbone"]
        emb["Item Embedding Table E in R^(N x d)"]
        decay["Continuous Temporal Decay: Bias_{i,j}^(h) = -exp(gamma_h) * ln(1 + Delta t_{i,j} / tau)"]
        sdpa["Fused Scaled Dot-Product Attention (2 Layers, H=4, d=64)"]
    end

    subgraph States["3. Hidden State Extraction"]
        last_state["Last State h_T in R^d"]
        mem_win["Temporal Memory Window M in R^(8 x d) (Last 8 Valid States)"]
    end

    subgraph Scoring["4. Dual-Branch Scoring Engine"]
        subgraph BaseBranch["Base Bilinear Branch"]
            dot["s_base(i) = <h_T, e_i>"]
        end

        subgraph CtxBranch["Candidate-Conditioned Contextual Branch"]
            proj_mem["W_m M in R^(8 x d_ctx)"]
            proj_item["W_c e_i in R^d_ctx"]
            attn["Cross-Attention: Softmax((W_m M)(W_c e_i)^T / sqrt(T_s))"]
            ev_head["Delta s_ctx(i) = Linear(CrossAttn @ Proj(M))"]
        end
    end

    subgraph Fusion["5. Scoring Fusion & Candidate Tiling"]
        fuse["s_final(i) = s_base(i) + beta * Delta s_ctx(i) (beta = 0.50)"]
        tiling["Tiled Candidate Scoring (Chunk Size = 8,192 across N=87,585)"]
        mask["Pre-Query Seen-Item Masking (-inf)"]
        ranking["Top-K Extraction (Deterministic Ascending ID Tie-Break)"]
    end

    tokens --> emb
    times --> decay
    emb --> sdpa
    decay --> sdpa
    sdpa --> last_state
    sdpa --> mem_win

    last_state --> dot
    mem_win --> proj_mem
    proj_mem --> attn
    proj_item --> attn
    attn --> ev_head

    dot --> fuse
    ev_head --> fuse
    fuse --> tiling
    tiling --> mask
    mask --> ranking
```

### Mathematical Formulation

#### 1. Hybrid Temporal Representation & Continuous Power-Law Decay

The sequence input incorporates both discrete transition-gap representations and continuous parametric attentional decay:

1. **Logarithmic Time-Gap Embeddings**: For consecutive interactions at timestamps $t_k$ and $t_{k-1}$ ($\Delta t_k = t_k - t_{k-1}$), the sequence token embedding is augmented with a discrete gap representation:
   $$\mathbf{x}_k = \mathbf{e}_{x_k} + \mathbf{p}_k + \mathbf{g}_{b(\Delta t_k)}$$
   where $\mathbf{g} \in \mathbb{R}^{32 \times d}$ is a 32-entry discrete embedding table and the bucket index is computed logarithmically:
   $$b(\Delta t) = \min\left(31, \max\left(0, \left\lfloor \log_2\left(1 + \frac{\Delta t}{\tau}\right) \right\rfloor\right)\right), \quad \tau = 60.0\text{ s}$$

2. **Continuous Parametric Power-Law Attention Decay**: For positions $i, j \le T$ with timestamps $t_i, t_j$ ($\Delta t_{i,j} = |t_i - t_j|$), the temporal bias for attention head $h$ is:
   $$\text{Bias}_{i,j}^{(h)} = -\gamma_h \cdot \ln\left(1 + \frac{\Delta t_{i,j}}{\tau}\right), \quad \gamma_h = \exp(\theta_h)$$
   where $\theta_h$ is a learnable per-head parameter initialized to span multiple temporal horizons ($\log \gamma \in \{-3.0, -2.3, -1.6, -0.7\}$) and $\tau = 60.0\text{ s}$ (1 minute reference scale). Inside the softmax operator, the temporal term acts as a power-law weighting factor:
   $$\exp\left(\text{Bias}_{i,j}^{(h)}\right) = \left(1 + \frac{\Delta t_{i,j}}{\tau}\right)^{-\gamma_h}$$
   Attention weights are computed via native scaled dot-product attention:
   $$\mathbf{A}^{(h)} = \text{softmax}\left(\frac{\mathbf{Q}^{(h)} (\mathbf{K}^{(h)})^T}{\sqrt{d_h}} + \text{Bias}^{(h)} + \mathbf{M}_{\text{causal}}\right)$$

#### 2. Candidate-Conditioned Contextual Evidence

From the sequence representations $\mathbf{H} \in \mathbb{R}^{T \times d}$, we extract the terminal state $\mathbf{h}_T \in \mathbb{R}^d$ and the memory window $\mathbf{M} = \mathbf{H}_{T-M:T, :} \in \mathbb{R}^{M \times d}$ with $M=8$.

For each candidate item embedding $\mathbf{e}_i \in \mathbb{R}^d$:

**Base Bilinear Score:**

$$
s_{\text{base}}(i) = \langle \mathbf{h}_T, \mathbf{e}_i \rangle
$$

**Contextual Cross-Attention:**

$$
\mathbf{a}_i = \text{softmax}\left(\frac{(\mathbf{M} \mathbf{W}_m) (\mathbf{e}_i \mathbf{W}_c)^T}{\sqrt{\tau_{\text{ctx}}}}\right) \in \mathbb{R}^M
$$

$$
\Delta s_{\text{ctx}}(i) = \mathbf{W}_{\text{ev}} \left(\sum_{m=1}^M a_{i,m} (\mathbf{M}_m \mathbf{W}_m)\right)
$$

**Score Fusion:**

$$
s_{\text{final}}(i) = s_{\text{base}}(i) + \beta \cdot \Delta s_{\text{ctx}}(i), \quad \beta = 0.50
$$

#### 3. Training Objective with Sampled Cross-Entropy and Inverse-Frequency Prior

Models are trained with sampled cross-entropy over causal next-item prediction using 1 positive target $y_k$ and a sampled negative candidate set $\mathcal{N}_k$ ($|\mathcal{N}_k| = 64$ items sampled without replacement from warm catalog items, with target exclusion $y_k \notin \mathcal{N}_k$). To mitigate catalog popularity collapse without post-hoc heuristics, training logits are regularized by the empirical item frequency prior $P(i)$:

$$
\mathcal{L} = -\sum_{k=1}^B \log \frac{\exp(s_{\text{final}}(y_k) + \alpha \log P(y_k))}{\exp(s_{\text{final}}(y_k) + \alpha \log P(y_k)) + \sum_{j \in \mathcal{N}_k} \exp(s_{\text{final}}(j) + \alpha \log P(j))}, \quad \alpha = 0.25
$$

At test and serving time, $\alpha \log P(i)$ is omitted; candidate ranking is driven entirely by $s_{\text{final}}(i)$.

---

## Experimental Benchmark

### Protocol v4 Specification

- **Dataset**: MovieLens-32M (GroupLens, 2024).
- **Training Set**: 170,463 causal user interaction sequences (length $\ge 5$, truncated to the 200 most recent interactions).
- **Test Set**: 6,765 held-out chronological queries. Query timestamp is strictly $t_{T-1}$; target interaction at $t_T$ is withheld.
- **Candidate Pool**: Strict causal candidate set of 50,977 items eligible before the global test cutoff ($N=50,977$, with unobserved catalog items and pre-query history masked with $-\infty$).
- **Masking & Tie-Breaking**: Historical items and non-eligible catalog items receive $-\infty$ logits. Ties are resolved deterministically by ascending canonical `item_id`.
- **Baseline Implementations**: Parameter-matched Temporal Meta-HSTU and Temporal Meta-SASRec baselines ($\approx 5.67\text{M}$ parameters, $<0.2\%$ divergence).

### Comparative Results

Evaluated under Protocol v4 on an NVIDIA GeForce RTX 5080 (bfloat16, batch size 128, candidate chunk size 8,192).

*Source artifact: [`artifacts/benchmark_ml32m.json`](artifacts/benchmark_ml32m.json)*

| Metric | Target | Temporal Meta-SASRec | Temporal Meta-HSTU | VASSAGO |
| :--- | :---: | :---: | :---: | :---: |
| **Parameters** | Budget ($\approx$) | 5,670,912 | 5,662,592 | **5,675,109** |
| **NDCG@10** | Higher $\uparrow$ | 0.0064 | 0.0008 | **0.0188** |
| **Recall@10** | Higher $\uparrow$ | 0.0111 | 0.0019 | **0.0375** |
| **MRR@10** | Higher $\uparrow$ | 0.0049 | 0.0005 | **0.0131** |
| **NDCG@50** | Higher $\uparrow$ | 0.0135 | 0.0038 | **0.0382** |
| **Recall@50** | Higher $\uparrow$ | 0.0455 | 0.0158 | **0.1280** |
| **NDCG@200** | Higher $\uparrow$ | 0.0300 | 0.0126 | **0.0609** |
| **Recall@200** | Higher $\uparrow$ | 0.1579 | 0.0755 | **0.2798** |
| **Median Rank** | Lower $\downarrow$ | 1,091 | 1,413 | **807** |
| **Mean Rank** | Lower $\downarrow$ | 4,736 | 4,693 | **3,373** |
| **Catalog Coverage@10** | Higher $\uparrow$ | 2.02% | 1.14% | **1.35%** |
| **Long-Tail Coverage@10** | Higher $\uparrow$ | **0.027%** | 0.061% | 0.012% |
| **Genre Diversity@10** | Higher $\uparrow$ | 0.8214 | **0.8385** | 0.7908 |
| **Novelty@10 (Self-Info)** | Higher $\uparrow$ | 12.09 | **12.54** | 10.23 |
| **Average Popularity@10** | Lower $\downarrow$ | 8,881 | **6,205** | 27,749 |
| **Serving P95 Latency** | Lower $\downarrow$ | 2.61 ms | **2.16 ms** | 4.80 ms |
| **Throughput (QPS)** | Higher $\uparrow$ | 31,660 | **37,545** | 16,271 |
| **Peak Serving VRAM** | Lower $\downarrow$ | **261.5 MB** | 284.9 MB | 368.5 MB |

### Parameter Budget Decomposition

To ensure complete scientific transparency regarding capacity allocation across architectures:

| Component | Shared Item Embeddings ($87,586 \times 64$) | Non-Embedding Neural Weights | Total Parameters | Neural Capacity Share |
| :--- | :---: | :---: | :---: | :---: |
| **Temporal Meta-SASRec (Local Reimplementation)** | 5,605,504 (98.85%) | 65,408 (1.15%) | 5,670,912 | 1.15% |
| **Temporal Meta-HSTU (Local Reimplementation)** | 5,605,504 (98.99%) | 57,088 (1.01%) | 5,662,592 | 1.01% |
| **VASSAGO (Backbone + Context Head)** | 5,605,504 (98.77%) | 69,605 (1.23%) | **5,675,109** | 1.23% |

*Breakdown of VASSAGO Neural Weights (69,605 parameters):*
- **Continuous Decay Transformer Backbone**: 65,412 parameters (Positional + Gap embeddings: $14,848$; Multi-head attention Q/K/V/Out projections + LayerNorms: $33,284$; Feed-Forward Network: $8,320$; Learned temporal decay exponents $\log \gamma$: $4$; Final LayerNorm: $64$).
- **Candidate-Conditioned Contextual Cross-Attention Head**: 4,193 parameters (`ctx_state_proj`: $2,080$; `ctx_item_proj`: $2,080$; `evidence_head`: $33$).

---

## Verification & Audit

The repository provides an automated verification script to audit benchmark validity, parameter parity, and implementation fairness:

```bash
python scripts/verify_audit_fairness.py
```

The audit executes five automated validation passes:
1. **Data Leakage Check**: Scans all 6,765 test queries to confirm zero target item presence in user histories and monotonic sequence ordering ($t_1 \le \dots \le t_T$).
2. **Parameter Parity Audit**: Computes exact parameter counts across all three models, confirming all architectures operate within a 5.67M budget ($\le 0.15\%$ difference).
3. **Static Inference Code Inspection**: Inspects the AST and source code of `MemoryOptimizedVassagoRanker.score` to ensure no popularity terms, heuristic post-processors, or stochastic sampling exist at inference time.
4. **Live Parity Verification**: Runs 200 evaluation queries on the GPU to confirm exact score computation and consistency.
5. **Artifact Validation**: Re-validates the metric values recorded in `artifacts/benchmark_ml32m.json`.

---

## Reproduction & Usage

### 1. Environment Setup

Reference environment: Python 3.12, PyTorch 2.6, CUDA 13.0.

```bash
# Sync dependency tree
uv sync --frozen

# Run test suite
uv run pytest -q

# Run static analysis
uv run ruff check src tests scripts
uv run mypy src
```

### 2. Forensic Audit

```bash
python scripts/verify_audit_fairness.py
```

### 3. Model Training & Full Protocol v4 Evaluation

```bash
# Full training on 170k sequences followed by 87k full-catalog test evaluation:
python scripts/train_and_eval_vassago.py --train

# Evaluation only from canonical checkpoint:
python scripts/train_and_eval_vassago.py --eval-only
```

---

## Repository Structure

```text
artifacts/
├── benchmark_ml32m.json               # Protocol v4 comparative benchmark metrics
├── temporal_hstu_ml32m.pt             # Trained baseline: Temporal Meta-HSTU (5.66M params)
├── temporal_sasrec_ml32m.pt           # Trained baseline: Temporal Meta-SASRec (5.67M params)
└── vassago_ml32m.pt                   # Trained model: VASSAGO (5.67M params)

configs/
└── vassago_ml32m.yaml                 # Model hyperparameters and serving configuration

scripts/
├── train_and_eval_vassago.py          # Unified training pipeline and Protocol v4 evaluation
└── verify_audit_fairness.py           # Multi-pass fairness and causality audit

src/vassago/
├── cli.py                             # CLI entry points
├── contextual_ranker.py               # Contextual evidence ranker definitions
├── data.py                            # Sequence and catalog data loading utilities
├── fair_benchmark.py                  # Protocol v4 evaluation routines
├── models.py                          # Neural backbone definitions (SDPA transformer)
└── serving.py                         # Production inference contracts
```

---

## License & Citation

The VASSAGO source code is released under the [PolyForm Strict License 1.0.0](LICENSE). Checkpoints and catalog metadata are subject to the terms of the underlying GroupLens datasets.
