# Data Card

The primary experimental evaluation uses the official, stable MovieLens 32M release (GroupLens, 2024).
Ratings range from 0.5 to 5.0. All source ratings and timestamps are preserved in Parquet format; chronological splits select test interactions. Source user identifiers are canonicalized. Internal item IDs are contiguous ($1 \le \text{item\_id} \le 87{,}585$), with mapping to MovieLens/TMDB/IMDb IDs maintained in the catalog metadata.

---

## Dataset Splits and Protocol v4 Causal Structure

1. **Training Sequences**:
   - 170,463 historical user sequences containing at least 5 positive interactions.
   - Truncated causally to the most recent 200 items per user.
   - Stored in `data/processed/ml32m-global-temporal-v4/hstu_training_sequences.csv`.

2. **Evaluation Queries (Test Split)**:
   - 6,765 held-out chronological test queries.
   - For each query, the model receives the interaction history up to interaction $T-1$. The target interaction at time $T$ is held out.
   - Offline query timestamp is explicitly set to the timestamp of the last observed interaction ($t_{T-1}$). The held-out target timestamp $t_T$ is strictly withheld from model inputs during ranking.
   - Stored in `data/processed/ml32m-global-temporal-v4/queries.parquet`.

3. **Catalog & Metadata**:
   - Total catalog size: 87,585 items.
   - Item titles, release years, and multi-label genres are loaded from `data/processed/ml32m-global-temporal-v4/catalog.json`.
   - Used for evaluating genre diversity (Jaccard dissimilarity across Top-10 recommendations) and catalog coverage.

---

## Evaluation Boundaries and Integrity

1. **Full-Catalog Scoring & Strict Candidate Masking**:
   - The global catalog contains 87,585 items. Under Protocol v4, scoring evaluates against the **50,977 candidate items** observed in training sequences prior to the global test cutoff (`candidate_item_ids`).
   - The 36,608 catalog items never observed prior to the cutoff receive $-\infty$ logits to eliminate random uninitialized embedding noise.
   - Negative candidate sampling is not used during test evaluation; all eligible candidate items are scored.

2. **Pre-Query Seen-Item Filtering**:
   - All items previously interacted with in the sequence history receive $-\infty$ logits before top-$K$ selection.

3. **Deterministic Tie-Breaking**:
   - In deterministic rankings where models output identical scores, ties are broken strictly by ascending canonical `item_id`.

4. **No Test-Time Popularity or Heuristics**:
   - Popularity debiasing ($\alpha = 0.25$) is applied solely as an inverse-frequency regularization term in the cross-entropy training loss.
   - At inference/test time, ranking is driven 100% by pure neural dot products and contextual attention scores.

---

## Methodological Limitations

1. **Protocol v4 Causal Boundary:** Offline query timestamps are strictly tied to interaction $T-1$, with the target at interaction $T$ withheld from model inputs. Cold-start interactions with sequence length $< 5$ are excluded from the test benchmark per protocol definition.
2. **Deterministic Tie-Breaking:** Ties are broken strictly by ascending canonical `item_id`.
3. **Hardware-Specific Serving Latencies:** Reported serving latencies (P95 of 4.80 ms, throughput of 16,271 QPS) and peak VRAM (368.5 MB) are measured on an NVIDIA GeForce RTX 5080 with Tensor-Core candidate tiling (chunk size 8,192, bfloat16 precision). Latency profiles will scale with GPU memory bandwidth and compute architecture.

Source data is derived from the public GroupLens MovieLens-32M release. Canonical model checkpoints are versioned under artifacts/.
