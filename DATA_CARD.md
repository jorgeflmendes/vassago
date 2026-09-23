# Data card

The general pipeline uses the stable MovieLens 32M release, not MovieLens Latest.
Ratings are 0.5–5, with configurable positive threshold 4 by default. Preserve
all source ratings in Parquet; chronological examples select positives. Source
user identifiers are namespaced. Canonical internal item IDs are contiguous;
MovieLens/TMDB/IMDb identifiers stay in the catalog and crosswalk report.

Five global time windows also preserve chronology within every user. Ties do not
cross boundaries. Train-only counts define collaborative exposure and cohorts.
Seeded item holdout removes the selected items' training interactions and graph
edges. Evaluation remains prequential with frozen weights. Availability and
stacking assumptions are recorded in the run manifests and
`REPRODUCIBILITY.md`.

Default MovieLens metadata contains title and genres. Earliest observed interaction
is a conservative availability bound when exact release dates are absent. This
does not identify theatrical versus regional/catalog release. Current rich metadata
and undated Tag Genome values are not historical snapshots. Their use must be
explicitly marked retrospective, with availability timestamps supplied by the user.

Synthetic data has invented titles, directors and events generated from a fixed
seed. It is intentionally small and not representative of real preference quality.
The benchmark documented in the README is a separate MovieLens 1M leave-one-out
development protocol. It keeps the latest 200 events as model input and excludes the
complete pre-query history from recommendations. Protocol v3 uses the last observed
event as the offline query time; the held-out event timestamp is audit metadata and is
never passed to the model.

MovieLens 100K is reserved for the registered external comparison. Its archive must
match MD5 `0e33842e24a9c977be4e0107933c0723`, and its protocol hash is fixed in
`configs/experiment/ml100k_external_benchmark.json` before test metrics are aggregated.
The resulting comparison uses all 943 users, the 1,682-item catalog, seeds 42–46, and
one frozen configuration for each model.

### Methodological Limitations and Proxy Biases

1. **Cold-start Onboarding Proxy:** The evaluation in `scripts/evaluate_profile_cold_start.py` constructs a profile proxy by querying the 3 most frequent genres across a user's entire pre-query history. This is an exploratory heuristic rather than a true cold-start evaluation, as it condenses years of established user preferences (marginal preference distribution) that would not be available for an unobserved, freshly onboarded user.
2. **Deterministic Tie-Breaking:** In deterministic rankings where models output identical scores (e.g., discrete popularity or coarse profiles), ties are broken strictly by ascending canonical `item_id`. Users must account for potential catalog-ordering biases when comparing models with high score degeneracy.
3. **Cross-Seed Bootstrap Interpretation:** The paired crossed multiplier bootstrap (`paired_seed_bootstrap`) over seeds 42–46 evaluates user variance conditioned on these specific runs; with 5 seeds, it does not approximate an asymptotic distribution across all stochastic training runs.

No copyrighted source data or trained artifact belongs in Git.
