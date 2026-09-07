# Data card

The primary source is the stable MovieLens 32M release, not MovieLens Latest.
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
No copyrighted source data or trained artifact belongs in Git.
