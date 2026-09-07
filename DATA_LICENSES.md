# Data sources and usage constraints

Checked 2026-09-07. This table records source policies, not a sublicense. Consult
the linked source terms for the exact intended use before acquiring data.

| Source / snapshot | Expected files | Redistribution / commercial use | Attribution and source terms |
|---|---|---|---|
| MovieLens 32M, stable release | ratings.csv, tags.csv, movies.csv, links.csv | No bundled redistribution; commercial/revenue-bearing use requires GroupLens permission under its README | [GroupLens release](https://grouplens.org/datasets/movielens/32m/); cite Harper and Konstan |
| MovieLens 100K, stable development release | u.data, u.item | Same release-specific GroupLens restrictions; user-triggered download only | [100K](https://grouplens.org/datasets/movielens/100k/) |
| Tag Genome 2021 | source score tables and tag dictionary | Preserve source terms; do not redistribute with this repository; no commercial grant inferred | [2021 source](https://grouplens.org/datasets/movielens/tag-genome-2021/); cite its supplied publication |
| MovieLens Beliefs 2024 | source CSV tables | Separate experiment; follow supplied research terms; no commercial grant inferred | [Beliefs](https://grouplens.org/datasets/movielens/ml_belief_2024/) |
| TMDB, user timestamped API snapshot | metadata JSON cache | Free API for non-commercial attributed use; commercial use requires agreement; no posters downloaded | [API FAQ](https://developer.themoviedb.org/docs/faq). Required notice: This product uses the TMDB API but is not endorsed or certified by TMDB. Follow logo attribution rules in any public UI. |
| IMDb, daily user download | title.basics, title.crew, title.principals, title.ratings, name.basics, title.akas TSV.gz | Official non-commercial datasets; do not redistribute raw files; daily ratings are excluded from historical features | [Official datasets and terms](https://data.imdb.com/non-commercial-datasets/) |
| Ultimate 1Million Movies, user-supplied Kaggle snapshot | CSV with exact TMDB id | Convenience only; uploader license does not override TMDB/IMDb restrictions; no commercial grant inferred | Record exact Kaggle version, checksum and upstream attribution |
| Amazon Reviews 2023 Movies_and_TV | review JSONL(.gz) | Separate product-domain research; check source license before use; not distributed here | [McAuley Lab](https://amazon-reviews-2023.github.io/); preserve ASIN/product identities |
| Synthetic fixture | generated in code | PolyForm Strict with repository code | Cite software version if used to reproduce tests |

Downloads record checksums where the downloader supports manifests. A locally
computed SHA256 pins downloaded bytes; it is not an upstream authenticity signature.
Do not treat metadata retrieved today as known in historical recommendation windows.
