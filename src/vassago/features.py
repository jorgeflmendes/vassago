"""Canonical metadata documents and revision-pinned cached text encoders."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from vassago.data import Movie, check_metadata


def movie_text(movie: Movie, cutoff: int) -> str:
    check_metadata(movie.metadata)
    if movie.metadata_available_at > cutoff:
        return ""
    fields = {"title": movie.title, "genres": movie.genres, **movie.metadata}
    return "\n".join(
        f"{key.replace('_', ' ').title()}: "
        + (", ".join(map(str, value)) if isinstance(value, list) else str(value))
        for key, value in sorted(fields.items())
        if value
    )


class TextEncoder:
    def __init__(self, name: str, dimension: int, revision: str | None = None) -> None:
        self.name, self.dimension, self.revision = name, dimension, revision

    def encode(self, documents: list[str], batch_size: int = 64) -> np.ndarray:
        if self.name != "hash":
            from sentence_transformers import SentenceTransformer

            if not self.revision:
                raise ValueError("A pinned model revision is required")
            model = SentenceTransformer(self.name, revision=self.revision, trust_remote_code=False)
            return np.asarray(
                model.encode(
                    documents,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        output = np.zeros((len(documents), self.dimension), dtype=np.float32)
        for i, document in enumerate(documents):
            for token in re.findall(r"\w+", document.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                output[i, int.from_bytes(digest[:4], "little") % self.dimension] += (
                    1 if digest[4] % 2 else -1
                )
        return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)

    def cached(self, documents: list[str], directory: Path) -> np.ndarray:
        manifest = {
            "encoder": self.name,
            "revision": self.revision,
            "dimension": self.dimension,
            "document_hash": hashlib.sha256(json.dumps(documents).encode()).hexdigest(),
        }
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{key}.npy"
        if not destination.exists():
            np.save(destination, self.encode(documents), allow_pickle=False)
            (directory / f"{key}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return np.load(destination, mmap_mode="r", allow_pickle=False)
