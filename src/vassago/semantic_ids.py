"""Residual quantization and valid-prefix autoregressive retrieval."""

from abc import ABC, abstractmethod
from collections import Counter
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GeneratedItem(TypedDict):
    movie_id: int
    log_probability: float
    beam_rank: int
    prefix: list[int]


class ResidualQuantizer(nn.Module):
    def __init__(self, input_dim: int, dimension: int, levels: int, size: int) -> None:
        super().__init__()
        self.levels, self.size = levels, size
        self.encoder = nn.Linear(input_dim, dimension)
        self.decoder = nn.Linear(dimension, input_dim)
        self.codebooks = nn.Parameter(torch.randn(levels, size, dimension) * 0.1)

    @torch.no_grad()
    def initialize(self, vectors: Tensor, iterations: int = 20) -> None:
        """Initialize the bottleneck and residual codebooks from the catalog.

        A data-aware initialization matters for small catalogs: random codebooks can
        assign every item to only a handful of codes before reconstruction training
        has a useful gradient. PCA preserves the semantic geometry and deterministic
        farthest-first k-means gives every level a well-spread starting vocabulary.
        """
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("Tokenizer initialization requires a non-empty matrix")
        if not torch.isfinite(vectors).all():
            raise ValueError("Tokenizer initialization vectors must be finite")
        device, dtype = vectors.device, vectors.dtype
        centered = vectors - vectors.mean(0, keepdim=True)
        _, _, right = torch.linalg.svd(centered.float().cpu(), full_matrices=False)
        basis = right[: self.encoder.out_features].to(device=device, dtype=dtype)
        if len(basis) < self.encoder.out_features:
            basis = F.pad(basis, (0, 0, 0, self.encoder.out_features - len(basis)))
        self.encoder.weight.copy_(basis)
        self.encoder.bias.copy_(-(vectors.mean(0) @ basis.T))
        self.decoder.weight.copy_(basis.T)
        self.decoder.bias.copy_(vectors.mean(0))

        residual = self.encoder(vectors)
        for level in range(self.levels):
            centroids = self._kmeans(residual, iterations)
            self.codebooks[level].copy_(centroids)
            assignments = torch.cdist(residual.float(), centroids.float()).argmin(-1)
            residual = residual - centroids[assignments]

    def _kmeans(self, vectors: Tensor, iterations: int) -> Tensor:
        count = min(self.size, len(vectors))
        chosen = [int(vectors.square().sum(-1).argmax())]
        distance = (vectors - vectors[chosen[0]]).square().sum(-1)
        for _ in range(1, count):
            index = int(distance.argmax())
            chosen.append(index)
            distance = torch.minimum(distance, (vectors - vectors[index]).square().sum(-1))
        centroids = vectors[chosen].clone()
        if count < self.size:
            centroids = torch.cat([centroids, centroids[:1].repeat(self.size - count, 1)])
        for _ in range(iterations):
            assignments = torch.cdist(vectors.float(), centroids.float()).argmin(-1)
            updated = centroids.clone()
            for code in range(count):
                members = vectors[assignments == code]
                if len(members):
                    updated[code] = members.mean(0)
            if torch.allclose(updated, centroids, atol=1e-5, rtol=1e-4):
                break
            centroids = updated
        return centroids

    def forward(
        self,
        vectors: Tensor,
        differentiable: bool = True,
        temperature: float = 1,
        exploration: float = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = self.encoder(vectors)
        quantized = torch.zeros_like(residual)
        codes, entropies = [], []
        for book in self.codebooks:
            logits = -(
                residual.square().sum(-1, keepdim=True)
                + book.square().sum(-1)
                - 2 * residual @ book.T
            )
            if self.training and exploration > 0:
                uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
                logits = logits - exploration * (-uniform.log()).log()
            soft = (logits / max(temperature, 0.05)).softmax(-1)
            code = logits.argmax(-1)
            hard = F.one_hot(code, self.size).to(soft.dtype)
            assignment = hard + soft - soft.detach() if differentiable else hard
            selected = assignment @ book
            quantized = quantized + selected
            residual = residual - selected
            codes.append(code)
            usage = soft.reshape(-1, self.size).mean(0).clamp_min(1e-9)
            entropies.append((usage * usage.log()).sum())
        return quantized, torch.stack(codes, -1), torch.stack(entropies).mean()

    def reconstruction_loss(self, vectors: Tensor, exploration: float = 0) -> Tensor:
        z, _, negative_entropy = self(vectors, temperature=0.25, exploration=exploration)
        return F.mse_loss(self.decoder(z), vectors) + 0.05 * negative_entropy

    @torch.no_grad()
    def diagnostics(self, vectors: Tensor) -> dict[str, object]:
        was_training = self.training
        self.eval()
        z, codes, _ = self(vectors)
        counts = [
            torch.bincount(codes[:, i], minlength=self.size).float() for i in range(self.levels)
        ]
        usage = [c / c.sum().clamp_min(1) for c in counts]
        collisions = Counter(map(tuple, codes.cpu().tolist()))
        result: dict[str, object] = {
            "reconstruction_mse": F.mse_loss(self.decoder(z), vectors).item(),
            "utilization": [(c > 0).float().mean().item() for c in counts],
            "dead_codes": [(c == 0).sum().item() for c in counts],
            "perplexity": [(-(p * p.clamp_min(1e-9).log()).sum()).exp().item() for p in usage],
            "colliding_items": sum(n for n in collisions.values() if n > 1),
            "unique_codes": len(collisions),
        }
        self.train(was_training)
        return result


class GenerativeDecoder(nn.Module, ABC):
    @abstractmethod
    def generate(
        self, context: Tensor, codes: Tensor, beam_width: int, k: int
    ) -> list[GeneratedItem]:
        """Return only canonical IDs belonging to valid terminal code sequences."""


class AutoregressiveSIDDecoder(GenerativeDecoder):
    def __init__(self, dimension: int, levels: int, size: int) -> None:
        super().__init__()
        self.levels, self.size = levels, size
        self.tokens = nn.Embedding(levels * size + 1, dimension)
        self.cell = nn.GRUCell(dimension, dimension)
        self.output = nn.Linear(dimension, size)
        self._indexed_codes: Tensor | None = None
        self._trie: dict[tuple[int, ...], set[int]] = {}
        self._terminals: dict[tuple[int, ...], list[int]] = {}

    def logits(self, context: Tensor, target_codes: Tensor) -> Tensor:
        state = context
        token = torch.full(
            (len(context),), self.levels * self.size, device=context.device, dtype=torch.long
        )
        outputs = []
        for level in range(self.levels):
            state = self.cell(self.tokens(token), state)
            outputs.append(self.output(state))
            token = target_codes[:, level] + level * self.size
        return torch.stack(outputs, 1)

    @torch.no_grad()
    def generate(
        self, context: Tensor, codes: Tensor, beam_width: int, k: int
    ) -> list[GeneratedItem]:
        if beam_width < 1 or k < 1:
            raise ValueError("Beam width and k must be positive")
        if self._indexed_codes is not codes:
            self._trie, self._terminals = {}, {}
            for item, code in enumerate(codes.cpu().tolist(), start=1):
                sequence = tuple(code)
                self._terminals.setdefault(sequence, []).append(item)
                for depth in range(self.levels):
                    self._trie.setdefault(sequence[:depth], set()).add(sequence[depth])
            self._indexed_codes = codes
        beams: list[tuple[tuple[int, ...], float, Tensor]] = [((), 0.0, context.reshape(1, -1))]
        for depth in range(self.levels):
            expanded = []
            if not beams:
                break
            previous = [
                self.levels * self.size if not prefix else prefix[-1] + (depth - 1) * self.size
                for prefix, _, _ in beams
            ]
            token = torch.tensor(previous, device=context.device)
            states = self.cell(self.tokens(token), torch.cat([state for _, _, state in beams]))
            log_prob = self.output(states).log_softmax(-1).cpu().tolist()
            for index, (prefix, score, _) in enumerate(beams):
                for token_id in sorted(self._trie.get(prefix, set())):
                    expanded.append(
                        (
                            (*prefix, token_id),
                            score + log_prob[index][token_id],
                            states[index : index + 1],
                        )
                    )
            beams = sorted(expanded, key=lambda entry: (-entry[1], entry[0]))[:beam_width]
        output: list[GeneratedItem] = []
        for rank, (prefix, score, _) in enumerate(beams, start=1):
            for movie_id in self._terminals[prefix]:
                output.append(
                    {
                        "movie_id": movie_id,
                        "log_probability": score,
                        "beam_rank": rank,
                        "prefix": list(prefix),
                    }
                )
        return output[:k]


class SIDExpert(nn.Module):
    embeddings: Tensor

    def __init__(self, embeddings: Tensor, dimension: int, levels: int, size: int) -> None:
        super().__init__()
        self.register_buffer("embeddings", embeddings)
        self.tokenizer = ResidualQuantizer(embeddings.shape[1], dimension, levels, size)
        self.history_encoder = nn.GRU(dimension, dimension, batch_first=True)
        self.decoder = AutoregressiveSIDDecoder(dimension, levels, size)

    def context(
        self, history: Tensor, differentiable: bool = True, exploration: float = 0
    ) -> Tensor:
        z, _, _ = self.tokenizer(
            self.embeddings[history], differentiable=differentiable, exploration=exploration
        )
        valid = history.ne(0)
        z = z * valid.unsqueeze(-1)
        encoded, _ = self.history_encoder(z)
        length = valid.sum(1)
        return encoded[
            torch.arange(len(history), device=history.device), (length - 1).clamp_min(0)
        ] * length.gt(0).unsqueeze(-1)

    def recommendation_loss(
        self,
        history: Tensor,
        targets: Tensor,
        differentiable: bool,
        exploration: float = 0,
        target_tokenizer: ResidualQuantizer | None = None,
    ) -> Tensor:
        context = self.context(history, differentiable, exploration)
        with torch.no_grad():
            tokenizer = target_tokenizer if target_tokenizer is not None else self.tokenizer
            _, codes, _ = tokenizer(self.embeddings[targets], exploration=0)
        logits = self.decoder.logits(context, codes)
        return F.cross_entropy(logits.flatten(0, 1), codes.flatten())

    @torch.no_grad()
    def codes(self) -> Tensor:
        return self.tokenizer(self.embeddings[1:], exploration=0)[1]
