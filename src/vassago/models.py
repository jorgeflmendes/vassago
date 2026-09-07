"""Compact independently parameterized sequential and collaborative models."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SASRec(nn.Module):
    """Causal, right-padded SASRec variant with next-item softmax training."""

    def __init__(
        self,
        n_items: int,
        dimension: int,
        max_length: int,
        heads: int = 2,
        layers: int = 1,
        dropout: float = 0.1,
        semantic: Tensor | None = None,
        finetune: bool = False,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        if semantic is None:
            self.items = nn.Embedding(n_items + 1, dimension, padding_idx=0)
            self.projection: nn.Module = nn.Identity()
        else:
            self.items = nn.Embedding.from_pretrained(semantic, freeze=not finetune, padding_idx=0)
            self.projection = nn.Linear(semantic.shape[1], dimension, bias=False)
        self.positions = nn.Embedding(max_length, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension, heads, dimension * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def item_vectors(self) -> Tensor:
        return F.normalize(self.projection(self.items.weight), dim=-1)

    def sequence_states(self, history: Tensor) -> Tensor:
        """Return a normalized causal representation at every non-padding position."""
        valid = history.ne(0)
        # Empty rows expose a zero sentinel key to prevent all-masked attention NaNs.
        key_padding = ~valid.clone()
        key_padding[:, 0] = False
        positions = torch.arange(history.shape[1], device=history.device)
        x = self.projection(self.items(history)) + self.positions(positions)
        x = self.dropout(x) * valid.unsqueeze(-1)
        causal = torch.ones(
            history.shape[1], history.shape[1], device=history.device, dtype=torch.bool
        ).triu(1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=key_padding)
        return F.normalize(self.norm(x), dim=-1) * valid.unsqueeze(-1)

    def forward(self, history: Tensor) -> Tensor:
        valid = history.ne(0)
        length = valid.sum(1)
        states = self.sequence_states(history)
        return states[
            torch.arange(len(history), device=history.device), (length - 1).clamp_min(0)
        ] * length.gt(0).unsqueeze(-1)

    def score(self, history: Tensor) -> Tensor:
        return self(history) @ self.item_vectors().T


class BPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, dimension: int) -> None:
        super().__init__()
        self.users = nn.Embedding(n_users + 1, dimension, padding_idx=0)
        self.items = nn.Embedding(n_items + 1, dimension, padding_idx=0)
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)
        with torch.no_grad():
            self.users.weight[0].zero_()
            self.items.weight[0].zero_()

    def representations(self) -> tuple[Tensor, Tensor]:
        return self.users.weight, self.items.weight

    def loss(self, users: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        u, i = self.representations()
        return -F.logsigmoid((u[users] * (i[positive] - i[negative])).sum(-1)).mean()

    def score_users(self, users: Tensor) -> Tensor:
        u, i = self.representations()
        return u[users] @ i.T


class LightGCN(BPR):
    """Degree-normalized bipartite propagation, no nonlinearities or feature transforms."""

    def __init__(
        self, n_users: int, n_items: int, dimension: int, edges: Tensor, layers: int = 2
    ) -> None:
        super().__init__(n_users, n_items, dimension)
        self.layers = layers
        self.n_users = n_users + 1
        total = n_users + n_items + 2
        src, dst = edges[0], edges[1] + self.n_users
        indices = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        degree = torch.bincount(indices[0], minlength=total).float().clamp_min(1)
        values = (degree[indices[0]] * degree[indices[1]]).rsqrt()
        self.register_buffer(
            "adjacency",
            torch.sparse_coo_tensor(
                indices, values, (total, total), check_invariants=True
            ).coalesce(),
        )

    def representations(self) -> tuple[Tensor, Tensor]:
        x = torch.cat([self.users.weight, self.items.weight])
        layers = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adjacency, x)
            layers.append(x)
        output = torch.stack(layers).mean(0)
        return output[: self.n_users], output[self.n_users :]
