"""Validated, serializable experiment configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentConfig(BaseModel):
    contextual_dimension: int = Field(default=32, ge=4)
    contextual_epochs: int = Field(default=30, ge=1)
    contextual_joint_epochs: int = Field(default=0, ge=0)
    contextual_joint_backbone_lr_scale: float = Field(default=0.1, gt=0, le=1)
    contextual_memory_window: int = Field(default=32, ge=1)
    contextual_positions_per_user: int = Field(default=4, ge=1)
    contextual_groups: int = Field(default=8, ge=1)
    contextual_hard_negatives: bool = False
    contextual_heads: int = Field(default=1, ge=1, le=8)
    contextual_temperature: float = Field(default=0.1, gt=0)
    model_config = ConfigDict(extra="forbid")
    dataset: str = "synthetic"
    seed: int = 42
    dimension: int = Field(default=32, ge=8)
    heads: int = Field(default=2, ge=1)
    layers: int = Field(default=1, ge=1)
    max_length: int = Field(default=20, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    epochs: int = Field(default=2, ge=1)
    stacker_epochs: int = Field(default=5, ge=1)
    tokenizer_epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=1)
    sampled_negatives: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=0.003, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    codebooks: int = Field(default=2, ge=1)
    codebook_size: int = Field(default=8, ge=2)
    beam_width: int = Field(default=20, ge=1)
    candidate_k: int = Field(default=30, ge=1)
    popularity_k: int = Field(default=10, ge=0)
    positive_threshold: float = Field(default=4, ge=0.5, le=5)
    rating_weighted: bool = False
    cold_fraction: float = Field(default=0.15, ge=0, lt=0.5)
    encoder: str = "hash"
    encoder_revision: str | None = None
    finetune_semantic: bool = False
    device: str = "cpu"
    mixed_precision: bool = False
    calibration: Literal["rank", "zscore", "temperature"] = "rank"
    split_quantiles: tuple[float, float, float, float] = (0.55, 0.65, 0.78, 0.88)
    cutoffs: tuple[int, int, int, int] | None = None
    evaluation_limit: int | None = Field(default=None, ge=1)
    mmr_lambda: float = Field(default=0.8, ge=0, le=1)
    refinement_weight: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def validate_structure(self) -> "ExperimentConfig":
        if self.dimension % self.heads:
            raise ValueError("dimension must be divisible by heads")
        if not all(0 < x < 1 for x in self.split_quantiles):
            raise ValueError("split quantiles must lie strictly between zero and one")
        if sorted(set(self.split_quantiles)) != list(self.split_quantiles):
            raise ValueError("split quantiles must be strictly increasing")
        if self.cutoffs and sorted(set(self.cutoffs)) != list(self.cutoffs):
            raise ValueError("cutoffs must be strictly increasing")
        if self.encoder != "hash" and not self.encoder_revision:
            raise ValueError("pin encoder_revision for reproducible pretrained embeddings")
        if self.contextual_groups > self.contextual_memory_window:
            raise ValueError("contextual_groups cannot exceed contextual_memory_window")
        return self

    @classmethod
    def read(cls, path: Path) -> "ExperimentConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
