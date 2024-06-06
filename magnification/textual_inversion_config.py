from typing import Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EmbeddingManagerConfig:
    placeholder_strings: list
    initializer_words: list
    per_image_tokens: bool
    num_vectors_per_token: int
    progressive_words: bool


@dataclass
class DatasetConfig:
    image_dir: Path
    image_edits_dir: Path
    repeats: int
    img_size: Optional[int] = 256


@dataclass
class LDMConfig:
    embedding_config: EmbeddingManagerConfig
    conditioning_dropout_prob: Optional[float] = None


@dataclass
class TextualInversionConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    batch_size: int
    epochs: int
    learning_rate: float
    device: int
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

