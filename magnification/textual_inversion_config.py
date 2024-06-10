import os
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
    eval_dir: Path
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
    num_inference_steps: int

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class BPTTConfig:
    # based on AlignProp config.train
    # optimizer
    learning_rate: float

    # flags
    use_lora: bool
    grad_checkpoint: bool

    total_batch_size: int
    total_samples_per_epoch: int
    num_gpus: int

    # optional
    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.99
    adam_weight_decay: Optional[float] = 1e-2
    adam_epsilon: Optional[float] = 1e-8

    samples_per_epoch_per_gpu: Optional[int] = 32
    batch_size_per_gpu: Optional[int] = 32
    batch_size_per_gpu_available: Optional[int] = 1
    gradient_accumulation_steps: Optional[int] = 32
    data_loader_iterations: Optional[int] = 32
    per_gpu_capacity: Optional[int] = 1

    def __post_init__(self):
        self.samples_per_epoch_per_gpu = self.total_samples_per_epoch // self.num_gpus
        self.batch_size_per_gpu = self.total_batch_size // self.num_gpus
        self.batch_size_per_gpu_available = self.per_gpu_capacity
        self.gradient_accumulation_steps = (
            self.batch_size_per_gpu // self.batch_size_per_gpu_available
        )
        self.data_loader_iterations = (
            self.samples_per_epoch_per_gpu // self.batch_size_per_gpu_available
        )

        assert (
            self.total_samples_per_epoch % self.num_gpus == 0
        ), "total_samples_per_epoch must be divisible by num_gpus"
        assert (
            self.total_batch_size % self.num_gpus == 0
        ), "total_batch_size must be divisible by num_gpus"
        assert (
            self.batch_size_per_gpu % self.batch_size_per_gpu_available == 0
        ), "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
        assert (
            self.samples_per_epoch_per_gpu % self.batch_size_per_gpu_available == 0
        ), "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"


@dataclass
class InstructInversionBPTTConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    train: BPTTConfig
    log_dir: Path
    run_name: str
    num_checkpoint_limit: int
    mixed_precision: str
    epochs: int
    num_inference_steps: int
    device: int
    allow_tf32: bool
    debug: Optional[bool] = False

    def __post_init__(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)
