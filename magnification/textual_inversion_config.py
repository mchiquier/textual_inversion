import os
from typing import Optional, Union
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
    repeats: Optional[int] = 1
    img_size: Optional[int] = 256
    subset: Optional[int] = None


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
    lr_text_embed: float
    lr_lora: float

    # flags
    use_lora: bool
    grad_checkpoint: bool
    truncated_backprop: bool

    batch_size: int # per gpu
    num_gpus: int

    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.99
    adam_weight_decay: Optional[float] = 1e-2
    adam_epsilon: Optional[float] = 1e-8

    truncated_backprop_minmax: Union[tuple, list] = (35, 45)

    samples_per_epoch_per_gpu: Optional[int] = None # per gpu
    total_batch_size: Optional[int] = None
    total_samples_per_epoch: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 32

    def __post_init__(self):
        self.total_batch_size = self.batch_size * self.num_gpus
        if self.samples_per_epoch_per_gpu is not None:
            self.total_samples_per_epoch = self.samples_per_epoch_per_gpu * self.num_gpus


@dataclass
class InstructInversionBPTTConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    train: BPTTConfig
    log_dir: Path
    mixed_precision: str
    epochs: int
    num_inference_steps: int
    device: Optional[int] = 0
    allow_tf32: Optional[bool] = True
    guidance_scale: Optional[float] = 7.5
    image_guidance_scale: Optional[float] = 1.5
    debug: Optional[bool] = False

    def __post_init__(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class EvalConfig:
    diffusion: LDMConfig
    dataset: DatasetConfig
    ckpt_path: Path
    output_dir: Path
    run_name: str
    batch_size: int
    num_inference_steps: int
    device: int

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)