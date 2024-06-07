import torch
from PIL import Image
from pathlib import Path
from torchvision.utils import make_grid
import numpy as np


def plot_grid(sample: torch.Tensor, save_path: Path):
    sample = sample.detach().cpu()
    grid = make_grid(sample, nrow=4)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    im = Image.fromarray(grid)
    im.save(save_path)