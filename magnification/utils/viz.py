import torch
from PIL import Image
from pathlib import Path
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


def plot_grid(sample: torch.Tensor, save_path: Path, nrow: int = 4):
    sample = sample.detach().cpu()
    grid = make_grid(sample, nrow=nrow)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    im = Image.fromarray(grid)
    im.save(save_path)


def plot_logits_and_predictions(
    logits: torch.Tensor, predictions: torch.Tensor, save_path: Path
):
    num_samples = logits.shape[0]
    num_classes = logits.shape[1]

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        logit_values = logits[i].detach().cpu().numpy()
        prediction = predictions[i].detach().cpu().numpy()

        # Plot logits
        ax.bar(
            np.arange(num_classes),
            logit_values,
            color="blue",
            alpha=0.6,
            label="Logits",
        )

        # Highlight the predicted class
        ax.bar(
            prediction,
            logit_values[prediction],
            color="red",
            alpha=0.6,
            label="Predicted Class",
        )

        ax.set_xticks(np.arange(num_classes))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Logits")
        ax.set_title(f"Sample {i + 1} - Prediction: Class {prediction}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
