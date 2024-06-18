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
    return im


def plot_logits_and_predictions(
    logits: torch.Tensor, probs: torch.Tensor, save_path: Path, max_samples: int = 8
):
    num_samples = min(logits.shape[0], max_samples)
    num_classes = logits.shape[1]

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        try:
            logit_values = logits[i].detach().cpu().numpy()
            prob_values = probs.detach().cpu().numpy()[i]
        except:
            print(1)

        prediction = np.argmax(prob_values)

        # Plot logits
        ax[0].bar(
            np.arange(num_classes),
            logit_values,
            color="blue",
            alpha=0.6,
            label="Logits",
        )
        ax[0].set_xticks(np.arange(num_classes))
        ax[0].set_xlabel("Classes")
        ax[0].set_ylabel("Logits")
        ax[0].set_title(f"Sample {i + 1} - Prediction: Class {prediction}")
        ax[0].legend()

        # Highlight the predicted class
        ax[1].bar(
            np.arange(num_classes),
            prob_values,
            color="red",
            alpha=0.6,
            label="Probs",
        )
        ax[1].set_xticks(np.arange(num_classes))
        ax[1].set_xlabel("Classes")
        ax[1].set_ylabel("Probs")
        ax[1].set_title(f"Sample {i + 1} - Prediction: Class {prediction}")
        ax[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return fig
