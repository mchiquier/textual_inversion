import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        with torch.no_grad():
            # Smooth the labels
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (logits.size(1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        # Calculate the cross-entropy loss with the smoothed labels
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(logits, dim=-1), dim=-1))


def pairwise_mse(embeddings: list[torch.Tensor]):
    """ Compute pairwise MSE for a list of embeddings. """
    num_embeddings = len(embeddings)
    mse_loss = nn.MSELoss(reduction='none')
    total_mse = 0.0
    count = 0

    # Calculate pairwise MSE
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            total_mse += mse_loss(embeddings[i], embeddings[j]).mean()
            count += 1

    # Average the MSE
    if count > 0:
        average_mse = total_mse / count
    else:
        average_mse = torch.tensor(0.0, requires_grad=True)

    return average_mse