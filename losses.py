import torch


def kl_dirichlet(alpha, beta):
    """KL(Dir(alpha) || Dir(beta))."""
    S_a = alpha.sum(-1, keepdim=True)
    S_b = beta.sum(-1, keepdim=True)
    kl = (torch.lgamma(S_a) - torch.lgamma(S_b)
          - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(-1, keepdim=True)
          + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_a))).sum(-1, keepdim=True))
    return kl.squeeze(-1)


def edl_loss(alpha, labels_onehot, epoch, total_epochs, weights=None, kl_coef=0.01):
    """
    EDL loss: Type II ML (digamma) + KL regularization with annealing.
    alpha: Dirichlet params (batch, K), all >= 1
    labels_onehot: One-hot labels (batch, K)
    kl_coef: Base KL coefficient (default 0.01)
    """
    S = alpha.sum(-1, keepdim=True)
    loss_fit = (labels_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
    if weights is not None:
        w = weights[labels_onehot.argmax(-1)]
        loss_fit = loss_fit * w
    alpha_tilde = labels_onehot * (alpha - 1) + 1
    kl = kl_dirichlet(alpha_tilde, torch.ones_like(alpha))
    lambda_t = min(1.0, epoch / (0.5 * total_epochs))
    return (loss_fit + kl_coef * lambda_t * kl).mean()
