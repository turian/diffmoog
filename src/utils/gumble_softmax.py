import torch
import torch.nn.functional as F


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False, device='cuda:0'):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """

    assert logits.ndim == 2, f"Expected 2D logits, but got {logits.ndim}D"
    assert logits.shape[1] > 1, f"Expected logits to have more than one class, but got {logits.shape[1]} classes"

    if not hard:
        y = gumbel_softmax_sample(logits, temperature, device)
        return y

    shape = logits.size()
    print(f"Shape of logits before Gumbel-Softmax: {logits.shape}")

    _, ind = logits.max(dim=-1)
    print(f"Max indices from logits: {ind.shape}, max value indices: {ind}")

    y_hard = torch.zeros_like(logits).view(-1, shape[-1])
    print(f"Shape of y_hard after scatter operation: {y_hard.shape}")

    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    print(f"Shape of y_hard after reshaping: {y_hard.shape}")

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - logits).detach() + logits
    return y_hard
