"""Filter-normalized random-direction loss landscape (Li et al., "Visualizing the
Loss Landscape of Neural Nets"). Not part of the invexapi package - a plain
examples-only helper, same as ``_data.py``.
"""

import torch


def _random_direction(model):
    return [torch.randn_like(p) for p in model.parameters()]


def _filter_normalize(direction, params):
    """Scale each filter of ``direction`` to match the norm of the corresponding
    filter in ``params`` (per output-channel/row slice). 1D params (biases) are
    zeroed out - they contribute negligibly and skew the normalization."""
    normalized = []
    for d, p in zip(direction, params):
        if p.dim() <= 1:
            normalized.append(torch.zeros_like(d))
            continue
        d_flat = d.view(d.shape[0], -1)
        p_flat = p.view(p.shape[0], -1)
        d_norms = d_flat.norm(dim=1, keepdim=True)
        p_norms = p_flat.norm(dim=1, keepdim=True)
        scale = p_norms / (d_norms + 1e-10)
        normalized.append((d_flat * scale).view_as(d))
    return normalized


def random_directions(model):
    """Two filter-normalized random directions in parameter space."""
    params = list(model.parameters())
    d1 = _filter_normalize(_random_direction(model), params)
    d2 = _filter_normalize(_random_direction(model), params)
    return d1, d2


@torch.no_grad()
def loss_surface(model, compute_loss, d1, d2, alphas, betas):
    """Evaluate ``compute_loss()`` on a grid of ``theta0 + alpha*d1 + beta*d2``.

    ``compute_loss`` is a zero-arg callable reading the model's current
    parameters (e.g. ``lambda: loss_fn(model(x), y).item()``). Restores the
    model's original parameters before returning.
    """
    base_params = [p.detach().clone() for p in model.parameters()]
    surface = torch.zeros(len(alphas), len(betas))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for p, p0, dd1, dd2 in zip(model.parameters(), base_params, d1, d2):
                p.copy_(p0 + a * dd1 + b * dd2)
            surface[i, j] = compute_loss()

    for p, p0 in zip(model.parameters(), base_params):
        p.copy_(p0)

    return surface
