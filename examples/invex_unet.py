"""A standard UNet encoder-decoder, built from ``ixConv2d`` instead of ``nn.Conv2d`` -
plus ``NormalUNet``, a genuine plain-UNet baseline with the *same* channel widths.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from invexapi.nn import ixConv2d
from invexapi.certificate import Certificate
from invexapi.nn.base import LAYER_REFERENCE


class _ixDoubleConv(nn.Module):
    """``ixConv2d`` -> activation -> + learnable per-channel bias.

    ``act(A2(act(A1(x)))) + bias``. Certified invex per the added bias
    (a constant shift is a full-row-rank affine map, so composing it after the
    already-invex ``ixConv2d``+activation stays invex).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, activation, **conv_kwargs):
        super().__init__()
        self.block = ixConv2d(
            in_channels, hidden_channels, out_channels, kernel_size, activation=activation, **conv_kwargs
        )
        self.activation = activation
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
        self._invex = Certificate(status="assumed", source="manual", reference=LAYER_REFERENCE)

    @property
    def invex(self):
        return self._invex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x)) + self.bias


class InvexUNet(nn.Module):
    """3-level encoder/decoder UNet; every block is a ``_ixDoubleConv``."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        activation: nn.Module = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.Softplus()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.down1 = _ixDoubleConv(in_channels, c1, c1, kernel_size=3, activation=activation, padding=1)
        self.down2 = _ixDoubleConv(c1, c2, c2, kernel_size=3, activation=activation, padding=1)
        self.down3 = _ixDoubleConv(c2, c3, c3, kernel_size=3, activation=activation, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _ixDoubleConv(c3, c4, c4, kernel_size=3, activation=activation, padding=1)

        self.up3 = _ixDoubleConv(c4 + c3, c3, c3, kernel_size=3, activation=activation, padding=1)
        self.up2 = _ixDoubleConv(c3 + c2, c2, c2, kernel_size=3, activation=activation, padding=1)
        self.up1 = _ixDoubleConv(c2 + c1, c1, c1, kernel_size=3, activation=activation, padding=1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _upsample_and_concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)
        s2 = self.down2(self.pool(s1))
        s3 = self.down3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        u3 = self.up3(self._upsample_and_concat(b, s3))
        u2 = self.up2(self._upsample_and_concat(u3, s2))
        u1 = self.up1(self._upsample_and_concat(u2, s1))

        return self.out_conv(u1)

    @property
    def invex(self):
        blocks = [self.down1, self.down2, self.down3, self.bottleneck, self.up3, self.up2, self.up1]
        return blocks[0].invex if all(b.invex is not None for b in blocks) else None


class _DoubleConv(nn.Module):
    """Textbook UNet block: ``act(A2(act(A1(x))))`` - activation after both convs."""

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, activation, **conv_kwargs):
        super().__init__()
        self.affine1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, **conv_kwargs)
        self.affine2 = nn.Conv2d(hidden_channels, out_channels, kernel_size, **conv_kwargs)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.affine2(self.activation(self.affine1(x))))


class NormalUNet(nn.Module):
    """Same architecture/channel widths as :class:`InvexUNet`, but with the textbook
    double-conv block (``_DoubleConv``, activation after both convs) instead of
    ``ixConv2d`` (activation only between the two convs) - a genuine plain-UNet
    baseline, not merely an uncertified version of the same computation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        activation: nn.Module = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.Softplus()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.down1 = _DoubleConv(in_channels, c1, c1, kernel_size=3, activation=activation, padding=1)
        self.down2 = _DoubleConv(c1, c2, c2, kernel_size=3, activation=activation, padding=1)
        self.down3 = _DoubleConv(c2, c3, c3, kernel_size=3, activation=activation, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(c3, c4, c4, kernel_size=3, activation=activation, padding=1)

        self.up3 = _DoubleConv(c4 + c3, c3, c3, kernel_size=3, activation=activation, padding=1)
        self.up2 = _DoubleConv(c3 + c2, c2, c2, kernel_size=3, activation=activation, padding=1)
        self.up1 = _DoubleConv(c2 + c1, c1, c1, kernel_size=3, activation=activation, padding=1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _upsample_and_concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)
        s2 = self.down2(self.pool(s1))
        s3 = self.down3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        u3 = self.up3(self._upsample_and_concat(b, s3))
        u2 = self.up2(self._upsample_and_concat(u3, s2))
        u1 = self.up1(self._upsample_and_concat(u2, s1))

        return self.out_conv(u1)


def synthetic_piecewise_constant_image(size: int = 64) -> torch.Tensor:
    """A [1, 1, size, size] image with a few constant-valued rectangular blocks."""
    img = torch.zeros(1, 1, size, size)
    img[:, :, size // 6 : size // 2, size // 6 : size // 2] = 0.8
    img[:, :, size // 2 :, size // 2 :] = 0.4
    img[:, :, size // 4 : size // 3, 2 * size // 3 :] = 1.0
    return img


def main():
    torch.manual_seed(0)
    clean = synthetic_piecewise_constant_image()
    noisy = clean + torch.randn_like(clean) * 0.15

    model = InvexUNet(in_channels=1, out_channels=1, base_channels=16)
    print(f"model.invex (all blocks certified): {model.invex is not None}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 151):
        optimizer.zero_grad()
        pred = model(noisy)
        loss = loss_fn(pred, clean)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 30 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

    noisy_mse = torch.mean((noisy - clean) ** 2).item()
    denoised_mse = torch.mean((model(noisy) - clean) ** 2).item()
    print(f"MSE noisy:    {noisy_mse:.4f}")
    print(f"MSE denoised: {denoised_mse:.4f}")


if __name__ == "__main__":
    main()
