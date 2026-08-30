import pytest
import torch
import torch.nn as nn

from invexapi.nn import ixConv2d, ixLinear, ixSequential


def test_ixlinear_output_shape():
    layer = ixLinear(in_features=10, hidden_features=6, out_features=4, activation=nn.Tanh())
    x = torch.randn(3, 10)
    y = layer(x)
    assert y.shape == (3, 4)


def test_ixlinear_gradient_flows():
    layer = ixLinear(in_features=5, hidden_features=5, out_features=2, activation=torch.sigmoid)
    x = torch.randn(2, 5, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_ixlinear_certifies_invex():
    layer = ixLinear(in_features=4, hidden_features=4, out_features=4, activation=nn.Tanh())
    cert = layer.invex
    assert cert is not None
    assert cert.status == "assumed"


def test_ixconv2d_output_shape():
    layer = ixConv2d(
        in_channels=3, hidden_channels=8, out_channels=5, kernel_size=3, activation=nn.Tanh(), padding=1
    )
    x = torch.randn(2, 3, 16, 16)
    y = layer(x)
    assert y.shape == (2, 5, 16, 16)


def test_ixconv2d_gradient_flows():
    layer = ixConv2d(
        in_channels=1, hidden_channels=4, out_channels=1, kernel_size=3, activation=torch.tanh, padding=1
    )
    x = torch.randn(1, 1, 8, 8, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_ixconv2d_certifies_invex():
    layer = ixConv2d(in_channels=2, hidden_channels=2, out_channels=2, kernel_size=1, activation=nn.Tanh())
    cert = layer.invex
    assert cert is not None
    assert cert.status == "assumed"


def test_ixsequential_certifies_when_all_children_certified():
    stack = ixSequential(
        ixLinear(8, 6, 6, nn.Tanh()),
        ixLinear(6, 6, 3, nn.Tanh()),
    )
    cert = stack.invex
    assert cert is not None
    assert cert.status == "assumed"


def test_ixsequential_forward_shape():
    stack = ixSequential(
        ixLinear(8, 6, 6, nn.Tanh()),
        ixLinear(6, 6, 3, nn.Tanh()),
    )
    x = torch.randn(5, 8)
    y = stack(x)
    assert y.shape == (5, 3)


def test_ixsequential_rejects_plain_module_child():
    with pytest.raises(TypeError):
        ixSequential(ixLinear(4, 4, 4, nn.Tanh()), nn.ReLU())


def test_ixsequential_rejects_bare_linear_child():
    with pytest.raises(TypeError):
        ixSequential(nn.Linear(4, 4), ixLinear(4, 4, 4, nn.Tanh()))


def test_nested_ixsequential_still_certifies():
    inner = ixSequential(ixLinear(4, 4, 4, nn.Tanh()))
    outer = ixSequential(inner, ixLinear(4, 4, 2, nn.Tanh()))
    cert = outer.invex
    assert cert is not None
    assert cert.status == "assumed"
