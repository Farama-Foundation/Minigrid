import torch
import torch.nn.functional as F
import torch.nn as nn

class BinaryTransform(object):
    def __init__(self, thr):
        self.thr = thr

    def __call__(self, x):
        return (x >= self.thr).to(x)  # do not change the data type or device


class FlipBinaryTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.abs(x - 1)


class ReshapeTransform(object):
    def __init__(self, *args):
        self.shape = args

    def __call__(self, x: torch.Tensor):
        return x.view(*self.shape)


class FlattenTransform(object):
    def __init__(self, *args):
        self.dims = args # start and end dim

    def __call__(self, x: torch.Tensor):
        return x.view(*self.dims)

class DilationTransform(object):
    def __init__(self, num_dilations: int = 1):
        self.num_dilations = num_dilations

    def __call__(self, x: torch.Tensor):
        # x has shape (B, C, H, W)
        n_channels = x.shape[1]
        weight = torch.ones((n_channels, 1, 1, 1))
        stride = 1 + self.num_dilations
        pad = [self.num_dilations for i in range(4)]

        out = F.conv_transpose2d(x, weight=weight, stride=stride, groups=n_channels)
        out = F.pad(out, pad)
        return out


class SelectChannelsTransform(object):
    def __init__(self, *args):
        self.selected_channels = args

    def __call__(self, x: torch.Tensor):
        return x[..., self.selected_channels]


class ToDeviceTransform(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, x: torch.Tensor):
        return x.to(self.device)