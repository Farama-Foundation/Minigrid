import torch


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