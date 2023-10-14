import torch


class ZeroOneClamp(object):
    def __init__(self):
        pass

    def __call__(self, module):
        w = module.weight.data
        module.weight.data = torch.clamp(w, 0, 1)
