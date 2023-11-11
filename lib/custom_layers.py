import torch


class ElementwiseMultiply(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features, out_features, lower_bound=0, upper_bound=1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        weights = torch.Tensor(in_features, out_features)
        self.weight = torch.nn.Parameter(weights)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):
        # Perform element-wise multiplication
        result = torch.mul(torch.Tensor(x), self.weight.t())

        # Bound the output between the specified lower and upper bounds
        result = torch.clamp(result, self.lower_bound, self.upper_bound)
        return result
