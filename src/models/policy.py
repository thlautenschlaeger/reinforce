import torch
from torch import nn
from torch.functional import F
from src.models.basic_nn import create_mlp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()

class CategoricalPolicy(nn.Module):

    def __init__(self, in_features: int, n_hidden: list, out_features: int, nonlin: str = 'relu',
                 layer_norm: bool = False):
        super(CategoricalPolicy, self).__init__()
        self.net = create_mlp(in_features, n_hidden, out_features, layer_norm, nonlin)

    def forward(self, x, **kwargs):
        for _, layer in enumerate(self.net):
            x = layer(x)
        # x = self.out(x)

        return [F.softmax(x, dim=0)]

class GaussianPolicy(nn.Module):

    def __init__(self, in_features: int, n_hidden: list, out_features: int, nonlin: str = 'tanh',
                 layer_norm: bool = False, std=1.0):
        super(GaussianPolicy, self).__init__()

        self.net = create_mlp(in_features, n_hidden, out_features, layer_norm, nonlin)
        self.log_std = nn.Parameter(torch.ones(1, out_features) * torch.tensor(std).log())


    def forward(self, x, **kwargs):
        for _, layer in enumerate(self.net):
            x = layer(x)

        mean = x
        std = self.log_std.exp()

        return mean, std