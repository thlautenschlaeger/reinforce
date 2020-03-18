import torch
from torch import nn
from torch.functional import F

def create_mlp(in_features: int, n_hidden: list, out_features: int, layer_norm: bool = False, nonlin: str = 'relu'):
    """
    Usage:
    def forward(self, x, **kwargs):
        for _, layer in enumerate(net):
            x = layer(x)

        return out(x)

    :param in_features: number of input features
    :param n_hidden: list containing number of neurons for each hidden layer
    :param out_features: number of output features
    :param layer_norm: boolean to check if layer norm
    :param nonlin: type of nonlinearity
    :return: mlp, output layer
    """
    nlist = dict(relu=nn.ReLU(), tanh=nn.Tanh(),
                 sigmoid=nn.Sigmoid(), softplus=nn.Softplus(), lrelu=nn.LeakyReLU())
    nonlin = nlist[nonlin]
    if layer_norm:
        net = nn.ModuleList([nn.Linear(in_features, n_hidden[0]),
                             nn.LayerNorm(n_hidden[0], n_hidden[0]),
                             nonlin])
    else:
        net = nn.ModuleList([nn.Linear(in_features, n_hidden[0]),
                             nonlin])
    if len(n_hidden) > 1:
        for i, h in enumerate(n_hidden[:-1]):
            net.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
            if layer_norm:
                net.append(nn.LayerNorm(n_hidden[i + 1], n_hidden[i + 1]))
            net.append(nonlin)
    out = nn.Linear(n_hidden[-1], out_features)
    net.append(out)

    return net