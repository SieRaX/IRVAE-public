import torch
import torch.nn as nn
import numpy as np

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")

class FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_vec, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)
    
class FC_image(nn.Module):
    def __init__(
        self,
        in_chan=784,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
        out_chan_num=1,
    ):
        super(FC_image, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.out_chan_num = out_chan_num
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for i_layer, [n_hidden, act] in enumerate(zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.view(-1, self.in_chan)
            out = self.net(x)
        else:
            dim = np.int(np.sqrt(self.out_chan / self.out_chan_num))
            out = self.net(x)
            out = out.reshape(-1, self.out_chan_num, dim, dim)
        return out

class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""

    def __init__(self, net, sigma=1):
        super().__init__()
        self.net = net
        sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        self.register_parameter("sigma", sigma)

    def log_likelihood(self, x, z):
        decoder_out = self.net(z)
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - decoder_out) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        return loglik

    def forward(self, z):
        return self.net(z)

    def sample(self, z):
        x_hat = self.net(z)
        return x_hat + torch.randn_like(x_hat) * self.sigma