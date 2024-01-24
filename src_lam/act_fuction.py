import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class LearnPsi(nn.Module):
    def __init__(self):
        super(LearnPsi, self).__init__()
        # register parameter
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可学习的参数
        self.beta = nn.Parameter(torch.tensor(2.0))  # 可学习的参数
        self.gamma = nn.Parameter(torch.tensor(3.0))  # 可学习的参数
        self.coff = nn.Parameter(torch.tensor([1.0, -3.0, 3.0, -1.0]))  # 可学习的参数
        self.omega1=nn.Parameter(torch.tensor(1.0))
        self.omega2=nn.Parameter(torch.tensor(2.0))
        self.omega3=nn.Parameter(torch.tensor(4.0))
        self.omega4=nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        return (
            self.coff[0] * F.relu(self.omega1*x - 0) ** 2
            + self.coff[1] * F.relu(self.omega2*x - self.alpha) ** 2
            + self.coff[2] * F.relu(self.omega3*x - self.beta) ** 2
            + self.coff[3] * F.relu(self.omega4*x - self.gamma) ** 2
        )
learn_psi= LearnPsi()
class Phi(nn.Module):  #是否要乘1/2
    def __init__(self):
        super(Phi, self).__init__()
        # register parameter
    def forward(self,x):

        return F.relu(x)**2 - 3 * F.relu(x - 1)**2 + 3 * F.relu(x - 2)**2  - F.relu(x - 3)**2
phi=Phi()

class SincPSi(nn.Module):
    def __init__(self):
        super(SincPSi, self).__init__()
        # 可学习的参数
        self.omega = nn.Parameter(torch.tensor(20.0))  # 调整 sinc 函数频率的参数
        self.offset=nn.Parameter(torch.tensor(1e-6))
    def forward(self, x):
        # sinc(omega * x) 的实现
        scaled_x = self.omega * x
        out=torch.where(torch.abs(x)<1e-20, torch.tensor(1.0, device=x.device), torch.sin(scaled_x) / (scaled_x+self.offset))
        return out
sinc_psi=SincPSi()
class St_act_in_4_subnet_space():
    def __init__(self):
        pass
    @classmethod
    def initializer_dict_torch(cls,identifier):
        return {
            "Glorot-normal": torch.nn.init.xavier_normal_,
            "Glorot-uniform": torch.nn.init.xavier_uniform_,
            "He-normal": torch.nn.init.kaiming_normal_,
            "He-uniform": torch.nn.init.kaiming_uniform_,
            "xavier_uniform": torch.nn.init.xavier_normal_,
        }[identifier]
    @classmethod
    def act_dict_torch(cls,identifier):
        return {
                "relu": torch.nn.ReLU(),
                "Sinc_psi": sinc_psi,
                "LearnPsi": learn_psi,
                "phi": phi,
                "sine": sine_layer,
                "tanh": torch.nn.Tanh(),
                
        }[identifier]
    @classmethod
    def activations_get(cls,act_info:np.ndarray)->list:

        '''
        return torch.nn.function,every layer every activation function
        every subnet should have different activation function
        5 个子网络，3层，说明是5*3的矩阵，每个元素是一个字符串，代表激活函数的名称
        '''
        print("act_info",act_info.shape)
        subs,layers=act_info.shape
        act_torch = nn.ModuleList()
        for i in range(subs):#every sub-net
            row = nn.ModuleList()
            for j in range(layers):
                activation_func = St_act_in_4_subnet_space.act_dict_torch(act_info[i][j])
                row.append(activation_func)
            act_torch.append(row)
        return act_torch

    @classmethod
    def weight_ini_method(self,init_info:np.ndarray)->list:
        '''

        Args:
            init_info:4*1,4个子网络

        Returns: [],4个torch.nn.init,method of weight initialization

        '''
        net_method=[]
        for i in range(init_info.shape[0]):
            net_method.append(St_act_in_4_subnet_space.initializer_dict_torch(init_info[i][0]))

        return net_method
class SineLayer(nn.Module):

# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

# If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
# nonlinearity. Different signals may require different omega_0 in the first layer - this is a
# hyperparameter.

# If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
# activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)


    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
sine_layer=SineLayer(in_features=2,out_features=1)

from collections import OrderedDict
class Siren(nn.Module):
    def __init__(self, in_features,
                hidden_features,
                hidden_layers,
                out_features,outermost_linear=False,
                first_omega_0=30,
                hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,\
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


if __name__=="__main__":
    pass