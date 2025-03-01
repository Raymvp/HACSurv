import torch.nn as nn
import torch
import numpy as np
from torch.autograd import grad

class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        y = exp.log1p()
        return x.where(torch.isinf(exp),y.half() if x.type()=='torch.cuda.HalfTensor' else y )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (-x).exp().half() if x.type()=='torch.cuda.HalfTensor' else (-x).exp()
        return grad_output / (1 + y)
log1plusexp = Log1PlusExp.apply

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        self.log_weight.data.abs_().sqrt_()

    def forward(self, input):
        if self.bias is not None:
            return nn.functional.linear(input, self.log_weight ** 2, self.bias)
            # return nn.functional.linear(input, self.log_weight.exp(), self.bias)            
        else:
            return nn.functional.linear(input, self.log_weight ** 2)
            # return nn.functional.linear(input, self.log_weight.exp())


def create_representation_positive(inputdim, layers, dropout = 0):
    modules = []
    prevdim = inputdim
    for hidden in layers[:-1]:
        modules.append(PositiveLinear(prevdim, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(nn.Tanh())
        # modules.append(nn.ReLU())
        prevdim = hidden
    modules.append(PositiveLinear(prevdim, layers[-1], bias=True))

    return nn.Sequential(*modules)

def create_representation(inputdim, layers, dropout = 0):
    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(nn.Tanh())
        # modules.append(nn.ReLU())
        prevdim = hidden

    return nn.Sequential(*modules)


class NDE(nn.Module):
    def __init__(self, inputdim, layers = [32, 32, 32], layers_surv = [100, 100, 100], 
               dropout = 0., optimizer = "Adam"):
        super(NDE, self).__init__()
        self.input_dim = inputdim
        self.dropout = dropout
        self.optimizer = optimizer
        self.embedding = create_representation(inputdim, layers, self.dropout) 
        self.outcome = create_representation_positive(1 + layers[-1], layers_surv + [1], self.dropout) 

    def forward(self, x, horizon, gradient = False):
        # Go through neural network




        # horizon = horizon.expand(x.shape[0])
        x_embed = self.embedding(x) # Extract unconstrained NN
        time_outcome = horizon.clone().detach().requires_grad_(gradient) # Copy with independent gradient
        # print(x_embed.shape)
        # # print(time_outcome.shape)
        #         # 确认维度
        # print(f"x_embed shape: {x_embed.shape}")  # 应显示 [batch_size, features]
        # print(f"time_outcome shape before unsqueeze: {time_outcome.shape}")  # 应显示 [batch_size]
        if time_outcome.size(0) != x_embed.size(0):
            time_outcome = time_outcome.repeat(x_embed.size(0), 1)
            survival = self.outcome(torch.cat((x_embed, time_outcome), 1)) # Compute survival
        else :
            survival = self.outcome(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1)) # Compute survival
        #推理的时候用这个
        # time_outcome = time_outcome.repeat(x_embed.size(0), 1)
        # print(time_outcome.shape)
        # survival = self.outcome(torch.cat((x_embed, time_outcome), 1)) # Compute survival
        # survival = self.outcome(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1)) # Compute survival



        survival = survival.sigmoid()
        # Compute gradients
        intensity = grad(survival.sum(), time_outcome, create_graph = True)[0].unsqueeze(1) if gradient else None

        return 1 - survival, intensity
        # return 1 - survival.squeeze(), intensity.squeeze() #推理的时候可以这样

    def survival(self, x, horizon):  
        with torch.no_grad():
            horizon = horizon.expand(x.shape[0])
            temp = self.forward(x, horizon)[0]
        return temp.squeeze()

def total_loss(model, x, t, e, eps = 1e-8):

    # Go through network
    survival, intensity = model.forward(x, t, gradient = True)
    with torch.no_grad():
        survival.clamp_(eps)
        intensity.clamp_(eps)

    # Likelihood error
    error = torch.log(survival[e == 0]).sum()
    error += torch.log(intensity[e != 0]).sum()

    return - error / len(x)