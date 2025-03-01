from torch.autograd import grad
import numpy as np
import torch.nn as nn
import torch

# All of this as dependence
class PositiveLinear(nn.Module):
  def __init__(self, in_features, out_features, bias = False):
    super().__init__()
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
    else:
      return nn.functional.linear(input, self.log_weight ** 2)


def create_representation_positive(inputdim, layers, dropout = 0):
  modules = []
  act = nn.Tanh()

  prevdim = inputdim
  for hidden in layers:
    modules.append(PositiveLinear(prevdim, hidden, bias = True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(act)
    prevdim = hidden

  # Need all values positive
  modules[-1] = nn.Softplus()

  return nn.Sequential(*modules)

def create_representation(inputdim, layers, activation, dropout = 0., last = None):
  if activation == 'ReLU6':
    act = nn.ReLU6()
  elif activation == 'ReLU':
    act = nn.ReLU()
  elif activation == 'Tanh':
    act = nn.Tanh()

  modules = []
  prevdim = inputdim

  for hidden in layers:
    modules.append(nn.Linear(prevdim, hidden, bias = True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(act)
    prevdim = hidden

  if not(last is None):
    modules[-1] = last
  
  return modules

class NeuralFineGrayTorch(nn.Module):

  def __init__(self, inputdim, layers = [100, 100, 100], act = 'ReLU', layers_surv = [100],
               risks = 1, dropout = 0., optimizer = "Adam", multihead = True):
    super().__init__()
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.dropout = dropout
    self.optimizer = optimizer

    self.embed = nn.Sequential(*create_representation(inputdim, layers + [inputdim], act, self.dropout)) # Assign each point to a cluster
    self.balance = nn.Sequential(*create_representation(inputdim, layers + [risks], act)) # Define balance between outcome (ensure sum < 1)
    self.outcome = nn.ModuleList(
                      [create_representation_positive(inputdim + 1, layers_surv + [1]) # Multihead (one for each outcome)
                  for _ in range(risks)]) if multihead \
                  else create_representation_positive(inputdim + 1, layers_surv + [risks])
    self.softlog = nn.LogSoftmax(dim = 1)

    self.forward = self.forward_multihead if multihead else self.forward_single

  def forward_multihead(self, x, horizon):
    x_rep = self.embed(x)
    log_beta = self.softlog(self.balance(x_rep)) # Balance

    # Compute cumulative hazard function
    sr = []
    tau_outcome = horizon.clone().detach().requires_grad_(True).unsqueeze(1)
    for outcome_competing in self.outcome:
      outcome = tau_outcome * outcome_competing(torch.cat((x_rep, tau_outcome), 1))
      sr.append(- outcome)

    sr = torch.cat(sr, -1)
    return sr, log_beta, tau_outcome
  
  def gradient(self, outcomes, horizon, e):
    # Compute gradient for points with observed risk - Faster: only one backpropagation
    return grad([- outcomes[:, r][e == (r + 1)].sum() for r in range(self.risks)], horizon, create_graph = True)[0].clamp_(1e-10)[:, 0]

  def forward_single(self, x, horizon):
    x_rep = self.embed(x)
    log_beta = self.softlog(self.balance(x_rep)) # Balance

    # Compute cumulative hazard function
    tau_outcome = horizon.clone().detach().requires_grad_(True).unsqueeze(1)
    outcome = tau_outcome * self.outcome(torch.cat((x_rep, tau_outcome), 1))
    return -outcome, log_beta, tau_outcome