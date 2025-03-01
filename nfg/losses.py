import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e):
  # Go through network
  log_sr, log_b, tau = model.forward(x, t)
  log_hr = model.gradient(log_sr, tau, e).log()

  # Compute competing risks
  log_balance_sr = log_b + log_sr

  # Likelihood error
  error = - torch.logsumexp(log_balance_sr[e == 0], dim = 1).sum() # Sum over the different risks and then across patient
  for k in range(model.risks):
    error -= (log_balance_sr[e == (k + 1)][:, k] + log_hr[e == (k + 1)]).sum() # Sum over patients with this risk

  return error / len(x)

def total_loss_cs(model, x, t, e):
  # Go through network
  log_sr, _, tau = model.forward(x, t)
  log_hr = model.gradient(log_sr, tau, e).log()

  # Likelihood error
  error = 0
  for k in range(model.risks):
    error -= log_sr[e != (k + 1)][:, k].sum()
    error -= log_hr[e == (k + 1)].sum()

  return error / len(x)