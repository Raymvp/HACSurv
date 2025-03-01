from dsm.dsm_api import DSMBase
from nfg.nfg_torch import NeuralFineGrayTorch
import nfg.losses as losses
from nfg.utilities import train_nfg

import torch
import numpy as np
from tqdm import tqdm

class NeuralFineGray(DSMBase):

  def __init__(self, cuda = torch.cuda.is_available(), cause_specific = False, **params):
    self.params = params
    self.fitted = False
    self.cuda = cuda
    self.cause_specific = cause_specific
    self.loss = losses.total_loss_cs if cause_specific else losses.total_loss

  def _gen_torch_model(self, inputdim, optimizer, risks):
    model = NeuralFineGrayTorch(inputdim, **self.params,
                                     risks = risks,
                                     optimizer = optimizer).double()
    if self.cuda > 0:
      model = model.cuda()
    return model

  def fit(self, x, t, e, vsize = 0.15, val_data = None,
          optimizer = "Adam", random_state = 100, **args):
    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    model = self._gen_torch_model(x_train.size(1), optimizer, risks = maxrisk)
    model, speed = train_nfg(model, self.loss,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val, cuda = self.cuda == 2,
                         **args)

    self.speed = speed # Number of iterations needed to converge
    self.torch_model = model.eval()
    self.fitted = True
    return self    

  def compute_nll(self, x, t, e):
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    if self.cuda == 2:
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    loss = self.loss(self.torch_model, x_val, t_val, e_val)
    return loss.item()

  def predict_survival(self, x, t, risk = 1):
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = torch.DoubleTensor([t_] * len(x)).to(x.device)
        log_sr, log_beta, _  = self.torch_model(x, t_)
        beta = 1 if self.cause_specific else log_beta.exp() 
        outcomes = 1 - beta * (1 - torch.exp(log_sr)) # Exp diff => Ignore balance but just the risk of one disease
        scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

  def feature_importance(self, x, t, e, n = 100):
    """
      This method computes the features' importance by a  random permutation of the input variables.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: np.ndarray
          A numpy array of the event/censoring times, \( t \).
      e: np.ndarray
          A numpy array of the event/censoring indicators, \( \delta \).
          \( \delta = 1 \) means the event took place.
      n: int
          Number of permutations used for the computation

      Returns:
        (dict, dict): Dictionary of the mean impact on likelihood and normal confidence interval

    """
    global_nll = self.compute_nll(x, t, e)
    permutation = np.arange(len(x))
    performances = {j: [] for j in range(x.shape[1])}
    for _ in tqdm(range(n)):
      np.random.shuffle(permutation)
      for j in performances:
        x_permuted = x.copy()
        x_permuted[:, j] = x_permuted[:, j][permutation]
        performances[j].append(self.compute_nll(x_permuted, t, e))
    return {j: np.mean((np.array(performances[j]) - global_nll)/abs(global_nll)) for j in performances}, \
           {j: 1.96 * np.std((np.array(performances[j]) - global_nll)/abs(global_nll)) / np.sqrt(n) for j in performances}
          