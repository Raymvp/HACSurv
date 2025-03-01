import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from sksurv.svm import FastKernelSurvivalSVM
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines.utils import concordance_index
from scipy.stats import gumbel_r, norm, logistic
import lifelines.datasets as dset
import math
from sklearn.calibration import calibration_curve, CalibrationDisplay
from matplotlib.gridspec import GridSpec
import warnings
from scipy.special import erf
from sklearn.preprocessing import StandardScaler
import sys
# sys.path.append('/home/liuxin/HACSurv')
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torchtuples as tt
from synthetic_dgp import linear_dgp, linear_dgp_hac
from truth_net import Weibull_linear
from metric import surv_diff
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


###生成数据
# batch_size = 20000

copula_form = 'Clayton'
sample_size = 80000

seed = 142857
device = torch.device("cuda:0")

# Set the number of threads and default tensor type
torch.set_num_threads(24)
torch.set_default_tensor_type(torch.DoubleTensor)

# Generate data
rng = np.random.default_rng(seed)
X, observed_time, event_indicator, _, _, beta_e1, beta_e2, beta_e3 = linear_dgp_hac(
    copula_name=copula_form, covariate_dim=10, theta=1, sample_size=sample_size, rng=rng
)


seeds=43
print(seeds)
# split train test
X_train, X_test, y_train, y_test, indicator_train, indicator_test = train_test_split(X, observed_time, event_indicator, test_size=0.2, stratify= event_indicator, random_state=seeds)
# split train val
X_train, X_val, y_train, y_val, indicator_train, indicator_val = train_test_split(X_train, y_train, indicator_train, test_size=0.2, stratify= indicator_train, random_state=seeds)


covariate_tensor_train = torch.tensor(X_train, dtype=torch.float64).to(device)
covariate_tensor_val = torch.tensor(X_val, dtype=torch.float64).to(device)
covariate_tensor_test = torch.tensor(X_test, dtype=torch.float64).to(device)


times_tensor_train = torch.tensor(y_train, dtype=torch.float64).to(device)
event_indicator_tensor_train = torch.tensor(indicator_train, dtype=torch.float64).to(device)


times_tensor_val= torch.tensor(y_val, dtype=torch.float64).to(device)
event_indicator_tensor_val = torch.tensor(indicator_val, dtype=torch.float64).to(device)

times_tensor_test = torch.tensor(y_test, dtype=torch.float64).to(device)
event_indicator_tensor_test = torch.tensor(indicator_test, dtype=torch.float64).to(device)

torch.set_num_threads(24)
torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_default_tensor_type(torch.DoubleTensor)
def cumulative_integral(df, step):
    # 对每一列进行累积积分
    df_integrated = df.cumsum(axis=0) * step
    return df_integrated


from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from survival import MixExpPhiStochastic,InnerGenerator,sample
import torch.optim as optim
from survival import HACSurvival_competing_shared,HACSurvival_competing3_shared
torch.set_default_tensor_type(torch.DoubleTensor)
psi = MixExpPhiStochastic(device)
ckpt_path_out =  './Competing_SYN/checkpoint/Example_competing_syn_outer_02_step1.pth'

ckpt_out = torch.load(ckpt_path_out)
phi_out_keys = {k.replace('phi.', ''): v for k, v in ckpt_out['model_state_dict'].items() if 'phi' in k and 'phi_inv' not in k}
#对于对称outer copula
# phi_out_keys = {k.replace('phi.', ''): v for k, v in ckpt_out.items() if 'phi' in k and 'phi_inv' not in k}
psi.load_state_dict(phi_out_keys) 
phi_01= InnerGenerator(psi,device)
phi_23= InnerGenerator(psi,device)
# phi_135= InnerGenerator(psi,device)
# model =HACSurvival_competing_shared(psi,phi_12, device = device, num_features=17, tol=1e-14, hidden_size = 100).to(device)


model =HACSurvival_competing3_shared(psi,phi_01,phi_23, device = device, num_features=10, tol=1e-14, hidden_size = 100).to(device)
ckpt_path_01 = './Competing_SYN/checkpoint/Example_competing_syn_inner01_step2.pth'
ckpt_01 = torch.load(ckpt_path_01)
phi_01_keys =  {k.replace('phi.', ''): v for k, v in ckpt_01['model_state_dict'].items() if 'phi' in k and 'phi_inv' not in k}
model.phi_01.load_state_dict(phi_01_keys)
phi_01_inv_keys = {k.replace('phi_inv.', ''): v for k, v in ckpt_01['model_state_dict'].items() if 'phi_inv.' in k}
model.phi_01_inv.load_state_dict(phi_01_inv_keys)

ckpt_path_23 = './Competing_SYN/checkpoint/Example_competing_syn_inner23_step2.pth'
ckpt_23 = torch.load(ckpt_path_23)
phi_23_keys =  {k.replace('phi.', ''): v for k, v in ckpt_23['model_state_dict'].items() if 'phi' in k and 'phi_inv' not in k}
model.phi_23.load_state_dict(phi_23_keys)
phi_23_inv_keys = {k.replace('phi_inv.', ''): v for k, v in ckpt_23['model_state_dict'].items() if 'phi_inv.' in k}
model.phi_23_inv.load_state_dict(phi_23_inv_keys)


optimizer_out = optim.Adam([
    {"params": model.shared_embedding.parameters(), "lr": 1e-4}, 
    {"params": model.sumo_e1.parameters(), "lr": 1e-4},
    {"params": model.sumo_e2.parameters(), "lr": 1e-4},
    {"params": model.sumo_e3.parameters(), "lr": 1e-4},
    {"params": model.sumo_c.parameters(), "lr": 1e-4},

],weight_decay=0)
from datetime import datetime
import os
def current_time():
    return datetime.now().strftime('%Y%m%d_%H%M%S')



best_avg_c_index = float('-inf')
best_val_loglikelihood = float('-inf')
epochs_no_improve = 0
num_epochs = 10000
early_stop_epochs = 1600
base_path = "./Competing_SYN/checkpoint"
best_model_filename = ""

for epoch in range(num_epochs):

    # Training process
    model.phi_01.psi.resample_M(100)
    model.phi_01.resample_M(100)
    model.phi_23.resample_M(100)
    optimizer_out.zero_grad()
    logloss = model(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train, max_iter=1000)
    (-logloss).backward(retain_graph=True)
    optimizer_out.step()

    if epoch % 80 == 0:
        model.eval()
        val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter=1000)
        print(f"Epoch {epoch}: Train loglikelihood {logloss.item()}, Val likelihood {val_loglikelihood.item()}")

        # # Compute metrics on the validation set
        # c_indexes_val_CIF, ibses_val_CIF = calculate_metrics_CIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        # c_indexes_val_SF, ibses_val_SF = calculate_metrics_SF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        # c_indexes_val_jointCIF, ibses_val_jointCIF = calculate_metrics_jointCIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        # c_indexes_val_Con_integral_CIF, ibses_val_Con_integral_CIF = calculate_metrics_Conditional_CIF_integral(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        
        # print('SF               ', c_indexes_val_SF, ibses_val_SF)
        # print('joint_CIF        ', c_indexes_val_jointCIF, ibses_val_jointCIF)
        # print('Con_integral_CIF ', c_indexes_val_Con_integral_CIF, ibses_val_Con_integral_CIF)
        # print('CIF              ', c_indexes_val_CIF, ibses_val_CIF)
        
        # Use the mean of c_indexes_val_Con_integral_CIF for early stopping
        # avg_c_index = np.mean(c_indexes_val_CIF)
        
        # Check if the new average C-index is better
        if val_loglikelihood > (best_val_loglikelihood + 1):
            best_val_loglikelihood = val_loglikelihood

            # Save the best model checkpoint
            if best_model_filename:
                os.remove(os.path.join(base_path, best_model_filename))  # Delete old best model file

            best_model_filename = f"BestModel_cindex_{best_val_loglikelihood:.4f}_{current_time()}seed{seeds}.pth"
            torch.save(model.state_dict(), os.path.join(base_path, best_model_filename))
            epochs_no_improve = 0
            print('Best model updated and saved.')
        else:
            epochs_no_improve += 100

        # Early stopping
        if epochs_no_improve >= early_stop_epochs:
            print(f'Early stopping triggered at epoch: {epoch}')
            break

# After early stopping, load the best model checkpoint
model.load_state_dict(torch.load(os.path.join(base_path, best_model_filename)))
model.eval()

# Compute metrics on the test set using the four methods
test_loglikelihood = model(covariate_tensor_test, times_tensor_test, event_indicator_tensor_test, max_iter=1000)
print(f"Test loglikelihood: {test_loglikelihood.item()}")


# c_indexes_test_CIF, ibses_test_CIF = calculate_metrics_CIF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
# c_indexes_test_SF, ibses_test_SF = calculate_metrics_SF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
# c_indexes_test_jointCIF, ibses_test_jointCIF = calculate_metrics_jointCIF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
# c_indexes_test_Con_integral_CIF, ibses_test_Con_integral_CIF = calculate_metrics_Conditional_CIF_integral(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)

# print('Seed:',seeds)
# print('Test Metrics:')
# print('SF               ', c_indexes_test_SF, ibses_test_SF)
# print('joint_CIF        ', c_indexes_test_jointCIF, ibses_test_jointCIF)
# print('Con_integral_CIF ', c_indexes_test_Con_integral_CIF, ibses_test_Con_integral_CIF)
# print('CIF              ', c_indexes_test_CIF, ibses_test_CIF)



# 初始化模型
truth_model1 = Weibull_linear(num_feature=X_test.shape[1], shape=6, scale=15, device=torch.device("cpu"), coeff=beta_e1)
truth_model2 = Weibull_linear(num_feature=X_test.shape[1], shape=5, scale=14, device=torch.device("cpu"), coeff=beta_e2)
truth_model3 = Weibull_linear(num_feature=X_test.shape[1], shape=4, scale=19, device=torch.device("cpu"), coeff=beta_e3)

# 准备评估数据
steps = np.linspace(y_test.min(), y_test.max(), 1000)
survival_l1 = []

# 计算每个真实模型与预测模型的L1范数差异
for event_index, truth_model in enumerate([truth_model1, truth_model2, truth_model3]):
    performance = surv_diff(truth_model, model, X_test, steps, event_index)
    survival_l1.append(performance)

# 输出每个模型的结果
for i, performance in enumerate(survival_l1, 1):
    print(f"Survival L1 difference for Model {i}: {performance}")
