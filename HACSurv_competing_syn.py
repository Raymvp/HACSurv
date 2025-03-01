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

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv

import pandas as pd
import numpy as np

# 固定随机数种子
# np.random.seed(42)
seeds=43
print(seeds)
# 从本地文件读取数据集
df = pd.read_csv('./MylinearSyndata_513.csv')
print(df.shape)
# 使用固定的随机数种子分割数据集
df_test = df.sample(frac=0.2, random_state=seeds)
df_train = df.drop(df_test.index)
df_val = df_train.sample(frac=0.2, random_state=seeds)
df_train = df_train.drop(df_val.index)

# 显示训练集的前几行
df_train.head()
device = torch.device("cuda:3")
get_x = lambda df: (df
                    .drop(columns=['observed_time', 'event_indicator'])
                    .values.astype('float32'))
get_target = lambda df: (df['observed_time'].values, df['event_indicator'].values)
covariate_tensor_train = torch.tensor(get_x(df_train), dtype=torch.float64).to(device)
covariate_tensor_val = torch.tensor(get_x(df_val), dtype=torch.float64).to(device)
covariate_tensor_test = torch.tensor(get_x(df_test), dtype=torch.float64).to(device)

t_train, c_train = get_target(df_train)
times_tensor_train = torch.tensor(t_train, dtype=torch.float64).to(device)
event_indicator_tensor_train = torch.tensor(c_train, dtype=torch.float64).to(device)

t_val, c_val = get_target(df_val)
times_tensor_val= torch.tensor(t_val, dtype=torch.float64).to(device)
event_indicator_tensor_val = torch.tensor(c_val, dtype=torch.float64).to(device)
t_test, c_test = get_target(df_test)
times_tensor_test = torch.tensor(t_test, dtype=torch.float64).to(device)
event_indicator_tensor_test = torch.tensor(c_test, dtype=torch.float64).to(device)

torch.set_num_threads(24)
torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_default_tensor_type(torch.DoubleTensor)
def cumulative_integral(df, step):
    # 对每一列进行累积积分
    df_integrated = df.cumsum(axis=0) * step
    return df_integrated
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from survival import MixExpPhiStochastic,InnerGenerator,InnerGenerator2,sample
import torch.optim as optim
from survival import HACSurvival_competing_shared,HACSurvival_competing3_shared
torch.set_default_tensor_type(torch.DoubleTensor)
psi = MixExpPhiStochastic(device)
ckpt_path_out =  './Competing_SYN/checkpoint/Example_competing_syn_outer_02_step1.pth'
ckpt_out = torch.load(ckpt_path_out)
phi_out_keys = {k.replace('phi.', ''): v for k, v in ckpt_out['model_state_dict'].items() if 'phi' in k and 'phi_inv' not in k}
psi.load_state_dict(phi_out_keys) 
phi_01= InnerGenerator(psi,device)
phi_23= InnerGenerator(psi,device)

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

def calculate_metrics_CIF(times_tensor_train,model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
    c_indexes = []
    ibses = []
    step = 1
    times = np.arange(0, times_tensor_train.max().cpu() + step, step)
    times_tensor = torch.tensor(times, dtype=torch.float64).unsqueeze(1).to(device)
    for event_index in range(3):
        survprob_matrix = []
        for time_tensor in times_tensor:
            time_tensor = time_tensor.expand(covariate_tensor_val.shape[0])
            survprob_matrix.append(model.survival_withCopula_condition_CIF_No_intergral(time_tensor, covariate_tensor_val,event_index).cpu().detach().numpy())
        survprob_matrix = np.vstack(survprob_matrix)
        surv_df = pd.DataFrame(survprob_matrix, index=times)

        surv_df = 1-surv_df.clip(0, 1)
        # print(surv_df)
        t_numpy = times_tensor_val.cpu().numpy()
        c_numpy = (event_indicator_tensor_val == (event_index + 1)).cpu().numpy()
        eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
        c_indexes.append(eval_surv.concordance_td())
        ibses.append(eval_surv.integrated_brier_score(times))

    return c_indexes, ibses

def calculate_metrics_jointCIF(times_tensor_train,model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
    c_indexes = []
    ibses = []
    step = 1
    # times = np.arange(times_tensor_train.min().cpu(), times_tensor_train.max().cpu() + step, step)
    times = np.arange(0, times_tensor_train.max().cpu()+step , step)
    # print(times_tensor_train.max())
    # print(times)
    times_tensor = torch.tensor(times, dtype=torch.float64).unsqueeze(1).to(device)
    for event_index in range(3):
        survprob_matrix = []
        for time_tensor in times_tensor:
            time_tensor = time_tensor.expand(covariate_tensor_val.shape[0])
            survprob_matrix.append(model.survival_withCopula_joint_CIF_( time_tensor, covariate_tensor_val,event_index).cpu().detach().numpy())
        survprob_matrix = np.vstack(survprob_matrix)
        surv_df = pd.DataFrame(survprob_matrix, index=times)
        surv_df = 1 - (surv_df* step).cumsum() 
        t_numpy = times_tensor_val.cpu().numpy()
        c_numpy = (event_indicator_tensor_val == (event_index + 1)).cpu().numpy()
        surv_df = surv_df.clip(0, 1)
        # print(surv_df)
        eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
        c_indexes.append(eval_surv.concordance_td())
        ibses.append(eval_surv.integrated_brier_score(times))
    
    return c_indexes, ibses
def calculate_metrics_Conditional_CIF_integral(times_tensor_train,model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
    c_indexes = []
    ibses = []
    step = 1
    # times = np.arange(times_tensor_train.min().cpu(), times_tensor_train.max().cpu() + step, step)
    times = np.arange(0, times_tensor_train.max().cpu()+step , step)
    # print(times_tensor_train.max())
    # print(times)
    times_tensor = torch.tensor(times, dtype=torch.float64).unsqueeze(1).to(device)
    for event_index in range(3):
        survprob_matrix = []
        for time_tensor in times_tensor:
            time_tensor = time_tensor.expand(covariate_tensor_val.shape[0])
            survprob_matrix.append(model.survival_withCopula_condition_CIF_intergral( time_tensor, covariate_tensor_val,event_index).cpu().detach().numpy())
        survprob_matrix = np.vstack(survprob_matrix)
        surv_df = pd.DataFrame(survprob_matrix, index=times)
        surv_df = 1 - (surv_df* step).cumsum() 
        t_numpy = times_tensor_val.cpu().numpy()
        c_numpy = (event_indicator_tensor_val == (event_index + 1)).cpu().numpy()
        surv_df = surv_df.clip(0, 1)
        # print(surv_df)
        eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
        c_indexes.append(eval_surv.concordance_td())
        ibses.append(eval_surv.integrated_brier_score(times))
    
    return c_indexes, ibses
def calculate_metrics_SF(times_tensor_train,model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
    c_indexes = []
    ibses = []
    step = 1
    times = np.arange(0, times_tensor_train.max().cpu() + step, step)
    times_tensor = torch.tensor(times, dtype=torch.float64).unsqueeze(1).to(device)
    for event_index in range(3):
        survprob_matrix = []
        for time_tensor in times_tensor:
            time_tensor = time_tensor.expand(covariate_tensor_val.shape[0])
            survprob_matrix.append(model.survival_event_onlySurvivalFunc(time_tensor, covariate_tensor_val,event_index).cpu().detach().numpy())
        survprob_matrix = np.vstack(survprob_matrix)
        surv_df = pd.DataFrame(survprob_matrix, index=times)
        surv_df = surv_df.clip(0, 1)
        t_numpy = times_tensor_val.cpu().numpy()
        c_numpy = (event_indicator_tensor_val == (event_index + 1)).cpu().numpy()
        eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
        c_indexes.append(eval_surv.concordance_td())
        ibses.append(eval_surv.integrated_brier_score(times))

    return c_indexes, ibses

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
num_epochs = 100000
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

        # Compute metrics on the validation set
        c_indexes_val_CIF, ibses_val_CIF = calculate_metrics_CIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        c_indexes_val_SF, ibses_val_SF = calculate_metrics_SF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        c_indexes_val_jointCIF, ibses_val_jointCIF = calculate_metrics_jointCIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        c_indexes_val_Con_integral_CIF, ibses_val_Con_integral_CIF = calculate_metrics_Conditional_CIF_integral(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
        
        print('SF               ', c_indexes_val_SF, ibses_val_SF)
        print('joint_CIF        ', c_indexes_val_jointCIF, ibses_val_jointCIF)
        print('Con_integral_CIF ', c_indexes_val_Con_integral_CIF, ibses_val_Con_integral_CIF)
        print('CIF              ', c_indexes_val_CIF, ibses_val_CIF)
        
        # Use the mean of c_indexes_val_Con_integral_CIF for early stopping
        avg_c_index = np.mean(c_indexes_val_CIF)
        
        # Check if the new average C-index is better
        if avg_c_index > best_avg_c_index:
            best_avg_c_index = avg_c_index

            # Save the best model checkpoint
            if best_model_filename:
                os.remove(os.path.join(base_path, best_model_filename))  # Delete old best model file

            best_model_filename = f"BestModel_cindex_{best_avg_c_index:.4f}_{current_time()}seed{seeds}.pth"
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

c_indexes_test_CIF, ibses_test_CIF = calculate_metrics_CIF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
c_indexes_test_SF, ibses_test_SF = calculate_metrics_SF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
c_indexes_test_jointCIF, ibses_test_jointCIF = calculate_metrics_jointCIF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
c_indexes_test_Con_integral_CIF, ibses_test_Con_integral_CIF = calculate_metrics_Conditional_CIF_integral(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)

print('Seed:',seeds)
print('Test Metrics:')
print('SF               ', c_indexes_test_SF, ibses_test_SF)
print('joint_CIF        ', c_indexes_test_jointCIF, ibses_test_jointCIF)
print('Con_integral_CIF ', c_indexes_test_Con_integral_CIF, ibses_test_Con_integral_CIF)
print('CIF              ', c_indexes_test_CIF, ibses_test_CIF)