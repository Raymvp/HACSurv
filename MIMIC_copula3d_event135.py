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
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchtuples as tt

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
import pandas as pd
import numpy as np
import os,sys
sys.path.append("../")
from dirac_phi import DiracPhi
from survival import MixExpPhiStochastic,InnerGenerator,InnerGenerator2,DCSurvival_competing,sample,HACSurv_3D_Sym_shared
import torch.optim as optim
import torch
from tqdm import tqdm


# 固定随机数种子
np.random.seed(42)

df = pd.read_csv('./mimic_time_data_withoutETT.csv')

# 数据分割
df_test = df.sample(frac=0.2, random_state=42)
df_train = df.drop(df_test.index)
df_val = df_train.sample(frac=0.2, random_state=42)
df_train = df_train.drop(df_val.index)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# 提取特征
x_train = df_train.drop(columns=['time', 'death_reason', 'label']).values.astype('float32')
x_val = df_val.drop(columns=['time', 'death_reason', 'label']).values.astype('float32')
x_test = df_test.drop(columns=['time', 'death_reason', 'label']).values.astype('float32')

# 提取时间和事件标签
get_target = lambda df: (df['time'].values, df['death_reason'].values)
time_train, event_train = get_target(df_train)
time_val, event_val = get_target(df_val)
time_test, event_test = get_target(df_test)

# 将数据转换为张量，并移动到指定设备上（GPU或CPU）
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

covariate_tensor_train = torch.tensor(x_train, dtype=torch.float64).to(device)
covariate_tensor_val = torch.tensor(x_val, dtype=torch.float64).to(device)
covariate_tensor_test = torch.tensor(x_test, dtype=torch.float64).to(device)

times_tensor_train = torch.tensor(time_train, dtype=torch.float64).to(device)
event_indicator_tensor_train = torch.tensor(event_train, dtype=torch.float64).to(device)

times_tensor_val = torch.tensor(time_val, dtype=torch.float64).to(device)
event_indicator_tensor_val = torch.tensor(event_val, dtype=torch.float64).to(device)

times_tensor_test = torch.tensor(time_test, dtype=torch.float64).to(device)
event_indicator_tensor_test = torch.tensor(event_test, dtype=torch.float64).to(device)

# print(covariate_tensor_val)
torch.set_num_threads(16)
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

num_epochs = 100000
batch_size = 300
early_stop_epochs = 3000

selected_indicators = [1,5]

model_dir = './MIMIC-III/checkpoint'
figures_dir = './MIMIC-III/figure'

phi = MixExpPhiStochastic(device)
model =HACSurv_3D_Sym_shared(phi, device = device, num_features=x_train.shape[1], tol=1e-14, hidden_size = 100).to(device)
optimizer = optim.AdamW([{"params": model.sumo_e1.parameters(), "lr": 1e-4},
                         {"params": model.sumo_e2.parameters(), "lr": 1e-4},
                        {"params": model.sumo_c.parameters(), "lr": 1e-4},
                        # {"params": model.phi.parameters(), "lr": 3e-4}
                        {"params": model.phi.parameters(), "lr": 2e-4}
                    ])

best_val_loglikelihood = float('-inf')
epochs_no_improve = 0
# for epoch in tqdm(range(num_epochs)):
for epoch in range(num_epochs):
    optimizer.zero_grad()
    model.phi.resample_M(200)
    logloss = model(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train, max_iter = 10000)
    # scaleloss = torch.square(torch.mean(model.phi.M)-1)
    # reg_loss = logloss+scaleloss
    (-logloss).backward() 
    optimizer.step()
    if epoch % 100 == 0 and epoch > 0:
        # Validation and logging
        print("Epoch", epoch, "Train loglikelihood:", logloss.item())
        model.phi.resample_M(200)
        val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter=10000)
        print("Validation likelihood:", val_loglikelihood.item())

        # Model checkpointing
        if val_loglikelihood > best_val_loglikelihood:
            best_val_loglikelihood = val_loglikelihood
            epochs_no_improve = 0
            indicators_str = ''.join(map(str, selected_indicators))
            checkpoint_path = os.path.join(model_dir, f'MIMIC_e135_sharedHACSurv.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': best_val_loglikelihood}, checkpoint_path)

            # Generate and save plots
            print('Scatter sampling')
            samples = sample(model, 2, 2000, device=device)
            plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
            plot_path = os.path.join(figures_dir, f'MIMIC_e135_sharedHACSurv.png')
            plt.savefig(plot_path)
            plt.clf()

        else:
            epochs_no_improve += 100

        # Early stopping condition
        if epochs_no_improve >= early_stop_epochs:
            print('Early stopping triggered at epoch:', epoch)
            break