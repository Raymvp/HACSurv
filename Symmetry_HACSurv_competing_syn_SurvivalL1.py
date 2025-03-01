import numpy as np
from sklearn.model_selection import train_test_split
import torch
import sys
# sys.path.append('/home/liuxin/HACSurv')
import warnings
warnings.filterwarnings("ignore")
import os

# 导入您需要的模块
from synthetic_dgp import linear_dgp_hac
from truth_net import Weibull_linear
from metric import surv_diff
from survival import MixExpPhiStochastic, HACSurv_4D_Sym_shared
import torch.optim as optim

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(24)
torch.set_default_tensor_type(torch.DoubleTensor)

# 定义种子列表
seeds_list = [41, 42, 43, 44, 45]

# 用于保存每个种子的生存 L1 差异
all_survival_l1 = []

# 遍历每个种子
for seeds in seeds_list:
    print(f"Running experiment with seed: {seeds}")
    # 生成数据
    copula_form = 'Clayton'
    sample_size = 80000
    rng = np.random.default_rng(142857)  # 固定的随机数生成器，用于数据生成
    X, observed_time, event_indicator, _, _, beta_e1, beta_e2, beta_e3 = linear_dgp_hac(
        copula_name=copula_form, covariate_dim=10, theta=1, sample_size=sample_size, rng=rng
    )

    # 分割数据集
    X_train, X_test, y_train, y_test, indicator_train, indicator_test = train_test_split(
        X, observed_time, event_indicator, test_size=0.2, stratify=event_indicator, random_state=seeds
    )
    X_train, X_val, y_train, y_val, indicator_train, indicator_val = train_test_split(
        X_train, y_train, indicator_train, test_size=0.2, stratify=indicator_train, random_state=seeds
    )

    # 转换为张量并移动到设备
    covariate_tensor_train = torch.tensor(X_train, dtype=torch.float64).to(device)
    covariate_tensor_val = torch.tensor(X_val, dtype=torch.float64).to(device)
    covariate_tensor_test = torch.tensor(X_test, dtype=torch.float64).to(device)
    times_tensor_train = torch.tensor(y_train, dtype=torch.float64).to(device)
    event_indicator_tensor_train = torch.tensor(indicator_train, dtype=torch.float64).to(device)
    times_tensor_val = torch.tensor(y_val, dtype=torch.float64).to(device)
    event_indicator_tensor_val = torch.tensor(indicator_val, dtype=torch.float64).to(device)
    times_tensor_test = torch.tensor(y_test, dtype=torch.float64).to(device)
    event_indicator_tensor_test = torch.tensor(indicator_test, dtype=torch.float64).to(device)

    # 定义模型
    phi = MixExpPhiStochastic(device)
    model = HACSurv_4D_Sym_shared(phi, device=device, num_features=10, tol=1e-14, hidden_size=100).to(device)

    optimizer_out = optim.Adam([
        {"params": model.shared_embedding.parameters(), "lr": 1e-4},
        {"params": model.sumo_e1.parameters(), "lr": 1e-4},
        {"params": model.sumo_e2.parameters(), "lr": 1e-4},
        {"params": model.sumo_e3.parameters(), "lr": 1e-4},
        {"params": model.sumo_c.parameters(), "lr": 1e-4},
        # {"params": model.phi.parameters(), "lr": 8e-4},
    ], weight_decay=0)

    from datetime import datetime

    def current_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    best_val_loglikelihood = float('-inf')
    epochs_no_improve = 0
    num_epochs = 10000
    early_stop_epochs = 1600
    base_path = "./Competing_SYN/checkpoint"
    best_model_filename = ""

    for epoch in range(num_epochs):

        # 训练过程
        model.phi.resample_M(100)
        optimizer_out.zero_grad()
        logloss = model(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train, max_iter=1000)
        (-logloss).backward(retain_graph=True)
        optimizer_out.step()

        if epoch % 80 == 0:
            model.eval()
            val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter=1000)
            print(f"Epoch {epoch}: Train loglikelihood {logloss.item()}, Val likelihood {val_loglikelihood.item()}")

            # 检查是否为最佳模型
            if val_loglikelihood > (best_val_loglikelihood + 1):
                best_val_loglikelihood = val_loglikelihood

                # 保存最佳模型
                if best_model_filename:
                    os.remove(os.path.join(base_path, best_model_filename))  # 删除旧的最佳模型文件

                best_model_filename = f"SurvL1_Independent_BestModel_loglik_{best_val_loglikelihood:.4f}_{current_time()}_seed{seeds}.pth"
                torch.save(model.state_dict(), os.path.join(base_path, best_model_filename))
                epochs_no_improve = 0
                print('Best model updated and saved.')
            else:
                epochs_no_improve += 100

            # 早停
            if epochs_no_improve >= early_stop_epochs:
                print(f'Early stopping triggered at epoch: {epoch}')
                break
            model.train()

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(base_path, best_model_filename)))
    model.eval()

    # 测试集上的对数似然
    test_loglikelihood = model(covariate_tensor_test, times_tensor_test, event_indicator_tensor_test, max_iter=1000)
    print(f"Test loglikelihood: {test_loglikelihood.item()}")

    # 初始化真实模型
    truth_model1 = Weibull_linear(num_feature=X_test.shape[1], shape=6, scale=15, device=torch.device("cpu"), coeff=beta_e1)
    truth_model2 = Weibull_linear(num_feature=X_test.shape[1], shape=5, scale=14, device=torch.device("cpu"), coeff=beta_e2)
    truth_model3 = Weibull_linear(num_feature=X_test.shape[1], shape=4, scale=19, device=torch.device("cpu"), coeff=beta_e3)

    # 准备评估数据
    steps = np.linspace(y_test.min(), y_test.max(), 1000)
    survival_l1 = []

    # 计算每个真实模型与预测模型的 L1 范数差异
    for event_index, truth_model in enumerate([truth_model1, truth_model2, truth_model3]):
        performance = surv_diff(truth_model, model, X_test, steps, event_index)
        survival_l1.append(performance)

    # 输出每个模型的结果
    for i, performance in enumerate(survival_l1, 1):
        print(f"Survival L1 difference for Model {i} with seed {seeds}: {performance}")

    # 将结果保存到总列表中
    all_survival_l1.append({
        'seed': seeds,
        'survival_l1': survival_l1
    })

# 在所有种子上输出结果
print("\nAll survival L1 differences Independent Copual:")
for result in all_survival_l1:
    seed = result['seed']
    survival_l1 = result['survival_l1']
    print(f"\nSeed: {seed}")
    for i, performance in enumerate(survival_l1, 1):
        print(f"Survival L1 difference for Model {i}: {performance}")
