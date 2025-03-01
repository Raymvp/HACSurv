import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv
import warnings
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 添加自定义模块路径
sys.path.append('../')
sys.path.append('DeepSurvivalMachines/')
sys.path.append("../")

# 导入自定义模块

from survival import MixExpPhiStochastic, InnerGenerator, InnerGenerator2, HACSurvival_competing_shared, sample
from nfg import datasets

# 定义当前时间函数用于文件命名
def current_time():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# 定义处理函数
def process_surv_df_CIF(surv_df, step=0.1):
    return 1 - surv_df.clip(0, 1)

def process_surv_df_jointCIF(surv_df, step=0.1):
    return 1 - (surv_df * step).cumsum()

def process_surv_df_SF(surv_df, step=0.1):
    return surv_df.clip(0, 1)

# 定义通用的计算指标函数
def calculate_metrics(model, device, times_tensor_train, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val, model_methods, event_ranges, surv_df_processings):
    c_indexes_all = []
    ibses_all = []
    step = 0.1
    times = np.arange(times_tensor_train.min().cpu(), times_tensor_train.max().cpu() + step, step)
    times_tensor = torch.tensor(times, dtype=torch.float64).unsqueeze(1).to(device)
    time_tensor_list = [time_tensor.expand(covariate_tensor_val.shape[0]) for time_tensor in times_tensor]

    # 预先计算共享的中间变量
    intermediates_all_times = []

    for time_tensor in time_tensor_list:
        intermediates = model.compute_intermediate_quantities(time_tensor, covariate_tensor_val)
        intermediates_all_times.append(intermediates)

    # 遍历每个预测方法
    for method_name, model_method, event_range, surv_df_processing in zip(
        ["Conditional CIF", "Joint CIF", "Conditional CIF Integral", "Marginal SF"],
        model_methods,
        event_ranges,
        surv_df_processings
    ):
        c_indexes = []
        ibses = []
        for event_index in event_range:
            survprob_matrix = []
            for intermediates in intermediates_all_times:
                survprob = model_method(event_index, intermediates).cpu().detach().numpy()
                survprob_matrix.append(survprob)
            survprob_matrix = np.vstack(survprob_matrix)
            surv_df = pd.DataFrame(survprob_matrix, index=times)
            surv_df = surv_df_processing(surv_df, step=step)
            t_numpy = times_tensor_val.cpu().numpy()
            c_numpy = (event_indicator_tensor_val == (event_index + 1)).cpu().numpy()
            surv_df = surv_df.clip(0, 1)
            eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
            c_indexes.append(eval_surv.concordance_td())
            ibses.append(eval_surv.integrated_brier_score(times))
        c_indexes_all.append((method_name, c_indexes))
        ibses_all.append((method_name, ibses))
    return c_indexes_all, ibses_all

# 定义主实验函数
def run_experiment(seed, base_path):
    print(f"\n=== Running experiment with seed: {seed} ===")

    # 加载数据
    x, t, e, columns = datasets.load_dataset('FRAMINGHAM', path='../', competing=True)
    # 如果使用其他数据集，请取消注释并修改路径
    # x, t, e, columns = datasets.load_dataset('PBC', path='../', competing=True)

    # 数据分割
    x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(
        x, t, e, test_size=0.2, random_state=seed
    )
    x_train, x_val, t_train, t_val, e_train, e_val = train_test_split(
        x_train, t_train, e_train, test_size=0.2, random_state=seed
    )

    # 归一化时间
    minmax = lambda x: x / t_train.max()
    t_train_ddh = minmax(t_train)
    t_test_ddh = minmax(t_test)
    t_val_ddh = minmax(t_val)

    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 转换为张量
    covariate_tensor_train = torch.tensor(x_train, dtype=torch.float64).to(device)
    covariate_tensor_val = torch.tensor(x_val, dtype=torch.float64).to(device)
    covariate_tensor_test = torch.tensor(x_test, dtype=torch.float64).to(device)

    times_tensor_train = torch.tensor(t_train_ddh, dtype=torch.float64).to(device)
    event_indicator_tensor_train = torch.tensor(e_train, dtype=torch.float64).to(device)

    times_tensor_val = torch.tensor(t_val_ddh, dtype=torch.float64).to(device)
    event_indicator_tensor_val = torch.tensor(e_val, dtype=torch.float64).to(device)

    times_tensor_test = torch.tensor(t_test_ddh, dtype=torch.float64).to(device)
    event_indicator_tensor_test = torch.tensor(e_test, dtype=torch.float64).to(device)

    # 设置线程数和默认张量类型
    torch.set_num_threads(16)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # 创建数据加载器
    batch_size = 10000
    train_dataset = TensorDataset(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    psi = MixExpPhiStochastic(device)
    ckpt_path_out = './Framingham/checkpoint/Framingham_e01_best.pth'
    ckpt_out = torch.load(ckpt_path_out)
    phi_out_keys = {k.replace('phi.', ''): v for k, v in ckpt_out['model_state_dict'].items() if 'phi' in k and 'phi_inv' not in k}
    psi.load_state_dict(phi_out_keys) 
    phi_12 = InnerGenerator(psi, device)
    model = HACSurvival_competing_shared(psi, phi_12, device=device, num_features=18, tol=1e-14, hidden_size=100).to(device)

    # 加载模型权重
    ckpt_path_12 = './Framingham/checkpoint/Shared_Model_inner_e1e2_step2.pth'
    ckpt_12 = torch.load(ckpt_path_12)
    phi_12_keys = {k.replace('phi.', ''): v for k, v in ckpt_12['model_state_dict'].items() if 'phi' in k and 'phi_inv' not in k}
    model.phi_in.load_state_dict(phi_12_keys)
    phi_12_inv_keys = {k.replace('phi_inv.', ''): v for k, v in ckpt_12['model_state_dict'].items() if 'phi_inv.' in k}
    model.phi_in_inv.load_state_dict(phi_12_inv_keys)

    # 设置优化器
    optimizer_out = optim.Adam([
        {"params": model.shared_embedding.parameters(), "lr": 1e-4}, 
        {"params": model.sumo_e1.parameters(), "lr": 1e-4},
        {"params": model.sumo_e2.parameters(), "lr": 1e-4},
        {"params": model.sumo_c.parameters(), "lr": 1e-4},
    ], weight_decay=0)

    # 准备模型方法、事件范围和处理函数
    model_methods = [
        model.survival_withCopula_condition_CIF_No_intergral,
        model.survival_withCopula_joint_CIF_,
        model.survival_withCopula_condition_CIF_intergral,
        model.survival_event_onlySurvivalFunc
    ]
    event_ranges = [range(2), range(2), range(2), range(2)]  # 事件范围调整为2
    surv_df_processings = [
        process_surv_df_CIF,
        lambda surv_df, step=0.1: process_surv_df_jointCIF(surv_df, step),
        lambda surv_df, step=0.1: process_surv_df_jointCIF(surv_df, step),
        process_surv_df_SF
    ]

    # 训练参数
    best_avg_c_index = -np.inf
    best_val_loglikelihood = float('-inf')
    epochs_no_improve = 0
    num_epochs = 100000
    early_stop_epochs = 240  # 调整为适当的早停阈值
    best_model_filename = ""

    # 初始化用于存储测试结果的字典
    test_metrics = {
        "Conditional CIF": {"Risk 1": {"C-index": [], "IBS": []},
                            "Risk 2": {"C-index": [], "IBS": []}},
        "Joint CIF": {"Risk 1": {"C-index": [], "IBS": []},
                     "Risk 2": {"C-index": [], "IBS": []}},
        "Conditional CIF Integral": {"Risk 1": {"C-index": [], "IBS": []},
                                      "Risk 2": {"C-index": [], "IBS": []}},
        "Marginal SF": {"Risk 1": {"C-index": [], "IBS": []},
                       "Risk 2": {"C-index": [], "IBS": []}},
    }

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        for covariates, times, events in train_loader:
            # 训练过程
            model.phi_in.psi.resample_M(100)
            model.phi_in.resample_M(100)
            optimizer_out.zero_grad()
            logloss = model(covariates, times, events, max_iter=1000)
            (-logloss).backward(retain_graph=True)
            optimizer_out.step()

        if epoch % 20 == 0:
            model.eval()
            
            val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter=1000)
            print(f"Epoch {epoch}: Train loglikelihood {logloss.item():.4f}, Val likelihood {val_loglikelihood.item():.4f}")

            # 计算指标
            c_indexes_all, ibses_all = calculate_metrics(
                model, device, times_tensor_train, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val,
                model_methods, event_ranges, surv_df_processings
            )

            # 提取 Conditional CIF 的 C-index 作为早停依据
            conditional_c_indexes = []
            for method_name, c_indexes in c_indexes_all:
                if method_name == "Conditional CIF":
                    conditional_c_indexes.extend(c_indexes)  # 收集所有风险的C-index
            avg_conditional_c_index = np.mean(conditional_c_indexes) if conditional_c_indexes else -np.inf

            print(f'Epoch {epoch}: Metrics:')
            for (method_name, c_indexes), (method_name_ibse, ibses) in zip(c_indexes_all, ibses_all):
                method_name_padded = f"{method_name:<25}"  # 固定宽度对齐方法名称
                c_index_str = f"Risk1 C-index: {c_indexes[0]:.4f}, Risk2 C-index: {c_indexes[1]:.4f}"
                ibse_str = f"Risk1 IBS: {ibses[0]:.4f}, Risk2 IBS: {ibses[1]:.4f}"
                print(f'  {method_name_padded}: {c_index_str} | {ibse_str}')

            # 检查 Conditional CIF 的 C-index 是否提升
            if avg_conditional_c_index > best_avg_c_index:
                best_avg_c_index = avg_conditional_c_index
                if best_model_filename:
                    os.remove(os.path.join(base_path, best_model_filename))  # 删除旧的最佳模型文件
                best_model_filename = f"New_HACSurv_shared_Framing2_Risks_cindex_{best_avg_c_index:.4f}_{current_time()}_{seed}.pth"
                torch.save(model.state_dict(), os.path.join(base_path, best_model_filename))
                epochs_no_improve = 0
                print('Best model updated and saved.')
            else:
                epochs_no_improve += 20  # 每次验证后增加20（因为每20个epoch验证一次）

            # 早停
            if epochs_no_improve >= early_stop_epochs:
                print(f'Early stopping triggered at epoch: {epoch}')
                break

    # 加载最佳模型并进行测试
    if best_model_filename:
        ckpt_path = os.path.join(base_path, best_model_filename)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        # 计算测试 log likelihood
        test_loglikelihood = model(covariate_tensor_test, times_tensor_test, event_indicator_tensor_test, max_iter=1000)
        
        # 计算测试指标
        c_indexes_all, ibses_all = calculate_metrics(
            model, device, times_tensor_train, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test,
            model_methods, event_ranges, surv_df_processings
        )

        print(f"\n=== Test Results for seed {seed} ===")
        print(f"Test loglikelihood: {test_loglikelihood.item():.4f}")
        for (method_name, c_indexes), (method_name_ibse, ibses) in zip(c_indexes_all, ibses_all):
            print(f'  {method_name}:')
            for event_idx, (c_index, ibse) in enumerate(zip(c_indexes, ibses)):
                print(f'    Risk {event_idx+1} - C-index: {c_index:.4f}, IBS: {ibse:.4f}')

            # 存储测试结果
            for event_idx, (c_index, ibse) in enumerate(zip(c_indexes, ibses)):
                test_metrics[method_name][f"Risk {event_idx+1}"]["C-index"].append(c_index)
                test_metrics[method_name][f"Risk {event_idx+1}"]["IBS"].append(ibse)

    return test_metrics  # 返回测试结果

# 设置保存模型的基础路径
base_path = "./Framingham/checkpoint"
os.makedirs(base_path, exist_ok=True)  # 确保路径存在

# 初始化用于汇总所有种子的测试结果的字典
all_test_metrics = {
    "Conditional CIF": {"Risk 1": {"C-index": [], "IBS": []},
                        "Risk 2": {"C-index": [], "IBS": []}},
    "Joint CIF": {"Risk 1": {"C-index": [], "IBS": []},
                 "Risk 2": {"C-index": [], "IBS": []}},
    "Conditional CIF Integral": {"Risk 1": {"C-index": [], "IBS": []},
                                  "Risk 2": {"C-index": [], "IBS": []}},
    "Marginal SF": {"Risk 1": {"C-index": [], "IBS": []},
                   "Risk 2": {"C-index": [], "IBS": []}},
}

# 运行种子从41到45的实验，并收集测试结果
for seed in range(41, 46):
    test_metrics = run_experiment(seed, base_path)
    # 将每个种子的测试结果添加到汇总字典中
    for method in all_test_metrics.keys():
        for risk in all_test_metrics[method].keys():
            all_test_metrics[method][risk]["C-index"].extend(test_metrics[method][risk]["C-index"])
            all_test_metrics[method][risk]["IBS"].extend(test_metrics[method][risk]["IBS"])

# 汇总并打印所有种子的测试结果
print("\n=== Summary of Test Results Across All Seeds ===")
for method, risks in all_test_metrics.items():
    print(f'\n{method}:')
    for risk, metrics in risks.items():
        c_index_mean = np.mean(metrics["C-index"])
        c_index_std = np.std(metrics["C-index"])
        ibs_mean = np.mean(metrics["IBS"])
        ibs_std = np.std(metrics["IBS"])
        print(f'  {risk}:')
        print(f'    C-index: {c_index_mean:.4f} ± {c_index_std:.4f}')
        print(f'    IBS: {ibs_mean:.4f} ± {ibs_std:.4f}')
