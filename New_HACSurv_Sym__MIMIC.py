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
# sys.path.append('/home/liuxin/HACSurv')
# sys.path.append('DeepSurvivalMachines/')

# 导入自定义模块
from survival import MixExpPhiStochastic, HACSurv_6D_Sym_shared, sample

# 读取数据
df = pd.read_csv('./mimic_time_data_withoutETT.csv')

def split_data(df, seed):
    # 数据分割
    df_test = df.sample(frac=0.2, random_state=seed)
    df_train = df.drop(df_test.index)
    df_val = df_train.sample(frac=0.2, random_state=seed)
    df_train = df_train.drop(df_val.index)
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_train, df_val, df_test

def get_features_and_labels(df):
    x = df.drop(columns=['time', 'death_reason', 'label']).values.astype('float32')
    time, event = df['time'].values, df['death_reason'].values
    return x, time, event

# 定义当前时间函数用于文件命名
def current_time():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# 定义处理函数
def process_surv_df_CIF(surv_df, step):
    return 1 - surv_df.clip(0, 1)

def process_surv_df_SF(surv_df, step):
    return surv_df.clip(0, 1)

# 定义通用的计算指标函数
def calculate_metrics(model, device, times_tensor_train, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val, model_methods, event_ranges, surv_df_processings, step):
    c_indexes_all = []
    ibses_all = []
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
        ["Conditional CIF", "Conditional CIF Integral", "Marginal SF"],
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
            surv_df = surv_df_processing(surv_df, step=step)  # 传入 step 参数
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

    # 分割数据集
    df_train, df_val, df_test = split_data(df, seed)
    # 获取特征和标签
    x_train, time_train, event_train = get_features_and_labels(df_train)
    x_val, time_val, event_val = get_features_and_labels(df_val)
    x_test, time_test, event_test = get_features_and_labels(df_test)
    # 转换为张量
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    covariate_tensor_train = torch.tensor(x_train, dtype=torch.float64).to(device)
    covariate_tensor_val = torch.tensor(x_val, dtype=torch.float64).to(device)
    covariate_tensor_test = torch.tensor(x_test, dtype=torch.float64).to(device)
    
    times_tensor_train = torch.tensor(time_train, dtype=torch.float64).to(device)
    event_indicator_tensor_train = torch.tensor(event_train, dtype=torch.float64).to(device)
    times_tensor_val = torch.tensor(time_val, dtype=torch.float64).to(device)
    event_indicator_tensor_val = torch.tensor(event_val, dtype=torch.float64).to(device)
    times_tensor_test = torch.tensor(time_test, dtype=torch.float64).to(device)
    event_indicator_tensor_test = torch.tensor(event_test, dtype=torch.float64).to(device)
    # 设置设备

    # 设置线程数和默认张量类型
    torch.set_num_threads(16)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # 创建数据加载器
    batch_size = 10000
    train_dataset = TensorDataset(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 对称情况下单个copula的消融实验
    phi = MixExpPhiStochastic(device)
    # ckpt_path_out = '/home/liuxin/HACSurv/MIMIC/copula_pth_supple/MIMIC_e13__Hacsurv1010adamw_nipssetting.pth'  #rebuttal


    # ckpt_out = torch.load(ckpt_path_out)
    # phi_out_keys = {k.replace('phi.', ''): v for k, v in ckpt_out['model_state_dict'].items() if 'phi' in k and 'phi_inv' not in k}
    # phi.load_state_dict(phi_out_keys) 

    model = HACSurv_6D_Sym_shared(phi, device=device, num_features=40, tol=1e-14, hidden_size=100).to(device)

    # 设置优化器
    optimizer_out = optim.Adam([
        {"params": model.shared_embedding.parameters(), "lr": 3e-4}, 
        {"params": model.sumo_e1.parameters(), "lr": 3e-4},
        {"params": model.sumo_e2.parameters(), "lr": 3e-4},
        {"params": model.sumo_e3.parameters(), "lr": 3e-4},
        {"params": model.sumo_e4.parameters(), "lr": 3e-4},
        {"params": model.sumo_e5.parameters(), "lr": 3e-4},
        {"params": model.sumo_c.parameters(), "lr": 3e-4},
        # {"params": model.phi.parameters(), "lr": 1e-3},
    ], weight_decay=0)

    # 准备模型方法、事件范围和处理函数
    model_methods = [
        model.survival_withCopula_condition_CIF_No_intergral,
        model.survival_withCopula_condition_CIF_intergral,
        model.survival_event_onlySurvivalFunc
    ]
    event_ranges = [range(5), range(5), range(5)]  # 事件范围调整为5
    surv_df_processings = [
        process_surv_df_CIF,
        process_surv_df_CIF,  # 由于 Joint CIF 已移除，这里改为 process_surv_df_CIF
        process_surv_df_SF
    ]

    # 训练参数
    best_avg_c_index = -np.inf
    best_val_loglikelihood = float('-inf')
    epochs_no_improve = 0
    num_epochs = 100000
    early_stop_epochs = 1200  # 调整为适当的早停阈值
    best_model_filename = ""

    # 初始化用于存储测试结果的字典
    test_metrics = {
        method_name: {f"Risk {i+1}": {"C-index": [], "IBS": []} for i in range(5)}
        for method_name in ["Conditional CIF", "Conditional CIF Integral", "Marginal SF"]
    }

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        for covariates, times, events in train_loader:
            # 训练过程
            optimizer_out.zero_grad()
            model.phi.resample_M(200)
            logloss = model(covariates, times, events, max_iter=1000)
            (-logloss).backward(retain_graph=True)
            optimizer_out.step()

        if epoch % 100 == 0:
            model.eval()
            
            val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter=1000)
            print(f"Epoch {epoch}: Train loglikelihood {logloss.item():.4f}, Val likelihood {val_loglikelihood.item():.4f}")

            # 计算指标（训练阶段使用 step=15）
            c_indexes_all, ibses_all = calculate_metrics(
                model, device, times_tensor_train, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val,
                model_methods, event_ranges, surv_df_processings, step=15
            )

            # 提取 Conditional CIF 的 C-index 作为早停依据
            conditional_c_indexes = []
            for method_name, c_indexes in c_indexes_all:
                if method_name == "Conditional CIF":
                    conditional_c_indexes.extend(c_indexes)  # 收集所有风险的C-index
            avg_conditional_c_index = np.mean(conditional_c_indexes) if conditional_c_indexes else -np.inf

            print(f'Epoch {epoch}: Metrics:')
            for (method_name, c_indexes), (_, ibses) in zip(c_indexes_all, ibses_all):
                method_name_padded = f"{method_name:<25}"  # 固定宽度对齐方法名称
                c_index_values = ', '.join([f'{c_idx:.4f}' for c_idx in c_indexes])
                ibse_values = ', '.join([f'{ibse:.4f}' for ibse in ibses])
                print(f'{method_name_padded} Cindex: {c_index_values}, IBS: {ibse_values}')

            # 检查 Conditional CIF 的 C-index 是否提升
            if avg_conditional_c_index > best_avg_c_index:
                best_avg_c_index = avg_conditional_c_index
                if best_model_filename:
                    os.remove(os.path.join(base_path, best_model_filename))  # 删除旧的最佳模型文件
                best_model_filename = f"New_HACSurv(independent_newexperi)_shared_MIMIC_cindex_{best_avg_c_index:.4f}_{current_time()}_{seed}.pth"
                torch.save(model.state_dict(), os.path.join(base_path, best_model_filename))
                epochs_no_improve = 0
                # 绘制并保存图像

                # samples = sample(model, 2, 3000, device=device)
                # plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                # plt.savefig(f'/home/liuxin/HACSurv/MIMIC/Figure/symHACSurv_MIMIC_phi_1e-3.png')
                # plt.clf()

                print('Best model updated and saved.')
            else:
                epochs_no_improve += 100  # 每次验证后增加100（因为每100个epoch验证一次）

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
        
        # 计算测试指标（测试阶段使用 step=4） 如果爆显存，就step=10 然后再写一个代码来test。
        # c_indexes_all, ibses_all = calculate_metrics(
        #     model, device, times_tensor_train, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test,
        #     model_methods, event_ranges, surv_df_processings, step=10
        # )
        c_indexes_all, ibses_all = calculate_metrics(
            model, device, times_tensor_train, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test,
            model_methods, event_ranges, surv_df_processings, step=4
        )
        print(f"\n=== Test Results for seed {seed} ===")
        print(f"Test loglikelihood: {test_loglikelihood.item():.4f}")
        for (method_name, c_indexes), (_, ibses) in zip(c_indexes_all, ibses_all):
            print(f'  {method_name}:')
            for event_idx, (c_index, ibse) in enumerate(zip(c_indexes, ibses)):
                print(f'    Risk {event_idx+1} - C-index: {c_index:.4f}, IBS: {ibse:.4f}')

            # 存储测试结果
            for event_idx, (c_index, ibse) in enumerate(zip(c_indexes, ibses)):
                test_metrics[method_name][f"Risk {event_idx+1}"]["C-index"].append(c_index)
                test_metrics[method_name][f"Risk {event_idx+1}"]["IBS"].append(ibse)
        # 将 return test_metrics 放在 for 循环外面
        return test_metrics  # 返回测试结果

# 设置保存模型的基础路径
base_path = "./MIMIC-III/checkpoint"
os.makedirs(base_path, exist_ok=True)  # 确保路径存在

# 初始化用于汇总所有种子的测试结果的字典
all_test_metrics = {
    method_name: {f"Risk {i+1}": {"C-index": [], "IBS": []} for i in range(5)}
    for method_name in ["Conditional CIF", "Conditional CIF Integral", "Marginal SF"]
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
print("\n=== Summary of Test Results of HACSurv(Independent) Across All Seeds ===")

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
