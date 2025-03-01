import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os
from datetime import datetime
sys.path.append('/home/liuxin/HACSurv')
import warnings
warnings.filterwarnings("ignore")

# 导入您需要的模块
from survival import MixExpPhiStochastic, HACSurv_4D_Sym_shared, sample
from truth_net import Weibull_linear
from metric import surv_diff
from pycox.evaluation import EvalSurv
import torch.optim as optim

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(24)
torch.set_default_tensor_type(torch.DoubleTensor)

# 定义种子列表
# seeds_list = [41, 42, 43, 44, 45]
seeds_list = [41]
# 用于保存每个种子的结果
all_results = []

# 遍历每个种子
for seeds in seeds_list:
    print(f"Running experiment with seed: {seeds}")

    # 从本地文件读取数据集
    df = pd.read_csv('./MylinearSyndata_513.csv')

    # 使用当前的种子分割数据集
    df_test = df.sample(frac=0.2, random_state=seeds)
    df_train = df.drop(df_test.index)
    df_val = df_train.sample(frac=0.2, random_state=seeds)
    df_train = df_train.drop(df_val.index)

    # 定义获取特征和标签的函数
    get_x = lambda df: (df.drop(columns=['observed_time', 'event_indicator']).values.astype('float32'))
    get_target = lambda df: (df['observed_time'].values, df['event_indicator'].values)

    # 准备数据
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

    # 定义模型
    phi = MixExpPhiStochastic(device)
    model = HACSurv_4D_Sym_shared(phi, device=device, num_features=covariate_tensor_train.shape[1], tol=1e-14, hidden_size=100).to(device)

    # 定义优化器
    optimizer_out = optim.Adam([
        {"params": model.shared_embedding.parameters(), "lr": 1e-4},
        {"params": model.sumo_e1.parameters(), "lr": 1e-4},
        {"params": model.sumo_e2.parameters(), "lr": 1e-4},
        {"params": model.sumo_e3.parameters(), "lr": 1e-4},
        {"params": model.sumo_c.parameters(), "lr": 1e-4},
        {"params": model.phi.parameters(), "lr": 8e-4},
    ], weight_decay=0)

    # 定义早停和模型保存参数
    def current_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    best_avg_c_index = float('-inf')
    best_val_loglikelihood = float('-inf')
    epochs_no_improve = 0
    num_epochs = 100000
    early_stop_epochs = 1600
    base_path = "/home/liuxin/HACSurv_Camera_Ready/Competing_SYN/checkpoint"
    best_model_filename = ""

    # 定义评价指标计算函数
    def calculate_metrics_CIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
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
            t_numpy = times_tensor_val.cpu().numpy()
            c_numpy = (event_indicator_tensor_val == (event_index + 1)).cpu().numpy()
            eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
            c_indexes.append(eval_surv.concordance_td())
            ibses.append(eval_surv.integrated_brier_score(times))
        return c_indexes, ibses

    def calculate_metrics_SF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
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

    def calculate_metrics_jointCIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
        c_indexes = []
        ibses = []
        step = 1
        times = np.arange(0, times_tensor_train.max().cpu()+step , step)
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
            eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
            c_indexes.append(eval_surv.concordance_td())
            ibses.append(eval_surv.integrated_brier_score(times))
        return c_indexes, ibses

    def calculate_metrics_Conditional_CIF_integral(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val):
        c_indexes = []
        ibses = []
        step = 1
        times = np.arange(0, times_tensor_train.max().cpu()+step , step)
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
            eval_surv = EvalSurv(surv_df, t_numpy, c_numpy, censor_surv='km')
            c_indexes.append(eval_surv.concordance_td())
            ibses.append(eval_surv.integrated_brier_score(times))
        return c_indexes, ibses

    # 训练模型
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

            # 计算验证集上的指标
            c_indexes_val_CIF, ibses_val_CIF = calculate_metrics_CIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
            # c_indexes_val_SF, ibses_val_SF = calculate_metrics_SF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
            # c_indexes_val_jointCIF, ibses_val_jointCIF = calculate_metrics_jointCIF(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)
            # c_indexes_val_Con_integral_CIF, ibses_val_Con_integral_CIF = calculate_metrics_Conditional_CIF_integral(times_tensor_train, model, device, times_tensor_val, covariate_tensor_val, event_indicator_tensor_val)

            # print('SF               ', c_indexes_val_SF, ibses_val_SF)
            # print('joint_CIF        ', c_indexes_val_jointCIF, ibses_val_jointCIF)
            # print('Con_integral_CIF ', c_indexes_val_Con_integral_CIF, ibses_val_Con_integral_CIF)
            print('CIF              ', c_indexes_val_CIF, ibses_val_CIF)

            # 使用 c_indexes_val_CIF 的平均值进行早停
            avg_c_index = np.mean(c_indexes_val_CIF)

            # 检查是否为最佳模型
            if avg_c_index > best_avg_c_index:
                best_avg_c_index = avg_c_index

                # 保存最佳模型
                if best_model_filename:
                    os.remove(os.path.join(base_path, best_model_filename))  # 删除旧的最佳模型文件

                # best_model_filename = f"Independent_BestModel_cindex_{best_avg_c_index:.4f}_{current_time()}seed{seeds}.pth"
                best_model_filename = f"Outer_Symmetry_SYN_cindex_{best_avg_c_index:.4f}_{current_time()}seed{seeds}.pth"
                torch.save(model.state_dict(), os.path.join(base_path, best_model_filename))
                epochs_no_improve = 0

                # 绘制并保存图像
                samples = sample(model, 2, 3000, device =  device)
                plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                plt.savefig(f'./Competing_SYN/figure/Outer_Symmetry_SYN.png' )
                plt.clf()

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

    # 计算测试集上的对数似然
    test_loglikelihood = model(covariate_tensor_test, times_tensor_test, event_indicator_tensor_test, max_iter=1000)
    print(f"Test loglikelihood: {test_loglikelihood.item()}")

    # 计算测试集上的指标
    c_indexes_test_CIF, ibses_test_CIF = calculate_metrics_CIF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
    c_indexes_test_SF, ibses_test_SF = calculate_metrics_SF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
    c_indexes_test_jointCIF, ibses_test_jointCIF = calculate_metrics_jointCIF(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)
    c_indexes_test_Con_integral_CIF, ibses_test_Con_integral_CIF = calculate_metrics_Conditional_CIF_integral(times_tensor_train, model, device, times_tensor_test, covariate_tensor_test, event_indicator_tensor_test)

    print('Seed:', seeds)
    print('Test Metrics:')
    print('SF               ', c_indexes_test_SF, ibses_test_SF)
    print('joint_CIF        ', c_indexes_test_jointCIF, ibses_test_jointCIF)
    print('Con_integral_CIF ', c_indexes_test_Con_integral_CIF, ibses_test_Con_integral_CIF)
    print('CIF              ', c_indexes_test_CIF, ibses_test_CIF)

    # 保存结果
    all_results.append({
        'seed': seeds,
        'test_loglikelihood': test_loglikelihood.item(),
        'c_indexes_test_SF': c_indexes_test_SF,
        'ibses_test_SF': ibses_test_SF,
        'c_indexes_test_jointCIF': c_indexes_test_jointCIF,
        'ibses_test_jointCIF': ibses_test_jointCIF,
        'c_indexes_test_Con_integral_CIF': c_indexes_test_Con_integral_CIF,
        'ibses_test_Con_integral_CIF': ibses_test_Con_integral_CIF,
        'c_indexes_test_CIF': c_indexes_test_CIF,
        'ibses_test_CIF': ibses_test_CIF,
    })

# 在所有种子上输出结果
print("\nAll results:")
for result in all_results:
    seed = result['seed']
    print(f"\nSeed: {seed}")
    print(f"Test loglikelihood: {result['test_loglikelihood']}")
    print('Test Metrics:')
    print('SF               ', result['c_indexes_test_SF'], result['ibses_test_SF'])
    print('joint_CIF        ', result['c_indexes_test_jointCIF'], result['ibses_test_jointCIF'])
    print('Con_integral_CIF ', result['c_indexes_test_Con_integral_CIF'], result['ibses_test_Con_integral_CIF'])
    print('CIF              ', result['c_indexes_test_CIF'], result['ibses_test_CIF'])
