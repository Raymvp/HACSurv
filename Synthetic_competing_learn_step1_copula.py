
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from datetime import datetime
import itertools
import multiprocessing
from multiprocessing import Pool
from PIL import Image  # 用于后续拼图
multiprocessing.set_start_method('spawn', force=True)
from survival import HACSurv_2D_shared, MixExpPhiStochastic, sample,HACSurv_2D

# 在代码开始运行时生成唯一时间戳
start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# 设置全局配置
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_num_threads(24)
device = torch.device("cuda:3")

# 参数设置
batch_size = 2000
num_epochs = 10000
seed = 142857
# 修改保存路径：checkpoint 与图像
checkpoint_dir = './Competing_SYN/checkpoint'
figure_dir = './Competing_SYN/figure'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

def main(selected_indicators):
    # print("Running experiment with indicators:", selected_indicators)
    print("Running experiment with indicators:", selected_indicators)
    # ===== 数据读取 =====
    csv_path = '/home/liuxin/HACSurv_Camera_Ready/MylinearSyndata_513.csv'
    # csv_path = '/home/liuxin/generated_data_camera_ready.csv'
    df = pd.read_csv(csv_path)
    print("CSV 数据读取完毕，形状：", df.shape)

    # 先筛选出符合 selected_indicators 的样本
    df = df[df['event_indicator'].isin(selected_indicators)]
    print("筛选后的数据形状：", df.shape)

    df = df.iloc[:3000]
    print("只取前3000个样本，形状：", df.shape)


    # 按照固定随机种子划分训练集与验证集（验证集占20%）
    df_val = df.sample(frac=0.3, random_state=seed)
    df_train = df.drop(df_val.index)
    # print(df_val)
    # 定义获取特征与标签的 lambda 函数
    get_x = lambda df: df.drop(columns=['observed_time', 'event_indicator']).values.astype('float32')
    get_target = lambda df: (df['observed_time'].values, df['event_indicator'].values)

    X_train = get_x(df_train)
    t_train, c_train = get_target(df_train)
    X_val = get_x(df_val)
    t_val, c_val = get_target(df_val)

    # 转为 tensor，并移动到指定 device 上
    covariate_tensor_train = torch.tensor(X_train, dtype=torch.float64).to(device)
    times_tensor_train = torch.tensor(t_train, dtype=torch.float64).to(device)
    event_indicator_tensor_train = torch.tensor(c_train, dtype=torch.float64).to(device)

    covariate_tensor_val = torch.tensor(X_val, dtype=torch.float64).to(device)
    times_tensor_val = torch.tensor(t_val, dtype=torch.float64).to(device)
    event_indicator_tensor_val = torch.tensor(c_val, dtype=torch.float64).to(device)

    # 构建 TensorDataset
    train_data = TensorDataset(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train)
    val_data = TensorDataset(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val)

    # 筛选 selected_indicators 对应的数据（冗余操作，此时所有样本均符合条件）
    train_mask = np.isin(train_data.tensors[2].cpu().numpy(), selected_indicators)
    train_data_filtered = TensorDataset(train_data.tensors[0][train_mask],
                                        train_data.tensors[1][train_mask],
                                        train_data.tensors[2][train_mask])

    val_mask = np.isin(val_data.tensors[2].cpu().numpy(), selected_indicators)
    val_data_filtered = TensorDataset(val_data.tensors[0][val_mask],
                                      val_data.tensors[1][val_mask],
                                      val_data.tensors[2][val_mask])

    # 输出训练集中每种事件的样本数
    unique_events, counts = np.unique(train_data_filtered.tensors[2].cpu().numpy(), return_counts=True)
    for event, count in zip(unique_events, counts):
        print(f"Event {event}: {count} samples")

    # 构建 DataLoader
    train_loader = DataLoader(train_data_filtered, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_filtered, batch_size=batch_size, shuffle=True)
    # print(val_loader)
    # ===== 模型与优化器 =====
    phi = MixExpPhiStochastic(device)
    # model = HACSurv_2D_shared(phi, device=device, num_features=covariate_tensor_train.shape[1], tol=1e-12).to(device)
    model = HACSurv_2D(phi, device=device, num_features=covariate_tensor_train.shape[1], tol=1e-12).to(device)
    
    
    optimizer_survival = optim.AdamW([{"params": model.sumo_e.parameters(), "lr": 0.0003},
                                      {"params": model.sumo_c.parameters(), "lr": 0.0003}])
    # optimizer_copula = optim.SGD([{"params": model.phi.parameters(), "lr": 0.0005}])
    optimizer_copula = optim.AdamW([{"params": model.phi.parameters(), "lr": 0.0006}])
    best_val_loglikelihood = float('-inf')
    epochs_no_improve = 0
    early_stop_epochs = 200
    train_loss_per_epoch = []
    indicator_str = "-".join(map(str, selected_indicators))
    best_checkpoint_path = None
    best_figure_path = None
    print("Start training!")
    
    # ===== 训练过程 =====
    for epoch in range(num_epochs):
        loss_per_minibatch = []
        model.train()
        for i, (x, t, c) in enumerate(train_loader, 0):
            optimizer_copula.zero_grad()
            optimizer_survival.zero_grad()
            model.phi.resample_M(500)
            p = model(x, t, c, max_iter=5000, selected_indicators=selected_indicators)
            logloss = -p
            logloss.backward(retain_graph=True)

            # scaleloss = torch.square(torch.mean(model.phi.M)-1)
            # reg_loss = logloss+scaleloss
            # reg_loss.backward(retain_graph=True)
            # 计算标量损失
            scalar_loss = (logloss / p.numel()).detach().cpu().numpy().item()
            optimizer_survival.step()
            if epoch > 50:
                optimizer_copula.step()
            loss_per_minibatch.append(scalar_loss / batch_size)
        
        train_loss_per_epoch.append(np.mean(loss_per_minibatch))
        
        if epoch % 10== 0:
            print('Training likelihood at epoch %s: %.5f' % (epoch, -train_loss_per_epoch[-1]))
            model.eval()
            for i, (x_val, t_val, c_val) in enumerate(val_loader, 0):
                val_loglikelihood = model(x_val, t_val, c_val, max_iter=1000, selected_indicators=selected_indicators) / len(val_data_filtered) 
                val_ll_value = val_loglikelihood.cpu().detach().numpy().item()
            print('Validation log-likelihood at epoch %s: %s' % (epoch, val_ll_value))
            
            if val_loglikelihood > best_val_loglikelihood:
                best_val_loglikelihood = val_loglikelihood
                epochs_no_improve = 0
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{start_time}_experiment_HAC_inner_indicator{indicator_str}_epoch{epoch}.pth')
                if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                    print("Deleted previous best checkpoint:", best_checkpoint_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': best_val_loglikelihood,
                }, checkpoint_path)
                best_checkpoint_path = checkpoint_path
                best_figure_path = os.path.join(figure_dir, f'{start_time}_epoch{epoch}_indicator{indicator_str}.png')
                model.phi.resample_M(1000)
                samples = sample(model, 2, 2000, device=device)
                plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                plt.savefig(best_figure_path)
                plt.clf()
                print("Best checkpoint and image saved to:", best_checkpoint_path, best_figure_path)
            else:
                epochs_no_improve += 10
            
            if epochs_no_improve == early_stop_epochs:
                print('Early stopping triggered at epoch: %s' % epoch)
                model.phi.resample_M(1000)
                samples = sample(model, 2, 2000, device=device)
                final_figure_path = os.path.join(figure_dir, f'{start_time}_epoch{epoch}_indicator{indicator_str}.png')
                plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                plt.savefig(final_figure_path)
                plt.clf()
                if best_figure_path is None:
                    best_figure_path = final_figure_path
                print("Figure saved to:", final_figure_path)
                break
            if epoch % 30 ==0:
                print('Scatter sampling')
                model.phi.resample_M(1000)
                samples = sample(model, 2, 2000, device=device)
                tmp_figure_path = os.path.join(figure_dir, f'{start_time}_epoch{epoch}_indicator{indicator_str}.png')
                plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                plt.savefig(tmp_figure_path)
                plt.clf()
                print("Figure saved to:", tmp_figure_path)
    
    return best_figure_path

def run_experiment_with_indicators(selected_indicators):
    return main(selected_indicators)

def create_composite_image(best_images, figure_dir):
    try:
        img_sample = Image.open(best_images["03"])
        img_width, img_height = img_sample.size
    except Exception as e:
        print("加载图片 03 失败:", e)
        return
    
    composite_img = Image.new("RGB", (3 * img_width, 3 * img_height))
    
    if best_images.get("03") and os.path.exists(best_images["03"]):
        composite_img.paste(Image.open(best_images["03"]), (0, 2 * img_height))
    if best_images.get("13") and os.path.exists(best_images["13"]):
        composite_img.paste(Image.open(best_images["13"]), (img_width, 2 * img_height))
    if best_images.get("23") and os.path.exists(best_images["23"]):
        composite_img.paste(Image.open(best_images["23"]), (2 * img_width, 2 * img_height))
    
    if best_images.get("02") and os.path.exists(best_images["02"]):
        composite_img.paste(Image.open(best_images["02"]), (0, img_height))
    if best_images.get("12") and os.path.exists(best_images["12"]):
        composite_img.paste(Image.open(best_images["12"]), (img_width, img_height))
    
    if best_images.get("01") and os.path.exists(best_images["01"]):
        composite_img.paste(Image.open(best_images["01"]), (0, 0))
    
    composite_image_path = os.path.join(figure_dir, f'Not_shared_embedd_composite_image_{start_time}_AdamW_00003_00006_3000sample.png')
    composite_img.save(composite_image_path)
    print("拼接后的组合图片已保存到:", composite_image_path)
    
    for key in best_images:
        try:
            if os.path.exists(best_images[key]) and key != "03":
                os.remove(best_images[key])
                print(f"已删除旧的图像文件：{best_images[key]}")
        except Exception as e:
            print(f"删除图像 {best_images[key]} 失败:", e)

    return composite_image_path

if __name__ == '__main__':
    combinations_list = list(itertools.combinations([0, 1, 2, 3], 2))
    combination_labels = ["01", "02", "03", "12", "13", "23"]
    # combinations_list = [[2,3]]
    max_processes = 6
    pool = Pool(processes=max_processes)
    results = pool.map(run_experiment_with_indicators, [list(comb) for comb in combinations_list])
    pool.close()
    pool.join()
    
    best_images = dict(zip(combination_labels, results))
    print("各组合最佳图片地址：", best_images)
    
    composite_image_path = create_composite_image(best_images, figure_dir)
