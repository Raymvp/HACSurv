import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
import os
import sys
import torch.optim as optim
from tqdm import tqdm
import multiprocessing
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Function to train copula for a given pair of events on a specified device
def train_copula_for_pair(pair_device):
    pair, device_id = pair_device
    i, j = pair
    selected_indicators = [i, j]
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training for event pair ({i}, {j}) on device {device}")

    # Fixed random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    df = pd.read_csv('./mimic_time_data_withoutETT.csv')
    
    # Data splitting
    df_test = df.sample(frac=0.2, random_state=42)
    df_train = df.drop(df_test.index)
    df_val = df_train.sample(frac=0.2, random_state=42)
    df_train = df_train.drop(df_val.index)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Extract features
    x_train = df_train.drop(columns=['time', 'death_reason', 'label']).values.astype('float32')
    x_val = df_val.drop(columns=['time', 'death_reason', 'label']).values.astype('float32')
    x_test = df_test.drop(columns=['time', 'death_reason', 'label']).values.astype('float32')
    
    # 对训练数据进行拟合和转换
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)

    # # 使用相同的 scaler 对验证集和测试集进行转换
    # x_val = scaler.transform(x_val)
    # x_test = scaler.transform(x_test)
    # Extract time and event labels
    get_target = lambda df: (df['time'].values, df['death_reason'].values)
    time_train, event_train = get_target(df_train)
    time_val, event_val = get_target(df_val)
    time_test, event_test = get_target(df_test)
    
    # Convert data to tensors and move to device
    covariate_tensor_train = torch.tensor(x_train, dtype=torch.float64).to(device)
    covariate_tensor_val = torch.tensor(x_val, dtype=torch.float64).to(device)
    covariate_tensor_test = torch.tensor(x_test, dtype=torch.float64).to(device)
    
    times_tensor_train = torch.tensor(time_train, dtype=torch.float64).to(device)/10.0
    event_indicator_tensor_train = torch.tensor(event_train, dtype=torch.float64).to(device)
    times_tensor_val = torch.tensor(time_val, dtype=torch.float64).to(device)/10.0
    event_indicator_tensor_val = torch.tensor(event_val, dtype=torch.float64).to(device)
    times_tensor_test = torch.tensor(time_test, dtype=torch.float64).to(device)
    event_indicator_tensor_test = torch.tensor(event_test, dtype=torch.float64).to(device)
    

    # # 获取 times_tensor_train 的最大值
    # max_time_train = times_tensor_train.max()

    # # 归一化 train, val, test 的时间张量
    # times_tensor_train = times_tensor_train / max_time_train
    # times_tensor_val = times_tensor_val / max_time_train
    # times_tensor_test = times_tensor_test / max_time_train

    # # 打印最大值以确认归一化
    # print(f'Max time (train) used for normalization: {max_time_train}')
    # print(times_tensor_val)




    torch.set_num_threads(16)
    torch.set_default_tensor_type(torch.DoubleTensor)
    # device is already set
    depth = 2
    widths = [10, 10]
    lc_w_range = (0, 1.0)
    shift_w_range = (0., 2.0)
    num_epochs = 100000
    batch_size = 1000
    early_stop_epochs = 400
    
    # model_dir and figures_dir for saving
    model_dir = './MIMIC-III/checkpoint'
    figures_dir = './MIMIC-III/figure'
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    from dirac_phi import DiracPhi
    from survival import MixExpPhiStochastic,InnerGenerator,InnerGenerator2,HACSurv_2D_shared,sample
    
    phi = MixExpPhiStochastic(device)
    model = HACSurv_2D_shared(phi, device=device, num_features=x_train.shape[1], tol=1e-14, hidden_size=100).to(device)
    optimizer = optim.AdamW([{"params": model.sumo_e.parameters(), "lr": 3e-4},
                             {"params": model.sumo_c.parameters(), "lr": 3e-4},
                            ])
    optimizer_phi = optim.AdamW([
                             {"params": model.phi.parameters(), "lr": 6e-4}
                            ])
    best_val_loglikelihood = float('-inf')
    epochs_no_improve = 0
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        optimizer_phi.zero_grad()
        model.phi.resample_M(200)
    
        logloss = model(covariate_tensor_train, times_tensor_train, event_indicator_tensor_train, max_iter=10000, selected_indicators=selected_indicators)
        (-logloss).backward() 
    
        optimizer.step()
        if epoch > 100:
            optimizer_phi.step()
        if epoch % 30 == 0 and epoch > 0:
            # Validation and logging
            print(f"Pair ({i},{j}) - Epoch {epoch}, Train loglikelihood: {logloss.item()}")
            model.phi.resample_M(200)
            val_loglikelihood = model(covariate_tensor_val, times_tensor_val, event_indicator_tensor_val, max_iter=10000, selected_indicators=selected_indicators)
            print(f"Pair ({i},{j}) - Validation likelihood: {val_loglikelihood.item()}")
    
            # Model checkpointing
            if val_loglikelihood > best_val_loglikelihood:
                best_val_loglikelihood = val_loglikelihood
                epochs_no_improve = 0
                indicators_str = ''.join(map(str, selected_indicators))
                checkpoint_path = os.path.join(model_dir, f'MIMIC_e{indicators_str}__adamw_1e-4.pth')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': best_val_loglikelihood}, checkpoint_path)
    
                # Generate and save plots
                print(f'Pair ({i},{j}) - Scatter sampling')
                samples = sample(model, 2, 2000, device=device)
                plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                plot_path = os.path.join(figures_dir, f'MIMIC_e{indicators_str}__adamw_1e-4.png')
                plt.savefig(plot_path)
                plt.clf()
    
            else:
                epochs_no_improve += 30
    
            # Early stopping condition
            if epochs_no_improve >= early_stop_epochs:
                print(f'Pair ({i},{j}) - Early stopping triggered at epoch: {epoch}')
                break

def combine_images_triangle():
    num_events = 6  # 事件数量 6（事件编号 0～5）
    figures_dir = './MIMIC-III/figure'
    
    # 使用“01”图片作为样本，获取图像尺寸
    sample_path = os.path.join(figures_dir, "MIMIC_e01__adamw_1e-4.png")
    try:
        sample_img = Image.open(sample_path)
    except Exception as e:
        print(f"加载样本图片 {sample_path} 失败: {e}")
        return
    img_width, img_height = sample_img.size
    
    grid_cols = num_events - 1  # 列数为事件数减 1
    grid_rows = num_events - 1  # 行数同样取最大对数（即第一列的图片数量）
    
    composite_img = Image.new("RGB", (grid_cols * img_width, grid_rows * img_height))
    
    # 遍历每一列，列号 i 对应事件 i
    for i in range(num_events - 1):
        # 该列图片数量
        count = num_events - i - 1  
        # 顶部空白行数：总行数减去该列图片数
        empty_rows = grid_rows - count  
        # 对于事件 i，遍历 j 从 i+1 到 num_events-1
        for j in range(i+1, num_events):
            # 在该列中，按照 j 的顺序排列，从空白行开始填充
            row = empty_rows + (j - i - 1)
            indicators_str = f"{i}{j}"
            image_path = os.path.join(figures_dir, f"MIMIC_e{indicators_str}__adamw_1e-4.png")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    # 若尺寸不一致，按样本图大小调整
                    if img.size != (img_width, img_height):
                        img = img.resize((img_width, img_height), Image.LANCZOS)
                    # 粘贴位置：x 为当前列偏移，y 为对应行偏移
                    composite_img.paste(img, (i * img_width, row * img_height))
                except Exception as e:
                    print(f"加载图片 {image_path} 时出错: {e}")
            else:
                print(f"未找到图片: {image_path}")
    
    composite_image_path = os.path.join(figures_dir, 'combined_copulas_triangle.png')
    composite_img.save(composite_image_path)
    print(f'拼接后的组合图片已保存到: {composite_image_path}')


if __name__ == "__main__":
    num_events = 6  # events numbered from 0 to 5
    devices = [0,2,3]
    processes_per_device = 4
    device_processes = {device_id: [] for device_id in devices}
    pairs = []
    for i in range(num_events):
        for j in range(i+1, num_events):
            pairs.append((i,j))
    index = 0
    total_pairs = len(pairs)
    import time

    while index < total_pairs or any(len(procs) > 0 for procs in device_processes.values()):
        # Check and remove finished processes
        for device_id in devices:
            device_procs = device_processes[device_id]
            for p in device_procs[:]:  # Copy the list to avoid modification during iteration
                if not p.is_alive():
                    p.join()
                    device_procs.remove(p)
                    print(f"Process on device {device_id} finished. Active processes on device {device_id}: {len(device_procs)}")
        # Start new processes if possible
        for device_id in devices:
            while len(device_processes[device_id]) < processes_per_device and index < total_pairs:
                pair = pairs[index]
                index +=1
                p = multiprocessing.Process(target=train_copula_for_pair, args=((pair, device_id),))
                p.start()
                device_processes[device_id].append(p)
                print(f"Started process for pair {pair} on device {device_id}. Active processes on device {device_id}: {len(device_processes[device_id])}")
        # If all processes are running and we have more pairs to process, wait for a short time
        if index >= total_pairs and all(len(procs) == 0 for procs in device_processes.values()):
            break
        time.sleep(1)  # Wait for a short time before checking again

    # Wait for any remaining processes
    for device_id in devices:
        for p in device_processes[device_id]:
            p.join()
    # After all processes are done, combine images
    combine_images_triangle()
