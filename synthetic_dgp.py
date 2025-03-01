
#####以下代码和513csv数据生成的参数应该一致。
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, random_split
import torch
import numpy as np
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula, FrankCopula, ClaytonCopula, IndependenceCopula)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.backends.cudnn.allow_tf32 = False
# def load_data(path, num_train=None, num_test=None):
#     '''
#     Loads dataset from `path` split into Pytorch train and test of 
#     given sizes. Train set is taken from the front while
#     test set is taken from behind.

#     :param path: path to .p file containing data.
#     '''
#     f = open(path, 'rb')
#     all_data = pickle.load(f)['samples']

#     ndata_all = all_data.size()[0]
    
#     if (num_train is None) and (num_test is None):
#         num_train = np.floor(all_data*2/3)
#         num_test = np.floor(all_data/3)
#     elif (num_train is None) and (num_test==0):
#         num_train = ndata_all
        
#     assert num_train+num_test <= ndata_all

#     train_data = all_data[:num_train]
#     test_data = all_data[(ndata_all-num_test):]

#     return train_data, test_data
# train_data, test_data = load_data('/home/liuxin/GenACSurvival-main/hac_clayton_dim4copy.p', 2000, 1000)



def LOG(x):
    return np.log(x+1e-20*(x<1e-20))

# Generate according to Algorithm 2 in "Copula-based Deep Survival Models for Dependent Censoring"
def inverse_transform(value, risk, shape, scale):
    return (-LOG(value)/np.exp(risk))**(1/shape)*scale       
    # return (-np.log(1-value)/np.exp(risk))**(1/shape)*scale

def linear_dgp(copula_name='Frank', sample_size=2000, covariate_dim=10, theta=10, rng=np.random.default_rng(), verbose=True):
    # Generate synthetic data (time-to-event and censoring indicator)
    v_e = 4; rho_e = 14; v_c = 3; rho_c = 16

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = rng.uniform(0, 1, (sample_size, covariate_dim))
    # generate censoring risk coefficient beta from 10 dimensional uniform distribution from 0 to 1
    beta_c = rng.uniform(0, 1, (covariate_dim, ))
    # generate event risk coefficient beta_e from 10 dimensional uniform distribution from 0 to 1
    beta_e = rng.uniform(0, 1, (covariate_dim,))
    # multiply beta_e with X to get event risk
    event_risk = np.matmul(X, beta_e).squeeze()
    # multiply beta_c with X to get censoring risk
    censoring_risk = np.matmul(X, beta_c).squeeze()

    if copula_name == 'Frank':
        copula = FrankCopula(theta=theta)
    elif copula_name == 'Gumbel':
        copula = GumbelCopula(theta=theta)
    elif copula_name == 'Clayton':
        copula = ClaytonCopula(theta=theta)
    elif copula_name == "Independent":
        copula = IndependenceCopula()
    else:
        raise ValueError('Copula not implemented')
        
    sample = copula.rvs(sample_size, random_state=rng)
    u = sample[:, 0]
    v = sample[:, 1]
    
    plt.scatter(u[:2000], v[:2000], s=15)
    # 生成包含 copula_name 和 theta 的文件名
    filename = f'./single_figure/singleRisk_{copula_name}_{theta}.png'
    plt.savefig(filename)
    print(f"图像已保存至 {filename}")

    event_time = inverse_transform(u, event_risk, v_e, rho_e)
    censoring_time = inverse_transform(v, censoring_risk, v_c, rho_c)

    # create observed time 
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time < censoring_time).astype(int)

    return X, observed_time, event_indicator, event_time, censoring_time, beta_e


def linear_dgp_hac( copula_name= 'Frank', sample_size = 30000, covariate_dim= 10, theta=10, rng = np.random.default_rng(), verbose=True):
    # Generate synthetic data (time-to-event and censoring indicator)

    #  513数据的margin参数
    # v越大越陡峭    p越小活越少

    v_e1 = 6; rho_e1 = 15  # 第一个竞争事件的形状和尺度参数
    v_e2 = 5; rho_e2 = 14 # 第二个竞争事件的形状和尺度参数
    
    v_e3 = 4; rho_e3 = 19 # 第三个竞争事件的形状和尺度参数
    v_c = 3; rho_c = 20 # 删失的形状和尺度参数


    # # # # v越大越陡峭    p越小活越少
    # v_e1 = 4; rho_e1 = 14  # 第一个竞争事件的形状和尺度参数
    # v_e2 = 4; rho_e2 = 14 # 第二个竞争事件的形状和尺度参数
    
    # v_e3 = 4; rho_e3 = 14 # 第三个竞争事件的形状和尺度参数
    # v_c = 4; rho_c = 14 # 删失的形状和尺度参数

    # v_e=4; rho_e=17; v_c=3; rho_c=16

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = rng.uniform(0, 1, (sample_size, covariate_dim))
    # generate censoring risk coefficient beta from 10 dimensional uniforma distribution from 0 to 1
    beta_c = rng.uniform(0, 1, (covariate_dim, ))
    # generate event risk coefficient beta_e from 10 dimensional uniforma distribution from 0 to 1
    beta_e1 = rng.uniform(0, 1, (covariate_dim,))
    beta_e2 = rng.uniform(0, 1, (covariate_dim,))
    beta_e3 = rng.uniform(0, 1, (covariate_dim,))
    # multiply beta_e with X to get event risk
    event_risk1 = np.matmul(X, beta_e1).squeeze()
    event_risk2 = np.matmul(X, beta_e2).squeeze()
    event_risk3 = np.matmul(X, beta_e3).squeeze()
    # multiple beta_c with X to get censoring risk
    censoring_risk = np.matmul(X, beta_c).squeeze()

    if copula_name=='Frank':
        copula = FrankCopula(theta=theta)
    elif copula_name=='Gumbel':
        copula = GumbelCopula(theta=theta)
    elif copula_name=='Clayton':
        copula = ClaytonCopula(theta=theta)
    elif copula_name=="Independent":
        copula = IndependenceCopula()
    else:
        raise ValueError('Copula not implemented')
    sample = copula.rvs(sample_size, random_state=rng)
    # u1 = sample[:, 0]
    # print(u1.shape)
    # v = sample[:, 1]
    # u1 = train_data[:, 0].detach().numpy()
    # u2 = train_data[:, 1].detach().numpy()
    # v1 = train_data[:, 2].detach().numpy()
    # v2 = train_data[:, 3].detach().numpy()

    train_data= np.loadtxt('./U_data.csv', delimiter=',')
    # train_data=train_data[:3000]
    # print(train_data.shape)

    u1 = train_data[:, 0]
    u2 = train_data[:, 1]
    v1 = train_data[:, 2]
    v2 = train_data[:, 3]
    # plt.scatter(u1[:3000], u2[:3000], s=15)
    # # os.makedirs('/home/liuxin/GenACSurvival-main/sample_figs/'+copula_form+'/'+str(theta_true), exist_ok=True)
    # plt.savefig('/home/liuxin/HACSurv_Camera_Ready/Competing_SYN/HACsyn_01_true.png' )

    # print(u.shape)
    event_time1 = inverse_transform(u2, event_risk1, v_e1, rho_e1)
    event_time2 = inverse_transform(v1, event_risk2, v_e2, rho_e2)
    event_time3 = inverse_transform(v2, event_risk3, v_e3, rho_e3)
    censoring_time = inverse_transform(u1, censoring_risk, v_c, rho_c)

    # create observed time 
    # 计算观察时间和事件指标
    observed_time = np.minimum.reduce([censoring_time,event_time1, event_time2, event_time3])
    # 计算事件发生的索引，删失为0，其他事件类型为1, 2, 3
    event_times = np.stack([censoring_time, event_time1, event_time2, event_time3])
    # print(event_times)
    event_indicator = np.argmin(event_times, axis=0)
    # event_indicator = np.argmin([censoring_time, event_time1, event_time2, event_time3], axis=0)


    return X, observed_time, event_indicator, event_time1, censoring_time, beta_e1,beta_e2,beta_e3


def nonlinear_dgp( copula_name= 'Frank', sample_size = 30000, theta=10, rng = np.random.default_rng(142857), verbose=True):
    # Generate synthetic data (time-to-event and censoring indicator)
    v_e=4; rho_e=17; v_c=3; rho_c=16

    # generate X from 10 dimensional uniform distribution from 0 to 1
    X = np.random.uniform(0, 1, (sample_size, 1))
    # # generate censoring risk coefficient beta from 10 dimensional uniforma distribution from 0 to 1
    # beta_c = np.random.uniform(0, 1, (covariate_dim, ))
    # # generate event risk coefficient beta_e from 10 dimensional uniforma distribution from 0 to 1
    # beta_e = np.random.uniform(0, 1, (covariate_dim,))
    # multiply beta_e with X to get event risk
    event_risk = 2* np.sin(X*np.pi).squeeze() 
    # multiple beta_c with X to get censoring risk
    censoring_risk = 2* np.sin(X*np.pi+0.5).squeeze()

    if copula_name=='Frank':
        copula = FrankCopula(theta=theta)
    elif copula_name=='Gumbel':
        copula = GumbelCopula(theta=theta)
    elif copula_name=='Clayton':
        copula = ClaytonCopula(theta=theta)
    elif copula_name=="Independent":
        copula = IndependenceCopula()
    else:
        raise ValueError('Copula not implemented')
    sample = copula.rvs(sample_size, random_state=rng)
    u = sample[:, 0]
    v = sample[:, 1]

    event_time = inverse_transform(u, event_risk, v_e, rho_e)
    censoring_time = inverse_transform(v, censoring_risk, v_c, rho_c)
    
    # create observed time 
    observed_time = np.minimum(event_time, censoring_time)
    event_indicator = (event_time<censoring_time).astype(int)

    return X, observed_time, event_indicator, event_time, censoring_time