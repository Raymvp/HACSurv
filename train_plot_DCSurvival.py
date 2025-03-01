from matplotlib import pyplot as plt
from synthetic_dgp import linear_dgp

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os
import numpy as np
# sys.path.append(os.path.abspath('../'))
from dirac_phi import DiracPhi
from survival import HACSurv_2D
from survival import sample
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_num_threads(24)

device = torch.device("cuda:0")
batch_size = 3000
num_epochs = 10000
copula_form = 'Clayton'
sample_size = 3000
val_size = 1000
seed = 142857
rng = np.random.default_rng(seed)

def main():
    for theta_true in [2,5,10]:
        X, observed_time, event_indicator, _, _, _ = linear_dgp( copula_name=copula_form, covariate_dim=10, theta=theta_true, sample_size=sample_size, rng=rng)
        times_tensor = torch.tensor(observed_time, dtype=torch.float64).to(device)
        event_indicator_tensor = torch.tensor(event_indicator, dtype=torch.float64).to(device)
        covariate_tensor = torch.tensor(X, dtype=torch.float64).to(device)
        train_data = TensorDataset(covariate_tensor[0:sample_size-val_size], times_tensor[0:sample_size-val_size], event_indicator_tensor[0:sample_size-val_size])
        val_data = TensorDataset(covariate_tensor[sample_size-val_size:], times_tensor[sample_size-val_size:], event_indicator_tensor[sample_size-val_size:])

        train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size= batch_size, shuffle=True)

        # Early stopping
        best_val_loglikelihood = float('-inf')
        epochs_no_improve = 0
        early_stop_epochs = 100

        # Parameters for ACNet
        depth = 2
        widths = [100, 100]
        lc_w_range = (0, 1.0)
        shift_w_range = (0., 2.0)

        phi = DiracPhi(depth, widths, lc_w_range, shift_w_range, device, tol = 1e-12).to(device)
        model = HACSurv_2D(phi, device = device, num_features=10, tol=1e-12).to(device)
        # separately optimize copula and survival parameters is sometimes helpful, but not necessary
        # optimizer_survival = optim.Adam([{"params": model.sumo_e.parameters(), "lr": 0.001},
        #                             {"params": model.sumo_c.parameters(), "lr": 0.001},
        #                         ])
        # optimizer_copula = optim.SGD([{"params": model.phi.parameters(), "lr": 0.0005}])
        optimizer_survival = optim.AdamW([{"params": model.sumo_e.parameters(), "lr": 0.0001},
                                    {"params": model.sumo_c.parameters(), "lr": 0.0001},
                                ])
        optimizer_copula = optim.AdamW([{"params": model.phi.parameters(), "lr": 0.0001}])

        train_loss_per_epoch = []
        print("Start training!")

        checkpoint_dir = './checkpoints/DCSurvival/'
        sample_figs_dir = './single_figure/DCSurvival/' + copula_form + '/' + str(theta_true) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if not os.path.exists(sample_figs_dir):
            os.makedirs(sample_figs_dir)

        for epoch in range(num_epochs):
            loss_per_minibatch = []
            for i, (x , t, c) in enumerate(train_loader, 0):
                optimizer_copula.zero_grad()
                optimizer_survival.zero_grad()

                p = model(x, t, c, max_iter = 10000)
                logloss = -p
                logloss.backward() 
                scalar_loss = (logloss/p.numel()).detach().cpu().numpy().item()

                optimizer_survival.step()
                
                if epoch > 200:
                    optimizer_copula.step()
                # optimizer_censoring.step()
                # optimizer.step()

                loss_per_minibatch.append(scalar_loss/batch_size)
            train_loss_per_epoch.append(np.mean(loss_per_minibatch))
            if epoch % 1 == 0:
                print('Training likilihood at epoch %s: %.5f' %
                        (epoch, -train_loss_per_epoch[-1]))
                # Check if validation loglikelihood has improved
                for i, (x_val, t_val, c_val) in enumerate(val_loader, 0):
                    val_loglikelihood = model(x_val, t_val, c_val, max_iter = 10000)/val_size
                print('Validation log-likelihood at epoch %s: %s' % (epoch, val_loglikelihood.cpu().detach().numpy().item()))
                if val_loglikelihood > best_val_loglikelihood:
                    best_val_loglikelihood = val_loglikelihood
                    epochs_no_improve = 0
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': best_val_loglikelihood,
                    }, './checkpoints/DCSurvival/checkpoint_experiment_'+copula_form + '_' +str(theta_true)+'.pth')
                
                else:
                    epochs_no_improve += 1
                # Early stopping condition
                if epochs_no_improve == early_stop_epochs:
                    print('Early stopping triggered at epoch: %s' % epoch)
                    break
            # Plot Samples from the learned copula
            if epoch % 200 == 0:
                print('Scatter sampling')
                samples = sample(model, 2, 2000, device =  device)
                plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=15)
                plt.savefig('./single_figure/DCSurvival/'+copula_form+'/'+str(theta_true)+'/epoch%s.png' %
                            (epoch))
                plt.clf()

        checkpoint = torch.load('./checkpoints/DCSurvival/checkpoint_experiment_'+copula_form + '_' +str(theta_true)+'.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        samples =  sample(model, 2, 2000, device =  device)
        plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s = 15)
        plt.savefig('./single_figure/DCSurvival/'+copula_form+'/'+str(theta_true)+'/best_epoch.png')
        plt.clf()


if __name__ == '__main__':
    main()
