import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def get_optimizer(models, lr, optimizer, **kwargs):
	parameters = list(models.parameters())

	if optimizer == 'Adam':
		return torch.optim.Adam(parameters, lr=lr, **kwargs)
	elif optimizer == 'SGD':
		return torch.optim.SGD(parameters, lr=lr, **kwargs)
	elif optimizer == 'RMSProp':
		return torch.optim.RMSprop(parameters, lr=lr, **kwargs)
	else:
		raise NotImplementedError('Optimizer '+optimizer+' is not implemented')

def train_nfg(model, total_loss,
			  x_train, t_train, e_train,
			  x_valid, t_valid, e_valid,
			  n_iter = 1000, lr = 1e-3, weight_decay = 0.001,
			  bs = 100, patience_max = 3, cuda = False):
	# Separate oprimizer as one might need more time to converge
	optimizer = get_optimizer(model, lr, model.optimizer, weight_decay = weight_decay)
	
	patience, best_loss, previous_loss = 0, np.inf, np.inf
	best_param = deepcopy(model.state_dict())
	
	nbatches = int(x_train.shape[0]/bs) + 1
	index = np.arange(len(x_train))
	t_bar = tqdm(range(n_iter))
	for i in t_bar:
		np.random.shuffle(index)
		model.train()
		
		# Train survival model
		for j in range(nbatches):
			xb = x_train[index[j*bs:(j+1)*bs]]
			tb = t_train[index[j*bs:(j+1)*bs]]
			eb = e_train[index[j*bs:(j+1)*bs]]
			
			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

			optimizer.zero_grad()
			loss = total_loss(model,
							  xb,
							  tb,
							  eb) 
			loss.backward()
			optimizer.step()

		model.eval()
		xb, tb, eb = x_valid, t_valid, e_valid
		if cuda:
			xb, tb, eb  = xb.cuda(), tb.cuda(), eb.cuda()

		valid_loss = total_loss(model,
								xb,
								tb,
								eb).item() 
		t_bar.set_description("Loss: {:.3f}".format(valid_loss))
		if valid_loss < previous_loss:
			patience = 0

			if valid_loss < best_loss:
				best_loss = valid_loss
				best_param = deepcopy(model.state_dict())

		elif patience == patience_max:
			break
		else:
			patience += 1

		previous_loss = valid_loss

	model.load_state_dict(best_param)
	return model, i
