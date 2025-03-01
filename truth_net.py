import torch 
# import matplotlib.pyplot as plt
import math

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def sine(x):
    return 2 * torch.sin(x * math.pi + 0.1)

class Weibull_linear:
    def __init__(self, num_feature, shape, scale, device, coeff = None):
        #torch.manual_seed(0)
        self.num_feature = num_feature
        self.alpha = torch.tensor([scale], device=device).type(torch.float64) # alpha is scale
        self.gamma = torch.tensor([shape], device=device).type(torch.float64) # gamma is shape       
        if coeff is None:
            self.coeff = torch.rand((num_feature,), device=device).type(torch.float64)
        else:
            self.coeff = torch.tensor(coeff, device=device).type(torch.float64)

    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t,x)
    
    def CDF(self ,t ,x):   
        return 1 - self.survival(t,x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t,x))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(torch.matmul(x, self.coeff))
        
    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(torch.matmul(x, self.coeff))
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(torch.matmul(x, self.coeff)))**(1/self.gamma))*self.alpha


class Weibull_nonlinear:
    #torch.manual_seed(0)
    def __init__(self, shape, scale, device):
        #torch.manual_seed(0)
        self.alpha = torch.tensor([scale],device=device).type(torch.float32)
        self.gamma = torch.tensor([shape], device=device).type(torch.float32)
        self.risk_function = sine

    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self ,t ,x):    
        return 1 - self.survival(t, x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t, x.squeeze()))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(self.risk_function(x))
        
    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(self.risk_function(x))
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(self.risk_function(x)))**(1/self.gamma))*self.alpha


if __name__ == "__main__":
    device = torch.device("cpu")
    dgp1 =Weibull_linear(2, 14, 3, device)
    dgp1.coeff = torch.tensor([0.3990, 0.5167])
    x = torch.rand((1000,2))
    t = dgp1.survival(x, torch.rand((1000,)))
    print(torch.min(t), torch.max(t))