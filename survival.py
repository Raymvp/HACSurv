import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
from scipy.integrate import dblquad
from nde import NDE


class PhiInv(nn.Module):
    def __init__(self, phi):
        super(PhiInv, self).__init__()
        self.phi = phi

    def forward(self, y, max_iter=400, tol=1e-10):
        with torch.no_grad():
            """
            # We will only run newton's method on entries which do not have
            # a manual inverse defined (via .inverse)
            inverse = self.phi.inverse(y)
            assert inverse.shape == y.shape
            no_inverse_indices = torch.isnan(inverse)
            # print(no_inverse_indices)
            # print(y[no_inverse_indices].shape)
            t_ = newton_root(
                self.phi, y[no_inverse_indices], max_iter=max_iter, tol=tol,
                t0=torch.ones_like(y[no_inverse_indices])*1e-10)

            inverse[no_inverse_indices] = t_
            t = inverse
            """
            t = newton_root(self.phi, y, max_iter=max_iter, tol=tol)

        topt = t.clone().detach().requires_grad_(True)
        f_topt = self.phi(topt)
        return self.FastInverse.apply(y, topt, f_topt, self.phi)

    class FastInverse(Function):
        '''
        Fast inverse function. To avoid running the optimization
        procedure (e.g., Newton's) repeatedly, we pass in the value
        of the inverse (obtained from the forward pass) manually.

        In the backward pass, we provide gradients w.r.t (i) `y`, and
        (ii) `w`, which are any parameters contained in PhiInv.phi. The
        latter is implicitly given by furnishing derivatives w.r.t. f_topt,
        i.e., the function evaluated (with the current `w`) on topt. Note
        that this should contain *values* approximately equal to y, but
        will have the necessary computational graph built up, but detached
        from y.
        '''
        @staticmethod
        def forward(ctx, y, topt, f_topt, phi):
            ctx.save_for_backward(y, topt, f_topt)
            ctx.phi = phi
            return topt

        @staticmethod
        def backward(ctx, grad):
            y, topt, f_topt = ctx.saved_tensors
            phi = ctx.phi

            with torch.enable_grad():
                # Call FakeInverse once again, in order to allow for higher
                # order derivatives to be taken.
                z = PhiInv.FastInverse.apply(y, topt, f_topt, phi)

                # Find phi'(z), i.e., take derivatives of phi(z) w.r.t z.
                f = phi(z)
                dev_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]

                # To understand why this works, refer to the derivations for
                # inverses. Note that when taking derivatives w.r.t. `w`, we
                # make use of autodiffs automatic application of the chain rule.
                # This automatically finds the derivative d/dw[phi(z)] which
                # when multiplied by the 3rd returned value gives the derivative
                # w.r.t. `w` contained by phi.
                return grad/dev_z, None, -grad/dev_z, None


def log_survival(t, shape, scale, risk):
    return -(torch.exp(risk + shape*torch.log(t) - shape*torch.log(scale))) # used log transform to avoid numerical issue


def survival(t, shape, scale, risk):
    return torch.exp(log_survival(t, shape, scale, risk))


def log_density(t,shape,scale,risk):
    log_hazard = risk + shape*torch.log(t) - shape*torch.log(scale )\
         + torch.log(1/t) + torch.log(shape)
    return log_hazard + log_survival(t, shape, scale, risk)
#DCSurvival 版本
# newtwon_root is used during phi_inverse
# def newton_root(phi, y, t0=None, max_iter=2000, tol=1e-10, guarded=False):
#     '''
#     Solve
#         f(t) = y
#     using the Newton's root finding method.

#     Parameters
#     ----------
#     f: Function which takes in a Tensor t of shape `s` and outputs
#     the pointwise evaluation f(t).
#     y: Tensor of shape `s`.
#     t0: Tensor of shape `s` indicating the initial guess for the root.
#     max_iter: Positive integer containing the max. number of iterations.
#     tol: Termination criterion for the absolute difference |f(t) - y|.
#         By default, this is set to 1e-14,
#         beyond which instability could occur when using pytorch `DoubleTensor`.
#     guarded: Whether we use guarded Newton's root finding method. 
#         By default False: too slow and is not necessary most of the time.

#     Returns:
#         Tensor `t*` of size `s` such that f(t*) ~= y
#     '''
#     if t0 is None:
#         t = torch.zeros_like(y)
#     else:
#         t = t0.clone().detach()

#     s = y.size()
#     for it in range(max_iter):

#         with torch.enable_grad():
#             f_t = phi(t.requires_grad_(True))
#             fp_t = torch.autograd.grad(f_t.sum(), t)[0]
#             assert not torch.any(torch.isnan(fp_t))

#         assert f_t.size() == s
#         assert fp_t.size() == s

#         g_t = f_t - y

#         # Terminate algorithm when all errors are sufficiently small.
#         if (torch.abs(g_t) < tol).all():
#             break

#         if not guarded:
#             t = t - g_t / fp_t
#         else:
#             step_size = torch.ones_like(t)
#             for num_guarded_steps in range(2000):
#                 t_candidate = t - step_size * g_t / fp_t
#                 f_t_candidate = phi(t_candidate.requires_grad_(True))
#                 g_candidate = f_t_candidate - y
#                 overstepped_indices = torch.abs(g_candidate) > torch.abs(g_t)
#                 if not overstepped_indices.any():
#                     t = t_candidate
#                     print(num_guarded_steps)
#                     break
#                 else:
#                     step_size[overstepped_indices] /= 2.

#     assert torch.abs(g_t).max() < tol, \
#         "t=%s, f(t)-y=%s, y=%s, iter=%s, max dev:%s" % (t, g_t, y, it, g_t.max())
#     assert t.size() == s
#     return t
###Gen-AC版本
def newton_root(phi, y, t0=None, max_iter=200, tol=1e-10):
    '''
    Solve
        f(t) = y
    using the Newton's root finding method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    t0: Tensor of shape `s` indicating the initial guess for the root.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the absolute difference |f(t) - y|.
        By default, this is set to 1e-14,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if t0 is None:
        t = torch.zeros_like(y) # why not 0.5?
    else:
        t = t0.detach().clone()

    s = y.size()
    for it in range(max_iter):


            
        if hasattr(phi,'ndiff'):
            f_t = phi(t)
            fp_t = phi.ndiff(t,ndiff=1)
        else:
            with torch.enable_grad():
                f_t = phi(t.requires_grad_(True))
                fp_t = torch.autograd.grad(f_t.sum(), t)[0]
        
        assert not torch.any(torch.isnan(fp_t))

        assert f_t.size() == s
        assert fp_t.size() == s

        g_t = f_t - y

        # Terminate algorithm when all errors are sufficiently small.
        if (torch.abs(g_t) < tol).all():
            break

        t = t - g_t / fp_t

    # error if termination criterion (tol) not met. 
    assert torch.abs(g_t).max() < tol, "t=%s, f(t)-y=%s, y=%s, iter=%s, max dev:%s" % (t, g_t, y, it, g_t.max())
    assert t.size() == s
    
    return t


# Only sampling use bisection root 
def bisection_root(phi, y, lb=None, ub=None, increasing=True, max_iter=100, tol=1e-10):
    '''
    Solve
        f(t) = y
    using the bisection method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    lb, ub: lower and upper bounds for t.
    increasing: True if f is increasing, False if decreasing.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the difference in upper and lower bounds.
        By default, this is set to 1e-10,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if lb is None:
        lb = torch.zeros_like(y)
    if ub is None:
        ub = torch.ones_like(y)

    assert lb.size() == y.size()
    assert ub.size() == y.size()
    assert torch.all(lb < ub)

    f_ub = phi(ub)
    f_lb = phi(lb)
    assert torch.all(
        f_ub >= f_lb) or not increasing, 'Need f to be monotonically non-decreasing.'
    assert torch.all(
        f_lb >= f_ub) or increasing, 'Need f to be monotonically non-increasing.'

    assert (torch.all(
        f_ub >= y) and torch.all(f_lb <= y)) or not increasing, 'y must lie within lower and upper bound. max min y=%s, %s. ub, lb=%s %s' % (y.max(), y.min(), ub, lb)
    assert (torch.all(
        f_ub <= y) and torch.all(f_lb >= y)) or increasing, 'y must lie within lower and upper bound. y=%s, %s. ub, lb=%s %s' % (y.max(), y.min(), ub, lb)

    for it in range(max_iter):
        t = (lb + ub)/2
        f_t = phi(t)

        if increasing:
            too_low, too_high = f_t < y, f_t >= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]
        else:
            too_low, too_high = f_t > y, f_t <= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]

        assert torch.all(ub - lb > 0. - tol), "lb: %s, ub: %s, tol: %s" % (lb, ub, tol)

    assert torch.all(ub - lb <= tol)
    return t


def bisection_default_increasing(phi, y,tol):
    '''
    Wrapper for performing bisection method when f is increasing.
    '''
    return bisection_root(phi, y, increasing=True,tol= tol)


def bisection_default_decreasing(phi, y):
    '''
    Wrapper for performing bisection method when f is decreasing.
    '''
    return bisection_root(phi, y, increasing=False)


class MixExpPhi(nn.Module):
    '''
    Sample net for phi involving the sum of 2 negative exponentials.
    phi(t) = m1 * exp(-w1 * t) + m2 * exp(-w2 * t)

    Network Parameters
    ==================
    mix: Tensor of size 2 such that such that (m1, m2) = softmax(mix)
    slope: Tensor of size 2 such that exp(m1) = w1, exp(m2) = w2

    Note that this implies
    i) m1, m2 > 0 and m1 + m2 = 1.0
    ii) w1, w2 > 0
    '''

    def __init__(self, init_w=None):
        import numpy as np
        super(MixExpPhi, self).__init__()

        if init_w is None:
            self.mix = nn.Parameter(torch.tensor(
                [np.log(0.2), np.log(0.8)], requires_grad=True))
            self.slope = nn.Parameter(
                torch.log(torch.tensor([1e1, 1e6], requires_grad=True)))
        else:
            assert len(init_w) == 2
            assert init_w[0].numel() == init_w[1].numel()
            self.mix = nn.Parameter(init_w[0])
            self.slope = nn.Parameter(init_w[1])

    def forward(self, t):
        s = t.size()
        t_ = t.flatten()
        nquery, nmix = t.numel(), self.mix.numel()

        mix_ = torch.nn.functional.softmax(self.mix)
        exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                         torch.exp(self.slope)[None, :].expand(nquery, nmix))

        ret = torch.sum(mix_ * exps, dim=1)
        return ret.reshape(s)

class InnerGenerator(nn.Module):

    def __init__(self, OuterGenerator,device, Nz=100):
        super(InnerGenerator, self).__init__()
        self.device = device 
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1)
        ).to(device)
        self.mu = nn.Parameter(torch.rand(1))
        self.beta = nn.Parameter(torch.rand(1))
        self.M = self.sample_M(Nz)
        
        for param in OuterGenerator.parameters():
            param.requires_grad = False
        self.psi = OuterGenerator

    def resample_M(self, Nz):
        self.M = self.sample_M(Nz)
        
    # def sample_M(self, N):
    #     return torch.exp(self.model(torch.rand(N).view(-1,1)).view(-1))

    def sample_M(self, N):
        # 创建随机张量，转换形状为 [N, 1]，符合线性层输入要求
        random_tensor = torch.rand(N, 1, device=self.device)
        # 通过模型处理，输出的张量需要再次确保为一维
        output_tensor = self.model(random_tensor)
        return torch.exp(output_tensor.view(-1))
    # def sample_M(self, N):
    #     return torch.exp(self.model(torch.rand(N,1, device=self.device).view(-1,1)).view(-1))
        
    def forward(self, t):
        with torch.autograd.set_detect_anomaly(True):
            s = t.size()
            t_ = t.flatten()
                
            nquery, nmix = t.numel(), self.M.numel()

            exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                            self.M[None, :].expand(nquery, nmix))

            ret = torch.exp(self.mu)*t_+torch.exp(self.beta)*(1-torch.mean(exps, dim=1))
            return self.psi(ret.reshape(s))
class InnerGenerator2(nn.Module):

    def __init__(self, OuterGenerator,device, Nz=100):
        super(InnerGenerator2, self).__init__()
        self.device = device 
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1)
        ).to(device)
        self.mu = nn.Parameter(torch.rand(1))
        self.beta = nn.Parameter(torch.rand(1))
        self.M = self.sample_M(Nz)
        
        for param in OuterGenerator.parameters():
            param.requires_grad = False
        self.psi = OuterGenerator

    def resample_M(self, Nz):
        self.M = self.sample_M(Nz)
        
    # def sample_M(self, N):
    #     return torch.exp(self.model(torch.rand(N).view(-1,1)).view(-1))

    def sample_M(self, N):
        # 创建随机张量，转换形状为 [N, 1]，符合线性层输入要求
        random_tensor = torch.rand(N, 1, device=self.device)
        # 通过模型处理，输出的张量需要再次确保为一维
        output_tensor = self.model(random_tensor)
        return torch.exp(output_tensor.view(-1))
    # def sample_M(self, N):
    #     return torch.exp(self.model(torch.rand(N,1, device=self.device).view(-1,1)).view(-1))
        
    def forward(self, t):
        with torch.autograd.set_detect_anomaly(True):
            s = t.size()
            t_ = t.flatten()
                
            nquery, nmix = t.numel(), self.M.numel()

            exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                            self.M[None, :].expand(nquery, nmix))

            ret = torch.exp(self.mu)*t_+torch.exp(self.beta)*(1-torch.mean(exps, dim=1))
            return self.psi(ret.reshape(s))
class MixExpPhiStochastic(nn.Module):
    '''
    Sample net for phi involving the mean of Nz negative exponentials.
    phi(t) = mean(exp(-wi * t))

    Network Parameters
    ==================
    slope: Tensor of size Nz such that exp(m1) = w1, ..., exp(mN) = wN

    Note that this implies
    w1, ..., wN > 0
    '''

    def __init__(self, device, Nz=100):
            super(MixExpPhiStochastic, self).__init__()
            self.device = device  # 存储设备信息
            self.model = nn.Sequential(
                nn.Linear(1, 10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(10, 10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(10, 1)
            ).to(device)  # 将模型移到指定的设备
            self.M = self.sample_M(Nz)  # 现在 self.sample_M 会使用 self.device

    def sample_M(self, N):
        # 创建随机张量，转换形状为 [N, 1]，符合线性层输入要求
        random_tensor = torch.rand(N, 1, device=self.device)
        # 通过模型处理，输出的张量需要再次确保为一维
        output_tensor = self.model(random_tensor)
        return torch.exp(output_tensor.view(-1))
    # def sample_M(self, N):
    #         return torch.exp(self.model(torch.rand(N,1, device=self.device).view(-1,1)).view(-1))

    def resample_M(self, Nz):
        self.M = self.sample_M(Nz)
        
    # def sample_M(self, N):
    #     return torch.exp(self.model(torch.rand(N).view(-1,1)).view(-1))
        
    def forward(self, t):
        with torch.autograd.set_detect_anomaly(True):
            s = t.size()
            t_ = t.flatten()
                
            nquery, nmix = t.numel(), self.M.numel()

            exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                            self.M[None, :].expand(nquery, nmix))

            ret = torch.mean(exps, dim=1)
            return ret.reshape(s)
    
    def ndiff(self, t, ndiff=0):
        s = t.size()
        t_ = t.flatten()
               
        nquery, nmix = t.numel(), self.M.numel()

        exps = torch.pow(-self.M[None, :].expand(nquery, nmix),ndiff) * \
                         torch.exp(-t_[:, None].expand(nquery, nmix) *
                         self.M[None, :].expand(nquery, nmix))

        ret = torch.mean(exps, dim=1)
        return ret.reshape(s)
class Copula(nn.Module):
    def __init__(self, phi, device):
        super(Copula, self).__init__()
        self.phi = phi.to(device)
        self.phi_inv = PhiInv(phi).to(device)
        self.device = device  # 存储设备信息

    def forward(self, y, mode='cdf', others=None, tol=1e-10):
        y = y.to(self.device)  # 确保输入在正确的设备上
        if not y.requires_grad:
            y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=tol)
        cdf = self.phi(inverses.sum(dim=1))

        if mode == 'cdf':
            return cdf

        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim].to(self.device)
            return cur

        if mode == 'pdf2':
            numerator = self.phi.ndiff(inverses.sum(dim=1), ndiff=ndims)
            denominator = torch.prod(self.phi.ndiff(inverses, ndiff=1), dim=1)
            return numerator / denominator
        
        elif mode == 'cond_cdf':
            target_dims = others['cond_dims']
            cur = cdf
            for dim in target_dims:
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim].to(self.device)
            numerator = cur

            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim].to(self.device)
            denominator = cur
            return numerator / denominator

    
class MixExpPhi2FixedSlope(nn.Module):
    def __init__(self, init_w=None):
        super(MixExpPhi2FixedSlope, self).__init__()

        self.mix = nn.Parameter(torch.tensor(
            [np.log(0.25)], requires_grad=True))
        self.slope = torch.tensor([1e1, 1e6], requires_grad=True)

    def forward(self, t):
        z = 1./(1+torch.exp(-self.mix[0]))
        return z * torch.exp(-t * self.slope[0]) + (1-z) * torch.exp(-t * self.slope[1])


class SurvivalCopula(nn.Module):
    # for known parametric survival marginals, e.g., Weibull distributions
    def __init__(self, phi, device, num_features, tol,  hidden_size=32, max_iter = 2000):
        super(SurvivalCopula, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)
        self.net_t = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.net_c = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.shape_t = nn.Parameter(torch.tensor(1.0)) # Event Weibull Shape
        self.scale_t = nn.Parameter(torch.tensor(1.0)) # Event Weibull Scale
        self.shape_c = nn.Parameter(torch.tensor(1.0)) # Censoring Weibull Shape   
        self.scale_c = nn.Parameter(torch.tensor(1.0)) # Censoring Weibull Scale


    def forward(self, x, t, c, max_iter = 2000):
        # the Covariates for Event and Censoring Model
        x_beta_t = self.net_t(x).squeeze()
        x_beta_c = self.net_c(x).squeeze()

        # In event density, censoring entries should be 0
        event_log_density = c * log_density(t, self.shape_t, self.scale_t, x_beta_t) 
        censoring_log_density = (1-c) * log_density(t, self.shape_c, self.scale_c, x_beta_c)

        S_E = survival(t, self.shape_t, self.scale_t, x_beta_t)
        S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        
        
        logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
    
        return torch.sum(logL)
        

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator


class DCSurvival(nn.Module):
    # with neural density estimators
    def __init__(self, phi, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(DCSurvival, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)
        self.sumo_e = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)

    def forward(self, x, t, c, max_iter = 2000):
        S_E, density_E = self.sumo_e(x, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        
        
        # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
        # why not:
        # logL = ((c==1) | (c==0)) * event_log_density + ((c==1) | (c==0)) * torch.log(cur1) + ((c==2) | (c==3)) * censoring_log_density + ((c==2) | (c==3)) * torch.log(cur2)
        # print(c)
        logL = (c==1) *  event_log_density + (c==1) * torch.log(cur1) + (c==0) *censoring_log_density + (c==0) * torch.log(cur2)  
        return torch.sum(logL)
        
class HACSurv_2D(nn.Module):
    # 使用神经密度估计
    def __init__(self, phi, device, num_features, tol, hidden_size=32, hidden_surv=32, max_iter=2000):
        super(HACSurv_2D, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)
        self.sumo_e = NDE(num_features, layers=[hidden_size, hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0.)
        self.sumo_c = NDE(num_features, layers=[hidden_size, hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0.)

    def forward(self, x, t, c, max_iter=2000, selected_indicators=[1,0]):
        S_E, density_E = self.sumo_e(x, t, gradient=True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        S_C, density_C = self.sumo_c(x, t, gradient=True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()

        # 验证生存函数的合法性
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )

        # Copula 部分导数计算
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter=max_iter)
        cdf = self.phi(inverses.sum(dim=1))

        cur1 = torch.autograd.grad(cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(cdf.sum(), y, create_graph=True)[0][:, 1]

        # 根据选择的事件指示器调整 logL 的计算
        indicator_1, indicator_2 = selected_indicators
        logL = (c == indicator_1) * (event_log_density + torch.log(cur1)) + \
               (c == indicator_2) * (censoring_log_density + torch.log(cur2))
        
        return torch.sum(logL)

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator

    def survival(self, t, X):
        with torch.no_grad():
        
            result = self.sumo_e.survival(X, t)
        return result

    def survival_withCopula_joint_CIF_(self, t,x , max_iter = 2000):

        S_E, density_E = self.sumo_e(x, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        


        Result =  density_E.squeeze()*cur1
        return Result 
    def survival_withCopula_condition_CIF_intergral(self, t,x , max_iter = 2000):

        S_E, density_E = self.sumo_e(x, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        

        # print(S_C)
        Result =  density_E.squeeze()*cur1/S_C

        
        return Result 
    def survival_withCopula_condition_CIF_No_intergral(self, t,x , max_iter = 2000):

        S_E, density_E = self.sumo_e(x, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        
        
        # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
        # why not:
        # logL = ((c==1) | (c==0)) * event_log_density + ((c==1) | (c==0)) * torch.log(cur1) + ((c==2) | (c==3)) * censoring_log_density + ((c==2) | (c==3)) * torch.log(cur2)
        # print(c)

 
        Result =1- cdf/S_C
        return Result 


    
class HACSurv_2D_shared(nn.Module):
    # 使用神经密度估计
    def __init__(self, phi, device, num_features, tol, hidden_size=32, hidden_surv=32, max_iter=2000):
        super(HACSurv_2D_shared, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)
        # self.sumo_e = NDE(num_features, layers=[hidden_size, hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0.)
        # self.sumo_c = NDE(num_features, layers=[hidden_size, hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0.)

        self.sumo_e = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_c = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.shared_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x, t, c, max_iter=2000, selected_indicators=[0, 1]):
        x_shared = self.shared_embedding(x)
        S_E, density_E = self.sumo_e(x_shared, t, gradient=True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()
        S_C, density_C = self.sumo_c(x_shared, t, gradient=True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()

        # 验证生存函数的合法性
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )

        # Copula 部分导数计算
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter=max_iter)
        cdf = self.phi(inverses.sum(dim=1))

        cur1 = torch.autograd.grad(cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(cdf.sum(), y, create_graph=True)[0][:, 1]

        # 根据选择的事件指示器调整 logL 的计算
        indicator_1, indicator_2 = selected_indicators
        logL = (c == indicator_1) * (event_log_density + torch.log(cur1)) + \
               (c == indicator_2) * (censoring_log_density + torch.log(cur2))
        
        return torch.sum(logL)

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator

    def survival(self, t, X):
        with torch.no_grad():
            x_shared = self.shared_embedding(X)
            result = self.sumo_e.survival(x_shared, t)
        return result

    def survival_withCopula_joint_CIF_(self, t,x , max_iter = 2000):
        x_shared = self.shared_embedding(x)
        S_E, density_E = self.sumo_e(x_shared, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        


        Result =  density_E.squeeze()*cur1
        return Result 
    def survival_withCopula_condition_CIF_intergral(self, t,x , max_iter = 2000):
        x_shared = self.shared_embedding(x)
        S_E, density_E = self.sumo_e(x_shared, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        

        # print(S_C)
        Result =  density_E.squeeze()*cur1/S_C

        
        return Result 
    def survival_withCopula_condition_CIF_No_intergral(self, t,x , max_iter = 2000):
        x_shared = self.shared_embedding(x)
        S_E, density_E = self.sumo_e(x_shared, t, gradient = True)
        S_E = S_E.squeeze()
        event_log_density = torch.log(density_E).squeeze()

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E >= 0.).all() and (
            S_E <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E, )
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]        
        
        # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
        # why not:
        # logL = ((c==1) | (c==0)) * event_log_density + ((c==1) | (c==0)) * torch.log(cur1) + ((c==2) | (c==3)) * censoring_log_density + ((c==2) | (c==3)) * torch.log(cur2)
        # print(c)

 
        Result =1- cdf/S_C
        return Result 


# class DCSurvival_competing(nn.Module):
#     # with neural density estimators
#     def __init__(self, phi, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
#         super(DCSurvival_competing, self).__init__()
#         self.tol = tol
#         self.phi = phi
#         self.phi_inv = PhiInv(phi).to(device)
#         self.sumo_e1 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
#         self.sumo_e2 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
#         self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)

#     def forward(self, x, t, c, max_iter = 2000):
#         # print(c)
                    
#         S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
#         S_E1 = S_E1.squeeze()
#         event1_log_density = torch.log(density_E1).squeeze()
#         S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
#         S_E2 = S_E2.squeeze()
#         event2_log_density = torch.log(density_E2).squeeze()        

#         # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
#         S_C, density_C = self.sumo_c(x, t, gradient = True)
#         S_C = S_C.squeeze()
#         censoring_log_density = torch.log(density_C).squeeze()
#         # Check if Survival Function of Event and Censoring are in [0,1]
#         assert (S_E1 >= 0.).all() and (
#             S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
#         assert (S_E2 >= 0.).all() and (
#             S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
        
#         assert (S_C >= 0.).all() and (
#             S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
#         # Partial derivative of Copula using ACNet
#         y = torch.stack([S_E1,S_E2, S_C], dim=1)
#         inverses = self.phi_inv(y, max_iter = max_iter)
#         cdf = self.phi(inverses.sum(dim=1))
#         # TODO: Only take gradients with respect to one dimension of y at at time
#         cur1 = torch.autograd.grad(
#             cdf.sum(), y, create_graph=True)[0][:, 0]
        
#         cur2 = torch.autograd.grad(
#             cdf.sum(), y, create_graph=True)[0][:, 1]      
#         cur3 = torch.autograd.grad(
#             cdf.sum(), y, create_graph=True)[0][:, 2]      
#         # print(cur1.shape)
#         # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
#         # logL = event1_log_density + (c==1) * torch.log(cur1) + event2_log_density + (c==2) * torch.log(cur2)+censoring_log_density + (c==0) * torch.log(cur3)  #density L1
#         logL = (c==1) *event1_log_density + (c==1) * torch.log(cur1) + (c==3) *event2_log_density + (c==3) * torch.log(cur2)+(c==5) *censoring_log_density + (c==5) * torch.log(cur3)     #density L2
#         return torch.sum(logL)
        

#     def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
#         if not y.requires_grad:
#             y = y.requires_grad_(True)
#         ndims = y.size()[1]
#         inverses = self.phi_inv(y, tol=self.tol)
#         cdf = self.phi(inverses.sum(dim=1))
        
#         if mode == 'cdf':
#             return cdf
#         if mode == 'pdf':
#             cur = cdf
#             for dim in range(ndims):
#                 # TODO: Only take gradients with respect to one dimension of y at at time
#                 cur = torch.autograd.grad(
#                     cur.sum(), y, create_graph=True)[0][:, dim]
#             return cur        
#         elif mode =='cond_cdf':
#             target_dims = others['cond_dims']
            
#             # Numerator
#             cur = cdf
#             for dim in target_dims:
#                 # TODO: Only take gradients with respect to one dimension of y at a time
#                 cur = torch.autograd.grad(
#                     cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
#             numerator = cur

#             # Denominator
#             trunc_cdf = self.phi(inverses[:, target_dims])
#             cur = trunc_cdf
#             for dim in range(len(target_dims)):
#                 cur = torch.autograd.grad(
#                     cur.sum(), y, create_graph=True)[0][:, dim]

#             denominator = cur
#             return numerator/denominator

#     def survival_event2(self, t, X):
#         with torch.no_grad():
#             result = self.sumo_e2.survival(X, t)
#         return result
#     def survival_event1(self, t, X):
#         with torch.no_grad():
#             result = self.sumo_e1.survival(X, t)

#         return result

# class DCSurvival_competing_shared(nn.Module):
class HACSurv_3D_Sym_shared(nn.Module):
    # with neural density estimators
    def __init__(self, phi, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(HACSurv_3D_Sym_shared, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)

        self.sumo_e1 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_e2 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_c = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.shared_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        # ##以下的是非shared的
        # self.sumo_e1 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        # self.sumo_e2 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        # self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        # #以下，shared embedding
        # self.shared_embedding = nn.Identity()

    def forward(self, x, t, c, max_iter = 2000):
        # print(c)
        x_shared = self.shared_embedding(x)
        S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
        S_E1 = S_E1.squeeze()
        event1_log_density = torch.log(density_E1).squeeze()
        S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
        S_E2 = S_E2.squeeze()
        event2_log_density = torch.log(density_E2).squeeze()        

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E1 >= 0.).all() and (
            S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
        assert (S_E2 >= 0.).all() and (
            S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
        
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E1,S_E2, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]      
        cur3 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 2]      
        # print(cur1.shape)
        # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
        # logL = event1_log_density + (c==1) * torch.log(cur1) + event2_log_density + (c==2) * torch.log(cur2)+censoring_log_density + (c==0) * torch.log(cur3)  #density L1
        logL = (c==1) *event1_log_density + (c==1) * torch.log(cur1) + (c==2) *event2_log_density + (c==2) * torch.log(cur2)+(c==0) *censoring_log_density + (c==0) * torch.log(cur3)     #density L2
        # logL = (c==1) *event1_log_density + (c==1) * torch.log(cur1) + (c==3) *event2_log_density + (c==3) * torch.log(cur2)+(c==5) *censoring_log_density + (c==5) * torch.log(cur3)     #density L2
        return torch.sum(logL)
        ###############旧的预测代码，参考Survival For compared 
    def compute_intermediate_quantities(self, t, x, max_iter=2000):
        with torch.autograd.set_detect_anomaly(True):
            x_shared = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
          
            # Partial derivative of Copula using ACNet
            y = torch.stack([S_E1,S_E2, S_C], dim=1)
            inverses = self.phi_inv(y, max_iter = max_iter)
            cdf = self.phi(inverses.sum(dim=1))
            # TODO: Only take gradients with respect to one dimension of y at at time
            cur1 = torch.autograd.grad(
                cdf.sum(), y, create_graph=True)[0][:, 0]
            
            cur2 = torch.autograd.grad(
                cdf.sum(), y, create_graph=True)[0][:, 1]      
            cur3 = torch.autograd.grad(
                cdf.sum(), y, create_graph=True)[0][:, 2]    
            


            y_E2C = torch.stack([S_E2, S_C], dim=1)
            inverses_E2C = self.phi_inv(y_E2C, max_iter = max_iter)
            cdf_E2C = self.phi(inverses_E2C.sum(dim=1))

            y_E1C = torch.stack([S_E1, S_C], dim=1)
            inverses_E1C = self.phi_inv(y_E1C, max_iter = max_iter)
            cdf_E1C = self.phi(inverses_E1C.sum(dim=1))  

        return {
            'S_E1': S_E1,
            'S_E2': S_E2,
            'S_C': S_C,
            'density_E1': density_E1,
            'density_E2': density_E2,
            'density_C': density_C,
            'cur1':cur1,
            'cur2':cur2,
            'cdf_E2C': cdf_E2C,
            'cdf_E1C': cdf_E1C,
            'cdf':cdf
        }

    # 修改模型方法以使用预计算的中间变量
    def survival_withCopula_condition_CIF_intergral(self, event_index, intermediates):
        density_E1 = intermediates['density_E1']
        density_E2 = intermediates['density_E2']
        cdf_E1C = intermediates['cdf_E1C']
        cdf_E2C = intermediates['cdf_E2C']
        cur1 = intermediates['cur1']
        cur2 = intermediates['cur2']

        if event_index == 0:
            return density_E1.squeeze()*cur1/cdf_E2C
        elif event_index == 1:
            return density_E2.squeeze()*cur2/cdf_E1C
        else:
            raise ValueError("Unsupported event index")

    def survival_withCopula_condition_CIF_No_intergral(self, event_index, intermediates):
        cdf = intermediates['cdf']
        cdf_E2C = intermediates['cdf_E2C']
        cdf_E1C = intermediates['cdf_E1C']

        if event_index == 0:
            return 1 - cdf / cdf_E2C
        elif event_index == 1:
            return 1 - cdf / cdf_E1C
        else:
            raise ValueError("Unsupported event index")

    def survival_withCopula_joint_CIF_(self, event_index, intermediates):
        density_E1 = intermediates['density_E1']
        density_E2 = intermediates['density_E2']
        cdf_E1C = intermediates['cdf_E1C']
        cdf_E2C = intermediates['cdf_E2C']
        cur1 = intermediates['cur1']
        cur2 = intermediates['cur2']

        if event_index == 0:
            return density_E1.squeeze()*cur1
        elif event_index == 1:
            return density_E2.squeeze()*cur2
        else:
            raise ValueError("Unsupported event index")

    def survival_event_onlySurvivalFunc(self, event_index, intermediates):
        if event_index == 0:
            return intermediates['S_E1']
        elif event_index == 1:
            return intermediates['S_E2']
        else:
            raise ValueError("Unsupported event index")

    # 定义处理函数
    def process_surv_df_CIF(surv_df):
        return 1 - surv_df.clip(0, 1)

    def process_surv_df_jointCIF(surv_df, step=0.1):
        return 1 - (surv_df * step).cumsum()

    def process_surv_df_SF(surv_df):
        return surv_df.clip(0, 1)

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator


# class DCSurvival_competing_3shared(nn.Module):
class HACSurv_4D_Sym_shared(nn.Module):
    # with neural density estimators
    def __init__(self, phi, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(HACSurv_4D_Sym_shared, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)
        # self.sumo_e1 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        # self.sumo_e2 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        # self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        #以下，shared embedding
        self.sumo_e1 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_e2 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_e3 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_c = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.shared_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
    def forward(self, x, t, c, max_iter = 2000):
        # print(c)
        x_shared = self.shared_embedding(x)
        S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
        S_E1 = S_E1.squeeze()
        event1_log_density = torch.log(density_E1).squeeze()
        S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
        S_E2 = S_E2.squeeze()
        event2_log_density = torch.log(density_E2).squeeze()        

        S_E3, density_E3 = self.sumo_e3(x_shared, t, gradient = True)
        S_E3 = S_E3.squeeze()
        event3_log_density = torch.log(density_E3).squeeze()       

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E1 >= 0.).all() and (
            S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
        assert (S_E2 >= 0.).all() and (
            S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )

        assert (S_E3 >= 0.).all() and (
            S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )     
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E1,S_E2,S_E3, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]      
        cur3 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 2]      
        cur4 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 3] 
        # print(cur1.shape)
        # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
        # logL = event1_log_density + (c==1) * torch.log(cur1) + event2_log_density + (c==2) * torch.log(cur2)+censoring_log_density + (c==0) * torch.log(cur3)  #density L1
        logL = (c==1) *event1_log_density + (c==1) * torch.log(cur1) + (c==2) *event2_log_density + (c==2) * torch.log(cur2)+(c==3) *event3_log_density + (c==3) * torch.log(cur3)+(c==0) *censoring_log_density + (c==0) * torch.log(cur4)     #density L2
        # logL = (c==1) *event1_log_density + (c==1) * torch.log(cur1) + (c==3) *event2_log_density + (c==3) * torch.log(cur2)+(c==5) *censoring_log_density + (c==5) * torch.log(cur3)     #density L2
        return torch.sum(logL)
        
    def survival_withCopula_condition_CIF_intergral(self, t,x, event_index,max_iter = 2000):

        x_shared = self.shared_embedding(x)
        S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
        S_E1 = S_E1.squeeze()
        event1_log_density = torch.log(density_E1).squeeze()
        S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
        S_E2 = S_E2.squeeze()
        event2_log_density = torch.log(density_E2).squeeze()        

        S_E3, density_E3 = self.sumo_e3(x_shared, t, gradient = True)
        S_E3 = S_E3.squeeze()
        event3_log_density = torch.log(density_E3).squeeze()       

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E1 >= 0.).all() and (
            S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
        assert (S_E2 >= 0.).all() and (
            S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )

        assert (S_E3 >= 0.).all() and (
            S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )     
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E1,S_E2,S_E3, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]      
        cur3 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 2]      
        cur4 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 3] 

        y_out_023 = torch.stack([S_E2,S_E3, S_C], dim=1)
        inverses_023 = self.phi_inv(y_out_023, max_iter = max_iter)
        cdf_023 = self.phi(inverses_023.sum(dim=1))

        y_out_013 = torch.stack([S_E1,S_E3, S_C], dim=1)
        inverses_013 = self.phi_inv(y_out_013, max_iter = max_iter)
        cdf_013 = self.phi(inverses_013.sum(dim=1))

        y_out_012 = torch.stack([S_E1,S_E2, S_C], dim=1)
        inverses_012 = self.phi_inv(y_out_012, max_iter = max_iter)
        cdf_012 = self.phi(inverses_012.sum(dim=1))

        if event_index == 0:
            return density_E1.squeeze()*cur1/cdf_023
        elif event_index == 1:
            return density_E2.squeeze()*cur2/cdf_013
        elif event_index == 2:
            return density_E3.squeeze()*cur3/cdf_012
        else:
            raise ValueError("Unsupported event index")  

 

    def survival_withCopula_condition_CIF_No_intergral(self, t,x, event_index,max_iter = 2000):
        # with torch.autograd.set_detect_anomaly(True):
            # print(c)
        x_shared = self.shared_embedding(x)
        S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
        S_E1 = S_E1.squeeze()
        event1_log_density = torch.log(density_E1).squeeze()
        S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
        S_E2 = S_E2.squeeze()
        event2_log_density = torch.log(density_E2).squeeze()        

        S_E3, density_E3 = self.sumo_e3(x_shared, t, gradient = True)
        S_E3 = S_E3.squeeze()
        event3_log_density = torch.log(density_E3).squeeze()       

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E1 >= 0.).all() and (
            S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
        assert (S_E2 >= 0.).all() and (
            S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )

        assert (S_E3 >= 0.).all() and (
            S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )     
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E1,S_E2,S_E3, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]      
        cur3 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 2]      
        cur4 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 3] 

        y_out_023 = torch.stack([S_E2,S_E3, S_C], dim=1)
        inverses_023 = self.phi_inv(y_out_023, max_iter = max_iter)
        cdf_023 = self.phi(inverses_023.sum(dim=1))

        y_out_013 = torch.stack([S_E1,S_E3, S_C], dim=1)
        inverses_013 = self.phi_inv(y_out_013, max_iter = max_iter)
        cdf_013 = self.phi(inverses_013.sum(dim=1))

        y_out_012 = torch.stack([S_E1,S_E2, S_C], dim=1)
        inverses_012 = self.phi_inv(y_out_012, max_iter = max_iter)
        cdf_012 = self.phi(inverses_012.sum(dim=1))

        if event_index == 0:
            return 1-cdf/cdf_023

        elif event_index == 1:
            return 1-cdf/cdf_013
        elif event_index == 2:
            return 1-cdf/cdf_012
        else:
            raise ValueError("Unsupported event index")  


    def survival_withCopula_joint_CIF_(self, t,x, event_index,max_iter = 2000):
        x_shared = self.shared_embedding(x)
        S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
        S_E1 = S_E1.squeeze()
        event1_log_density = torch.log(density_E1).squeeze()
        S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
        S_E2 = S_E2.squeeze()
        event2_log_density = torch.log(density_E2).squeeze()        

        S_E3, density_E3 = self.sumo_e3(x_shared, t, gradient = True)
        S_E3 = S_E3.squeeze()
        event3_log_density = torch.log(density_E3).squeeze()       

        # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
        S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()
        # Check if Survival Function of Event and Censoring are in [0,1]
        assert (S_E1 >= 0.).all() and (
            S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
        assert (S_E2 >= 0.).all() and (
            S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )

        assert (S_E3 >= 0.).all() and (
            S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )     
        assert (S_C >= 0.).all() and (
            S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
          
        # Partial derivative of Copula using ACNet
        y = torch.stack([S_E1,S_E2,S_E3, S_C], dim=1)
        inverses = self.phi_inv(y, max_iter = max_iter)
        cdf = self.phi(inverses.sum(dim=1))
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur1 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 0]
        
        cur2 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 1]      
        cur3 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 2]      
        cur4 = torch.autograd.grad(
            cdf.sum(), y, create_graph=True)[0][:, 3] 

        y_out_023 = torch.stack([S_E2,S_E3, S_C], dim=1)
        inverses_023 = self.phi_inv(y_out_023, max_iter = max_iter)
        cdf_023 = self.phi(inverses_023.sum(dim=1))

        y_out_013 = torch.stack([S_E1,S_E3, S_C], dim=1)
        inverses_013 = self.phi_inv(y_out_013, max_iter = max_iter)
        cdf_013 = self.phi(inverses_013.sum(dim=1))

        y_out_012 = torch.stack([S_E1,S_E2, S_C], dim=1)
        inverses_012 = self.phi_inv(y_out_012, max_iter = max_iter)
        cdf_012 = self.phi(inverses_012.sum(dim=1))

        if event_index == 0:
            return density_E1.squeeze()*cur1
        elif event_index == 1:
            return density_E2.squeeze()*cur2
        elif event_index == 2:
            return density_E3.squeeze()*cur3
        else:
            raise ValueError("Unsupported event index")  


    def survival_event2(self, t, X):
        with torch.no_grad():
            X = self.shared_embedding(X)
            result = self.sumo_e2.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
        # return cdf
    def survival_event3(self, t, X):
        with torch.no_grad():
            X= self.shared_embedding(X)
            result = self.sumo_e3.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
        # return cdf

    def survival_event1(self, t, X):
        with torch.no_grad():
            X = self.shared_embedding(X)
            result = self.sumo_e1.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=self.tol)
        cdf = self.phi(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator
    # def survival_event_predict_withCop(self, t, X, event_index):
    #     # 根据事件索引选择对应的生存预测函数
    #     if event_index == 0:
    #         return self.survival_event1_withCopula(t, X)
    #     elif event_index == 1:
    #         return self.survival_event2_withCopula(t, X)
    #     elif event_index == 2:
    #         return self.survival_event3_withCopula(t, X)
    #     # elif event_index == 3:
    #     #     return self.survival_event4(t, X)
    #     # elif event_index == 4:
    #     #     return self.survival_event5(t, X)
    #     else:
    #         raise ValueError("Unsupported event index")  
        
        # return cdf
    def survival_event_onlySurvivalFunc(self, t,X, event_index):
        # 根据事件索引选择对应的生存预测函数
        if event_index == 0:
            return self.survival_event1(t, X)
        elif event_index == 1:
            return self.survival_event2(t, X)
        elif event_index == 2:
            return self.survival_event3(t, X)
        # elif event_index == 3:
        #     return self.survival_event4(t, X)
        # elif event_index == 4:
        #     return self.survival_event5(t, X)
        else:
            raise ValueError("Unsupported event index")  
        ########
# class DCSurvival_competing_shared5(nn.Module):
class HACSurv_6D_Sym_shared(nn.Module):
    # with neural density estimators
    def __init__(self, phi, device, num_features, tol, hidden_size=32, hidden_surv=32, max_iter=2000):
        super(HACSurv_6D_Sym_shared, self).__init__()
        self.tol = tol
        self.phi = phi
        self.phi_inv = PhiInv(phi).to(device)

        # 定义5个事件的NDE模型
        self.sumo_e1 = NDE(hidden_size, layers=[hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0)
        self.sumo_e2 = NDE(hidden_size, layers=[hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0)
        self.sumo_e3 = NDE(hidden_size, layers=[hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0)
        self.sumo_e4 = NDE(hidden_size, layers=[hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0)
        self.sumo_e5 = NDE(hidden_size, layers=[hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0)
        self.sumo_c = NDE(hidden_size, layers=[hidden_size, hidden_size], layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0)

        # 共享的嵌入层
        self.shared_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x, t, c, max_iter=2000):
        x_shared = self.shared_embedding(x)

        # 计算每个事件和删失的生存函数和密度函数
        S_Es = []
        densities_Es = []
        event_log_densities = []
        for i in range(1, 6):  # 事件1到5
            sumo_ei = getattr(self, f'sumo_e{i}')
            S_Ei, density_Ei = sumo_ei(x_shared, t, gradient=True)
            S_Ei = S_Ei.squeeze()
            S_Es.append(S_Ei)
            densities_Es.append(density_Ei)
            event_log_densities.append(torch.log(density_Ei).squeeze())

        S_C, density_C = self.sumo_c(x_shared, t, gradient=True)
        S_C = S_C.squeeze()
        censoring_log_density = torch.log(density_C).squeeze()

        # 检查生存函数是否在 [0,1] 之间
        for S_E in S_Es + [S_C]:
            assert (S_E >= 0.).all() and (S_E <= 1.+1e-10).all(), f"t {t}, output {S_E}"

        # 将生存函数堆叠为 y
        y = torch.stack([S_C] + S_Es, dim=1)
        inverses = self.phi_inv(y, max_iter=max_iter)
        cdf = self.phi(inverses.sum(dim=1))

        # 计算偏导数
        partials = torch.autograd.grad(cdf.sum(), y, create_graph=True)[0]
        cur_c = partials[:, 0]
        curs = [partials[:, i] for i in range(1, 6)]  # 对应于5个事件

        # 计算对数似然
        logL = (c == 0) * (censoring_log_density + torch.log(cur_c))
        for i in range(1, 6):
            event_indicator = (c == i)
            logL += event_indicator * (event_log_densities[i - 1] + torch.log(curs[i - 1]))

        return torch.sum(logL)

    def compute_intermediate_quantities(self, t, x, max_iter=2000):
        with torch.autograd.set_detect_anomaly(True):
            x_shared = self.shared_embedding(x)

            # 计算每个事件和删失的生存函数和密度函数
            S_Es = []
            densities_Es = []
            event_log_densities = []
            for i in range(1, 6):  # 事件1到5
                sumo_ei = getattr(self, f'sumo_e{i}')
                S_Ei, density_Ei = sumo_ei(x_shared, t, gradient=True)
                S_Ei = S_Ei.squeeze()
                S_Es.append(S_Ei)
                densities_Es.append(density_Ei)
                event_log_densities.append(torch.log(density_Ei).squeeze())

            S_C, density_C = self.sumo_c(x_shared, t, gradient=True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()

            # 将生存函数堆叠为 y
            y = torch.stack([S_C] + S_Es, dim=1)
            inverses = self.phi_inv(y, max_iter=max_iter)
            cdf = self.phi(inverses.sum(dim=1))

            # 计算偏导数
            partials = torch.autograd.grad(cdf.sum(), y, create_graph=True)[0]
            cur_c = partials[:, 0]
            curs = [partials[:, i] for i in range(1, 6)]  # 对应于5个事件

            # 计算排除预测事件后的 cdf
            cdfs_excl_event = []
            for i in range(1, 6):  # 事件索引从1到5
                # 排除当前事件
                y_excl_event = torch.stack(
                    [S_C] + [S_Es[j - 1] for j in range(1, 6) if j != i],
                    dim=1
                )
                inverses_excl_event = self.phi_inv(y_excl_event, max_iter=max_iter)
                cdf_excl_event = self.phi(inverses_excl_event.sum(dim=1))
                cdfs_excl_event.append(cdf_excl_event)

        return {
            'S_Es': S_Es,
            'S_C': S_C,
            'densities_Es': densities_Es,
            'density_C': density_C,
            'curs': curs,
            'cur_c': cur_c,
            'cdf': cdf,
            'cdfs_excl_event': cdfs_excl_event
        }

    # 修改模型方法以使用预计算的中间变量
    def survival_withCopula_condition_CIF_intergral(self, event_index, intermediates):
        # event_index 从 0 到 4，对应事件1到5
        density_E = intermediates['densities_Es'][event_index]
        cur = intermediates['curs'][event_index]
        cdf_excl_event = intermediates['cdfs_excl_event'][event_index]
        return density_E.squeeze() * cur / cdf_excl_event

    def survival_withCopula_condition_CIF_No_intergral(self, event_index, intermediates):
        cdf = intermediates['cdf']
        cdf_excl_event = intermediates['cdfs_excl_event'][event_index]
        return 1 - cdf / cdf_excl_event

    def survival_withCopula_joint_CIF_(self, event_index, intermediates):
        density_E = intermediates['densities_Es'][event_index]
        cur = intermediates['curs'][event_index]
        return density_E.squeeze() * cur

    def survival_event_onlySurvivalFunc(self, event_index, intermediates):
        return intermediates['S_Es'][event_index]

    # 定义处理函数
    def process_surv_df_CIF(surv_df):
        return 1 - surv_df.clip(0, 1)

    def process_surv_df_jointCIF(surv_df, step=0.1):
        return 1 - (surv_df * step).cumsum()

    def process_surv_df_SF(surv_df):
        return surv_df.clip(0, 1)



class HACSurvival_competing(nn.Module):
    # with neural density estimators
    def __init__(self, phi_out,phi_in, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(HACSurvival_competing, self).__init__()
        self.tol = tol
        self.phi_out = phi_out
        self.phi_out_inv = PhiInv(phi_out).to(device)
        self.phi_in=phi_in
        self.phi_in_inv = PhiInv(phi_in).to(device)
        self.sumo_e1 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_e2 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        # self.sumo_e1 = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        # self.sumo_e2 = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        # self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        
        
        
        # self.shared_embedding = nn.Sequential(
        #     nn.Linear(num_features, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU()
        # )
    def forward(self, x, t, c, max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_in = torch.stack([S_E1,S_E2], dim=1)
            inverses_in = self.phi_in_inv(y_in, max_iter = max_iter)
            cdf_in = self.phi_in(inverses_in.sum(dim=1))

            y_out = torch.stack([cdf_in,S_C], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))
            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]      
            grad_cdf_out_SE1 = torch.autograd.grad(outputs=cdf_out, inputs=S_E1, grad_outputs=torch.ones_like(cdf_out), create_graph=True,retain_graph=True)[0]
            grad_cdf_out_SE2 = torch.autograd.grad(outputs=cdf_out, inputs=S_E2, grad_outputs=torch.ones_like(cdf_out), create_graph=True,retain_graph=True)[0]

            # print('y_out',y_out.shape)
            # TODO: Only take gradients with respect to one dimension of y at at time

            # y_in = torch.stack([S_E1,S_E2], dim=1)
            # print('yin',y_in.shape)

            cur_in1 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_in2 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 1]     

            
            # cur3 = torch.autograd.grad(
            #     cdf.sum(), y, create_graph=True)[0][:, 2]      
            # *cur_out1*cur_in1
            # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
            # logL = event1_log_density + (c==1) * torch.log(cur1) + event2_log_density + (c==2) * torch.log(cur2)+censoring_log_density + (c==0) * torch.log(cur3)  #density L1
            # print(cur_out1)

            # print(cur_in1)
            logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_out1*cur_in1)) + (c==2) *event2_log_density + (c==2) * (torch.log(cur_out1*cur_in2))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out2)     #density L2
            # logL = (c==1) *event1_log_density + (c==1) * (torch.log(grad_cdf_out_SE1)) + (c==2) *event2_log_density + (c==2) * (torch.log(grad_cdf_out_SE2))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out2)
            # logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_in1)) + (c==2) *event2_log_density + (c==2) * (torch.log(cur_in2))+(c==0) *censoring_log_density      #density L2
        return torch.sum(logL)
    def survival_event1_withCopula(self,t, x,max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_in = torch.stack([S_E1,S_E2], dim=1)
            inverses_in = self.phi_in_inv(y_in, max_iter = max_iter)
            cdf_in = self.phi_in(inverses_in.sum(dim=1))

            y_out = torch.stack([cdf_in,S_C], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))
            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]      
            # print('y_out',y_out.shape)
            # TODO: Only take gradients with respect to one dimension of y at at time

            # y_in = torch.stack([S_E1,S_E2], dim=1)
            # print('yin',y_in.shape)

            cur_in1 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_in2 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 1]    
            result=density_E1.squeeze()*cur_out1*cur_in1
            return 1-result
    def survival_event2_withCopula(self,t, x,max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_in = torch.stack([S_E1,S_E2], dim=1)
            inverses_in = self.phi_in_inv(y_in, max_iter = max_iter)
            cdf_in = self.phi_in(inverses_in.sum(dim=1))

            y_out = torch.stack([cdf_in,S_C], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))
            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]      
            # print('y_out',y_out.shape)
            # TODO: Only take gradients with respect to one dimension of y at at time

            # y_in = torch.stack([S_E1,S_E2], dim=1)
            # print('yin',y_in.shape)

            cur_in1 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_in2 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 1]    
            result=density_E2.squeeze()*cur_out1*cur_in2
            return 1-result


    def survival_event2(self, t, X):
        with torch.no_grad():
            result = self.sumo_e2.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
        # return cdf


    def survival_event1(self, t, X):
        with torch.no_grad():
            result = self.sumo_e1.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
        # return cdf

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_01_inv(y, tol=self.tol)
        cdf = self.phi_01(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi_01(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator
        # return cdf
    def survival_event_predict_withCop(self, t, X, event_index):
        # 根据事件索引选择对应的生存预测函数
        if event_index == 0:
            return self.survival_event1_withCopula(t, X)
        elif event_index == 1:
            return self.survival_event2_withCopula(t, X)
        # elif event_index == 2:
        #     return self.survival_event3_withCopula(t, X)
        # elif event_index == 3:
        #     return self.survival_event4(t, X)
        # elif event_index == 4:
        #     return self.survival_event5(t, X)
        else:
            raise ValueError("Unsupported event index")  
        
        # return cdf
    def survival_event_onlySurvivalFunc(self, t, X, event_index):
        # 根据事件索引选择对应的生存预测函数
        if event_index == 0:
            return self.survival_event1(t, X)
        elif event_index == 1:
            return self.survival_event2(t, X)
        # elif event_index == 2:
        #     return self.survival_event3(t, X)
        # elif event_index == 3:
        #     return self.survival_event4(t, X)
        # elif event_index == 4:
        #     return self.survival_event5(t, X)
        else:
            raise ValueError("Unsupported event index")  
        
        # return cdf
class HACSurvival_competing_shared(nn.Module):
    # with neural density estimators
    def __init__(self, phi_out,phi_in, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(HACSurvival_competing_shared, self).__init__()
        self.tol = tol
        self.phi_out = phi_out
        self.phi_out_inv = PhiInv(phi_out).to(device)
        self.phi_in=phi_in
        self.phi_in_inv = PhiInv(phi_in).to(device)
        self.sumo_e1 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_e2 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.sumo_c = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.shared_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
    def forward(self, x, t, c, max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            x_shared = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_in = torch.stack([S_E1,S_E2], dim=1)
            inverses_in = self.phi_in_inv(y_in, max_iter = max_iter)
            cdf_in = self.phi_in(inverses_in.sum(dim=1))

            y_out = torch.stack([cdf_in,S_C], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))
            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]      
            grad_cdf_out_SE1 = torch.autograd.grad(outputs=cdf_out, inputs=S_E1, grad_outputs=torch.ones_like(cdf_out), create_graph=True,retain_graph=True)[0]
            grad_cdf_out_SE2 = torch.autograd.grad(outputs=cdf_out, inputs=S_E2, grad_outputs=torch.ones_like(cdf_out), create_graph=True,retain_graph=True)[0]

            # print('y_out',y_out.shape)
            # TODO: Only take gradients with respect to one dimension of y at at time

            # y_in = torch.stack([S_E1,S_E2], dim=1)
            # print('yin',y_in.shape)

            cur_in1 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_in2 = torch.autograd.grad(
                cdf_in.sum(), y_in, create_graph=True,retain_graph=True)[0][:, 1]     

            
            # cur3 = torch.autograd.grad(
            #     cdf.sum(), y, create_graph=True)[0][:, 2]      
            # *cur_out1*cur_in1
            # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
            # logL = event1_log_density + (c==1) * torch.log(cur1) + event2_log_density + (c==2) * torch.log(cur2)+censoring_log_density + (c==0) * torch.log(cur3)  #density L1
            # print(cur_out1)

            # print(cur_in1)
            logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_out1*cur_in1)) + (c==2) *event2_log_density + (c==2) * (torch.log(cur_out1*cur_in2))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out2)     #density L2
            # logL = (c==1) *event1_log_density + (c==1) * (torch.log(grad_cdf_out_SE1)) + (c==2) *event2_log_density + (c==2) * (torch.log(grad_cdf_out_SE2))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out2)
            # logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_in1)) + (c==2) *event2_log_density + (c==2) * (torch.log(cur_in2))+(c==0) *censoring_log_density      #density L2
        return torch.sum(logL)
    # 在模型类中添加一个计算共享中间变量的函数
    def compute_intermediate_quantities(self, t, x, max_iter=2000):
        with torch.autograd.set_detect_anomaly(True):
            x_shared = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient=True)
            S_E1 = S_E1.squeeze()
            S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient=True)
            S_E2 = S_E2.squeeze()
            S_C, density_C = self.sumo_c(x_shared, t, gradient=True)
            S_C = S_C.squeeze()

            y_in = torch.stack([S_E1, S_E2], dim=1)
            inverses_in = self.phi_in_inv(y_in, max_iter=max_iter)
            cdf_in = self.phi_in(inverses_in.sum(dim=1))

            y_out = torch.stack([cdf_in, S_C], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter=max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out = torch.autograd.grad(cdf_out.sum(), y_out, create_graph=True, retain_graph=True)[0]
            cur_in = torch.autograd.grad(cdf_in.sum(), y_in, create_graph=True, retain_graph=True)[0]

            y_E2C = torch.stack([S_C, S_E2], dim=1)
            inverses_out_E2C = self.phi_out_inv(y_E2C, max_iter=max_iter)
            cdf_E2C = self.phi_out(inverses_out_E2C.sum(dim=1))

            y_E1C = torch.stack([S_C, S_E1], dim=1)
            inverses_out_E1C = self.phi_out_inv(y_E1C, max_iter=max_iter)
            cdf_E1C = self.phi_out(inverses_out_E1C.sum(dim=1))

        return {
            'S_E1': S_E1,
            'S_E2': S_E2,
            'S_C': S_C,
            'density_E1': density_E1,
            'density_E2': density_E2,
            'density_C': density_C,
            'cdf_in': cdf_in,
            'cdf_out': cdf_out,
            'cur_out': cur_out,
            'cur_in': cur_in,
            'cdf_E2C': cdf_E2C,
            'cdf_E1C': cdf_E1C
        }

    # 修改模型方法以使用预计算的中间变量
    def survival_withCopula_condition_CIF_intergral(self, event_index, intermediates):
        S_E1 = intermediates['S_E1']
        S_E2 = intermediates['S_E2']
        density_E1 = intermediates['density_E1']
        density_E2 = intermediates['density_E2']
        cur_out1 = intermediates['cur_out'][:, 0]
        cur_in1 = intermediates['cur_in'][:, 0]
        cur_in2 = intermediates['cur_in'][:, 1]
        cdf_E2C = intermediates['cdf_E2C']
        cdf_E1C = intermediates['cdf_E1C']

        if event_index == 0:
            return density_E1.squeeze() * cur_out1 * cur_in1 / cdf_E2C
        elif event_index == 1:
            return density_E2.squeeze() * cur_out1 * cur_in2 / cdf_E1C
        else:
            raise ValueError("Unsupported event index")

    def survival_withCopula_condition_CIF_No_intergral(self, event_index, intermediates):
        cdf_out = intermediates['cdf_out']
        cdf_E2C = intermediates['cdf_E2C']
        cdf_E1C = intermediates['cdf_E1C']

        if event_index == 0:
            return 1 - cdf_out / cdf_E2C
        elif event_index == 1:
            return 1 - cdf_out / cdf_E1C
        else:
            raise ValueError("Unsupported event index")

    def survival_withCopula_joint_CIF_(self, event_index, intermediates):
        density_E1 = intermediates['density_E1']
        density_E2 = intermediates['density_E2']
        cur_out1 = intermediates['cur_out'][:, 0]
        cur_in1 = intermediates['cur_in'][:, 0]
        cur_in2 = intermediates['cur_in'][:, 1]

        if event_index == 0:
            return density_E1.squeeze() * cur_out1 * cur_in1
        elif event_index == 1:
            return density_E2.squeeze() * cur_out1 * cur_in2
        else:
            raise ValueError("Unsupported event index")

    def survival_event_onlySurvivalFunc(self, event_index, intermediates):
        if event_index == 0:
            return intermediates['S_E1']
        elif event_index == 1:
            return intermediates['S_E2']
        else:
            raise ValueError("Unsupported event index")

    # 定义处理函数
    def process_surv_df_CIF(surv_df):
        return 1 - surv_df.clip(0, 1)

    def process_surv_df_jointCIF(surv_df, step=0.1):
        return 1 - (surv_df * step).cumsum()

    def process_surv_df_SF(surv_df):
        return surv_df.clip(0, 1)
    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_01_inv(y, tol=self.tol)
        cdf = self.phi_01(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi_01(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator
     
class HACSurvival_competing5(nn.Module):
    def __init__(self, phi_out,phi_24,phi_135, device, num_features, tol, hidden_size=32, hidden_surv=32, max_iter=2000):
        #   def __init__(self, phi_out,phi_in, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(HACSurvival_competing5, self).__init__()
        self.tol = tol
        # self.phi = phi
        # self.phi_inv = PhiInv(phi).to(device)

        self.phi_out = phi_out
        self.phi_out_inv = PhiInv(phi_out).to(device)
        self.phi_24=phi_24
        self.phi_24_inv = PhiInv(phi_24).to(device)
        self.phi_135=phi_135
        self.phi_135_inv = PhiInv(phi_135).to(device)
        
        # 初始化5个事件的神经密度估计器和1个删失的估计器
        self.sumo_e1 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e2 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e3 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e4 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e5 = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        # self.sumo_e1 = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv], dropout = 0.1)
        # self.sumo_e2 = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv], dropout = 0.1)
        # self.sumo_e3 = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv], dropout = 0.1)
        # self.sumo_e4 = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv], dropout = 0.1)
        # self.sumo_e5 = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv], dropout = 0.1)
        # self.sumo_c = NDE(num_features, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv], dropout = 0.1)

        # self.sumo_c = NDE(num_features, layers=[hidden_size, hidden_size, hidden_size], 
        #                   layers_surv=[hidden_surv, hidden_surv, hidden_surv], dropout=0.0)

    def forward(self, x, t, c, max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        
            S_E3, density_E3 = self.sumo_e3(x, t, gradient = True)
            S_E3 = S_E3.squeeze()
            event3_log_density = torch.log(density_E3).squeeze()   
            S_E4, density_E4 = self.sumo_e4(x, t, gradient = True)
            S_E4 = S_E4.squeeze()
            event4_log_density = torch.log(density_E4).squeeze()   
            S_E5, density_E5 = self.sumo_e5(x, t, gradient = True)
            S_E5 = S_E5.squeeze()
            event5_log_density = torch.log(density_E5).squeeze()   

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            assert (S_E3 >= 0.).all() and (
                S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )  
            assert (S_E4 >= 0.).all() and (
                S_E4 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E4, )    
            assert (S_E5 >= 0.).all() and (
                S_E5 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E5, )      
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            # Partial derivative of Copula using Gen-AC
            y_24 = torch.stack([S_E2,S_E4], dim=1)
            inverses_24 = self.phi_24_inv(y_24, max_iter = max_iter)
            cdf_24 = self.phi_24(inverses_24.sum(dim=1))
            y_135 = torch.stack([S_E1, S_E3, S_E5], dim=1)
            inverses_135 = self.phi_135_inv(y_135, max_iter = max_iter)
            cdf_135 = self.phi_135(inverses_135.sum(dim=1))

            y_out = torch.stack([ S_C, cdf_24, cdf_135], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]  
            cur_out3 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 2] 

            cur_24_2 = torch.autograd.grad(
                cdf_24.sum(), y_24, create_graph=True,retain_graph=True)[0][:, 0]
            cur_24_4 = torch.autograd.grad(
                cdf_24.sum(), y_24, create_graph=True,retain_graph=True)[0][:, 1]
            
            cur_135_1 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 0]
            cur_135_3 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 1]
            cur_135_5 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 2]
            
            logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_out3*cur_135_1)) +(c==3) *event3_log_density + (c==3) * (torch.log(cur_out3*cur_135_3))+(c==5) *event5_log_density + (c==5) * (torch.log(cur_out3*cur_135_5))+ (c==2) *event2_log_density + (c==2) * (torch.log(cur_out2*cur_24_2))+(c==4) *event4_log_density + (c==4) * (torch.log(cur_out2*cur_24_4))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out1)
        return torch.sum(logL)

    def survival_event_predict_withCop(self, x, t, event_index,max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        
            S_E3, density_E3 = self.sumo_e3(x, t, gradient = True)
            S_E3 = S_E3.squeeze()
            event3_log_density = torch.log(density_E3).squeeze()   
            S_E4, density_E4 = self.sumo_e4(x, t, gradient = True)
            S_E4 = S_E4.squeeze()
            event4_log_density = torch.log(density_E4).squeeze()   
            S_E5, density_E5 = self.sumo_e5(x, t, gradient = True)
            S_E5 = S_E5.squeeze()
            event5_log_density = torch.log(density_E5).squeeze()   

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            assert (S_E3 >= 0.).all() and (
                S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )  
            assert (S_E4 >= 0.).all() and (
                S_E4 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E4, )    
            assert (S_E5 >= 0.).all() and (
                S_E5 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E5, )      
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            # Partial derivative of Copula using Gen-AC
            y_24 = torch.stack([S_E2,S_E4], dim=1)
            inverses_24 = self.phi_24_inv(y_24, max_iter = max_iter)
            cdf_24 = self.phi_24(inverses_24.sum(dim=1))
            y_135 = torch.stack([S_E1, S_E3, S_E5], dim=1)
            inverses_135 = self.phi_135_inv(y_135, max_iter = max_iter)
            cdf_135 = self.phi_135(inverses_135.sum(dim=1))

            y_out = torch.stack([ S_C, cdf_24, cdf_135], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]  
            cur_out3 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 2] 

            cur_24_2 = torch.autograd.grad(
                cdf_24.sum(), y_24, create_graph=True,retain_graph=True)[0][:, 0]
            cur_24_4 = torch.autograd.grad(
                cdf_24.sum(), y_24, create_graph=True,retain_graph=True)[0][:, 1]
            
            cur_135_1 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 0]
            cur_135_3 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 1]
            cur_135_5 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 2]
            
        # if event_index == 0:
        #     return density_C.squeeze()*cur_out1
    
        if event_index == 0:
            return density_E1.squeeze() * cur_out3*cur_135_1
        elif event_index == 1:
            return density_E2.squeeze() * cur_out2*cur_24_2
        elif event_index == 2:
            return density_E3.squeeze() * cur_out3*cur_135_3
        elif event_index == 3:
            return density_E4.squeeze() * cur_out2*cur_24_4
        elif event_index == 4:
            return density_E5.squeeze() * cur_out3*cur_135_5
        else:
            raise ValueError("Unsupported event index")  

    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_01_inv(y, tol=self.tol)
        cdf = self.phi_01(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi_01(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator    
    
class HACSurvival_competing5_shared(nn.Module):
    def __init__(self, phi_out,phi_24,phi_135, device, num_features, tol, hidden_size=32, hidden_surv=32, max_iter=2000):
        #   def __init__(self, phi_out,phi_in, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(HACSurvival_competing5_shared, self).__init__()
        self.tol = tol
        # self.phi = phi
        # self.phi_inv = PhiInv(phi).to(device)

        self.phi_out = phi_out
        self.phi_out_inv = PhiInv(phi_out).to(device)
        self.phi_24=phi_24
        self.phi_24_inv = PhiInv(phi_24).to(device)
        self.phi_135=phi_135
        self.phi_135_inv = PhiInv(phi_135).to(device)
        
        # 初始化5个事件的神经密度估计器和1个删失的估计器
        self.sumo_e1 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e2 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e3 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e4 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e5 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)

        self.sumo_c = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0)
        self.shared_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        ) 


    def forward(self, x, t, c, max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            x_shared = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        
            S_E3, density_E3 = self.sumo_e3(x_shared, t, gradient = True)
            S_E3 = S_E3.squeeze()
            event3_log_density = torch.log(density_E3).squeeze()   
            S_E4, density_E4 = self.sumo_e4(x_shared, t, gradient = True)
            S_E4 = S_E4.squeeze()
            event4_log_density = torch.log(density_E4).squeeze()   
            S_E5, density_E5 = self.sumo_e5(x_shared, t, gradient = True)
            S_E5 = S_E5.squeeze()
            event5_log_density = torch.log(density_E5).squeeze()   

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x_shared, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            assert (S_E3 >= 0.).all() and (
                S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )  
            assert (S_E4 >= 0.).all() and (
                S_E4 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E4, )    
            assert (S_E5 >= 0.).all() and (
                S_E5 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E5, )      
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            # Partial derivative of Copula using Gen-AC
            y_24 = torch.stack([S_E2,S_E4], dim=1)
            inverses_24 = self.phi_24_inv(y_24, max_iter = max_iter)
            cdf_24 = self.phi_24(inverses_24.sum(dim=1))
            y_135 = torch.stack([S_E1, S_E3, S_E5], dim=1)
            inverses_135 = self.phi_135_inv(y_135, max_iter = max_iter)
            cdf_135 = self.phi_135(inverses_135.sum(dim=1))

            y_out = torch.stack([ S_C, cdf_24, cdf_135], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]  
            cur_out3 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 2] 

            cur_24_2 = torch.autograd.grad(
                cdf_24.sum(), y_24, create_graph=True,retain_graph=True)[0][:, 0]
            cur_24_4 = torch.autograd.grad(
                cdf_24.sum(), y_24, create_graph=True,retain_graph=True)[0][:, 1]
            
            cur_135_1 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 0]
            cur_135_3 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 1]
            cur_135_5 = torch.autograd.grad(
                cdf_135.sum(), y_135, create_graph=True,retain_graph=True)[0][:, 2]
            
            logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_out3*cur_135_1)) +(c==3) *event3_log_density + (c==3) * (torch.log(cur_out3*cur_135_3))+(c==5) *event5_log_density + (c==5) * (torch.log(cur_out3*cur_135_5))+ (c==2) *event2_log_density + (c==2) * (torch.log(cur_out2*cur_24_2))+(c==4) *event4_log_density + (c==4) * (torch.log(cur_out2*cur_24_4))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out1)
        return torch.sum(logL)
    def compute_intermediate_quantities(self, t, x, max_iter=2000):
        with torch.autograd.set_detect_anomaly(True):
            x_shared = self.shared_embedding(x)
            # 生存函数和密度函数
            S_E1, density_E1 = self.sumo_e1(x_shared, t, gradient=True)
            S_E1 = S_E1.squeeze()
            density_E1 = density_E1.squeeze()
            S_E2, density_E2 = self.sumo_e2(x_shared, t, gradient=True)
            S_E2 = S_E2.squeeze()
            density_E2 = density_E2.squeeze()
            S_E3, density_E3 = self.sumo_e3(x_shared, t, gradient=True)
            S_E3 = S_E3.squeeze()
            density_E3 = density_E3.squeeze()
            S_E4, density_E4 = self.sumo_e4(x_shared, t, gradient=True)
            S_E4 = S_E4.squeeze()
            density_E4 = density_E4.squeeze()
            S_E5, density_E5 = self.sumo_e5(x_shared, t, gradient=True)
            S_E5 = S_E5.squeeze()
            density_E5 = density_E5.squeeze()
            S_C, density_C = self.sumo_c(x_shared, t, gradient=True)
            S_C = S_C.squeeze()
            density_C = density_C.squeeze()

            # 检查生存函数是否在 [0,1] 范围内
            assert (S_E1 >= 0.).all() and (S_E1 <= 1.+1e-10).all()
            assert (S_E2 >= 0.).all() and (S_E2 <= 1.+1e-10).all()
            assert (S_E3 >= 0.).all() and (S_E3 <= 1.+1e-10).all()
            assert (S_E4 >= 0.).all() and (S_E4 <= 1.+1e-10).all()
            assert (S_E5 >= 0.).all() and (S_E5 <= 1.+1e-10).all()
            assert (S_C >= 0.).all() and (S_C <= 1.+1e-10).all()

            # 计算 Copula 函数值
            y_24 = torch.stack([S_E2, S_E4], dim=1)
            inverses_24 = self.phi_24_inv(y_24, max_iter=max_iter)
            cdf_24 = self.phi_24(inverses_24.sum(dim=1))

            y_135 = torch.stack([S_E1, S_E3, S_E5], dim=1)
            inverses_135 = self.phi_135_inv(y_135, max_iter=max_iter)
            cdf_135 = self.phi_135(inverses_135.sum(dim=1))

            y_out = torch.stack([S_C, cdf_24, cdf_135], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter=max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            # 计算偏导数
            grads_cdf_out = torch.autograd.grad(cdf_out.sum(), y_out, create_graph=True, retain_graph=True)[0]
            cur_out1 = grads_cdf_out[:, 0]
            cur_out2 = grads_cdf_out[:, 1]
            cur_out3 = grads_cdf_out[:, 2]

            grads_cdf_24 = torch.autograd.grad(cdf_24.sum(), y_24, create_graph=True, retain_graph=True)[0]
            cur_24_2 = grads_cdf_24[:, 0]
            cur_24_4 = grads_cdf_24[:, 1]

            grads_cdf_135 = torch.autograd.grad(cdf_135.sum(), y_135, create_graph=True, retain_graph=True)[0]
            cur_135_1 = grads_cdf_135[:, 0]
            cur_135_3 = grads_cdf_135[:, 1]
            cur_135_5 = grads_cdf_135[:, 2]

            # 计算积分方法需要的额外 CDF
            y_35 = torch.stack([S_E3, S_E5], dim=1)
            inverses_35 = self.phi_135_inv(y_35, max_iter=max_iter)
            cdf_35 = self.phi_135(inverses_35.sum(dim=1))

            y_13 = torch.stack([S_E1, S_E3], dim=1)
            inverses_13 = self.phi_135_inv(y_13, max_iter=max_iter)
            cdf_13 = self.phi_135(inverses_13.sum(dim=1))

            y_15 = torch.stack([S_E1, S_E5], dim=1)
            inverses_15 = self.phi_135_inv(y_15, max_iter=max_iter)
            cdf_15 = self.phi_135(inverses_15.sum(dim=1))

            y_02345 = torch.stack([S_C, cdf_24, cdf_35], dim=1)
            inverses_02345 = self.phi_out_inv(y_02345, max_iter=max_iter)
            cdf_02345 = self.phi_out(inverses_02345.sum(dim=1))

            y_01345 = torch.stack([S_C, S_E4, cdf_135], dim=1)
            inverses_01345 = self.phi_out_inv(y_01345, max_iter=max_iter)
            cdf_01345 = self.phi_out(inverses_01345.sum(dim=1))

            y_01245 = torch.stack([S_C, cdf_24, cdf_15], dim=1)
            inverses_01245 = self.phi_out_inv(y_01245, max_iter=max_iter)
            cdf_01245 = self.phi_out(inverses_01245.sum(dim=1))

            y_01235 = torch.stack([S_C, S_E2, cdf_135], dim=1)
            inverses_01235 = self.phi_out_inv(y_01235, max_iter=max_iter)
            cdf_01235 = self.phi_out(inverses_01235.sum(dim=1))

            y_01234 = torch.stack([S_C, cdf_24, cdf_13], dim=1)
            inverses_01234 = self.phi_out_inv(y_01234, max_iter=max_iter)
            cdf_01234 = self.phi_out(inverses_01234.sum(dim=1))

        # 将所有计算的变量存储在字典中
        return {
            'x_shared': x_shared,
            'S_E1': S_E1,
            'S_E2': S_E2,
            'S_E3': S_E3,
            'S_E4': S_E4,
            'S_E5': S_E5,
            'S_C': S_C,
            'density_E1': density_E1,
            'density_E2': density_E2,
            'density_E3': density_E3,
            'density_E4': density_E4,
            'density_E5': density_E5,
            'density_C': density_C,
            'cdf_24': cdf_24,
            'cdf_135': cdf_135,
            'cdf_out': cdf_out,
            'cur_out1': cur_out1,
            'cur_out2': cur_out2,
            'cur_out3': cur_out3,
            'cur_24_2': cur_24_2,
            'cur_24_4': cur_24_4,
            'cur_135_1': cur_135_1,
            'cur_135_3': cur_135_3,
            'cur_135_5': cur_135_5,
            'cdf_35': cdf_35,
            'cdf_13': cdf_13,
            'cdf_15': cdf_15,
            'cdf_02345': cdf_02345,
            'cdf_01345': cdf_01345,
            'cdf_01245': cdf_01245,
            'cdf_01235': cdf_01235,
            'cdf_01234': cdf_01234
        }
    
    def survival_withCopula_joint_CIF_(self, event_index, intermediates):
        # 提取需要的中间变量
        density_E1 = intermediates['density_E1']
        density_E2 = intermediates['density_E2']
        density_E3 = intermediates['density_E3']
        density_E4 = intermediates['density_E4']
        density_E5 = intermediates['density_E5']
        cur_out2 = intermediates['cur_out2']
        cur_out3 = intermediates['cur_out3']
        cur_24_2 = intermediates['cur_24_2']
        cur_24_4 = intermediates['cur_24_4']
        cur_135_1 = intermediates['cur_135_1']
        cur_135_3 = intermediates['cur_135_3']
        cur_135_5 = intermediates['cur_135_5']
        cdf_02345 = intermediates['cdf_02345']
        cdf_01345 = intermediates['cdf_01345']
        cdf_01245 = intermediates['cdf_01245']
        cdf_01235 = intermediates['cdf_01235']
        cdf_01234 = intermediates['cdf_01234']

        if event_index == 0:
            return density_E1 * cur_out3 * cur_135_1 
        elif event_index == 1:
            return density_E2 * cur_out2 * cur_24_2 
        elif event_index == 2:
            return density_E3 * cur_out3 * cur_135_3 
        elif event_index == 3:
            return density_E4 * cur_out2 * cur_24_4 
        elif event_index == 4:
            return density_E5 * cur_out3 * cur_135_5 
        else:
            raise ValueError("Unsupported event index")
    def survival_withCopula_condition_CIF_intergral(self, event_index, intermediates):
        # 提取需要的中间变量
        density_E1 = intermediates['density_E1']
        density_E2 = intermediates['density_E2']
        density_E3 = intermediates['density_E3']
        density_E4 = intermediates['density_E4']
        density_E5 = intermediates['density_E5']
        cur_out2 = intermediates['cur_out2']
        cur_out3 = intermediates['cur_out3']
        cur_24_2 = intermediates['cur_24_2']
        cur_24_4 = intermediates['cur_24_4']
        cur_135_1 = intermediates['cur_135_1']
        cur_135_3 = intermediates['cur_135_3']
        cur_135_5 = intermediates['cur_135_5']
        cdf_02345 = intermediates['cdf_02345']
        cdf_01345 = intermediates['cdf_01345']
        cdf_01245 = intermediates['cdf_01245']
        cdf_01235 = intermediates['cdf_01235']
        cdf_01234 = intermediates['cdf_01234']

        if event_index == 0:
            return density_E1 * cur_out3 * cur_135_1 / cdf_02345
        elif event_index == 1:
            return density_E2 * cur_out2 * cur_24_2 / cdf_01345
        elif event_index == 2:
            return density_E3 * cur_out3 * cur_135_3 / cdf_01245
        elif event_index == 3:
            return density_E4 * cur_out2 * cur_24_4 / cdf_01235
        elif event_index == 4:
            return density_E5 * cur_out3 * cur_135_5 / cdf_01234
        else:
            raise ValueError("Unsupported event index")

    def survival_withCopula_condition_CIF_No_intergral(self, event_index, intermediates):
        cdf_out = intermediates['cdf_out']
        cdf_02345 = intermediates['cdf_02345']
        cdf_01345 = intermediates['cdf_01345']
        cdf_01245 = intermediates['cdf_01245']
        cdf_01235 = intermediates['cdf_01235']
        cdf_01234 = intermediates['cdf_01234']

        if event_index == 0:
            return 1 - cdf_out / cdf_02345
        elif event_index == 1:
            return 1 - cdf_out / cdf_01345
        elif event_index == 2:
            return 1 - cdf_out / cdf_01245
        elif event_index == 3:
            return 1 - cdf_out / cdf_01235
        elif event_index == 4:
            return 1 - cdf_out / cdf_01234
        else:
            raise ValueError("Unsupported event index")
    def survival_event_onlySurvivalFunc(self, event_index, intermediates):
        if event_index == 0:
            return intermediates['S_E1']
        elif event_index == 1:
            return intermediates['S_E2']
        elif event_index == 2:
            return intermediates['S_E3']
        elif event_index == 3:
            return intermediates['S_E4']
        elif event_index == 4:
            return intermediates['S_E5']
        else:
            raise ValueError("Unsupported event index")


    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_01_inv(y, tol=self.tol)
        cdf = self.phi_01(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi_01(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator    
class HACSurvival_competing3_shared(nn.Module):
    # with neural density estimators
    def __init__(self, phi_out,phi_01,phi_23, device, num_features, tol,  hidden_size=32, hidden_surv = 32, max_iter = 2000):
        super(HACSurvival_competing3_shared, self).__init__()
        self.tol = tol
        self.phi_out = phi_out
        self.phi_out_inv = PhiInv(phi_out).to(device)
        self.phi_01=phi_01
        self.phi_01_inv = PhiInv(phi_01).to(device)
        self.phi_23=phi_23
        self.phi_23_inv = PhiInv(phi_23).to(device)
        self.sumo_e1 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e2 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_e3 = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        self.sumo_c = NDE(hidden_size, layers = [hidden_size,hidden_size], layers_surv = [hidden_surv,hidden_surv,hidden_surv], dropout = 0.)
        
        self.shared_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x, t, c, max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            x = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        
            S_E3, density_E3 = self.sumo_e3(x, t, gradient = True)
            S_E3 = S_E3.squeeze()
            event3_log_density = torch.log(density_E3).squeeze()   

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            assert (S_E3 >= 0.).all() and (
                S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )          
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_01 = torch.stack([S_C,S_E1], dim=1)
            inverses_01 = self.phi_01_inv(y_01, max_iter = max_iter)
            cdf_01 = self.phi_01(inverses_01.sum(dim=1))
            y_23 = torch.stack([S_E2,S_E3], dim=1)
            inverses_23 = self.phi_23_inv(y_23, max_iter = max_iter)
            cdf_23 = self.phi_23(inverses_23.sum(dim=1))

            y_out = torch.stack([cdf_01,cdf_23], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]     
            
            # grad_cdf_out_SE1 = torch.autograd.grad(outputs=cdf_out, inputs=S_E1, grad_outputs=torch.ones_like(cdf_out), create_graph=True,retain_graph=True)[0]
            # grad_cdf_out_SE2 = torch.autograd.grad(outputs=cdf_out, inputs=S_E2, grad_outputs=torch.ones_like(cdf_out), create_graph=True,retain_graph=True)[0]

            # print('y_out',y_out.shape)
            # TODO: Only take gradients with respect to one dimension of y at at time

            # y_in = torch.stack([S_E1,S_E2], dim=1)
            # print('yin',y_in.shape)

            cur_01_1 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 0]
            cur_01_2 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 1]
            cur_23_1 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 0]     

            cur_23_2 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 1]     

        
            # cur3 = torch.autograd.grad(
            #     cdf.sum(), y, create_graph=True)[0][:, 2]      
            # *cur_out1*cur_in1
            # logL = event_log_density + c * torch.log(cur1) + censoring_log_density + (1-c) * torch.log(cur2)
            # logL = event1_log_density + (c==1) * torch.log(cur1) + event2_log_density + (c==2) * torch.log(cur2)+censoring_log_density + (c==0) * torch.log(cur3)  #density L1
            # print(cur_out1)

            # print(cur_in1)
            # logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_out1*cur_in1)) + (c==2) *event2_log_density + (c==2) * (torch.log(cur_out1*cur_in2))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out2)     #density L2
            logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_out1*cur_01_2)) + (c==2) *event2_log_density + (c==2) * (torch.log(cur_out2*cur_23_1))+ (c==3) *event3_log_density + (c==3) * (torch.log(cur_out2*cur_23_2))+(c==0) *censoring_log_density + (c==0) * torch.log(cur_out1*cur_01_1)
            # logL = (c==1) *event1_log_density + (c==1) * (torch.log(cur_in1)) + (c==2) *event2_log_density + (c==2) * (torch.log(cur_in2))+(c==0) *censoring_log_density      #density L2
        return torch.sum(logL)
    


        

    def survival_withCopula_condition_CIF_intergral(self, t,x, event_index,max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            x = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        
            S_E3, density_E3 = self.sumo_e3(x, t, gradient = True)
            S_E3 = S_E3.squeeze()
            event3_log_density = torch.log(density_E3).squeeze()   

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            assert (S_E3 >= 0.).all() and (
                S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )          
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_01 = torch.stack([S_C,S_E1], dim=1)
            inverses_01 = self.phi_01_inv(y_01, max_iter = max_iter)
            cdf_01 = self.phi_01(inverses_01.sum(dim=1))
            y_23 = torch.stack([S_E2,S_E3], dim=1)
            inverses_23 = self.phi_23_inv(y_23, max_iter = max_iter)
            cdf_23 = self.phi_23(inverses_23.sum(dim=1))

            y_out = torch.stack([cdf_01,cdf_23], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]     
            
            cur_01_1 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 0]
            cur_01_2 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 1]
            cur_23_1 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 0]     

            cur_23_2 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 1]     

            y_out_023 = torch.stack([S_C,cdf_23], dim=1)
            inverses_out_023 = self.phi_out_inv(y_out_023, max_iter = max_iter)
            cdf_out_023 = self.phi_out(inverses_out_023.sum(dim=1))

            y_out_013 = torch.stack([cdf_01,S_E3], dim=1)
            inverses_out_013 = self.phi_out_inv(y_out_013, max_iter = max_iter)
            cdf_out_013 = self.phi_out(inverses_out_013.sum(dim=1))

            y_out_012 = torch.stack([cdf_01,S_E2], dim=1)
            inverses_out_012 = self.phi_out_inv(y_out_012, max_iter = max_iter)
            cdf_out_012 = self.phi_out(inverses_out_012.sum(dim=1))

        if event_index == 0:
            return density_E1.squeeze()*cur_out1*cur_01_2/cdf_out_023
        elif event_index == 1:
            return density_E2.squeeze()*cur_out2*cur_23_1/cdf_out_013
        elif event_index == 2:
            return density_E3.squeeze()*cur_out2*cur_23_2/cdf_out_012
        else:
            raise ValueError("Unsupported event index")  

    def survival_withCopula_condition_CIF_No_intergral(self, t,x, event_index,max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            x = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        
            S_E3, density_E3 = self.sumo_e3(x, t, gradient = True)
            S_E3 = S_E3.squeeze()
            event3_log_density = torch.log(density_E3).squeeze()   

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            assert (S_E3 >= 0.).all() and (
                S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )          
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_01 = torch.stack([S_C,S_E1], dim=1)
            inverses_01 = self.phi_01_inv(y_01, max_iter = max_iter)
            cdf_01 = self.phi_01(inverses_01.sum(dim=1))
            y_23 = torch.stack([S_E2,S_E3], dim=1)
            inverses_23 = self.phi_23_inv(y_23, max_iter = max_iter)
            cdf_23 = self.phi_23(inverses_23.sum(dim=1))

            y_out = torch.stack([cdf_01,cdf_23], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]     
            
            cur_01_1 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 0]
            cur_01_2 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 1]
            cur_23_1 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 0]     

            cur_23_2 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 1]        

            y_out_023 = torch.stack([S_C,cdf_23], dim=1)
            inverses_out_023 = self.phi_out_inv(y_out_023, max_iter = max_iter)
            cdf_out_023 = self.phi_out(inverses_out_023.sum(dim=1))

            y_out_013 = torch.stack([cdf_01,S_E3], dim=1)
            inverses_out_013 = self.phi_out_inv(y_out_013, max_iter = max_iter)
            cdf_out_013 = self.phi_out(inverses_out_013.sum(dim=1))

            y_out_012 = torch.stack([cdf_01,S_E2], dim=1)
            inverses_out_012 = self.phi_out_inv(y_out_012, max_iter = max_iter)
            cdf_out_012 = self.phi_out(inverses_out_012.sum(dim=1))

        if event_index == 0:
            return 1-cdf_out/cdf_out_023
        elif event_index == 1:
            return 1-cdf_out/cdf_out_013
        elif event_index == 2:
            return 1-cdf_out/cdf_out_012
        else:
            raise ValueError("Unsupported event index")  

    def survival_withCopula_joint_CIF_(self, t,x, event_index,max_iter = 2000):
        with torch.autograd.set_detect_anomaly(True):
            # print(c)
            x = self.shared_embedding(x)
            S_E1, density_E1 = self.sumo_e1(x, t, gradient = True)
            S_E1 = S_E1.squeeze()
            event1_log_density = torch.log(density_E1).squeeze()
            S_E2, density_E2 = self.sumo_e2(x, t, gradient = True)
            S_E2 = S_E2.squeeze()
            event2_log_density = torch.log(density_E2).squeeze()        
            S_E3, density_E3 = self.sumo_e3(x, t, gradient = True)
            S_E3 = S_E3.squeeze()
            event3_log_density = torch.log(density_E3).squeeze()   

            # S_C = survival(t, self.shape_c, self.scale_c, x_beta_c)
            S_C, density_C = self.sumo_c(x, t, gradient = True)
            S_C = S_C.squeeze()
            censoring_log_density = torch.log(density_C).squeeze()
            # Check if Survival Function of Event and Censoring are in [0,1]
            assert (S_E1 >= 0.).all() and (
                S_E1 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E1, )
            assert (S_E2 >= 0.).all() and (
                S_E2 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E2, )
            assert (S_E3 >= 0.).all() and (
                S_E3 <= 1.+1e-10).all(), "t %s, output %s" % (t, S_E3, )          
            assert (S_C >= 0.).all() and (
                S_C <= 1.+1e-10).all(), "t %s, output %s" % (t, S_C, )      
            
            # Partial derivative of Copula using ACNet
            y_01 = torch.stack([S_C,S_E1], dim=1)
            inverses_01 = self.phi_01_inv(y_01, max_iter = max_iter)
            cdf_01 = self.phi_01(inverses_01.sum(dim=1))
            y_23 = torch.stack([S_E2,S_E3], dim=1)
            inverses_23 = self.phi_23_inv(y_23, max_iter = max_iter)
            cdf_23 = self.phi_23(inverses_23.sum(dim=1))

            y_out = torch.stack([cdf_01,cdf_23], dim=1)
            inverses_out = self.phi_out_inv(y_out, max_iter = max_iter)
            cdf_out = self.phi_out(inverses_out.sum(dim=1))

            cur_out1 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 0]
            
            cur_out2 = torch.autograd.grad(
                cdf_out.sum(), y_out, create_graph=True,retain_graph=True)[0][:, 1]     
            
            cur_01_1 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 0]
            cur_01_2 = torch.autograd.grad(
                cdf_01.sum(), y_01, create_graph=True,retain_graph=True)[0][:, 1]
            cur_23_1 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 0]     

            cur_23_2 = torch.autograd.grad(
                cdf_23.sum(), y_23, create_graph=True,retain_graph=True)[0][:, 1]         
        if event_index == 0:
            return density_E1.squeeze()*cur_out1*cur_01_2
        elif event_index == 1:
            return density_E2.squeeze()*cur_out2*cur_23_1
        elif event_index == 2:
            return density_E3.squeeze()*cur_out2*cur_23_2
        else:
            raise ValueError("Unsupported event index")  


    def survival_event2(self, t, X):
        with torch.no_grad():
            X = self.shared_embedding(X)
            result = self.sumo_e2.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
        # return cdf
    def survival_event3(self, t, X):
        with torch.no_grad():
            X= self.shared_embedding(X)
            result = self.sumo_e3.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
        # return cdf

    def survival_event1(self, t, X):
        with torch.no_grad():
            X = self.shared_embedding(X)
            result = self.sumo_e1.survival(X, t)
            # S_E1,a= self.sumo_e1(X, t)
            # S_E1 = S_E1.squeeze()
            # S_E2 ,b= self.sumo_e2(X, t)
            # S_E2 = S_E2.squeeze()
            # S_C = torch.full_like(S_E1, 0.5)
            # y = torch.stack([S_E1,S_E2, S_C], dim=1)
            # inverses = self.phi_inv(y, max_iter = 2000)
            # cdf = self.phi(inverses.sum(dim=1))
        return result
    def cond_cdf(self, y, mode='cond_cdf', others=None, tol=1e-8):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_01_inv(y, tol=self.tol)
        cdf = self.phi_01(inverses.sum(dim=1))
        
        if mode == 'cdf':
            return cdf
        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur        
        elif mode =='cond_cdf':
            target_dims = others['cond_dims']
            
            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi_01(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator
    # def survival_event_predict_withCop(self, t, X, event_index):
    #     # 根据事件索引选择对应的生存预测函数
    #     if event_index == 0:
    #         return self.survival_event1_withCopula(t, X)
    #     elif event_index == 1:
    #         return self.survival_event2_withCopula(t, X)
    #     elif event_index == 2:
    #         return self.survival_event3_withCopula(t, X)
    #     # elif event_index == 3:
    #     #     return self.survival_event4(t, X)
    #     # elif event_index == 4:
    #     #     return self.survival_event5(t, X)
    #     else:
    #         raise ValueError("Unsupported event index")  
        
        # return cdf
    def survival_event_onlySurvivalFunc(self, t,X, event_index):
        # 根据事件索引选择对应的生存预测函数
        if event_index == 0:
            return self.survival_event1(t, X)
        elif event_index == 1:
            return self.survival_event2(t, X)
        elif event_index == 2:
            return self.survival_event3(t, X)
        # elif event_index == 3:
        #     return self.survival_event4(t, X)
        # elif event_index == 4:
        #     return self.survival_event5(t, X)
        else:
            raise ValueError("Unsupported event index")  
        
        # return cdf

####################################################################################
# Tests
####################################################################################

def sample(net, ndims, N, device, seed=142857):
    """
    Note: this does *not* use the efficient method described in the paper.
    Instead, we will use the naive method, i.e., conditioning on each 
    variable in turn and then applying the inverse CDF method on the resultant conditional
    CDF. 

    This method will work on all generators (even those defined by ACNet), and is
    the simplest method assuming no knowledge of the mixing variable M is known.
    """
    # Store old seed and set new seed
    old_rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    # random variable generation
    U = torch.rand(N, ndims).to(device)

    for dim in range(1, ndims):
        print('Sampling from dim: %s' % dim)
        y = U[:, dim].detach().clone()

        def cond_cdf_func(u):
            U_ = U.clone().detach()
            U_[:, dim] = u
            cond_cdf = net.cond_cdf(U_[:, :(dim+1)], "cond_cdf",
                           others={'cond_dims': list(range(dim))})
            return cond_cdf

        # Call inverse using the conditional cdf `M` as the function.
        # Note that the weight parameter is set to None since `M` is not parameterized,
        # i.e., hardcoded as the conditional cdf itself.
        U[:, dim] = bisection_default_increasing(cond_cdf_func, y,tol=1e-8).detach()

    # Revert to old random state.
    torch.random.set_rng_state(old_rng_state)

    return U



def test_grad_of_phi():
    phi_net = MixExpPhi()
    phi_inv = PhiInv(phi_net)
    query = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [1., 1., 1.]]).requires_grad_(True)

    gradcheck(phi_net, (query), eps=1e-9)
    gradgradcheck(phi_net, (query,), eps=1e-9)


def test_grad_y_of_inverse():
    phi_net = MixExpPhi()
    phi_inv = PhiInv(phi_net)
    query = torch.tensor(
        [[0.1, 0.2], [0.2, 0.3], [0.25, 0.7]]).requires_grad_(True)

    gradcheck(phi_inv, (query, ), eps=1e-10)
    gradgradcheck(phi_inv, (query, ), eps=1e-10)


def test_grad_w_of_inverse():
    phi_net = MixExpPhi2FixedSlope()
    phi_inv = PhiInv(phi_net)

    eps = 1e-8
    new_phi_inv = copy.deepcopy(phi_inv)

    # Jitter weights in new_phi.
    new_phi_inv.phi.mix.data = phi_inv.phi.mix.data + eps

    query = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.99, 0.99, 0.99]]).requires_grad_(True)
    old_value = phi_inv(query).sum()
    old_value.backward()
    anal_grad = phi_inv.phi.mix.grad
    new_value = new_phi_inv(query).sum()
    num_grad = (new_value-old_value)/eps

    print('gradient of weights (anal)', anal_grad)
    print('gradient of weights (num)', num_grad)


def test_grad_y_of_pdf():
    phi_net = MixExpPhi()
    query = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.99, 0.99, 0.99]]).requires_grad_(True)
    cop = SurvivalCopula(phi_net)
    def f(y): return cop(y, mode='pdf')
    gradcheck(f, (query, ), eps=1e-8)
    # This fails sometimes if rtol is too low..?
    gradgradcheck(f, (query, ), eps=1e-8, atol=1e-6, rtol=1e-2)


def plot_pdf_and_cdf_over_grid():
    phi_net = MixExpPhi()
    cop = Copula(phi_net)

    n = 500
    x1 = np.linspace(0.001, 1, n)
    x2 = np.linspace(0.001, 1, n)
    xv1, xv2 = np.meshgrid(x1, x2)
    xv1_tensor = torch.tensor(xv1.flatten())
    xv2_tensor = torch.tensor(xv2.flatten())
    query = torch.stack((xv1_tensor, xv2_tensor)
                        ).double().t().requires_grad_(True)
    cdf = cop(query, mode='cdf')
    pdf = cop(query, mode='pdf')

    assert abs(pdf.mean().detach().numpy().sum() -
               1) < 1e-6, 'Mean of pdf over grid should be 1'
    assert abs(cdf[-1].detach().numpy().sum() -
               1) < 1e-6, 'CDF at (1..1) should be should be 1'


def plot_cond_cdf():
    phi_net = MixExpPhi()
    cop = Copula(phi_net)

    n = 500
    xv2 = np.linspace(0.001, 1, n)
    xv2_tensor = torch.tensor(xv2.flatten())
    xv1_tensor = 0.9 * torch.ones_like(xv2_tensor)
    x = torch.stack([xv1_tensor, xv2_tensor], dim=1).requires_grad_(True)
    cond_cdf = cop(x, mode="cond_cdf", others={'cond_dims': [0]})

    plt.figure()
    plt.plot(cond_cdf.detach().numpy())
    plt.title('Conditional CDF')
    plt.draw()
    plt.pause(0.01)


def plot_samples():
    phi_net = MixExpPhi()
    cop = Copula(phi_net)

    s = sample(cop, 2, 2000, seed=142857)
    s_np = s.detach().numpy()

    plt.figure()
    plt.scatter(s_np[:, 0], s_np[:, 1])
    plt.title('Sampled points from Copula')
    plt.draw()
    plt.pause(0.01)


def plot_loss_surface():
    phi_net = MixExpPhi2FixedSlope()
    cop = Copula(phi_net)

    s = sample(cop, 2, 2000, seed=142857)
    s_np = s.detach().numpy()

    l = []
    x = np.linspace(-1e-2, 1e-2, 1000)
    for SS in x:
        new_cop = copy.deepcopy(cop)
        new_cop.phi.mix.data = cop.phi.mix.data + SS

        loss = -torch.log(new_cop(s, mode='pdf')).sum()
        l.append(loss.detach().numpy().sum())

    plt.figure()
    plt.plot(x, l)
    plt.title('Loss surface')
    plt.draw()
    plt.pause(0.01)


def test_training(test_grad_w=False):
    gen_phi_net = MixExpPhi()
    gen_phi_inv = PhiInv(gen_phi_net)
    gen_cop = Copula(gen_phi_net)

    s = sample(gen_cop, 2, 2000, seed=142857)
    s_np = s.detach().numpy()

    ideal_loss = -torch.log(gen_cop(s, mode='pdf')).sum()

    train_cop = copy.deepcopy(gen_cop)
    train_cop.phi.mix.data *= 1.5
    train_cop.phi.slope.data *= 1.5
    print('Initial loss', ideal_loss)
    optimizer = optim.Adam(train_cop.parameters(), lr=1e-3)

    def numerical_grad(cop):
        # Take gradients w.r.t to the first mixing parameter
        print('Analytic gradients:', cop.phi.mix.grad[0])

        old_cop, new_cop = copy.deepcopy(cop), copy.deepcopy(cop)
        # First order approximation of gradient of weights
        eps = 1e-6
        new_cop.phi.mix.data[0] = cop.phi.mix.data[0] + eps
        x2 = -torch.log(new_cop(s, mode='pdf')).sum()
        x1 = -torch.log(cop(s, mode='pdf')).sum()

        first_order_approximate = (x2-x1)/eps
        print('First order approx.:', first_order_approximate)

    for iter in range(100000):
        optimizer.zero_grad()
        loss = -torch.log(train_cop(s, mode='pdf')).sum()
        loss.backward()
        print('iter', iter, ':', loss, 'ideal loss:', ideal_loss)
        if test_grad_w:
            numerical_grad(train_cop)
        optimizer.step()


if __name__ == '__main__':
    import torch.optim as optim
    from torch.autograd import gradgradcheck, gradcheck
    import numpy as np
    import logging as log
    import matplotlib.pyplot as plt
    import copy

    torch.set_default_tensor_type(torch.DoubleTensor)

    test_grad_of_phi()
    test_grad_y_of_inverse()
    test_grad_w_of_inverse()
    test_grad_y_of_pdf()

    plot_pdf_and_cdf_over_grid()
    plot_cond_cdf()
    plot_samples()
    """ Uncomment for rudimentary training. 
        Note: very slow and unrealistic.
    plot_loss_surface()
    test_training()
    """



