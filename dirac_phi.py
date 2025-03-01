"""
Contains the generator for ACNet (in the class DiracPhi).
This is named as such since the mixing variable is a convex combination of dirac delta functions.
"""

import torch
import torch.nn as nn


class DiracPhi(nn.Module):
    '''
    TODO: streamline 3 cases in forward pass.
    '''

    def __init__(self, depth, widths, lc_w_range, shift_w_range, device, tol):
        super(DiracPhi, self).__init__()

        # Depth is the number of hidden layers.
        self.depth = depth
        self.widths = widths
        self.lc_w_range = lc_w_range
        self.shift_w_range = shift_w_range
        self.device = device
        self.tol = tol
        assert self.depth == len(self.widths)

        self.shift_raw_, self.lc_raw_ = self.init_w()
        # print(self.shift_raw_, self.lc_raw_)
        self.shift_raw = nn.ParameterList(
            [nn.Parameter(x) for x in self.shift_raw_])
        self.lc_raw = nn.ParameterList([nn.Parameter(x) for x in self.lc_raw_])

    def init_w(self):
        sizes = self.get_sizes_w_()
        shift_sizes, lc_sizes = sizes[:self.depth], sizes[self.depth:]
        shift_tensors, lc_tensors = [], []

        for shift_size in shift_sizes:
            w = torch.zeros(shift_size)
            torch.nn.init.uniform_(w, *self.shift_w_range)
            shift_tensors.append(w)

        for lc_size in lc_sizes:
            w = torch.zeros(lc_size)
            torch.nn.init.uniform_(w, *self.lc_w_range)
            lc_tensors.append(w)

        return shift_tensors, lc_tensors

    def get_sizes_w_(self):
        depth, widths = self.depth, self.widths
        lc_sizes, shift_sizes = [], []

        # Shift weights
        prev_width = 1
        for pos in range(depth):
            width = widths[pos]
            shift_sizes.append((width,))
            prev_width = width

        # Linear combination weights
        for pos in range(depth):
            width = widths[pos]
            if pos < depth-1:
                next_width = widths[pos+1]
            else:
                next_width = 1
            lc_sizes.append((next_width, width))

        return shift_sizes + lc_sizes

    def forward(self, t_raw):
        s_raw, lc_raw = self.shift_raw, self.lc_raw
        depth = self.depth
        num_queries = t_raw.numel()
        t = t_raw.flatten()

        # State[i] has a dimension of N x num_inputs (to current layer)
        # Initial state has a dimension of N x 1.
        initial_state = torch.ones((num_queries, 1))
        states = [initial_state]

        # Positive function.
        def pf(x): return torch.exp(x)

        for ell in range(depth+1):
            F_prev = states[-1]
            if ell == 0:
                # In the first layer, there is only a shift, since convex combinations
                # are meaningless.
                n_outputs, n_inputs = s_raw[ell].size()[0], 1
                s = pf(s_raw[ell])
                s = s[None, :].expand(num_queries, -1)

                Fp = F_prev[:, None].expand(-1, 1, n_outputs).squeeze(dim=1).to(self.device)
                t_2d = t[:, None].expand(-1, n_outputs)

                next_state = Fp * torch.exp(-t_2d * s)
                states.append(next_state)

            elif ell == depth:
                # In the last layer, we only perform convex combinations.
                n_outputs, n_inputs = lc_raw[ell-1].size()
                lc = torch.softmax(lc_raw[ell-1], dim=1)
                lc = lc[None, :, :].expand(num_queries, -1, -1)
                Fp = F_prev[:, None, :].expand(-1, n_outputs, -1)
                next_state = (Fp * lc).sum(dim=2)
                states.append(next_state)

            else:
                # Main case.
                n_outputs, n_inputs = lc_raw[ell-1].size()
                s = pf(s_raw[ell])
                s = s[None, :].expand(num_queries, -1)
                lc = torch.softmax(lc_raw[ell-1], dim=1)
                lc = lc[None, :, :].expand(num_queries, -1, -1)
                Fp = F_prev[:, None, :].expand(-1, n_outputs, -1)
                t_2d = t[:, None].expand(-1, n_outputs)

                next_state = (Fp * lc).sum(dim=2) * torch.exp(-t_2d * s)
                states.append(next_state)

        output = states[-1]
        assert (output >= 0.).all() and \
        (output <= 1.+ self.tol).all(), "t %s, output %s, tol %s, max %s, min %s" % (t, output, self.tol, torch.max(output), torch.min(output))

        return output.reshape(t_raw.size())
