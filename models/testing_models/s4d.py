""" Standalone version of Structured (Sequence) State Space (S4) model. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from opt_einsum import contract

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class Modrelu(modrelu):
    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'modrelu':
        return Modrelu(size)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, **kernel_args):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.H = H
        self.N = N // 2

        C = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))

        B = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.B = nn.Parameter(_c2r(B))

        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        t: (L,)
        """

        # Materialize parameters
        C = _r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        dt = torch.exp(self.log_dt)  # (H)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def step(self, u, state):
        ''' dt: float'''
        C = _r2c(self.C)  # (C H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)
        dt = torch.exp(self.log_dt)  # (H)

        dtA = A * dt.unsqueeze(-1)  # (H N)
        self.dA = torch.exp(dtA)  # (H N)
        self.dC = C  # (C H N)
        self.dB = self.dC.new_ones(self.H, self.N) * (torch.exp(dtA) - 1.) / A  # (H N)

        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                     + contract("h n, b h -> b h n", self.dB, u)
        y = contract("h n, b h n -> b h", self.dC, next_state)

        return 2 * y.real, next_state

    def default_state(self, *batch_shape, device='cuda'):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=device)
        return state


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, activation='gelu', bidirectional=False,
                 postact='glu', **kernel_args):
        super().__init__()
        ##TODO: implement bidirection
        assert not bidirectional, 'not implemented yet'
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        # self.activation = nn.SiLU() #nn.GELU()
        self.activation = Activation(activation)
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact == 'glu':

            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )
        else:
            self.output_linear = nn.Conv1d(self.h, self.h, kernel_size=1)

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)
        # breakpoint()
        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None  # Return a dummy state to satisfy this repo's interface, but this can be modified

    def step(self, u, state, **kwargs):
        """ Step one time step as a recurrent model. Intended to be used during validation.
        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        # assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)

        y = y + contract('bh,h->bh', u, self.D)

        y = self.activation(y)
        y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)

        return y, next_state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class S4DJointKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, **kernel_args):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.H = H
        self.N = N // 2

        C = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))

        C_aux = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.C_aux = nn.Parameter(_c2r(C_aux))

        E = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.E = nn.Parameter(_c2r(E))

        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        C = _r2c(self.C)  # (H N)
        C_aux = _r2c(self.C_aux)  # (H N)
        E = _r2c(self.E)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C_main = C * (torch.exp(dtA) - 1.) / A
        C_aux = C_aux * (torch.exp(dtA) - 1.) / A
        E_main = E * C_main
        E_aux = E * C_aux
        Ku = 2 * torch.einsum('hn, hnl -> hl', C_main, torch.exp(K)).real
        Kx = 2 * torch.einsum('hn, hnl -> hl', E_main, torch.exp(K)).real

        Ku_aux = 2 * torch.einsum('hn, hnl -> hl', C_aux, torch.exp(K)).real
        Kx_aux = 2 * torch.einsum('hn, hnl -> hl', E_aux, torch.exp(K)).real

        return Ku, Kx, Ku_aux, Kx_aux

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def step(self, u, x, state, **kwargs):
        C = _r2c(self.C)  # (C H N)
        C_aux = _r2c(self.C_aux)  # (H N)
        E = _r2c(self.E)  # (C H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)
        dt = torch.exp(self.log_dt)  # (H)
        # Incorporate dt into A
        dtA = A * dt.unsqueeze(-1)  # (H N)

        self.dA = torch.exp(dtA)  # (H N)
        self.dC_main = C  # (C H N)
        self.dC_aux = C_aux  # (C H N)
        self.dB = self.dC_main.new_ones(self.H, self.N) * (torch.exp(dtA) - 1.) / A  # (H N)
        self.dE = E * (torch.exp(dtA) - 1.) / A
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                     + contract("h n, b h -> b h n", self.dB, u) \
                     + contract("h n, b h -> b h n", self.dE, x)
        y = contract("h n, b h n -> b h", self.dC_main, next_state)
        y_aux = contract("h n, b h n -> b h", self.dC_aux, next_state)

        return 2 * y.real, 2 * y_aux.real, next_state

    def default_state(self, *batch_shape, device='cuda'):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=device)
        return state


class S4DJoint(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, activation='gelu', bidirectional=False,
                 postact='glu', **kernel_args):
        super().__init__()
        ##TODO: implement bidirection
        assert not bidirectional, 'not implemented yet'
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D_main = nn.Parameter(torch.randn(self.h))
        self.F_main = nn.Parameter(torch.randn(self.h))
        self.D_aux = nn.Parameter(torch.randn(self.h))
        self.F_aux = nn.Parameter(torch.randn(self.h))
        # SSM Kernel
        self.kernel = S4DJointKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        # self.activation = nn.SiLU() #nn.GELU()
        self.activation = Activation(activation)
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact == 'glu':

            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )
        else:
            self.output_linear = nn.Conv1d(self.h, self.h, kernel_size=1)

        if postact == 'glu':

            self.output_linear_aux = nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )
        else:
            self.output_linear_aux = nn.Conv1d(self.h, self.h, kernel_size=1)

    def forward(self, ins, t=None, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        u, x = ins
        if not self.transposed: x = x.transpose(-1, -2)
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        ku, kx, ku_aux, kx_aux = self.kernel(L=L)  # (H L)
        # Convolution
        ku_f = torch.fft.rfft(ku, n=2 * L)  # (H L)
        kx_f = torch.fft.rfft(kx, n=2 * L)  # (H L)
        ku_aux_f = torch.fft.rfft(ku_aux, n=2 * L)  # (H L)
        kx_aux_f = torch.fft.rfft(kx_aux, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        x_f = torch.fft.rfft(x, n=2 * L)  # (B H L)

        yu_main = torch.fft.irfft(u_f * ku_f, n=2 * L)[..., :L]  # (B H L)
        yx_main = torch.fft.irfft(x_f * kx_f, n=2 * L)[..., :L]  # (B H L)

        yu_aux = torch.fft.irfft(u_f * ku_aux_f, n=2 * L)[..., :L]  # (B H L)
        yx_aux = torch.fft.irfft(x_f * kx_aux_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y_main = yu_main + yx_main + u * self.D_main.unsqueeze(-1) + x * self.F_main.unsqueeze(-1)
        y_aux = yu_aux + yx_aux + u * self.D_aux.unsqueeze(-1) + x * self.F_aux.unsqueeze(-1)

        y_main = self.dropout(self.activation(y_main))
        y_main = self.output_linear(y_main)

        y_aux = self.dropout(self.activation(y_aux))
        y_aux = self.output_linear_aux(y_aux)
        if not self.transposed: y_main, y_aux = y_main.transpose(-1, -2), y_aux.transpose(-1, -2)
        return (y_main, y_aux), None  # Return a dummy state to satisfy this repo's interface, but this can be modified

    def step(self, ins, state, **kwargs):
        """ Step one time step as a recurrent model. Intended to be used during validation.
        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        # assert not self.training
        u, x = ins
        y, y_aux, next_state = self.kernel.step(u, x, state)  # (B C H)

        y = y + contract('bh,h->bh', u, self.D_main) + contract('bh,h->bh', x, self.F_main)
        y_aux = y_aux + contract('bh,h->bh', u, self.D_aux) + contract('bh,h->bh', x, self.F_aux)
        y = self.activation(y)
        y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        y_aux = self.activation(y_aux)
        y_aux = self.output_linear_aux(y_aux.unsqueeze(-1)).squeeze(-1)
        return (y, y_aux), next_state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


def test_step():
    device = 'cuda'
    B = 2
    H = 1
    N = 4
    L = 784
    # s4 = LatentS4Model(H, d_state=N, d_output=H, d_model=10, n_layers=2, dropout=0, transposed=True )
    s4 = S4DJoint(d_state=N, d_model=10)
    s4.to(device)
    # s4.eval()

    for module in s4.modules():
        if hasattr(module, 'setup_step'): module.setup_step()

    u = torch.rand(B, H, L).to(device)
    x = torch.rand(B, H, L).to(device)
    initial_state = s4.default_state(B, device=device)

    # initial_state = initial_state[..., :N // 2]
    # state = [iss.clone() for iss in initial_state]
    y, yx, _ = s4((u, x))
    print("output mean:\n", y, y.shape)
    print("output std:\n", yx, yx.shape)
    # print("final state:\n", final_state, final_state.shape)

    # Use Stepping

    # state =[iss.clone() for iss in initial_state]
    state = initial_state
    ys = []
    yxs = []
    for i, u_ in enumerate(torch.unbind(u, dim=-1)):
        y_, yx_, state = s4.step((u_, x[..., i]), state=state)
        ys.append(y_)
        yxs.append(yx_)
        # ystds.append(y_std_)
    ys = torch.stack(ys, dim=-1)
    yxs = torch.stack(yxs, dim=-1)
    # ystds = torch.stack(ystds, dim=-1)
    print("step outputs y:\n", ys, ys.shape)
    print("step outputs yx:\n", yx, yxs.shape)
    # print("step final state:\n", state)

    breakpoint()


if __name__ == '__main__':
    test_step()
