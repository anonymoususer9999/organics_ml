import torch
from neuro_spectra.model.ORGaNICs_models.ORGaNICs import ORGaNICs
import torch.nn.functional as F
from torch.func import jacrev, vmap
import torch.nn as nn


def dynm_fun(f):
    """A wrapper for the dynamical function"""
    def wrapper(self, t, var, input):
        new_fun = lambda t, var, input: f(self, t, var, input)
        return new_fun(t, var, input)
    return wrapper


class ORGaNICs2Dspectra(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int, 
                 sigma=0.1,
                 device=None,
                 dtype=None,
                 set_diag_one=False):

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ORGaNICs2Dspectra, self).__init__()
        # Dimensionality of the circuit
        self.n = output_size
        input_size = input_size

        # Define the parameters for the weight matrices
        wzx, _ = ORGaNICs.make_weight_matrices(input_size, output_size)
        self.Wzx = nn.Parameter(torch.tensor(wzx, **factory_kwargs), requires_grad=False)
        self.log_Way = nn.Parameter(- 1. * torch.ones((output_size, output_size), **factory_kwargs))
        self.log_Way.data.fill_diagonal_(0.)

        # Define the base b0 vector and semiseturation constant
        self.b0 = nn.Parameter(torch.zeros((output_size), **factory_kwargs), requires_grad=True)
        self.b1 = self.b0
        self.sigma = nn.Parameter(torch.tensor(sigma, **factory_kwargs), requires_grad=False)
        self.set_diag_one = set_diag_one

        # Define the time constants
        self.log_tauy = nn.Parameter(torch.tensor([-6.9]), requires_grad=True)
        self.log_taua = nn.Parameter(torch.tensor([-6.9]), requires_grad=True)

        # Define the noise parameters
        self.eta = nn.Parameter(1000 * torch.ones(2 * self.n, requires_grad=True, dtype=torch.cdouble))

        self.I = torch.eye(2 * self.n, requires_grad=False)

    def Way(self):
        return self.log_Way.exp()
    
    def tauy(self):
        return self.log_tauy.exp()
    
    def taua(self):
        return self.log_taua.exp()
    
    def B0(self):
        return torch.sigmoid(self.b0)
    
    def B1(self):
        return torch.sigmoid(self.b1)
    
    def Q(self):
        return torch.diag(self.eta ** 2)
    
    def ss(self, x):
        """
        This function calculates the steady state of the system.
        Note that the input here has  abatch dimension at location 0.
        """
        z = F.linear(x, self.Wzx, bias=None)

        if self.set_diag_one:
            with torch.no_grad():
                self.log_Way.data.fill_diagonal_(0.)

        B0 = self.B0()
        B1 = self.B1()

        gated_z = B1 ** 2 * F.relu(z) ** 2
        pooled_response = F.linear(gated_z, self.Way(), bias=None)
        norm_response = gated_z / ((self.sigma * B0) ** 2 + pooled_response)

        return norm_response, ((self.sigma * B0) ** 2 + pooled_response)
    
    def spectral_matrix(self, omega, J):
        return torch.inverse(J + 1j * omega * self.I) @ self.Q() @ torch.inverse(torch.transpose(J - 1j * omega * self.I, 0, 1))

    def forward(self, x, omega):
        """Return the spectral density matrix for different inputs and frequencies"""
        # Calculate the steady state
        ss_y, ss_a = self.ss(x)
        # stack the steady state
        ss = torch.cat((ss_y, ss_a), dim=1)

        # Calculate the jacobian at the steady state
        J = []
        for i in range(x.shape[0]):
            J.append(jacrev(self._dynamical_fun, argnums=1)(0, ss[i, :], x[i, :]))

        # Calculate the spectral density matrix for each input and frequency
        S = torch.empty(x.shape[0], len(omega))

        for i in range(x.shape[0]):
            s = torch.zeros(len(omega))

            for j in range(len(omega)):
                spect_mat = self.spectral_matrix(omega[j], J[i])
                s[j] = torch.abs(torch.sum(spect_mat[0:self.n, 0:self.n])) / self.n ** 2

            S[i, :] = s

        # Calculate the spectral density prediction
        return J, S

    @dynm_fun
    def _dynamical_fun(self, t, var, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        var = var.squeeze(0)  # Remove the extra dimension
        y = var[0:self.n]
        a = var[self.n:]
        cc = (self.sigma * self.B0()) ** 2
        z = self.Wzx @ x
        dydt = (1 / self.tauy()) * (self.B1() * z - torch.sqrt(torch.relu(a)) *  y)
        dadt = (1 / self.taua()) * (-a + self.Way() @ (a * torch.relu(y)**2) + cc)
        return torch.cat((dydt, dadt))


