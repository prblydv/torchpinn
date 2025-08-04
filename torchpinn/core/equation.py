# Base class for PDEs
import torch
from abc import ABC, abstractmethod


class PDE(ABC):
    """Abstract base class for a PDE."""

    @abstractmethod
    def residual(self, x, t, model):
        """Compute the PDE residual at input points."""
        pass


class BlackScholesPDE(PDE):
    def __init__(self, r: float = 0.05, sigma: float = 0.2):
        self.r = r
        self.sigma = sigma

    def residual(self, x, t, model):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = model(torch.cat([x, t], dim=1))

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        bs_residual = u_t + 0.5 * self.sigma**2 * x**2 * u_xx + self.r * x * u_x - self.r * u
        return bs_residual
