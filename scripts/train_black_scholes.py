# CLI training script for Black-Scholes
import torch
from torchpinn.core.equation import BlackScholesPDE
from torchpinn.core.model import PINN
from torchpinn.solvers.adam_solver import AdamSolver

def domain_sampler(batch_size=512):
    x = torch.rand(batch_size, 1) * 1.0  # x in [0, 1]
    t = torch.rand(batch_size, 1) * 1.0  # t in [0, 1]
    return x, t

def main():  
    model = PINN(in_dim=2, out_dim=1)
    pde = BlackScholesPDE(r=0.05, sigma=0.2)
    solver = AdamSolver(model=model, pde=pde, domain_sampler=domain_sampler, epochs=2000, lr=1e-3)
    solver.train()


if __name__ == "__main__":
    main()