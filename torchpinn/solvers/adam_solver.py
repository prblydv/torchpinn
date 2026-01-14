# Adam solver implementation
import torch
import torch.nn as nn
import torch.optim as optim

class AdamSolver:
    def __init__(self, model, pde, domain_sampler, epochs=5000, lr=1e-3, verbose=True):
        self.model = model 
        self.pde = pde
        self.domain_sampler = domain_sampler
        self.epochs = epochs
        self.lr = lr
 
        self.verbose = verbose
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_loss(self, x, t):
        residual = self.pde.residual(x, t, self.model)
        return torch.mean(residual**2)

    def train(self):
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            x, t = self.domain_sampler()
            loss = self.compute_loss(x, t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if self.verbose and epoch % 100 == 0:
                print(f"[{epoch}/{self.epochs}] Loss: {loss.item():.6f}")