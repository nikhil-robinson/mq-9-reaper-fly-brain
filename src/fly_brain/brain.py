"""Minimal leaky-integrate-and-fire connectome sim (PLAN.md Section 2)."""

import torch


class LIFBrain:
    def __init__(self, W, n_neurons, device="cpu", decay=0.9, threshold=1.0):
        self.W = W
        self.n = n_neurons
        self.device = device
        self.decay = decay
        self.threshold = threshold
        self.V = torch.zeros(n_neurons, device=device)
        self.S = torch.zeros(n_neurons, device=device)

    @torch.no_grad()
    def step(self, external_current=None):
        input_current = torch.sparse.mm(self.W, self.S.unsqueeze(1)).squeeze(1)
        if external_current is not None:
            input_current = input_current + external_current

        self.V = self.V * self.decay + input_current
        self.S = (self.V > self.threshold).float()
        self.V = self.V * (1 - self.S)
        return self.S
