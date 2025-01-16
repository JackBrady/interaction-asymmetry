import torch
from torch import nn
from torch.nn import functional as F


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))
        self.d = dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.query_pos = nn.Parameter(torch.rand(1, self.num_slots, dim))

    def iterate(self, f, x):
        for _ in range(self.iters):
            x = f(x)
        return x

    def step(self, slots, k, v):
        b = k.shape[0]
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps

        attn = attn / attn.sum(dim=-1, keepdim=True)
        updates = torch.einsum('bjd,bij->bid', v, attn)
        slots = self.gru(updates.reshape(-1, self.d), slots.reshape(-1, self.d)).reshape(b, -1, self.d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots

    def forward(self, inputs):
        b, n, d = inputs.shape
        slots = self.query_pos.expand(b, -1, -1)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        slots = self.iterate(lambda z: self.step(z, k, v), slots)
        slots = self.step(slots.detach(), k, v)
        return slots
