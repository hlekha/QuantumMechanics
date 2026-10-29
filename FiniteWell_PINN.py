#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import pi
from scipy import constants

a = 0.5
v0 = 20.0
x_max = 3.0

def potential(x):
    return torch.where(
        torch.abs(x) < a,
        torch.zeros_like(x),
        torch.full_like(x, v0)
    )


energy = nn.Parameter(torch.tensor(-1.0))
def _get_energy():
    return v0 * torch.sigmoid(energy)


def _get_tensor():
    tensor = torch.linspace(-x_max, x_max, 600).reshape(-1, 1)
    tensor.requires_grad_(True)

    return tensor


model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

E = nn.Parameter(torch.tensor(10.0))

def _get_psi_prime(psi, x):
    dpsi_dx = torch.autograd.grad(
        outputs=psi,
        inputs=x,
        grad_outputs=torch.ones_like(psi),
        create_graph=True
    )[0]

    return dpsi_dx

def _get_second_deriv(dpsi_dx, x):
    d2psi_dx2 = torch.autograd.grad(
        outputs=dpsi_dx,
        inputs=x,
        grad_outputs=torch.ones_like(dpsi_dx),
        create_graph=True
    )[0]

    return d2psi_dx2

"""
grad_outputs=torch.ones_like(psi):
    Differentiates sum of outputs - network processes each position independently so this gives derivative at each position.

[0]:
    gets gradient tensor
"""
def _get_physics_loss(psi, second_deriv, x):

    residual = second_deriv + (E- potential(x))*psi
    physics_loss = torch.mean(residual**2)

    return physics_loss

# Boundary Loss

def _get_boundary_loss():
    walls = torch.tensor([[-x_max], [x_max]])
    psi_walls = model(walls)

    boundary_loss = torch.mean(psi_walls**2)

    return boundary_loss

# Normalization Loss

def _get_normalization_loss(psi, x):
    normal = torch.trapezoid(
        (psi**2).squeeze(-1),
        x.squeeze(-1),
        dim=0
    )

    normal_loss = (normal - 1)**2

    return normal_loss, normal

""" 
We use unsqueeze so we can integrate along each position - changes tensor from [100,1] to [100,]
"""


optimizer = torch.optim.Adam(
    list(model.parameters()) + [energy],
    lr = 0.001
)

# Training

loss_history = []
physics_history = []
boundary_history = []
norm_history = []

for epoch in range(5000):
    optimizer.zero_grad()

    x = _get_tensor()
    psi = model(x)

    dpsi_dx = _get_psi_prime(psi, x)
    d2psi_dx2 = _get_second_deriv(dpsi_dx, x)

    physics_loss = _get_physics_loss(psi, d2psi_dx2, x)
    boundary_loss = _get_boundary_loss()
    norm_loss, normal = _get_normalization_loss(psi, x)
    loss = physics_loss + 10*boundary_loss + 10*norm_loss

    loss_history.append(loss.item())
    physics_history.append(physics_loss.item())
    boundary_history.append(boundary_loss.item())
    norm_history.append(norm_loss.item())

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss.item():.6f} | "
            f"Energy: {_get_energy().item():.6f} | "
            f"Norm: {normal.item():.6f}"
        )


x_plot = torch.linspace(-x_max, x_max, 1000).reshape(-1, 1)

with torch.no_grad():
    psi_pred = model(x_plot).squeeze(-1).numpy()
    V_plot = potential(x_plot).squeeze(-1).numpy()
    energy = _get_energy().item()

positions = x_plot.squeeze(-1).numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(positions, psi_pred, color="blue", label="PINN")
ax.axvline(-a, color="gray", linestyle=":")
ax.axvline(a, color="gray", linestyle=":")
ax.set_xlabel("Dimensionless position x")
ax.set_ylabel("Wavefunction", color="blue")
ax.grid(alpha=0.3)

ax_potential = ax.twinx()
ax_potential.plot(
    positions, V_plot,
    color="black", drawstyle="steps-mid", label="V(x)"
)
ax_potential.axhline(
    energy, color="red", linestyle="--",
    label=f"Energy = {energy:.4f}"
)
ax_potential.set_ylabel("Dimensionless potential / energy")
ax_potential.set_ylim(-0.05 * v0, 1.15 * v0)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_potential.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax.set_title("Finite square well")

axes[1].semilogy(
    loss_history, color="black", linewidth=2, label="Total"
)
axes[1].semilogy(physics_history, label="Physics", alpha=0.8)
axes[1].semilogy(
    10 * np.array(boundary_history),
    label="10 × boundary", alpha=0.8
)
axes[1].semilogy(
    10 * np.array(norm_history),
    label="10 × normalization", alpha=0.8
)
axes[1].set_xlabel("Training iteration")
axes[1].set_ylabel("Loss")
axes[1].set_title("Finite-well training loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:




