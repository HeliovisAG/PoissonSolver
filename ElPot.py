import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PoissonSolver2D import *


N = 200
L = 2             # Ausdehnung in m
h = L/N
k_soil=50e-3,     # Elektrische Leitfähigkeit Boden (lehmig) in S/m
I = 40000               # Strom der eingespeist wird

k = np.ones((N, N))*k_soil  # Erdreich
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing="ij")


Q = np.zeros((N, N))                # Quellmatrix
Q[100:110, 0:10] = -I  # A/m³
k[100:110, 0:10] = 358              # Kupferleiter 

k[105, :] = 358  


# Startwert
phi0 = np.ones((N, N)) * 20.0

bc_type = {
    "left": "dirichlet",
    "right": "dirichlet",
    "bottom": "neumann",
    "top": "dirichlet"
}

bc_values = {
    "left": 0.0,
    "right": 0.0,
    "bottom": None,
    "top": 0
}

solver = PoissonSolver2D(
    N, L,
    k_field=k,
    Q_field=Q,
    phi_init=phi0,
    bc_type=bc_type,
    bc_values=bc_values
)

fig, ax = plt.subplots()

result = solver.solve(omega=1, max_iter=30000, tol=1e-4)
print(result["converged"], result["iterations"])

Phi = result["phi"]
dPhidy, dPhidx = np.gradient(Phi, h, h)

Ex = -dPhidx
Ey = -dPhidy
E = np.sqrt(Ex**2 + Ey**2)

cf = ax.contourf(X, Y, Phi, levels=40, cmap="inferno")
#cf = ax.contourf(X, Y, E, levels=40, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperatur [V]")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Potentialfeld")
plt.show()
