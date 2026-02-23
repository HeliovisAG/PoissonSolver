import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PoissonSolver2D import *

N = 200
L = 2             # Ausdehnung in m
k_soil=50e-3,     # Elektrische Leitfähigkeit Boden (lehmig) in S/m
I = 40000               # Strom der über den Staberder eingespeist wird

k = np.ones((N, N))*k_soil  # Erdreich
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing="ij")

# Definition des Staberders
xE = 1      # x-Position des Erders in m
LE = 1.3            # Länge des Erders in m
dE = 0.02         # Dicke des Erders in m
mask = (X>=xE) & (X<=xE+dE) & (Y>=L-LE) & (Y<=L)
k[mask] = 358              # Leitfähigkeit von Kupfer in S/m

Q = np.zeros((N, N))                # Quellmatrix
V = LE*dE**2          # Volumen des Erders in m³
Q[mask] = -I/V  # A/m³    

# Startwert
phi0 = np.ones((N, N)) * 20.0

bc_type = {
    "left": "dirichlet",
    "right": "dirichlet",
    "bottom": "dirichlet",
    "top": "neumann"
}

bc_values = {
    "left": 0.0,
    "right": 0.0,
    "bottom": 0.0,
    "top": None
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
h = L/N
dPhidy, dPhidx = np.gradient(Phi, h, h)

Ex = -dPhidx
Ey = -dPhidy
E = np.sqrt(Ex**2 + Ey**2)

cf = ax.contourf(X, Y, Phi, levels=40, cmap="inferno")
#cf = ax.contourf(X, Y, E, levels=40, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Potential [V]")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Potentialfeld")
plt.show()
