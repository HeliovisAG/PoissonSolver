import numpy as np
import matplotlib.pyplot as plt
from PoissonMatrixSolver2D import PoissonMatrixSolver2D

N = 200
L = 20             # Ausdehnung in m
k_soil=50e-3     # Elektrische Leitfähigkeit Boden (lehmig) in S/m
I = 20000         # Strom der über den Staberder eingespeist wird
# Definition des Staberders
xE = 10      # x-Position des Erders in m
LE = 13            # Länge des Erders in m
dE = 1         # Dicke des Erders in m
ds = 0.05 # Dicke der nichtleitenden Oberflöche in m

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

sigma = np.ones((N, N))*k_soil  # Erdreich

mask = (X>=xE) & (X<=xE+dE) & (Y>=L-LE) & (Y<=L-ds)
mask_surf = Y > L-ds
sigma[mask] = 358              # Leitfähigkeit von Kupfer in S/m
sigma[mask_surf] = 1e-8
mask_left = X < ds
sigma[mask_left] = 1e-8
mask_right = X > L-ds
sigma[mask_right] = 1e-8
mask_bottom = Y < ds
sigma[mask_bottom] = 1e-8

Q = np.zeros((N, N))                # Quellmatrix
V = LE*dE**2          # Volumen des Erders in m³
Q[mask] = -I/V  # A/m³    

# Randbedingungen
bc = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
# Startwert
phi_init = np.ones((N, N)) * 0.0
ps = PoissonMatrixSolver2D(N=N, L=L, sigma=sigma, Q=Q, phi_init=phi_init, bc_values=bc)
phi = ps.solve()


fig, ax = plt.subplots()

dPhidy, dPhidx = np.gradient(phi, L/N, L/N)

Ex = -dPhidx
Ey = -dPhidy
E = np.sqrt(Ex**2 + Ey**2)

cf = ax.contourf(X, Y, phi, levels=40, cmap="inferno")
#cf = ax.contourf(X, Y, E, levels=40, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Potential [V]")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Potentialfeld")
plt.show()
