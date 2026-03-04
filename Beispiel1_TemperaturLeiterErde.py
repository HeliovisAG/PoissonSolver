import numpy as np
import matplotlib.pyplot as plt
from poissonSolverMatrix2D import PoissonSolverMatrix2D

N = 200
L = 5.0
ps = PoissonSolverMatrix2D(N=N, L=L)
X = ps.X
Y = ps.Y

# Leitfähigkeitsmatrix
ps.sigma = ps.sigma*1                                # Wärmeleitfähigkeit Boden W/mK

# Leiter
x0 = 2
y0 = 3.2
r=0.2
RR = np.sqrt((X-x0)**2 + (Y-y0)**2)
mask = RR < r 
ps.sigma[mask] = 380.0                       # Wärmeleitfähigkeit Kupfer W/mK

# dünne isolierende Schicht an der Oberfläche, adiabatische Randbedingung
mask_surf = Y > L*0.99
ps.sigma[mask_surf] = 1e-8

# Quellenmatrix
ps.Q[mask] = 100                            # Quellendichte in A/m3

# Randbedingungen
def phi_left(x, y):  # Dirichlet links
    return 20.0
def phi_right(x, y):  # Dirichlet rechts
    return 20.0
def phi_bottom(x, y):  # Dirichlet unten
    return 20.0
def phi_top(x, y):  # Dirichlet oben
    return 20.0
bc = {'left':phi_left, 'right':phi_right, 'bottom':phi_bottom, 'top':phi_top}

ps.solve(bc)

# --- Visualisierung ---
fig, ax = plt.subplots()
#ps.drawImage(ps.phi, fig, ax, 'Temperatur °C')
ps.drawContour(ps.phi, ax)
plt.show()
