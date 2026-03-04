import numpy as np
import matplotlib.pyplot as plt
from poissonSolverMatrix2D import PoissonSolverMatrix2D

N = 400
L = 20
h = L/N             # örtliche Auflösung
ps = PoissonSolverMatrix2D(N=N, L=L)
X = ps.X
Y = ps.Y

sigma_soil=50e-3    # Elektrische Leitfähigkeit Boden (lehmig) in S/m
sigmaCu = 5.8e7     # Elektrische Leitfähigkeit von Cu bei 20°C in 1/Ohm m
I = 20000           # Strom der über den Staberder eingespeist wird

# Definition des Staberders
xE = 10             # x-Position des Erders in m
LE = 5             # Länge des Erders in m
dE = 0.05              # Dicke des Erders in m
ds = 20*h           # Dicke der nichtleitenden Oberflöche in m

# Leitfähigkeitsmatrix
ps.sigma = ps.sigma*sigma_soil                                
mask = (X>=xE) & (X<=xE+dE) & (Y>=L-LE-ds) & (Y<=L-ds)
ps.sigma[mask] = sigmaCu                       

mask_surf = Y > L-ds                      # dünne isolierende Schicht an der Oberfläche, adiabatische Randbedingung
ps.sigma[mask_surf] = 1e-8

# Quellenmatrix
V = LE*dE**2          # Volumen des Erders in m³
ps.Q[mask] = I/V  # A/m³

# Randbedingungen
def phi_left(x, y):  # Dirichlet links
    return 0.0
def phi_right(x, y):  # Dirichlet rechts
    return 0.0
def phi_bottom(x, y):  # Dirichlet unten
    return 0.0
def phi_top(x, y):  # Dirichlet oben
    return 0.0
bc = {'left':phi_left, 'right':phi_right, 'bottom':phi_bottom, 'top':phi_top}

ps.solve(bc)            # Berechnet das Potential durch Lösen der Poissongleichung
ps.fieldvektor()        # Berechnet die Feldstärke aus dem Potential

# --- Visualisierung ---
fig, ax = plt.subplots()
ps.drawImage(ps.phi, fig, ax, 'Elektrostatisches Potential V')
#ps.drawContour(phi, ax)
plt.show()
