import numpy as np
import matplotlib.pyplot as plt
from poissonSolverMatrix2D import PoissonSolverMatrix2D

N = 400
L = 6.0
ps = PoissonSolverMatrix2D(N=N, L=L)
X = ps.X
Y = ps.Y

# Leitfähigkeitsmatrix
ps.sigma = ps.sigma*1                                # Wärmeleitfähigkeit Boden W/mK

# Leiter
x0 = 3
y0 = L-0.8
I = 774.0
A = 240e-6       # Querschnitt in m2
sigma20 = 5.8e7    # El. Leitfähigkeit von Cu bei 20°C in 1/Ohm m
alpha = 3.93e-4        # Temperaturkoeffizient von Cu
Topmax = 90

R20 = 1/sigma20/A  # Widerstand in Ohm/m bei 20°C
R_DC = R20*(1-alpha*(Topmax-20))
R_AC = R_DC * 1.03                         # Skin- und Proximity Effekt mit 3% angenommen
P = R_AC*I**2       # Leistung in W/m

r=np.sqrt(A/np.pi)
RR = np.sqrt((X-x0)**2 + (Y-y0)**2)
mask = RR < r 
ps.sigma[mask] = 380.0                       # Wärmeleitfähigkeit Kupfer W/mK

ps.sigma[Y > L*0.99] = 1e-8     # dünne isolierende Schicht oben, adiabatische Randbedingung 
#ps.sigma[X < L*0.01] = 1e-8    # dünne isolierende Schicht links, adiabatische Randbedingung 
#ps.sigma[X > L*0.99] = 1e-8    # dünne isolierende Schicht rechts, adiabatische Randbedingung 

# Quellenmatrix
ps.Q[mask] = P/A                            # Quellendichte in A/m3

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
ps.drawContour(ps.phi, fig, ax, 'Temperatur °C')
plt.show()
