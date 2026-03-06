import numpy as np
import matplotlib.pyplot as plt
from poissonSolverMatrix2D import PoissonSolverMatrix2D

N = 400
L = 200
h = L/N             # örtliche Auflösung
dspx = 2          # Dicke der nichtleitenden Oberfläche in pixel
ds = dspx*h         # Dicke in m

sigma=0.01    # Elektrische Leitfähigkeit Raum  in S/m
sigmaCu = 5.8e7     # Elektrische Leitfähigkeit von Cu bei 20°C in 1/Ohm m
I = 2000          # Strom
Le = 18.1         # Tiefe Kugelerder

ps = PoissonSolverMatrix2D(N=N, L=L)
X = ps.X
Y = ps.Y

# Leiter1
x0 = L/2
y0 = L/2-Le
r=h
RR = np.sqrt((X-x0)**2 + (Y-y0)**2)
mask = RR < r 
ps.sigma[mask] = sigmaCu
V = 4*np.pi*r**2/3                       
ps.Q[mask]  = I/V            

# Leiter2
x0 = L/2
y0 = L/2+Le
r=h
RR = np.sqrt((X-x0)**2 + (Y-y0)**2)
mask = RR < r 
ps.sigma[mask] = sigmaCu
V = 4*np.pi*r**2/3                       
ps.Q[mask]  = I/V            

# dünne isolierende Schicht an der Oberfläche, adiabatische Randbedingung
#ps.sigma[:, N-dspx:N] = 1e-8                # oben
ps.sigma[:dspx, :] = 1e-8             # links   
ps.sigma[N-dspx:, :] = 1e-8         # rechts
#ps.sigma[:, :dspx] = 1e-8         # unten

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
phi = ps.phi
ps.fieldvektor()        # Berechnet die Feldstärke aus dem PotentialE
E, Ex, Ey = ps.E, ps.Ex, ps.Ey

# --- Visualisierung ---
fig, ax = plt.subplots()
#ps.drawImage(phi, fig, ax, 'Elektrostatisches Potential V')
ps.drawContour(phi, ax, levels=40)
E_s = np.abs(Ey[:,N//2])
E_s = E_s[dspx:N-dspx]
fig, ax = plt.subplots()
ax.plot(np.linspace(0, L, N-2*dspx), E_s, label='E_y')
plt.show()