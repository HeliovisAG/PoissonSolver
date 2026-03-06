import numpy as np
import matplotlib.pyplot as plt
from poissonSolverMatrix2D import PoissonSolverMatrix2D

N = 400
L = 1200
h = L/N             # örtliche Auflösung
print(h)
dspx = 2          # Dicke der nichtleitenden Oberfläche in pixel
ds = dspx*h         # Dicke in m

sigma_soil=0.01    # Elektrische Leitfähigkeit Boden (lehmig) in S/m
sigmaCu = 5.8e7     # Elektrische Leitfähigkeit von Cu bei 20°C in 1/Ohm m
I = 2000           # Strom der über den Staberder eingespeist wird

# Definition des Staberders
xE = L/2             # x-Position des Erders in m
xE2 = L/2+150
LE = 18             # Länge des Erders in m
dE = 3             # Dicke des Erders in m
V = LE*dE**2          # Volumen des Erders in m³
qD = I/V            # Quellendichte in A/m³

ps = PoissonSolverMatrix2D(N=N, L=L)
X = ps.X
Y = ps.Y

# Leitfähigkeitsmatrix
ps.sigma = ps.sigma*sigma_soil                                
mask = (X>=xE) & (X<=xE+dE) & (Y>=L-LE-ds) & (Y<=L-ds)                    
ps.sigma[mask] = sigmaCu 
mask2 = (X>=xE2) & (X<=xE2+dE) & (Y>=L-LE-ds) & (Y<=L-ds) 
#ps.sigma[mask2] = sigmaCu 

#mask_surf = Y > L-ds                      # dünne isolierende Schicht an der Oberfläche, adiabatische Randbedingung
#ps.sigma[mask_surf] = 1e-8
ps.sigma[:, N-dspx:N] = 1e-8                # oben
ps.sigma[:dspx, :] = 1e-8             # links   
ps.sigma[N-dspx:, :] = 1e-8         # rechts


# Quellenmatrix
#ps.Q[int(xE/h)-1,:N-dspx-1] = qD
ps.Q[mask] = qD
#ps.Q[mask2] = qD/2


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
#fig, ax = plt.subplots()
#ps.drawImage(phi, fig, ax, 'Elektrostatisches Potential V')
#ps.drawContour(phi, ax, levels=400)
# Oberfläche
lx = np.linspace(0, L, N)
ly = np.ones(N)*(L-ds)
#ax.plot(lx, ly)

#fig, ax = plt.subplots()

phi_surf = phi[N-dspx, :]
phi_surf = phi_surf[dspx:-dspx]-phi_surf[2]
x = np.linspace(0, L, N-2*dspx)
#ax.plot(x, phi_surf,'.')  # Potentialverlauf an der Oberfläche

E_surf = Ey[N-dspx, :]
E_surf = E_surf[dspx:-dspx]
x = np.linspace(0, L, N-2*dspx)
#ax.plot(x, E_surf,'.')  # Feldstärkeverlauf an der Oberfläche
print(np.max(E_surf)-np.min(E_surf))
plt.show()
