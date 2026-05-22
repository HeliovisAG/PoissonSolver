import numpy as np
import matplotlib.pyplot as plt
from poissonSolverMatrix2D import PoissonSolverMatrix2D

N = 400
L = 20            # in m
h = L/N           # räumliche Auflösung
dspx = 2          # Dicke der nichtleitenden Oberfläche in pixel
ds = dspx*h         # Dicke in m

sigma_soil=0.01    # Elektrische Leitfähigkeit Boden (lehmig) in S/m
sigmaCu = 5.8e7     # Elektrische Leitfähigkeit von Cu bei 20°C in 1/Ohm m
I = 20000           # Strom der über den Kugelerder eingespeist wird in A

# Definition des Kugelerders
xE = L/2             # x-Position des Erders in m
yE = L-3               # y-Position des Erders in m   
rE = 0.05
#V = 4/3*np.pi*rE**3          # Volumen des Erders in m³
V = np.pi*rE**2          # Querschnittsfläche des Zylindererders in m3 (Länge=1m)
qD = I/V            # Quellendichte in A/m³

ps = PoissonSolverMatrix2D(N=N, L=L)
X = ps.X
Y = ps.Y

# Leitfähigkeitsmatrix
ps.sigma[:,:] = sigma_soil
RR = np.sqrt((X-xE)**2 + (Y-yE)**2)                                
mask1 = (RR < rE) & (Y < L)                     
ps.sigma[mask1] = sigmaCu 

#RR = np.sqrt((X-L/4)**2 + (Y-yE)**2)                                
#mask2 = (RR < rE) & (Y < L)                     
#ps.sigma[mask2] = sigmaCu 

# dünne isolierende Schicht, adiabatische Randbedingung
ps.sigma[:, N-dspx:N] = 1e-8                # oben
ps.sigma[:dspx, :] = 1e-8             # links   
ps.sigma[N-dspx:, :] = 1e-8         # rechts


# Quellenmatrix
ps.Q[mask1] = qD
#ps.Q[mask2] = qD

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
ps.drawContour(phi, fig, ax, 'Elektrostatisches Potential V', levels=40)

fig, ax = plt.subplots()

phi_surf = phi[N-2*dspx, :]
phi_surf = phi_surf[dspx:-dspx]-phi_surf[2]
x = np.linspace(0, L, N-2*dspx)
ax.plot(x, phi_surf)  # Potentialverlauf an der Oberfläche

# Analytische Funktion für Potentialverlauf an der Oberfläche
r=np.linspace(1,L/2,100)         # horizontaler Abstand an der Oberfläche vom Leiter
h=3         # Einbautiefe
print(I/2/np.pi/sigma_soil*np.log(2*h/rE))
V = I/2/np.pi/sigma_soil*np.log(np.sqrt(r**2+4*h**2)/(r+1e-16))
ax.plot(r+xE,V)


#E_surf = Ey[N-2*dspx, :]
#E_surf = E_surf[dspx:-dspx]
#x = np.linspace(0, L, N-2*dspx)
#ax.plot(x, E_surf,'.')  # Feldstärkeverlauf an der Oberfläche
#print(np.max(E_surf)-np.min(E_surf))
plt.show()
