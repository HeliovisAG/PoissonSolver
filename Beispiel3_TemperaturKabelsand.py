import numpy as np
import matplotlib.pyplot as plt
from poissonSolverMatrix2D import PoissonSolverMatrix2D

def addXLPECable(ps:PoissonSolverMatrix2D, x:float, y:float, A:float, d:float, r_ins:float, r_pvc:float, I:float):
    r_cond=np.sqrt(A/np.pi)                     # Radius Leiter in m
    sigma20 = 5.8e7    # El. Leitfähigkeit von Cu bei 20°C in 1/Ohm m
    alpha = 3.93e-4        # Temperaturkoeffizient von Cu
    Topmax = 90
    R20 = 1/sigma20/A  # Widerstand in Ohm/m bei 20°C
    R_DC = R20*(1+alpha*(Topmax-20))
    R_AC = R_DC * 1.03                         # Skin- und Proximity Effekt mit 3% angenommen
    P = R_AC*I**2       # Leistung in W/m

    X = ps.X
    Y = ps.Y
    RR = np.sqrt((X-x)**2 + (Y-y)**2)
    mask_cond = RR <= r_cond
    mask_ins  = (RR>r_cond) & (RR<=r_ins)
    mask_pvc  = (RR>r_ins) & (RR<=r_pvc)

    # Leitfähigkeitsmatrix
    ps.sigma[mask_cond] = 380.0                 # Wärmeleitfähigkeit Kupfer W/mK
    ps.sigma[mask_ins]  = 0.4                  # Wärmeleitfähigkeit XLPE-Isolation
    ps.sigma[mask_pvc] = 0.20                  # Wärmeleitfähigkeit Mantel PVC (vereinfacht)
    
    # Quellenmatrix
    ps.Q[mask_cond] = P/A                            # Quellendichte in A/m3


N = 2000
L = 8.0
ps = PoissonSolverMatrix2D(N=N, L=L)
X = ps.X
Y = ps.Y

# Leitfähigkeitsmatrix
k_soil = 1/2.5                              # Wärmeleitfähigkeit Boden W/mK
ps.sigma = ps.sigma*k_soil                                

# Leiter
I = 411.0
A = 300e-6                                  # Querschnitt in m2
r_cond=np.sqrt(A/np.pi)                     # Radius Leiter in m
r_ins = 0.034                               # Radius Leiter + Isolator
r_pvc = 0.036                               # Radius Leiter + Isolator + äußere PVC-Schicht
d = 0.8                                     # Verlege-Tiefe des Kabels in m
x1 = L/2
y1 = L-d
x2 = L/2+2*r_pvc
y2 = L-d
x3 = L/2+r_pvc
y3 = L-d+np.sqrt(3)*r_pvc
sd = 0.2                                    # Sanddicke
y_sd1 = L-d+r_pvc + sd/2                    # y-Koordinate Begin Sandschicht
y_sd2 = L-d+r_pvc - sd/2                    # y-Koordinate Ende Sandschicht

addXLPECable(ps, x1, y1, A, d, r_ins, r_pvc, I)
addXLPECable(ps, x2, y2, A, d, r_ins, r_pvc, I)
addXLPECable(ps, x3, y3, A, d, r_ins, r_pvc, I)

ps.sigma[Y > L*0.99] = 1e-8     # dünne isolierende Schicht oben, adiabatische Randbedingung 
#ps.sigma[X < L*0.01] = 1e-8    # dünne isolierende Schicht links, adiabatische Randbedingung 
#ps.sigma[X > L*0.99] = 1e-8    # dünne isolierende Schicht rechts, adiabatische Randbedingung 

mask_sand = (Y < y_sd1) & (Y > y_sd2) 
#ps.sigma[mask_sand] = 1     # Wärmeleitfähigkeit Kabelsand 


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
isolevels = np.linspace(20,100,61)
ps.drawContour(ps.phi, fig, ax, 'Temperatur °C', levels=isolevels)
alpha = np.linspace(0,2*np.pi,100)
ss = np.sin(alpha)
cc = np.cos(alpha)
xx = r_cond*ss + x1
yy = r_cond*cc + y1
ax.plot(xx, yy,  'black')
xx = r_cond*ss + x2
yy = r_cond*cc + y2
ax.plot(xx, yy, 'black')
xx = r_cond*ss + x3
yy = r_cond*cc + y3
ax.plot(xx, yy, 'black')

xx = r_ins*ss + x1
yy = r_ins*cc + y1
ax.plot(xx, yy, 'black')
xx = r_ins*ss + x2
yy = r_ins*cc + y2
ax.plot(xx, yy, 'black')
xx = r_ins*ss + x3
yy = r_ins*cc + y3
ax.plot(xx, yy, 'black')

xx = r_pvc*ss + x1
yy = r_pvc*cc + y1
ax.plot(xx, yy, 'black')
xx = r_pvc*ss + x2
yy = r_pvc*cc + y2
ax.plot(xx, yy, 'black')
xx = r_pvc*ss + x3
yy = r_pvc*cc + y3
ax.plot(xx, yy, 'black')

# Sandlayer einzeichnen
#ax.plot([0, L], [y_sd1, y_sd1], 'black', linewidth=0.3)
#ax.plot([0, L], [y_sd2, y_sd2], 'black', linewidth=0.3)

ax.set_xlim(L/2-1, L/2+1)
ax.set_ylim(L-2, L)

h = L/N             # örtliche Auflösung
y_px = int((L-d)/h)

# Kabel-Einbautiefe einzeichnen
ax.plot([0, L], [y_px*h, y_px*h], linewidth=0.3)

x1_px = int((L/2-1)/h)
x2_px = int((L/2+1)/h)
T = ps.phi[y_px, x1_px:x2_px]
ax2 = fig.add_axes([0.4, 0.2, 0.25, 0.25])
x = X[x1_px:x2_px, y_px]
ax2.plot(x, T)
ax2.set_ylabel("Temperatur [°C]")
ax2.set_ylim([40,100])
ax2.grid()

plt.show()
