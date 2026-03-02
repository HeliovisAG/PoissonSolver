import numpy as np
import matplotlib.pyplot as plt
from PoissonMatrixSolver2D import PoissonMatrixSolver2D

N = 200
L = 2.0
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

k_soil=1,       # Thermische Leitfähigkeit Boden (feucht ~1.0..1.5)
k_xlpe=0.4,      # Thermische Leitfähigkeit XLPE-Isolation
k_pvc=0.20,       # Thermische Leitfähigkeit Mantel (vereinfacht)
k_cu=385.0,       # Thermische Leitfähigkeit Kupfer
I = 618.0
A = 240e-6       # Querschnitt in m2
sigma20 = 5.8e7    # El. Leitfähigkeit von Cu bei 20°C in 1/Ohm m
alpha = 3.93e-4        # Temperaturkoeffizient von Cu
Topmax = 90

R20 = 1/sigma20/A  # Widerstand in Ohm/m bei 20°C
R_DC = R20*(1-alpha*(Topmax-20))
R_AC = R_DC * 1.03                         # Skin- und Proximity Effekt mit 3% angenommen
P = R_AC*I**2       # Leistung in W/m

sigma = np.ones((N, N))*k_soil
x0 = 1
y0 = 1.2
RR = np.sqrt((X-x0)**2 + (Y-y0)**2)
r_cond = np.sqrt(A/np.pi)   # Radius Leiter in m
r_ins = 0.034
r_pvc = 0.039
mask_cond = RR <= r_cond
mask_ins  = (RR>r_cond) & (RR<=r_ins)
mask_pvc  = (RR>r_ins) & (RR<=r_pvc)
mask_surf = Y > 1.97
#mask_surf = (X > 1.97) | (X < 0.03)

sigma[mask_surf] = 1e-8
sigma[mask_cond] = k_cu
sigma[mask_ins]  = k_xlpe
sigma[mask_pvc] = k_pvc

Q = np.zeros((N, N))
Q[mask_cond] = P/A  # W/m³

bc = {'left': 20.0, 'right': 20.0, 'bottom': 20.0, 'top': 20}
phi_init=20*np.ones((N, N))
ps = PoissonMatrixSolver2D(N=N, L=L, sigma=sigma, Q=Q, phi_init=phi_init, bc_values=bc)
phi = ps.solve()

# --- Visualisierung ---
fig, ax = plt.subplots()

#im = ax.imshow(phi, origin='lower', extent=[0, L, 0, L], cmap='inferno', aspect='auto')
#cb = fig.colorbar(im)
#cb.set_label('Temperatur [°C]')

cs = ax.contourf(np.linspace(0, L, N), np.linspace(0, L, N), phi, cmap="inferno", levels=40, linewidths=0.6)
#ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f', colors='white',)

ax.set_title("Temperaturfeld um das erdverlegte Kabel [°C]")
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect("equal")
plt.show()
