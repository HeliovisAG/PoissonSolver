
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PoissonSolver2D import *


N = 200
L = 2             # Ausdehnung in m
k_soil=1,       # Thermische Leitfähigkeit Boden (feucht ~1.0..1.5)
k_xlpe=0.4,      # Thermische Leitfähigkeit XLPE-Isolation
k_pvc=0.20,       # Thermische Leitfähigkeit Mantel (vereinfacht)
k_cu=385.0,       # Thermische Leitfähigkeit Kupfer

sigma20 = 5.8e7    # El. Leitfähigkeit von Cu bei 20°C in 1/Ohm m
alpha = 3.93e-4        # Temperaturkoeffizient von Cu
Topmax = 90

# Beispiel: Kreisförmige XLPE-Region im Erdreich
k = np.ones((N, N))*k_soil  # Erdreich
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing="ij")

I = 774.0
A = 240e-6       # Querschnitt in m2
R20 = 1/sigma20/A  # Widerstand in Ohm/m bei 20°C
R_DC = R20*(1-alpha*(Topmax-20))
R_AC = R_DC * 1.03                         # Skin- und Proximity Effekt mit 3% angenommen
P = R_AC*I**2       # Leistung in W/m
print(P)

x0 = 1
y0 = 1.3
RR = np.sqrt((X-x0)**2 + (Y-y0)**2)
r_cond = np.sqrt(A/np.pi)   # Radius Leiter in m
r_ins = 0.034
r_pvc = 0.039
mask_cond = RR <= r_cond
mask_ins  = (RR>r_cond) & (RR<=r_ins)
mask_pvc  = (RR>r_ins) & (RR<=r_pvc)
mask_surf = Y > 1.9

k[mask_surf] = 1e-4
k[mask_cond] = k_cu
k[mask_ins]  = k_xlpe
k[mask_pvc] = k_pvc


Q = np.zeros((N, N))
A = np.pi*r_cond**2
Q[mask_cond] = -P/A  # W/m³

T0 = np.ones((N, N)) * 20.0

bc_type = {
    "left": "dirichlet",
    "right": "dirichlet",
    "bottom": "dirichlet",
    "top": "dirichlet"
}

bc_values = {
    "left": 20.0,
    "right": 20.0,
    "bottom": 20.0,
    "top": 40
}

solver = PoissonSolver2D(
    N, L,
    k_field=k,
    Q_field=Q,
    phi_init=T0,
    bc_type=bc_type,
    bc_values=bc_values
)

frames = []
fig, ax = plt.subplots()

result = solver.solve(omega=1, max_iter=300000, tol=1e-5)
print(result["converged"], result["iterations"])

T = result["phi"]

#ani = animation.ArtistAnimation(fig, frames, interval=15, blit=True)
#ani.save('T_kabel_in_erde.gif', writer='pillow')
#plt.close()

cf = ax.contourf(X, Y, T, levels=40, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperatur [°C]")
alpha = np.linspace(0,2*np.pi,100)
ss = np.sin(alpha)
cc = np.cos(alpha)
xx = r_cond*ss + x0
yy = r_cond*cc + y0
ax.plot(xx, yy)
xx = r_ins*ss + x0
yy = r_ins*cc + y0
ax.plot(xx, yy)
xx = r_pvc*ss + x0
yy = r_pvc*cc + y0
ax.plot(xx, yy)

ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Temperaturfeld um das erdverlegte Kabel [°C]")



#plt.figure(figsize=(6,5))
#plt.imshow(result["T"].T, origin="lower", cmap="inferno")
#plt.colorbar(label="Temperatur [°C]")
#plt.title("Stationäres Temperaturfeld")
plt.show()


