import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from PoissonSolver2D import PoissonSolver2D_MatrixInverse

# --- Solver mit Neumann am oberen Rand (y = Ly) ---
def solve_poisson_sigma_mixedBC(nx, ny, Lx, Ly, sigma, f, phi_dirichlet, top_flux):
    """
    Loest -div(sigma * grad(phi)) = f auf [0,Lx]x[0,Ly]
    Dirichlet: links, rechts, unten  -> phi_dirichlet(x,y)
    Neumann (Flux): oben y=Ly        -> n·(sigma grad phi) = qN(x) = top_flux(x)

    Diskretisierung: regulaeres FD-Gitter, Kanten-Leitfaehigkeit mitteln.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    N = nx * ny

    rows, cols, data = [], [], []
    b = np.zeros(N)

    def idx(i, j):
        return i * nx + j

    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)

    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)

            # Ränder: links/rechts/unten -> Dirichlet
            is_left   = (j == 0)
            is_right  = (j == nx - 1)
            is_bottom = (i == 0)
            is_top    = (i == ny - 1)

            if is_left or is_right or is_bottom or is_top:
                rows.append(k); cols.append(k); data.append(1.0)
                b[k] = float(phi_dirichlet(xs[j], ys[i]))
                continue

            #if is_top:
                # Neumann: qN(x) = n·(sigma grad phi) am oberen Rand
                # x-Richtung wie gewohnt:
            #    sx_e = 0.5 * (sigma[i, j] + sigma[i, j+1])
            #    sx_w = 0.5 * (sigma[i, j] + sigma[i, j-1])
            #    ae = sx_e / dx**2
            #    aw = sx_w / dx**2

                # y-Richtung: nur "nördlicher" Nachbar (i-1)
            #    sy_n = 0.5 * (sigma[i-1, j] + sigma[i, j])
            #    an = sy_n / dy**2

                # Diagonale (kein oberer Nachbar):
            #    ap = -(ae + aw + an)

            #    rows += [k, k, k, k]
            #    cols += [idx(i, j+1), idx(i, j-1), idx(i-1, j), k]
            #    data += [ae, aw, an, ap]

            #    qN = float(top_flux(xs[j]))  # physikalischer Neumann-Flux
            #    b[k] = -float(f[i, j]) - qN / dy
            #    continue

            # Innere Punkte
            sx_e = 0.5 * (sigma[i, j] + sigma[i, j+1])
            sx_w = 0.5 * (sigma[i, j] + sigma[i, j-1])
            sy_n = 0.5 * (sigma[i-1, j] + sigma[i, j])
            sy_s = 0.5 * (sigma[i+1, j] + sigma[i, j])

            ae = sx_e / dx**2
            aw = sx_w / dx**2
            an = sy_n / dy**2
            aS = sy_s / dy**2
            ap = -(ae + aw + an + aS)

            rows += [k, k, k, k, k]
            cols += [idx(i, j+1), idx(i, j-1), idx(i-1, j), idx(i+1, j), k]
            data += [ae, aw, an, aS, ap]

            b[k] = -float(f[i, j])

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    phi = spla.spsolve(A, b)
    return phi.reshape((ny, nx))

# --- Beispielproblem (oben: homogener Neumann-Flux qN=0) ---
nx, ny = 200, 200
Lx, Ly = 2.0, 2.0
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, nx)
X, Y = np.meshgrid(x, y)

k_soil=1,       # Thermische Leitfähigkeit Boden (feucht ~1.0..1.5)
k_xlpe=0.4,      # Thermische Leitfähigkeit XLPE-Isolation
k_pvc=0.20,       # Thermische Leitfähigkeit Mantel (vereinfacht)
k_cu=385.0,       # Thermische Leitfähigkeit Kupfer
I = 618.0
A = 240e-6       # Querschnitt in m2
sigma20 = 5.8e7    # El. Leitfähigkeit von Cu bei 20°C in 1/Ohm m
alpha = 3.93e-4        # Temperaturkoeffizient von Cu
Topmax = 70

R20 = 1/sigma20/A  # Widerstand in Ohm/m bei 20°C
R_DC = R20*(1-alpha*(Topmax-20))
R_AC = R_DC * 1.03                         # Skin- und Proximity Effekt mit 3% angenommen
P = R_AC*I**2       # Leistung in W/m

sigma = np.ones((ny, nx))*k_soil
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

sigma[mask_surf] = 1e-8
sigma[mask_cond] = k_cu
sigma[mask_ins]  = k_xlpe
sigma[mask_pvc] = k_pvc

f = np.zeros((ny, nx))
f[mask_cond] = P/A  # W/m³

def phi_D(x, y):  # Dirichlet unten/links/rechts
    return 20.0

def qN_top(x):    # Neumann-Flux oben: qN = 0 (isoliert)
    return 0.0

#phi = solve_poisson_sigma_mixedBC(nx, ny, Lx, Ly, sigma, f, phi_D, qN_top)

bc = {'left': 20.0, 'right': 20.0, 'bottom': 20.0, 'top': 20}
ps = PoissonSolver2D_MatrixInverse(N=nx, L=Lx, sigma=sigma, Q=f, phi_init=20*np.ones((ny, nx)), bc_values=bc)
phi = ps.solve()

# --- Visualisierung ---
extent = [0, Lx, 0, Ly]

fig, ax = plt.subplots()

#im = ax.imshow(phi, origin='lower', extent=extent, cmap='inferno', aspect='auto')
#cb = fig.colorbar(im)
#cb.set_label('Temperatur [°C]')

cs = ax.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), phi,
                 cmap="inferno", linewidths=0.6, levels=40)
#ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
ax.set_title("Temperaturfeld um das erdverlegte Kabel [°C]")
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect("equal")
plt.show()
