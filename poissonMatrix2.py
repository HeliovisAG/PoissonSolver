import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

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

            if is_left or is_right or is_bottom:
                rows.append(k); cols.append(k); data.append(1.0)
                b[k] = float(phi_dirichlet(xs[j], ys[i]))
                continue

            if is_top:
                # Neumann: qN(x) = n·(sigma grad phi) am oberen Rand
                # x-Richtung wie gewohnt:
                sx_e = 0.5 * (sigma[i, j] + sigma[i, j+1])
                sx_w = 0.5 * (sigma[i, j] + sigma[i, j-1])
                ae = sx_e / dx**2
                aw = sx_w / dx**2

                # y-Richtung: nur "nördlicher" Nachbar (i-1)
                sy_n = 0.5 * (sigma[i-1, j] + sigma[i, j])
                an = sy_n / dy**2

                # Diagonale (kein oberer Nachbar):
                ap = -(ae + aw + an)

                rows += [k, k, k, k]
                cols += [idx(i, j+1), idx(i, j-1), idx(i-1, j), k]
                data += [ae, aw, an, ap]

                qN = float(top_flux(xs[j]))  # physikalischer Neumann-Flux
                b[k] = -float(f[i, j]) - qN / dy
                continue

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
Lx, Ly = 1.0, 0.8

sigma = np.ones((ny, nx))
sigma[90:120, 120:150] = 500.0  # rechte Haelfte: hoehere Leitfaehigkeit

f = np.zeros((ny, nx))
cy, cx = ny//2, nx//2
f[cy-1:cy+2, cx-1:cx+2] = 10.0 / 9.0

def phi_D(x, y):  # Dirichlet unten/links/rechts
    return 0.0

def qN_top(x):    # Neumann-Flux oben: qN = 0 (isoliert)
    return 0.0

phi = solve_poisson_sigma_mixedBC(nx, ny, Lx, Ly, sigma, f, phi_D, qN_top)

# --- Visualisierung ---
extent = [0, Lx, 0, Ly]

plt.figure(figsize=(7, 5))
im = plt.imshow(phi, origin='lower', extent=extent, cmap='inferno', aspect='auto')
cb = plt.colorbar(im)
cb.set_label('Potential phi [V]')
plt.title('Poisson: Dirichlet (links/rechts/unten), Neumann-Flux oben (qN=0)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
cs = plt.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), phi,
                 colors='white', linewidths=0.6, levels=12, alpha=0.7)
plt.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
plt.tight_layout()
plt.show()

#plt.figure(figsize=(7, 5))
#im2 = plt.imshow(phi, origin='lower', extent=extent, cmap='viridis', aspect='equal')
#cb2 = plt.colorbar(im2)
#cb2.set_label('Potential phi [V]')
#plt.title('Heatmap (gleiches Seitenverhaeltnis) - Neumann oben')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.tight_layout()
#plt.show()