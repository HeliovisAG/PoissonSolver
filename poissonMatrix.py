
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# --- Solver-Funktion ---
def solve_poisson_inhomogeneous_sigma(nx, ny, Lx, Ly, sigma, f, phi_bc):
    """
    Loest -∇·(σ∇φ) = f auf einem regulaeren Gitter mit Dirichlet-Randbedingungen.
    sigma, f: Arrays [ny, nx]; phi_bc: Funktion (x,y)-> φ am Rand.
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

            # Dirichlet-Rand
            if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
                rows.append(k); cols.append(k); data.append(1.0)
                b[k] = float(phi_bc(xs[j], ys[i]))
                continue

            # Innere Punkte: gemittelte Leitfaehigkeit an Kanten
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

# --- Beispielproblem ---
nx, ny = 200, 200
Lx, Ly = 1.0, 0.8

sigma = np.ones((ny, nx))
sigma[90:120, 120:150] = 500.0  # rechte Haelfte: hoehere Leitfaehigkeit

f = np.zeros((ny, nx))
cy, cx = ny//2, nx//2
f[cy-1:cy+20, cx-1:cx+20] = 10.0 / 9.0  # kleine kompakte Quelle

phi_bc = lambda x, y: 0.0

phi = solve_poisson_inhomogeneous_sigma(nx, ny, Lx, Ly, sigma, f, phi_bc)

# --- Heatmap 1: mit Konturen ---
extent = [0, Lx, 0, Ly]
plt.figure(figsize=(7, 5))
im = plt.imshow(phi, origin='lower', extent=extent, cmap='inferno', aspect='auto')
cb = plt.colorbar(im)
cb.set_label('Potential φ [V]')
plt.title('Poisson-Loesung: φ mit inhomogener Leitfaehigkeit')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

cs = plt.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), phi,
                 colors='white', linewidths=0.6, levels=12, alpha=0.7)
plt.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
plt.tight_layout()
plt.show()

# --- Heatmap 2: ohne Konturen, gleiches Seitenverhältnis ---
#plt.figure(figsize=(7, 5))
#im2 = plt.imshow(phi, origin='lower', extent=extent, cmap='viridis', aspect='equal')
#cb2 = plt.colorbar(im2)
#cb2.set_label('Potential φ [V]')
#plt.title('Heatmap des Potentials (gleiches Seitenverhaeltnis)')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.tight_layout()
#plt.show()
