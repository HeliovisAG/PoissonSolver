import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

class PoissonSolverMatrix2D:
    def __init__(self, N, L):
        """
        N       Anzahl Punkte in x- und y-Richtung
        L       Ausdehnung in [m] in x- und y-Richtung
        """
        self.N = N
        self.L = L
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        self.sigma = np.ones((N, N))             # Leitfähigkeitsmatrix
        self.Q = np.zeros((N, N))                # Quellenmatrix

    def solve(self, bc):
        """
        Loest ∇·(σ∇φ) = -Q auf einem quadratischen Gitter mit Dirichlet-Randbedingungen.

        sigma    Leitfähigkeitsmatrix [N, N]
        Q        Quellenmatrix [N, N]
        bc:     Dictionary {'left', 'right', 'bottom', 'top'} mit den Randfunktionen (x,y).

        Return
        phi     Potentialmatrix [N, N]
        """
        Q = self.Q
        sigma = self.sigma

        L = self.L
        N = self.N
        h = L / (N - 1)
        N2 = N * N

        rows, cols, data = [], [], []
        b = np.zeros(N2)

        def idx(i, j):
            #return i * N + j
            return j * N + i

        xs = np.linspace(0, L, N)
        ys = np.linspace(0, L, N)

        for i in range(N):
            for j in range(N):
                k = idx(i, j)

                # Dirichlet-Randbedingungen
                is_left   = (i == 0)
                is_right  = (i == N - 1)
                is_bottom = (j == 0)
                is_top    = (j == N - 1)

                if is_left:
                    rows.append(k); cols.append(k); data.append(1.0)
                    fun = bc['left']
                    b[k] = float(fun(xs[j], ys[i]))
                    continue

                if is_right:
                    rows.append(k); cols.append(k); data.append(1.0)
                    fun = bc['right']
                    b[k] = float(fun(xs[j], ys[i]))
                    continue

                if is_bottom:
                    rows.append(k); cols.append(k); data.append(1.0)
                    fun = bc['bottom']
                    b[k] = float(fun(xs[j], ys[i]))
                    continue

                if is_top:
                    rows.append(k); cols.append(k); data.append(1.0)
                    fun = bc['top']
                    b[k] = float(fun(xs[j], ys[i]))
                    continue

                
                # Innere Punkte: gemittelte Leitfaehigkeit an Kanten
                sx_e = 0.5 * (sigma[i, j] + sigma[i, j+1])
                sx_w = 0.5 * (sigma[i, j] + sigma[i, j-1])
                sy_n = 0.5 * (sigma[i-1, j] + sigma[i, j])
                sy_s = 0.5 * (sigma[i+1, j] + sigma[i, j])

                ae = sx_e / h**2
                aw = sx_w / h**2
                an = sy_n / h**2
                aS = sy_s / h**2
                ap = -(ae + aw + an + aS)

                rows += [k, k, k, k, k]
                cols += [idx(i, j+1), idx(i, j-1), idx(i-1, j), idx(i+1, j), k]
                data += [ae, aw, an, aS, ap]

                b[k] = -Q[i, j]

        A = sp.csr_matrix((data, (rows, cols)), shape=(N2, N2))
        self.phi = spla.spsolve(A, b)
        self.phi = self.phi.reshape((N, N))

    def fieldvektor(self):
        dPhidy, dPhidx = np.gradient(self.phi, self.L/self.N, self.L/self.N)

        self.Ex = -dPhidx
        self.Ey = -dPhidy
        self.E = np.sqrt(self.Ex**2 + self.Ey**2)

    def drawImage(self, field, fig, ax, titel:str):
        im = ax.imshow(field, origin='lower', extent=[0, self.L, 0, self.L], cmap='inferno', aspect='auto')
        cb = fig.colorbar(im)
        cb.set_label(titel)
        ax.set_xlabel('y [m]')
        ax.set_ylabel('x [m]')
        ax.set_aspect("equal")

    def drawContour(self, field, ax):
        #isolevels = [20, 40, 60, 70]
        cs = ax.contourf(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N), field, cmap="inferno", levels=40)
        #ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
        ax.set_xlabel('y [m]')
        ax.set_ylabel('x [m]')
        ax.set_aspect("equal")
    

    