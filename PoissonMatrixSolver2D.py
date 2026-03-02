import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class PoissonMatrixSolver2D:
    """
    Löser für stationäre 2D-Poisson-Gleichung:

        div(sigma * grad phi) = -Q
    Unterstützt:
    - variable Leitfähigkeit sigma(x,y) 
    - Dirichlet-Ränder (fixe Temperatur)
    """
    
    def __init__(self, N, L, sigma, Q, phi_init, bc_values):
        """
        Parameter:
            N               Anzahl der Gitterpunkte pro Richtung        
            L               Länge der Seite des quadratischen Gebiets in m
            sigma           Matrix mit der ortsabhängigen Leitfähigkeit     
            Q               Matrix mit dem Quellenfeld
            phi_init        Matrix mit den Startwerten für das Potentialfeld    
            bc_values       dict mit randbezogenen Werten für Dirichlet-Ränder:
                            - 'left', 'right', 'bottom', 'top' : Temperaturwert
        """
        self.N = N
        self.L = L
        self.h = L/(N-1)

        self.sigma = sigma
        self.Q = Q
        self.phi = phi_init.copy()

        self.bc_values = bc_values

    def solve(self):
        h = self.h
        N = self.N
        L = self.L
        sigma = self.sigma
        Q = self.Q
        N2 = self.N**2

        rows, cols, data = [], [], []
        b = np.zeros(N2)

        def idx(i, j):
            return i * N + j

        for i in range(N):
            for j in range(N):
                k = idx(i, j)

                # Ränder: links/rechts/unten -> Dirichlet
                is_left   = (j == 0)
                is_right  = (j == self.N - 1)
                is_bottom = (i == 0)
                is_top    = (i == self.N - 1)

                rows.append(k); cols.append(k); data.append(1.0)
                if is_left:
                    b[k] =  float(self.bc_values['left']) 
                if is_right:
                    b[k] =  float(self.bc_values['right']) 
                if is_bottom:
                    b[k] =  float(self.bc_values['bottom']) 
                if is_top:
                    b[k] =  float(self.bc_values['top']) 
                    
                # Innere Punkte
                if not (is_left or is_right or is_bottom or is_top):
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

                    b[k] = -float(Q[i, j])

        A = sp.csr_matrix((data, (rows, cols)), shape=(N2, N2))
        phi = spla.spsolve(A, b)
        return phi.reshape((N, N))

