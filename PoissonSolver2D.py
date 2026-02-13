import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PoissonSolver2D:
    """
    Löser für stationäre 2D-Poisson-Gleichung:

        div(k * grad T) = -Q

    Unterstützt:
    - variable Wärmeleitfähigkeit k(x,y)
    - Dirichlet-Ränder (fixe Temperatur)
    - Neumann-Ränder (adiabatisch: dT/dn = 0)
    """

    def __init__(self, N, L,
                 k_field,
                 Q_field,
                 T_init,
                 bc_type,
                 bc_values):
        """
        Parameter:
            bc_type  : dict mit 'left','right','bottom','top'
                       Wert = "dirichlet" oder "neumann"

            bc_values: dict mit randbezogenen Werten:
                       - bei Dirichlet: Temperaturwert
                       - bei Neumann:   nichts (None)
        """

        self.N = N, N
        self.dx = L/N

        self.k = k_field
        self.Q = Q_field
        self.T = T_init.copy()

        self.bc_type = bc_type
        self.bc_values = bc_values

        self.dx2 = self.dy2 = self.dx*self.dx
        
        # Anfangsrandbedingungen
        self.apply_boundary()

    # -------------------------------------------------------------
    # Randbedingungen
    # -------------------------------------------------------------

    def apply_boundary(self):

        # ---- LINKS ----
        if self.bc_type["left"] == "dirichlet":
            self.T[0, :] = self.bc_values["left"]
        else:  # Neumann → adiabatisch
            self.T[0, :] = self.T[1, :]

        # ---- RECHTS ----
        if self.bc_type["right"] == "dirichlet":
            self.T[-1, :] = self.bc_values["right"]
        else:
            self.T[-1, :] = self.T[-2, :]

        # ---- UNTEN ----
        if self.bc_type["bottom"] == "dirichlet":
            self.T[:, 0] = self.bc_values["bottom"]
        else:
            self.T[:, 0] = self.T[:, 1]

        # ---- OBEN ----
        if self.bc_type["top"] == "dirichlet":
            self.T[:, -1] = self.bc_values["top"]
        else:
            self.T[:, -1] = self.T[:, -2]

    # -------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------

    def solve(self, max_iter=20000, tol=1e-6, omega=1.7):

        for it in range(max_iter):

            T_old = self.T.copy()

            # Wärmeleitfähigkeit (harmonische Mittelwerte)
            k = self.k
            kE = 2*k[1:-1,1:-1] * k[2:,1:-1] / (k[1:-1,1:-1] + k[2:,1:-1])
            kW = 2*k[1:-1,1:-1] * k[0:-2,1:-1] / (k[1:-1,1:-1] + k[0:-2,1:-1])
            kN = 2*k[1:-1,1:-1] * k[1:-1,2:]   / (k[1:-1,1:-1] + k[1:-1,2:])
            kS = 2*k[1:-1,1:-1] * k[1:-1,0:-2] / (k[1:-1,1:-1] + k[1:-1,0:-2])

            numerator = (
                kE * self.T[2:, 1:-1] +
                kW * self.T[0:-2, 1:-1] +
                kN * self.T[1:-1, 2:] +
                kS * self.T[1:-1, 0:-2]
                - self.Q[1:-1,1:-1] * self.dx2
            )

            denominator = (kE + kW + kN + kS)

            T_new = numerator / denominator

            # SOR
            #self.T[1:-1,1:-1] = (
            #    omega * T_new + (1 - omega) * self.T[1:-1,1:-1]
            #)
            self.T[1:-1,1:-1] = T_new
            # Ränder anwenden
            self.apply_boundary()

            # Fehler
            err = np.max(np.abs(self.T - T_old))
            print(err, it, np.max(self.T))


            # store frame
            #if it % 5 == 0:
            #    im = ax.imshow(self.T, origin='lower', extent=[0,L,0,L],
            #           cmap='inferno', animated=True)
            #    ax.set_title(f"Iteration {it}")
            #    frames.append([im])


            if err < tol:
                return {
                    "converged": True,
                    "iterations": it,
                    "error": err,
                    "T": self.T
                }

        return {
            "converged": False,
            "iterations": max_iter,
            "error": err,
            "T": self.T
        }

