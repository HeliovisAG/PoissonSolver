import numpy as np

class PoissonSolver2D:
    """
    Löser für stationäre 2D-Poisson-Gleichung:

        div(k * grad phi) = -Q

    Unterstützt:
    - variable Leitfähigkeit k(x,y)
    - Dirichlet-Ränder (fixe Temperatur)
    - Neumann-Ränder (adiabatisch: dT/dn = 0)
    """

    def __init__(self, N, L,
                 k_field,
                 Q_field,
                 phi_init,
                 bc_type,
                 bc_values):
        """
        Parameter:
            N               Anzahl der Gitterpunkte pro Richtung
            L               Länge der Seite des quadratischen Gebiets in m
            k               Matrix mit der ortsabhängigen Leitfähigkeit
            Q               Matrix mit dem Quellenfeld
            phi_init        Matrix mit den Startwerten für das Potentialfeld

            bc_type  :      dict mit 'left','right','bottom','top'
                            Wert = "dirichlet" oder "neumann"

            bc_values: dict mit randbezogenen Werten:
                       - bei Dirichlet: Temperaturwert
                       - bei Neumann:   nichts (None)
        """

        self.N = N
        self.L = L
        self.h = L/N

        self.k = k_field
        self.Q = Q_field
        self.phi = phi_init.copy()

        self.bc_type = bc_type
        self.bc_values = bc_values

        self.h2 = self.h**2
        
        # Anfangsrandbedingungen
        self.apply_boundary()

    # -------------------------------------------------------------
    # Randbedingungen
    # -------------------------------------------------------------

    def apply_boundary(self):

        # ---- LINKS ----
        if self.bc_type["left"] == "dirichlet":
            self.phi[0, :] = self.bc_values["left"]
        else:  # Neumann → adiabatisch
            self.phi[0, :] = self.phi[1, :]

        # ---- RECHTS ----
        if self.bc_type["right"] == "dirichlet":
            self.phi[-1, :] = self.bc_values["right"]
        else:
            self.phi[-1, :] = self.phi[-2, :]

        # ---- UNTEN ----
        if self.bc_type["bottom"] == "dirichlet":
            self.phi[:, 0] = self.bc_values["bottom"]
        else:
            self.phi[:, 0] = self.phi[:, 1]

        # ---- OBEN ----
        if self.bc_type["top"] == "dirichlet":
            self.phi[:, -1] = self.bc_values["top"]
        else:
            self.phi[:, -1] = self.phi[:, -2]

    # -------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------

    def solve(self, max_iter=20000, tol=1e-6, omega=1.7):

        for it in range(max_iter):

            phi_old = self.phi.copy()

            # Wärmeleitfähigkeit (harmonische Mittelwerte)
            k = self.k
            kE = 2*k[1:-1,1:-1] * k[2:,1:-1] / (k[1:-1,1:-1] + k[2:,1:-1])
            kW = 2*k[1:-1,1:-1] * k[0:-2,1:-1] / (k[1:-1,1:-1] + k[0:-2,1:-1])
            kN = 2*k[1:-1,1:-1] * k[1:-1,2:]   / (k[1:-1,1:-1] + k[1:-1,2:])
            kS = 2*k[1:-1,1:-1] * k[1:-1,0:-2] / (k[1:-1,1:-1] + k[1:-1,0:-2])

            numerator = (
                kE * self.phi[2:, 1:-1] +
                kW * self.phi[0:-2, 1:-1] +
                kN * self.phi[1:-1, 2:] +
                kS * self.phi[1:-1, 0:-2]
                - self.Q[1:-1,1:-1] * self.h2
            )

            denominator = (kE + kW + kN + kS)

            phi_new = numerator / denominator

            # SOR
            self.phi[1:-1,1:-1] = (
                omega * phi_new + (1 - omega) * self.phi[1:-1,1:-1]
            )
            #self.phi[1:-1,1:-1] = phi_new
            # Ränder anwenden
            self.apply_boundary()

            # Fehler
            err = np.max(np.abs(self.phi - phi_old))
            print(err, it, np.max(self.phi))

           
            # store frame
            #if it % 5 == 0:
            #    im = ax.imshow(self.T, origin='lower', extent=[0,self.L,0,self.L],
            #           cmap='inferno', animated=True)
            #    ax.set_title(f"Iteration {it}")
            #    frames.append([im])


            if err < tol:
                return {
                    "converged": True,
                    "iterations": it,
                    "error": err,
                    "phi": self.phi
                }

        return {
            "converged": False,
            "iterations": max_iter,
            "error": err,
            "phi": self.phi
        }

