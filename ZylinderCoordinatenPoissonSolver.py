
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# Parameters
# -----------------------------------------------
rho = 100.0
sigma = 1.0 / rho
I = 1.0
L = 2.0
r_e = 0.05

dr = 0.005
r_max = 5.0
r = np.arange(r_e, r_max, dr)

# Distributed source inside electrode
J0 = I / (np.pi * r_e**2 * L)
J = np.where(r <= r_e, J0, 0.0)

# Initial potential
V = np.zeros_like(r)

# Over-relaxation parameter
omega = 2

# -----------------------------------------------
# Solve Poisson: (1/r d/dr (r dV/dr)) = -J/sigma
# -----------------------------------------------
for it in range(20000):
    V_old = V.copy()
    
    for i in range(1, len(r)-1):
        rp = r[i] + dr/2
        rm = r[i] - dr/2
        
        A = rp / (r[i] * dr**2)
        B = rm / (r[i] * dr**2)
        C = -(A + B)
        
        rhs = -J[i] / sigma
        
        V_new = (A*V[i+1] + B*V[i-1] - rhs) / (-C)
        
        V[i] = (1-omega)*V[i] + omega*V_new

    # Convergence check
    if np.max(np.abs(V - V_old)) < 1e-9:
        print("Converged in iteration", it)
        break

# -----------------------------------------------
# Analytical reference
# -----------------------------------------------
def V_analytical(r):
    return (rho * I)/(2*np.pi*L) * np.log(r_e / r)

# -----------------------------------------------
# Plot
# -----------------------------------------------
plt.plot(r, V, label="Numerical")
plt.plot(r, V_analytical(r), "--", label="Analytical")
plt.xlabel("r (m)")
plt.ylabel("V (V)")
plt.grid()
plt.legend()
plt.show()
