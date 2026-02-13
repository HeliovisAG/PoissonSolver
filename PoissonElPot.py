import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Grid
Nx, Ny = 181, 181
Lx, Ly = 0.4, 0.4
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)
hx = x[1]-x[0]
hy = y[1]-y[0]

# Radii (conductor + insulation)
r_cu  = 0.005
r_iso = 0.010

# Phase center positions (triangular layout)
L1x, L1y = -0.04,  0.00
L2x, L2y =  0.02, -0.035
L3x, L3y =  0.02,  0.035

R1 = np.sqrt((X-L1x)**2 + (Y-L1y)**2)
R2 = np.sqrt((X-L2x)**2 + (Y-L2y)**2)
R3 = np.sqrt((X-L3x)**2 + (Y-L3y)**2)

# Conductivity map
sigma = np.full_like(X, 0.05)      # soil
sigma[R1 < r_iso] = 1e-15
sigma[R2 < r_iso] = 1e-15
sigma[R3 < r_iso] = 1e-15
sigma[R1 < r_cu]  = 5.8e7
sigma[R2 < r_cu]  = 5.8e7
sigma[R3 < r_cu]  = 5.8e7

# Potential
phi = np.zeros_like(X)
phi[R1 < r_cu] =  1.0
phi[R2 < r_cu] = -0.5
phi[R3 < r_cu] = -0.5

# Dirichlet mask
mask_dir = (R1 < r_cu) | (R2 < r_cu) | (R3 < r_cu)

# Animation
fig, ax = plt.subplots(figsize=(5,5))
frames = []

# Gauss–Seidel iterations
for step in range(220):
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            if mask_dir[j,i]:
                continue
            sxp = (sigma[j,i] + sigma[j,i+1]) / 2
            sxm = (sigma[j,i] + sigma[j,i-1]) / 2
            syp = (sigma[j,i] + sigma[j+1,i]) / 2
            sym = (sigma[j,i] + sigma[j-1,i]) / 2
            A = sxp + sxm + syp + sym
            phi[j,i] = (sxp*phi[j,i+1] + sxm*phi[j,i-1] +
                        syp*phi[j+1,i] + sym*phi[j-1,i]) / A

    if step % 10 == 0:
        ax.clear()
        ax.set_title(f"Potential Iteration {step}")
        im = ax.imshow(phi, cmap="plasma", origin="lower",
                       extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
        frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=150, blit=False)
ani.save("3phase_potential.gif", writer="pillow")
plt.close()

# Compute E and J
Ex = np.zeros_like(phi)
Ey = np.zeros_like(phi)
Ex[:,1:-1] = -(phi[:,2:] - phi[:,:-2])/(2*hx)
Ey[1:-1,:] = -(phi[2:,:] - phi[:-2,:])/(2*hy)

Jx = sigma * Ex
Jy = sigma * Ey

# Streamplot
plt.figure(figsize=(5,5))
plt.streamplot(x, y, Jx, Jy, color=np.hypot(Jx,Jy), cmap="turbo")
plt.title("J-field of 3-phase cable system")
plt.savefig("3phase_J_field.png")
plt.close()