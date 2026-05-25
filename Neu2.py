import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETER UND GITTER DEFINIEREN
# =============================================================================
L_x, L_y = 2.0, 1.5
Nx, Ny = 100, 75
dx, dy = L_x / (Nx - 1), L_y / (Ny - 1)

X, Y = np.meshgrid(np.linspace(0, L_x, Nx), np.linspace(0, L_y, Ny))
Y_phys = L_y - Y  # y=0 ist oben

T_surf = 10.0
h = 0.8           # Kabeltiefe (m)
kabel_x = 1.0     # Kabel X-Position (m)
P_meter = 50.0    # Höhere Verlustleistung für deutliche Effekte (W/m)

# =============================================================================
# 2. MATERIALMATRIX (LAMBDA) DEFINIEREN
# =============================================================================
# Standardmäßig überall normaler Boden
Lambda = np.ones((Ny, Nx)) * 1.0  # lambda_soil = 1.0 W/(m*K)

# Sandbett-Abmessungen definieren (z. B. 40cm breit, 30cm hoch um das Kabel)
sand_breite = 0.4
sand_hoehe = 0.3
lambda_sand = 1.4  # Bessere Wärmeleitfähigkeit von verdichtetem Kabelsand

# Maske für das Sandbett erstellen
sand_maske = (
    (X >= (kabel_x - sand_breite/2)) & (X <= (kabel_x + sand_breite/2)) &
    (Y_phys >= (h - sand_hoehe/2)) & (Y_phys <= (h + sand_hoehe/2))
)
# Sandbett in die Materialmatrix einprägen
Lambda[sand_maske] = lambda_sand

# =============================================================================
# 3. QUELLTERM INJIZIEREN
# =============================================================================
kabel_idx_x = int(kabel_x / dx)
kabel_idx_y = int((L_y - h) / dy)

q_dot = P_meter / (dx * dy)
Q = np.zeros((Ny, Nx))
Q[kabel_idx_y, kabel_idx_x] = q_dot

# =============================================================================
# 4. NUMERISCHE ITERATION (Erweiterte Poisson-Gleichung)
# =============================================================================
T = np.ones((Ny, Nx)) * T_surf
max_iter = 10000
tolerance = 1e-5

for it in range(max_iter):
    T_old = T.copy()
    
    # Harmonische Mittelwerte für Lambda an den Grenzflächen (wichtig für Materialübergänge)
    L_east  = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[1:-1, 2:])
    L_west  = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[1:-1, :-2])
    L_north = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[2:, 1:-1])
    L_south = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[:-2, 1:-1])
    
    # Nenner für die Normierung des Knotens
    Denom = (L_east + L_west) / dx**2 + (L_north + L_south) / dy**2
    
    # Erweiterter FDM-Schritt
    T[1:-1, 1:-1] = (
        (L_east * T_old[1:-1, 2:] + L_west * T_old[1:-1, :-2]) / dx**2 +
        (L_north * T_old[2:, 1:-1] + L_south * T_old[:-2, 1:-1]) / dy**2 +
        Q[1:-1, 1:-1]
    ) / Denom
    
    # Randbedingungen
    T[-1, :] = T_surf           # Oben (Dirichlet)
    T[0, :] = T[1, :]           # Unten (Neumann)
    T[:, 0] = T[:, 1]           # Links (Neumann)
    T[:, -1] = T[:, -2]         # Rechts (Neumann)
    
    if np.max(np.abs(T - T_old)) < tolerance:
        print(f"Konvergiert nach {it} Iterationen.")
        break

T_plot = np.flipud(T)

# =============================================================================
# 5. VISUALISIERUNG
# =============================================================================
plt.figure(figsize=(10, 7))
levels = np.linspace(10, 30, 21)

# Temperaturfeld plotten
cp = plt.contourf(X, Y_phys, T_plot, levels=levels, cmap='jet', extend='max')
plt.colorbar(cp, label='Temperatur (°C)')

# Umriss des Sandbetts einzeichnen
sand_kontur = np.flipud(sand_maske.astype(float))
plt.contour(X, Y_phys, sand_kontur, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
plt.text(kabel_x - 0.18, h - 0.18, 'Sandbett', color='white', weight='bold', fontsize=10)

# Kabel einzeichnen
plt.scatter([kabel_x], [h], color='black', marker='o', s=50, label='Kabel', zorder=5)

plt.title('Temperaturverteilung mit optimiertem Sandbett (weiße Kontur)')
plt.xlabel('Breite (m)')
plt.ylabel('Tiefe (m)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.2)
plt.legend()
plt.show()
