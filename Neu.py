import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. GEMEINSAME PHYSIKALISCHE PARAMETER
# =============================================================================
L_x = 2.0          # Breite des Bodensegments (m)
L_y = 1.5          # Tiefe des Bodensegments (m)
lambda_soil = 1.0  # Wärmeleitfähigkeit des Bodens (W/(m*K))
T_surf = 10.0      # Erdoberflächentemperatur (°C)

h = 0.8            # Vergrabungstiefe des Kabels (m)
kabel_x = 1.0      # X-Position des Kabels (Mitte)
P_meter = 40.0     # Verlustleistung des Kabels pro Meter (W/m)

# =============================================================================
# 2. NUMERISCHES VERFAHREN (Finite Differenzen)
# =============================================================================
Nx, Ny = 100, 75
dx = L_x / (Nx - 1)
dy = L_y / (Ny - 1)

X, Y = np.meshgrid(np.linspace(0, L_x, Nx), np.linspace(0, L_y, Ny))

# Numerische Gitterkoordinaten (y=0 ist oben für die physikalische Logik)
# Für das Rechengitter transformieren wir: Gitter-Zeile 0 ist unten, Zeile Ny-1 ist oben
kabel_idx_x = int(kabel_x / dx)
kabel_idx_y = int((L_y - h) / dy)

# Quellterm: Leistung geteilt durch das Volumen (Fläche * 1m Länge) der Gitterzelle
q_dot = P_meter / (dx * dy) 
Q = np.zeros((Ny, Nx))
Q[kabel_idx_y, kabel_idx_x] = q_dot

T_num = np.ones((Ny, Nx)) * T_surf
max_iter = 8000
tolerance = 1e-5

for it in range(max_iter):
    T_old = T_num.copy()
    
    # Poisson-Schritt für innere Punkte
    T_num[1:-1, 1:-1] = 0.25 * (T_old[1:-1, 2:] + T_old[1:-1, :-2] + 
                                T_old[2:, 1:-1] + T_old[:-2, 1:-1] + 
                                (dx * dy * Q[1:-1, 1:-1] / lambda_soil))
    
    # Randbedingungen
    T_num[-1, :] = T_surf   # Oben: Feste Temperatur (Dirichlet)
    T_num[0, :] = T_num[1, :]   # Unten: Isoliert (Neumann)
    T_num[:, 0] = T_num[:, 1]   # Links: Isoliert (Neumann)
    T_num[:, -1] = T_num[:, -2] # Rechts: Isoliert (Neumann)
    
    if np.max(np.abs(T_num - T_old)) < tolerance:
        break

# Matrix für Plot umdrehen, damit y=0 oben ist
T_num_plot = np.flipud(T_num)

# =============================================================================
# 3. ANALYTISCHES VERFAHREN (Spiegelungs-Methode nach Kennelly)
# =============================================================================
# Physikalische Koordinaten relativ zur Oberfläche (y_phys = 0 ist oben, geht nach unten)
Y_phys = L_y - Y
X_rel = X - kabel_x

# Singularität am Kabelkern vermeiden (Radius einrechnen, z.B. r = 2cm)
r_kabel = 0.02
R_dist = np.sqrt(X_rel**2 + (Y_phys - h)**2)
R_dist = np.maximum(R_dist, r_kabel) # Schneidet die unendliche Spitze im Kern ab

# Spiegelquellen-Distanz (Abstand zur fiktiven negativen Quelle über der Oberfläche)
R_spiegel = np.sqrt(X_rel**2 + (Y_phys + h)**2)

# Kennelly-Formel
T_ana = T_surf + (P_meter / (2 * np.pi * lambda_soil)) * np.log(R_spiegel / R_dist)

# =============================================================================
# 4. VISUALISIERUNG UND GEGENÜBERSTELLUNG
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
levels = np.linspace(10, 30, 21) # Einheitliche Skalierung

# Plot Numerisch
cp1 = axes[0].contourf(X, Y_phys, T_num_plot, levels=levels, cmap='jet', extend='max')
axes[0].set_title('Numerische Lösung (Finite Differenzen)')
axes[0].set_xlabel('Breite (m)')
axes[0].set_ylabel('Tiefe (m)')
axes[0].invert_yaxis()
axes[0].scatter([kabel_x], [h], color='black', marker='o', label='Kabel')
axes[0].grid(True, alpha=0.3)

# Plot Analytisch
cp2 = axes[1].contourf(X, Y_phys, T_ana, levels=levels, cmap='jet', extend='max')
axes[1].set_title('Analytische Lösung (Kennelly-Spiegelung)')
axes[1].set_xlabel('Breite (m)')
axes[1].scatter([kabel_x], [h], color='black', marker='o', label='Kabel')
axes[1].grid(True, alpha=0.3)

# Gemeinsame Formatierung
fig.colorbar(cp2, ax=axes.ravel().tolist(), label='Temperatur (°C)', orientation='vertical', shrink=0.8)
plt.show()