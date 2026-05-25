import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETER UND REALE GEOMETRIE (300 mm² Kabel)
# =============================================================================
L_x, L_y = 2.0, 1.5
Nx, Ny = 150, 112  # Feineres Gitter für bessere Kreisauflösung
dx, dy = L_x / (Nx - 1), L_y / (Ny - 1)

X, Y = np.meshgrid(np.linspace(0, L_x, Nx), np.linspace(0, L_y, Ny))
Y_phys = L_y - Y  # y=0 ist oben

T_surf = 10.0
h = 0.8           # Kabeltiefe (m)
kabel_x = 1.0     # Kabel X-Position (m)
P_meter = 45.0    # Typische Verlustleistung im Lastbetrieb (W/m)

# Radius eines 300 mm² Kabels inkl. Isolierung/Mantel (ca. 3 cm)
r_kabel = 0.03    

# =============================================================================
# 2. MATERIALMATRIX & KABELMASKE DEFINIEREN
# =============================================================================
Lambda = np.ones((Ny, Nx)) * 1.0  # Erdboden: lambda = 1.0 W/(m*K)
lambda_sand = 1.4                 # Sandbett: lambda = 1.4 W/(m*K)

# Sandbett-Maske (40cm breit, 30cm hoch)
sand_maske = (
    (X >= (kabel_x - 0.4/2)) & (X <= (kabel_x + 0.4/2)) &
    (Y_phys >= (h - 0.3/2)) & (Y_phys <= (h + 0.3/2))
)
Lambda[sand_maske] = lambda_sand

# Kabel-Maske (Kreis mit Radius r_kabel)
kabel_maske = (X - kabel_x)**2 + (Y_phys - h)**2 <= r_kabel**2

# Optionale physikalische Verfeinerung: 
# Dem Kabelinneren eine sehr hohe Wärmeleitfähigkeit (Kupfer/Alu) zuweisen
Lambda[kabel_maske] = 200.0  

# =============================================================================
# 3. QUELLTERM (VOLUMENSTROM) BERECHNEN
# =============================================================================
# Fläche des Kabels im Gitter zählen, um die Leistung korrekt aufzuteilen
anzahl_kabel_zellen = np.sum(kabel_maske)
volumen_pro_zelle = dx * dy * 1.0  # Fläche * 1m Länge

# Verlustleistung pro Zelle innerhalb des Kabels
Q = np.zeros((Ny, Nx))
Q[kabel_maske] = P_meter / (anzahl_kabel_zellen * volumen_pro_zelle)

# =============================================================================
# 4. NUMERISCHE ITERATION (Finite Differenzen)
# =============================================================================
T = np.ones((Ny, Nx)) * T_surf
max_iter = 12000
tolerance = 1e-5

for it in range(max_iter):
    T_old = T.copy()
    
    # Harmonische Mittelwerte für Materialübergänge
    L_east  = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[1:-1, 2:])
    L_west  = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[1:-1, :-2])
    L_north = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[2:, 1:-1])
    L_south = 2 / (1/Lambda[1:-1, 1:-1] + 1/Lambda[:-2, 1:-1])
    
    Denom = (L_east + L_west) / dx**2 + (L_north + L_south) / dy**2
    
    T[1:-1, 1:-1] = (
        (L_east * T_old[1:-1, 2:] + L_west * T_old[1:-1, :-2]) / dx**2 +
        (L_north * T_old[2:, 1:-1] + L_south * T_old[:-2, 1:-1]) / dy**2 +
        Q[1:-1, 1:-1]
    ) / Denom
    
    # Randbedingungen
    T[-1, :] = T_surf
    T[0, :] = T[1, :]
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]
    
    if np.max(np.abs(T - T_old)) < tolerance:
        break

T_plot = np.flipud(T)

# =============================================================================
# 5. VISUALISIERUNG
# =============================================================================
plt.figure(figsize=(10, 7))
levels = np.linspace(10, 45, 25)

# Temperaturfeld
cp = plt.contourf(X, Y_phys, T_plot, levels=levels, cmap='jet', extend='max')
plt.colorbar(cp, label='Temperatur (°C)')

# Sandbett und Kabelgrenzen einzeichnen
plt.contour(X, Y_phys, np.flipud(sand_maske.astype(float)), levels=[0.5], colors='white', linestyles='--')
plt.contour(X, Y_phys, np.flipud(kabel_maske.astype(float)), levels=[0.5], colors='black', linewidths=2)

plt.title('Temperaturfeld eines 300 mm² Erdkabels (r = 3 cm) im Sandbett')
plt.xlabel('Breite (m)')
plt.ylabel('Tiefe (m)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.2)
plt.show()

# Maximale Leitertemperatur ausgeben
print(f"Maximale Temperatur im Kabelkern: {np.max(T):.1f} °C")
