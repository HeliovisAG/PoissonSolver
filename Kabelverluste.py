import numpy as np

A = 400e-6
I = 8.78e6/10.5e3/np.sqrt(3)
r_cond=np.sqrt(A/np.pi)                     # Radius Leiter in m
sigma20 = 5.8e7    # El. Leitfähigkeit von Cu bei 20°C in 1/Ohm m
alpha = 3.93e-4        # Temperaturkoeffizient von Cu
Topmax = 30
R20 = 1/sigma20/A  # Widerstand in Ohm/m bei 20°C
R_DC = R20*(1+alpha*(Topmax-20))
R_AC = R_DC * 1.03                         # Skin- und Proximity Effekt mit 3% angenommen
P = 3*R_AC*I**2       # Leistung in W/m
print(I, P)