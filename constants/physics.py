# ---GLOBAL CONSTANTS FOR THE CREATURE---

# --FRICTION STUFF--
# Friction forces apply according to mu*g*M where M is the mass of creature
g = 10  # gravity constant.
mu_friction = 0.15  # kinetic friction coefficent

# --PUMPS/MASS STUFF--
# The creature weighs M_total across its 3 legs,
# but must always have >= M_min mass on each leg at any time
M_min = 1.0  # minimum mass on each vertex
M_total = 15.0  # total mass of the creature
# The creature pumps mass according to dM/dt = -K_Mass(M - M_eq) where M_eq is the equlibrium mass it is aiming for
K_mass = 4.0  # pump constant for the masses

# --MUSCLES/SPRINGS STUFF--
L_min = 1.6  # minimum spring eq lenth, when muscle = 0
L_max = 2.8  # maximum spring eq length, when muscle = 1
K_spring = 9.0  # spring constant F = K_spring(L - L_eq)
