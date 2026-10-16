"""Stores physical constants in CGS.
"""
import astropy.constants as a_cst
import numpy as np
import scipy.constants as s_cst

# Defined constants
c = s_cst.c * 1e2
h = s_cst.h * 1e7
kB = s_cst.k * 1e7
nA = s_cst.N_A
e = s_cst.e * np.sqrt(1e9 / (4 * s_cst.pi * s_cst.epsilon_0))

# Measured constants
G = s_cst.G * 1e3
m_elec = s_cst.electron_mass * 1e3
e_molar_mass = m_elec * s_cst.Avogadro  # (g.mol-1) e- molar mass

# Derived exact constants
sigma = s_cst.sigma * 1e3
L0 = s_cst.physical_constants['Loschmidt constant (273.15 K, 101.325 kPa)'][0] * 1e-6
R = s_cst.R

# Units conversion factors to CGS
bar = 1e6
atm = s_cst.atm * 1e1
au = s_cst.au * 1e2
pc = s_cst.parsec * 1e2
light_year = s_cst.light_year * 1e2
amu = s_cst.physical_constants['atomic mass constant'][0] * 1e3

# Astronomical constants
r_sun = float(a_cst.R_sun.cgs.value)  # type: ignore
r_jup = float(a_cst.R_jup.cgs.value)  # type: ignore
r_earth = float(a_cst.R_earth.cgs.value)  # type: ignore
m_sun = float(a_cst.M_sun.cgs.value)  # type: ignore
m_jup = float(a_cst.M_jup.cgs.value)  # type: ignore
m_earth = float(a_cst.M_earth.cgs.value)  # type: ignore
l_sun = float(a_cst.L_sun.cgs.value)  # type: ignore

r_jup_mean = 6.9911e9  # (cm) Jupiter's mean radius
s_earth = 1.3654e6  # (erg.s-1.cm-2) source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010GL045777
