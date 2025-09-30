import numpy as np
import matplotlib.pyplot as plt
import scipy

# -----------------------------
# Pressure grid / runtime flags
# -----------------------------
p_start = 1000
p_end   = 0.1
dp      = 100
uselast = 0  # unused here but preserved

# -----------------------------
# Molecular masses (amu order)
# -----------------------------
mass = np.array([44, 18, 16, 28, 2, 64, 34, 64])

# -----------------------------
# Compositions (mass fractions)
# -----------------------------
Mt_etna = {
    'SiO2': 0.516, 'TiO2': 0.014, 'Al2O3': 0.11, 'FeO': 0.091,
    'MgO': 0.092, 'CaO': 0.126, 'Na2O': 0.035, 'K2O': 0.002, 'P2O5': 0.016
}

basalt = {
    'SiO2': 0.4795, 'TiO2': 0.0167, 'Al2O3': 0.1732, 'FeO': 0.1024,
    'MnO': 0.0017, 'MgO': 0.0576, 'CaO': 0.1093, 'Na2O': 0.0345,
    'K2O': 0.0199, 'P2O5': 0.0051
}

# -----------------------------
# Oxide molar masses (g/mol)
# -----------------------------
melt_mass = {
    'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.096, 'FeO': 71.84, 'MnO': 70.93,
    'MgO': 40.3, 'CaO': 56.07, 'Na2O': 61.97, 'K2O': 94.2, 'P2O5': 283.89
}


def massfractomolfrac(comp):
    """
    Convert oxide mass fractions to mol fractions and compute melt mean molar mass.

    Parameters
    ----------
    comp : dict
        Oxide mass fractions by weight.

    Returns
    -------
    (mol, mumelt) : (dict, float)
        mol : oxide mol fractions (normalized)
        mumelt : mean molar mass of the melt (g/mol)
    """
    mol = {}
    total = 0.0
    for key in basalt.keys():
        mol[key] = comp[key] / melt_mass[key]
        total += comp[key] / melt_mass[key]
    for key in basalt.keys():
        mol[key] /= total

    mumelt = 0.0
    for key in basalt.keys():
        mumelt += mol[key] * melt_mass[key]
    return mol, mumelt


def O2_fugacity2(P, dfmq_fo2, Temp):
    """
    Oxygen fugacity using FMQ buffer parameterization.

    Parameters
    ----------
    P : float
        Pressure (bar).
    dfmq_fo2 : float
        ΔFMQ offset.
    Temp : float
        Temperature (K).

    Returns
    -------
    float
        fO2 (bar).
    """
    A, B, C, D = 5.5976, 24505, 0.8099, 0.0937
    logFMQ = A - (B / Temp) + C * np.log10(Temp) + D * (P - 1.0) / Temp
    fo2 = 10 ** (logFMQ + dfmq_fo2)
    return fo2


def equilibrium_const(T):
    """
    Equilibrium constants (K) for simple redox reactions (Arrhenius-style fits).

    Returns
    -------
    K1..K5 : floats
        K for:
          1) H2O = H2 + O2/2
          2) CO2 = CO + O2/2
          3) CO2 + 2H2O = CH4 + 2O2
          4) 0.5 S2 + O2 = SO2
          5) H2S + 1.5 O2 = SO2 + H2O
    """
    K1 = np.exp(-29755 / T + 6.55)
    K2 = np.exp(-33979 / T + 10.42)
    K3 = np.exp(-96444 / T + 0.22)
    K4 = 10 ** (18880 / T - 3.8018)
    K5 = 10 ** (27103 / T - 4.1973)
    return K1, K2, K3, K4, K5


def symonds_equation(l0, l1, l2, l3, l4, T):
    """Helper polynomial-in-1/T form (base-10 log pieces mixed)."""
    return l0 + l1 / T + l2 * T + l3 / (T**2) + l4 * np.log10(T)


def symonds_equi_const(T):
    """
    Symonds-style equilibrium constants set.

    Returns (base-10 exponentiated values):
      K1: H2O = H2 + 0.5 O2
      K2: CO2 = CO + 0.5 O2
      K3: CO2 + 2H2O = CH4 + 2O2
      K4: 0.5 S2 + O2 = SO2
      K5: 0.5 O2 + H2S = 0.5 S2 + H2O
    """
    # K1
    K1 = -(symonds_equation(4.414, 2.48e4, 3.26e-4, 6.036e3, -3.1546, T)) / 2
    # K2
    K2 = K1 - symonds_equation(-7.768, 2.5445e3, -1.115e-4, -2.4619e4, 1.8435, T)
    # K3
    K3 = 4 * K1 - symonds_equation(-12.869, -7.7318e3, -1.101e-3, 4.447e3, 7.7549, T)
    # K4
    K4 = symonds_equation(-7.905, -5.9248e3, -2.835e-4, -1.6076e4, 3.029, T) - 2 * K1
    # K5
    K5 = -symonds_equation(4.741, 8.5123e3, 0.451e-3, -1.518e3, -3.1279, T) / 2 - K1

    # A sixth constant (K6) appears in the original code but is not returned/used here.
    K6 = 4.98e-12 * np.exp(-85.4 / (8.31e-3 * T))  # kept for parity (unused)

    return 10**K1, 10**K2, 10**K3, 10**K4, 10**K5


def feo_fe2o3_ratio(P, T, fo2, comp):
    """
    Compute FeO/Fe2O3 ratio following empirical parameterization.

    Parameters
    ----------
    P : float
        Pressure (bar).
    T : float
        Temperature (K).
    fo2 : float
        Oxygen fugacity (bar).
    comp : dict
        Composition (mass fractions).

    Returns
    -------
    float
        FeO/Fe2O3 ratio (dimensionless).
    """
    P *= 1e5  # to Pa
    a, b, c = 0.196, 1.1492e4, -6.675
    dal2o3, dfeo, dcao, dna2o, dk2o = -2.243, -1.828, 3.201, 5.854, 6.215
    e, f, g, h = -3.36, -7.01e-7, -1.54e-10, 3.85e-17
    T0 = 1673.0

    comp, mumelt = massfractomolfrac(comp)
    ratio = np.exp(
        a * np.log(fo2) + b / T + c
        + dal2o3 * comp['Al2O3'] + dfeo * comp['FeO'] + dcao * comp['CaO']
        + dna2o * comp['Na2O'] + dk2o * comp['K2O']
        + e * (1.0 - T0 / T - np.log(T / T0))
        + f * P / T + g * (T - T0) * P / T + h * P**2 / T
    )
    P *= 1e-5  # back to bar (not used further but kept)
    return ratio


def fo2_fe(P, T, ratio, comp):
    """
    Invert FeO/Fe2O3 relation to oxygen fugacity.

    Returns
    -------
    float
        fO2 (bar).
    """
    P *= 1e5  # to Pa
    a, b, c = 0.196, 1.1492e4, -6.675
    dal2o3, dfeo, dcao, dna2o, dk2o = -2.243, -1.828, 3.201, 5.854, 6.215
    e, f, g, h = -3.36, -7.01e-7, -1.54e-10, 3.85e-17
    T0 = 1673.0

    comp, mumelt = massfractomolfrac(comp)
    fo2 = np.exp(
        (
            np.log(ratio) - b / T - c
            - dal2o3 * comp['Al2O3'] - dfeo * comp['FeO'] - dcao * comp['CaO']
            - dna2o * comp['Na2O'] - dk2o * comp['K2O']
            - e * (1.0 - T0 / T - np.log(T / T0))
            - f * P / T - g * (T - T0) * P / T - h * P**2 / T
        ) / a
    )
    return fo2 * 1e-5  # back to bar

# Quick check/example call (preserved)
print(fo2_fe(1, 1573, 1e-2, basalt))


# ============================================================
# Gibbs free energies (from external MAT file; paths preserved)
# ============================================================
matlab_arrays = {}
scipy.io.loadmat(
    '/Users/rahularora/Desktop/Project/Work/degassing_model/Gibbsenergies.mat',
    mdict=matlab_arrays
)

# Data origin note (kept from original)
# NIST JANAF tables for most species; S2O from IVTANTHERMO.

# dataT: temperatures (K); dataG: Gibbs energies (kJ/mol/K or as original source states)
dataG = matlab_arrays['dataG']
dataT = np.squeeze(matlab_arrays['dataT'])

# Species order matches original indexing mapping
dataG_dict = {}
list_of_species = [
    'O2', 'O', 'OH', 'H', 'H2', 'H2O', 'HO2', 'N2', 'N', 'NO', 'NO2',
    'N2O', 'NH3', 'CO', 'CO2', 'CH4', 'HCN', 'S2', 'S', 'SO', 'SO2',
    'SO3', 'S2O', 'H2S', 'SH', 'COS'
]
for ind in range(len(list_of_species)):
    dataG_dict[list_of_species[ind]] = np.squeeze(dataG[:, ind])

# -----------------------------
# Equilibrium constant utilities
# -----------------------------
R_J_mol  = 8.3144621
R_kJ_mol = R_J_mol * 1.0e-3  # kJ K^-1 mol^-1


def interp_dataG(species, T):
    """Linear interpolation helper over tabulated Gibbs energies."""
    return np.interp(T, dataT, dataG_dict[species])  # preserves original behavior


# ---- CHO system ----

def K_p_R1(T):
    """R1: CO + 0.5 O2 -> CO2"""
    dG0 = interp_dataG('CO2', T) - (0.5 * interp_dataG('O2', T) + interp_dataG('CO', T))
    return np.exp(-dG0 / (T * R_kJ_mol))


def K_p_R2(T):
    """R2: H2 + 0.5 O2 -> H2O"""
    dG0 = interp_dataG('H2O', T) - (0.5 * interp_dataG('O2', T) + interp_dataG('H2', T))
    return np.exp(-dG0 / (T * R_kJ_mol))


def K_p_R3(T):
    """R3: CH4 + 2 O2 -> CO2 + 2 H2O"""
    dG0 = (2.0 * interp_dataG('H2O', T) + interp_dataG('CO2', T)) - (2.0 * interp_dataG('O2', T) + interp_dataG('CH4', T))
    return np.exp(-dG0 / (T * R_kJ_mol))


# ---- Sulfur system ----

def K_p_R4(T):
    """R4: O2 + 0.5 S2 -> SO2"""
    dG0 = interp_dataG('SO2', T) - (interp_dataG('O2', T) + 0.5 * interp_dataG('S2', T))
    return np.exp(-dG0 / (T * R_kJ_mol))


def K_p_R5(T):
    """R5: 0.5 O2 + H2S -> 0.5 S2 + H2O"""
    dG0 = (interp_dataG('H2O', T) + 0.5 * interp_dataG('S2', T)) - (interp_dataG('H2S', T) + 0.5 * interp_dataG('O2', T))
    return np.exp(-dG0 / (T * R_kJ_mol))


# ---- Nitrogen system ----

def K_p_R6(T):
    """R6: N2 + 3 H2 -> 2 NH3"""
    dG0 = 2.0 * interp_dataG('NH3', T) - (interp_dataG('N2', T) + 3.0 * interp_dataG('H2', T))
    return np.exp(-dG0 / (T * R_kJ_mol))


def K_p_R7(T):
    """R7: N2 + O2 -> 2 NO (combustion-dominant NO formation)"""
    dG0 = 2.0 * interp_dataG('NO', T) - (interp_dataG('N2', T) + interp_dataG('O2', T))
    return np.exp(-dG0 / (T * R_kJ_mol))


# -----------------------------
# (Example usage/plots retained as comments)
# -----------------------------

# temp = np.arange(500, 2000, 50)
# K1, K2, K3, K4, K5 = symonds_equi_const(temp)
# K11 = K_p_R5(temp)
# plt.plot(temp, K5)
# plt.plot(temp, K11, '.')
# plt.xlabel('Temperature(K)')
# plt.ylabel('Equilibrium constant K')
# plt.yscale('log')
# plt.legend()
# plt.show()
