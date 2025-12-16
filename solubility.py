from subfunctions import *           # project-specific helpers (left as-is)
import numpy as np
from scipy import special

# ---------------------------------
# Global parameters / fit constants
# ---------------------------------
aco2   = 1
ah2o   = 0.54
dh2o   = 2.3
mu_melt = 64.53
muh2o  = 18.02
muco2  = 44.01
Cco2   = 0.14
Ch2o   = 0.02
Bco2   = -5.3
bco2   = 15.8
Bh2o   = -2.95
bh2o   = 1.24
dal2o3 = 3.8
dfeo   = -16.3
dna2o  = 20.1
mus    = 32
muH2   = 2


def x_H2_m_gaillard2003(f_H2_bar):
    """
    Convert H2 fugacity (bar) to H2 mole fraction in melt (Gaillard et al., 2003).

    Parameters
    ----------
    f_H2_bar : float or ndarray
        H2 fugacity in bar (calibrated range ~0.02–70 bar).

    Returns
    -------
    float or ndarray
        Mole fraction of H2 in the melt.

    Notes
    -----
    - Uses a mass-concentration correction (~2.3) per Liggins confirmation,
      matching Table 4 of Gaillard et al. (2003).
    - Retains original units and constants from your code.
    """
    # concentration of H2 in melt in g H2 / cm^2 magma (as in original)
    conc_H2 = (3.4e-7 * f_H2_bar**1.28)
    # convert to mass concentration (factor 2.3 per comment)
    w_H2 = conc_H2 / 2.3
    # convert weight fraction -> mole fraction
    x_H2 = w_H2 * mu_melt / muH2
    return x_H2


def IaconoMarziano2012(comp, P, T):
    """
    Iacono-Marziano et al. (2012) CO2 & H2O solubility terms S1, S2.

    Parameters
    ----------
    comp : dict
        Melt composition (oxide mass fractions).
    P : float
        Pressure (bar).
    T : float
        Temperature (K).

    Returns
    -------
    (S1, S2) : tuple of floats
        Log-solubility terms for CO2 (S1) and H2O (S2).
    """
    comp, mumelt = massfractomolfrac(comp)
    NBOO = (
        2 * (comp['K2O'] + comp['Na2O'] + comp['CaO'] + comp['MgO'] + comp['FeO'] - comp['Al2O3'])
        / (2 * comp['SiO2'] + 2 * comp['TiO2'] + comp['K2O'] + comp['Na2O'] + comp['CaO']
           + comp['MgO'] + comp['FeO'] + 3 * comp['Al2O3'])
    )

    # cite1 eq:A1 (CO2)
    S1 = (
        np.log(mu_melt / (muco2 * 1e6)) + Cco2 * P / T + Bco2 + bco2 * NBOO
        + (comp['Al2O3'] / (comp['K2O'] + comp['Na2O'] + comp['CaO'])) * dal2o3
        + (comp['MgO'] + comp['FeO']) * dfeo
        + (comp['K2O'] + comp['Na2O']) * dna2o
    )

    # cite1 eq:A2 (H2O)
    S2 = np.log(mu_melt / (muh2o * 1e2)) + Ch2o * P / T + Bh2o + bh2o * NBOO

    return S1, S2


def Boulliung2023(comp, P, fs2, fo2, T):
    """
    Boulliung (2023) total S (S2- + S6+) capacity (Eq. 7 & combination).

    Returns
    -------
    float
        Total S capacity (fraction), i.e., (exp(S4) + exp(S3)) / 1e6
    """
    comp, mumelt = massfractomolfrac(comp=comp)

    # cite2 eq 7 (S2- term)
    S3 = (
        0.225
        + (25237 * comp['FeO'] + 5214 * comp['CaO'] + 12705 * comp['MnO']
           + 19829 * comp['K2O'] - 1109 * comp['SiO2'] / 2 - 8879) / T
        + 0.5 * np.log(fs2) - 0.5 * np.log(fo2)
    )

    # S6+ term
    S4 = (
        -12.498
        + (28649 * comp['Na2O'] + 15602 * comp['CaO'] + 9496 * comp['MgO']
           + 16016 * comp['MnO'] + 4194 * comp['Al2O3'] / 3 + 29244) / T
        + 0.5 * np.log(fs2) + 1.5 * np.log(fo2)
    )

    S34 = np.exp(S4) + np.exp(S3)
    return S34 / 1e6


def S_oneill_sulphide(comp, P, fs2, fo2, T):
    """
    O'Neill sulphide capacity -> melt S2- content (as mole fraction).

    Returns
    -------
    float
        x_S_m : S (as S2-) mole fraction in melt.
    """
    # Normalize oxide weights to total (logic preserved)
    wtsio2 = comp["SiO2"]; wttio2 = comp["TiO2"]; wtal2o3 = comp["Al2O3"]
    wtfeo  = comp["FeO"];  wtmno  = comp["MnO"];  wtmgo   = comp["MgO"]
    wtcao  = comp["CaO"];  wtna2o = comp["Na2O"]; wtk2o   = comp["K2O"]
    wtp2o5 = comp["P2O5"]

    oxide_tot = (wtsio2 + wttio2 + wtal2o3 + wtfeo + wtmno
                 + wtmgo + wtcao + wtna2o + wtk2o + wtp2o5)

    wtsio2 /= oxide_tot; wttio2 /= oxide_tot; wtal2o3 /= oxide_tot
    wtfeo  /= oxide_tot; wtmno  /= oxide_tot; wtmgo   /= oxide_tot
    wtcao  /= oxide_tot; wtna2o /= oxide_tot; wtk2o   /= oxide_tot
    wtp2o5 /= oxide_tot

    # Moles of oxides
    nsio2 = wtsio2 / (28.086 + 15.999 * 2)
    ntio2 = wttio2 / (47.867 + 15.999 * 2)
    nal2o3 = wtal2o3 / (26.982 * 2 + 15.999 * 3)
    nfeo = wtfeo / (55.845 + 15.999)
    nmno = wtmno / (54.938 + 15.999)
    nmgo = wtmgo / (24.305 + 15.999)
    ncao = wtcao / (40.078 + 15.999)
    nna2o = wtna2o / (22.9898 * 2 + 15.999)
    nk2o = wtk2o / (39.098 * 2 + 15.999)
    np2o5 = wtp2o5 / (30.973 * 2 + 15.999 * 5)

    # Site totals and cation fractions X_i (as in original)
    xtot = nsio2 + ntio2 + 0.5 * nal2o3 + nfeo + nmno + nmgo + nmgo + 0.5 * nna2o + 0.5 * nk2o
    X_Na = (wtna2o / 30.99) / xtot
    X_Mg = (wtmgo / 40.32) / xtot
    X_Al = (wtal2o3 / 50.98) / xtot
    X_Si = (wtsio2 / 60.08) / xtot
    X_K  = (wtk2o  / 47.1)  / xtot
    X_Ca = (wtcao  / 56.08) / xtot
    X_Ti = (wttio2 / 79.9)  / xtot
    X_Mn = (wtmno  / 70.94) / xtot
    X_Fe = (wtfeo  / 71.85) / xtot

    # ln capacity (kept exactly, incl. erf term)
    ln_C_S_silicate_ppmw = (
        8.77
        + (-23590 + 1673 * (6.7 * (X_Na + X_K) + 1.8 * X_Al + 4.9 * X_Mg
                            + 8.1 * X_Ca + 5 * X_Ti + 8.9 * (X_Fe + X_Mn)
                            - 22.2 * (X_Fe + X_Mn) * X_Ti
                            + 7.2 * (X_Fe + X_Mn) * X_Si)) / T
        - 2.06 * special.erf(-7.2 * (X_Fe + X_Mn))
    )

    C_S_m = np.exp(ln_C_S_silicate_ppmw) / 1e6  # ppmw -> fraction
    # eq. 3 Wogan+2020: weight -> mole fraction (retain ppm factor path)
    w_S_m = C_S_m * ((fs2 / fo2) ** 0.5)
    x_S_m = w_S_m * mu_melt / mus
    return x_S_m


def S_oneill_sulphate(comp, P, fs2, fo2, T):
    """
    O'Neill sulphate capacity -> melt S6+ content (as mole fraction).

    Returns
    -------
    float
        x_S_m : S (as S6+) mole fraction in melt.
    """
    # Normalize oxides
    wtsio2 = comp["SiO2"]; wttio2 = comp["TiO2"]; wtal2o3 = comp["Al2O3"]
    wtfeo  = comp["FeO"];  wtmno  = comp["MnO"];  wtmgo   = comp["MgO"]
    wtcao  = comp["CaO"];  wtna2o = comp["Na2O"]; wtk2o   = comp["K2O"]
    wtp2o5 = comp["P2O5"]

    oxide_tot = (wtsio2 + wttio2 + wtal2o3 + wtfeo + wtmno
                 + wtmgo + wtcao + wtna2o + wtk2o + wtp2o5)

    wtsio2 /= oxide_tot; wttio2 /= oxide_tot; wtal2o3 /= oxide_tot
    wtfeo  /= oxide_tot; wtmno  /= oxide_tot; wtmgo   /= oxide_tot
    wtcao  /= oxide_tot; wtna2o /= oxide_tot; wtk2o   /= oxide_tot
    wtp2o5 /= oxide_tot

    # Moles of oxides
    nsio2 = wtsio2 / (28.086 + 15.999 * 2)
    ntio2 = wttio2 / (47.867 + 15.999 * 2)
    nal2o3 = wtal2o3 / (26.982 * 2 + 15.999 * 3)
    nfeo = wtfeo / (55.845 + 15.999)
    nmno = wtmno / (54.938 + 15.999)
    nmgo = wtmgo / (24.305 + 15.999)
    ncao = wtcao / (40.078 + 15.999)
    nna2o = wtna2o / (22.9898 * 2 + 15.999)
    nk2o = wtk2o / (39.098 * 2 + 15.999)
    np2o5 = wtp2o5 / (30.973 * 2 + 15.999 * 5)

    # Cation fractions
    xtot = nsio2 + ntio2 + 0.5 * nal2o3 + nfeo + nmno + nmgo + nmgo + 0.5 * nna2o + 0.5 * nk2o
    X_Na = (wtna2o / 30.99) / xtot
    X_Mg = (wtmgo / 40.32) / xtot
    X_Al = (wtal2o3 / 50.98) / xtot
    X_Si = (wtsio2 / 60.08) / xtot
    X_K  = (wtk2o  / 47.1)  / xtot
    X_Ca = (wtcao  / 56.08) / xtot
    X_Ti = (wttio2 / 79.9)  / xtot
    X_Mn = (wtmno  / 70.94) / xtot
    X_Fe = (wtfeo  / 71.85) / xtot

    # ln capacity (H2SO4 side), then ppmw path to mole fraction
    ln_C_S_silicate_ppmw = (
        -8.02
        + (21100 + 44000 * X_Na + 18700 * X_Mg + 4300 * X_Al + 44200 * X_K
           + 35600 * X_Ca + 12600 * X_Mn + 16500 * X_Fe) / T
    )

    ln_w_S_m_ppmw = ln_C_S_silicate_ppmw + 0.5 * np.log(fs2) + 1.5 * np.log(fo2)
    w_S_m_ppmw = np.exp(ln_w_S_m_ppmw)           # ppmw
    x_S_m = (w_S_m_ppmw) * mu_melt / mus         # ppmw path preserved
    return x_S_m


# ----------------------------
# (Alternative sulphate via Fe3+/Fe2+ relation retained as comments)
# ----------------------------

# def S_oneill_sulphate(comp, P,fs2,fo2,T):
#     F = 2* feo_fe2o3_ratio(P,T,fo2,comp)
#     ratio = 10 ** (8 * np.log10(F) + (8.7436e6 / T**2) - (27703 / T) + 20.273)
#     return ratio * S_oneill_sulphide(comp,P,fs2,fo2, T)


def N_libourel(pn2, fo2):
    """
    Libourel-style N content proxy (units preserved from original code).

    Parameters
    ----------
    pn2 : float or ndarray
        Partial pressure of N2 (bar).
    fo2 : float or ndarray
        Oxygen fugacity (bar).

    Returns
    -------
    float or ndarray
        mN2 * 1e3 (units as in original).
    """
    mN2 = 2.21e-9 * pn2 + fo2**(-0.75) * 2.13e-17 * (pn2**0.5)
    return mN2 * 1e3


# ----------------------------
# Temperature / redox sweeps
# ----------------------------
Temp = np.arange(873, 1873, 10)       # K
dfmq = np.arange(-4, 4, 0.5)          # ΔFMQ range

# Solubility terms vs T at fixed P (left as-is)
H2o, co2 = IaconoMarziano2012(basalt, 1000, Temp)
S6 = S_oneill_sulphate(basalt, 1, 1e-6, 1e-6, Temp)

# ----------------------------
# (Plotting/diagnostics preserved as comments)
# ----------------------------

# for i in range(len(dfmq)):
#     S2 = S_oneill_sulphide(basalt,1,1e-10,O2_fugacity2(1e-3,dfmq[i],Temp),Temp) 
#     plt.plot(Temp,S2,label=f'FMQ {dfmq[i]}')
# plt.xlabel('fO2 (FMQ)')
# plt.ylabel('Sulphur mole fraction in melt')
# plt.yscale('log')
# plt.legend(ncol = 3)
# plt.savefig('/Users/rahularora/Desktop/Project/Work/degassing_model/temp_s2-.png')
# plt.clf()

# for i in range(len(dfmq)):
#     S2 = S_oneill_sulphate(basalt,1,1e-1,O2_fugacity2(1e-3,dfmq[i],Temp),Temp)
#     plt.plot(Temp,S2,label=f'FMQ {dfmq[i]}')
# plt.xlabel('fO2 (FMQ)')
# plt.ylabel('Sulphur mole fraction in melt')
# plt.yscale('log')
# plt.legend(ncol = 3)
# plt.savefig('/Users/rahularora/Desktop/Project/Work/degassing_model/temp_s6+.png')
# plt.clf()

# for i in range(len(dfmq)):
#     S2 = (S_oneill_sulphate(basalt,1,1e-1,O2_fugacity2(1e-3,dfmq[i],Temp),Temp)
#           + S_oneill_sulphide(basalt,1,1e-10,O2_fugacity2(1e-3,dfmq[i],Temp),Temp))
#     plt.plot(Temp,S2,label=f'FMQ {dfmq[i]}')
# plt.xlabel('fO2 (FMQ)')
# plt.ylabel('Sulphur mole fraction in melt')
# plt.yscale('log')
# plt.legend(ncol = 3)
# plt.savefig('/Users/rahularora/Desktop/Project/Work/degassing_model/temp_solubility.png')

# ----------------------------
# Constant-P sweep (logic preserved; plotting commented in original)
# ----------------------------
temp = np.arange(973, 1873, 100)
dfmq = np.arange(-5, 5.5, 0.5)
P = np.ones_like(dfmq)

for t in range(len(temp)):
    K1, K2, K3, K4, K5 = symonds_equi_const(temp[t])
    fo2 = O2_fugacity2(1, dfmq, temp[t])

    # Solve for fs2 using quadratic in each redox state (coeffs preserved)
    fs2 = np.ones_like(fo2)
    for i in range(len(fo2)):
        fs2_coeff = [1, -2 * P[i] + 2 * fo2[i] - (K4 * fo2[i])**2, P[i]**2 + fo2[i]**2 - 2 * P[i] * fo2[i]]
        fs2_roots = np.roots(fs2_coeff)
        root = fs2_roots[fs2_roots <= 1]
        fs2[i] = root

    fso2 = P - fo2 - fs2

    # Melt S contents vs fO2 at this T (not plotted here)
    S2_minus = S_oneill_sulphide(basalt, P, fs2, fo2, temp[t])
    S6_plus  = S_oneill_sulphate(basalt, P, fs2, fo2, temp[t])



def sulphur_weigth_ratio(comp, fo2, T):
    """
    Compute (C_S_sulphate / C_S_sulphide) * fo2^2 and return with C_S_sulphide.

    Returns
    -------
    (ratio, C_S_sulphide) : tuple of floats
        ratio = (C_S_sulphate / C_S_sulphide) * fo2**2
    """
    # Normalize oxides
    wtsio2 = comp["SiO2"]; wttio2 = comp["TiO2"]; wtal2o3 = comp["Al2O3"]
    wtfeo  = comp["FeO"];  wtmno  = comp["MnO"];  wtmgo   = comp["MgO"]
    wtcao  = comp["CaO"];  wtna2o = comp["Na2O"]; wtk2o   = comp["K2O"]
    wtp2o5 = comp["P2O5"]

    oxide_tot = (wtsio2 + wttio2 + wtal2o3 + wtfeo + wtmno
                 + wtmgo + wtcao + wtna2o + wtk2o + wtp2o5)

    wtsio2 /= oxide_tot; wttio2 /= oxide_tot; wtal2o3 /= oxide_tot
    wtfeo  /= oxide_tot; wtmno  /= oxide_tot; wtmgo   /= oxide_tot
    wtcao  /= oxide_tot; wtna2o /= oxide_tot; wtk2o   /= oxide_tot
    wtp2o5 /= oxide_tot

    # Moles & cation fractions
    nsio2 = wtsio2 / (28.086 + 15.999 * 2)
    ntio2 = wttio2 / (47.867 + 15.999 * 2)
    nal2o3 = wtal2o3 / (26.982 * 2 + 15.999 * 3)
    nfeo = wtfeo / (55.845 + 15.999)
    nmno = wtmno / (54.938 + 15.999)
    nmgo = wtmgo / (24.305 + 15.999)
    ncao = wtcao / (40.078 + 15.999)
    nna2o = wtna2o / (22.9898 * 2 + 15.999)
    nk2o = wtk2o / (39.098 * 2 + 15.999)
    np2o5 = wtp2o5 / (30.973 * 2 + 15.999 * 5)

    xtot = nsio2 + ntio2 + 0.5 * nal2o3 + nfeo + nmno + nmgo + nmgo + 0.5 * nna2o + 0.5 * nk2o
    X_Na = (wtna2o / 30.99) / xtot
    X_Mg = (wtmgo / 40.32) / xtot
    X_Al = (wtal2o3 / 50.98) / xtot
    X_Si = (wtsio2 / 60.08) / xtot
    X_K  = (wtk2o  / 47.1)  / xtot
    X_Ca = (wtcao  / 56.08) / xtot
    X_Ti = (wttio2 / 79.9)  / xtot
    X_Mn = (wtmno  / 70.94) / xtot
    X_Fe = (wtfeo  / 71.85) / xtot

    # Sulphide capacity (ppmw pathway)
    ln_C_S_sulphide = (
        8.77
        + (-23590 + 1673 * (6.7 * (X_Na + X_K) + 1.8 * X_Al + 4.9 * X_Mg
                            + 8.1 * X_Ca + 5 * X_Ti + 8.9 * (X_Fe + X_Mn)
                            - 22.2 * (X_Fe + X_Mn) * X_Ti
                            + 7.2 * (X_Fe + X_Mn) * X_Si)) / T
        - 2.06 * special.erf(-7.2 * (X_Fe + X_Mn))
    )
    C_S_sulphide = np.exp(ln_C_S_sulphide) / 1e6  # ppmw -> fraction

    # Sulphate capacity
    ln_C_S_sulphate = (
        -8.02
        + (21100 + 44000 * X_Na + 18700 * X_Mg + 4300 * X_Al + 44200 * X_K
           + 35600 * X_Ca + 12600 * X_Mn + 16500 * X_Fe) / T
    )
    C_S_sulphate = np.exp(ln_C_S_sulphate)

    # Return ratio*fo2^2 and sulphide capacity, as in original
    return (C_S_sulphate / C_S_sulphide) * (fo2**2), C_S_sulphide

