import numpy as np
from scipy import optimize
import os
from subfunctions import *          # project-specific helpers (left as-is)
from solubility import *           # project-specific solubility functions (left as-is)
import pandas as pd

"""
References (for your own bookkeeping; no code-level effect):
- cite1: https://doi.org/10.48550/arXiv.2009.07761
- cite2: https://doi.org/10.1007/s00410-023-02033-9
"""

# ----------------------------
# Global constants / parameters
# ----------------------------

# Empirical/fit parameters (left as provided)
aco2   = 1
ah2o   = 0.54
dh2o   = 2.3
mu_melt = 64.53   # mean molar mass of melt [g/mol]
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
mus    = 32        # molar mass for S in accounting below [g/mol]

# Pressure sweep defaults (not used directly—kept to preserve original definitions)
p_start = 1000
p_end   = 0.1
dp      = 100

# Control flags
uselast = 1     # try to seed solver with last saved row if available
newrun  = 1     # if 0, skip cases that already have output CSVs

# Gas molar masses used for "mmw" calculation order-aligned with "molecules" below
mass = np.array([44, 18, 16, 28, 2, 64, 34, 64])

# Output set selector
fileno = 0

# Gas species labels (degassed partial pressures); order matches solver outputs
molecules = [
    'pco2', 'ph2o', 'pch4', 'pco', 'ph2', 'pso2', 'ph2s', 'ps2',
    'Total S flux', 'h2s_so2'
]

# Output directory variants (unchanged)
outfile = [
    'Output_with_gc_COHS',
]

# Representative compositions and molar masses (kept as-is)
Mt_etna = {
    'SiO2': 0.516, 'TiO2': 0.014, 'Al2O3': 0.11,  'FeO': 0.091,
    'MgO': 0.092,  'CaO': 0.126, 'Na2O': 0.035, 'K2O': 0.002, 'P2O5': 0.016
}

basalt = {
    'SiO2': 0.4795, 'TiO2': 0.0167, 'Al2O3': 0.1732, 'FeO': 0.1024, 'MnO': 0.0017,
    'MgO': 0.0576,  'CaO': 0.1093, 'Na2O': 0.0345,  'K2O': 0.0199,  'P2O5': 0.0051
}

melt_mass = {
    'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.096, 'FeO': 71.84, 'MnO': 70.93,
    'MgO': 40.3,   'CaO': 56.07,   'Na2O': 61.97,    'K2O': 94.2,  'P2O5': 283.89
}

# ----------------------------
# Helper thermodynamic relations
# ----------------------------

def log10_K1(T: float, P: float) -> float:
    """log10 of equilibrium constant K1 as a function of T [K] and P [bar]."""
    return 40.07639 - 2.53932e-2 * T + 5.27096e-6 * T**2 + 0.0267 * (P - 1) / T

def log10_K2(T: float, P: float) -> float:
    """log10 of equilibrium constant K2 as a function of T [K] and P [bar]."""
    return -6.24763 - (282.56 / T) - 0.119242 * (P - 1000) / T

def X_CO3_melt(T: float, P: float, Fo2_val: float) -> float:
    """
    Fraction of carbonate species in melt (X_CO3) as function of T, P, and fO2.
    Uses K1, K2 and provided fO2.
    """
    K1 = 10 ** log10_K1(T, P)
    K2 = 10 ** log10_K2(T, P)
    numerator = K1 * K2 * Fo2_val
    denominator = 1 + K1 * K2 * Fo2_val
    return numerator / denominator

def X_CO2_final(M_CO2: float, fwm: float, X_CO3_melt_val: float) -> float:
    """
    CO2 saturation term used to cap m_co2_tot (as in original code).
    Parameters:
      - M_CO2:  (kept as original: value 44 used)
      - fwm:    (kept as original: value 36.594 used)
      - X_CO3_melt_val: carbonate fraction in melt
    """
    numerator = (M_CO2 / fwm) * X_CO3_melt_val
    denominator = 1 - (1 - (M_CO2 / fwm)) * X_CO3_melt_val
    return numerator / denominator

def massfractomolfrac(comp: dict) -> dict:
    """
    Convert oxide mass fractions to mole fractions for a given composition dict.
    Returns a dict of mole fractions across keys in 'basalt'.
    """
    mol = {}
    total = 0.0
    for ox in basalt.keys():
        mol[ox] = comp[ox] / melt_mass[ox]
        total += mol[ox]
    for ox in basalt.keys():
        mol[ox] /= total
    return mol

# ----------------------------
# Nonlinear system definitions
# ----------------------------

def COHS_sys_eqs(
    guesses, pressure, mco2_tot, mh2o_tot, ms_tot, dfmq, temp, flag
):
    """
    Full CO-H-S-O system:
    Unknowns (in order):
      mf_co2, mf_h2o, mf_s, mf_h2,
      pco2, ph2o, pch4, pco, ph2, pso2, ph2s, ps2,
      alpha_gas
    Returns tuple of residuals eq1..eq14 (eq10 intentionally omitted).
    """
    # Unpack unknowns (exact order preserved)
    mf_co2, mf_h2o, mf_s, mf_h2, \
    pco2, ph2o, pch4, pco, ph2, pso2, ph2s, ps2, alpha_gas = guesses

    # Oxygen fugacity
    fO2 = O2_fugacity2(pressure, dfmq, temp)

    # Total pressure closure (includes fO2 as in original)
    eq1 = 1 - (ph2o + pco2 + pco + pch4 + ph2 + pso2 + ps2 + ph2s + fO2) / pressure

    # Solubilities (CO2, H2O, S) at given P, T; sulfur split into sulphide + sulphate
    SCo2, SH2o = IaconoMarziano2012(basalt, pressure, temp)
    Ss = (S_oneill_sulphide(basalt, pressure, ps2, fO2, temp)
          + S_oneill_sulphate(basalt, pressure, ps2, fO2, temp))
    Sh2 = x_H2_m_gaillard2003(ph2)

    # Solubility constraints
    # CO2 (switch to fixed-mass case if flag==1)
    if flag == 0:
        eq2 = 1 - np.exp(mf_h2o * dh2o + aco2 * np.log(pco2) + SCo2) / mf_co2
    else:
        eq2 = 1 - (mf_co2 / mco2_tot)

    # H2O
    eq3 = 1 - np.exp(ah2o * np.log(ph2o) + SH2o) / mf_h2o

    # S2 (total S in melt vs solubility)
    eq4 = 1 - (Ss / mf_s)

    # Gas-phase equilibrium constants (Symonds-style)
    K1, K2, K3, K4, K5 = symonds_equi_const(temp)

    # Gas equilibria (kept identical)
    eq5 = 1 - (np.log(ph2) + 0.5 * np.log(fO2) - np.log(ph2o)) / np.log(K1)
    eq6 = 1 - (np.log(pco) + 0.5 * np.log(fO2) - np.log(pco2)) / np.log(K2)
    eq7 = 1 - (np.log(pch4) + 2 * np.log(fO2) - np.log(pco2) - 2 * np.log(ph2o)) / np.log(K3)
    eq8 = 1 - (np.log(pso2) - 0.5 * np.log(ps2) - np.log(fO2)) / np.log(K4)
    eq9 = 1 - ((np.log(ps2) / 2) + np.log(ph2o) - np.log(ph2s) - (np.log(fO2) / 2)) / np.log(K5)
    # eq10 reserved in original code (P-OCS equilibrium), intentionally omitted

    # Elemental mass balance (CO2, H2O, S)
    eq11 = 1 - ((pco2 + pco + pch4) * alpha_gas / pressure
                + (1 - alpha_gas) * mf_co2) / (mco2_tot * mu_melt / muco2)
    eq12 = 1 - ((ph2 + ph2o + 2 * pch4) * alpha_gas / pressure
                + (1 - alpha_gas) * mf_h2o) / (mh2o_tot * mu_melt / muh2o)
    eq13 = 1 - ((ph2s + pso2 + 2 * ps2) * alpha_gas / pressure
                + (1 - alpha_gas) * mf_s) / (ms_tot * mu_melt / mus)

    # H2 solubility constraint
    eq14 = 1 - (Sh2 / mf_h2)

    return (eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq11, eq12, eq13, eq14)

def COHS_sys_eqs_gc(
    guesses, pressure, mco2_tot, mh2o_tot, ms_tot, dfmq, temp
):
    """
    Variant where pCO2 is directly tied to fO2 via:
      pCO2 = fO2 * exp(47457/T + 0.136)
    All other relations remain as in COHS_sys_eqs.
    """
    mf_co2, mf_h2o, mf_s, mf_h2, \
    pco2, ph2o, pch4, pco, ph2, pso2, ph2s, ps2, alpha_gas = guesses

    fO2 = O2_fugacity2(pressure, dfmq, temp)
    pco2 = fO2 * np.exp((47457 / temp) + 0.136)

    eq1 = 1 - (ph2o + pco2 + pco + pch4 + ph2 + pso2 + ps2 + ph2s + fO2) / pressure

    SCo2, SH2o = IaconoMarziano2012(basalt, pressure, temp)
    Ss = (S_oneill_sulphide(basalt, pressure, ps2, fO2, temp)
          + S_oneill_sulphate(basalt, pressure, ps2, fO2, temp))
    Sh2 = x_H2_m_gaillard2003(ph2)

    eq2 = 1 - np.exp(mf_h2o * dh2o + aco2 * np.log(pco2) + SCo2) / mf_co2
    eq3 = 1 - np.exp(ah2o * np.log(ph2o) + SH2o) / mf_h2o
    eq4 = 1 - (Ss / mf_s)

    K1, K2, K3, K4, K5 = symonds_equi_const(temp)

    eq5 = 1 - (np.log(ph2) + 0.5 * np.log(fO2) - np.log(ph2o)) / np.log(K1)
    eq6 = 1 - (np.log(pco) + 0.5 * np.log(fO2) - np.log(pco2)) / np.log(K2)
    eq7 = 1 - (np.log(pch4) + 2 * np.log(fO2) - np.log(pco2) - 2 * np.log(ph2o)) / np.log(K3)
    eq8 = 1 - (np.log(pso2) - 0.5 * np.log(ps2) - np.log(fO2)) / np.log(K4)
    eq9 = 1 - ((np.log(ps2) / 2) + np.log(ph2o) - np.log(ph2s) - (np.log(fO2) / 2)) / np.log(K5)

    # eq11 deliberately set to 0 in original; keep it that way
    eq11 = 0

    eq12 = 1 - ((ph2 + ph2o + 2 * pch4) * alpha_gas / pressure
                + (1 - alpha_gas) * mf_h2o) / (mh2o_tot * mu_melt / muh2o)
    eq13 = 1 - ((ph2s + pso2 + 2 * ps2) * alpha_gas / pressure
                + (1 - alpha_gas) * mf_s) / (ms_tot * mu_melt / mus)
    eq14 = 1 - (Sh2 / mf_h2)

    return (eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq11, eq12, eq13, eq14)

# ----------------------------
# Solver wrapper
# ----------------------------

def COHS_sys_solver(temp, mco2_tot, mh2o_tot, ms_tot, dfmq, contfile):
    """
    March in pressure (logspace from 10^3 to 10^-3 bar) and solve the COHS system
    at each level using previous solution as the initial guess.
    """
    # Initial fO2 at 1000 bar (kept as-is)
    fO2 = O2_fugacity2(1000, dfmq, temp)

    # Pressure grid (unchanged)
    if contfile == 'na':
        press_log = np.logspace(3, -3, 100)
    else:
        press_log = np.logspace(3, -3, 100)

    Pres = press_log[0]

    # Equilibrium constants & solubilities at top pressure (for initial guesses)
    K1, K2, K3, K4, K5 = symonds_equi_const(temp)
    SCo2, SH2o = IaconoMarziano2012(basalt, Pres, temp)

    # Try to initialize from last file row if requested and available
    if (uselast == 1 and os.path.exists(
        f'./{outfile[fileno]}/{contfile}'
    )):
        lastfile = np.genfromtxt(
            f'./{outfile[fileno]}/{contfile}',
            delimiter=','
        )
        ind = np.argmin(np.abs(Pres * np.ones_like(lastfile[1:, 0]) - lastfile[1:, 0]))
        guess1 = lastfile[ind + 1, 2:]
    else:
        # Construct physically-consistent seed at top pressure
        alpha_gas_guess = 1e-3
        mf_co2_guess = (1 - alpha_gas_guess) * mco2_tot * mu_melt / muco2
        mf_h2o_guess = (1 - alpha_gas_guess) * mh2o_tot * mu_melt / muh2o
        mf_s_guess   = (1 - alpha_gas_guess) * ms_tot   * mu_melt / mus

        ph2o_guess = np.exp((np.log(mf_h2o_guess) - SH2o) / ah2o)
        pco2_guess = np.exp((np.log(mf_co2_guess) - mf_h2o_guess * dh2o - SCo2) / aco2)
        ph2_guess  = K1 * ph2o_guess / (fO2 ** 0.5)
        pco_guess  = K2 * pco2_guess / (fO2 ** 0.5)
        pch4_guess = K3 * (ph2o_guess ** 2) * pco2_guess / (fO2 ** 2)
        mf_h2_guess = x_H2_m_gaillard2003(ph2_guess)

        # Initial S-bearing gas guesses: split the pressure evenly (as in original)
        pso2_guess = press_log[0] / 20
        ph2s_guess = press_log[0] / 20
        ps2_guess  = press_log[0] / 20

        guess1 = np.array([
            mf_co2_guess, mf_h2o_guess, mf_s_guess, mf_h2_guess,
            pco2_guess, ph2o_guess, pch4_guess, pco_guess, ph2_guess,
            pso2_guess, ph2s_guess, ps2_guess, 1e-3
        ])

    sol = []
    press = []

    # Pressure loop (top -> bottom)
    for Pres in press_log:
        ind = np.where(press_log == Pres)[0][0]

        if Pres == press_log[0]:
            # CO2 saturation limit for first step
            chi_co2 = X_CO2_final(44, 36.594, X_CO3_melt(temp, Pres, fO2))
            flag = 0
            m_co2_tot = np.min([chi_co2, mco2_tot])
            if chi_co2 < mco2_tot:
                flag = 1
                print(f'flag = {flag} , {chi_co2} , {mco2_tot}')

            # Solve from initial guess
            result = optimize.root(
                COHS_sys_eqs, guess1,
                args=(Pres, m_co2_tot, mh2o_tot, ms_tot, dfmq, temp, flag),
                method='lm',
                options={'maxiter': 1000000, 'xtol': 5.0e-16, 'ftol': 5.0e-16}
            )

            # Optional GC check (left in place)
            fO2 = O2_fugacity2(Pres, dfmq, temp)
            gc_check = result['x'][4] / (fO2 * np.exp(47457 / temp + 0.136))

            sol.append(result['x'])
            func_eval = np.abs(result['fun'])

            # Hard stop if solver failed or tolerance not met
            if (result['success'] is False) or (len(func_eval[func_eval > 1e-4]) != 0):
                print(result)
                print(result['fun'])
                print(Pres)
                exit()

        else:
            # Continuation: cap mco2_tot at saturation if needed
            chi_co2 = X_CO2_final(44, 36.594, X_CO3_melt(temp, Pres, fO2))
            flag = 0
            m_co2_tot = np.min([chi_co2, mco2_tot])
            if chi_co2 < mco2_tot:
                flag = 1

            # Use previous solution as guess
            result = optimize.root(
                COHS_sys_eqs, sol[-1],
                args=(Pres, m_co2_tot, mh2o_tot, ms_tot, dfmq, temp, flag),
                method='lm',
                options={'maxiter': 1000000, 'xtol': 5.0e-16, 'ftol': 5.0e-16}
            )

            # Optional GC check (left in place)
            fO2 = O2_fugacity2(Pres, dfmq, temp)
            gc_check = result['x'][4] / (fO2 * np.exp(47457 / temp + 0.136))

            sol.append(result['x'])
            func_eval = np.abs(result['fun'])

            if (result['success'] is False) or (len(func_eval[func_eval > 1e-4]) != 0):
                print(result)
                print(result['fun'])
                print(Pres)
                exit()

        press.append(Pres)

    sol = np.array(sol)
    return press, sol

# ----------------------------
# Output writer
# ----------------------------

def writeplot(filename, pres, sol, xxco2tot, xxh2otot, title):
    """
    Write CSV of solutions vs pressure.
    Also computes a single 'mmw' based on the *last* row's degassed vector
    (matches original logic).
    """
    # Compute 'mmw' from the last solution row and normalize by last pressure
    mmw = 0.0
    for i in range(len(mass)):
        mmw += sol[-1, i + 2] * mass[i]
    mmw /= pres[-1]

    # Write CSV to the selected output folder
    path = f'./{outfile[fileno]}/{filename}.csv'
    with open(path, 'w+') as file:
        file.writelines('P,mmw,mfco2,mfh2o,mfs,mfh2,'
                        'pco2,ph2o,pch4,pco,ph2,pso2,ph2s,ps2,alphagas\n')
        for i in range(len(pres)):
            file.writelines(
                f'{pres[i]},{mmw/1e3},'
                f'{sol[i,0]},{sol[i,1]},{sol[i,2]},{sol[i,3]},'
                f'{sol[i,4]},{sol[i,5]},{sol[i,6]},{sol[i,7]},'
                f'{sol[i,8]},{sol[i,9]},{sol[i,10]},{sol[i,11]},{sol[i,12]}\n'
            )
    # Note: Plotting code intentionally left commented out (unchanged from original).

# ----------------------------
# Parameter sweeps (kept identical)
# ----------------------------

# Smaller grids than the commented-out block above (as in original)
wh2o = np.logspace(-5, -1, 10)[::-1]
wco2 = np.logspace(-5, -2, 10)[::-1]
ws   = np.logspace(-4, -3, 10)[::-1]
fo2  = np.arange(0, 5, 1)
Temp = np.array([1573])

# Continuation seed (file name) across nested loops
x = 'na'

for t in range(len(Temp)):
    for i in range(len(fo2)):
        for j in range(len(wh2o)):
            for k in range(len(wco2)):
                for l in range(len(ws)):
                    print(f'{fo2[i]}_{np.log10(wh2o[j])}_{np.log10(wco2[k])}_{np.log10(ws[l])}')
                    print(x)

                    outdir = f'./{outfile[fileno]}'
                    outcsv = f'{fo2[i]}_{np.log10(wh2o[j])}_{np.log10(wco2[k])}_{np.log10(ws[l])}.csv'
                    outpath = f'{outdir}/{outcsv}'

                    # Skip if output already exists and newrun==0 (unchanged behavior)
                    if (newrun == 0) and os.path.exists(outpath):
                        print('continue')
                    else:
                        pres, sol = COHS_sys_solver(
                            Temp[t], wco2[k], wh2o[j], ws[l], fo2[i], contfile=x
                        )
                        title = (f'Temp = {Temp[t]}, fO2 = FMQ {fo2[i]} , '
                                 f'H2O = {wh2o[j]*1e6} ppm, CO2 = {wco2[k]*1e6} ppm, '
                                 f'S = {ws[l]*1e6} ppm')
                        writeplot(
                            f'{fo2[i]}_{np.log10(wh2o[j])}_{np.log10(wco2[k])}_{np.log10(ws[l])}',
                            pres, sol, wco2[k], wh2o[j], title=title
                        )

                    # Update continuation file name after each inner iteration (same logic)
                    x = outcsv
                x = f'{fo2[i]}_{np.log10(wh2o[j])}_{np.log10(wco2[k])}_{np.log10(ws[0])}.csv'
            x = f'{fo2[i]}_{np.log10(wh2o[j])}_{np.log10(wco2[0])}_{np.log10(ws[0])}.csv'
        x = f'{fo2[i]}_{np.log10(wh2o[0])}_{np.log10(wco2[0])}_{np.log10(ws[0])}.csv'
    x = f'{fo2[0]}_{np.log10(wh2o[0])}_{np.log10(wco2[0])}_{np.log10(ws[0])}.csv'

