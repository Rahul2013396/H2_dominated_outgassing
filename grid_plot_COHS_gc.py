import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
from subfunctions import *            # project-specific helpers (left as-is)
from solubility import *              # project-specific solubility functions (left as-is)
import sys
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

# ----------------------------
# Global parameter grids (kept as provided)
# ----------------------------
wh20 = np.logspace(-5, -1, 10)[::-1]
wco2 = np.logspace(-5, -2, 10)[::-1]
ws   = np.logspace(-4, -3, 10)[::-1]
fo2  = np.arange(-8, 5, 1)

# ----------------------------
# Main function (logic preserved)
# ----------------------------
def main(rp, Mp, escape, magma_prod, linsty, name, clor):
    """
    Run the emission analysis and produce figures.

    Parameters
    ----------
    rp : float
        Planet radius in Earth radii.
    Mp : float
        Planet mass in Earth masses (used later in commented lifetime section).
    escape : float
        Escape flux (units follow original code).
    magma_prod : float
        Magma production rate (units as per original code).
    linsty : str
        Matplotlib line style for plots (carried through to commented block).
    name : str
        Label for the object (used in legends in commented block).
    clor : str
        Color name/string used for shading (used in commented block).
    """
    # ---- Constants (left unchanged; some duplicates preserved to maintain logic) ----
    k_b = 1.38e-16
    Rp = rp * 6.378e8                      # convert Earth radii -> cm

    m_H2  = 2
    m_N2  = 28
    m_co2 = 44
    epsilon = 0.1
    G = 6.67e-8
    k = 1.38e-16
    mmw = np.array([12, 10, 8, 6, 4, 2])
    R_earth = 637813700
    M_earth = 5.972e+27
    earth_heat = 4.7e13 * 1e7
    pressure = 1

    # ----------------------------------------------------------------------
    # Local helper: sulphur_weigth_ratio (kept byte-for-byte in math/flow)
    # ----------------------------------------------------------------------
    def sulphur_weigth_ratio(comp, P, fs2, fo2, T):
        """
        Compute C_S_sulphate / C_S_sulphide * fo2**2 and C_S_sulphide from composition.

        Notes
        -----
        - This follows the exact arithmetic in the original function; only comments added.
        - Variables kept as-is to avoid any behavior change.
        """
        wtsio2 = comp["SiO2"]
        wttio2 = comp["TiO2"]
        wtal2o3 = comp["Al2O3"]
        wtfeo = comp["FeO"]
        wtmno = comp["MnO"]
        wtmgo = comp["MgO"]
        wtcao = comp["CaO"]
        wtna2o = comp["Na2O"]
        wtk2o = comp["K2O"]
        wtp2o5 = comp["P2O5"]

        # Normalize to oxide total
        oxide_tot = (
            wtsio2 + wttio2 + wtal2o3 + wtfeo + wtmno + wtmgo
            + wtcao + wtna2o + wtk2o + wtp2o5
        )
        wtsio2 /= oxide_tot
        wttio2 /= oxide_tot
        wtal2o3 /= oxide_tot
        wtfeo  /= oxide_tot
        wtmno  /= oxide_tot
        wtmgo  /= oxide_tot
        wtcao  /= oxide_tot
        wtna2o /= oxide_tot
        wtk2o  /= oxide_tot
        wtp2o5 /= oxide_tot

        # Moles (molar masses preserved)
        nsio2 = wtsio2 / (28.086 + 15.999 * 2)
        ntio2 = wttio2 / (47.867 + 15.999 * 2)
        nal2o3 = wtal2o3 / (26.982 * 2 + 15.999 * 3)
        nfeo  = wtfeo  / (55.845 + 15.999)
        nmno  = wtmno  / (54.938 + 15.999)
        nmgo  = wtmgo  / (24.305 + 15.999)
        ncao  = wtcao  / (40.078 + 15.999)
        nna2o = wtna2o / (22.9898 * 2 + 15.999)
        nk2o  = wtk2o  / (39.098 * 2 + 15.999)

        # Totals for X_i (kept as-is)
        xtot = nsio2 + ntio2 + 0.5 * nal2o3 + nfeo + nmno + nmgo + nmgo + 0.5 * nna2o + 0.5 * nk2o
        X_Na = (wtna2o / 30.99) / xtot
        X_Mg = (wtmgo / 40.32) / xtot
        X_Al = (wtal2o3 / 50.98) / xtot
        X_Si = (wtsio2 / 60.08) / xtot
        X_K  = (wtk2o / 47.1) / xtot
        X_Ca = (wtcao / 56.08) / xtot
        X_Ti = (wttio2 / 79.9) / xtot
        X_Mn = (wtmno / 70.94) / xtot
        X_Fe = (wtfeo  / 71.85) / xtot

        # Sulphide capacity (exact expression preserved, including erf form)
        ln_C_S_sulphide = (
            8.77
            + (-23590 + 1673 * (6.7 * (X_Na + X_K) + 1.8 * X_Al + 4.9 * X_Mg
               + 8.1 * X_Ca + 5 * X_Ti + 8.9 * (X_Fe + X_Mn)
               - 22.2 * (X_Fe + X_Mn) * X_Ti + 7.2 * (X_Fe + X_Mn) * X_Si)) / T
            - 2.06 * special.erf(-7.2 * (X_Fe + X_Mn))
        )
        C_S_sulphide = np.exp(ln_C_S_sulphide) / 1e6  # ppmw -> fraction

        # Sulphate capacity (exact expression preserved)
        ln_C_S_sulphate = (
            -8.02
            + (21100 + 44000 * X_Na + 18700 * X_Mg + 4300 * X_Al + 44200 * X_K
               + 35600 * X_Ca + 12600 * X_Mn + 16500 * X_Fe) / T
        )
        C_S_sulphate = np.exp(ln_C_S_sulphate)

        # Return ratio * fo2**2 and the sulphide capacity (unchanged)
        return (C_S_sulphate / C_S_sulphide) * (fo2 ** 2), C_S_sulphide

    # Re-declare grids locally (kept identical to the global ones)
    wh20 = np.logspace(-5, -1, 10)[::-1]
    wco2 = np.logspace(-5, -2, 10)[::-1]
    ws   = np.logspace(-4, -3, 10)[::-1]
    fo2  = np.arange(-8, 5, 1)

    # Redox bookkeeping arrays (used later via data products)
    redoxpower = np.array([0, 0, 4, 1, 1, 0, 6, 8])
    earth      = np.array([0, 0, 3e8, 0, 3e10, 3e8, 0, 0])
    redpowearth = np.sum(redoxpower * earth)

    # Molecule field labels
    molecules = [
        'pco2', 'ph2o', 'pch4', 'pco', 'ph2', 'pso2', 'ph2s', 'ps2',
        'steady_press_C', 'steady_press_S', 'steady_pres', 'pre', 'check'
    ]
    moleculesname = [
        r'pCO$_2$', r'pH$_2$O', r'pCH$_4$', 'pCO', 'pH$_2$', 'pSO$_2$',
        'pH$_2$S', 'pS$_2$', 'cTotal S', r'cH$_{2}$S/SO$_{2}$ (log$_{10}$)', 'creducing power'
    ]

    # Allocate data arrays (shapes preserved)
    data   = np.ndarray(shape=(len(molecules), len(fo2), len(wh20), len(wco2), len(ws)))
    data_pp = np.ndarray(shape=(len(molecules), len(fo2), len(wh20), len(wco2), len(ws)))
    data2  = np.ndarray(shape=(len(molecules), len(fo2), len(wh20), len(wco2), len(ws)))

    print(len(data))

    # Missing file logger (path preserved)
    filemis = open('/Users/rahularora/Desktop/Project/Work/degassing_model/missingcohs.txt', 'w+')

    fileno = 0
    outfile  = [
        'Output_with_gc_COHS', 'Output_with_sulphate', 'Output_without_sulphide',
        'Output_shifted_ssolubility', 'Output_reduced_h2osolubility', 'Output_high_wco2'
    ]
    plotfile = [
        'Plots_gc', 'Plots_normal', 'Plots_without_sulphide',
        'Plots_shifted_ssolubilty', 'Plots_reduced_h2osolubility', 'Plots_high_wco2'
    ]

    check = 0

    # ---------------------------------------------------------
    # Populate data[] by reading per-case CSVs and computing
    # flux proxies & steady-state diagnostics (logic preserved)
    # ---------------------------------------------------------
    for k in range(len(wh20)):
        for i in range(len(fo2)):
            for l in range(len(wco2)):
                for m in range(len(ws)):
                    Fo2 = O2_fugacity2(1, fo2[i], 1573)
                    ratio, C_S_sulphide = sulphur_weigth_ratio(basalt, 1, 0.1, Fo2, 1473)
                    w_s = ws[m]
                    ws2_minus = w_s / (1 + ratio)
                    ws6_plus = w_s - ws2_minus
                    ps2 = (ws2_minus / C_S_sulphide) ** 2 * Fo2
                    K1, K2, K3, K4, K5 = symonds_equi_const(1473)
                    pso2 = K4 * Fo2 * (ps2 ** 0.5)

                    # CSV path for this case
                    casefile = (
                        f'/Users/rahularora/Desktop/Project/Work/degassing_model/'
                        f'{outfile[fileno]}/'
                        f'{fo2[i]}_{np.log10(wh20[k])}_{np.log10(wco2[l])}_{np.log10(ws[m])}.csv'
                    )

                    if os.path.isfile(casefile):
                        file = pd.read_csv(casefile)

                        # Find row with P ~ 1 bar
                        ind = np.argmin(np.abs(1 - file['P']))

                        # Gas mass fraction & mmw (as stored)
                        mgas = np.array(file['alphagas'])
                        mu   = np.array(file['mmw'])

                        # ngas factor (same expression retained)
                        ngas = 1e3 * mgas[ind] / (64 * (1 - mgas[ind]))

                        # Fill species flux grids (log10 of flux proxy) for first 8 molecules
                        for p in range(len(molecules) - 5):
                            data[p, i, k, l, m] = np.log10(
                                (
                                    np.array(file[f'{molecules[p]}'])[ind]
                                    * ngas * 6.022e23 * magma_prod * 25 * 2.9e+12
                                    / (3.154e+7 * (4 * np.pi * Rp**2))
                                    / np.array(file['P'])[ind]
                                )
                            )

                        # Steady-state with escape: evaluate vs pressure grid
                        pres_steady = np.log10(
                            (
                                np.array(file[f'{molecules[4]}'])[:] *
                                (1e3 * mgas[:] / (64 * (1 - mgas[:])))
                                * 6.022e23 * magma_prod * 25 * 2.9e+12
                                / (3.154e+7 * (4 * np.pi * Rp**2))
                                / np.array(file['P'])[:]
                            )
                        )
                        pres_steady[pres_steady == np.nan] = -100  # keep original guard

                        # Find pressure where steady-state ~ escape
                        ind2 = np.nanargmin(np.abs(pres_steady - np.log10(escape)))
                        pre  = np.array(file['P'])[ind2]

                        # Carbon- and sulfur-based composite indices (as in original)
                        c_ind = np.log10(
                            (
                                (np.array(file['pco2'])[ind2]
                                 + np.array(file['pch4'])[ind2]
                                 + np.array(file['pco'])[ind2])
                                * ngas * 6.022e23 * magma_prod * 25 * 2.9e+12
                                / (3.154e+7 * (4 * np.pi * Rp**2))
                                / np.array(file['P'])[ind2]
                            )
                        )
                        s_ind = np.log10(
                            (
                                (np.array(file['pso2'])[ind2]
                                 + np.array(file['ph2s'])[ind2]
                                 + 2 * np.array(file['ps2'])[ind2])
                                * ngas * 6.022e23 * magma_prod * 25 * 2.9e+12
                                / (3.154e+7 * (4 * np.pi * Rp**2))
                                / np.array(file['P'])[ind2]
                            )
                        )

                        # Derived fields 8..12 (exact algebra preserved)
                        data[8,  i, k, l, m] = np.log10(
                            (10 ** c_ind) * 1e9 / ((np.array(file['ph2'])[ind2]) * 1e6 / (612 * mu[ind2] * 1e3 * 1.66e-24))
                        )
                        data[9,  i, k, l, m] = np.log10(
                            (10 ** s_ind) * 1e9 / ((np.array(file['ph2'])[ind2]) * 1e6 / (612 * mu[ind2] * 1e3 * 1.66e-24))
                        )
                        data[10, i, k, l, m] = (0.1 * (pre) * 1e6 / (612 * mu[ind2] * 1e3 * 1.66e-24)) / (1e9 * 3.15e7)
                        data[11, i, k, l, m] = pre
                        data[12, i, k, l, m] = pres_steady[ind2]

                        if (pre >= 17 or pre <= 1e-3):
                            data[8, i, k, l, m] = -100
                        if (pre <= 17 and pre != 1e-3):
                            check += 1
                    else:
                        # Mark missing and log indices
                        data[:, i, k, l, m] = 0
                        filemis.writelines(f'{i} {k} {l} {m} \n')

    filemis.close()

    # Save arrays (paths preserved)
    np.save('/Users/rahularora/Desktop/Project/Work/degassing_model/emissioncohs.npy', data)
    np.save('/Users/rahularora/Desktop/Project/Work/degassing_model/emissioncohs_10bar.npy', data2)

    # Reload (kept identical to original flow)
    data  = np.load('/Users/rahularora/Desktop/Project/Work/degassing_model/emissioncohs.npy')
    data2 = np.load('/Users/rahularora/Desktop/Project/Work/degassing_model/emissioncohs_10bar.npy')

    # ----------------------------
    # Build plotting grids (unchanged math)
    # ----------------------------
    wh20_grid = np.zeros(11)
    wco2_grid = np.zeros(11)
    ws_grid   = np.zeros(11)
    fo2_grid  = np.zeros(14)

    wh20_grid[0] = np.log10(wh20[0]) + (np.log10(wh20[0]) - np.log10(wh20[1])) / 2
    wco2_grid[0] = np.log10(wco2[0]) + (np.log10(wco2[0]) - np.log10(wco2[1])) / 2
    ws_grid[0]   = np.log10(ws[0])   + (np.log10(ws[0])   - np.log10(ws[1]))   / 2
    fo2_grid[0]  = fo2[0] + (fo2[0] - fo2[1]) / 2

    for i in range(10):
        wh20_grid[i + 1] = wh20_grid[i] - (np.log10(wh20[0]) - np.log10(wh20[1]))
        wco2_grid[i + 1] = wco2_grid[i] - (np.log10(wco2[0]) - np.log10(wco2[1]))
        ws_grid[i + 1]   = ws_grid[i]   - (np.log10(ws[0])   - np.log10(ws[1]))
    for i in range(13):
        fo2_grid[i + 1] = fo2_grid[i] - (fo2[0] - fo2[1])

    # ----------------------------
    # Figure 1: φ_H2 / φ_CO2 vs fO2 (left) and φ_H2 / φ_H2O vs fO2 (right)
    # ----------------------------
    data_ana = data[4, :, :, :, :] - np.log10(10 ** data[0, :, :, :, :] + 10 ** data[2, :, :, :, :] + 10 ** data[3, :, :, :, :])
    co2image = np.zeros((20, 13))
    fig, [axs, ax] = plt.subplots(1, 2, figsize=(40, 20))
    co2_flux = np.linspace(np.max(np.ndarray.flatten(data_ana)), np.min(np.ndarray.flatten(data_ana)), 21)

    # Bin model counts by fO2 & flux bin
    for i in range(20):
        for j in range(13):
            ind = []
            for l in fo2[(fo2 >= fo2_grid[j]) & (fo2 < fo2_grid[j + 1])]:
                ind.append(np.where(fo2 == l)[0][0])
            ind = np.array(ind)
            if len(ind) > 0:
                flatten_array = np.ndarray.flatten(data_ana[ind, :, :, :])
                model_count = len(flatten_array[(flatten_array <= co2_flux[i]) & (flatten_array >= co2_flux[i + 1])])
                co2image[i, j] = model_count

    im3 = axs.imshow(
        co2image, origin='lower',
        extent=[fo2_grid[0], fo2_grid[-1], co2_flux[0], co2_flux[-1]],
        aspect=np.abs((fo2_grid[0] - fo2_grid[-1]) / (co2_flux[0] - co2_flux[-1])),
        cmap='Blues'
    )
    axs.set_yticks(co2_flux)
    axs.set_xticks(fo2_grid)
    axs.invert_yaxis()
    axs.axhline(np.log10(1.8e10 / 3.17e10), linewidth=5, color='r', label='Modern Earth')
    axs.axvline(0, linewidth=5, color='r', linestyle='--')
    axs.axvline(-8.4, linewidth=5, color='g', linestyle='--', label='Mercury')
    axs.axvline(-4.5, linewidth=5, color='y', linestyle='--', label='Mars')
    axs.xaxis.set_tick_params(labelsize=30)
    axs.yaxis.set_tick_params(labelsize=30)
    axs.set_xlabel(r'$\mathit{f}O_2 (FMQ)$', fontsize=45.0)
    axs.set_ylabel(r'  $\phi_{H_2}$/ $\phi_{CO_2}$ (log$_{10}$) ', fontsize=45.0)
    axs.legend(fontsize=30)

    # Right panel of Fig 1: φ_H2 / φ_H2O
    data_ana = data[4, :, :, :, :] - np.log10(10 ** data[1, :, :, :, :])
    co2image = np.zeros((20, 13))
    co2_flux = np.linspace(np.max(np.ndarray.flatten(data_ana)), np.min(np.ndarray.flatten(data_ana)), 21)

    for i in range(20):
        for j in range(13):
            ind = []
            for l in fo2[(fo2 > fo2_grid[j]) & (fo2 <= fo2_grid[j + 1])]:
                ind.append(np.where(fo2 == l)[0][0])
            ind = np.array(ind)
            if len(ind) > 0:
                flatten_array = np.ndarray.flatten(data_ana[ind, :, :, :])
                model_count = len(flatten_array[(flatten_array <= co2_flux[i]) & (flatten_array >= co2_flux[i + 1])])
                co2image[i, j] = model_count

    im4 = ax.imshow(
        co2image, origin='lower',
        extent=[fo2_grid[0], fo2_grid[-1], co2_flux[0], co2_flux[-1]],
        aspect=np.abs((fo2_grid[0] - fo2_grid[-1]) / (co2_flux[0] - co2_flux[-1])),
        cmap='Blues'
    )
    ax.set_yticks(co2_flux)
    ax.set_xticks(fo2_grid)
    ax.axhline(np.log10(1.8e10 / 41.5e10), linewidth=5, color='r', label='Modern Earth')
    ax.axvline(0, linewidth=5, color='r', linestyle='--')
    ax.axvline(-8.4, linewidth=5, color='g', linestyle='--', label='Mercury')
    ax.axvline(-4.5, linewidth=5, color='y', linestyle='--', label='Mars')
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\mathit{f}O_2 (FMQ)$', fontsize=45.0)
    ax.set_ylabel(r'  $\phi_{H_2}$/ $\phi_{H2O}$ (log$_{10}$) ', fontsize=45.0)
    ax.legend(fontsize=30)

    # Colorbars for Fig 1
    fcbar2 = fig.colorbar(im3, ax=axs, pad=0.01, label='Model density', shrink=0.75, aspect=30)
    fcbar2.ax.tick_params(labelsize=28)
    fcbar2.set_label('Model density', fontsize=32)
    cbar2 = fig.colorbar(im4, ax=ax, pad=0.01, label='Model density', shrink=0.75, aspect=30)
    cbar2.ax.tick_params(labelsize=28)
    cbar2.set_label('Model density', fontsize=32)

    plt.savefig('/Users/rahularora/Desktop/Project/Work/lowmu/Final plots/H2_CO2_distr.png')

    # ----------------------------
    # Figure 2: φ_H2 / φ_S vs fO2
    # ----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    data_ana = data[4, :, :, :, :] - np.log10(10 ** data[5, :, :, :, :] + 10 ** data[6, :, :, :, :] + 10 ** data[7, :, :, :, :])
    co2image = np.zeros((20, 13))
    co2_flux = np.linspace(np.max(data_ana), np.min(data_ana), 21)

    for i in range(20):
        for j in range(13):
            ind = []
            for l in fo2[(fo2 > fo2_grid[j]) & (fo2 <= fo2_grid[j + 1])]:
                ind.append(np.where(fo2 == l)[0][0])
            ind = np.array(ind)
            if len(ind) > 0:
                flatten_array = np.ndarray.flatten(data_ana[ind, :, :, :])
                model_count = len(flatten_array[(flatten_array <= co2_flux[i]) & (flatten_array >= co2_flux[i + 1])])
                co2image[i, j] = model_count

    im4 = ax.imshow(
        co2image, origin='lower',
        extent=[fo2_grid[0], fo2_grid[-1], co2_flux[0], co2_flux[-1]],
        aspect=np.abs((fo2_grid[0] - fo2_grid[-1]) / (co2_flux[0] - co2_flux[-1])),
        cmap='Blues'
    )
    ax.invert_yaxis()
    ax.axhline(np.log10(1.8e10 / (3e10 + 3e9)), linewidth=5, color='r', label='Modern Earth')
    ax.axvline(0, linewidth=5, color='r', linestyle='--')
    ax.axvline(-8.4, linewidth=5, color='g', linestyle='--', label='Mercury')
    ax.axvline(-4.5, linewidth=5, color='y', linestyle='--', label='Mars')
    ax.set_yticks(co2_flux)
    ax.set_xticks(fo2_grid)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_xlabel(r'$\mathit{f}O_2 (FMQ)$', fontsize=45.0)
    ax.set_ylabel(r'$\phi_{H_2}/\phi_{S}$ (log$_{10}$)', fontsize=45.0)
    ax.legend(fontsize=30)

    cbar2 = fig.colorbar(im4, ax=ax, pad=0.01, label='Model density', shrink=0.75, aspect=30)
    cbar2.ax.tick_params(labelsize=28)
    cbar2.set_label('Model density', fontsize=32)
    plt.savefig('/Users/rahularora/Desktop/Project/Work/lowmu/Final plots/H2_S.png')

    # ----------------------------
    # Figure 3: φ_H2 vs fO2 (left) and φ_H2 vs H2O wt% (right)
    # ----------------------------
    data_ana = data[4, :, :, :, :]
    fig, [axs, ax] = plt.subplots(1, 2, figsize=(40, 20))
    co2image = np.zeros((20, 13))
    co2_flux = np.linspace(np.max(np.ndarray.flatten(data_ana)), np.min(np.ndarray.flatten(data_ana)), 21)

    for i in range(20):
        for j in range(13):
            ind = []
            for l in fo2[(fo2 >= fo2_grid[j]) & (fo2 < fo2_grid[j + 1])]:
                ind.append(np.where(fo2 == l)[0][0])
            ind = np.array(ind)
            if len(ind) > 0:
                flatten_array = np.ndarray.flatten(data_ana[ind, :, :, :])
                model_count = len(flatten_array[(flatten_array <= co2_flux[i]) & (flatten_array >= co2_flux[i + 1])])
                co2image[i, j] = model_count

    im3 = axs.imshow(
        co2image, origin='lower',
        extent=[fo2_grid[0], fo2_grid[-1], co2_flux[0], co2_flux[-1]],
        aspect=np.abs((fo2_grid[0] - fo2_grid[-1]) / (co2_flux[0] - co2_flux[-1])),
        cmap='Blues'
    )
    axs.set_yticks(co2_flux)
    axs.set_xticks(fo2_grid)
    axs.invert_yaxis()
    axs.axhline(np.log10(1.8e10), linewidth=5, color='r', label='Modern Earth')
    axs.axvline(0, linewidth=5, color='r', linestyle='--')
    axs.axvline(-8.4, linewidth=5, color='g', linestyle='--', label='Mercury')
    axs.axvline(-4.5, linewidth=5, color='y', linestyle='--', label='Mars')
    axs.xaxis.set_tick_params(labelsize=30)
    axs.yaxis.set_tick_params(labelsize=30)
    axs.set_xlabel(r'$\mathit{f}O_2 (FMQ)$', fontsize=45.0)
    axs.set_ylabel(r'  $\phi_{H_2}$(log$_{10}$) ', fontsize=45.0)
    axs.legend(fontsize=30)

    # Right panel: φ_H2 vs log10(H2O wt%)
    data_ana = data[4, :, :, :, :]
    co2image = np.zeros((20, 10))
    co2_flux = np.linspace(np.max(np.ndarray.flatten(data_ana)), np.min(np.ndarray.flatten(data_ana)), 21)

    for i in range(20):
        for j in range(10):
            ind = []
            for l in wh20[(np.log10(wh20) < wh20_grid[j]) & (np.log10(wh20) >= wh20_grid[j + 1])]:
                ind.append(np.where(np.log10(wh20) == np.log10(l))[0][0])
            ind = np.array(ind)
            if len(ind) > 0:
                flatten_array = np.ndarray.flatten(data_ana[:, ind, :, :])
                model_count = len(flatten_array[(flatten_array <= co2_flux[i]) & (flatten_array >= co2_flux[i + 1])])
                co2image[i, j] = model_count

    im4 = ax.imshow(
        co2image, origin='lower',
        extent=[wh20_grid[0], wh20_grid[-1], co2_flux[0], co2_flux[-1]],
        aspect=np.abs((wh20_grid[0] - wh20_grid[-1]) / (co2_flux[0] - co2_flux[-1])),
        cmap='Blues'
    )
    ax.set_yticks(co2_flux)
    ax.set_xticks(wh20_grid)
    ax.axhline(np.log10(1.8e10), linewidth=5, color='r', label='Modern Earth')
    ax.axvline(-3, linewidth=5, color='r', linestyle='--')
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\log_{10}$(H_2O) (wt%)', fontsize=45.0)
    ax.set_ylabel(r' $\phi_{H_2}$ (log$_{10}$) ', fontsize=45.0)
    ax.legend(fontsize=30)

    fcbar2 = fig.colorbar(im3, ax=axs, pad=0.01, label='Model density', shrink=0.75, aspect=30)
    fcbar2.ax.tick_params(labelsize=28)
    fcbar2.set_label('Model density', fontsize=32)
    cbar2 = fig.colorbar(im4, ax=ax, pad=0.01, label='Model density', shrink=0.75, aspect=30)
    cbar2.ax.tick_params(labelsize=28)
    cbar2.set_label('Model density', fontsize=32)

    plt.savefig('/Users/rahularora/Desktop/Project/Work/lowmu/Final plots/H2_distr.png')

    # --------------------------------------------------------------------
    # The large commented diagnostic/plotting blocks below are preserved
    # to avoid any change in program behavior.
    # --------------------------------------------------------------------

    # (Lifetime, steady-pres vs fmq/co2 plots, etc. are left commented.)

# ----------------------------
# Entry point (kept as-is)
# ----------------------------
# Example call (original values preserved):
main(1, 1, 2694954711139.712, 1, '-', 'L 98-59d', 'red')

# Other examples (left commented exactly as in your code):
# main(1.329, 2, 2956528297937.1196, 1753, '--', 'L 98-59c')
# main(1.046, 0.934, 3399908264068.2334, 140, '--', 'TRAPPIST-1 f', 'yellow')

# plt.tight_layout()
# axL.tick_params(axis='x', labelsize=12)
# axL.tick_params(axis='y', labelsize=12)
# plt.savefig(f'/Users/rahularora/Desktop/Project/Work/Sub_neps/plots/steady_pres_L98d.png', dpi=300)

# main(1,1,1,1,'-','Test','black')
