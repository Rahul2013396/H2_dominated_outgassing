
"""
Outgassing Requirement Module

Purpose
-------
This script computes outgassing requirements for maintaining H₂-dominated atmospheres
on rocky exoplanets. It compares volcanic supply rates to atmospheric escape rates
under a range of planetary and stellar parameters.


"""

# -------------------- Imports --------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
epsilon = 0.1 # escape efficieny factor
G = 6.67e-8 # gravitatinal constant (cgs)
k = 1.38e-16 # boltzmann constant (cgs)
mmw = np.array([12,10,8,6,4,2]) # mean molecular weight (amu)
R_earth = 637813700 # Radius of Earth (cm)
M_earth =5.972e+27 # Mass of earth (g)
planet_name = ['Earth','Venus','Mars ','TRAPPIST-1 b','TRAPPIST-1 c','TRAPPIST-1 d','TRAPPIST-1 e','TRAPPIST-1 f','TRAPPIST-1 g','TRAPPIST-1 h','L 98-59b','L 98-59c','L 98-59d','LP 791-18d']


data = np.genfromtxt('./Table1_data.csv',delimiter=',',skip_header=24)
data2 = np.genfromtxt('./Table1_data.csv',delimiter=',', dtype='unicode',skip_header=24)

Rp = data[:,4]
Mp = data[:,6]
a = data[:,3]*1.496e+13
star_xray = data[:,15]
star_EUV = data[:,16]
XUV_star = (star_xray+star_EUV)/((a/1.496e+13)**2)
T_plan =data[:,10]
P_i = data[:,2]
e = data[:,8]
Ms = data[:,13]


def energy_lim_esc(R,M,FXUV,K,a):
    """
    energy_lim_esc

    Parameters
    ----------
    See function body for details.

    Returns
    -------
    See function body for details.

    Notes
    -----
    Added automatically during cleanup; no functional changes.
    """
    m = epsilon*np.pi*((R*R_earth)**3)*FXUV*(a**2)/(K*G*(M*M_earth))
    return m


def tidal_heat(P,Rp,e,J):
    """
    tidal_heat

    Parameters
    ----------
    See function body for details.

    Returns
    -------
    See function body for details.

    Notes
    -----
    Added automatically during cleanup; no functional changes.
    """
    heat = 3.4e25*(P**(-5))*(Rp**5)*((e/1e-2)**2)*J/1e-2
    return heat


def planet_effective_temperature(star_temp, star_radius, distance, albedo=0.3):
    """
    Calculate the planet's effective temperature as a function of distance.

    Parameters:
    star_temp (float): Star's effective temperature (K).
    star_radius (float): Star's radius (in meters).
    distance (numpy.ndarray): Distance of the planet from the star (in meters).
    albedo (float): Planet's albedo (default: 0.3).

    Returns:
    numpy.ndarray: Planet's effective temperature (K).
    """
    # Calculate the effective temperature
    temp = star_temp * np.sqrt((star_radius*6.96e8) / (2 * distance)) * np.sqrt(1 - albedo)
    return temp


def calculate_luminosity(mass):
    """
    Calculate the stellar luminosity using mass and temperature.
    
    Parameters:
    mass (float): Stellar mass in solar masses (M_sun).
    temperature (float): Stellar temperature in Kelvin.
    
    Returns:
    float: Stellar luminosity in terms of solar luminosity (L_sun).
    """
    # Approximation: L ~ M^3.5 (valid for main-sequence stars)
    mass = mass
    luminosity = mass ** 3.5
    return luminosity


def calculate_habitable_zone(mass):
    """
    Calculate the inner and outer edges of the habitable zone (HZ) based on stellar luminosity.
    
    Parameters:
    luminosity (float): Stellar luminosity in terms of the solar luminosity (L_sun).
    
    Returns:
    tuple: Inner and outer edges of the HZ in AU.
    """
    luminosity  = calculate_luminosity(mass)
    # Constants for the habitable zone boundaries
    S_inner = 0.95  # Solar flux received at the inner edge of the HZ
    S_outer = 1.37  # Solar flux received at the outer edge of the HZ
    
    # Calculate the distance in AU for the inner and outer edges of the HZ
    inner_edge = (luminosity / S_outer) ** 0.5
    outer_edge = (luminosity / S_inner) ** 0.5
    
    return inner_edge, outer_edge

# effect of mean molecular weight


sx = np.logspace(-2,1,1000)
earth_heat = 4.7e13 * 1e7


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_context("talk")
# Use dark background and set high-contrast colors for text/ticks/legend
# plt.style.use('dark_background')
# plt.rcParams.update({
#     'font.family': 'serif',
#     'figure.facecolor': 'black',
#     'axes.facecolor': 'black',
#     'savefig.facecolor': 'black',
#     'axes.edgecolor': 'white',
#     'axes.labelcolor': 'white',
#     'xtick.color': 'white',
#     'ytick.color': 'white',
#     'text.color': 'white',
#     'legend.facecolor': 'black',
#     'legend.edgecolor': 'white'
# })

# Use a bright palette for lines/patches so they show on black background
sns.set_context("talk")
plt.rcParams.update({'font.family': 'serif'})

# Colorblind-friendly palette
colors = sns.color_palette("colorblind")

plt.figure(figsize=(10, 7))

# Define the habitable zone (for simplicity, let's use some example values)
inner_hz = 0.086  # Inner boundary in AU
outer_hz = 0.224  # Outer boundary in AU

# Plot the habitable zone as a shaded region
plt.fill_betweenx([1e7, 1e19], inner_hz, outer_hz, color='gray', alpha=0.2, label='Habitable Zone')

for i in range(len(planet_name)):
    thirdlaw = (G * Ms[i] * 1.98e33 / (4 * np.pi**2))
    P = ((((sx * 1.496e13) ** 3) / thirdlaw) ** 0.5) * 1.15741e-5  # Convert to days
    XUV_a = XUV_star[i] * ((a[i] / 1.496e13) ** 2) / (sx ** 2)
    
    escape_rate = energy_lim_esc(Rp[i], Mp[i], XUV_a, 1, 1)  # g/s
    mol_escape_rate = escape_rate * 6.022e23 / 2 / (4 * np.pi * (Rp[i] * R_earth) ** 2)
    
    heat = tidal_heat(P, Rp[i], e[i], 1)
    planet_outgas_rate = (heat*1.8e10/earth_heat)/(Rp[i]**2)

    # Plot only L 98-59b and L 98-59d
    if i == 10:
        plt.plot(sx, mol_escape_rate, color=colors[0], label='Escape (L 98-59b)', linewidth=2)
        plt.plot(sx, planet_outgas_rate, '--', color=colors[0], label='Outgassing (L 98-59b)', linewidth=2)
        plt.scatter(a[i]/1.496e13, planet_outgas_rate[np.argmin(abs(sx - a[i]/1.496e13))],
                    color=colors[0], marker='o', edgecolor='black', zorder=5)
        plt.text(a[i]/1.496e13, 2e8, 'L 98-59b', ha='center', fontsize=14, color=colors[0])
        print(P_i[i], Rp[i], e[i])
        print((tidal_heat(P_i[i], Rp[i], e[i], 1)*1.8e10/earth_heat)/(Rp[i]**2),energy_lim_esc(Rp[i], Mp[i], XUV_star[i], 1, 1) * 6.022e23 / 2 / (4 * np.pi * (Rp[i] * R_earth) ** 2))

        # Shade region where outgassing > escape
        mask = planet_outgas_rate > mol_escape_rate
        plt.fill_between(sx, mol_escape_rate, planet_outgas_rate, where=mask,
                         color=colors[0], alpha=0.2, label='Outgassing > Escape (b)')

    if i == 12:
        plt.plot(sx, mol_escape_rate, color=colors[1], label='Escape (L 98-59d)', linewidth=2)
        plt.plot(sx, planet_outgas_rate, '--', color=colors[1], label='Outgassing (L 98-59d)', linewidth=2)
        plt.scatter(a[i]/1.496e13, planet_outgas_rate[np.argmin(abs(sx - a[i]/1.496e13))],
                    color=colors[1], marker='s', edgecolor='black', zorder=5)
        plt.text(a[i]/1.496e13, 2e8, 'L 98-59d', ha='center', fontsize=14, color=colors[1])

        # Shade region where outgassing > escape
        mask = planet_outgas_rate >= mol_escape_rate
        print(mask)
        plt.fill_between(sx, mol_escape_rate, planet_outgas_rate, where=mask,
                         color=colors[1], alpha=0.2, label='Outgassing > Escape (d)')

plt.xscale('log')
plt.yscale('log')
plt.ylim(1e7, 1e19)
plt.xlim(1e-2, 5)
plt.xlabel('Semi-major axis (AU)')
plt.ylabel(r'H$_2$ Outgassing & Escape flux (molecules cm$^{-2}$ s$^{-1}$)')
plt.legend(loc='upper right', fontsize=13)
plt.tight_layout()
plt.savefig('./orbital_dependence.png', dpi=300, facecolor=plt.gcf().get_facecolor())
plt.clf()


