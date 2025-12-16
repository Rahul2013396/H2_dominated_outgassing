

"""
volatile_inventory.py
"""

# -------------------- Imports --------------------
import matplotlib.pyplot as plt
import numpy as np


chi_H2o_max = 0.033
chi_H2o_min = 1000e-6
chi_H2o_mean = 1e-3

mantle_mass_frac = 1
m_H2 = 2
m_N2 = 28
m_co2 = 44
epsilon = 0.1
G = 6.67e-8
k = 1.38e-16
mmw = np.array([12,10,8,6,4,2])
R_earth = 637813700
M_earth =5.972e+27
earth_heat  = 4.7e13*1e7

chi_H2 = ((mmw - 28))/-26
xuv_flux = np.logspace(-1,4,6)*1.26e-2
T = 400
g=980
bh2= 1.46e19

data = np.genfromtxt('/Users/rahularora/Desktop/Project/Work/lowmu/PS_2025.01.29_12.51.35.csv',delimiter=',',skip_header=24)
data2 = np.genfromtxt('/Users/rahularora/Desktop/Project/Work/lowmu/PS_2025.01.29_12.51.35.csv',delimiter=',', dtype='unicode',skip_header=24)

planet_name = ['Earth','Venus','Mars','TRAPPIST-1 b','TRAPPIST-1 c','TRAPPIST-1 d','TRAPPIST-1 e','TRAPPIST-1 f','TRAPPIST-1 g','TRAPPIST-1 h','L 98-59b','L 98-59c','L 98-59d','LP 791-18d']

tableau20 = [(31, 119, 180),(255, 127, 14),(44, 160, 44),(214, 39, 40),(148, 103, 189),(140, 86, 75), (227, 119, 194),(127, 127, 127),(188, 189, 34),(23, 190, 207),\
(174, 199, 232),(255, 187, 120),(152, 223, 138),(255, 152, 150),(197, 176, 213),(196, 156, 148),(247, 182, 210),(199, 199, 199),(219, 219, 141),(158, 218, 229)] 
# 


# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
# Loop over collection
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

    

# Earth
Rp_earth = 1
Mp_earth = 1
Earth_a = 1
M_sun = 1
P_earth = 365
e_Earth = 0.03
Sun_xuv = 1.26e-2
Temp_sun = 5800
R_sun =1 

H2_pressure = np.array([1,10,100])*1e6
outgassing_eff  =np.logspace(0,-4,100)
#print(outgassing_eff)


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
    Added during code cleanup; logic unchanged.
    """
    m = epsilon*np.pi*((R*R_earth)**3)*FXUV*(a**2)/(K*G*(M*M_earth))
    return m


def diff_lim_esc(b,chi,T,mu,g):
    """
    diff_lim_esc

    Parameters
    ----------
    See function body for details.

    Returns
    -------
    See function body for details.

    Notes
    -----
    Added during code cleanup; logic unchanged.
    """
    H = k*T/(mu*1.66054e-24*g)
    Phi = b*chi/(H)
    return Phi



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
    Added during code cleanup; logic unchanged.
    """
    heat = 3.4e25*(P**(-5))*(Rp**5)*((e/1e-2)**2)*J/1e-2
    return heat

symbol = ['o','d','s','.']

flag = np.full(len(planet_name), False)
#flag = np.zeros(len(planet_name))
#print(flag)
time_high = np.zeros((len(planet_name),len(outgassing_eff)))
time_low = np.zeros((len(planet_name),len(outgassing_eff)))
time_mean_arr = np.zeros((len(planet_name),len(outgassing_eff)))

fig = plt.figure(figsize=(12,12))
ind = np.array([3,4,10,11,12])
# Loop over collection
for i in ind:
# Loop over collection
    for j in range(len(outgassing_eff)):
        Rp = data[i,4]
        Mp = data[i,6]
        M_star = data[i,13]
        e = data[i,8]
        a = data[i,3]*1.496e+13
        star_xray = data[i,15]
        star_EUV = data[i,16]
        #print(star_xray+star_EUV)
        XUV_star = (star_xray+star_EUV)/((a/1.496e+13)**2)
        Temp_star = data[i,11]
        R_star = data[i,12]
        melting_eff = 0.01
        period = data[i,2]
        T_plan =data[i,10]
        g = G*Mp*M_earth/((R_earth*Rp)**2)
        H2_escape_rate = energy_lim_esc(Rp,Mp,XUV_star,1,1) # g/s
        diff_esc = diff_lim_esc(bh2,chi_H2[1],T_plan,mmw[1],g)*2*(4*np.pi*(Rp*R_earth)**2)/6.022e23 
        
        
        heat = tidal_heat(period,Rp,e,1)
        planet_outgas_rate = (heat*1.8e10/(earth_heat*0.1))*(4*np.pi*(R_earth*Rp)**2)*outgassing_eff[j] /(Rp**2)
        print(Rp,Mp,heat/earth_heat,planet_outgas_rate/((4*np.pi*(R_earth*Rp)**2)/outgassing_eff[j]),H2_escape_rate*6.022e23/2/(4*np.pi*(R_earth*Rp)**2))
        print(planet_name[i],1/(planet_outgas_rate/(H2_escape_rate*6.022e23/2)),outgassing_eff[j])
# Conditional block
        if (planet_outgas_rate/outgassing_eff[j]/6.022e23>= H2_escape_rate/2):
            flag[i] = True

        time_max = (chi_H2o_max*1000*6.022e23/18)*(mantle_mass_frac*Mp*M_earth/1000)/planet_outgas_rate/3.15e7/1e9
        time_min = (chi_H2o_min*1000*6.022e23/18)*(mantle_mass_frac*Mp*M_earth/1000)/planet_outgas_rate/3.15e7/1e9
        time_mean = (chi_H2o_mean*1000*6.022e23/18)*(mantle_mass_frac*Mp*M_earth/1000)/planet_outgas_rate/3.15e7/1e9
        time_high[i,j] = time_max
        #print(time_max)
        time_low[i,j] = time_min
        time_mean_arr[i,j]  =time_mean
            
    flag2 = np.full(len(outgassing_eff), False)
# Conditional block
    if (planet_outgas_rate/outgassing_eff[-1]/6.022e23>= H2_escape_rate/2 ):
# Loop over collection
        for l in range(len(outgassing_eff)):
            if(planet_outgas_rate/outgassing_eff[-1]*outgassing_eff[l]/6.022e23>= H2_escape_rate/2 ):
                flag2[l] = True
        #if(j == 0):
        print(flag2)
        plt.plot(outgassing_eff[flag2],time_high[i][flag2],'-',color = tableau20[i], label = f'{planet_name[i]}' ,linewidth = 3)
        plt.plot(outgassing_eff[flag2], time_low[i][flag2],'--',color = tableau20[i],linewidth = 3)
        # else:    
        #     plt.plot(outgassing_eff[j],time_hig[],f'{symbol[0]}',color = tableau20[i],markersize = 15 )
        #     plt.plot(outgassing_eff[j], time_min,f'{symbol[1]}',color = tableau20[i],markersize = 15)

plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\eta_{og} \eta_{heat}$',fontsize = 31)
plt.ylabel('Lifetime of Hydrogen inventory (Gyr)',fontsize = 31)
plt.fill_between(outgassing_eff,1,13.6, facecolor='blue', alpha=.1)
plt.fill_between(outgassing_eff,1e-5,1, facecolor='yellow', alpha=.1)
plt.text(4e-4,6,'Geologically long',fontsize = 32)
plt.text(3e-3,2e-5,'Geologically short',fontsize = 32)

legend = plt.legend(fontsize =22)
# for text in legend.get_texts():
#     text.set_weight('bold')
plt.xticks(fontsize=29)
plt.yticks(fontsize=29)
plt.xlim(outgassing_eff[-1],outgassing_eff[0])
plt.ylim(1e-5,13.6)
#plt.xlim(1e-4,1)
plt.savefig('./inventory.png')
plt.clf()

