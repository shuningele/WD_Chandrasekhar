import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as sc
import matplotlib.pyplot as plt
import scipy
m_n = (sc.m_n+sc.m_p)/2# nucleon mass
m_e = sc.m_e#electron mass
Z_C = 2.
A_C = 4.
c = sc.c #speed of light in vacuum
wd_dens = 10**9 # kg/m^3

def epsilon_e(fermi_p):
# returns energy density due to a) Pauli exclusion principle at Fermi momentum, b) electron rest mass
    const = m_e**4*c**5/(np.pi**2*sc.hbar**3)
    act = const*scipy.integrate.quad(intg, 1, fermi_p/(m_e*c), points=0)[0]
    return act
    
def intg(u):
    return np.sqrt(u**2+1)*u**2

def ep_nuc(dens):
#returns rest mass energy of nucleons
    #n: electron number density
    return dens*c**2

# singular composition
def fermi_momentum(A, Z, mass_dens):
    return sc.hbar*(3*np.pi**2*mass_dens*Z/(m_n*A))**(1/3.)

# neglecting energy density due to electron rest mass
def epsilon(dens):
    ep_e = epsilon_e(fermi_momentum(A_C, Z_C, dens))
    return ep_nuc(dens)
    # +ep_e
# below is commented out: for the milestone, we split the cases into rel and non-rel only
#integrate below to find pressue for cases between non-rel and rel:
#def central_pressure(fermi_p):
#    const1 = m_e**4*c**5/(3*np.pi**2*sc.hbar**3)
#    return scipy.integrate.quad(intg2, 0,fermi_p/(m_e*c))[0]
    
#def intg2(v):
#    return np.exp(-0.5*(u**2+1))*u**4

# [for the milstone] rel and non-rel is only determined with the central pressure. ie. the three functions below is only run once for every central density

def press_rel(ep, A, Z):
    k_rel = (sc.hbar*c)/(12*np.pi**2)*(3*np.pi**2*Z/(A*m_n*c**2))**(4./3)
    return k_rel*ep**(4./3.)
    
def press_nr(ep, A, Z):
    k_non = (sc.hbar**2)/(15*np.pi**2*m_e)*(3*np.pi**2*Z/(A*m_n*c**2))**(5./3)
    return k_non*ep**(5./3.)
    
def p_rel_or_not(cent_dens):
    ep = epsilon(cent_dens)
    fermi_p = fermi_momentum(A_C, Z_C, cent_dens)
    if fermi_p > m_e*c:
        return (press_rel(ep, A_C, Z_C), "rel")
    else:
        return (press_nr(ep, A_C, Z_C), "non-rel")

def update_dens(re, pressure, A, Z):
    if re == "rel":
        k = sc.hbar*c/(12*np.pi**2)*(3*np.pi**2*Z/(A*m_n*c**2))**(4./3)
        return ((pressure/k)**(3./4))/c**2
    else:
        k = sc.hbar**2/(15*np.pi**2*m_e)*(3*np.pi**2*Z/(A*m_n*c**2))**(5./3)
        return ((pressure/k)**(3./5))/c**2
xyzs = []
def mass_press(r, state, re, A, Z):
    m, p = state
    d = update_dens(re, p, A, Z)
    ep = epsilon(d)
    dmdr = 4*np.pi*r**2*ep/c**2
    #with GR correction:
    nt = -sc.G*ep*m/(c*r)**2
    # x = 1+p/ep
    # y = 1+4*np.pi*r**3*p/(m*c**2)
    # z = 1/(1-2*sc.G*m/(c**2*r))
    # valid = not np.isnan(y) and not np.isnan(x) and not np.isnan(z)
    # finite = np.isfinite(y) and np.isfinite(x) and np.isfinite(z)
    # if valid and finite:
    #     xyzs.append([x,y,z])
    #     gr_corr = x*y*z
    #     print('valid:', gr_corr)
    # else:
    #     gr_corr = xyzs[-1][0]*xyzs[-1][1]*xyzs[-1][2]
    #     print("last_valid:",gr_corr)
    gr_corr=1.
    dpdr = nt*gr_corr
    # plt.hist(xyzs)
    # plt.show()
    
    return [dmdr, dpdr]
    
def integrate(rs, func, A, Z, p_start, m_start, re):
    ts = np.linspace(rs[0], rs[-1], 20)
    results = solve_ivp(func,[rs[0], rs[-1]], [m_start, p_start], args=(re, A, Z), t_eval=ts)
    # , max_step=0.1
    return (results.t, results.y[0,:], results.y[1,:])

def find_zero_cross(A, Z, d0):
    p0,re = p_rel_or_not(d0)
    m0 = 4*np.pi/3*d0
    rs = [1., 50000000.]
    rs, ms, ps = integrate(rs, mass_press, A, Z, p0, m0, re)
    # the three variables below should return arrays of identical length
    rs = rs[np.nonzero(ps>0)]
    ps = ps[ps>0]
    ms = ms[np.nonzero(ps>0)]
    stepsize = 10000000
    i=1
    while ps[-1]>1 and stepsize>5:
        i+=1
        # dens = update_dens(re,ps[-1], A, Z)
        stepsize/= 2.
        r_range = [rs[-1],rs[-1]+stepsize]
        # print(len(ps))
        # print("The updated radius range at {}th reiteration is: {}".format(i, r_range))
        rs, ms, ps = integrate(r_range, mass_press, A, Z, ps[-1], ms[-1], re)
        rs = rs[np.nonzero(ps>0)]
        # print(rs)
        ps = ps[np.nonzero(ps>0)]
        ms = ms[np.nonzero(ps>0)]
    boundary_r = rs[-1] 
    return boundary_r, p0, re

def density_range(dens_crit, sigma, n):
    dens_l = np.log10(dens_crit/float(sigma))
    dens_u = np.log10(dens_crit*float(sigma))
#    creating an interval between dens_crit*sigma and crit_dens/sigma where the denses are uniformedly separated on the log scale
    denses = 10.**(np.linspace(dens_l, dens_u, n))
    # fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    # subfigs = fig.subfigures(1, 2, wspace=0.07)
    # leftax = subfigs[0].subplots(len(denses), sharex = 'col')
    # rightax = subfigs[1].subplots(len(denses), sharex = 'col')
    chandrasekhar_masses = []
    p0s = []
    Rs = []
    for i, de in enumerate(denses):
        R, p0, re = find_zero_cross(A_C, Z_C, de)
        Rs.append(R/1000.)
        # print("The boundary R found is {}".format(R))
        p0s.append(p0)
        # print("The central pressure at d0= {} kg/m^3 is {}".format(de,p0))
        rs = np.linspace(100,R, 100)
        rs, ms, ps = integrate(rs, mass_press, A_C, Z_C, p0, 0, re)
        rs = rs/1000.
        ms = ms/(1.989*10**30)   
        chandrasekhar_masses.append(ms[-1])    
        # leftax[4-i].scatter(rs[1:],ms[1:], marker='x', s=5)
        # rightax[4-i].scatter(rs[1:],ps[1:], marker='x', s=5)
    # leftax[-1].set_xlabel('Radius ($\mathrm{km}$)', labelpad=10) # Use argument `labelpad` to move label downwards.
    # leftax[2].set_ylabel('Mass ($\mathrm{M_{\odot}})$', labelpad=10)
    # subfigs[0].suptitle("M(r)")
    # subfigs[1].suptitle("P(r)")
    # rightax[-1].set_xlabel('Radius ($\mathrm{km}$)', labelpad=10) # Use argument `labelpad` to move label downwards.
    # rightax[2].set_ylabel('Pressure ($\mathrm{Pa}$)', labelpad=10)
    # fig.suptitle('Mass and Pressure over Radial Distances (including electron mass density)')
    # # plt.tight_layout()    
    # plt.show()
    # print(chandrasekhar_masses)
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.scatter(denses, chandrasekhar_masses, marker='+', color = 'skyblue')
    ax.set_xlabel('Central Density [$\\mathrm{kg/m^3}$]')
    ax.set_ylabel("Mass [$\mathrm{M_{\odot}}]$")
            # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()
    ax2.set_xscale('log')
    ax2.scatter(denses, Rs, color='green', marker='+')
    ax2.set_ylabel("Radius [$\mathrm{km}$]", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    print(chandrasekhar_masses)
    # plt.tight_layout()
        # break
    plt.show()
    
density_range(wd_dens, 10**2, 10)
