# -----------------------------
# CUFT Cosmology: MCMC with SN + BAO + Growth Data
# -----------------------------
!pip install emcee corner

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid as cumtrapz, quad
from numpy.linalg import inv
import emcee
import corner

# -----------------------------
# File paths
# -----------------------------
sn_file = '/content/Pantheon-SH0ES.dat.txt'
bao_file = '/content/sdss_DR12Consensus_final.dat (1).txt'
cov_file = '/content/BAO_consensus_covtot_dM_Hz (2).txt'
growth_file = '/content/growth_data.txt'
mcmc_csv_file = '/content/dcuft_mcmc_chain.csv'

# -----------------------------
# Constants
# -----------------------------
OMEGA_B_H2 = 0.0224
OMEGA_GAMMA_H2 = 2.47e-5
C_LIGHT = 299792.458  # km/s

# -----------------------------
# Load data
# -----------------------------
def load_data(sn_file, bao_file, cov_file, growth_file):
    # Supernovae
    data_sn_all = pd.read_csv(sn_file, sep='\s+', skiprows=1, header=None, dtype=str)
    data_sn_all = data_sn_all.iloc[:, [2, 8, 9]].apply(pd.to_numeric, errors='coerce')
    data_sn_all.dropna(inplace=True)
    z_sn = data_sn_all.iloc[:, 0].values
    mu_sn = data_sn_all.iloc[:, 1].values
    mu_err_sn = data_sn_all.iloc[:, 2].values

    # BAO
    data_bao = pd.read_csv(bao_file, sep='\s+', header=None, comment='#', names=['z','value','type'])
    z_list = [0.38, 0.51, 0.61]
    V_data = []
    for z in z_list:
        dm_rs = data_bao[(data_bao['z']==z)&(data_bao['type']=='DM_over_rs')]['value'].values[0]
        hz_rs = data_bao[(data_bao['z']==z)&(data_bao['type']=='bao_Hz_rs')]['value'].values[0]
        V_data.extend([dm_rs, hz_rs])
    V_data = np.array(V_data)
    z_bao_vec = np.repeat(z_list,2)

    # Covariance
    C = np.loadtxt(cov_file)
    C_inv = inv(C)

    # Growth data
    growth_data = np.loadtxt(growth_file)
    z_g = growth_data[:,0]
    fsigma8_obs = growth_data[:,1]
    fsigma8_err = growth_data[:,2]

    return z_sn, mu_sn, mu_err_sn, z_bao_vec, V_data, C_inv, z_g, fsigma8_obs, fsigma8_err

# -----------------------------
# D-CUFT 3-param Model
# -----------------------------
class DCUFT2:
    OMEGA_R = 4.15e-5
    def __init__(self,H0=70.0, logA=-35.0, Om=0.3, Phi_i=0.1):
        self.H0 = H0
        self.logA = logA
        self.Lambda = 10**self.logA
        self.Om = Om
        self.Ode0 = 1.0 - Om - self.OMEGA_R*(self.H0/100)**-2
        self.Phi_i = Phi_i
        self.h = self.H0 / 100

    def V(self,phi): return self.Lambda*phi**2
    def dV_dphi(self,phi): return 2*self.Lambda*phi

    def equations(self,t,y):
        phi, phidot, rho_m, rho_r, rho_de = y
        H2 = (8*np.pi/3)*(rho_m + rho_r + rho_de + 0.5*phidot**2 + self.V(phi))
        H = np.sqrt(max(H2,1e-18))
        return [phidot, -3*H*phidot - self.dV_dphi(phi),
                -3*H*rho_m, -4*H*rho_r, -4*H*rho_de]

    def solve_background(self,zmax=3.0,npts=300):
        H0_phys = self.H0 / 3.086e19
        y0 = [self.Phi_i,0.0,3*H0_phys**2*self.Om/(8*np.pi),
              3*H0_phys**2*self.OMEGA_R/(8*np.pi),
              3*H0_phys**2*self.Ode0/(8*np.pi)]
        t_span = (np.log(1/(1+zmax)),0)
        sol = solve_ivp(self.equations,t_span,y0,method='BDF',dense_output=True,rtol=1e-6,atol=1e-8)
        t_eval = np.linspace(t_span[0],t_span[1],npts)
        y_eval = sol.sol(t_eval)
        a_vals = np.exp(t_eval)
        z_vals = 1/a_vals -1
        H_vals = np.zeros_like(z_vals)
        for i in range(len(z_vals)):
            phi,phidot,rho_m,rho_r,rho_de = y_eval[:,i]
            H_vals[i] = np.sqrt(max(8*np.pi/3*(rho_m+rho_r+rho_de+0.5*phidot**2+self.V(phi)),1e-18))*3.086e19
        return z_vals, H_vals

# -----------------------------
# Growth Solver
# -----------------------------
def solve_growth_ode(a_vals, H_vals, Om):
    delta_vals = np.zeros_like(a_vals)
    f_vals = np.zeros_like(a_vals)
    delta_vals[0] = a_vals[0]
    for i in range(1,len(a_vals)):
        da = a_vals[i] - a_vals[i-1]
        H = H_vals[i]
        Om_a = Om / a_vals[i]**3 / (Om / a_vals[i]**3 + (1-Om))
        ddelta = delta_vals[i-1] * da * np.sqrt(Om_a)/H
        delta_vals[i] = delta_vals[i-1] + ddelta
        f_vals[i] = ddelta / (delta_vals[i-1]*da)
    return delta_vals, f_vals

# -----------------------------
# MCMC Likelihood (SN+BAO+Growth)
# -----------------------------
def log_likelihood(theta, data):
    H0, logA, Om = theta
    if H0<=0 or Om<=0 or Om>=1: return -np.inf
    dcuft = DCUFT2(H0, logA, Om)
    z_sn, mu_sn, mu_err_sn, z_bao, V_data, Cinv, z_g, fsigma8_obs, fsigma8_err = data
    try:
        z_vals, H_vals = dcuft.solve_background(zmax=max(np.max(z_sn), np.max(z_bao), np.max(z_g)))
        H_interp = lambda z: np.interp(z, z_vals[::-1], H_vals[::-1])
        Dl_integrand = lambda z: C_LIGHT / H_interp(z)
        Dl_integral_sn = cumtrapz(Dl_integrand(z_sn), z_sn, initial=0)
        mu_model_sn = 5*np.log10(np.maximum((1+z_sn)*Dl_integral_sn,1e-8)) + 25
        chi2_sn = np.sum(((mu_sn - mu_model_sn)/mu_err_sn)**2)

        # BAO
        V_model = []
        for z in np.unique(z_bao):
            H_z = H_interp(z)
            Dm = cumtrapz(Dl_integrand(np.linspace(0,z,50)), np.linspace(0,z,50), initial=0)[-1]
            V_model.append(Dm)
            V_model.append(H_z)
        V_model = np.array(V_model)
        Delta = V_data - V_model
        chi2_bao = Delta.T @ Cinv @ Delta

        # Growth
        a_vals = 1/(1+z_g)
        delta,f_vals = solve_growth_ode(a_vals,H_vals[::-1],Om)
        fsigma8_model = f_vals*delta
        chi2_growth = np.sum(((fsigma8_obs - fsigma8_model)/fsigma8_err)**2)

    except:
        return -1e12

    return -0.5*(chi2_sn + chi2_bao + chi2_growth)

def log_prior(theta):
    H0, logA, Om = theta
    if 60<H0<80 and -36<logA<-34 and 0.1<Om<0.5: return 0.0
    return -np.inf

def log_prob(theta, data):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta, data)

# -----------------------------
# Run MCMC
# -----------------------------
def run_mcmc(nwalkers=64,nsteps=5000):
    data = load_data(sn_file, bao_file, cov_file, growth_file)
    ndim = 3
    start_pos = np.array([70.0,-35.0,0.3])
    pos = start_pos + np.array([0.5,0.05,0.05])*np.random.randn(nwalkers,ndim)
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_prob,args=[data])
    sampler.run_mcmc(pos,nsteps,progress=True)
    flat_samples = sampler.get_chain(flat=True,discard=int(0.1*nsteps),thin=1)
    np.savetxt(mcmc_csv_file, flat_samples, delimiter=',', header='H0,logA,Om', comments='')
    print(f"MCMC chain saved to {mcmc_csv_file}")
    return flat_samples, sampler

# -----------------------------
# Diagnostics
# -----------------------------
def diagnostics(flat_samples,sampler):
    medians = np.median(flat_samples,axis=0)
    perc = np.percentile(flat_samples,[16,84],axis=0)

    print(f"H0 = {medians[0]:.4f} +{perc[1,0]-medians[0]:.4f}/-{medians[0]-perc[0,0]:.4f}")
    print(f"log10 Λ = {medians[1]:.4f} +{perc[1,1]-medians[1]:.4f}/-{medians[1]-perc[0,1]:.4f}")
    print(f"Ω_m = {medians[2]:.4f} +{perc[1,2]-medians[2]:.4f}/-{medians[2]-perc[0,2]:.4f}")

    if corner is not None:
        fig = corner.corner(flat_samples, labels=["H0","logA","Ω_m"], show_titles=True)
        fig.suptitle("D-CUFT 3-param SN+BAO+Growth", fontsize=16)
        fig.savefig("dcuft_corner.png")

    dcuft = DCUFT2(*medians)
    zvals, Hvals = dcuft.solve_background(zmax=3.0)
    plt.figure()
    plt.plot(zvals,Hvals,label="H(z)")
    plt.xlabel("z"); plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("D-CUFT Expansion H(z) Best-fit")
    plt.savefig("dcuft_Hz.png")
    plt.close('all')

    # Growth plot
    growth_data = np.loadtxt(growth_file)
    z_g = growth_data[:,0]
    fsigma8_obs = growth_data[:,1]
    fsigma8_err = growth_data[:,2]
    a_vals = 1/(1+z_g)
    delta,f_vals = solve_growth_ode(a_vals,Hvals[::-1],medians[2])
    fsigma8_model = f_vals*delta

    plt.figure(figsize=(7,5))
    plt.errorbar(z_g,fsigma8_obs,yerr=fsigma8_err,fmt='o',label="Observed fσ8")
    plt.plot(z_g,fsigma8_model,'-',color='red',label="D-CUFT Best-fit")
    plt.xlabel("z"); plt.ylabel("fσ8(z)")
    plt.title("Growth Rate Comparison")
    plt.legend()
    plt.savefig("dcuft_fsigma8.png")
    plt.close('all')
    print("Diagnostics complete. Corner, H(z), fσ8 plots saved.")

# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    flat_samples, sampler = run_mcmc()
    diagnostics(flat_samples,sampler)