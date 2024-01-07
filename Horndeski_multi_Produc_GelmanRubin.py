import camb
from camb import model, initialpower
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import emcee
import corner
import sys, platform, os
from scipy.stats import norminvgauss
from scipy.stats import invgauss
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import integrate, special
import math
import configparser
import os, shutil, configparser
import os, sys
import stat
from scipy import stats
from scipy.interpolate import interp1d
import time
import scienceplots
from scipy.integrate import quad
from iminuit import Minuit
from iminuit.util import describe
from scipy.special import legendre
from scipy.integrate import quad
import logging
from scipy import integrate, interpolate, optimize
import BAO_tool as BER
from scipy import interpolate
from scipy.stats import linregress

import numpy as np
from numpy.linalg import inv
from numpy.random import normal

from scipy.interpolate import interp1d as i1d
from scipy.special import legendre
from scipy.signal import find_peaks
from scipy.linalg import eig

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mcfit
from operator import itemgetter 
import emcee
from classy import Class

import multiprocessing as mp

if len(sys.argv)!=12:
    raise Exception('syntaxis: [baofit.py,  survey,  fidmodel,  datamodel,  recon,  hexadecapole,pshot,   output directory]')
if sys.argv[5] != '0' and sys.argv[5] != '1':
    raise Exception('Please use 1 (True) or 0 (False) to include or not density field reconstruction')
if sys.argv[6] != '0' and sys.argv[6] != '1':
    raise Exception('Please use 1 (True) or 0 (False) to include or not the hexadecapole of the Pk')
if sys.argv[7] != '0' and sys.argv[7] != '1':
    raise Exception('Please use 1 (True) or 0 (False) to include or not the dispersion around the prediction for Pk')
    
SURVEY = sys.argv[1]
FIDMODEL = sys.argv[2]
DATAMODEL = sys.argv[3]
directory = sys.argv[4]
RECON = bool(int(sys.argv[5]))
HEXADECAPOLE = bool(int(sys.argv[6]))
DISPERSION = bool(int(sys.argv[7]))
Pshot_T = bool(int(sys.argv[8]))
OUTPUT = sys.argv[9]
Smooth_method = sys.argv[10]
Partindex = sys.argv[11]
print(SURVEY,FIDMODEL,DATAMODEL,RECON,directory,HEXADECAPOLE,DISPERSION,OUTPUT,Smooth_method,Partindex)
#print '...'
print ("Generating Mock data for a "+DATAMODEL+" cosmology")
#print '...'

#prepare Horndeski data
# npoin = 128

#Horndeski data

    
# def textiowrapper_to_arrays(textiowrapper_object):
#     data1 = []
#     data2 = []
#     for line in textiowrapper_object:
#         if line.startswith('#'):  # skip comment lines
#             continue
#         columns = line.split()
#       #  if float(columns[0]) <= 0.3:
#         data1.append(float(columns[0]))
#         data2.append(float(columns[1]))
#     return np.array(data1), np.array(data2)
    
def textiowrapper_to_arrays(textiowrapper_object):
    data1 = []
    data2 = []
    for line in textiowrapper_object:
        if line.startswith('#'):  # skip comment lines
            continue
        columns = line.split()
        #if float(columns[0]) <= 2.6:
        if (float(columns[0]) <= 1.1) and (float(columns[0]) >= 0.9e-3):
          data1.append(float(columns[0]))
          data2.append(float(columns[1]))
    return np.array(data1), np.array(data2)
    
def extrapolate_powerspectrum(kh,pk):
    Nk=4096
    kmin3 = 1e-5
    kmax3 = 2.7
    kh3 = np.logspace(np.log10(kmin3),np.log10(kmax3),Nk)

    lower = 0.00015
    linear_mask = np.where(kh <= lower)
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(kh[linear_mask]), np.log10(pk[linear_mask]))

    # Extrapolation: Extend the linear regression line to higher k values
    extrapolated_k = np.log10(kh3[np.where(kh3<=lower)])
    extrapolated_pk = slope * extrapolated_k + intercept
    extrapolated_pk_nolog = 10**(extrapolated_pk)
    extrapolated_k2 = np.log10(kh3[np.where(kh3>lower)])
    Plin = interpolate.UnivariateSpline(kh, pk, s=20 )


    # Maintain original P(k) values by blending with extrapolated values
    blended_pk = np.concatenate((extrapolated_pk,np.log10(Plin(10**extrapolated_k2))))
    return kh3,10**blended_pk

kvalues_Horndeski = []
pk_Horn_Horndeski = []
filtered_file_names = []
pk_base_index = []
for filename in os.listdir(directory):
   # if filename.endswith('0.65.dat'): #and filename.startswith('1.25'):
        file_path = os.path.join(directory, filename)
        split_filename = filename.split('_')
      # print(split_filename)
        base_paras = int(split_filename[2])
        pk_base_index.append(base_paras)
        with open(file_path, 'r') as file:
            kvalues, pk_Horn = textiowrapper_to_arrays(file)
        #    kvalues, pk_Horn = extrapolate_powerspectrum(kvalues, pk_Horn)
            kvalues_Horndeski.append(kvalues)
            pk_Horn_Horndeski.append(pk_Horn)
            filtered_file_names.append(filename)

kvalues_Horndeski = np.array(kvalues_Horndeski)
pk_Horn_Horndeski = np.array(pk_Horn_Horndeski)


def extract_float_value(filename_parts, prefix, index):
    try:
        value_string = filename_parts[filename_parts.index(prefix) + index]
        return float(value_string)
    except (ValueError, IndexError):
        return 0  # Return a default value of 0 if extraction fails

# Collect all filenames in the directory
file_names = os.listdir(directory)

# Filter and sort the filenames based on g1 values in descending order
filtered_file_names = sorted(
    [filename for filename in file_names],
    key=lambda filename: extract_float_value(filename.split('_'), 'g1', 1),
    reverse=True
)

def extract_float_value2(filename_parts, prefix, index1, index2):
    try:
        combined_value = filename_parts[index1] + '.' + filename_parts[index2]
        return float(combined_value)
    except (ValueError, IndexError):
        return 0

cmap = plt.get_cmap('viridis')  # You can choose any other colormap
basic_dict = BER.basic_cosmo(FIDMODEL)
if FIDMODEL == 'lcdm_planck':
    modelpars = {'A_s':2.1e-09,'n_s' : 0.9649,'h' : 0.6736,
                 'tau_reio' : 0.0544,'omega_cdm' : 0.12,'omega_b' : 0.02237} 
                #First tries
                #'h':0.6766,'omega_b':0.02242,'omega_cdm':0.11933,
                #'A_s':2.105e-9,'n_s':0.9665}
    modelpars = BER.merge_dicts([basic_dict,modelpars])
elif FIDMODEL == 'lcdm_horndeski':
    modelpars = {'A_s':2.101e-9,'n_s' : 0.96605,'h' : 0.6732,
                 'tau_reio' : 0.0544,'omega_cdm' : 0.12011,'omega_b' : 0.022383} 
# elif FIDMODEL == 'paper_horn':
#     modelpars = {'A_s':2.10e-9,'n_s' : 0.9649,'h' : 0.6736,
#                  'tau_reio' : 0.0544,'omega_cdm' : 0.14,'omega_b' : 0.0187} 
elif FIDMODEL == 'paper_horn':
    modelpars = {'A_s':2.10e-9,'n_s' : 0.9649,'h' : 0.6736,
                 'tau_reio' : 0.0544,'omega_cdm' : 0.119,'omega_b' : 0.022,'z_max_pk': 2.5} 
elif FIDMODEL == 'lcdm_planck_2':
    modelpars = {'A_s':2.1e-09,'n_s' : 0.9649,'h' : 0.6736,
                 'tau_reio' : 0.0544,'omega_cdm' : 0.14,'omega_b' : 0.0187,'z_max_pk': 10.0} 
                #First tries
                #'h':0.6766,'omega_b':0.02242,'omega_cdm':0.11933,
                #'A_s':2.105e-9,'n_s':0.9665}
    modelpars = BER.merge_dicts([basic_dict,modelpars])
else:
    raise Exception('Please, note that your model of choice is not coded, choose lcdm_planck or\
         include it here (and in basic_cosmo)')
    
cosmofid = BER.get_cosmo(modelpars)
hfid = cosmofid.h()

vecparam = BER.fiducial(SURVEY,HEXADECAPOLE,RECON)
#vecparam[7] *= hfid**3
coeffs = vecparam[4:-3]

vecparam_list = []
hhorn_list = []
base_index = []
csv_file_path = DATAMODEL#'samps_data.csv'

data = pd.read_csv(csv_file_path)
for i, filename in enumerate(filtered_file_names):
    file_path = os.path.join(directory, filename)
    split_filename = filename.split('_')
   # print(split_filename)
    base_paras = int(split_filename[2])-int(1) #this is the index corresponding to the comsologies used
    base_index.append(int(split_filename[2]))
    #read in cosmology
   # print('cosmology index',base_paras)
    # Extract the required columns
    omegabh2_val = data['omegabh2'].values[int(base_paras)]
    omegach2_val = data['omegach2'].values[int(base_paras)]
    logA_val = data['logA'].values[int(base_paras)]
    tau_val = data['tau'].values[int(base_paras)]
    ns_val = data['ns'].values[int(base_paras)]
    h_horn_val = data['H0'].values[int(base_paras)]
   # print(h_horn_val)
    # Calculate the corresponding values
    logA_list_val = np.exp(logA_val) * 10**(-10)
    DATAMODELhold = 'Production'
    # Populate the modelpars dictionary with extracted values
    modelpars = []
    basic_dict = BER.basic_cosmo(DATAMODELhold)
    if DATAMODELhold == 'Production':  # Use '==' for comparison
        modelpars = {
            'h': f'{h_horn_val*0.01:.5f}',
            'omega_b': f'{omegabh2_val:.5f}',
            'omega_cdm': f'{omegach2_val:.5f}',
            'A_s': f'{logA_list_val:.5e}',
            'n_s': f'{ns_val:.5f}',
            'tau_reio': f'{tau_val:.5f}'
            ,'z_max_pk': 2.5
        }
        # Rest of your code

        
    modelpars = BER.merge_dicts([basic_dict,modelpars])
    cosmodat = BER.get_cosmo(modelpars)
    vecparam = BER.fiducial(SURVEY,HEXADECAPOLE,RECON)
    #vecparam[7] *= hfid**3
    coeffs = vecparam[4:-3]

    #Get Mock Data
    z,V,k,dk,n,bias,beta = BER.survey_pars(survey=SURVEY)
    Pshot = 1./n#*hfid**3.
    Vh = V#*hfid**3. #Volume in (Mpc/h)^3
    #alpha_perp,alpha_par = get_AP_rescale(z,cosmofid,cosmodat)
    aperp = cosmodat.angular_distance(z)/cosmofid.angular_distance(z)
    apar = cosmofid.Hubble(z)/cosmodat.Hubble(z)
    vecparam[0] = aperp
    vecparam[1] = apar
  #  print('alpha', aperp, apar)
    vecparam_list.append(vecparam)
    hhorn_list.append(h_horn_val*0.01)


#Get Mock Data
z,V,k,dk,n,bias,beta = BER.survey_pars(survey=SURVEY)
Pshot = 1./n#*hfid**3.
Vh = V#*hfid**3. #Volume in (Mpc/h)^3



z, V, k, dk, n, bias, beta = BER.survey_pars(survey=SURVEY)
Pshot = 1. / n  # *hfid**3.
Vh = V  # *hfid**3. #Volume in (Mpc/h)^3
#Smooth_method = 'Bernal'
Pmuldat_list = []
cosmo=0
#hhorn_list = hhorn_list#[x * 0.01 for x in hhorn_list]
for khorn, pkhorn, pkindex in zip(kvalues_Horndeski, pk_Horn_Horndeski, pk_base_index):
    for h_horn, vecparam, baseindex in zip(hhorn_list, vecparam_list, base_index):
        if baseindex == pkindex:
            if Smooth_method == 'Bernal':
                iPk_nw, iOlin = BER.get_templates_horn(khorn, pkhorn, z, hfid=hfid, h_horn=h_horn, r1bounds=[240, 280])
            elif Smooth_method == 'Barry':
                iPk_nw, iOlin = BER.get_templates_my_horn(khorn, pkhorn, hfid=hfid, h_horn=h_horn)
           
            coeffs = vecparam[4:-3]
            nmu = 1000
            mu_edge = np.linspace(-1, 1, nmu + 1)
            mu = (mu_edge[0:nmu + 1 - 1] + mu_edge[1:nmu + 1]) / 2.
            ki_grid, mui_grid = np.meshgrid(k, mu)

            Pkmu = BER.Pk_AP(ki_grid, mui_grid, iPk_nw, iOlin, vecparam, RECON) + Pshot
            Pmuldat = BER.get_multipoles(ki_grid, mui_grid, Pkmu, coeffs, HEXADECAPOLE, RECON)
            Pmuldat_list.append(Pmuldat)
if DISPERSION:
    #Get the covariance to add the dispersion
    covmat_dat = BER.get_covmat(ki_grid,mui_grid,Pkmu,dk,Vh,HEXADECAPOLE)
    #Get dispersion for P(k) and add it to the multipoles of Pk
    dispersion = BER.dispersion_k(covmat_dat)
    Pmuldat += dispersion
    
# if not HEXADECAPOLE:
#     MAT = np.zeros((len(k)*2,2))
#     MAT[:,0] = np.concatenate((k,k))
#     MAT[:,1] = Pmuldat
# else:
#     MAT = np.zeros((len(k)*3,3))
#     MAT[:,0] = np.concatenate((k,k,k))
#     MAT[:,1] = Pmuldat
#np.savetxt(OUTPUT+f'pkdat2_{file_name}.txt',MAT)

if Smooth_method == 'Bernal':
    iPk_nw_fid, iOlin_fid = BER.get_templates(cosmofid,z,r1bounds=[240,280])
elif Smooth_method == 'Barry':
    iPk_nw_fid, iOlin_fid = BER.get_templates_barry(cosmofid,z,r1bounds=[240,280])
    
nmu = 1000
mu_edge = np.linspace(-1, 1, nmu + 1)
mu = (mu_edge[0:nmu + 1 - 1] + mu_edge[1:nmu + 1]) / 2.
ki_grid, mui_grid = np.meshgrid(k, mu)
# The covariance is computed from the fiducial cosmology
vecparam[0] = 1.
vecparam[1] = 1.
Pkmu_fid = BER.Pk_AP(ki_grid,mui_grid,iPk_nw_fid,iOlin_fid,vecparam,RECON)+Pshot
covmat = BER.get_covmat(ki_grid,mui_grid,Pkmu_fid,dk,Vh,HEXADECAPOLE)
invcovmat = inv(covmat)

cosmofid.struct_cleanup()
cosmodat.struct_cleanup()



#Set number of walkers, number of steps and initial conditions
nwalkers = 100
max_steps = 2000#2000#200000#200000 11000

pos = np.zeros((nwalkers,0))
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[0]+normal(0.,0.15,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[1]+normal(0.,0.15,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[2]+normal(0.,0.5,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[3]+normal(0.,0.2,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[4]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[5]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[6]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[7]+normal(0,100,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[8]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[9]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[10]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[11]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[12]+normal(0.,0.1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[13]+normal(0.,0.1,(nwalkers,1)),axis = 1)

if HEXADECAPOLE:
    prior_min = [0.5,0.5,0.01,0.,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,
                -1e5,-1e5,-1e5,-1e5,-1e5,0.,0.,0.]
    prior_max = [1.5,1.5,100,10,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5,
                1e5,100,100,100]
    ndim = len(prior_min)
    pos = np.append(pos,np.ones((nwalkers,1))*vecparam[14]+normal(0.,0.1,(nwalkers,1)),axis = 1)
    pos = np.append(pos,np.ones((nwalkers,1))*vecparam[15]+normal(0.,0.1,(nwalkers,1)),axis = 1)
    pos = np.append(pos,np.ones((nwalkers,1))*vecparam[16]+normal(0.,0.1,(nwalkers,1)),axis = 1)
    pos = np.append(pos,np.ones((nwalkers,1))*vecparam[17]+normal(0.,0.1,(nwalkers,1)),axis = 1)
    pos = np.append(pos,np.ones((nwalkers,1))*vecparam[18]+normal(0.,0.1,(nwalkers,1)),axis = 1)
    
else:
    # prior_min = [0.8,0.8,0.01,0.,-1e3,-1e3,-1e3,-1e3,-1e3,-1e3,-1e3,-1e3,-1e3,-1e3,
    #             0.,0.,0.]
    # prior_max = [1.2,1.2,100,10,1e3,1e3,1e3,1e3,1e3,1e3,1e3,1e3,1e3,1e3,100,100,100]
    prior_min = [0.8,0.8,0.01,0.,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,-1e5,
                0.,0.,0.]
    prior_max = [1.2,1.2,100,10,5e4,5e4,5e4,5e4,5e4,5e4,5e4,5e4,5e4,5e4,100,100,100]
    ndim = len(prior_min)
    
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[-3]+normal(0.,1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[-2]+normal(0.,1,(nwalkers,1)),axis = 1)
pos = np.append(pos,np.ones((nwalkers,1))*vecparam[-1]+normal(0.,1,(nwalkers,1)),axis = 1)

def remove_dat_extension(file_name):
    if file_name.endswith('.dat'):
        new_file_name = file_name[:-4]  # Remove the last 4 characters (.dat)
        return new_file_name
    else:
        return file_name
# Define your function to be executed in parallel
def run_mcmc_parallel(args):
    Pmuldat, file_name = args
    new_name = remove_dat_extension(file_name)
    if not HEXADECAPOLE:
        MAT = np.zeros((len(k)*2,2))
        MAT[:,0] = np.concatenate((k,k))
        MAT[:,1] = Pmuldat
    else:
        MAT = np.zeros((len(k)*3,3))
        MAT[:,0] = np.concatenate((k,k,k))
        MAT[:,1] = Pmuldat
    np.savetxt(OUTPUT+f'pkdat_LRGappB{Partindex}BarryGelRu_{new_name}.txt',MAT)
   # print(Pmuldat, file_name)
    backend = emcee.backends.HDFBackend(OUTPUT+f'store_ChainLRGappB{Partindex}BarryGelRu_{SURVEY}_{new_name}.h5')
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, BER.lnprob, a = 1.4, 
            args=(k,Pmuldat,invcovmat,iPk_nw_fid,iOlin_fid,HEXADECAPOLE,RECON,
                  prior_min,prior_max,1/Pshot,Pshot_T))

   # print('Total max_steps = ', nwalkers * max_steps)

    index = 0
    autocorr = np.empty(max_steps)
    old_tau = 1e10

    dummy = None

    state = sampler.run_mcmc(pos, 300, blobs0=dummy)
    sampler.reset()

    print("Running MCMC...")

    for sample in sampler.sample(pos, iterations=max_steps, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 200:
            continue
        if sampler.iteration > 0:
            chain_means = np.mean(sampler.get_chain(), axis=1)
            chain_variances = np.var(sampler.get_chain(), axis=1, ddof=1)
            mean_within_chain_variance = np.mean(chain_variances)
            mean_chain_mean = np.mean(chain_means, axis=0)
            between_chain_variance = np.var(chain_means, axis=0, ddof=1) * nwalkers
            W = mean_within_chain_variance
            B = between_chain_variance
            var_estimate = ((sampler.chain.shape[1] - 1) / sampler.chain.shape[1]) * W + B / sampler.chain.shape[1]
            R_hat_values = np.sqrt(var_estimate / W)
        
        # Check convergence using Gelman-Rubin statistic
        if np.all(R_hat_values < 1.0010):
            break


    sampler.run_mcmc(state, max_steps, rstate0=np.random.get_state(), blobs0=dummy)

    print("Done.")

    acceptance_fraction = np.sum(sampler.acceptance_fraction) / nwalkers

    print("Sample acceptance fraction", np.sum(sampler.acceptance_fraction) / nwalkers)

    burnin = 800
    thin = 5
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    minus_lkl = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    MAT = np.zeros((samples.shape[0],ndim+1))
    MAT[:,0] = minus_lkl
    MAT[:,1:] = samples

    np.savetxt(OUTPUT + f'Chaindesi_LRGappB{Partindex}BarryProduction_{SURVEY}_{new_name}.txt', MAT)

    return acceptance_fraction


if __name__ == '__main__':
    # Define your input parameters
   # k_fixed = #...  # Fixed value of k
    Pmuldat_values = Pmuldat_list #[...]  # List of Pmuldat values to loop through

    # Define the number of processes to use
    num_processes = mp.cpu_count()

    # Create a list of arguments to pass to the function
    arguments = [(Pmuldat, file_name) for Pmuldat, file_name in zip(Pmuldat_values, filtered_file_names)]
  #  print(arguments)
    # Create a multiprocessing Pool and execute the function in parallel
    with mp.Pool(processes=num_processes) as pool:
        acceptance_fractions = pool.map(run_mcmc_parallel, arguments)

    print("Sample acceptance fractions:", acceptance_fractions)





# f1 = open(OUTPUT + '_log6.txt', 'w')
# f1.write('####Log file of the chain####\n\nChain with ')
# #f1.write(repr(nwalkers) + ' walkers and ' + repr(steps) + ' steps: Total steps = ' + repr(nwalkers * steps) + '\n\n')
# f1.write('\n\nMean acceptance fraction' + repr(np.sum(sampler.acceptance_fraction) / nwalkers))
# #f1.write('\n\nAnd the minimum of the -Loglkl = ' + repr(bestfit[0]))
# f1.write('Autocorrelation times for each parameter:\n' + repr(sampler.acor))
# f1.close()
