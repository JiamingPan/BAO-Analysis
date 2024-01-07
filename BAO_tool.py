import numpy as np
from numpy.linalg import inv
from numpy.random import normal

from scipy.interpolate import interp1d as i1d
from scipy.special import legendre
from scipy.signal import find_peaks
from scipy.linalg import eig
from scipy import interpolate

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import mcfit

from operator import itemgetter 
import emcee

from classy import Class
#import Class
def log_interp1d(xx, yy, kind='linear',bounds_error=False,fill_value='extrapolate'):
    '''
    log-log interpolation
    '''
    try:
        logx = np.log10(xx.value)
    except:
        logx = np.log10(xx)
    try:
        logy = np.log10(yy.value)
    except:
        logy = np.log10(yy)
    lin_interp = i1d(logx, logy, kind=kind,bounds_error=bounds_error,fill_value=fill_value)
    
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))

    return log_interp
    
    
def semilogx_interp1d(xx, yy, kind='linear',bounds_error=False,fill_value='extrapolate'):
    '''
    log-linear interpolation
    '''
    try:
        logx = np.log10(xx.value)
    except:
        logx = np.log10(xx)
    lin_interp = i1d(logx, yy, kind=kind,bounds_error=bounds_error,fill_value=fill_value)
    
    #log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))

    log_interp = lambda zz: lin_interp(np.log10(zz))
    return log_interp
    
    
def merge_dicts(D):
    '''
    Merges dictionaries
    '''
    dic = {}
    for k in D:
        if k:
            dic.update(k)
    return dic
    
    
def polyk(k,a1,a2,a3,a4,a5,recon):
    '''
    Broadband model for the power spectrum multipoles.
    Note: for pre-recon:   a1*k**-3 + a2*k**-2 + a3*k**-1 + a4 + a5*k
          for post-recon:  a1*k**-3 + a2*k**-2 + a3*k**-1 + a4 + a5*k**2
    '''
    if not recon:
        return a1*k**-3 + a2*k**-2 + a3*k**-1 + a4 + a5*k
    else:
        return a1*k**-3 + a2*k**-2 + a3*k**-1 + a4 + a5*k**2
        
        
def get_multipoles(ki_grid,mui_grid,Pkhmu,coeffs,hexadecapole,recon):
    '''
    Gets the P(k) multipoles from P(k,mu), including the broad-band polynomials
    '''
    L2 = legendre(2)(mui_grid)
    
    a01,a02,a03,a04,a05 = itemgetter(0,1,2,3,4)(coeffs)
    a21,a22,a23,a24,a25 = itemgetter(5,6,7,8,9)(coeffs)
    PK0 = 0.5*np.trapz(Pkhmu,mui_grid,axis=0) + polyk(ki_grid[0,:],a01,a02,a03,a04,a05,recon)
    PK2 = 2.5*np.trapz(Pkhmu*L2,mui_grid,axis=0) + polyk(ki_grid[0,:],a21,a22,a23,a24,a25,recon)
    
    if hexadecapole:
        a41,a42,a43,a44,a45 = itemgetter(10,11,12,13,14)(coeffs)
        L4 = legendre(4)(mui_grid)
        PK4 = 4.5*np.trapz(Pkhmu*L4,mui_grid,axis=0) + polyk(ki_grid[0,:],a41,a42,a43,a44,a45,recon)
        return np.concatenate((PK0,PK2,PK4))
    else:
        return np.concatenate((PK0,PK2))
        
############################
##### Prepare the data #####
############################

def survey_pars(survey):
    '''
    Defines each possible survey with a given name.
    
    Note that each for surveys with several redshift bins, you should enter each redshift bin 
        as a different "survey"
    '''
    if survey == 'example':
        zcent = 0.6
        V_inbins = 5e9 
        khmin = 0.02
        khmax = 0.2
        nk = 128
        n = 1e-4
        bias = 2.
        beta = 0.4
    # elif survey == 'BOSS_CMASS':
    #     zcent = 0.57
    #     V_inbins = 3.7044e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128#29
    #     n = 2.1e-4#(Mpc/h)^-3 
    #     bias = 1.85
    #     beta = 0.4225
    # elif survey == 'ELG_0.75':
    #     zcent = 0.75
    #     V_inbins = 6.8e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 4.22e-4#(Mpc/h)^-3 
    #     bias = 1.95
    #     beta = 0.451
    # elif survey == 'ELG_1.25':
    #     zcent = 1.25
    #     V_inbins = 4.01e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 3.31e-4#(Mpc/h)^-3
    #     bias = 2.42
    #     beta = 0.233
    # elif survey == 'LRG_0.55':
    #     zcent = 0.55
    #     V_inbins = 4.24e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 4.84e-4#(Mpc/h)^-3
    #     bias = 3.6
    #     beta = 0.234
    # elif survey == 'ELG_1.25':
    #     zcent = 1.25
    #     V_inbins = 4.01e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 138.34e-4#(Mpc/h)^-3 
    #     bias = 3.6
    #     beta = 0.233
    elif survey == 'ELG_1.25':
        zcent = 1.25
        V_inbins = 4.01e9 #(Mpc/h)^3
        khmin = 0.01
        khmax = 0.3
        nk = 128
        n = 4.86e-4#(Mpc/h)^-3
        bias = 1.44
        beta = 0.233
    elif survey == 'LRG_0.55':
        zcent = 0.55
        V_inbins = 4.24e9 #(Mpc/h)^3
        khmin = 0.01
        khmax = 0.3
        nk = 128
        n = 4.84e-4#(Mpc/h)^-3
        bias = 2.16
        beta = 0.234
    # elif survey == 'LRG_0.65':
    #     zcent = 0.65
    #     V_inbins = 1.76e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 1.21e-4#(Mpc/h)^-3
    #     bias = 3.77
    #     beta = 0.223
    elif survey == 'QSO_1.25':
        zcent = 1.25
        V_inbins = 17.707e9#(Mpc/h)^3
        khmin = 0.01
        khmax = 0.3
        nk = 128
        n = 5.223e-4#(Mpc/h)^-3
        bias = 2.06
        beta = 0.194
    # elif survey == 'DESII_2.25':
    #     zcent = 2.25
    #     V_inbins = 12.03e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 2.95e-4#(Mpc/h)^-3
    #     bias = 2.44
    #     beta = 0.160
    # elif survey == 'QSO_1.85':
    #     zcent = 1.85
    #     V_inbins = 0.7e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 0.3e-4#(Mpc/h)^-3
    #     bias = 4.32
    #     beta = 0.194
    elif survey == 'Mega_2.25':
        zcent = 2.25
        V_inbins = 10.87e9 #(Mpc/h)^3
        khmin = 0.01
        khmax = 0.3
        nk = 128
        n = 7.9e-4#(Mpc/h)^-3
        bias = 2.12
        beta = 0.160
    elif survey == 'DESII_2.25':
        zcent = 2.25
        V_inbins = 7.76e9 #(Mpc/h)^3
        khmin = 0.01
        khmax = 0.3
        nk = 128
        n = 2.77e-4#(Mpc/h)^-3
        bias = 2.12
        beta = 0.160
    # elif survey == 'LRG_0.55':
    #     zcent = 0.55
    #     V_inbins = 4.24e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 23.68e-4#(Mpc/h)^-3  #updated Aug 26
    #     bias = 3.6
    #     beta = 0.234
    # elif survey == 'QSO_1.65':
    #     zcent = 1.65
    #     V_inbins = 1.4e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 0.711e-4#(Mpc/h)^-3 
    #     bias = 4.03
    #     beta = 0.208
    # elif survey == 'QSO_1.75':
    #     zcent = 1.75
    #     V_inbins = 0.47e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 3.55e-4#(Mpc/h)^-3 
    #     bias = 4.17
    #     beta = 0.201
    # elif survey == 'QSO_1.85':
    #     zcent = 1.85
    #     V_inbins = 0.7e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 10.98e-4#(Mpc/h)^-3
    #     bias = 4.32
    #     beta = 0.194
    # elif survey == 'QSOL_1.75':
    #     zcent = 1.75
    #     V_inbins = 4.7e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 3.55e-4#(Mpc/h)^-3 
    #     bias = 4.17
    #     beta = 0.201
    # elif survey == 'BOSS_CMASS_0.65':
    #     zcent = 0.75
    #     V_inbins = 3.7044e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128#29
    #     n = 2.1e-4#(Mpc/h)^-3 
    #     bias = 1.85
    #     beta = 0.4225
    # elif survey == 'BOSS_LOWZ':
    #     zcent = 0.32
    #     V_inbins = 1.2691e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 2.85e-4#(Mpc/h)^-3 
    #     bias = 1.85
    #     beta = 0.3743
    # elif survey == 'BOSS_LOWZ_0.65':
    #     zcent = 0.65
    #     V_inbins = 1.2691e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 2.85e-4#(Mpc/h)^-3 
    #     bias = 1.85
    #     beta = 0.3743
    # elif survey == 'DESI_0.8':
    #     zcent = 0.8
    #     V_inbins = 6.8e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 1.226e-3#(Mpc/h)^-3 
    #     bias = 1.86
    #     beta = 0.451
    # elif survey == 'QSO_1.25':
    #     zcent = 1.25
    #     V_inbins = 15.75e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 5.223e-4#(Mpc/h)^-3
    #     bias = 3.59
    #     beta = 0.194
    # elif survey == 'Mega_2.25':
    #     zcent = 2.25
    #     V_inbins = 13.04e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 6.62e-4#(Mpc/h)^-3
    #     bias = 2.5
    #     beta = 0.160
  #  elif survey == 'DESI_0.65':
    #     zcent = 0.65
    #     V_inbins = 6.8e9 #(Mpc/h)^3
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 1.226e-3#(Mpc/h)^-3 
    #     bias = 1.86
    #     beta = 0.451
    # elif survey == 'DESI_1.0':
    #     zcent = 1.0
    #     V_inbins = 8.73e9 #(Mpc/h)^3 
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 6.41e-4 #(Mpc/h)^-3 
    #     bias = 1.5
    #     beta = 0.5831
    # elif survey == 'DESI_1.05':
    #     zcent = 1.05
    #     V_inbins = 8.73e9 #(Mpc/h)^3 
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 6.41e-4 #(Mpc/h)^-3 
    #     bias = 1.5
    #     beta = 0.5831
    # elif survey == 'DESI_1.25':
    #     zcent = 1.25
    #     V_inbins = 8.73e9 #(Mpc/h)^3 
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 6.41e-4 #(Mpc/h)^-3 
    #     bias = 1.5
    #     beta = 0.5831
    # elif survey == 'DESI_1.65':
    #     zcent = 1.65
    #     V_inbins = 8.73e9 #(Mpc/h)^3 
    #     khmin = 0.01
    #     khmax = 0.3
    #     nk = 128
    #     n = 6.41e-4 #(Mpc/h)^-3 
    #     bias = 1.5
    #     beta = 0.5831
    else:
        raise TypeError('Please use an option between: survey = "example","BOSS_CMASS","BOSS_LOWZ","DESI_0.8","DESI_1.0" ')
        
    k_edge = np.linspace(khmin,khmax,nk+1)
    
    kh = 0.5*(k_edge[0:nk+1-1]+k_edge[1:nk+1])
    dkh = np.diff(k_edge)
     
    return zcent,V_inbins,kh,dkh,n,bias,beta


def fiducial(SURVEY,hexadecapole,recon):
    '''
    Fiducial values of the measured/varied parameters in the analysis.
    In this case, it corresponds to: alpha_perp,alpha_par,B,beta,a_ell^{1-5},sigma_fog, Sigma_perp,Sigma_par
    '''
    zcent,V_inbins,kh,dkh,n,bias,beta = survey_pars(survey=SURVEY)
    #a04 = 1./n
    if recon:
        Sigma_par = 4.
        Sigma_perp = 2.
    else:
        Sigma_par = 8. 
        Sigma_perp = 4.
        
    if not hexadecapole:
        return [1.,1.,bias**2,beta,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,10,Sigma_perp,Sigma_par]
    else:
        return [1.,1.,bias**2,beta,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,10,Sigma_perp,Sigma_par]


def fitting(r0,r1,dire):

    iCF = dire['iCF']
    iPk = dire['iPk']
    rvec = dire['rvec']
    kdum = dire['kdum']
    ind=np.concatenate((np.where(rvec<=r0)[0],np.where(rvec>=r1)[0]))
    CF_nw = semilogx_interp1d(rvec[ind],rvec[ind]**2*iCF(rvec[ind]),
                           kind='cubic',fill_value='extrapolate',bounds_error=False)(rvec)/(rvec**2)
    iCF_nw = semilogx_interp1d(rvec,CF_nw,
                               kind='cubic',fill_value='extrapolate',bounds_error=False)
    #Undo the FT to get the unwiggled Pk (use rh to avoid ringing)
    ift = mcfit.xi2P(rvec, l=0, lowring=True)
    kkvec, Pk_nw = ift(CF_nw, extrap=True)
    iPk_nw = semilogx_interp1d(kkvec,Pk_nw,kind='cubic',fill_value='extrapolate',bounds_error=False)
    iOlin = semilogx_interp1d(kkvec,iPk(kkvec)/iPk_nw(kkvec),kind='cubic',fill_value='extrapolate',bounds_error=False)

    return np.sqrt(np.sum((iOlin(kdum)-1)**2))


def get_templates(COSMO,z,hfid=None,r0bounds=None,r1bounds=None):
    Nk=4096
    kmin = 1e-5
    kmax = 20.
    factor=0.01
    
    kmax_class = 3.
    
    kh = np.logspace(np.log10(kmin),np.log10(kmax_class),Nk) 
    pk = np.zeros(Nk)
    if not hfid:
        k = kh*COSMO.h() #so here the kh does not have unit of h. 
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*COSMO.h()**3
    else:
        k = kh*hfid
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*hfid**3
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    kh = np.logspace(np.log10(kmin),np.log10(kmax),Nk) #The previous kh up to k_max (not k_max class)
    Pkh = iPk(kh)
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)    #Probably not needed, but for consistency                               
                               
    #Fourier Transform to get it to CF
    ft = mcfit.P2xi(kh, l=0, lowring=True)
    rvec, CF = ft(Pkh, extrap=True)
    iCF = semilogx_interp1d(rvec,CF,kind='cubic',bounds_error=False,fill_value='extrapolate')
    kdum = np.logspace(-4,-3,1000)
    dire = dict(iCF=iCF,iPk=iPk,rvec=rvec,kdum=kdum)
    
    if not r0bounds:
        r0_vec = np.linspace(60,69,40)
    else:
        r0_vec = np.linspace(r0bounds[0],r0bounds[1],40)
    res = np.zeros(len(r0_vec))
    r1 = np.zeros(len(r0_vec))
    if not r1bounds:
        r1b_1,r1b_2 = 200,300
    else:
        r1b_1,r1b_2 = r1bounds[0],r1bounds[1]
    for i in range(len(r0_vec)):
        r1_1 = r1b_1
        r1_2 = r1b_2
        for j in range(150):
            res1 = fitting(r0_vec[i],r1_1,dire)
            res2 = fitting(r0_vec[i],r1_2,dire)
            
            dr = (r1_2-r1_1)
            if res1 > res2:
                r1_1 = r1_1 + 0.5*dr
                res[i] = res2

            elif res2 > res1:
                r1_2 = r1_2 - 0.5*dr
                res[i] = res1

            else:
                r1[i] = r1_1
                res[i] = res1
                break
            r1[i] = 0.5*(r1_1+r1_2)
    
    indr = np.argmin(res)
    r0,r1 = r0_vec[indr],r1[indr]

    ind=np.concatenate((np.where(rvec<=r0)[0],np.where(rvec>=r1)[0]))
    
    CF_nw = semilogx_interp1d(rvec[ind],rvec[ind]**2*iCF(rvec[ind]),
                               kind='cubic',fill_value='extrapolate',bounds_error=False)(rvec)/(rvec**2)
    
    #Undo the FT to get the unwiggled Pk (use rh to avoid ringing)
    ift = mcfit.xi2P(rvec, l=0, lowring=True)
    kkvec, Pk_nw = ift(CF_nw, extrap=True)
    iPk_nw = semilogx_interp1d(kkvec,Pk_nw,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    iOlin = semilogx_interp1d(kkvec,iPk(kkvec)/iPk_nw(kkvec),kind='cubic',fill_value='extrapolate',bounds_error=False)
    #return iPkh, iPkh_nw, iCFh, iCFh_nw
    return iPk_nw, iOlin

def get_templatesk04(COSMO,z,hfid=None,r0bounds=None,r1bounds=None):
    Nk=4096
    kmin = 0.0010
    kmax = 20.
    factor=0.01
    
    kmax_class = 3.
    
    kh = np.logspace(np.log10(kmin),np.log10(kmax_class),Nk) 
    pk = np.zeros(Nk)
    if not hfid:
        k = kh*COSMO.h() #so here the kh does not have unit of h. 
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*COSMO.h()**3
    else:
        k = kh*hfid
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*hfid**3
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    kh = np.logspace(np.log10(kmin),np.log10(kmax),Nk) #The previous kh up to k_max (not k_max class)
    Pkh = iPk(kh)
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)    #Probably not needed, but for consistency                               
                               
    #Fourier Transform to get it to CF
    ft = mcfit.P2xi(kh, l=0, lowring=True)
    rvec, CF = ft(Pkh, extrap=True)
    iCF = semilogx_interp1d(rvec,CF,kind='cubic',bounds_error=False,fill_value='extrapolate')
    kdum = np.logspace(-4,-3,1000)
    dire = dict(iCF=iCF,iPk=iPk,rvec=rvec,kdum=kdum)
    
    if not r0bounds:
        r0_vec = np.linspace(60,69,40)
    else:
        r0_vec = np.linspace(r0bounds[0],r0bounds[1],40)
    res = np.zeros(len(r0_vec))
    r1 = np.zeros(len(r0_vec))
    if not r1bounds:
        r1b_1,r1b_2 = 200,300
    else:
        r1b_1,r1b_2 = r1bounds[0],r1bounds[1]
    for i in range(len(r0_vec)):
        r1_1 = r1b_1
        r1_2 = r1b_2
        for j in range(150):
            res1 = fitting(r0_vec[i],r1_1,dire)
            res2 = fitting(r0_vec[i],r1_2,dire)
            
            dr = (r1_2-r1_1)
            if res1 > res2:
                r1_1 = r1_1 + 0.5*dr
                res[i] = res2

            elif res2 > res1:
                r1_2 = r1_2 - 0.5*dr
                res[i] = res1

            else:
                r1[i] = r1_1
                res[i] = res1
                break
            r1[i] = 0.5*(r1_1+r1_2)
    
    indr = np.argmin(res)
    r0,r1 = r0_vec[indr],r1[indr]

    ind=np.concatenate((np.where(rvec<=r0)[0],np.where(rvec>=r1)[0]))
    
    CF_nw = semilogx_interp1d(rvec[ind],rvec[ind]**2*iCF(rvec[ind]),
                               kind='cubic',fill_value='extrapolate',bounds_error=False)(rvec)/(rvec**2)
    
    #Undo the FT to get the unwiggled Pk (use rh to avoid ringing)
    ift = mcfit.xi2P(rvec, l=0, lowring=True)
    kkvec, Pk_nw = ift(CF_nw, extrap=True)
    iPk_nw = semilogx_interp1d(kkvec,Pk_nw,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    iOlin = semilogx_interp1d(kkvec,iPk(kkvec)/iPk_nw(kkvec),kind='cubic',fill_value='extrapolate',bounds_error=False)
    #return iPkh, iPkh_nw, iCFh, iCFh_nw
    return iPk_nw, iOlin
    
def smoothing_function(interpolation_points, pk_camb, degree=13, sigma=1, weight=0.5, **kwargs):
    """ Smooth power spectrum based on Hinton 2017 polynomial method """
    # logging.debug("Smoothing spectrum using Hinton 2017 method")
    log_ks = np.log(interpolation_points)
    log_pk = np.log(pk_camb)
    index = np.argmax(pk_camb)
    maxk2 = log_ks[index]
    gauss = np.exp(-0.5 * np.power(((log_ks - maxk2) / sigma), 2))
    w = np.ones(pk_camb.size) - weight * gauss
    z = np.polyfit(log_ks, log_pk, degree, w=w)
    p = np.poly1d(z)
    polyval = p(log_ks)
    pk_smoothed = np.exp(polyval)
    return pk_smoothed
def create_Pmsm_template(interpolation_points,pk_camb,smooth_factor):
    """ fixed assuming fiducial cosmology (Bernal. Equation 7)"""
    ys_smooth = smoothing_function(interpolation_points, pk_camb, degree=13, sigma=1, weight=0.5)
    Pmsm = interpolate.UnivariateSpline(interpolation_points, ys_smooth, s=smooth_factor )
    return Pmsm
    
def get_templates_barry(COSMO,z,hfid=None,r0bounds=None,r1bounds=None):
    Nk=4096*10
    kmin = 1e-5
    kmax = 20.
    factor=0.01

    kmax_class = 3.

    kh = np.logspace(np.log10(kmin),np.log10(kmax_class),Nk)
    pk = np.zeros(Nk)
    if not hfid:
        k = kh*COSMO.h() #so here the kh does not have unit of h.
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*COSMO.h()**3
    else:
        k = kh*hfid
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*hfid**3

    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)

    kh = np.logspace(np.log10(kmin),np.log10(kmax),Nk) #The previous kh up to k_max (not k_max class)
    Pkh = iPk(kh)

    iPk_nw = create_Pmsm_template(kh,Pkh,smooth_factor=20)
    iOlin = semilogx_interp1d(kh,iPk(kh)/iPk_nw(kh),kind='cubic',fill_value='extrapolate',bounds_error=False)
    return iPk_nw, iOlin
    
def get_templates_my_horn(kh,Pkh,hfid=None,h_horn=None):
    Nk=4096
    kmin = 1e-4 #1e-5
    kmax = 0.29 #20.
    factor=0.01

    kmax_class = 3.

    kh = kh*h_horn/hfid
    Pkh = Pkh*hfid**3/h_horn**3

    iPk = interpolate.UnivariateSpline(kh,Pkh, s=20 ) #interp1d(kh, Pkh, kind='cubic') #
   # iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)
   # kh = np.logspace(np.log10(kmin),np.log10(kmax),Nk)
    Pkh2 = iPk(kh)
    iPk_nw = create_Pmsm_template(kh,Pkh,smooth_factor=20)
   # plt.plot(kh,iPk_nw(kh))

    iOlin = semilogx_interp1d(kh,iPk(kh)/iPk_nw(kh),kind='cubic',fill_value='extrapolate',bounds_error=False)
    return iPk_nw, iOlin
    
def get_templatesh(COSMO,z,hfid=None,hh=None,r0bounds=None,r1bounds=None):
    Nk=4096
    kmin = 1e-5
    kmax = 20.
    factor=0.01
    
    kmax_class = 3.
    
    kh = np.logspace(np.log10(kmin),np.log10(kmax_class),Nk) 
    pk = np.zeros(Nk)
    if not hfid:
        k = kh*hh #so here the kh does not have unit of h. 
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*hh**3
    else:
        k = kh*hfid
        for ik in range(Nk):
            pk[ik] = COSMO.pk(k[ik],z)
        Pkh = pk*hfid**3
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    kh = np.logspace(np.log10(kmin),np.log10(kmax),Nk) #The previous kh up to k_max (not k_max class)
    Pkh = iPk(kh)
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)    #Probably not needed, but for consistency                               
                               
    #Fourier Transform to get it to CF
    ft = mcfit.P2xi(kh, l=0, lowring=True)
    rvec, CF = ft(Pkh, extrap=True)
    iCF = semilogx_interp1d(rvec,CF,kind='cubic',bounds_error=False,fill_value='extrapolate')
    kdum = np.logspace(-4,-3,1000)
    dire = dict(iCF=iCF,iPk=iPk,rvec=rvec,kdum=kdum)
    
    if not r0bounds:
        r0_vec = np.linspace(60,69,40)
    else:
        r0_vec = np.linspace(r0bounds[0],r0bounds[1],40)
    res = np.zeros(len(r0_vec))
    r1 = np.zeros(len(r0_vec))
    if not r1bounds:
        r1b_1,r1b_2 = 200,300
    else:
        r1b_1,r1b_2 = r1bounds[0],r1bounds[1]
    for i in range(len(r0_vec)):
        r1_1 = r1b_1
        r1_2 = r1b_2
        for j in range(150):
            res1 = fitting(r0_vec[i],r1_1,dire)
            res2 = fitting(r0_vec[i],r1_2,dire)
            
            dr = (r1_2-r1_1)
            if res1 > res2:
                r1_1 = r1_1 + 0.5*dr
                res[i] = res2

            elif res2 > res1:
                r1_2 = r1_2 - 0.5*dr
                res[i] = res1

            else:
                r1[i] = r1_1
                res[i] = res1
                break
            r1[i] = 0.5*(r1_1+r1_2)
    
    indr = np.argmin(res)
    r0,r1 = r0_vec[indr],r1[indr]

    ind=np.concatenate((np.where(rvec<=r0)[0],np.where(rvec>=r1)[0]))
    
    CF_nw = semilogx_interp1d(rvec[ind],rvec[ind]**2*iCF(rvec[ind]),
                               kind='cubic',fill_value='extrapolate',bounds_error=False)(rvec)/(rvec**2)
    
    #Undo the FT to get the unwiggled Pk (use rh to avoid ringing)
    ift = mcfit.xi2P(rvec, l=0, lowring=True)
    kkvec, Pk_nw = ift(CF_nw, extrap=True)
    iPk_nw = semilogx_interp1d(kkvec,Pk_nw,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    iOlin = semilogx_interp1d(kkvec,iPk(kkvec)/iPk_nw(kkvec),kind='cubic',fill_value='extrapolate',bounds_error=False)
    #return iPkh, iPkh_nw, iCFh, iCFh_nw
    return iPk_nw, iOlin
def get_templates_camb(plin_camb,z,hfid=None,r0bounds=None,r1bounds=None):
    Nk=4096
    kmin = 1e-5
    kmax = 20.
    factor=0.01

    kmax_class = 3.

    Pkh = plin_camb
    kh = np.logspace(np.log10(kmin),np.log10(kmax_class),Nk)

    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    kh = np.logspace(np.log10(kmin),np.log10(kmax),Nk) #The previous kh up to k_max (not k_max class)
    Pkh = iPk(kh)
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)    #Probably not needed, but for consistency                               
                               
    #Fourier Transform to get it to CF
    ft = mcfit.P2xi(kh, l=0, lowring=True)
    rvec, CF = ft(Pkh, extrap=True)
    iCF = semilogx_interp1d(rvec,CF,kind='cubic',bounds_error=False,fill_value='extrapolate')
    kdum = np.logspace(-4,-3,1000)
    dire = dict(iCF=iCF,iPk=iPk,rvec=rvec,kdum=kdum)
    
    if not r0bounds:
        r0_vec = np.linspace(60,69,40)
    else:
        r0_vec = np.linspace(r0bounds[0],r0bounds[1],40)
    res = np.zeros(len(r0_vec))
    r1 = np.zeros(len(r0_vec))
    if not r1bounds:
        r1b_1,r1b_2 = 200,300
    else:
        r1b_1,r1b_2 = r1bounds[0],r1bounds[1]
    for i in range(len(r0_vec)):
        r1_1 = r1b_1
        r1_2 = r1b_2
        for j in range(150):
            res1 = fitting(r0_vec[i],r1_1,dire)
            res2 = fitting(r0_vec[i],r1_2,dire)
            
            dr = (r1_2-r1_1)
            if res1 > res2:
                r1_1 = r1_1 + 0.5*dr
                res[i] = res2

            elif res2 > res1:
                r1_2 = r1_2 - 0.5*dr
                res[i] = res1

            else:
                r1[i] = r1_1
                res[i] = res1
                break
            r1[i] = 0.5*(r1_1+r1_2)
    
    indr = np.argmin(res)
    r0,r1 = r0_vec[indr],r1[indr]

    ind=np.concatenate((np.where(rvec<=r0)[0],np.where(rvec>=r1)[0]))
    
    CF_nw = semilogx_interp1d(rvec[ind],rvec[ind]**2*iCF(rvec[ind]),
                               kind='cubic',fill_value='extrapolate',bounds_error=False)(rvec)/(rvec**2)
    
    #Undo the FT to get the unwiggled Pk (use rh to avoid ringing)
    ift = mcfit.xi2P(rvec, l=0, lowring=True)
    kkvec, Pk_nw = ift(CF_nw, extrap=True)
    iPk_nw = semilogx_interp1d(kkvec,Pk_nw,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    iOlin = semilogx_interp1d(kkvec,iPk(kkvec)/iPk_nw(kkvec),kind='cubic',fill_value='extrapolate',bounds_error=False)
    #return iPkh, iPkh_nw, iCFh, iCFh_nw
    return iPk_nw, iOlin, iPk
def get_templates_horn(kh,Pkh,z,hfid=None,h_horn=None,r0bounds=None,r1bounds=None):
    Nk=4096
    kmin = 1e-5
    kmax = 20.
    factor=0.01
    
    kmax_class = 3.
    
   # kh = kh#np.logspace(np.log10(kmin),np.log10(kmax_class),Nk) 
    # pk = np.zeros(Nk)
     #if not hfid:
    #     k = kh*COSMO.h() #so here the kh does not have unit of h. 
    #     for ik in range(Nk):
    #         pk[ik] = COSMO.pk(k[ik],z)
    #     Pkh = pk*COSMO.h()**3
    # else:
    kh = kh*h_horn/hfid #first get rid of unit h. Then add a unit of hfid. 
    Pkh = Pkh*hfid**3/h_horn**3
 #   iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
   # kh = np.logspace(np.log10(kmin),np.log10(kmax),Nk) #The previous kh up to k_max (not k_max class)
  #  Pkh = iPk(kh)
    iPk = log_interp1d(kh,Pkh,kind='cubic',fill_value='extrapolate',bounds_error=False)    #Probably not needed, but for consistency                               
                               
    #Fourier Transform to get it to CF
    ft = mcfit.P2xi(kh, l=0, lowring=True)
    rvec, CF = ft(Pkh, extrap=True)
    iCF = semilogx_interp1d(rvec,CF,kind='cubic',bounds_error=False,fill_value='extrapolate')
    kdum = np.logspace(-4,-3,1000)
    dire = dict(iCF=iCF,iPk=iPk,rvec=rvec,kdum=kdum)
    
    if not r0bounds:
        r0_vec = np.linspace(60,69,40)
    else:
        r0_vec = np.linspace(r0bounds[0],r0bounds[1],40)
    res = np.zeros(len(r0_vec))
    r1 = np.zeros(len(r0_vec))
    if not r1bounds:
        r1b_1,r1b_2 = 200,300
    else:
        r1b_1,r1b_2 = r1bounds[0],r1bounds[1]
    for i in range(len(r0_vec)):
        r1_1 = r1b_1
        r1_2 = r1b_2
        for j in range(150):
            res1 = fitting(r0_vec[i],r1_1,dire)
            res2 = fitting(r0_vec[i],r1_2,dire)
            
            dr = (r1_2-r1_1)
            if res1 > res2:
                r1_1 = r1_1 + 0.5*dr
                res[i] = res2

            elif res2 > res1:
                r1_2 = r1_2 - 0.5*dr
                res[i] = res1

            else:
                r1[i] = r1_1
                res[i] = res1
                break
            r1[i] = 0.5*(r1_1+r1_2)
    
    indr = np.argmin(res)
    r0,r1 = r0_vec[indr],r1[indr]

    ind=np.concatenate((np.where(rvec<=r0)[0],np.where(rvec>=r1)[0]))
    
    CF_nw = semilogx_interp1d(rvec[ind],rvec[ind]**2*iCF(rvec[ind]),
                               kind='cubic',fill_value='extrapolate',bounds_error=False)(rvec)/(rvec**2)
    
    #Undo the FT to get the unwiggled Pk (use rh to avoid ringing)
    ift = mcfit.xi2P(rvec, l=0, lowring=True)
    kkvec, Pk_nw = ift(CF_nw, extrap=True)
    iPk_nw = semilogx_interp1d(kkvec,Pk_nw,kind='cubic',fill_value='extrapolate',bounds_error=False)
    
    iOlin = semilogx_interp1d(kkvec,iPk(kkvec)/iPk_nw(kkvec),kind='cubic',fill_value='extrapolate',bounds_error=False)
    #return iPkh, iPkh_nw, iCFh, iCFh_nw
    return iPk_nw, iOlin
    
def get_covmat(ki_grid,mui_grid,Pkmu,dk,V,hexadecapole):
    '''
    Compute the cross-terms of the covariance matrix (cosmic variance, including shot noise)
    '''
    mu = mui_grid[:,0]
    nk = len(dk)
    Nmodes = ki_grid**2*dk*V/4./np.pi**2.

    L2 = legendre(2)(mui_grid)

    covmat_00 = 0.5*np.trapz(Pkmu**2/Nmodes,mu,axis=0)
    covmat_02 = 5./2.*np.trapz(Pkmu**2*L2/Nmodes,mu,axis=0) 
    covmat_22 = 25./2.*np.trapz(Pkmu**2*L2*L2/Nmodes,mu,axis=0)

    if hexadecapole:
        L4 = legendre(4)(mui_grid)
        covmat_04 = 9./2.*np.trapz(Pkmu**2*L4/Nmodes,mu,axis=0)
        covmat_24 = 45./2.*np.trapz(Pkmu**2*L2*L4/Nmodes,mu,axis=0)
        covmat_44 = 81./2.*np.trapz(Pkmu**2*L4*L4/Nmodes,mu,axis=0)
        Nmul = 3
                
        errh = np.zeros((Nmul*nk,Nmul*nk))
        errh[np.ix_(np.arange(0,nk),np.arange(0,nk))] = np.diag(covmat_00)
        errh[np.ix_(np.arange(0,nk),np.arange(nk,nk*2))] = np.diag(covmat_02)
        errh[np.ix_(np.arange(nk,nk*2),np.arange(0,nk))] = np.diag(covmat_02)
        errh[np.ix_(np.arange(nk,nk*2),np.arange(nk,nk*2))] = np.diag(covmat_22)
        errh[np.ix_(np.arange(nk*2,nk*3),np.arange(nk,nk*2))] = np.diag(covmat_24)
        errh[np.ix_(np.arange(nk,nk*2),np.arange(nk*2,nk*3))] = np.diag(covmat_24)
        errh[np.ix_(np.arange(nk*2,nk*3),np.arange(nk*2,nk*3))] = np.diag(covmat_44)
    else:
        Nmul = 2
        
        errh = np.zeros((Nmul*nk,Nmul*nk))
        errh[np.ix_(np.arange(0,nk),np.arange(0,nk))] = np.diag(covmat_00)
        errh[np.ix_(np.arange(0,nk),np.arange(nk,nk*2))] = np.diag(covmat_02)
        errh[np.ix_(np.arange(nk,nk*2),np.arange(0,nk))] = np.diag(covmat_02)
        errh[np.ix_(np.arange(nk,nk*2),np.arange(nk,nk*2))] = np.diag(covmat_22)
        
    return errh
    
# ~ def get_AP_rescale(z,COSMOfid,COSMOdat):
    # ~ '''
    # ~ Get the value for alpha_perp and alpha_par for a given cosmology of the data and an assumed fiducial
    
    # ~ Returns alpha_perp,alpha_par
    # ~ '''    
    # ~ #fiducial distances
    # ~ fid_DA_rs = COSMOfid.angular_distance(z)/COSMOfid.rs_drag()
    # ~ fid_H_rs = COSMOfid.Hubble(z)*COSMOfid.rs_drag()
    
    # ~ #data distances
    # ~ data_DA_rs = COSMOdat.angular_distance(z)/COSMOdat.rs_drag()
    # ~ data_H_rs = COSMOdat.Hubble(z)*COSMOdat.rs_drag()
    
    # ~ alpha_perp = data_DA_rs/fid_DA_rs
    # ~ alpha_par = fid_H_rs/data_H_rs
    
    # ~ return alpha_perp,alpha_par
    
def dispersion_k(covmat):
    '''
    Get the dispersion in k space for the power spectrum around the true values
    accounting for the covariance between multipoles
    '''
    size = covmat.shape[0]
    #diagonalize covariance
    w,P = eig(covmat)
    D = np.dot(inv(P),np.dot(covmat,P))
    disp_diag = np.zeros(size)
    for i in range(size):
        disp_diag[i] = normal(0,D[i,i]**0.5)
    disp_k = np.dot(P,disp_diag)
    return disp_k

#####################
##### Functions #####
#####################

def get_cosmo(modelpars):
    '''
    Initialize CLASS, using the cosmological parameters for a given cosmology (modelpars for a dictionary)
    '''
    cosmo_common = dict(N_ncdm = 1,T_ncdm = 0.71611,N_ur=2.0328,m_ncdm=0.06)
    cosmo_output = dict(output = 'mPk',z_pk='0.38,1.',modes = 's')
    cosmo_maxpk = {'P_k_max_h/Mpc':'40'}
    cosmo_halofit = {'non linear':'halofit'}

    #COSMO_dict = merge_dicts([cosmo_dict,cosmo_common,cosmo_output,cosmo_halofit,cosmo_maxpk])
    COSMO_dict = merge_dicts([cosmo_common,cosmo_output,cosmo_maxpk,modelpars])
        
    COSMO = Class()
    COSMO.set(COSMO_dict)
    COSMO.compute()
    
    return COSMO
    
def basic_cosmo(model):
    '''
    Returns the basic precision and needed params for a model
    '''
    if 'EDE' in model and 'lcdm' not in model:

        return {'scf_potential':'axion',
                'gauge':'synchronous','scf_evolve_as_fluid':'no',
                'scf_evolve_like_axionCAMB':'no','threshold_scf_fluid_m_over_H':1e-3,'do_shooting':'yes',
                'do_shooting_scf':'yes','back_integration_stepsize':1e-5,'scf_has_perturbations':'yes',
                'attractor_ic_scf':'no'}
    else:
        return {} 
    # ~ else:
        # ~ raise Exception('Please, note that your model of choice is not coded, choose between EDE and lcdm, or\
         # ~ include it in basic_cosmo')
    

def Pk_AP(ki_grid,mui_grid,iP_nw,iOlin,theta,recon):
    alpha_perp = theta[0]
    alpha_par = theta[1]
    B = theta[2]
    beta = theta[3]
    sigma_fog = theta[-3]
    Sigma_perp = theta[-2]
    Sigma_par = theta[-1]
    
    F = alpha_par/alpha_perp
    prefac = 1./alpha_perp**2/alpha_par
    
    #Get "real" k and mu
    kprime = np.zeros(ki_grid.shape)
    mu_prime = mui_grid/F/np.sqrt(1.+mui_grid**2.*(1./F/F-1))
    for imu in range(ki_grid.shape[0]):
        kprime[imu,:] = ki_grid[imu,:]/alpha_perp*np.sqrt(1.+mui_grid[imu,:]**2*(1./F/F-1))
    #Obtain the corresponding P_nw and Olin for the "real" k
    Pknw = iP_nw(kprime)
    Oklin = iOlin(kprime)
    
    #NL BAO evolution. Values of R smoothing
    if recon:
        Sigma_smooth = 15.
        R = 1. - np.exp(-0.5*(kprime*Sigma_smooth)**2)
    else: 
        R = 1.
    NL_BAO = np.exp(-0.5*(kprime**2*mu_prime**2*Sigma_par**2+kprime**2*(1.-mu_prime**2)*Sigma_perp**2))
    #Apply RSD for the "real" mu
    kaiser = (1.+beta*mu_prime**2*R)**2
   # loren = 1./(1.+0.5*(kprime*mu_prime*sigma_fog)**2)
    loren = 1./(1.+0.5*(kprime*mu_prime*sigma_fog)**2)**2
    
    P_dewig = prefac*B*kaiser*loren*Pknw
    #Pkmu with wiggles
    Pkmu = P_dewig*(1.+(Oklin-1.)*NL_BAO)
    
    return Pkmu

######################
##### Likelihood #####
######################

def lkl_pk(theta,k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon):
    alpha_perp = theta[0]
    alpha_par = theta[1]
    B = theta[2]
    beta = theta[3]
    coeffs = theta[4:-3]
    sigma_fog = theta[-3]
    Sigma_perp = theta[-2]
    Sigma_par = theta[-1]
    
    #Get kgrid and mugrid
    nmu = 1000
    mu_edge = np.linspace(-1,1,nmu+1)
    mu = (mu_edge[0:nmu+1-1]+mu_edge[1:nmu+1])/2.
    ki_grid,mui_grid = np.meshgrid(k,mu)
    
    #AP effect for Pkmu:
    Pkmu = Pk_AP(ki_grid,mui_grid,iPk_nw,iOlin,theta,recon)
    
    Pmul = get_multipoles(ki_grid,mui_grid,Pkmu,coeffs,hexadecapole,recon)
    
    chi2 = np.dot(np.dot(Pmul-datPk,invcovmat),Pmul-datPk)

    return -0.5*chi2
    
def lkl_pk_Pshot(theta,k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon,n):
    alpha_perp = theta[0]
    alpha_par = theta[1]
    B = theta[2]
    beta = theta[3]
    coeffs = theta[4:-3]
    sigma_fog = theta[-3]
    Sigma_perp = theta[-2]
    Sigma_par = theta[-1]
    
    #Get kgrid and mugrid
    nmu = 1000
    mu_edge = np.linspace(-1,1,nmu+1)
    mu = (mu_edge[0:nmu+1-1]+mu_edge[1:nmu+1])/2.
    ki_grid,mui_grid = np.meshgrid(k,mu)
    
    #AP effect for Pkmu:
    Pkmu = Pk_AP(ki_grid,mui_grid,iPk_nw,iOlin,theta,recon)+1/n
    
    Pmul = get_multipoles(ki_grid,mui_grid,Pkmu,coeffs,hexadecapole,recon)
    
    chi2 = np.dot(np.dot(Pmul-datPk,invcovmat),Pmul-datPk)

    return -0.5*chi2
    
def lkl_pk_k04(theta,k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon):
    alpha_perp = theta[0]
    alpha_par = theta[1]
    B = theta[2]
    beta = theta[3]
    coeffs = theta[4:-3]
    sigma_fog = theta[-3]
    Sigma_perp = theta[-2]
    Sigma_par = theta[-1]
    
    #Get kgrid and mugrid
    nmu = 1000
    mu_edge = np.linspace(-1,1,nmu+1)
    mu = (mu_edge[0:nmu+1-1]+mu_edge[1:nmu+1])/2.
    ki_grid,mui_grid = np.meshgrid(k,mu)
    
    #AP effect for Pkmu:
    Pkmu = Pk_AP(ki_grid,mui_grid,iPk_nw,iOlin,theta,recon)
    
    Pmul = get_multipoles(ki_grid,mui_grid,Pkmu,coeffs,hexadecapole,recon)
    
    chi2 = np.dot(np.dot(Pmul-datPk,invcovmat),Pmul-datPk)

    return -0.5*chi2
#probsu
def lnprob(theta, k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon,prior_min,prior_max,n,Pshot_T):

    lp = lnprior(theta, prior_min, prior_max)
    if not np.isfinite(lp):
        return -np.inf
    if Pshot_T == False:
        lkl = lkl_pk(theta, k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon)
    elif Pshot_T == True:
        lkl = lkl_pk_Pshot(theta, k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon,n)

    # ~ global iters
    # ~ global chi2_0
    
    # ~ chi2_0 = min(chi2_0,-2*lkl)
    # ~ count = iters/1e2
    # ~ if round(count)==count:
        # ~ print "iters = ", iters, 'chi2 = ', chi2_0
    # ~ iters += 1
    
    if not np.isfinite(lkl):
        return -np.inf
    else:
        return lp + lkl
     
def lnprobk04(theta, k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon,prior_min,prior_max,n,Pshot_T):

    lp = lnprior(theta, prior_min, prior_max)
    if not np.isfinite(lp):
        return -np.inf
    if Pshot_T == False:
        lkl = lkl_pk_k04(theta, k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon)
    elif Pshot_T == True:
        lkl = lkl_pk_Pshot(theta, k,datPk,invcovmat,iPk_nw,iOlin,hexadecapole,recon,n)

    # ~ global iters
    # ~ global chi2_0
    
    # ~ chi2_0 = min(chi2_0,-2*lkl)
    # ~ count = iters/1e2
    # ~ if round(count)==count:
        # ~ print "iters = ", iters, 'chi2 = ', chi2_0
    # ~ iters += 1
    
    if not np.isfinite(lkl):
        return -np.inf
    else:
        return lp + lkl   
        
#flat priors
def lnprior(theta,prior_min,prior_max):

    if (prior_min <= theta).all() == True and (theta <= prior_max).all() == True:
        return 0.0
    else:
        return -np.inf
