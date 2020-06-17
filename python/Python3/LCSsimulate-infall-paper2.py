#!/usr/bin/env python

'''

USAGE
- from within ipython

%run ~/Dropbox/pythonCode/LCSsimulate-infall.py

t = run_sim(tmax=0,drdt_step=0.05,nrandom=1000)
t = run_sim(tmax=1,drdt_step=0.05,nrandom=1000)
t = run_sim(tmax=2,drdt_step=0.05,nrandom=1000)
t = run_sim(tmax=3,drdt_step=0.05,nrandom=1000)
t = run_sim(tmax=4,drdt_step=0.05,nrandom=1000)
t = run_sim(tmax=5,drdt_step=0.05,nrandom=1000)


Written by Rose A. Finn, 2/21/18

'''



from astropy.io import fits,ascii
from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import argparse
from scipy.stats import ks_2samp
# the incomplete gamma function, for integrating the sersic profile
from scipy.special import gammainc
#from astropy.table import Table

#from anderson import *
###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Program to run simulation for LCS paper 2')
parser.add_argument('--use24', dest = 'use24', default = False, action='store_true',help = 'use 24um profile parameters when calculating expected SFR of sim galaxies (should set this).')
#parser.add_argument('--diskonly', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')

args = parser.parse_args()

###########################
##### DEFINITIONS
###########################

mipspixelscale=2.45

#R24/Rd err  core_flag   B/T 

'''
infile = '/Users/rfinn/research/LocalClusters/catalogs/sizes.txt'
infile = '/home/rfinn/research/LCS/tables/sizes.txt'
sizes = np.loadtxt(infile)
size_ratio = np.array(sizes[:,0],'f')
size_err = np.array(sizes[:,1],'f')
core_flag= np.array(sizes[:,2],'bool')
bt = np.array(sizes[:,3],'f')
'''
###########################
##### READ IN DATA FILE
##### WITH FITTED PARAMETERS
###########################

# updated input file to include SFR, n
#infile = '/home/rfinn/research/LCS/tables/LCS-simulation-data.fits'
infile = '/home/rfinn/research/LCS/tables/LCS_all_size_KE_SFR.fits'
sizes = Table.read(infile)
# keep only galaxies in paper 1 sample
sizes = sizes[sizes['sampleflag']]
              
size_ratio = sizes['sizeratio']
size_err = sizes['sizeratio_err']
core_flag= sizes['membflag']
bt = sizes['B_T_r']
logSFR = sizes['logSFR_NUVIR_KE']
nsersic = sizes['ng'] # simard sersic index
Re = sizes['Re'] # rband disk scale length from simard
Re24 = sizes['fcre1']*mipspixelscale # 24um Re in arcsec
nsersic24 = sizes['fcnsersic1']*mipspixelscale # 24um Re in arcsec

if args.use24:
    Re = Re24
    nsersic = nsersic24
    
'''
sizes = ascii.read(infile)
size_ratio = np.array(sizes['col0'],'f')
size_err = np.array(sizes['col1'],'f')
core_flag = np.array(sizes['col2'],'bool')
#bt = np.array(sizes[:,3],'f')
'''


## split sizes
core = size_ratio[core_flag]
external = size_ratio[~core_flag]

core_logsfr = logSFR[core_flag]
external_logsfr = logSFR[~core_flag]

core_nsersic = nsersic[core_flag]
external_nsersic = nsersic[~core_flag]

## infall rates
# uniform distribution between 0 and tmax Gyr
tmax = 2. # max infall time in Gyr

def get_frac_flux_retained(n,ratio_before,ratio_after):
    # ratio_before = the initial value of R/Re
    # ratio_after = the final value of R/Re
    # n = sersic index of profile

    # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
    
    # calculate the loss in light
    bn = 1.999*n-0.327
    x_before = bn*(ratio_before)**(1./n)
    x_after = bn*(ratio_after)**(1./n)    
    frac_retained = gammainc(2*n,x_after)/gammainc(2*n,x_before) 
    return frac_retained

'''
I think the right way to do the integration (inspiration during my jog today) is to integrate
the sersic profile of the external profile(Re24, n24) from zero to inf.

for sim core galaxy, integrate sersic profile with new Re from zero to inf.  
Don't know what to do with sersic index.  As a first attempt, leave it 
the same as for the external galaxy.

gammainc = 1 when integrating to infinity

'''
def get_frac_flux_retained0(n,ratio_before,ratio_after):
    # ratio_before = the initial value of R24/Re_r
    # ratio_after = the final value of R24/Re_r
    # n = sersic index of profile

    # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
    
    # calculate the loss in light
    bn = 1.999*n-0.327
    x_before = bn*(ratio_before)**(1./n)
    x_after = bn*(ratio_after)**(1./n)

    # everything is the same except for Re(24)
    frac_retained = (ratio_before/ratio_after)**2
    return frac_retained

def run_sim(tmax = 2.,nrandom=100,drdt_step=.1,plotsingle=True):
    ks_D_min = 0
    ks_p_max = 0
    drdt_best = 0
    drdt_multiple = []
    drdtmin=-2
    drdtmax=0
    all_p = np.zeros(int(nrandom*(drdtmax-drdtmin)/drdt_step))
    all_p_sfr = np.zeros(int(nrandom*(drdtmax-drdtmin)/drdt_step))    
    all_drdt = np.zeros(int(nrandom*(drdtmax - drdtmin)/drdt_step))
    i=0
    for drdt in np.arange(drdtmin,drdtmax,drdt_step):
        #print drdt
        for j in range(nrandom):
            #sim_core = np.random.choice(external,size=len(core)) + drdt*np.random.uniform(low=0, high=tmax, size=len(core))
            infall_times = np.linspace(0,tmax,len(external))
            actual_infall_times = np.random.choice(infall_times, len(infall_times))
            sim_core = external + drdt*actual_infall_times

            # to implement SFR decrease
            # need SFR, sersic index, and Re for all galaxies
            # greg has equation for integral of sersic profile
            # create function that gives flux/flux_0 for sersic profile that
            # has Re shrink by some factor
            #frac_retained = get_frac_flux_retained(external_nsersic,external,sim_core)
            frac_retained = get_frac_flux_retained0(external_nsersic,external,sim_core)            
            ########################################################
            # get predicted SFR of core galaxies by multiplying the
            # distribution of SFRs from the external samples by the
            # flux ratio you would expect from shrinking the
            # external sizes to the sim_core sizes
            # SFRs are logged
            sim_core_logsfr = np.log10(frac_retained) + external_logsfr

            # not sure what this statement does...
            if sum(sim_core > 0.)*1./len(sim_core) < .2:
                continue

            # compare distribution of sizes between core (measured)
            # and the simulated core
            D,p=ks_2samp(core,sim_core[sim_core > 0.])
            #D,t,p=anderson_ksamp([core,sim_core[sim_core > 0.]])
            #print D,p
            if p > ks_p_max:
                #print 'got a good one, drdt = ',drdt
                ks_p_max = p
                best_sim_core = sim_core
                best_drdt = drdt
                drdt_multiple = []
            elif abs(p - ks_p_max) < 1.e-5:
                #print('found an equally good model at drdt = ',drdt,p)
                drdt_multiple.append(drdt)
            D2,p2 = ks_2samp(core_logsfr,sim_core_logsfr)
            all_p[i] = p
            all_drdt[i] = drdt
            all_p_sfr[i] = p2
            i += 1
    print('best dr/dt = ',best_drdt)
    print('disk is quenched in %.1f Gyr'%(1./abs(best_drdt)))
    print('fraction that are quenched = %.2f'%(1.*sum(best_sim_core < 0.)/len(best_sim_core)))
    print('KS p value = %.8e'%(ks_p_max))
    if len(drdt_multiple) > 0.:
        print('drdt multiple values of dr/dt')
        for i in range(len(drdt_multiple)):
            print('#################')
            print('\t best dr/dt = ',drdt_multiple[i])
            print('\t disk is quenched in %.1f Gyr'%(1./abs(drdt_multiple[i])))
    #plot_results(core,external,best_sim_core,best_drdt,tmax)
    plot_hexbin(all_drdt,all_p,best_drdt,tmax,gridsize = int(1./drdt_step),plotsingle=plotsingle)

    return best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr


def plot_hexbin(all_drdt,all_p,best_drdt,tmax,gridsize=10,plotsingle=True):
    if plotsingle:
        plt.figure()
    plt.subplots_adjust(bottom=.15,left=.12)
    myvmax = 1.*len(all_drdt)/(gridsize**2)*4
    #print 'myvmax = ',myvmax 
    plt.hexbin(all_drdt, all_p,gridsize=gridsize,cmap='gray_r',vmin=0,vmax=myvmax)
    if plotsingle:
        plt.colorbar(fraction=0.08)
    plt.xlabel(r'$dr/dt \ (Gyr^{-1}) $',fontsize=18)
    plt.ylabel(r'$p-value$',fontsize=18)
    #s = r'$t_{max} = %.1f \ Gyr, \ dr/dt = %.2f \ Gyr^{-1}, \ t_{quench} = %.1f \ Gyr$'%(tmax, best_drdt,1./abs(best_drdt))
    s = r'$t_{max} = %.1f \ Gyr$'%(tmax)
    #plt.text(0.02,.7,s,transform = plt.gca().transAxes)
    plt.title(s,fontsize=18)
    output = 'sim_infall_tmax_%.1f.png'%(tmax)
    plt.savefig(output)

def plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax,plotsingle=True):
    if plotsingle:
        plt.figure(figsize=(8,6))
    plt.scatter(all_p,all_p_sfr,c=all_drdt,s=10,vmin=-1,vmax=0)
    plt.xlabel('$p-value \ size$',fontsize=18)
    plt.ylabel('$p-value \ SFR$',fontsize=18)

    plt.axhline(y=.05,ls='--')
    plt.axvline(x=.05,ls='--')
    plt.axis([-.09,1,-.09,1])
    ax = plt.gca()
    #ax.set_yscale('log')
    if plotsingle:
        plt.colorbar(label='$dr/dt$')        
        plt.savefig('pvalue-SFR-size-tmax'+str(tmax)+'Gyr-shrink0.png')

def plot_multiple_tmax(nrandom=100):
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr=run_sim(tmax=1,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax)    
    plt.subplot(2,2,2)
    run_sim(tmax=2,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplot(2,2,3)
    run_sim(tmax=3,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplot(2,2,4)
    run_sim(tmax=4,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplots_adjust(hspace=.5,bottom=.1)
    plt.savefig('sim_infall_multiple_tmax.pdf')
    plt.savefig('fig18.pdf')

def plot_multiple_tmax_wsfr(nrandom=100):
    plt.figure(figsize=(12,6))
    mytmax = [1,2,3,4]
    allax = []
    for i,tmax in enumerate(mytmax):
        plt.subplot(2,4,i+1)
        best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr=run_sim(tmax=tmax,drdt_step=.05,nrandom=nrandom,plotsingle=False)
        allax.append(plt.gca())
        plt.subplot(2,4,i+5)    
        plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax,plotsingle=False)    
        allax.append(plt.gca())


    plt.subplots_adjust(hspace=.5,wspace=.7,bottom=.1)
    cb = plt.colorbar(ax=allax,label='$dr/dt$')    
    plt.savefig('sim_infall_multiple_tmax_wsfr.pdf')
    plt.savefig('sim_infall_multiple_tmax_wsfr.png')    
    #plt.savefig('fig18.pdf')

def plot_results(core,external,sim_core,best_drdt,tmax):
    plt.figure()
    mybins = np.arange(0,2,.2)
    plt.hist(core,bins=mybins,color='r',histtype='step',label='Core',lw='3',normed=True)
    plt.hist(external,bins=mybins,color='b',ls='-',lw=3,histtype='step',label='External',normed=True)
    plt.hist(sim_core,bins=mybins,color='k',hatch='//',histtype='step',label='Sim Core',normed=True)
    plt.subplots_adjust(bottom=.15)
    plt.xlabel('$R_{24}/R_d$', fontsize=22)
    plt.ylabel('$Frequency$',fontsize=22)
    s = '$dr/dt = %.2f /Gyr$'%(best_drdt)
    plt.text(0.02,.7,s,transform = plt.gca().transAxes)
    s = '$t_{quench} = %.1f  Gyr$'%(1./abs(best_drdt))
    plt.text(0.02,.65,s,transform = plt.gca().transAxes)
    s = '$t_{max} = %.1f  Gyr$'%(tmax)
    plt.text(0.02,.6,s,transform = plt.gca().transAxes)
    plt.legend(loc='upper left')

