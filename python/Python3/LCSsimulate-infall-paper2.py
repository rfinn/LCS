#!/usr/bin/env python

'''
GOAL:
- test several models that describe how disks shrink and integrated SFR decreases as a result of outside-in quenching


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
Updated 2019-2020 to incorporate total SFRs into the comparison.

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
import os
import LCScommon as lcommon

#from anderson import *

homedir = os.getenv("HOME")
plotdir = homedir+'/research/LCS/plots/'
###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Program to run simulation for LCS paper 2')
parser.add_argument('--use24', dest = 'use24', default = True, action='store_false',help = 'use 24um profile parameters when calculating expected SFR of sim galaxies.  default is true')
parser.add_argument('--model', dest = 'model', default = 1, help = 'infall model to use.  default is 1.  \n\tmodel 1 is shrinking 24um effective radius \n\tmodel 2 is truncatingthe 24um emission')
parser.add_argument('--sfrint', dest = 'sfrint', default = 1, help = 'method for integrating the SFR in model 2.  \n\tmethod 1 = integrate external sersic profile out to truncation radius.\n\tmethod 2 = integrate fitted sersic profile out to rmax.')
parser.add_argument('--pvalue', dest = 'pvalue', default = .005, help = 'pvalue threshold to use when plotting fraction of trials below this pvalue.  Default is 0.05 (2sigma).  For ref, 3sigma is .003.')
parser.add_argument('--tmax', dest = 'tmax', default = 3., help = 'maximum infall time.  default is 3 Gyr.  ')

parser.add_argument('--rmax', dest = 'rmax', default = 6., help = 'maximum size of SF disk in terms of Re.  default is 6.  ')
parser.add_argument('--btcut', dest = 'btcut', default = False, action='store_true',help = 'cut sample by B/T < 0.3.  This should be set, but leaving this as an option for backwards compatability..  ')
parser.add_argument('--gsw', dest = 'gsw', default = False, action='store_true',help = 'use GSWLC sfrs instead of MIPS 24+UV SFRs')
parser.add_argument('--sampleks', dest = 'sampleks', default = False, action='store_true',help = 'run KS test to compare core/external size, SFR, Re24 and nsersic24.  default is False.')


args = parser.parse_args()
args.model = int(args.model)
args.sfrint = int(args.sfrint)
args.tmax = float(args.tmax)
args.pvalue = float(args.pvalue)

###########################
##### DEFINITIONS
###########################

mycolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mipspixelscale=2.45
######################################################
## FROM FITTING FIT VS INPUT
## SERSIC PARAMETERS AS A FUNCTION OF RTRUNC
######################################################
sersicN_fit = [ 1.00321952, -1.33892353, -1.58502679]
sersicRe_fit = [ 1.0051858,  -1.6640543,  -1.43415337]
sersicIe_fit = [ 1.03890568, 30.02634861, -3.38602545]

## infall rates
# uniform distribution between 0 and tmax Gyr
tmax = 2. # max infall time in Gyr


###########################
##### READ IN DATA FILE
##### WITH SIZE INFO
###########################

# updated input file to include SFR, n
#infile = homedir+'/research/LCS/tables/LCS-simulation-data.fits'

if args.gsw:
    print('USING GSWLC SFRS')
    # data for the 1490 galaxies that are matched to GSWLC
    infile = homedir+'/research/LCS/tables/LCS_all_size_KE_SFR_GSWLC2_X2.fits'
else:
    print('USING MIPS+GALEX NUV SFRS')
    # data for the full sample of 1800 galaxies
    # this path will use MIPS+UV sfrs
    infile = homedir+'/research/LCS/tables/LCS_all_size_KE_SFR.fits'

sizes = Table.read(infile)
# keep only galaxies in paper 1 sample
sizes = sizes[sizes['sampleflag']]


bt = sizes['B_T_r']
btflag = sizes['B_T_r'] < 0.3
if args.btcut:
    sizes = sizes[btflag]
# keep only B/T < 0.3              

if args.gsw:
    # use gswlc sfrs
    logSFR = sizes['logSFR']
    SFR = 10.**(logSFR)

else:
    # use MIPS+GALEX sfrs
    logSFR = sizes['logSFR_NUVIR_KE']
    SFR = 10.**(logSFR)
print('\nMIN SFR = ',min(SFR))
size_ratio = sizes['sizeratio']
size_err = sizes['sizeratio_err']
core_flag= sizes['membflag']
nsersic = sizes['ng'] # simard sersic index

Re = sizes['Re'] # rband disk scale length from simard
Re24 = sizes['fcre1']*mipspixelscale # 24um Re in arcsec
nsersic24 = sizes['fcnsersic1']*mipspixelscale # 24um Re in arcsec

if args.use24:
    print('\nUsing 24um sizes (this is the right thing to do)\n')
    Re = Re24
    nsersic = nsersic24
    # can't remember what this does, or why I would use something else
    # maybe at some point I was using the R-band effective radius?
    
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

core_sfr = 10.**(logSFR[core_flag])
external_sfr = 10.**(logSFR[~core_flag])

core_nsersic = nsersic[core_flag]
external_nsersic = nsersic[~core_flag]

core_Re24 = Re24[core_flag]
external_Re24 = Re24[~core_flag]

core_nsersic24 = nsersic24[core_flag]
external_nsersic24 = nsersic24[~core_flag]

###########################
##### compare core/external
###########################
if args.sampleks:
    print('\ncore vs external: size distribution')
    lcommon.ks(core,external,run_anderson=True)
    print('\ncore vs external: SFR distribution')
    lcommon.ks(core_sfr,external_sfr,run_anderson=True)
    print('\ncore vs external: Re 24')
    lcommon.ks(core_Re24,external_Re24,run_anderson=True)
    print('\ncore vs external: n sersic distribution')
    lcommon.ks(core_nsersic,external_nsersic,run_anderson=True)
    print("")

###########################
##### FUNCTIONS
###########################


def model2_get_fitted_param(input,coeff):
    # here rtrunc is the truncation radius/Re
    return coeff[0]+coeff[1]*np.exp(coeff[2]*input)

def integrate_sersic(n,Re,Ie,rmax=6):
    bn = 1.999*n-0.327
    x = bn*(rmax/Re)**(1./n)    
    return Ie*Re**2*2*np.pi*n*np.exp(bn)/bn**(2*n)*gammainc(2*n, x)

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
def get_frac_flux_retained0(n,ratio_before,ratio_after,Iboost=1):
    '''
    calculate fraction of flux retained after shrinking Re of sersic profile

    PARAMS
    ------
    * ratio_before: the initial value of R24/Re_r
    * ratio_after: the final value of R24/Re_r
    * n: sersic index of profile

    # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)

    RETURNS
    -------
    * frac_retained: fraction of flux retained
    '''
    
    # calculate the loss in light
    bn = 1.999*n-0.327
    x_before = bn*(ratio_before)**(1./n)
    x_after = bn*(ratio_after)**(1./n)

    # everything is the same except for Re(24)

    ### check this!!!  gamma function might be different???
    frac_retained = Iboost*(ratio_after/ratio_before)**2
    return frac_retained



def get_frac_flux_retained_model2(n,Re,rtrunc=1,rmax=4,version=1,Iboost=1):
    '''
    return fraction of the flux retained by a truncated profile.

    this model integrates the after model to the truncation radius AND
    boosts the central intensity of the after model.
    The sersic index is unchanged.  

    PARAMS
    ------
    * n: sersic index of profile
    * Re: effective radius of sersic profile
    * rtrunc: truncation radius, in terms of Re; default=1
    * rmax: maximum extent of the disk, in terms of Re, for the purpose of integration; default=6
      - this is how far out the original profile is integrated to, rather than infinity
    * version:
      - 1 = integrate the truncated profile
      - 2 = integrate the sersic profile we would measure by fitting a sersic profile to the truncated profile
      - best to use option 1
    * Iboost: factor to boost central intensity by

    RETURNS
    -------
    * fraction of the flux retained
    
    '''
    if version == 1:
        # sersic index of profile
        # Re = effective radius of profile
        # n = sersic index of profile
        # rmax = multiple of Re to use a max radius of integration in the "before" integral
        # rtrunc = max radius to integrate to in "after" integral

        # ORIGINAL
        # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
        # PROCESSED
        # L(<R) = boost*Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
            
        # calculate the loss in light
        
        # this should simplify to the ratio of the incomplete gamma functions
        # ... I think ...

        # this is the same for both
        bn = 1.999*n-0.327
        
        x_after = bn*(rtrunc/Re)**(1./n)
        x_before = bn*(rmax)**(1./n)
        frac_retained = Iboost*gammainc(2*n,x_after)/gammainc(2*n,x_before)
        
    elif version == 2:
        # use fitted values of model to get integrated flux after
        # integral of input sersic profile with integral of fitted sersic profile
        Ie = 1
        sfr_before = integrate_sersic(n,Re,Ie,rmax=rmax)
        
        n2 = n*model2_get_fitted_param(rtrunc/Re,sersicN_fit)
        Re2 = Re*model2_get_fitted_param(rtrunc/Re,sersicRe_fit)
        Ie2 = Ie*model2_get_fitted_param(rtrunc/Re,sersicIe_fit)
        sfr_after = integrate_sersic(n2,Re2,Ie2,rmax=rmax)        
        
        frac_retained = Iboost*sfr_after/sfr_before

        
    return frac_retained


###############################
##### MAIN SIMULATION FUNCTION
###############################


def run_sim(tmax = 2.,nrandom=100,drdtmin=-2,drdt_step=.1,model=1,plotsingle=True,maxboost=5,plotflag=True,rmax=4,boostflag=False):
    '''
    run simulations of disk shrinking

    PARAMS
    ------
    * tmax: maximum time that core galaxies have been in cluster, in Gyr; default = 2
    * nrandom : number of random iterations for each value of dr/dt
    * drdt_step : step size for drdt; range is between -2 and 0
    * model : quenching model to use; can be 1 or 2
      - 1 = shrink Re
      - 2 = truncate disk
    * boostflag : set this to boost central intensity; can set this for both model 1 and 2
    * maxboost : max factor to boost SFRs by; Iboost/Ie
    * rmax : max extent of disk for truncation model in terms of Re; disk shrinks as (rmax - dr/dt*tinfall)*Re_input
    * plotsingle : default is True;
      - use this to print a separate figure;
      - set to False if creating a multi-panel plot
    * plotflag : don't remember what this does


    RETURNS
    -------
    * best_drdt : best dr/dt value (basically meaningless)
      - b/c KS test is good at rejecting null hypothesis, but pvalue of 0.98 is not better than pvalue=0.97
    * best_sim_core : best distribution of core sizes? (basically meaningless)
    * ks_p_max : pvalue for best model (basically meaningless)
    * all_drdt : dr/dt for every model
    * all_p : p value for size comparison for every model
    * all_p_sfr : p value for SFR comparison for every model
    * all_boost : boost value for each model; this will be zeros if model != 3

    '''
    # pass in rmax
    #rmax = float(args.rmax)
    ks_D_min = 0
    ks_p_max = 0
    drdt_best = 0
    drdt_multiple = []
    #drdtmin=-2
    drdtmax=0
    nstep_drdt = int((drdtmax-drdtmin)/drdt_step)
    npoints = int(nstep_drdt*nrandom)
    all_p = np.zeros(npoints)
    all_p_sfr = np.zeros(npoints)
    all_drdt = np.zeros(npoints)
    all_boost = np.zeros(npoints)
    fquench_size = np.zeros(npoints)
    fquench_sfr = np.zeros(npoints)

    if model == 2:
        print('USING MODEL 2')
    for i in range(nstep_drdt):
        drdt  = drdtmin + i*drdt_step

        # repeat nrandom times for each value of dr/dt and tmax
        for j in range(nrandom):
            aindex = nrandom*i+j
            #sim_core = np.random.choice(external,size=len(core)) + drdt*np.random.uniform(low=0, high=tmax, size=len(core))
            infall_times = np.linspace(0,tmax,len(external))
            actual_infall_times = np.random.choice(infall_times, len(infall_times))
            if boostflag:
                # model 3 involves boosting Ie in addition to truncating the disk,
                # so this requires another loop where Iboost/Ie0 ranges from 1 to 5
                # not sure if I can implement this as a third case
                # or I could just assign a random boost for each iteration
                # and increase nrandom when running model 3

                #
                # going with one boost factor per iteration for now
                # obviously, it's not realistic that ALL galaxies
                # would be boosted by the SAME factor
                # but this is an easy place to start
                
                boost = np.random.random()*(maxboost-1)+1 # boost factor will range between 1 and maxboost
                
            else:
                boost = 1.0
            
            if model == 1:
                # shrink effective radius
                sim_core = external + drdt*actual_infall_times

                # don't allow shrunk radii to go negative

                zero_size_flag = sim_core <= 0
                #if sum(zero_size_flag) > 0:
                #    print('number of size = 0 :',sum(zero_size_flag),len(sim_core))
                sim_core[zero_size_flag] = np.zeros(sum(zero_size_flag))

                
                # to implement SFR decrease
                # need SFR, sersic index, and Re for all galaxies
                # greg has equation for integral of sersic profile
                # create function that gives flux/flux_0 for sersic profile that
                # has Re shrink by some factor
                #frac_retained = get_frac_flux_retained(external_nsersic,external,sim_core)
                frac_retained = get_frac_flux_retained0(external_nsersic,external,sim_core,Iboost=boost)
                
                ########################################################
                # get predicted SFR of core galaxies by multiplying the
                # distribution of SFRs from the external samples by the
                # flux ratio you would expect from shrinking the
                # external sizes to the sim_core sizes
                # SFRs are logged, so add log of frac_retained 
                sim_core_sfr = frac_retained*external_sfr
                
            if model == 2:
                # get the truncation radius (actual physical radius)
                #sim_core_Re24 = external_Re24 + drdt*actual_infall_times
                # our choice of rmax will affect the inferred infall time
                sim_core_Re24 = (rmax + drdt*actual_infall_times)*external_Re24

                # new size ratio
                sim_core = model2_get_fitted_param(sim_core_Re24/external_Re24,sersicRe_fit)*external

                # don't allow shrunk radii to go negative

                zero_size_flag = sim_core < 0
                sim_core[zero_size_flag] = np.zeros(sum(zero_size_flag))

                
                # get the SFR by integrating profile to truncation radius
                if args.sfrint == 1:
                    # integrated truncated profile
                    frac_retained = get_frac_flux_retained_model2(external_nsersic24,external_Re24,rtrunc=sim_core_Re24,rmax=rmax,Iboost=boost)
                elif args.sfrint == 2:
                    # integrate sersic model you would fit to truncated profile
                    frac_retained = get_frac_flux_retained_model2(external_nsersic24,external_Re24,rtrunc=sim_core_Re24,rmax=rmax,version=2,Iboost=boost)
                    
                sim_core_sfr = frac_retained*external_sfr
            ## NEED TO REVIEW THIS
            ## THIS REQUIRES THAT > 20% OF SIMULATED CORE GALAXIES
            ## HAVE A SIZE > 0
            ## do we need this?????
            # not sure what this statement does...

            ## SKIPPING FOR NOW
            #if sum(sim_core > 0.)*1./len(sim_core) < .05:
            #    continue



            ## NEED TO CUT SIMULATED GALAXIES BASED ON SFR AND SIZE
            ## EASIEST IS TO USE MIN VALUE OF SFR AND SIZE IN THE CATALOG
            ## WE COULD PUT THE ACTUAL CUTS IN
            
            #print('\nFraction of Galaxies with sim_core < 0 = %.2f (%i/%i) (drdt = %.2f)'%(sum(sim_core <= 0)/len(sim_core),sum(sim_core <=0),len(sim_core),drdt))
            # compare distribution of sizes between core (measured)
            # and the simulated core

            # keep track of # that drop out due to size
            flag = sim_core_sfr > min(SFR)
            fquench_sfr[aindex] = sum(~flag)/len(flag)
            fquench_size[aindex] = sum(sim_core <= 0)/len(sim_core)
            quench_flag = flag & (sim_core <=0)

            # removing flag for now to make sure things work as expected
            D,p=ks_2samp(core,sim_core[~quench_flag])
            #D,p=ks_2samp(core,sim_core)


            # keep track of number that drop out due to SFR
            
            #if sum(quench_flag) < 0.2*len(quench_flag):
            #    print('Warning: < 20% of sim core sample are unquenched')

            # removing flag to make sure things work as expected
            D2,p2 = ks_2samp(core_sfr,sim_core_sfr[~quench_flag])
            #D2,p2 = ks_2samp(core_sfr,sim_core_sfr)            
            all_p[aindex] = p
            all_drdt[aindex] = drdt
            all_p_sfr[aindex] = p2
            all_boost[aindex] = boost

            if plotflag & ((i+j) == 0):
                #print(core_sfr)
                #print('hey')
                #print(sim_core_sfr)
                #print('hey hey')
                #print(frac_retained)        
                plt.figure(figsize=(10,8))
                # compare size dist
                plt.subplot(1,2,1)
                
                plt.hist(core,label='core',histtype='step',bins=25,lw=2)
                plt.hist(sim_core,label='sim core',histtype='step',bins=25,lw=2)
                plt.hist(external,label='external',histtype='step',bins=25,lw=2)
                plt.xlabel('Size')
                plt.legend()
                plt.title('dr/dt = '+str(drdt))                
                plt.subplot(1,2,2)
                plt.hist(core_sfr,label='core',histtype='step',bins=25,lw=2)
                
                plt.hist(sim_core_sfr,label='sim core',histtype='step',bins=25,lw=2)
                plt.hist(external_sfr,label='external',histtype='step',bins=25,lw=2)
                plt.xlabel('SFR')
                plt.legend()
                plt.title('boost = '+str(boost))

        
        
                

    
    return all_drdt,all_p,all_p_sfr,all_boost,fquench_size,fquench_sfr

###########################
##### PLOT FUNCTIONS
###########################


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

def plot_frac_below_pvalue(all_drdt,all_p,all_p_sfr,tmax,nbins=100,plotsingle=True):
    pvalue = args.pvalue
    if plotsingle:
        plt.figure()
    plt.subplots_adjust(bottom=.15,left=.12)
    mybins = np.linspace(min(all_drdt),max(all_drdt),100)
    t= np.histogram(all_drdt,bins=mybins)
    #print(t)
    ytot = t[0]
    xtot = t[1]
    flag = all_p < 0.05
    t = np.histogram(all_drdt[flag],bins=mybins)
    #print(t)
    y1 = t[0]/ytot
    flag = all_p_sfr < 0.05
    t = np.histogram(all_drdt[flag],bins=mybins)
    y2 = t[0]/ytot
    #plt.figure()

    # calculate the position of the bin centers
    xplt = 0.5*(xtot[0:-1]+xtot[1:])
    plt.plot(xplt,y1,'bo',color=mycolors[0],markersize=6,label='R24/Re')
    plt.plot(xplt,y2,'rs',color=mycolors[1],markersize=6,label='SFR')
    plt.legend()

    plt.xlabel(r'$dr/dt \ (Gyr^{-1}) $',fontsize=18)
    plt.ylabel(r'$Fraction(p<{:.3f})$'.format(pvalue),fontsize=18)
    #s = r'$t_{max} = %.1f \ Gyr, \ dr/dt = %.2f \ Gyr^{-1}, \ t_{quench} = %.1f \ Gyr$'%(tmax, best_drdt,1./abs(best_drdt))
    s = r'$t_{max} = %.1f \ Gyr$'%(tmax)
    #plt.text(0.02,.7,s,transform = plt.gca().transAxes)
    plt.title(s,fontsize=18)
    output = 'frac_pvalue_infall_tmax_%.1f.png'%(tmax)
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
        #plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax,plotsingle=False)
        plot_frac_below_pvalue(all_drdt,all_p,all_p_sfr,tmax,nbins=100,pvalue=0.05,plotsingle=False)        
        allax.append(plt.gca())


    plt.subplots_adjust(hspace=.5,wspace=.7,bottom=.1)
    cb = plt.colorbar(ax=allax,label='$dr/dt$')    
    plt.savefig('sim_infall_multiple_tmax_wsfr.pdf')
    plt.savefig('sim_infall_multiple_tmax_wsfr.png')    
    #plt.savefig('fig18.pdf')
def plot_multiple_tmax_wsfr2(nrandom=100):
    plt.figure(figsize=(10,8))
    mytmax = [1,2,3,4]
    allax = []
    for i,tmax in enumerate(mytmax):
        plt.subplot(2,2,i+1)
        if args.model < 3:
            best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr=run_sim(tmax=tmax,drdt_step=.05,nrandom=nrandom,plotsingle=False,plotflag=False)
        else:
            best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr,boost=run_sim(tmax=tmax,drdt_step=.05,nrandom=nrandom,plotsingle=False,plotflag=False)

        plot_frac_below_pvalue(all_drdt,all_p,all_p_sfr,tmax,nbins=100,plotsingle=False)        
        allax.append(plt.gca())


    plt.subplots_adjust(hspace=.5,wspace=.5,bottom=.1)
    #cb = plt.colorbar(ax=allax,label='$dr/dt$')    
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


def plot_model3(all_drdt,all_p,all_p_sfr,boost,tmax=2):
    '''plot boost factor vs dr/dt, colored by pvalue'''
    plt.figure(figsize=(12,4))
    plt.subplots_adjust(wspace=.5)
    colors = [all_p,all_p_sfr]
    labels = ['size p value','sfr p value']
    titles = ['Size Constraints','SFR Constraints']
    v2 = .005
    allax = []
    for i in range(len(colors)):
        plt.subplot(1,2,i+1)
        plt.scatter(all_drdt,boost,c=colors[i],vmin=0,vmax=v2,s=15)
    
        plt.title(titles[i])
        plt.xlabel('dr/dt',fontsize=16)
        plt.ylabel('I boost/I0',fontsize=16)
        allax.append(plt.gca())
    cb = plt.colorbar()
    cb.set_label('KS p value')
    plt.savefig(plotdir+'/model3-tmax'+str(tmax)+'-size-sfr-constraints.png')
    plt.savefig(plotdir+'/model3-tmax'+str(tmax)+'-size-sfr-constraints.pdf')

def plot_boost_3panel(all_drdt,all_p,all_p_sfr,boost,tmax=2,v2=.005,model=3):
    plt.figure(figsize=(14,4))
    plt.subplots_adjust(wspace=.01,bottom=.15)
    colors = [all_p,all_p_sfr]
    labels = ['size p value','sfr p value']
    titles = ['Size Constraints','SFR Constraints']
    allax = []
    psize=30
    for i in range(len(colors)+1):
        plt.subplot(1,3,i+1)
        if i < 2:
            plt.scatter(all_drdt,boost,c=colors[i],vmin=0,vmax=v2,s=psize)
            plt.title(titles[i])
        else:
            # plot both together
            flag = np.arange(0,len(all_drdt),2)
            plt.scatter(all_drdt[flag],boost[flag],c=colors[0][flag],vmin=0,vmax=v2,s=psize)
            flag = np.arange(1,len(all_drdt),2)
            plt.scatter(all_drdt[flag],boost[flag],c=colors[1][flag],vmin=0,vmax=v2,s=psize)
            #plt.scatter(all_drdt,boost,c=colors[0]+colors[1],vmin=0,vmax=v2,s=10)
            plt.title('Size & SFR Constraints')
        if i == 0:
            plt.ylabel('Iboost/Ie',fontsize=16)
        else:
            y1,y2 = plt.ylim()
            #t = plt.yticks()
            #print(t)
            plt.yticks([])
            plt.ylim(y1,y2)
        plt.xlabel('dr/dt',fontsize=16)
        
        allax.append(plt.gca())
    cb = plt.colorbar(ax=allax,fraction=.08)
    cb.set_label('KS p value')
    plt.savefig(plotdir+'/model3-tmax'+str(tmax)+'-size-sfr-constraints-3panel.png')
    plt.savefig(plotdir+'/model'+str(model)+'-tmax'+str(tmax)+'-size-sfr-constraints-3panel.pdf')

def plot_model1_3panel(all_drdt,all_p,all_p_sfr,tmax=2,v2=.005,model=1,vmin=-4):
    '''
    make a 1x3 plot showing
     (1) pvalue vs dr/dt for size
     (2) pvalue vs dr/dt for SFR
     (3) pvalue size vs pvalue SFR, color coded by dr/dt

    PARAMS
    ------
    * all_drdt : output from run_sum; disk-shrinking rate for each model
    * all_p : output from run_sum; KS pvalue for size comparision
    * all_p_sfr : output from run_sum; KS pvalue for SFR comparison
    * tmax : tmax of simulation, default is 2 Gyr
    * v2 : max value for colorbar; default is 0.005 for 2sigma
    * model : default is 1; could use this plot for models 1 and 2

    OUTPUT
    ------
    * save png and pdf plot in plotdir
    * title is: model3-tmax'+str(tmax)+'-size-sfr-constraints-3panel.pdf
    '''
    
    plt.figure(figsize=(14,4))
    plt.subplots_adjust(wspace=.5,bottom=.15)
    xvars = [all_drdt, all_drdt, all_p]
    yvars = [all_p, all_p_sfr, all_p_sfr]
    xlabels=['dr/dt','dr/dt','pvalue Size']
    ylabels=['pvalue Size','pvalue SFR','pvalue SFR']    
    titles = ['Size Constraints','SFR Constraints','']
    allax = []
    for i in range(len(xvars)):
        plt.subplot(1,3,i+1)
        if i < 2:
            plt.scatter(xvars[i],yvars[i],s=10,alpha=.5)
            plt.title(titles[i])
        else:
            # plot pvalue vs pvalue, color coded by dr/dt
            plt.scatter(all_p,all_p_sfr,c=all_drdt,vmin=vmin,vmax=0,s=5)
            plt.title('Size & SFR Constraints')

        plt.xlabel(xlabels[i],fontsize=16)
        plt.ylabel(ylabels[i],fontsize=16)        
        plt.ylim(-.02,1.02)
        allax.append(plt.gca())
    cb = plt.colorbar(ax=allax,fraction=.08)
    cb.set_label('dr/dt')
    plt.axhline(y=.05,ls='--')
    plt.axvline(x=.05,ls='--')
    ax = plt.gca()
    #plt.axis([-.01,.35,-.01,.2])
    xl = np.linspace(.05,1,100)
    y1 = np.ones(len(xl))
    y2 = .05*np.ones(len(xl))
    plt.fill_between(xl,y1=y1,y2=y2,alpha=.1)
    plt.savefig(plotdir+'/model'+str(model)+'-tmax'+str(tmax)+'-size-sfr-constraints-3panel.png')
    plt.savefig(plotdir+'/model'+str(model)+'-tmax'+str(tmax)+'-size-sfr-constraints-3panel.pdf')

def plot_quenched_fraction(all_drdt,all_boost, fquench_size,fquench_sfr,vmax=.5):
    plt.figure(figsize=(10,4))
    allax=[]
    colors = [fquench_size, fquench_sfr]
    for i in range(len(colors)):
        plt.subplot(1,2,i+1)
        plt.scatter(all_drdt,all_boost,c=colors[i],vmin=0,vmax=vmax)
        allax.append(plt.gca())
        plt.xlabel('$dr/dt$',fontsize=20)
        
        if i == 0:
            plt.ylabel('$I_{boost}/I_e$',fontsize=20)
            plt.title('$Frac \ with\ R_{24} = 0$')
        if i == 1:
            plt.yticks([])
            plt.title('$Frac \ with \ SFR < Limit $')
    plt.colorbar(ax=allax,label='Quenched Fraction')

    
    
if __name__ == '__main__':

    # run program
    print('Welcome!')
    #if args.model == 3:
    #    best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr,all_boost = run_sim(tmax=args.tmax,drdt_step=0.05,nrandom=100,rmax=6)
    #else:
    #    best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr = run_sim(tmax=args.tmax,drdt_step=0.05,nrandom=100,rmax=6)
    # plot
    #plot_frac_below_pvalue(all_drdt,all_p,best_drdt,args.tmax,nbins=100,pvalue=0.05,plotsingle=True)
    pass
