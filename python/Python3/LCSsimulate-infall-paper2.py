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
parser.add_argument('--model', dest = 'model', default = 1, help = 'infall model to use.  default is 1.  \n\tmodel 1 is shrinking 24um effective radius \n\tmodel 2 is truncatingthe 24um emission')
parser.add_argument('--sfrint', dest = 'sfrint', default = 1, help = 'method for integrating the SFR in model 2.  \n\tmethod 1 = integrate external sersic profile out to truncation radius.\n\tmethod 2 = integrate fitted sersic profile out to rmax.')
parser.add_argument('--pvalue', dest = 'pvalue', default = .005, help = 'pvalue threshold to use when plotting fraction of trials below this pvalue.  Default is 0.05 (2sigma).  For ref, 3sigma is .003.')parser.add_argument('--tmax', dest = 'tmax', default = 3., help = 'maximum infall time.  default is 3 Gyr.  ')

parser.add_argument('--rmax', dest = 'rmax', default = 6., help = 'maximum size of SF disk in terms of Re.  default is 6.  ')
#parser.add_argument('--diskonly', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')

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

sersicN_fit = [ 1.00321952, -1.33892353, -1.58502679]
sersicRe_fit = [ 1.0051858,  -1.6640543,  -1.43415337]
sersicIe_fit = [ 1.03890568, 30.02634861, -3.38602545]

## infall rates
# uniform distribution between 0 and tmax Gyr
tmax = 2. # max infall time in Gyr


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

core_Re24 = Re24[core_flag]
external_Re24 = Re24[~core_flag]

core_nsersic24 = nsersic24[core_flag]
external_nsersic24 = nsersic24[~core_flag]



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


def get_frac_flux_retained_model2(n,Re,rtrunc=1,rmax=6,version=1):
    if version == 1:
        # sersic index of profile
        # Re = effective radius of profile
        # n = sersic index of profile
        # rmax = multiple of Re to use a max radius of integration in the "before" integral
        # rtrunc = max radius to integrate to in "after" integral
        
        # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
        
        # calculate the loss in light
        
        # this should simplify to the ratio of the incomplete gamma functions
        # ... I think ...
        
        bn = 1.999*n-0.327
        x_after = bn*(rtrunc/Re)**(1./n)
        x_before = bn*(rmax)**(1./n)
        frac_retained = gammainc(2*n,x_after)/gammainc(2*n,x_before)
        
    elif version == 2:
        # use fitted values of model to get integrated flux after
        # integral of input sersic profile with integral of fitted sersic profile
        Ie = 1
        sfr_before = integrate_sersic(n,Re,Ie,rmax=rmax)
        
        n2 = n*model2_get_fitted_param(rtrunc/Re,sersicN_fit)
        Re2 = Re*model2_get_fitted_param(rtrunc/Re,sersicRe_fit)
        Ie2 = Ie*model2_get_fitted_param(rtrunc/Re,sersicIe_fit)
        sfr_after = integrate_sersic(n2,Re2,Ie2,rmax=rmax)        
        
        frac_retained = sfr_after/sfr_before

        
    return frac_retained


###########################
##### MAIN SIMULATION FUNCTION
###########################


def run_sim(tmax = 2.,nrandom=100,drdt_step=.1,plotsingle=True,plotflag=True):
    rmax = float(args.rmax)
    ks_D_min = 0
    ks_p_max = 0
    drdt_best = 0
    drdt_multiple = []
    drdtmin=-4
    drdtmax=0
    all_p = np.zeros(int(nrandom*(drdtmax-drdtmin)/drdt_step))
    all_p_sfr = np.zeros(int(nrandom*(drdtmax-drdtmin)/drdt_step))    
    all_drdt = np.zeros(int(nrandom*(drdtmax - drdtmin)/drdt_step))
    i=0
    if args.model == 2:
        print('USING MODEL 2')
    for drdt in np.arange(drdtmin,drdtmax,drdt_step):
        #print drdt
        for j in range(nrandom):
            #sim_core = np.random.choice(external,size=len(core)) + drdt*np.random.uniform(low=0, high=tmax, size=len(core))
            infall_times = np.linspace(0,tmax,len(external))
            actual_infall_times = np.random.choice(infall_times, len(infall_times))
            if args.model == 1:
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
            elif args.model == 2:
                # get the truncation radius (actual physical radius)
                sim_core_Re24 = external_Re24 + drdt*actual_infall_times
                sim_core_Re24 = (rmax + drdt*actual_infall_times)*external_Re24

                # new size ratio
                sim_core = model2_get_fitted_param(sim_core_Re24/external_Re24,sersicRe_fit)*external
                # get the SFR by integrating profile to truncation radius
                if args.sfrint == 1:
                    frac_retained = get_frac_flux_retained_model2(external_nsersic24,external_Re24,rtrunc=sim_core_Re24,rmax=rmax)
                elif args.sfrint == 2:
                    frac_retained = get_frac_flux_retained_model2(external_nsersic24,external_Re24,rtrunc=sim_core_Re24,rmax=rmax,version=2)
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
    if plotflag:
        plot_hexbin(all_drdt,all_p,best_drdt,tmax,gridsize = int(1./drdt_step),plotsingle=plotsingle)

    return best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr

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
        best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr=run_sim(tmax=tmax,drdt_step=.05,nrandom=nrandom,plotsingle=False,plotflag=False)

        plot_frac_below_pvalue(all_drdt,all_p,all_p_sfr,tmax,nbins=100,pvalue=0.05,plotsingle=False)        
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

if __name__ == '__main__':

    # run program
    print('Welcome!')
    #best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr = run_sim(tmax=args.tmax,drdt_step=0.05,nrandom=100,rmax=6)
    # plot
    #plot_frac_below_pvalue(all_drdt,all_p,best_drdt,args.tmax,nbins=100,pvalue=0.05,plotsingle=True)
    pass
