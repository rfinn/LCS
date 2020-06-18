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



from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
#from anderson import *

#R24/Rd err  core_flag   B/T 
infile = '/Users/rfinn/research/LocalClusters/catalogs/sizes.txt'

sizes = np.loadtxt(infile)
size_ratio = np.array(sizes[:,0],'f')
size_err = np.array(sizes[:,1],'f')
core_flag = np.array(sizes[:,2],'bool')
bt = np.array(sizes[:,3],'f')

## split sizes
core = size_ratio[core_flag]
external = size_ratio[~core_flag]

## infall rates
# uniform distribution between 0 and tmax Gyr
tmax = 2. # max infall time in Gyr



def run_sim(tmax = 2.,nrandom=100,drdt_step=.1,plotsingle=True):
    ks_D_min = 0
    ks_p_max = 0
    drdt_best = 0
    drdt_multiple = []
    drdtmin=-2
    drdtmax=0
    all_p = np.zeros(int(nrandom*(drdtmax-drdtmin)/drdt_step))
    all_drdt = np.zeros(int(nrandom*(drdtmax - drdtmin)/drdt_step))
    i=0
    for drdt in np.arange(drdtmin,drdtmax,drdt_step):
        #print drdt
        for j in range(nrandom):
            #sim_core = np.random.choice(external,size=len(core)) + drdt*np.random.uniform(low=0, high=tmax, size=len(core))
            infall_times = np.linspace(0,tmax,len(external))
            sim_core = external + drdt*np.random.choice(infall_times, len(infall_times)) 
            if sum(sim_core > 0.)*1./len(sim_core) < .2:
                continue
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
                print 'found an equally good model at drdt = ',drdt,p
                drdt_multiple.append(drdt)
            all_p[i] = p
            all_drdt[i] = drdt
            i += 1
    print 'best dr/dt = ',best_drdt
    print 'disk is quenched in %.1f Gyr'%(1./abs(best_drdt))
    print 'fraction that are quenched = %.2f'%(1.*sum(best_sim_core < 0.)/len(best_sim_core))
    print 'KS p value = %.8e'%(ks_p_max)
    if len(drdt_multiple) > 0.:
        print 'drdt multiple values of dr/dt'
        for i in range(len(drdt_multiple)):
            print '#################'
            print '\t best dr/dt = ',drdt_multiple[i]
            print '\t disk is quenched in %.1f Gyr'%(1./abs(drdt_multiple[i]))
    #plot_results(core,external,best_sim_core,best_drdt,tmax)
    plot_hexbin(all_drdt,all_p,best_drdt,tmax,gridsize = int(1./drdt_step),plotsingle=plotsingle)
    return best_drdt, best_sim_core,ks_p_max,all_drdt,all_p


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

def plot_multiple_tmax(nrandom=100):
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    run_sim(tmax=1,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplot(2,2,2)
    run_sim(tmax=2,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplot(2,2,3)
    run_sim(tmax=3,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplot(2,2,4)
    run_sim(tmax=4,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplots_adjust(hspace=.5,bottom=.1)
    plt.savefig('sim_infall_multiple_tmax.pdf')
    plt.savefig('fig18.pdf')
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

