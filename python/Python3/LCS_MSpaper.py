#!/usr/bin/env python

'''
GOAL:
- this code contains all of the code to make figures for MS paper


REQUIRED MODULES
- LCSbase_py3.py


**************************
written by Gregory Rudnick
June 2018
**************************

'''
###########################
###### IMPORT MODULES
###########################

import LCSbase as lb
from matplotlib import pyplot as plt
import numpy as np
import os
from LCScommon_py3 import *
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo

import argparse# here is min mass = 9.75

###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Run sextractor, scamp, and swarp to determine WCS solution and make mosaics')
parser.add_argument('--minmass', dest = 'minmass', default = 9., help = 'minimum stellar mass for sample.  default is log10(M*) > 7.9')
parser.add_argument('--diskonly', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')

args = parser.parse_args()

###########################
##### DEFINITIONS
###########################

USE_DISK_ONLY = np.bool(np.float(args.diskonly))#True # set to use disk effective radius to normalize 24um size
if USE_DISK_ONLY:
    print('normalizing by radius of disk')
minsize_kpc=1.3 # one mips pixel at distance of hercules
#minsize_kpc=2*minsize_kpc

mstarmin=float(args.minmass)
mstarmax=10.8
minmass=mstarmin #log of M*
ssfrmin=-12.
ssfrmax=-9
spiralcut=0.8
truncation_ratio=0.5

exterior=.68
colors=['k','b','c','g','m','y','r','sienna','0.5']
shapes=['o','*','p','d','s','^','>','<','v']
#colors=['k','b','c','g','m','y','r','sienna','0.5']

truncated=np.array([113107,140175,79360,79394,79551,79545,82185,166185,166687,162832,146659,99508,170903,18236,43796,43817,43821,70634,104038,104181],'i')

# figure setup
plotsize_single=(6.8,5)
plotsize_2panel=(10,5)
params = {'backend': 'pdf',
          'axes.labelsize': 24,
          'font.size': 20,
          'legend.fontsize': 12,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          #'lines.markeredgecolor'  : 'k',  
          #'figure.titlesize': 20,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'text.usetex': True,
          'figure.figsize': plotsize_single}
plt.rcParams.update(params)
#figuredir = '/Users/rfinn/Dropbox/Research/MyPapers/LCSpaper1/submit/resubmit4/'
figuredir = '/Users/grudnick/Work/Local_cluster_survey/Papers/Finn_MS/Plots/'

###########################
##### START OF GALAXIES CLASS
###########################

class galaxies(lb.galaxies):
    def plotSFRStellarmassall(self):
        figure()
        ax=gca()
        ax.set_yscale('log')
        axis([8.8,12,1.e-3,40.])
        plt.plot(self.logstellarmass[~self.agnflag],self.s.SFR_ZDIST[~self.agnflag],'bo')
        plt.plot(self.logstellarmass[self.sampleflag],self.s.SFR_ZDIST[self.sampleflag],'ro')
        axhline(y=.086,c='k',ls='--')
        axvline(x=9.5,c='k',ls='--')
        plt.xlabel(r'$ M_* \ (M_\odot/yr) $')
        plt.ylabel('$ SFR \ (M_\odot/yr) $')
 
    def plotelbaz(self):
        xe=arange(8.5,11.5,.1)
        xe=10.**xe
        ye=(.08e-9)*xe
        plot(log10(xe),(ye),'k-',lw=1,label='$Elbaz+2011$')
        plot(log10(xe),(2*ye),'k:',lw=1,label='$2 \ SFR_{MS}$')
        plot(log10(xe),(ye/5.),'k-',lw=1,label='$Elbaz+2011$')


        
if __name__ == '__main__':
    homedir = os.environ['HOME']
    g = galaxies(homedir+'/github/LCS/')
