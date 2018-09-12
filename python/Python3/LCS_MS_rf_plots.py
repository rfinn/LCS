#!/usr/bin/env python

###########################
###### IMPORT MODULES
###########################

import LCSbase as lb
from matplotlib import pyplot as plt
import numpy as np
import os
from LCScommon import *
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
import scipy.stats as st

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

###########################
##### Plot defaults
###########################

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
figuredir='/Users/grudnick/Work/Local_cluster_survey/Papers/Finn_MS/Plots/'
#figuredir = '/Users/grudnick/Work/Local_cluster_survey/Analysis/MS_paper/Plots/'

#colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

###########################
##### START OF GALAXIES CLASS
###########################

class galaxies(lb.galaxies):
