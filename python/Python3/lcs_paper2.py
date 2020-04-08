#!/usr/bin/env python

###########################
###### IMPORT MODULES
###########################

import LCSbase as lb
from LCScommon import *

from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.stats as st
from scipy.stats import ks_2samp
import argparse# here is min mass = 9.75

from astropy.io import fits, ascii
from astropy.cosmology import WMAP9 as cosmo
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.table import Table
from astropy.coordinates import SkyCoord

from astropy import units as u
from astropy.stats import median_absolute_deviation as MAD


###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Program to run analysis for LCS paper 2')
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


zmin = 0.0137
zmax = 0.0433

exterior=.68
colors=['k','b','c','g','m','y','r','sienna','0.5']
shapes=['o','*','p','d','s','^','>','<','v']
#colors=['k','b','c','g','m','y','r','sienna','0.5']

truncated=np.array([113107,140175,79360,79394,79551,79545,82185,166185,166687,162832,146659,99508,170903,18236,43796,43817,43821,70634,104038,104181],'i')

###########################
##### Functions
###########################

def plotsalim07():
    #plot the main sequence from Salim+07 for a Chabrier IMF

    lmstar=np.arange(8.5,11.5,0.1)

    #use their equation 11 for pure SF galaxies
    lssfr = -0.35*(lmstar - 10) - 9.83

    #use their equation 12 for color-selected galaxies including
    #AGN/SF composites.  This is for log(Mstar)>9.4
    #lssfr = -0.53*(lmstar - 10) - 9.87

    lsfr = lmstar + lssfr -.3
    sfr = 10.**lsfr

    plt.plot(lmstar, lsfr, 'w-', lw=4)
    plt.plot(lmstar, lsfr, c='salmon',ls='-', lw=2, label='$Salim+07$')
    plt.plot(lmstar, lsfr-np.log10(5.), 'w--', lw=4)
    plt.plot(lmstar, lsfr-np.log10(5.), c='salmon',ls='--', lw=2)
        
def plotelbaz():
    #plot the main sequence from Elbaz+13
        
    xe=np.arange(8.5,11.5,.1)
    xe=10.**xe

    #I think that this comes from the intercept of the
    #Main-sequence curve in Fig. 18 of Elbaz+11.  They make the
    #assumption that galaxies at a fixed redshift have a constant
    #sSFR=SFR/Mstar.  This is the value at z=0.  This is
    #consistent with the value in their Eq. 13

    #This is for a Salpeter IMF
    ye=(.08e-9)*xe   
        
        
    plt.plot(log10(xe),np.log19(ye),'w-',lw=3)
    plt.plot(log10(xe),np.log10(ye),'k-',lw=2,label='$Elbaz+2011$')
    #plot(log10(xe),(2*ye),'w-',lw=4)
    #plot(log10(xe),(2*ye),'k:',lw=2,label='$2 \ SFR_{MS}$')
    plt.plot(log10(xe),np.log10(ye/5.),'w--',lw=4)
    plt.plot(log10(xe),np.log10(ye/5.),'k--',lw=2,label='$SFR_{MS}/5$')

###########################
##### Plot parameters
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

########################################
##### STATISTICS TO COMPARE CORE VS EXTERNAL
########################################

test_statistics = lambda x: (np.mean(x), np.var(x), MAD(x), st.skew(x), st.kurtosis(x))
stat_cols = ['mean','var','MAD','skew','kurt']
###########################
##### START OF GALAXIES CLASS
###########################


class gswlc_full():
    def __init__(self,catalog):
        self.gsw = ascii.read(catalog)
        self.cut_redshift()
        self.save_trimmed_cat()
    def cut_redshift(self):
        z1 = zmin
        z2 = zmax
        zflag = (self.gsw['Z'] > z1) & (self.gsw['Z'] < z2)
        massflag = self.gsw['logMstar'] > 0
        self.gsw = self.gsw[zflag & massflag]
    def save_trimmed_cat(self):
        t = Table(self.gsw)
        t.write('/home/rfinn/research/LCS/tables/GSWLC-X2-LCS-Zoverlap.fits',format='fits',overwrite=True)
        
# functions that will be applied to LCS-GSWLC catalog and GSWLC catalog
class gswlc_base():
    def base_init(self):
        self.calc_ssfr()
        
    def calc_ssfr(self):
        self.ssfr = self.cat['logSFR'] - self.cat['logMstar']
        
    def plot_ms(self,plotsingle=True):
        if plotsingle:
            plt.figure(figsize=(8,6))
        x = self.cat['logMstar']
        y = self.cat['logSFR']
        plt.plot(x,y,'k.',alpha=.1)
        #plt.hexbin(x,y,gridsize=30,vmin=5,cmap='gray_r')
        #plt.colorbar()
        xl=np.linspace(8,12,50)
        ssfr_limit = -10
        #plt.plot(xl,xl+ssfr_limit,'r-')
        plotsalim07()
        plt.xlabel('logMstar',fontsize=16)
        plt.ylabel('logSFR',fontsize=16)
        
    def plot_positions(self, plotsingle = True):
        if plotsingle:
            plt.figure(figsize=(14,8))

        plt.scatter(self.cat['RA'],selfgsw['DEC'],c=np.log10(self.densNN*u.Mpc),vmin=-.5,vmax=1.,s=10)
        plt.colorbar(label='$\log_{10}(N/d_N)$')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.gca().invert_xaxis()
        plt.axhline(y=5)
        plt.axhline(y=60)
        plt.axvline(x=135)
        plt.axvline(x=225)

class gswlc(gswlc_base):
    def __init__(self,catalog):
        self.cat = fits.getdata(catalog) 
        self.base_init()
        self.calc_local_density()
        self.get_field1()
        
    def calc_local_density(self,NN=10):
        pos = SkyCoord(ra=self.cat['RA']*u.deg,dec=self.cat['DEC']*u.deg, distance=self.cat['Z']*3.e5/70*u.Mpc,frame='icrs')

        idx, d2d, d3d = pos.match_to_catalog_3d(pos,nthneighbor=NN)
        self.densNN = NN/d3d
        self.sigmaNN = NN/d2d
    def get_field1(self):
        ramin=135
        ramax=225
        decmin=5
        decmax=60
        gsw_position_flag = (self.cat['RA'] > ramin) & (self.cat['RA'] < ramax) & (self.cat['DEC'] > decmin) & (self.cat['DEC'] < decmax)
        gsw_density_flag = np.log10(self.densNN*u.Mpc) < .2
        self.field1 = gsw_position_flag & gsw_density_flag
    def plot_field1(self):
        plt.figure(figsize=(14,8))
        plt.scatter(self.cat['RA'],self.cat['DEC'],c=np.log10(self.densNN*u.Mpc),vmin=-.5,vmax=1.,s=10)
        plt.colorbar(label='$\log_{10}(N/d_N)$')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.gca().invert_xaxis()
        plt.axhline(y=5)
        plt.axhline(y=60)
        plt.axvline(x=135)
        plt.axvline(x=225)
        plt.title('Entire GSWLC (LCS z range)')
        plt.figure()
        t = plt.hist(np.log10(self.densNN*u.Mpc), bins=20)
        plt.xlabel('$ \log_{10} (N/d_N)$')
        print('median local density =  %.3f'%(np.median(np.log10(self.densNN*u.Mpc))))
        plt.axvline(x = (np.median(np.log10(self.densNN*u.Mpc))),c='k')
        plt.title('Distribution of Local Density')
    
class lcsgsw(gswlc_base):
    def __init__(self,catalog):
        # read in catalog of LCS matched to GSWLC
        self.cat = fits.getdata(catalog)
        self.base_init()
        self.calc_membflag()
    def calc_membflag(self):
        self.membflag = 
    def get_mstar_limit(self):
        print(zmax)
        # assume hubble flow
        dmax = zmax*3.e5/70
        r_limit = 18.
        # abs r
        # m - M = 5logd_pc - 5
        # M = m - 5logd_pc + 5
        ## r = 22.5 - np.log10(lcsgsw['NMGY'][:,4])
        Mr = r_limit - 5*np.log10(dmax*1.e6) +5
        print(Mr)
        ssfr = self.cat['logSFR'] - self.cat['logMstar']
        flag = (self.cat['logMstar'] > 0) & (ssfr > -11.5)
        Mr = self.cat['ABSMAG'][:,4]
        plt.figure()
        plt.plot(Mr[flag],self.cat['logMstar'][flag],'bo',alpha=.2,markersize=3)
        plt.axvline(x=-18.33)
        plt.axhline(y=9.8)
        plt.xlabel('Mr')
        plt.ylabel('logMstar GSWLC')
        plt.grid(True)        

class comp_lcs_gsw():
    def __init__(self,lcs,gsw,minmstar = 9.8, minssfr = -11.5):
        self.lcs = lcs
        self.gsw = gsw
        self.minmstar = minmstar
        self.minssfr = minssfr
    
    def plot_ssfr_mstar(self):

        pass

    def plot_sfr_mstar(self):
        flag1 = lcsgsw['membflag'] &  (lcsgsw['logMstar']> masscut)  & (ssfr > ssfrcut)
        flag2 = (gsw['logMstar'] > masscut) & (gswssfr > ssfrcut)  & field_gsw
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs['logMstar'][flag1]
        y1 = self.lcs['logSFR'][flag1]
        x2 = self.gsw['logMstar'][flag1]
        y2 = self.gsw['logSFR'][flag1]
        
        colormass(self.lcs['logMstar'][flag1],self.lcs[[flag1],gsw['logMstar'][flag2],gsw['logSFR'][flag2],'LCS core','GSWLC','sfr-mstar-gswlc-field.pdf',ymin=-2,ymax=1.6,xmin=9.75,xmax=11.5,nhistbin=10,ylabel='$\log_{10}(SFR)$',contourflag=False,alphagray=.8)
        plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.pdf')
plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.png')        
        pass
if __name__ == '__main__':
    trimgswlc = False
    if trimgswlc:
        g = gswlc_full('/home/rfinn/research/GSWLC/GSWLC-X2.dat')
        g.cut_redshift()
        g.save_trimmed_cat()

    g = gswlc('/home/rfinn/research/LCS/tables/GSWLC-X2-LCS-Zoverlap.fits')    
    #g.plot_ms()
    #g.plot_field1()
    lcs = lcsgsw('/home/rfinn/research/LCS/tables/lcs-gswlc-x2-match.fits')

    b = comp_lcs_gsw(lcs,g)
