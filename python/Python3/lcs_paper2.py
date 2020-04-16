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
parser.add_argument('--minmass', dest = 'minmass', default = 10., help = 'minimum stellar mass for sample.  default is log10(M*) > 7.9')
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
colorblind1='#F5793A' # orange
colorblind2 = '#85C0F9' # light blue
colorblind3='#0F2080' # dark blue
def colormass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
              xlabel='$\log_{10}(M_\star/M_\odot) $', ylabel='$(g-i)_{corrected} $', color1=colorblind3,color2=colorblind2,\
             nhistbin=50, alphagray=.1):
    fig = plt.figure(figsize=(8,8))
    nrow = 4
    ncol = 4
    
    # for purposes of this plot, only keep data within the 
    # window specified by [xmin:xmax, ymin:ymax]
    
    keepflag1 = (x1 >= xmin) & (x1 <= xmax) & (y1 >= ymin) & (y1 <= ymax)
    keepflag2 = (x2 >= xmin) & (x2 <= xmax) & (y2 >= ymin) & (y2 <= ymax)
    
    x1 = x1[keepflag1]
    y1 = y1[keepflag1]
    
    x2 = x2[keepflag2]
    y2 = y2[keepflag2]
    
    ax1 = plt.subplot2grid((nrow,ncol),(1,0),rowspan=nrow-1,colspan=ncol-1, fig=fig)
    if hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)

        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label=name1)
    else:
        plt.plot(x1,y1,'k.',color=color1,alpha=alphagray,label=name1, zorder=2)
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=1,colors=color2, label='__nolegend__')
        #plt.legend()
    else:
        plt.plot(x2,y2,'c.',color=color2,alpha=.3, label=name2)
        
        
        #plt.legend()
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()
    
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=22)
    plt.ylabel(ylabel,fontsize=22)
    #plt.axis([7.9,11.6,-.05,2])
    ax2 = plt.subplot2grid((nrow,ncol),(0,0),rowspan=1,colspan=ncol-1, fig=fig, sharex = ax1, yticks=[])
    t = plt.hist(x1, normed=True, bins=nhistbin,color=color1,histtype='step',lw=1.5, label=name1)
    t = plt.hist(x2, normed=True, bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
    #plt.legend()
    ax2.legend(fontsize=10,loc='upper right')
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    t=plt.hist(y1, normed=True, orientation='horizontal',bins=nhistbin,color=color1,histtype='step',lw=1.5, label=name1)
    t=plt.hist(y2, normed=True, orientation='horizontal',bins=nhistbin,color=color2,histtype='step',lw=1.5, label=name2)
    
    plt.yticks(rotation='horizontal')
    ax3.yaxis.tick_right()
    plt.savefig(figname)

    print('############################################################# ')
    print('KS test comparising galaxies within range shown on the plot')
    print('')
    print('STELLAR MASS')
    t = ks(x1,x2,run_anderson=False)
    print('')
    print('COLOR')
    t = ks(y1,y2,run_anderson=False)


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
        if catalog.find('.fits') > -1:
            self.gsw = Table(fits.getdata(catalog))
            self.redshift_field = 'zobs'
            self.outfile = catalog.split('.fits')[0]+'-LCS-Zoverlap.fits'                        
        else:
            self.gsw = ascii.read(catalog)
            self.redshift_field = 'Z'
            self.outfile = catalog.split('.dat')[0]+'-LCS-Zoverlap.fits'            
        self.cut_redshift()
        self.save_trimmed_cat()

    def cut_redshift(self):
        z1 = zmin
        z2 = zmax
        zflag = (self.gsw[self.redshift_field] > z1) & (self.gsw[self.redshift_field] < z2)
        massflag = self.gsw['logMstar'] > 0
        self.gsw = self.gsw[zflag & massflag]
    def save_trimmed_cat(self):
        t = Table(self.gsw)

        t.write(self.outfile,format='fits',overwrite=True)
        
# functions that will be applied to LCS-GSWLC catalog and GSWLC catalog
class gswlc_base():
    def base_init(self):
        self.calc_ssfr()
        
    def calc_ssfr(self):
        self.ssfr = self.cat['logSFR'] - self.cat['logMstar']
        
    def plot_ms(self,plotsingle=True,outfile1=None,outfile2=None):
        if plotsingle:
            plt.figure(figsize=(8,6))
        x = self.cat['logMstar']
        y = self.cat['logSFR']
        plt.plot(x,y,'k.',alpha=.1)
        #plt.hexbin(x,y,gridsize=30,vmin=5,cmap='gray_r')
        #plt.colorbar()
        xl=np.linspace(8,12,50)
        ssfr_limit = -11.5
        plt.plot(xl,xl+ssfr_limit,'c-')
        plotsalim07()
        plt.xlabel('logMstar',fontsize=20)
        plt.ylabel('logSFR',fontsize=20)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/sfms-ssfr-cut.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/sfms-ssfr-cut.png')
        else:
            plt.savefig(outfile2)
        
    def plot_positions(self,plotsingle=True, filename=None):
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
        #plt.axis([130,230,0,65])
        if filename is not None:
            plt.savefig(filename)
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
    def plot_field1(self,figname1=None,figname2=None):
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
        plt.axis([130, 230, 0, 65])
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)
    def plot_dens_hist(self,figname1=None,figname2=None):
        plt.figure()
        t = plt.hist(np.log10(self.densNN*u.Mpc), bins=20)
        plt.xlabel('$ \log_{10} (N/d_N)$')

        print('median local density =  %.3f'%(np.median(np.log10(self.densNN*u.Mpc))))
        plt.axvline(x = (np.median(np.log10(self.densNN*u.Mpc))),c='k')
        plt.title('Distribution of Local Density')
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)

class lcsgsw(gswlc_base):
    def __init__(self,catalog,sigma_split=600):
        # read in catalog of LCS matched to GSWLC
        self.cat = fits.getdata(catalog)
        self.base_init()
        self.group = self.cat['CLUSTER_SIGMA'] < sigma_split
        self.cluster = self.cat['CLUSTER_SIGMA'] > sigma_split
        #lcspath = homedir+'/github/LCS/'
        #self.lcsbase = lb.galaxies(lcspath)
    def get_mstar_limit(self,rlimit=17.7):
        
        print(rlimit,zmax)
        # assume hubble flow
        
        dmax = zmax*3.e5/70
        # abs r
        # m - M = 5logd_pc - 5
        # M = m - 5logd_pc + 5
        ## r = 22.5 - np.log10(lcsgsw['NMGY'][:,4])
        Mr = rlimit - 5*np.log10(dmax*1.e6) +5
        print(Mr)
        ssfr = self.cat['logSFR'] - self.cat['logMstar']
        flag = (self.cat['logMstar'] > 0) & (ssfr > -11.5)
        Mr = self.cat['ABSMAG'][:,4]
        plt.figure()
        plt.plot(Mr[flag],self.cat['logMstar'][flag],'bo',alpha=.2,markersize=3)
        plt.axvline(x=-18.6)
        plt.axhline(y=10)
        plt.xlabel('Mr')
        plt.ylabel('logMstar GSWLC')
        plt.grid(True)        
    def plot_dvdr(self, figname1=None,figname2=None,plotsingle=True):
        # log10(chabrier) = log10(Salpeter) - .25 (SFR estimate)
        # log10(chabrier) = log10(diet Salpeter) - 0.1 (Stellar mass estimates)
        xmin,xmax,ymin,ymax = 0,3.5,0,3.5
        if plotsingle:
            plt.figure(figsize=(8,6))
            ax=plt.gca()
            plt.subplots_adjust(left=.1,bottom=.15,top=.9,right=.9)
            plt.ylabel('$ \Delta v/\sigma $',fontsize=26)
            plt.xlabel('$ \Delta R/R_{200}  $',fontsize=26)
            plt.legend(loc='upper left',numpoints=1)

        if USE_DISK_ONLY:
            clabel=['$R_{24}/R_d$','$R_{iso}(24)/R_{iso}(r)$']
        else:
            clabel=['$R_e(24)/R_e(r)$','$R_{iso}(24)/R_{iso}(r)$']
        
        x=(self.cat['DR_R200'])
        y=abs(self.cat['DELTA_V'])
        plt.hexbin(x,y,extent=(xmin,xmax,ymin,ymax),cmap='gray_r',gridsize=50,vmin=0,vmax=10)
        xl=np.arange(0,2,.1)
        plt.plot(xl,-4./3.*xl+2,'k-',lw=3,color=colorblind1)
        props = dict(boxstyle='square', facecolor='0.8', alpha=0.8)
        plt.text(.1,.1,'CORE',transform=plt.gca().transAxes,fontsize=18,color=colorblind3,bbox=props)
        plt.text(.6,.6,'EXTERNAL',transform=plt.gca().transAxes,fontsize=18,color=colorblind3,bbox=props)        
        #plt.plot(xl,-3./1.2*xl+3,'k-',lw=3)
        plt.axis([xmin,xmax,ymin,ymax])
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)            
    def compare_sfrs(self,shift=None,masscut=None,nbins=20):
        '''
        plot distribution of LCS external and core SFRs

        '''
        if masscut is None:
            masscut = minmass
        flag = (self.cat['logSFR'] > -99) & (self.cat['logSFR']-self.cat['logMstar'] > -11.5) & (self.cat['logMstar'] > masscut)
        sfrcore = self.cat['logSFR'][self.cat['membflag'] & flag] 
        sfrext = self.cat['logSFR'][~self.cat['membflag']& flag]
        plt.figure()
        mybins = np.linspace(-2.5,1.5,nbins)

        plt.hist(sfrext,bins=mybins,histtype='step',label='External',lw=3)
        if shift is not None:
            plt.hist(sfrext+np.log10(1-shift),bins=mybins,histtype='step',label='Shifted External',lw=3)
        plt.hist(sfrcore,bins=mybins,histtype='step',label='Core',lw=3)            
        plt.legend()
        plt.xlabel('SFR')
        plt.ylabel('Normalized Counts')
        print('CORE VS EXTERNAL')
        t = ks(sfrcore,sfrext,run_anderson=False)

class comp_lcs_gsw():
    def __init__(self,lcs,gsw,minmstar = 10, minssfr = -11.5):
        self.lcs = lcs
        self.gsw = gsw
        self.masscut = minmstar
        self.ssfrcut = minssfr
    
    def plot_ssfr_mstar(self):

        pass

    def plot_sfr_mstar(self,lcsflag=None,outfile1=None,outfile2=None):
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        
        flag1 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        # removing field1 cut because we are now using Tempel catalog that only
        # includes galaxies in halo masses logM < 12.5
        flag2 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.cat['logSFR'][flag1]
        x2 = self.gsw.cat['logMstar'][flag2]
        y2 = self.gsw.cat['logSFR'][flag2]
        
        colormass(x1,y1,x2,y2,'LCS core','GSWLC','sfr-mstar-gswlc-field.pdf',ymin=-2,ymax=1.6,xmin=9.75,xmax=11.5,nhistbin=10,ylabel='$\log_{10}(SFR)$',contourflag=False,alphagray=.8)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.png')
        else:
            plt.savefig(outfile2)
    def plot_ssfr_mstar(self,lcsflag=None,outfile1=None,outfile2=None):
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        
        flag1 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        flag2 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.ssfr[flag1]
        x2 = self.gsw.cat['logMstar'][flag2]
        y2 = self.gsw.ssfr[flag2]
        
        colormass(x1,y1,x2,y2,'LCS core','GSWLC','sfr-mstar-gswlc-field.pdf',ymin=-11.6,ymax=-8.75,xmin=9.75,xmax=11.5,nhistbin=20,ylabel='$\log_{10}(sSFR)$',contourflag=False,alphagray=.8)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.png')
        else:
            plt.savefig(outfile2)


    
if __name__ == '__main__':
    trimgswlc = True
    if trimgswlc:
        #g = gswlc_full('/home/rfinn/research/GSWLC/GSWLC-X2.dat')
        g = gswlc_full('/home/rfinn/research/LCS/tables/GSWLC-Tempel-12.5.fits')
        g.cut_redshift()
        g.save_trimmed_cat()

    #g = gswlc('/home/rfinn/research/LCS/tables/GSWLC-X2-LCS-Zoverlap.fits')
    g = gswlc('/home/rfinn/research/LCS/tables/GSWLC-Tempel-12.5-LCS-Zoverlap.fits')        
    #g.plot_ms()
    #g.plot_field1()
    lcs = lcsgsw('/home/rfinn/research/LCS/tables/lcs-gswlc-x2-match.fits')
    #lcs.compare_sfrs()
    b = comp_lcs_gsw(lcs,g)
