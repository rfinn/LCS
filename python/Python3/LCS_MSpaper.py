#!/usr/bin/env python

'''
GOAL:
- this code contains all of the code to make figures for MS paper


REQUIRED MODULES
- LCSbase.py


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
from LCScommon import *
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from scipy.optimize import curve_fit

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
    def plotSFRStellarmassall(self):
        #plot SFR-Mstar showing all galaxies that make our final cut and those that don't

        figure()
        ax=gca()
        ax.set_yscale('log')
        axis([8.8,12,5.e-4,40.])
        plt.plot(self.logstellarmass,self.SFR_BEST,'ko',label='rejected')
        plt.plot(self.logstellarmass[self.sampleflag],self.SFR_BEST[self.sampleflag],'ro',label='final sample')
        #axhline(y=.086,c='k',ls='--')
        #axvline(x=9.7,c='k',ls='--')
        plt.xlabel(r'$ M_* \ (M_\odot/yr) $')
        plt.ylabel('$ SFR \ (M_\odot/yr) $')
        g.plotelbaz()
        g.plotlims()

        #determine MS fit and output fit results
        xmod,ymod = g.MSfit()
        #plot our own MS
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        plt.title("All Galaxies")
        plt.legend(loc='lower right',numpoints=1,scatterpoints=1, markerscale=0.7,fontsize='x-small')

    def plotSFRStellarmassallenv(self, savefig=False, btcutflag=True):
        #plot SFR-Mstar showing all galaxies that make our final cut
        #and those that don't.  Split by environment.
        
        figure(figsize=(10,5))
        subplots_adjust(left=.12,bottom=.19,wspace=.02,hspace=.02)

        limits=[8.1,12,1.e-3,40.]

        #what is the B/T cut
        btcut = 0.3

        #decide whether to apply the B/T<btcut 
        if btcutflag:
            sampflag = self.sampleflag & self.membflag & (self.gim2d.B_T_r < btcut)
            nosampflag = ~self.sampleflag & self.membflag & (self.gim2d.B_T_r < btcut)
        else:
            sampflag = self.sampleflag & self.membflag
            nosampflag = ~self.sampleflag & self.membflag

        
        #plot selection for core galaxies
        plt.subplot(1,2,1)
        ax=gca()
        ax.set_yscale('log')
        bothax=[]
        plt.axis(limits)


        #determine MS fit and output fit results
        xmod,ymod = g.MSfit()

        plt.plot(self.logstellarmass[nosampflag],self.SFR_BEST[nosampflag],'ko',label='rejected')
        plt.plot(self.logstellarmass[sampflag],self.SFR_BEST[sampflag],'ro',label='final sample')
        #plt.xlabel(r'$ M_* \ (M_\odot/yr) $')
        plt.ylabel('$ SFR \ (M_\odot/yr) $')
        g.plotsalim07()
        g.plotelbaz()
        g.plotlims()

        #plot our own MS
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        plt.title('$Core$', fontsize=22)
        #plt.legend(loc='lower right',numpoints=1,scatterpoints=1, markerscale=0.7,fontsize='x-small')
        text(1.,-.2,'$log_{10}(M_*/M_\odot)$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)

        #plot our own MS
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        #plot selection for external galaxies
        plt.subplot(1,2,2)
        ax=gca()
        ax.set_yscale('log')
        bothax=[]
        plt.axis(limits)
        
        plt.plot(self.logstellarmass[nosampflag],self.SFR_BEST[nosampflag],'ko',label='rejected')
        plt.plot(self.logstellarmass[sampflag],self.SFR_BEST[sampflag],'ro',label='final sample')
        g.plotsalim07()
        g.plotelbaz()
        g.plotlims()
        ax.set_yticklabels(([]))
        plt.title('$External$', fontsize=22)

        #plot our own MS
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            text(-0.15,1.05,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        if savefig:
            plt.savefig(figuredir + 'sfr_mstar_allsel_env.pdf')


    def plotSFRStellarmasssel(self,subsample='all'):
        #detrmine how the different selection flags remove galaxies in
        #the SFR-Mstar space

        #the default is that it will plot all the galaxies
        #(subsample='all') .  By selecting subsample='core' or
        #'exterior' you can choose the other subsamples

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[8.1,12,1.e-3,40.]

        #determine MS fit and output fit results
        xmod,ymod = g.MSfit()

        #plot each subplot separately
        if subsample == 'all':
            subsampflag = (self.logstellarmass > 0)
        if subsample == 'core':
            subsampflag = (self.membflag)
        if subsample == 'exterior':
            subsampflag = (~self.membflag)

        flag = (subsampflag & ~self.agnflag)
        plt.subplot(2,3,1)
        ax=plt.gca()
        g.sfrmasspanel(subsampflag,flag,limits,bothax,ax,xmod,ymod)
        ax.set_xticklabels(([]))
        text(0.1,0.9,'AGN',transform=ax.transAxes,horizontalalignment='left',fontsize=12)
        text(-0.25,-0,'$SFR \ (M_\odot/yr)$',transform=ax.transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=24)

        flag = (subsampflag & self.galfitflag)
        plt.subplot(2,3,2)
        ax=plt.gca()
        g.sfrmasspanel(subsampflag,flag,limits,bothax,ax,xmod,ymod)
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        text(0.1,0.9,'galfitflag',transform=ax.transAxes,horizontalalignment='left',fontsize=12)
        if subsample == 'all':
            plt.title('All galaxies')
        if subsample == 'core':
            plt.title('Core galaxies')
        if subsample == 'exterior':
            plt.title('Exterior galaxies')

        flag = (subsampflag & self.lirflag)
        plt.subplot(2,3,3)
        ax=plt.gca()
        g.sfrmasspanel(subsampflag,flag,limits,bothax,ax,xmod,ymod)
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        text(0.1,0.9,'lirflag',transform=ax.transAxes,horizontalalignment='left',fontsize=12)

        flag = (subsampflag & self.sizeflag)
        plt.subplot(2,3,4)
        ax=plt.gca()
        g.sfrmasspanel(subsampflag,flag,limits,bothax,ax,xmod,ymod)
        text(0.1,0.9,'size',transform=ax.transAxes,horizontalalignment='left',fontsize=12)

        flag = (subsampflag & self.sbflag)
        plt.subplot(2,3,5)
        ax=plt.gca()
        g.sfrmasspanel(subsampflag,flag,limits,bothax,ax,xmod,ymod)
        ax.set_yticklabels(([]))
        text(0.1,0.9,'SB',transform=ax.transAxes,horizontalalignment='left',fontsize=12)
        text(0.5,-.2,'$log_{10}(M_*/M_\odot)$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)

        flag = (subsampflag & self.gim2dflag)
        plt.subplot(2,3,6)
        ax=plt.gca()
        g.sfrmasspanel(subsampflag,flag,limits,bothax,ax,xmod,ymod)
        ax.set_yticklabels(([]))
        text(0.1,0.9,'GIM2D',transform=ax.transAxes,horizontalalignment='left',fontsize=12)

    def plotsalim07(self):
        #plot the main sequence from Salim+07 for a Chabrier IMF

        lmstar=arange(8.5,11.5,0.1)

        #use their equation 11 for pure SF galaxies
        lssfr = -0.35*(lmstar - 10) - 9.83

        #use their equation 12 for color-selected galaxies including
        #AGN/SF composites.  This is for log(Mstar)>9.4
        #lssfr = -0.53*(lmstar - 10) - 9.87

        lsfr = lmstar + lssfr
        sfr = 10**lsfr

        plot(lmstar, sfr, 'w-', lw=4)
        plot(lmstar, sfr, c='salmon',ls='-', lw=2, label='$Salim+07$')
        plot(lmstar, sfr/5., 'w--', lw=4)
        plot(lmstar, sfr/5., c='salmon',ls='--', lw=2)
        
    def plotelbaz(self):
        #plot the main sequence from Elbaz+13
        
        xe=arange(8.5,11.5,.1)
        xe=10.**xe

        #I think that this comes from the intercept of the
        #Main-sequence curve in Fig. 18 of Elbaz+11.  They make the
        #assumption that galaxies at a fixed redshift have a constant
        #sSFR=SFR/Mstar.  This is the value at z=0.  This is
        #consistent with the value in their Eq. 13

        #This is for a Salpeter IMF
        ye=(.08e-9)*xe   
        
        
        plot(log10(xe),(ye),'w-',lw=3)
        plot(log10(xe),(ye),'k-',lw=2,label='$Elbaz+2011$')
        #plot(log10(xe),(2*ye),'w-',lw=4)
        #plot(log10(xe),(2*ye),'k:',lw=2,label='$2 \ SFR_{MS}$')
        plot(log10(xe),(ye/5.),'w--',lw=4)
        plot(log10(xe),(ye/5.),'k--',lw=2,label='$SFR_{MS}/5$')

    def plotlims(self):
        #0.086 is the MS SFR that corresponds to our LIR limit.
        #The factor of 1.74 converts this to Chabrier
        axhline(y=.086/1.74,c='w',lw=4,ls='--')
        axhline(y=.086/1.74,c='goldenrod',lw=3,ls='--')

        axvline(x=9.7,c='w',lw=4,ls='--')
        axvline(x=9.7,c='goldenrod',lw=3,ls='--')

    def sfrmasspanel(self,subsampflag,flag,limits,bothax,ax,xmod,ymod):
        #make SFR-Mstar plots of individual panels if given a subset of sources
        plt.plot(self.logstellarmass[subsampflag],self.SFR_BEST[subsampflag],'ko',markersize=5)
        #sample with selection
        plt.plot(self.logstellarmass[flag],self.SFR_BEST[flag],'ro',markersize=4)
        plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        g.plotelbaz()

        #plot our own MS
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        g.plotlims()

    def plotSFRStellarmass_sizebin(self, savefig=False, btcutflag=True):
        #Make Mstar-SFR plots for exterior and core samples including
        #a binned plot

        minsize=.4
        maxsize=1.5
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[8.8,11.2,3e-2,15.]

        #core galaxies - individual points
        plt.subplot(2,2,1)
        ax=plt.gca()

        #determine MS fit and output fit results
        xmod,ymod = g.MSfit()

        #select members with B/T<btcut
        if btcutflag:
            flag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (self.membflag & self.sampleflag)
        plt.scatter(self.logstellarmass[flag],self.SFR_BEST[flag],c=self.sizeratio[flag],vmin=minsize,vmax=maxsize,cmap='jet_r',s=60)

        
        plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        ax.set_xticklabels(([]))
        #g.plotelbaz()
        text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        plt.title('$SF \ Galaxies$',fontsize=22)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        
        #core plot binned points
        plt.subplot(2,2,2)
        ax=plt.gca()
        xmin = 9.7
        xmax = 10.9
        nbin = (xmax - xmin) / 0.2
        #median SFR  in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_BEST[flag])

        #median size ratio  in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sizeratio[flag])
        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='jet_r',vmin=minsize,vmax=maxsize,marker='s')
        plt.scatter(xbin,ybin,c=sbin,s=300,cmap='jet_r',vmin=minsize,vmax=maxsize,marker='s')
        gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        plt.title('$Median $',fontsize=22)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        #g.plotelbaz()
        legend(loc='upper left',numpoints=1)

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            text(-0.15,1.1,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)


        ###############
        plt.subplot(2,2,3)
        ax=plt.gca()

        #select non-members with B/T<btcut
        flag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        plt.scatter(self.logstellarmass[flag],self.SFR_BEST[flag],c=self.sizeratio[flag],vmin=minsize,vmax=maxsize,cmap='jet_r',s=60)
        plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)
        #g.plotelbaz()

        text(-0.2,1.,'$SFR \ (M_\odot/yr)$',transform=ax.transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=24)
        text(0.1,0.9,'$External$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        #external plot binned points
        plt.subplot(2,2,4)
        ax=plt.gca()
        xmin = 9.4
        xmax = 11.0
        nbin = (xmax - xmin) / 0.2

        #median SFR  in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_BEST[flag])

        #median size ratio  in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sizeratio[flag])
        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='jet_r',vmin=minsize,vmax=maxsize,marker='s')
        plt.scatter(xbin,ybin,c=sbin,s=300,cmap='jet_r',vmin=minsize,vmax=maxsize,marker='s')
        gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)
        #g.plotelbaz()


        text(-0.02,-.2,'$log_{10}(M_*/M_\odot)$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)

        c=colorbar(ax=bothax,fraction=.05,ticks=arange(minsize,maxsize,.1),format='%.1f')
        c.ax.text(2.2,.5,'$R_e(24)/R_e(r)$',rotation=-90,verticalalignment='center',fontsize=20)

        
        if savefig:
            plt.savefig(figuredir + 'sfr_mstar_sizecolor.pdf')
        
    def plotSFRStellarmass_musfrbin(self, savefig=False, btcutflag=True):
        #Make Mstar-SFR plots for exterior and core samples including
        #a binned plot

        minlsfrdense=-2.5
        maxlsfrdense=-0.5
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[8.8,11.2,3e-2,15.]

        self.sfrdense = 0.5 * self.SFR_BEST / (np.pi * self.mipssize**2)
        
        #core galaxies - individual points
        plt.subplot(2,2,1)
        ax=plt.gca()

        #determine MS fit and output fit results
        xmod,ymod = g.MSfit()

        #select members with B/T<btcut
        if btcutflag:
            flag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (self.membflag & self.sampleflag)

        #plot the individual points.
        plt.scatter(self.logstellarmass[flag],self.SFR_BEST[flag],c=log10(self.sfrdense[flag]),vmin=minlsfrdense,vmax=maxlsfrdense,cmap='jet_r',s=60)

        
        plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        ax.set_xticklabels(([]))
        #g.plotelbaz()
        text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        plt.title('$SF \ Galaxies$',fontsize=22)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        
        #core plot binned points
        plt.subplot(2,2,2)
        ax=plt.gca()
        xmin = 9.7
        xmax = 10.9
        nbin = (xmax - xmin) / 0.2
        #median SFR  in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_BEST[flag])

        #median SFR density in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sfrdense[flag])
        #median log(SFR density) in bins of mass
        xbin,lsbin,lsbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],log10(self.sfrdense[flag]))

        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

        #color code by log of the median SFR density
        plt.scatter(xbin,ybin,c=log10(sbin),s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

        #color code by the median of log(SFR density)
        #plt.scatter(xbin,ybin,c=lsbin,s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')
        gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        plt.title('$Median $',fontsize=22)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        #g.plotelbaz()
        legend(loc='upper left',numpoints=1)

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            text(-0.15,1.1,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)


        ###############
        plt.subplot(2,2,3)
        ax=plt.gca()

        #select non-members with B/T<btcut
        flag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        plt.scatter(self.logstellarmass[flag],self.SFR_BEST[flag],c=log10(self.sfrdense[flag]),vmin=minlsfrdense,vmax=maxlsfrdense,cmap='jet_r',s=60)
        plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)
        #g.plotelbaz()

        text(-0.2,1.,'$SFR \ (M_\odot/yr)$',transform=ax.transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=24)
        text(0.1,0.9,'$External$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        #external plot binned points
        plt.subplot(2,2,4)
        ax=plt.gca()
        xmin = 9.4
        xmax = 11.0
        nbin = (xmax - xmin) / 0.2
        #median SFRs in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_BEST[flag])

        #median SFR density in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sfrdense[flag])

        #median log(SFR density) in bins of mass
        xbin,lsbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],log10(self.sfrdense[flag]))
        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

        #color code by log of the median SFR density
        plt.scatter(xbin,ybin,c=log10(sbin),s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')
        
        #color code by the median of log(SFR density)
        #plt.scatter(xbin,ybin,c=lsbin,s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

        gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)
        #g.plotelbaz()


        text(-0.02,-.2,'$log_{10}(M_*/M_\odot)$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)

        c=colorbar(ax=bothax,fraction=.05,ticks=arange(minlsfrdense,maxlsfrdense,.1),format='%.1f')
        c.ax.text(2.2,.5,'$log_{10}(\mu_{SFR}/(M_\odot~yr^{-1}~kpc^{-2}))$',rotation=-90,verticalalignment='center',fontsize=20)

        
        if savefig:
            plt.savefig(figuredir + 'sfr_mstar_musfrcolor.pdf')
        
    def binitbins(self,xmin,xmax,nbin,x,y):#use equally spaced bins
        #compute median values of a quantity for data binned by another quantity
        
        dx=float((xmax-xmin)/(nbin))
        xbin=np.arange(xmin,(xmax),dx)+dx/2.
        ybin=np.zeros(len(xbin),'d')
        ybinerr=np.zeros(len(xbin),'d')
        xbinnumb=np.array(len(x),'d')
        x1=np.compress((x >= xmin) & (x <= xmax),x)
        y1=np.compress((x >= xmin) & (x <= xmax),y) 
        x=x1
        y=y1
        xbinnumb=((x-xmin)*nbin/(xmax-xmin))#calculate x  bin number for each point 
        j=-1
        for i in range(len(xbin)):
            ydata=np.compress(abs(xbinnumb-float(i))<.5,y)
            try:
                #ybin[i]=np.average(ydata)
                ybin[i] = np.median(ydata)
                ybinerr[i] = np.std(ydata)/np.sqrt(float(len(ydata)))
            except ZeroDivisionError:
                ybin[i]=0.
                ybinerr[i]=0.
        return xbin,ybin,ybinerr
            
    def MSfit(self):
        #fit M-S from own data. Use all galaxies above IR limits that
        #aren't AGN.  Also limit ourselves to things were we are
        #reasonably mass complete and to galaxies with SFR>SFR_Ms(Mstar)/X
        fitflag = self.lirflag & ~self.agnflag & (self.logstellarmass > 9.5) & (self.SFR_BEST >=  (.08e-9/6.)*10**self.logstellarmass)

        #popt, pcov = curve_fit(self.linefunc, self.logstellarmass[fitflag], log10(self.SFR_BEST[fitflag]), p0=(1.0,-10.5), bounds=([0.,-12.],[3., -5.])
        p = np.polyfit(self.logstellarmass[fitflag], log10(self.SFR_BEST[fitflag]), 1.)
        print("****************fit parameters",p)
        xmod=arange(8.5,12.,0.2)
        ymod=10**(p[0] * xmod + p[1])
        return xmod,ymod
    #plt.plot(xmod,ymod,'w-',lw=5)
    #plt.plot(xmod,ymod,'b-',lw=3)

    
if __name__ == '__main__':
    homedir = os.environ['HOME']
    g = galaxies(homedir+'/github/LCS/')
