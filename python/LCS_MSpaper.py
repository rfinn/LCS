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
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
import scipy.stats as st
import scipy.special as scisp

import argparse# here is min mass = 9.75

###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Run sextractor, scamp, and swarp to determine WCS solution and make mosaics')
parser.add_argument('--minmass', dest = 'minmass', default = 9., help = 'minimum stellar mass for sample.  default is log10(M*) > 7.9')
parser.add_argument('--diskonly', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')
parser.add_argument('--uvirsfr', dest = 'uvirsfr', default = 1, help = 'True/False (enter 1 or 0). If true, use UV+IR SFRs as opposed to purely IR SFRs.  Default is true.  ')

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
#mstarmax=10.8
minmass=mstarmin #log of M*
#ssfrmin=-12.
#ssfrmax=-9
#spiralcut=0.8
#truncation_ratio=0.5

exterior=.68
colors=['k','b','c','g','m','y','r','sienna','0.5']
shapes=['o','*','p','d','s','^','>','<','v']
#colors=['k','b','c','g','m','y','r','sienna','0.5']

#truncated=np.array([113107,140175,79360,79394,79551,79545,82185,166185,166687,162832,146659,99508,170903,18236,43796,43817,43821,70634,104038,104181],'i')

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

    def setSFR(self):
        USE_UVIRSFR = np.bool(np.float(args.uvirsfr))#True # set to use disk effective radius to normalize 24um size
        self.SFR_USE = self.SFR_BEST
        if USE_UVIRSFR:
            print('Using UV + IR SFR')
            self.SFR_USE = self.SFR_NUV_BEST
            self.logSFR_USE = self.logSFR_NUV_BEST

    
    def plotSFRStellarmassall(self):
        #plot SFR-Mstar showing all galaxies that make our final cut and those that don't

        figure()
        ax=gca()
        ax.set_yscale('log')
        axis([8.8,12,5.e-4,40.])
        #plt.plot(self.logstellarmass,self.SFR_USE,'ko',label='rejected')
        plt.plot(self.logstellarmass,self.SFR_USE,'ko',label='rejected')
        plt.plot(self.logstellarmass[self.sampleflag],self.SFR_USE[self.sampleflag],'ro',label='final sample')
        #axhline(y=.086,c='k',ls='--')
        #axvline(x=9.7,c='k',ls='--')
        plt.xlabel(r'$ M_* \ (M_\odot/yr) $')
        plt.ylabel('$ SFR \ (M_\odot/yr) $')
        g.plotelbaz()
        g.plotlims()

        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()
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
            csampflag = self.sampleflag & self.membflag & (self.gim2d.B_T_r < btcut)
            cnosampflag = ~self.sampleflag & self.membflag & (self.gim2d.B_T_r < btcut)
            esampflag = self.sampleflag & ~self.membflag & (self.gim2d.B_T_r < btcut)
            enosampflag = ~self.sampleflag & ~self.membflag & (self.gim2d.B_T_r < btcut)
        else:
            csampflag = self.sampleflag & self.membflag
            cnosampflag = ~self.sampleflag & self.membflag
            esampflag = self.sampleflag & ~self.membflag
            enosampflag = ~self.sampleflag & ~self.membflag

        
        #plot selection for core galaxies
        plt.subplot(1,2,1)
        ax=gca()
        ax.set_yscale('log')
        bothax=[]
        plt.axis(limits)


        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()

        plt.plot(self.logstellarmass[cnosampflag],self.SFR_USE[cnosampflag],'ko',label='rejected')
        plt.plot(self.logstellarmass[csampflag],self.SFR_USE[csampflag],'ro',label='final sample')
        #plt.xlabel(r'$ M_* \ (M_\odot/yr) $')
        plt.ylabel('$ SFR \ (M_\odot/yr) $')
        #g.plotsalim07()
        #g.plotelbaz()
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
        
        plt.plot(self.logstellarmass[enosampflag],self.SFR_USE[enosampflag],'ko',label='rejected')
        plt.plot(self.logstellarmass[esampflag],self.SFR_USE[esampflag],'ro',label='final sample')
        #g.plotsalim07()
        #g.plotelbaz()
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
        else:
            show()


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
        xmod,ymod,param = g.MSfit()

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
        plt.plot(self.logstellarmass[subsampflag],self.SFR_USE[subsampflag],'ko',markersize=5)
        #sample with selection
        plt.plot(self.logstellarmass[flag],self.SFR_USE[flag],'ro',markersize=4)
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

    def plotSFRStellarmass_sizebin(self, savefig=False, btcutflag=True, specialpoi=False):
        #Make Mstar-SFR plots for exterior and core samples including
        #a binned plot

        minsize=.4
        maxsize=1.5
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(15,12))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[8.8,11.2,3e-2,15.]

        #core galaxies - individual points
        plt.subplot(2,2,1)
        ax=plt.gca()

        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()

        #select members with B/T<btcut
        if btcutflag:
            flag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (self.membflag & self.sampleflag)
        #plt.scatter(self.logstellarmass[flag],self.SFR_USE[flag],c=self.sizeratio[flag],vmin=minsize,vmax=maxsize,cmap='jet_r',s=60, edgecolors='k')
        plt.scatter(self.logstellarmass[flag],self.SFR_USE[flag],c=self.sizeratio[flag],vmin=minsize,vmax=maxsize,cmap='inferno',s=60, edgecolors='k')

        
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

        #plot a subset of points
        if specialpoi:
            spflag = (self.membflag & self.sampleflag) & ((self.s.NSAID==68305) |  (self.s.NSAID == 146606) | (self.s.NSAID == 72738) |  (self.s.NSAID == 166167) |  (self.s.NSAID == 103648)  | (self.s.NSAID == 103791))
            plt.plot(self.logstellarmass[spflag],self.SFR_USE[spflag],'gx',markersize=10,markeredgewidth=2)

        #core plot binned points
        plt.subplot(2,2,2)
        ax=plt.gca()
        xmin = 9.7
        xmax = 10.9
        nbin = (xmax - xmin) / 0.2
        #median SFR  in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_USE[flag])

        #median size ratio  in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sizeratio[flag])
        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='inferno',vmin=minsize,vmax=maxsize,marker='s', edgecolors='k')
        plt.scatter(xbin,ybin,c=sbin,s=300,cmap='inferno',vmin=minsize,vmax=maxsize,marker='s', edgecolors='k')
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
        plt.scatter(self.logstellarmass[flag],self.SFR_USE[flag],c=self.sizeratio[flag],vmin=minsize,vmax=maxsize,cmap='inferno',s=60, edgecolors='k')
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
        xmin = 9.7
        xmax = 10.9
        nbin = (xmax - xmin) / 0.2

        #median SFR  in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_USE[flag])

        #median size ratio  in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sizeratio[flag])
        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='inferno',vmin=minsize,vmax=maxsize,marker='s', edgecolors='k')
        plt.scatter(xbin,ybin,c=sbin,s=300,cmap='inferno',vmin=minsize,vmax=maxsize,marker='s', edgecolors='k')
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
        else:
            show()
        
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

        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)
        
        #core galaxies - individual points
        plt.subplot(2,2,1)
        ax=plt.gca()

        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()

        #select members with B/T<btcut
        if btcutflag:
            flag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (self.membflag & self.sampleflag)

        #plot the individual points.
        plt.scatter(self.logstellarmass[flag],self.SFR_USE[flag],c=log10(self.sfrdense[flag]),vmin=minlsfrdense,vmax=maxlsfrdense,cmap='jet_r',s=60)

        
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
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_USE[flag])

        #median SFR density in bins of mass
        xbin,mubin,mubinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sfrdense[flag])
        #median log(SFR density) in bins of mass
        xbin,lmubin,lmubinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],log10(self.sfrdense[flag]))

        #print(xbin,mubin,mubinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

        #color code by log of the median SFR density
        plt.scatter(xbin,ybin,c=log10(mubin),s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

        #color code by the median of log(SFR density)
        #plt.scatter(xbin,ybin,c=lmubin,s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')
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
        if btcutflag:
            flag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (~self.membflag & self.sampleflag)
            
        plt.scatter(self.logstellarmass[flag],self.SFR_USE[flag],c=log10(self.sfrdense[flag]),vmin=minlsfrdense,vmax=maxlsfrdense,cmap='jet_r',s=60)
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
        xmin = 9.7
        xmax = 10.9
        nbin = (xmax - xmin) / 0.2
        #median SFRs in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[flag],self.SFR_USE[flag])

        #median SFR density in bins of mass
        xbin,mubin,mubinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],self.sfrdense[flag])

        #median log(SFR density) in bins of mass
        xbin,lmubin,mubinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[flag],log10(self.sfrdense[flag]))
        #print(xbin,mubin,mubinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,ybin,c='k',s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

        #color code by log of the median SFR density
        plt.scatter(xbin,ybin,c=log10(mubin),s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')
        
        #color code by the median of log(SFR density)
        #plt.scatter(xbin,ybin,c=lmubin,s=300,cmap='jet_r',vmin=minlsfrdense,vmax=maxlsfrdense,marker='s')

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
        c.ax.text(2.2,.5,'$log_{10}(\Sigma_{SFR}/(M_\odot~yr^{-1}~kpc^{-2}))$',rotation=-90,verticalalignment='center',fontsize=20)

        
        if savefig:
            plt.savefig(figuredir + 'sfr_mstar_musfrcolor.pdf')


    def plotmusfr_optdisksize(self, savefig=False, btcutflag=True):
        #Plot SFR surface density vs optical disk size

        minlsfrdense=-2.5
        maxlsfrdense=-0.5
        minlmstardense = 7.0
        maxlmstardense = 8.5
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[0.,11.9, -2.99, -0.001]

        #SFR surface density
        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)

        #stellar mass surface density
        self.optsize = self.s.SERSIC_TH50 * self.DA
        self.mstardense = 0.5 * 10**(self.logstellarmass) / (np.pi * self.optsize**2)
        
        #core galaxies - individual points
        plt.subplot(2,2,1)
        ax=plt.gca()

        #select members with B/T<btcut
        if btcutflag:
            flag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (self.membflag & self.sampleflag)

        #plot the individual points, color coded by total stellar surface mass density 
        plt.scatter(self.gim2d.Rd[flag],log10(self.sfrdense[flag]),c=log10(self.mstardense[flag]),vmin=minlmstardense,vmax=maxlmstardense,cmap='jet_r',s=60)
        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        ax.set_xticklabels(([]))
        text(0.6,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        plt.title('$SF \ Galaxies$',fontsize=22)

        text(-3.,-3,'$log_{10}(\Sigma_{SFR}/(M_\odot~yr^{-1}~kpc^{-2}))$',rotation=90,verticalalignment='center',fontsize=20)

        print("Core - Spearman Rank")
        (rho,p) = st.spearmanr(self.gim2d.Rd[flag],log10(self.sfrdense[flag]))
        print("rho = ",rho)
        print("p = ",p)
        
        #core plot binned points
        plt.subplot(2,2,2)
        ax=plt.gca()
        xmin = 0.
        xmax = 12.
        #nbin = (xmax - xmin) / 1.
        nbin = (xmax - xmin) / 2.
        #median musfr in bins of Rd
        xbin,musfrbin,musfrbin_err,musfrbin_err_btlow,musfrbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,self.gim2d.Rd[flag],self.sfrdense[flag])

        #median log(Sigma_Mstar) in bins of Rd
        xbin,mumstarbin,mumstarbin_err,mumstarbin_err_btlow,mumstarbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,self.gim2d.Rd[flag],self.mstardense[flag])

        #have to redo the errors because it's in log space
        lmusfrbin_err_btlow = log10(musfrbin) - log10(musfrbin - musfrbin_err_btlow)
        lmusfrbin_err_bthigh = log10(musfrbin + musfrbin_err_bthigh) - log10(musfrbin)
        
        #print(xbin,mubin,mubinerr)
        errorbar(xbin,log10(musfrbin),yerr=[lmusfrbin_err_btlow,lmusfrbin_err_bthigh],fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,log10(musfrbin),c='k',s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        plt.scatter(xbin,log10(musfrbin),c=log10(mumstarbin),s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        
        #gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        plt.title('$Median $',fontsize=22)

        axhline(y=-2.0,c='k',ls='--')

        legend(loc='upper left',numpoints=1)

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            text(-0.15,1.1,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        ###############
        #external individual points
        plt.subplot(2,2,3)
        ax=plt.gca()

        #select non-members with B/T<btcut
        if btcutflag:
            flag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (~self.membflag & self.sampleflag)
            
       #plot the individual points.
        plt.scatter(self.gim2d.Rd[flag],log10(self.sfrdense[flag]),c=log10(self.mstardense[flag]),vmin=minlmstardense,vmax=maxlmstardense,cmap='jet_r',s=60)
        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        text(0.6,0.9,'$External$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        
        print("External - Spearman Rank")
        (rho,p) = st.spearmanr(self.gim2d.Rd[flag],log10(self.sfrdense[flag]))
        print("rho = ",rho)
        print("p = ",p)

        #external plot binned points
        plt.subplot(2,2,4)
        ax=plt.gca()
        xmin = 0.
        xmax = 12.
        nbin = (xmax - xmin) / 2.
        #median musfr in bins of Rd
        xbin,musfrbin,musfrbin_err,musfrbin_err_btlow,musfrbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,self.gim2d.Rd[flag],self.sfrdense[flag])

        #median log(Sigma_Mstar) in bins of Rd
        xbin,mumstarbin,mumstarbin_err,mumstarbin_err_btlow,mumstarbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,self.gim2d.Rd[flag],self.mstardense[flag])

        lmusfrbin_err_btlow = log10(musfrbin) - log10(musfrbin - musfrbin_err_btlow)
        lmusfrbin_err_bthigh = log10(musfrbin + musfrbin_err_bthigh) - log10(musfrbin)
        
        #print(xbin,mubin,mubinerr)
        errorbar(xbin,log10(musfrbin),yerr=[lmusfrbin_err_btlow,lmusfrbin_err_bthigh],fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,log10(musfrbin),c='k',s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        plt.scatter(xbin,log10(musfrbin),c=log10(mumstarbin),s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        
        #gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))

        axhline(y=-2.0,c='k',ls='--')

        legend(loc='upper left',numpoints=1)

        text(-0.02,-.2,'$R_e(r)/kpc$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)
        
        c=colorbar(ax=bothax,fraction=.05,ticks=arange(minlmstardense,maxlmstardense,.1),format='%.1f')
        c.ax.text(2.2,.5,'$log_{10}(\Sigma_{\star}/(M_\odot~kpc^{-2}))$',rotation=-90,verticalalignment='center',fontsize=20)
        
        if savefig:
            plt.savefig(figuredir + 'musfr_optdisksize.pdf')


    def plotmusfr_mustar(self, savefig=False, btcutflag=True):
        #Plot SFR surface density vs optical disk size

        minlsfrdense=-2.5
        maxlsfrdense=-0.5
        minlmstardense = 7.0
        maxlmstardense = 8.5
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[6.51,8.99, -2.99, -0.001]

        #SFR surface density
        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)

        #stellar mass surface density
        self.optsize = self.s.SERSIC_TH50 * self.DA
        self.mstardense = 0.5 * 10**(self.logstellarmass) / (np.pi * self.optsize**2)
        
        #core galaxies - individual points
        plt.subplot(2,2,1)
        ax=plt.gca()

        #select members with B/T<btcut
        if btcutflag:
            flag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (self.membflag & self.sampleflag)

        #plot the individual points, color coded by total stellar surface mass density 
        plt.scatter(log10(self.mstardense[flag]),log10(self.sfrdense[flag]),c='r',vmin=minlmstardense,vmax=maxlmstardense,cmap='jet_r',s=60)
        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        ax.set_xticklabels(([]))
        text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        plt.title('$SF \ Galaxies$',fontsize=22)

        #plt.ylabel('$log_{10}(\Sigma_{SFR}/(M_\odot~yr^{-1}~kpc^{-2}))$',fontsize=20)
        text(6.0, -3,'$log_{10}(\Sigma_{SFR}/(M_\odot~yr^{-1}~kpc^{-2}))$',rotation=90,verticalalignment='center',fontsize=20)

        print("Core - Spearman Rank")
        (rho,p) = st.spearmanr(log10(self.mstardense[flag]),log10(self.sfrdense[flag]))
        print("rho = ",rho)
        print("p = ",p)
        
        #core plot binned points
        plt.subplot(2,2,2)
        ax=plt.gca()
        xmin = 7.
        xmax = 8.4
        nbin = (xmax - xmin) / 0.2
        #median musfr in bins of Rd
        xbin,musfrbin,musfrbin_err,musfrbin_err_btlow,musfrbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,log10(self.mstardense[flag]),self.sfrdense[flag])

        #median log(Sigma_Mstar) in bins of Rd
        #xbin,mumstarbin,mumstarbin_err,mumstarbin_err_btlow,mumstarbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,self.gim2d.Rd[flag],self.mstardense[flag])

        #have to redo the errors because it's in log space
        lmusfrbin_err_btlow = log10(musfrbin) - log10(musfrbin - musfrbin_err_btlow)
        lmusfrbin_err_bthigh = log10(musfrbin + musfrbin_err_bthigh) - log10(musfrbin)
        
        #print(xbin,mubin,mubinerr)
        errorbar(xbin,log10(musfrbin),yerr=[lmusfrbin_err_btlow,lmusfrbin_err_bthigh],fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,log10(musfrbin),c='r',s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        #plt.scatter(xbin,log10(musfrbin),c=log10(mumstarbin),s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        
        #gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        plt.title('$Median $',fontsize=22)

        axhline(y=-2.0,c='k',ls='--')

        legend(loc='upper left',numpoints=1)

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            text(-0.15,1.1,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        ###############
        #external individual points
        plt.subplot(2,2,3)
        ax=plt.gca()

        #select non-members with B/T<btcut
        if btcutflag:
            flag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (~self.membflag & self.sampleflag)
            
       #plot the individual points.
        plt.scatter(log10(self.mstardense[flag]),log10(self.sfrdense[flag]),c='r',vmin=minlmstardense,vmax=maxlmstardense,cmap='jet_r',s=60)
        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        text(0.1,0.9,'$External$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        
        print("External - Spearman Rank")
        (rho,p) = st.spearmanr(log10(self.mstardense[flag]),log10(self.sfrdense[flag]))
        print("rho = ",rho)
        print("p = ",p)

        #external plot binned points
        plt.subplot(2,2,4)
        ax=plt.gca()
        xmin = 7.
        xmax = 8.4
        nbin = (xmax - xmin) / 0.2
        #median musfr in bins of Rd
        xbin,musfrbin,musfrbin_err,musfrbin_err_btlow,musfrbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,log10(self.mstardense[flag]),self.sfrdense[flag])

        #median log(Sigma_Mstar) in bins of Rd
        #xbin,mumstarbin,mumstarbin_err,mumstarbin_err_btlow,mumstarbin_err_bthigh =self.binitbinsbt(xmin, xmax, nbin ,self.gim2d.Rd[flag],self.mstardense[flag])

        lmusfrbin_err_btlow = log10(musfrbin) - log10(musfrbin - musfrbin_err_btlow)
        lmusfrbin_err_bthigh = log10(musfrbin + musfrbin_err_bthigh) - log10(musfrbin)
        
        #print(xbin,mubin,mubinerr)
        errorbar(xbin,log10(musfrbin),yerr=[lmusfrbin_err_btlow,lmusfrbin_err_bthigh],fmt=None,color='k',markersize=16,ecolor='k')
        plt.scatter(xbin,log10(musfrbin),c='r',s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        #plt.scatter(xbin,log10(musfrbin),c=log10(mumstarbin),s=300,cmap='jet_r',vmin=minlmstardense,vmax=maxlmstardense,marker='s')
        
        #gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))

        axhline(y=-2.0,c='k',ls='--')

        legend(loc='upper left',numpoints=1)

        text(-0.02,-.2,'$log_{10}(\Sigma_{\star}/(M_\odot~kpc^{-2}))$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)
        
        #c=colorbar(ax=bothax,fraction=.05,ticks=arange(minlmstardense,maxlmstardense,.1),format='%.1f')
        #c.ax.text(2.2,.5,'$log_{10}(\Sigma_{\star}/(M_\odot~kpc^{-2}))$',rotation=-90,verticalalignment='center',fontsize=20)
        
        if savefig:
            plt.savefig(figuredir + 'musfr_mustar.pdf')

    def sizediff_musfrdiff_mass(self, savefig=False, btcutflag=True,sfrlimflag=False):
        #make a plot of the difference in size ratios and SFR surface
        #density between the core and external sample as a function of
        #stellar mass
        #if sfrlim==True the limit galaxies to within a factor of the main sequence.

        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]

        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)

        #determine offsets w.r.t. main sequence
        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()
        
        sfrpred = 10**(param[0] * self.logstellarmass + param[1])
        lsfrdiff = log10(self.SFR_USE) - log10(sfrpred)

        #what is the difference in SFR with respect to the main
        #sequence for which we will perform the  calculation
        lsfrthresh = 100.0
        if sfrlimflag:
            #lsfrthresh = 0.6 #factor of 4
            #lsfrthresh = 0.477 #factor of 3
            lsfrthresh = 0.3 #factor of 2
        
        #select core with B/T<btcut and within a certain distance of the main sequence
        if btcutflag:
            cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (abs(lsfrdiff) < lsfrthresh)
        else:
            cflag = (self.membflag & self.sampleflag) & (abs(lsfrdiff) < lsfrthresh)

        #select external galaxies with B/T<btcut
        if btcutflag:
            eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (abs(lsfrdiff) < lsfrthresh)
        else:
            eflag = (~self.membflag & self.sampleflag) & (abs(lsfrdiff) < lsfrthresh)

        logmassmin = 9.7
        logmassmax = 10.9
        nbin = (logmassmax - logmassmin) / 0.2

        ################
        #the size difference as a function of mass, binned
        plt.subplot(3,1,1)
        ax=plt.gca()


        #median size ratio in bins of mass for core
        xbin,sbinc,sbinc_err,sbinc_err_btlow,sbinc_err_bthigh = self.binitbinsbt(logmassmin, logmassmax, nbin,self.logstellarmass[cflag],self.sizeratio[cflag])

        #median size ratio in bins of mass for external
        xbin,sbine,sbine_err,sbine_err_btlow,sbine_err_bthigh = self.binitbinsbt(logmassmin, logmassmax, nbin,self.logstellarmass[eflag],self.sizeratio[eflag])

        #symmetrize the 2-sided bootstrap errors
        symarrc = (sbinc_err_bthigh + sbinc_err_btlow) / 2.
        symarre = (sbine_err_bthigh + sbine_err_btlow) / 2.

        #ratio of median size ratio
        sizerat_rat = sbinc / sbine
        sizerat_rat_err = np.sqrt(sizerat_rat**2 * ((sbine_err/sbine)**2 + (sbinc_err/sbinc)**2))
        sizerat_rat_errbt = np.sqrt(sizerat_rat**2 * ((symarre/sbine)**2 + (symarrc/sbinc)**2))
        #asymmetric errorbars
        sizerat_rat_errbthigh = np.sqrt(sizerat_rat**2 * ((sbine_err_bthigh/sbine)**2 + (sbinc_err_bthigh/sbinc)**2))
        sizerat_rat_errbtlow = np.sqrt(sizerat_rat**2 * ((sbine_err_btlow/sbine)**2 + (sbinc_err_btlow/sbinc)**2))
        
        #print(xbin,sbin,sbinerr)
        #errorbar(xbin,sizerat_rat,yerr=sizerat_rat_errbt,fmt='rs',color='k',markersize=8,ecolor='k',mfc='r')
        errorbar(xbin,sizerat_rat,yerr=[ sizerat_rat_errbthigh, sizerat_rat_errbtlow],fmt='rs',color='k',markersize=8,ecolor='k',mfc='r')
    #plt.scatter(xbin,sizerat_rat,c='r',s=30,marker='s')

        limits=[logmassmin - 0.2,logmassmax + 0.2,0,2.49]
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        #plt.title('$Median $',fontsize=22)

        axhline(y=1,c='k',ls='--')

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            plt.title(s,fontsize=20)
            #text(-0.15,1.1,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        #plt.ylabel('$[R_e(24)/R_e(r)]_{core}/[R_e(24)/R_e(r)]_{ext}$',fontsize=16)
        plt.ylabel('$R_e(24)/R_e(r)~ratio$',fontsize=18)
        text(0.35,0.8,'$[R_e(24)/R_e(r)]_{core}/[R_e(24)/R_e(r)]_{ext}$',transform=ax.transAxes,horizontalalignment='left',fontsize=18)
        #plt.ylabel('$\frac{(R_e(24)/R_e(r))_{core}}{(R_e(24)/R_e(r))_{ext}}$',fontsize=12)

        if sfrlimflag:
            s = '$|log(SFR) - log(SFR_{MS})|<  \  %.2f$'%(lsfrthresh)
            text(0.4,0.8,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)
            
        ##########################
        #the musfr difference as a function of mass
        plt.subplot(3,1,2)
        ax=plt.gca()

        #median size ratio in bins of mass for core
        xbin,mubinc,mubinc_err ,mubinc_err_btlow,mubinc_err_bthigh = self.binitbinsbt(logmassmin, logmassmax, nbin,self.logstellarmass[cflag],self.sfrdense[cflag])

        #median size ratio in bins of mass for external
        xbin,mubine,mubine_err,mubine_err_btlow,mubine_err_bthigh = self.binitbinsbt(logmassmin, logmassmax, nbin,self.logstellarmass[eflag],self.sfrdense[eflag])

        #symmetrize the 2-sided bootstrap errors
        symarrc = (mubinc_err_bthigh + mubinc_err_btlow) / 2.
        symarre = (mubine_err_bthigh + mubine_err_btlow) / 2.

        #ratio of median size ratio
        musfr_rat = mubinc / mubine
        musfr_rat_err = np.sqrt(musfr_rat**2 * ((mubine_err/mubine)**2 + (mubinc_err/mubinc)**2))
        musfr_rat_errbt = np.sqrt(musfr_rat**2 * ((symarre/mubine)**2 + (symarrc/mubinc)**2))

        #asymmetric errorbars
        musfrrat_rat_errbthigh = np.sqrt(musfr_rat**2 * ((mubine_err_bthigh/mubine)**2 + (mubinc_err_bthigh/mubinc)**2))
        musfrrat_rat_errbtlow = np.sqrt(musfr_rat**2 * ((mubine_err_btlow/mubine)**2 + (mubinc_err_btlow/mubinc)**2))

        #errorbar(xbin,musfr_rat,yerr=musfr_rat_errbt,fmt='rs',color='k',markersize=8,ecolor='k',mfc='r')
        #errorbar(xbin,musfr_rat,yerr=musfr_rat_errbt,fmt='none',color='k',ecolor='k')
        errorbar(xbin,musfr_rat,yerr=[musfrrat_rat_errbthigh ,musfrrat_rat_errbtlow],fmt='rs',color='k',markersize=8,ecolor='k',mfc='r')
        errorbar(xbin,musfr_rat,yerr=[musfrrat_rat_errbthigh ,musfrrat_rat_errbtlow],fmt='none',color='k',ecolor='k')

        limits=[logmassmin - 0.2,logmassmax + 0.2,0,3.99]
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        #plt.title('$Median $',fontsize=22)

        plt.ylabel('$\Sigma_{SFR}~ratio$',fontsize=18)
        text(0.35,0.8,'$\Sigma_{SFR,core}/\Sigma_{SFR,ext}$',transform=ax.transAxes,horizontalalignment='left',fontsize=18)
        axhline(y=1,c='k',ls='--')


        ##########################
        #the SFR difference as a function of mass
        plt.subplot(3,1,3)
        ax=plt.gca()

        #median SFR in bins of mass for core
        xbin,sfrbinc,sfrbinc_err ,sfrbinc_err_btlow,sfrbinc_err_bthigh = self.binitbinsbt(logmassmin, logmassmax, nbin,self.logstellarmass[cflag],self.SFR_USE[cflag])

        #median SFR in bins of mass for external
        xbin,sfrbine,sfrbine_err,sfrbine_err_btlow,sfrbine_err_bthigh = self.binitbinsbt(logmassmin, logmassmax, nbin,self.logstellarmass[eflag],self.SFR_USE[eflag])

        #symmetrize the 2-sided bootstrap errors
        symarrc = (sfrbinc_err_bthigh + sfrbinc_err_btlow) / 2.
        symarre = (sfrbine_err_bthigh + sfrbine_err_btlow) / 2.

        #ratio of median size ratio
        sfr_rat = sfrbinc / sfrbine
        sfr_rat_err = np.sqrt(sfr_rat**2 * ((sfrbine_err/sfrbine)**2 + (sfrbinc_err/sfrbinc)**2))
        sfr_rat_errbt = np.sqrt(sfr_rat**2 * ((symarre/sfrbine)**2 + (symarrc/sfrbinc)**2))
        
        #asymmetric errorbars
        sfr_rat_errbthigh = np.sqrt(sfr_rat**2 * (((sfrbine_err_bthigh/sfrbine)**2 + (sfrbinc_err_bthigh/sfrbinc)**2)))
        sfr_rat_errbtlow = np.sqrt(sfr_rat**2 * (((sfrbine_err_btlow/sfrbine)**2 + (sfrbinc_err_btlow/sfrbinc)**2)))

        #errorbar(xbin,sfr_rat,yerr=sfr_rat_errbt,fmt='rs',color='k',markersize=8,ecolor='k',mfc='r')
        #errorbar(xbin,sfr_rat,yerr=sfr_rat_errbt,fmt='none',color='k',ecolor='k')
        errorbar(xbin,sfr_rat,yerr=[sfr_rat_errbthigh,sfr_rat_errbtlow],fmt='rs',color='k',markersize=8,ecolor='k',mfc='r')
        errorbar(xbin,sfr_rat,yerr=[sfr_rat_errbthigh,sfr_rat_errbtlow],fmt='none',color='k',ecolor='k')

        limits=[logmassmin - 0.2,logmassmax + 0.2,0,3.99]
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_yticklabels(([]))
        #ax.set_xticklabels(([]))
        #plt.title('$Median $',fontsize=22)
        plt.ylabel('${SFR}~ratio$',fontsize=18)
        text(0.35,0.8,'$SFR_{core}/SFR_{ext}$',transform=ax.transAxes,horizontalalignment='left',fontsize=18)

        axhline(y=1,c='k',ls='--')
            
        plt.xlabel('$log_{10}(M_*/M_\odot)$',fontsize=20)

        if savefig:
            plt.savefig(figuredir + 'sizediff_musfrdiff_mass.pdf')

    def musfr_size(self, savefig=False, btcutflag=True):
        #make a plot of the relation between SFR surface density and size ratio
        
        minsize=0.0
        maxsize=2.49

        minlsfrdense=-3
        maxlsfrdense=-0.0

        minlogmass = 8.8
        maxlogmass = 11.2
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[minsize,maxsize,minlsfrdense,maxlsfrdense]

        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)
        
        #core galaxies - individual points
        plt.subplot(1,2,1)
        ax=plt.gca()

        #select members with B/T<btcut
        if btcutflag:
            flag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (self.membflag & self.sampleflag)

        #plot the individual points.
        plt.scatter(self.sizeratio[flag],log10(self.sfrdense[flag]),c=self.logstellarmass[flag],vmin=minlogmass,vmax=maxlogmass,cmap='jet_r',s=60)

        #spearman-rank coefficient 
        rho,p = st.spearmanr(self.sizeratio[flag],log10(self.sfrdense[flag]))
        text(.4,.8,r'$\rho = %4.2f$'%(rho),horizontalalignment='left',transform=ax.transAxes,fontsize=20)
        text(.4,.7,'$p = %5.2e$'%(p),horizontalalignment='left',transform=ax.transAxes,fontsize=20)

        #plot line showing intrinsic correlation
        x = arange(0.01,2.49,0.01)
        y = log10(1/x**2) - 2.
        plt.plot(x,y,'k--',lw=3)
        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_yticklabels(([]))
        text(0.5,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        s1 = '$SF~Galaxies$'
        text(0.9,1.02,s1,transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        if btcutflag:
            s2 = '$B/T \ <  \  %.2f$'%(btcut)
            text(0.9,1.07,s2,transform=ax.transAxes,horizontalalignment='left',fontsize=20)
           
        plt.ylabel('$log_{10}(\Sigma_{SFR}/(M_\odot~yr^{-1}~kpc^{-2}))$')

        ##############################
        #external galaxies - individual points
        plt.subplot(1,2,2)
        ax=plt.gca()

        #select non-members with B/T<btcut
        if btcutflag:
            flag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
        else:
            flag = (~self.membflag & self.sampleflag)

        #plot the individual points.
        plt.scatter(self.sizeratio[flag],log10(self.sfrdense[flag]),c=self.logstellarmass[flag],vmin=8.8,vmax=11.2,cmap='jet_r',s=60)

        rho,p = st.spearmanr(self.sizeratio[flag],log10(self.sfrdense[flag]))
        text(.4,.8,r'$\rho = %4.2f$'%(rho),horizontalalignment='left',transform=ax.transAxes,fontsize=20)
        text(.4,.7,'$p = %5.2e$'%(p),horizontalalignment='left',transform=ax.transAxes,fontsize=20)

        #plot line showing intrinsic correlation
        x = arange(0.01,2.49,0.01)
        y = log10(1/x**2) - 2.
        plt.plot(x,y,'k--',lw=3)

        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        ax.set_yticklabels(([]))
        text(0.5,0.9,'$External$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        text(-0.02,-.2,'$R_e(24)/R_e(r)$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)

        c=colorbar(ax=bothax,fraction=.05,ticks=arange(minlogmass,maxlogmass,.1),format='%.1f')
        c.ax.text(2.2,.5,'$log_{10}(M_*/M_\odot)$',rotation=-90,verticalalignment='center',fontsize=20)

        if savefig:
            plt.savefig(figuredir + 'size_musfr_masscolor.pdf')

    def sfr_offset(self,savefig=False,btcutflag=True,logmassmin=9.7, logmassmax=12.0,morphsamp=False):
        #compares the distribution w.r.t. the main sequence for both
        #the core and external galaxies
        
        #logmassmin and logmassmax set limits on mass for the histograms
                
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,4))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[-1.9,1.3,0, 22]

        #determine offsets w.r.t. main sequence
        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()
        
        sfrpred = 10**(param[0] * self.logstellarmass + param[1])
        lsfrdiff = log10(self.SFR_USE) - log10(sfrpred)

        #core galaxies. plot MS offsets
        plt.subplot(2,1,1)
        ax=plt.gca()

        #choose samples based on morphological cuts or not.
        if morphsamp:
        #select members with B/T<btcut
            if btcutflag:
                cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
            else:
                    cflag = (self.membflag & self.sampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
        else:
            cflag = (self.membflag & self.sfsampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
        
        lsfrdiffmin = limits[0]
        lsfrdiffmax = limits[1]
        mybins=np.arange(lsfrdiffmin, lsfrdiffmax, 0.1)
        plt.hist(lsfrdiff[cflag],bins=mybins,histtype='stepfilled',color='r',label='$Core$',lw=1.5,alpha=1,edgecolor='k')#,normed=True)

        #median SFR diff
        medlsfrdiff = np.median(lsfrdiff[cflag])
        plt.axvline(x=medlsfrdiff,c='k',ls='-',lw=3)

        #68% confidence intervals on SFRdiff
        lsfrdiffsort = sort(lsfrdiff[cflag])
        confint1 = 0.68                           #confidence interval
        lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
        highind1 = int(round((1-((1 - confint1) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind1],c='k',ls='--',lw=3)
        plt.axvline(x=lsfrdiffsort[highind1],c='k',ls='--',lw=3)
                
        confint2 = 0.90                           #confidence interval
        lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
        highind2 = int(round((1-((1 - confint2) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind2],c='k',ls=':',lw=3)
        plt.axvline(x=lsfrdiffsort[highind2],c='k',ls=':',lw=3)

        #bootstrap the sample to determine the robustness of the lower
        #SFR tail to sampling issues
        niter = 1000
        with NumpyRNGContext(1):    #assures reproducibility of monte carlo
            btlsfrdiff = bootstrap(lsfrdiff[cflag],bootnum=niter)

        btconflimlowc1 = np.zeros(niter)        #bootstrap 68% lower limits
        btconflimlowc2 = np.zeros(niter)        #bootstrap 90% lower limits
        for iter in range(niter):
            btlsfrdiffsort = sort(btlsfrdiff[iter,:])
            confint1 = 0.68                           #confidence interval
            lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
            btconflimlowc1[iter] = btlsfrdiffsort[lowind1]

            confint2 = 0.90                           #confidence interva
            lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
            btconflimlowc2[iter] = btlsfrdiffsort[lowind2]

        print("Core")
        print("median lower 68% SFRdiff confidence interval",np.median(btconflimlowc1))
        print("median lower 90% SFRdiff confidence interval",np.median(btconflimlowc2))
                
        plt.axis(limits)
        bothax.append(ax)
        ax.set_xticklabels(([]))
        plt.ylabel('$N_{gal}$',fontsize=18)
        plt.legend(loc='upper right')

        if morphsamp and btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            #text(0.5,1.05,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)
            plt.title(s,fontsize=20)

        if logmassmin>9.7:
            s = '$%.2f < log(M_\star/M_\odot) < %.2f$'%(logmassmin, logmassmax)
            text(0.4,0.8,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        ##################
        #external galaxies. plot MS offsets
        plt.subplot(2,1,2)
        ax=plt.gca()

        #choose samples based on morphological cuts or not.
        if morphsamp:
            #select external galaxies with B/T<btcut
            if btcutflag:
                eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
            else:
                eflag = (~self.membflag & self.sampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)

        else:
            eflag = (~self.membflag & self.sfsampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)

        hist(lsfrdiff[eflag],bins=mybins,histtype='stepfilled',color='b',label='$External$',lw=1.5,alpha=1,edgecolor='k')#,normed=True)

        #median SFR diff
        medlsfrdiff = np.median(lsfrdiff[eflag])
        plt.axvline(x=medlsfrdiff,c='k',ls='-',lw=3)

        #68% confidence intervals on SFRdiff
        lsfrdiffsort = sort(lsfrdiff[eflag])
        confint1 = 0.68                           #confidence interval
        lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
        highind1 = int(round((1-((1 - confint1) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind1],c='k',ls='--',lw=3)
        plt.axvline(x=lsfrdiffsort[highind1],c='k',ls='--',lw=3)
                
        confint2 = 0.90                           #confidence interval
        lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
        highind2 = int(round((1-((1 - confint2) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind2],c='k',ls=':',lw=3)
        plt.axvline(x=lsfrdiffsort[highind2],c='k',ls=':',lw=3)

        #bootstrap the sample to determine the robustness of the lower
        #SFR tail to sampling issues
        niter = 1000
        with NumpyRNGContext(1):    #assures reproducibility of monte carlo
            btlsfrdiff = bootstrap(lsfrdiff[eflag],bootnum=niter)

        btconflimlowe1 = np.zeros(niter)        #bootstrap 68% lower limits
        btconflimlowe2 = np.zeros(niter)        #bootstrap 90% lower limits
        for iter in range(niter):
            btlsfrdiffsort = sort(btlsfrdiff[iter,:])
            confint1 = 0.68                           #confidence interval
            lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
            btconflimlowe1[iter] = btlsfrdiffsort[lowind1]

            confint2 = 0.90                           #confidence interva
            lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
            btconflimlowe2[iter] = btlsfrdiffsort[lowind2]

        print("External")
        print("median lower 68% SFRdiff confidence interval",np.median(btconflimlowe1))
        print("median lower 90% SFRdiff confidence interval",np.median(btconflimlowe2))

        #find how often the core lower 68 and 90% confidence limits are below the field
        nlow1 = np.where(btconflimlowc1 < btconflimlowe1)
        nlow2 = np.where(btconflimlowc2 < btconflimlowe2)
        nlow1norm = float(size(nlow1)) / float(niter)
        nlow2norm = float(size(nlow2)) / float(niter)

        print("Core has lower 68% SFR limit",nlow1norm,"of the time")
        print("Core has lower 90% SFR limit",nlow2norm,"of the time")
        
        #K-S test
        a,b=ks(lsfrdiff[cflag],lsfrdiff[eflag])

        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        plt.text(-.2,1,'$N_{gal}$',transform=gca().transAxes,verticalalignment='center',rotation=90,fontsize=24)
        plt.legend(loc='upper right')

        plt.ylabel('$N_{gal}$',fontsize=18)
        plt.xlabel('log(SFR) - log(SFR$_{MS}$)',fontsize=18)
        
        if savefig:
            plt.savefig(figuredir + 'sfrdiff.pdf')
        else:
            plt.show()

    def sfr_offset_matchsamp(self,savefig=False,btcutflag=True,logmassmin=9.7, logmassmax=12.0):
        #compares the distribution w.r.t. the main sequence for core`
        #galaxies that have different size offsets w.r.t. field
        
        matchflag, matchind = self.matchsamp_masssize(btcutflag=btcutflag)
                
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[-1.9,1.3,0, 22]

        #determine offsets w.r.t. main sequence
        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()
        
        sfrpred = 10**(param[0] * self.logstellarmass + param[1])
        lsfrdiff  = log10(self.SFR_USE) - log10(sfrpred)

        ################################################
        #first, just select members with SFRs, with no extra selection on morphology
        plt.subplot(4,1,1)
        ax=plt.gca()

        cflag =  (self.membflag & self.sfsampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)

        
        # #matchflag = cflag & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
        # #choose samples based on morphological cuts or not.
        # if morphsamp:
        # #select members with B/T<btcut
        #     if btcutflag:
        #         cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
        #     else:
        #             cflag = (self.membflag & self.sampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
        # else:
        #     cflag = (self.membflag & self.sfsampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)

        lsfrdiff_sfsamp = lsfrdiff[cflag]
        lsfrdiffmin = limits[0]
        lsfrdiffmax = limits[1]
        mybins=np.arange(lsfrdiffmin, lsfrdiffmax, 0.1)
        plt.hist(lsfrdiff_sfsamp,bins=mybins,histtype='stepfilled',color='r',label='$All$',lw=1.5,alpha=1,edgecolor='k')#,normed=True)

        #median SFR diff
        medlsfrdiff = np.median(lsfrdiff_sfsamp)
        plt.axvline(x=medlsfrdiff,c='k',ls='-',lw=3)

        #68% confidence intervals on SFRdiff
        lsfrdiffsort = sort(lsfrdiff_sfsamp)
        confint1 = 0.68                           #confidence interval
        lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
        highind1 = int(round((1-((1 - confint1) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind1],c='k',ls='--',lw=3)
        plt.axvline(x=lsfrdiffsort[highind1],c='k',ls='--',lw=3)
                
        confint2 = 0.90                           #confidence interval
        lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
        highind2 = int(round((1-((1 - confint2) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind2],c='k',ls=':',lw=3)
        plt.axvline(x=lsfrdiffsort[highind2],c='k',ls=':',lw=3)
        plt.legend(loc='upper right')

        plt.ylabel('$N_{gal}$',fontsize=18)

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            #text(0.5,1.05,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)
            plt.title(s,fontsize=20)
            
        ###############################
        #select morphological subset of galaxies

        plt.subplot(4,1,2)
        ax=plt.gca()

        cflag =  matchflag & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)

        lsfrdiff_morphsamp = lsfrdiff[cflag]
        lsfrdiffmin = limits[0]
        lsfrdiffmax = limits[1]
        mybins=np.arange(lsfrdiffmin, lsfrdiffmax, 0.1)
        plt.hist(lsfrdiff_morphsamp,bins=mybins,histtype='stepfilled',color='r',label='$Morph$',lw=1.5,alpha=1,edgecolor='k')#,normed=True)

        #median SFR diff
        medlsfrdiff = np.median(lsfrdiff_morphsamp)
        plt.axvline(x=medlsfrdiff,c='k',ls='-',lw=3)

        #68% confidence intervals on SFRdiff
        lsfrdiffsort = sort(lsfrdiff_morphsamp)
        confint1 = 0.68                           #confidence interval
        lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
        highind1 = int(round((1-((1 - confint1) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind1],c='k',ls='--',lw=3)
        plt.axvline(x=lsfrdiffsort[highind1],c='k',ls='--',lw=3)
                
        confint2 = 0.90                           #confidence interval
        lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
        highind2 = int(round((1-((1 - confint2) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind2],c='k',ls=':',lw=3)
        plt.axvline(x=lsfrdiffsort[highind2],c='k',ls=':',lw=3)
        plt.legend(loc='upper right')


        plt.ylabel('$N_{gal}$',fontsize=18)

        ###############################
        #select morphological subset of galaxies that have same 24um
        #size as matched field

        lsizediffthresh = -0.2
        
        plt.subplot(4,1,3)
        ax=plt.gca()

        #I need to redo these because my matched sample only includes
        #cluster members with good profile fits.
        sfrpred = 10**(param[0] * self.diffsamplmstar + param[1])
        lsfrdiff = log10(self.diffsampSFR) - log10(sfrpred)


        cflag = (self.diffsamplmstar > logmassmin) & (self.diffsamplmstar < logmassmax) & (self.difflmipssize > lsizediffthresh)

        lsfrdiff_normsize = lsfrdiff[cflag]
        lsfrdiffmin = limits[0]
        lsfrdiffmax = limits[1]
        labstr = '$\Delta log R_{24}> \ %0.2f$'%(lsizediffthresh)
        mybins=np.arange(lsfrdiffmin, lsfrdiffmax, 0.1)
        plt.hist(lsfrdiff_normsize,bins=mybins,histtype='stepfilled',color='r',label=labstr,lw=1.5,alpha=1,edgecolor='k')#,normed=True)

        #median SFR diff
        medlsfrdiff = np.median(lsfrdiff_normsize)
        plt.axvline(x=medlsfrdiff,c='k',ls='-',lw=3)

        #68% confidence intervals on SFRdiff
        lsfrdiffsort = sort(lsfrdiff_normsize)
        confint1 = 0.68                           #confidence interval
        lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
        highind1 = int(round((1-((1 - confint1) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind1],c='k',ls='--',lw=3)
        plt.axvline(x=lsfrdiffsort[highind1],c='k',ls='--',lw=3)
                
        confint2 = 0.90                           #confidence interval
        lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
        highind2 = int(round((1-((1 - confint2) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind2],c='k',ls=':',lw=3)
        plt.axvline(x=lsfrdiffsort[highind2],c='k',ls=':',lw=3)
        plt.legend(loc='upper right')

        plt.ylabel('$N_{gal}$',fontsize=18)

        ###############################
        #select morphological subset of galaxies that have  24um
        #size significantly less than the matched field

        plt.subplot(4,1,4)
        ax=plt.gca()

        cflag = (self.diffsamplmstar > logmassmin) & (self.diffsamplmstar < logmassmax) & (self.difflmipssize <= lsizediffthresh)
        
        lsfrdiff_smallsize = lsfrdiff[cflag]
        lsfrdiffmin = limits[0]
        lsfrdiffmax = limits[1]
        labstr = '$\Delta log R_{24}\leq \ %0.2f$'%(lsizediffthresh)
        mybins=np.arange(lsfrdiffmin, lsfrdiffmax, 0.1)
        plt.hist(lsfrdiff_smallsize,bins=mybins,histtype='stepfilled',color='r',label=labstr,lw=1.5,alpha=1,edgecolor='k')#,normed=True)

        #median SFR diff
        medlsfrdiff = np.median(lsfrdiff_smallsize)
        plt.axvline(x=medlsfrdiff,c='k',ls='-',lw=3)

        #68% confidence intervals on SFRdiff
        lsfrdiffsort = sort(lsfrdiff_smallsize)
        confint1 = 0.68                           #confidence interval
        lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
        highind1 = int(round((1-((1 - confint1) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind1],c='k',ls='--',lw=3)
        plt.axvline(x=lsfrdiffsort[highind1],c='k',ls='--',lw=3)
                
        confint2 = 0.90                           #confidence interval
        lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
        highind2 = int(round((1-((1 - confint2) / 2)) * len(lsfrdiffsort),2))
        
        plt.axvline(x=lsfrdiffsort[lowind2],c='k',ls=':',lw=3)
        plt.axvline(x=lsfrdiffsort[highind2],c='k',ls=':',lw=3)


        # #bootstrap the sample to determine the robustness of the lower
        # #SFR tail to sampling issues
        # niter = 1000
        # with NumpyRNGContext(1):    #assures reproducibility of monte carlo
        #     btlsfrdiff = bootstrap(lsfrdiff[cflag],bootnum=niter)

        # btconflimlowc1 = np.zeros(niter)        #bootstrap 68% lower limits
        # btconflimlowc2 = np.zeros(niter)        #bootstrap 90% lower limits
        # for iter in range(niter):
        #     btlsfrdiffsort = sort(btlsfrdiff[iter,:])
        #     confint1 = 0.68                           #confidence interval
        #     lowind1 = int(round((1 - confint1) / 2 * len(lsfrdiffsort),2))
        #     btconflimlowc1[iter] = btlsfrdiffsort[lowind1]

        #     confint2 = 0.90                           #confidence interva
        #     lowind2 = int(round((1 - confint2) / 2 * len(lsfrdiffsort),2))
        #     btconflimlowc2[iter] = btlsfrdiffsort[lowind2]

        # print("Core")
        # print("median lower 68% SFRdiff confidence interval",np.median(btconflimlowc1))
        # print("median lower 90% SFRdiff confidence interval",np.median(btconflimlowc2))
                
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        plt.ylabel('$N_{gal}$',fontsize=18)
        plt.legend(loc='upper right')


        if logmassmin>9.7:
            s = '$%.2f < log(M_\star/M_\odot) < %.2f$'%(logmassmin, logmassmax)
            text(0.4,0.8,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        
        #K-S test morph sample and SF sample
        print("##############")
        print("SF compared to morphological sample")
        a,b=ks(lsfrdiff_sfsamp, lsfrdiff_morphsamp)

        print("##############")
        print("morphological sample compared to normal sizes")
        a,b=ks(lsfrdiff_morphsamp, lsfrdiff_normsize)

        print("##############")
        print("morphological sample compared to small sizes")
        a,b=ks(lsfrdiff_morphsamp, lsfrdiff_smallsize)

        print("##############")
        print("normal sizes  compared to small sizes")
        a,b=ks(lsfrdiff_normsize, lsfrdiff_smallsize)

        
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        plt.text(-.2,1,'$N_{gal}$',transform=gca().transAxes,verticalalignment='center',rotation=90,fontsize=24)
        plt.legend(loc='upper right')

        plt.ylabel('$N_{gal}$',fontsize=18)
        plt.xlabel('log(SFR) - log(SFR$_{MS}$)',fontsize=18)
        
        if savefig:
            plt.savefig(figuredir + 'sfrdiff_matchsamp.pdf')
        else:
            plt.show()

    def r24_re_msdist(self,savefig=False,btcutflag=True):

        '''plot r24/re vs. distance from the main sequence
        '''
        
        #matchflag, matchind = self.matchsamp_mass(btcutflag=btcutflag)
                
        logmassmin = 9.7
        logmassmax = 10.9
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(14,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[-1.3,1.0,0.0, 1.6]

        #determine offsets w.r.t. main sequence
        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()
        
        sfrpred = 10**(param[0] * self.logstellarmass + param[1])
        lsfrdiff  = log10(self.SFR_USE) - log10(sfrpred)

        if btcutflag:
             cflag =  (self.membflag & self.sampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax) & (self.gim2d.B_T_r < btcut)
             eflag =  (~self.membflag & self.sampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax) & (self.gim2d.B_T_r < btcut)
        else:
             cflag =  (self.membflag & self.sampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)
             eflag =  (~self.membflag & self.sampleflag) & (self.logstellarmass > logmassmin) & (self.logstellarmass < logmassmax)

        ################################################
        ##core galaxies
        plt.subplot(1,2,1)
        ax=plt.gca()

        plt.plot(lsfrdiff[cflag],self.sizeratio[cflag],'ko')

        plt.text(-1,1.45,'Core',fontsize=24,color='r')
        plt.ylabel('$R_{24}/R_d $')
        plt.xlabel('log(SFR) - log(SFR$_{MS}$)',fontsize=18)
        plt.axis(limits)

        #######################
        #external galaxies
        plt.subplot(1,2,2)
        ax=plt.gca()

        plt.plot(lsfrdiff[eflag],self.sizeratio[eflag],'ko')

        plt.text(-1,1.45,'External',fontsize=24,color='r')
        plt.xlabel('log(SFR) - log(SFR$_{MS}$)',fontsize=18)
        ax.set_yticklabels(([]))
        
        plt.axis(limits)
        bothax.append(ax)
        
        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            text(-0.15,1.1,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        if savefig:
            plt.savefig(figuredir + 'r24_re_msdist.pdf')
        else:
            plt.show()
        
            
    def matchsamp_mass(self,savefig=False,btcutflag=True):
        '''This create massmatched samples from the external population
        for every galaxy in the core population.  It will compute how
        each galaxy in the core deviates from the median of the
        mass-matched sample along various axes.  It will plot these
        deviations against each other.

        '''
        #what is the B/T cut
        btcut = 0.3

        if btcutflag:
            cind = np.where((self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut))
            eind = np.where((~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut))
            cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
            eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)

        else:
            cind = np.where(self.membflag & self.sampleflag)
            eind = np.where(~self.membflag & self.sampleflag)
            cflag = (self.membflag & self.sampleflag)
            eflag = (~self.membflag & self.sampleflag)

        #This is the mass interval around which each mass matched
        #sample should be consrtucted
        dlMstarsel = 0.3           #dex

        #initialize the differences of each mass-matched sample with
        #respect to the core galaxy
        self.diffoptsize = np.zeros(len(cind[0]))     #r-band size
        self.diffoptdisksize = np.zeros(len(cind[0]))     #r-band size
        self.diffSFR = np.zeros(len(cind[0]))         #absolute SFR
        self.diffSFRdense = np.zeros(len(cind[0]))     #SFR surface density
        self.diffsizeratio = np.zeros(len(cind[0]))      #R24/Rd
        self.diffmstardense = np.zeros(len(cind[0]))    #stellar mass surface density
        self.diffmstar = np.zeros(len(cind[0]))

        #stellar mass surface density
        self.optsize = self.s.SERSIC_TH50 * self.DA
        self.mstardense = 0.5 * 10**(self.logstellarmass) / (np.pi * self.optsize**2)

        #SFR surface density
        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)
        
        #loop through all cluster members
        jcore=0           #the index of the differences, sequential for every core galaxy
        for i in cind[0]:     #index into the self.<value> array for each core galaxy

            #construct mass matched sample of external galaxies within
            #dMstar.  Exclude the galaxy itself
            dlMstar = self.logstellarmass[i] - self.logstellarmass
            mmatchflag = (eflag) & (abs(dlMstar) < dlMstarsel) & (self.s.NSAID[i] != self.s.NSAID)

            #compute the difference between the mass of each core
            #galaxy and all external galaxies.

            #test if there are any galaxies in mass-matched sample.
            if mmatchflag.any():
                self.diffmstar[jcore] = self.logstellarmass[i] - np.median(self.logstellarmass[mmatchflag])
                self.diffoptdisksize[jcore] = log10(self.optdisksize[i]) - log10(np.median(self.optdisksize[mmatchflag]))
                self.diffoptsize[jcore] = log10(self.optsize[i]) - log10(np.median(self.optsize[mmatchflag]))
                self.diffSFR[jcore] = log10(self.SFR_USE[i]) - log10(np.median(self.SFR_USE[mmatchflag]))
                self.diffSFRdense[jcore] = log10(self.sfrdense[i]) - log10(np.median(self.sfrdense[mmatchflag]))
                self.diffsizeratio[jcore] = log10(self.sizeratio[i]) - log10(np.median(self.sizeratio[mmatchflag]))
                self.diffmstardense[jcore] = log10(self.mstardense[i]) - log10(np.median(self.mstardense[mmatchflag]))

            if np.isnan(self.diffsizeratio[jcore]):
                print("*****NaN Detection******")
                print(self.sizeratio[i],np.median(self.sizeratio[mmatchflag]),self.sizeratio[mmatchflag],mmatchflag)
                print("***********")
            jcore += 1

            
        figure(figsize=(10,8))
        #subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        subplots_adjust(left=.12,bottom=.15,wspace=.3,hspace=.3)
        bothax=[]
        limits=[-1.,1.,-2.,2.]

        plt.subplot(2,2,1)
        ax=plt.gca()

        plt.plot(self.diffsizeratio,self.diffSFR,'ko')


        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        #text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        #plt.title('$SF \ Galaxies$',fontsize=22)
        plt.xlabel(r'$ \Delta log(R_{24}/R_d )$')
        #plt.ylabel('$ \Delta log(\Sigma_{SFR})$')
        plt.ylabel('$ \Delta log(SFR)$')


        plt.subplot(2,2,2)
        ax=plt.gca()

        plt.plot(self.diffsizeratio,self.diffSFRdense,'ko')
        #print(self.diffsizeratio)
        #print(self.diffSFRdense)

        #fit the relation
        p = np.polyfit(self.diffsizeratio,self.diffSFRdense, 1.)
        print("*********fit parameters",p)
        xmod=arange(-3,3,0.1)
        ymod=p[0] * xmod + p[1]
        plt.plot(xmod,ymod,'b-',lw=3)
        
        #plot line showing intrinsic correlation
        x = arange(0.001,20.,0.01)
        y = -2 * log10(x) 
        plt.plot(log10(x),y,'r--',lw=3)

        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        #text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        #plt.title('$SF \ Galaxies$',fontsize=22)
        plt.xlabel(r'$ \Delta log(R_{24}/R_d )$')
        plt.ylabel('$ \Delta log(\Sigma_{SFR})$')
        #plt.ylabel('$ \Delta log(SFR)$')

        if savefig:
            plt.savefig(figuredir + 'matchsamp_mass.pdf')
        else:
            plt.show()


        return cflag

    def matchsamp_masssize(self,btcutflag=True):
        '''This create mass and size-matched samples from the external population
        for every galaxy in the core population.  It will compute how
        each galaxy in the core deviates from the median of the
        mass-matched sample along various axes.  

        '''
        #what is the B/T cut
        btcut = 0.3

        if btcutflag:
            cind = np.where((self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut))
            eind = np.where((~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut))
            cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
            eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)

            #cind = np.where((self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass < 10.0))
            #eind = np.where((~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass < 10.0))
            #cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)  & (self.logstellarmass < 10.0)
            #eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)  & (self.logstellarmass < 10.0)

        else:
            cind = np.where(self.membflag & self.sampleflag)
            eind = np.where(~self.membflag & self.sampleflag)
            cflag = (self.membflag & self.sampleflag)
            eflag = (~self.membflag & self.sampleflag)

        #This is the mass interval around which each mass matched
        #sample should be consrtucted
        dlMstarsel = 0.3           #dex
        #size interval for each mass-matched sample
        drdsel = 0.5           #kpc

        #initialize the differences of each mass-matched sample with
        #respect to the core galaxy
        self.diffsamplmstar = self.logstellarmass[cind[0]]     #masses of cluster sample
        self.diffsampSFR = self.SFR_USE[cind[0]]     #SFRs of cluster sample
        self.difflsersicn = np.zeros(len(cind[0]))     #MIPS sersic index
        self.diffoptsize = np.zeros(len(cind[0]))     #r-band size
        self.difflmipssize = np.zeros(len(cind[0]))     #MIPS size
        self.diffoptdisksize = np.zeros(len(cind[0]))     #r-band disk size
        self.diffSFR = np.zeros(len(cind[0]))         #absolute SFR
        self.diffSFRdense = np.zeros(len(cind[0]))     #SFR surface density
        self.diffsizeratio = np.zeros(len(cind[0]))      #R24/Rd
        self.diffmstardense = np.zeros(len(cind[0]))    #stellar mass surface density
        self.diffmstar = np.zeros(len(cind[0]))
        self.diffID = np.zeros(len(cind[0]))

        #stellar mass surface density
        self.optsize = self.s.SERSIC_TH50 * self.DA
        self.mstardense = 0.5 * 10**(self.logstellarmass) / (np.pi * self.optsize**2)

        #SFR surface density
        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)
        
        #loop through all cluster members
        jcore=0           #the index of the differences, sequential for every core galaxy
        for i in cind[0]:     #index into the self.<value> array for each core galaxy

            #construct mass matched sample of external galaxies within
            #dMstar.  Exclude the galaxy itself
            dlMstar = self.logstellarmass[i] - self.logstellarmass
            drd = self.optdisksize[i] - self.optdisksize
            mmatchflag = (eflag) & (abs(dlMstar) < dlMstarsel) & (self.s.NSAID[i] != self.s.NSAID) & (abs(drd) < drdsel)

            #compute the difference between the mass of each core
            #galaxy and all external galaxies.

            #test if there are any galaxies in mass-matched sample.
            #print(cind[0])
            #print('number in matched sample for gal', i, 'is', np.sum(mmatchflag))
            if mmatchflag.any():
                self.difflsersicn[jcore] = log10(self.s.fcnsersic1[i]) - np.median(log10(self.s.fcnsersic1[mmatchflag]))
                self.difflmipssize[jcore] = log10(self.mipssize[i]) - np.median(log10(self.mipssize[mmatchflag]))
                self.diffmstar[jcore] = self.logstellarmass[i] - np.median(self.logstellarmass[mmatchflag])
                self.diffoptdisksize[jcore] = log10(self.optdisksize[i]) - log10(np.median(self.optdisksize[mmatchflag]))
                self.diffoptsize[jcore] = log10(self.optsize[i]) - log10(np.median(self.optsize[mmatchflag]))
                self.diffSFR[jcore] = log10(self.SFR_USE[i]) - log10(np.median(self.SFR_USE[mmatchflag]))
                self.diffSFRdense[jcore] = log10(self.sfrdense[i]) - log10(np.median(self.sfrdense[mmatchflag]))
                self.diffsizeratio[jcore] = log10(self.sizeratio[i]) - log10(np.median(self.sizeratio[mmatchflag]))
                self.diffmstardense[jcore] = log10(self.mstardense[i]) - log10(np.median(self.mstardense[mmatchflag]))
                self.diffID[jcore] = self.s.NSAID[i]
                #print(sum(mmatchflag))

            if np.isnan(self.diffsizeratio[jcore]):
                print("*****NaN Detection******")
                print(self.sizeratio[i],np.median(self.sizeratio[mmatchflag]),self.sizeratio[mmatchflag],mmatchflag)
                print("***********")
            jcore += 1

        #print(self.difflmipssize)
        return cflag,cind
        
    def plot_n24diff_mipssizediff(self, savefig=False, btcutflag=True):
        '''Create matched samples in mass and optical size and then plot how
sersic indices, 24um sizes, SFRs, and SFR surface densities compare among these matched samples.

        '''

        btcut = 0.3

        cflag,cind = self.matchsamp_masssize(btcutflag=btcutflag)

        figure(figsize=(15,12))
        #subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        subplots_adjust(left=.12,bottom=.15,wspace=.3,hspace=.9)
        bothax=[]
        limits=[-1.,1.,-2.,2.]

        plt.subplot(2,2,1)
        ax=plt.gca()

        #plt.plot(self.difflmipssize,self.diffsersicn,'ko')
        plt.scatter(self.difflmipssize,self.difflsersicn,c=self.diffSFR,vmin=-1.0,vmax=1.0,cmap='inferno',s=60)
        c=colorbar(ax=ax,fraction=.05,ticks=arange(-1.0, 1.0,.2),format='%.1f')
        #plt.colorbar()
        c.ax.text(3.2,.5,'$\Delta log(SFR)$',rotation=-90,verticalalignment='center',fontsize=20)

        (rho,p) = st.spearmanr(self.difflmipssize,self.difflsersicn)
        print("for the left panel")
        print("rho = ",rho)
        print("p = ",p)

        print("Galaxies with low  R24, high n, and low Sigma SFR, all relative")
        istrange = np.where((self.difflsersicn>0.45) & (self.diffSFR<-0.2))
        print(self.diffID[istrange],self.difflsersicn[istrange],self.diffSFR[istrange],)

        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        #text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        #plt.title('$SF \ Galaxies$',fontsize=22)
        plt.xlabel(r'$ \Delta log(R_{24})$')
        #plt.ylabel('$ \Delta log(\Sigma_{SFR})$')
        plt.ylabel('$ \Delta log(n_{24})$')

        if btcutflag:
            s2 = '$B/T \ <  \  %.2f$'%(btcut)
            text(0.9,1.18,s2,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        
#    def test():
        plt.subplot(2,2,2)
        ax=plt.gca()

        plt.plot(self.difflmipssize,self.diffSFRdense,'ko')
        #print(self.diffsizeratio)
        #print(self.diffSFRdense)

        x=np.array([-1.0,1.0])
        y=np.array([np.median(self.diffSFRdense),np.median(self.diffSFRdense)])
        plt.plot(x,y,'g-',lw=3)
        
        y=np.array([-2.0,2.0])
        x=np.array([np.median(self.difflmipssize),np.median(self.difflmipssize)])
        plt.plot(x,y,'g-',lw=3)
        
        #fit the relation
        p = np.polyfit(self.difflmipssize,self.diffSFRdense, 1.)
        print("*********fit parameters for the right panel",p)
        xmod=arange(-3,3,0.1)
        ymod=p[0] * xmod + p[1]
        plt.plot(xmod,ymod,'b-',lw=3)
        
        #plot line showing intrinsic correlation
        x = arange(0.001,20.,0.01)
        y = -2 * log10(x) 
        plt.plot(log10(x),y,'r--',lw=3)

        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        #text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        #plt.title('$SF \ Galaxies$',fontsize=22)
        plt.xlabel(r'$ \Delta log(R_{24})$')
        plt.ylabel('$ \Delta log(\Sigma_{SFR})$')
        #plt.ylabel('$ \Delta log(SFR)$')

        if savefig:
            plt.savefig(figuredir + 'matchsamp_masssize.pdf')
        else:
            plt.show()

    def plotSFRStellarmass_matchsamp(self, savefig=False, btcutflag=True):
        '''Plot SFR vs. Mstar color coded by 24um size of each galaxy compared to
its field comparison sample that is matched in stellar mass and size.

        This code is limited to the subsample of galaxies that is
        selected in matchsamp_masssize()

        '''
        
        
        cflag,cind = self.matchsamp_masssize(btcutflag=btcutflag)

        minsizediff=-0.5
        maxsizediff=0.5
        btcut = 0.3

        #set up the figure and the subplots
        figure(figsize=(10,10))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]
        limits=[8.8,11.2,3e-2,15.]

        ##############################
        #mipssize difference
        #core galaxies - individual points
        plt.subplot(2,2,1)
        ax=plt.gca()

        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()

        plt.scatter(self.logstellarmass[cflag],self.SFR_USE[cflag],c=self.difflmipssize,vmin=minsizediff,vmax=maxsizediff,cmap='inferno',s=60)
        
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

        plt.ylabel('$ SFR \ (M_\odot/yr) $')
        
#############################
        #core plot binned points
        plt.subplot(2,2,2)
        ax=plt.gca()
        xmin = 9.7
        xmax = 10.9
        nbin = (xmax - xmin) / 0.2
        #median SFR  in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[cflag],self.SFR_USE[cflag])

        #median size ratio  in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[cflag],self.difflmipssize)
        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt="none",color='k',markersize=16,ecolor='k')
        #plt.scatter(xbin,ybin,c='k',s=300,cmap='inferno',vmin=minsizediff,vmax=maxsizediff,marker='s', edgecolors='k')
        plt.scatter(xbin,ybin,c=sbin,s=300,cmap='inferno',vmin=minsizediff,vmax=maxsizediff,marker='s', edgecolors='k')
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
        #legend(loc='upper left',numpoints=1)

        if btcutflag:
            s = '$B/T \ <  \  %.2f$'%(btcut)
            text(-0.15,1.1,s,transform=ax.transAxes,horizontalalignment='left',fontsize=20)

        #c=colorbar(ax=bothax,fraction=.05,ticks=arange(minsizediff,maxsizediff,.1),format='%.1f')
        c=colorbar(fraction=.05,ticks=arange(minsizediff,maxsizediff,.1),format='%.1f')
        c.ax.text(3.0,.5,'$ \Delta log(R_{24})$',rotation=-90,verticalalignment='center',fontsize=20)

        ########################
        #sersic index difference
        #core galaxies - individual points

        minndiff = -1.2
        maxndiff = 1.0

        plt.subplot(2,2,3)
        #ax=plt.gca()

        #determine MS fit and output fit results
        xmod,ymod,param = g.MSfit()

        plt.scatter(self.logstellarmass[cflag],self.SFR_USE[cflag],c=self.difflsersicn,vmin=minndiff,vmax=maxndiff,cmap='inferno',s=60)
        
        plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        #g.plotelbaz()
        #text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        #plt.title('$SF \ Galaxies$',fontsize=22)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        plt.ylabel('$ SFR \ (M_\odot/yr) $')
        
#############################
        #core plot binned points
        plt.subplot(2,2,4)
        #ax=plt.gca()
        xmin = 9.7
        xmax = 10.9
        nbin = (xmax - xmin) / 0.2
        #median SFR  in bins of mass
        xbin,ybin,ybinerr=g.binitbins(xmin, xmax, nbin ,self.logstellarmass[cflag],self.SFR_USE[cflag])

        #median size ratio  in bins of mass
        xbin,sbin,sbinerr = g.binitbins(xmin, xmax, nbin,self.logstellarmass[cflag],self.difflsersicn)
        #print(xbin,sbin,sbinerr)
        errorbar(xbin,ybin,yerr=ybinerr,fmt="none",color='k',markersize=16,ecolor='k')
        #plt.scatter(xbin,ybin,c='k',s=300,cmap='inferno',vmin=minsizediff,vmax=maxsizediff,marker='s', edgecolors='k')
        plt.scatter(xbin,ybin,c=sbin,s=300,cmap='inferno',vmin=minndiff,vmax=maxndiff,marker='s', edgecolors='k')
        gca().set_yscale('log')
        plt.axis(limits)
        ax=plt.gca()
        bothax.append(ax)
        ax.set_yticklabels(([]))
        #ax.set_xticklabels(([]))
       # plt.title('$Median $',fontsize=22)

        #plot our MS fit
        plt.plot(xmod,ymod,'w-',lw=3)
        plt.plot(xmod,ymod,'b-',lw=2)
        plt.plot(xmod,ymod/5.,'w--',lw=3)
        plt.plot(xmod,ymod/5.,'b--',lw=2)

        #g.plotelbaz()
        #legend(loc='upper left',numpoints=1)

        text(-0.02,-.2,'$log_{10}(M_*/M_\odot)$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)

        #c=colorbar(ax=bothax,fraction=.05,ticks=arange(minsizediff,maxsizediff,.1),format='%.1f')
        c=colorbar(fraction=.05,ticks=arange(minndiff,maxndiff,.1),format='%.1f')
        c.ax.text(3.0,.5,'$ \Delta log(n_{24})$',rotation=-90,verticalalignment='center',fontsize=20)

        if savefig:
            plt.savefig(figuredir + 'sfr_mstar_matchsamp_masssize.pdf')
        else:
            show()

    def phasespace_matchsamp_masssize(self,savefig=False,btcutflag=True):
        '''This create mass and size-matched samples from the external
        population for every galaxy in the core population.  It will
        compute how each galaxy in the core deviates from the median
        of the mass-matched sample along various axes.  It will plot
        galaxies in phase space according to the deviation in R24
        '''
        #what is the B/T cut
        btcut = 0.3

        if btcutflag:
            cind = np.where((self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut))
            eind = np.where((~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut))
            cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)
            eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)

            #cind = np.where((self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass < 10.0))
            #eind = np.where((~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass < 10.0))
            #cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)  & (self.logstellarmass < 10.0)
            #eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut)  & (self.logstellarmass < 10.0)

        else:
            cind = np.where(self.membflag & self.sampleflag)
            eind = np.where(~self.membflag & self.sampleflag)
            cflag = (self.membflag & self.sampleflag)
            eflag = (~self.membflag & self.sampleflag)

        #This is the mass interval around which each mass matched
        #sample should be consrtucted
        dlMstarsel = 0.3           #dex
        #size interval for each mass-matched sample
        drdsel = 0.5           #kpc

        #initialize the differences of each mass-matched sample with
        #respect to the core galaxy
        self.difflsersicn = np.zeros(len(cind[0]))     #MIPS sersic index
        self.diffoptsize = np.zeros(len(cind[0]))     #r-band size
        self.difflmipssize = np.zeros(len(cind[0]))     #MIPS size
        self.diffoptdisksize = np.zeros(len(cind[0]))     #r-band disk size
        self.diffSFR = np.zeros(len(cind[0]))         #absolute SFR
        self.diffSFRdense = np.zeros(len(cind[0]))     #SFR surface density
        self.diffsizeratio = np.zeros(len(cind[0]))      #R24/Rd
        self.diffmstardense = np.zeros(len(cind[0]))    #stellar mass surface density
        self.diffmstar = np.zeros(len(cind[0]))
        self.diffID = np.zeros(len(cind[0]))

        #stellar mass surface density
        self.optsize = self.s.SERSIC_TH50 * self.DA
        self.mstardense = 0.5 * 10**(self.logstellarmass) / (np.pi * self.optsize**2)

        #SFR surface density
        self.sfrdense = 0.5 * self.SFR_USE / (np.pi * self.mipssize**2)
        
        #loop through all cluster members
        jcore=0           #the index of the differences, sequential for every core galaxy
        for i in cind[0]:     #index into the self.<value> array for each core galaxy

            #construct mass matched sample of external galaxies within
            #dMstar.  Exclude the galaxy itself
            dlMstar = self.logstellarmass[i] - self.logstellarmass
            drd = self.optdisksize[i] - self.optdisksize
            mmatchflag = (eflag) & (abs(dlMstar) < dlMstarsel) & (self.s.NSAID[i] != self.s.NSAID) & (abs(drd) < drdsel)

            #compute the difference between the mass of each core
            #galaxy and all external galaxies.

            #test if there are any galaxies in mass-matched sample.
            if mmatchflag.any():
                self.difflsersicn[jcore] = log10(self.s.fcnsersic1[i]) - np.median(log10(self.s.fcnsersic1[mmatchflag]))
                self.difflmipssize[jcore] = log10(self.mipssize[i]) - np.median(log10(self.mipssize[mmatchflag]))
                self.diffmstar[jcore] = self.logstellarmass[i] - np.median(self.logstellarmass[mmatchflag])
                self.diffoptdisksize[jcore] = log10(self.optdisksize[i]) - log10(np.median(self.optdisksize[mmatchflag]))
                self.diffoptsize[jcore] = log10(self.optsize[i]) - log10(np.median(self.optsize[mmatchflag]))
                self.diffSFR[jcore] = self.logSFR_USE[i] - log10(np.median(self.SFR_USE[mmatchflag]))
                self.diffSFRdense[jcore] = log10(self.sfrdense[i]) - log10(np.median(self.sfrdense[mmatchflag]))
                self.diffsizeratio[jcore] = log10(self.sizeratio[i]) - log10(np.median(self.sizeratio[mmatchflag]))
                self.diffmstardense[jcore] = log10(self.mstardense[i]) - log10(np.median(self.mstardense[mmatchflag]))
                self.diffID[jcore] = self.s.NSAID[i]
                print(sum(mmatchflag))

            if np.isnan(self.diffsizeratio[jcore]):
                print("*****NaN Detection******")
                print(self.sizeratio[i],np.median(self.sizeratio[mmatchflag]),self.sizeratio[mmatchflag],mmatchflag)
                print("***********")
            jcore += 1

            
        figure(figsize=(10,8))
        #subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        subplots_adjust(left=.12,bottom=.15,wspace=.3,hspace=.3)
        bothax=[]
        limits=[-1.,1.,-2.,2.]

        plt.subplot(2,2,1)
        ax=plt.gca()

        #plot phase space color-coded by R24 difference
        #g.s.DR_R200
        #g.dv
        plt.scatter(self.s.DR_R200, self.dv,c=self.difflmipssize,vmin=-1.0,vmax=1.0,cmap='jet_r',s=60)
        
        #plt.plot(self.difflmipssize,self.diffsersicn,'ko')
        #plt.scatter(self.difflmipssize,self.difflsersicn,c=self.diffSFR,vmin=-1.0,vmax=1.0,cmap='jet_r',s=60)
        #c=colorbar(ax=bothax,fraction=.05,ticks=arange(-1.0, 1.0,.2),format='%.1f')
        plt.colorbar()
        #c.ax.text(2.2,.5,'$\Delta log(SFR)$',rotation=-90,verticalalignment='center',fontsize=20)

        #(rho,p) = st.spearmanr(self.difflmipssize,self.difflsersicn)
        #print("for the left panel")
        #print("rho = ",rho)
        #print("p = ",p)

        print("Galaxies with low  R24, high n, and low Sigma SFR, all relative")
        istrange = np.where((self.difflsersicn>0.45) & (self.diffSFR<-0.2))
        print(self.diffID[istrange],self.difflsersicn[istrange],self.diffSFR[istrange],)

        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        #text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        #plt.title('$SF \ Galaxies$',fontsize=22)
        plt.xlabel(r'$ \Delta log(R_{24})$')
        #plt.ylabel('$ \Delta log(\Sigma_{SFR})$')
        plt.ylabel('$ \Delta log(n_{24})$')


        plt.subplot(2,2,2)
        ax=plt.gca()

        plt.plot(self.difflmipssize,self.diffSFRdense,'ko')
        #print(self.diffsizeratio)
        #print(self.diffSFRdense)

        x=np.array([-1.0,1.0])
        y=np.array([np.median(self.diffSFRdense),np.median(self.diffSFRdense)])
        plt.plot(x,y,'g-',lw=3)
        
        y=np.array([-2.0,2.0])
        x=np.array([np.median(self.difflmipssize),np.median(self.difflmipssize)])
        plt.plot(x,y,'g-',lw=3)
        
        #fit the relation
        p = np.polyfit(self.difflmipssize,self.diffSFRdense, 1.)
        print("*********fit parameters for the right panel",p)
        xmod=arange(-3,3,0.1)
        ymod=p[0] * xmod + p[1]
        plt.plot(xmod,ymod,'b-',lw=3)
        
        #plot line showing intrinsic correlation
        x = arange(0.001,20.,0.01)
        y = -2 * log10(x) 
        plt.plot(log10(x),y,'r--',lw=3)

        
        #plt.gca().set_yscale('log')
        plt.axis(limits)
        bothax.append(ax)
        #ax.set_xticklabels(([]))
        #text(0.1,0.9,'$Core$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)
        #plt.title('$SF \ Galaxies$',fontsize=22)
        plt.xlabel(r'$ \Delta log(R_{24})$')
        plt.ylabel('$ \Delta log(\Sigma_{SFR})$')
        #plt.ylabel('$ \Delta log(SFR)$')

        if savefig:
            plt.savefig(figuredir + 'matchsamp_masssize.pdf')
        else:
            plt.show()

    def binitbinsbt(self,xmin,xmax,nbin,x,y):#use equally spaced bins
        #compute median values of a quantity for data binned by another quantity
        #also compute 68% confidence limits on bootstrapped value of median

        
        dx=float((xmax-xmin)/(nbin))                    #width of each bin
        xbin=np.arange(xmin,(xmax),dx)+dx/2.    #centers of each bin
        ybin=np.zeros(len(xbin),'d')                      #initialize y-values of each bin
        ybinerr=np.zeros(len(xbin),'d')                  #initialize yerror-values of each bin

        #initialize bootstrap errors on the median
        ybinerrbtlow=np.zeros(len(xbin),'d')         
        ybinerrbthigh=np.zeros(len(xbin),'d')
        
        xbinnumb=np.array(len(x),'d')                  #give each bin a number
        x1=np.compress((x >= xmin) & (x <= xmax),x)
        y1=np.compress((x >= xmin) & (x <= xmax),y) 
        x=x1
        y=y1
        xbinnumb=((x-xmin)*nbin/(xmax-xmin))     #calculate x  bin number for each point 
        j=-1

        #iterate through bin number
        for i in range(len(xbin)):

            #find all data in that bin
            ydata=np.compress(abs(xbinnumb-float(i))<.5,y)
            
            nydata = len(ydata)
            
            #calculate median
            if nydata>0:
                #ybin[i]=np.average(ydata)
                ybin[i] = np.median(ydata)
                ybinerr[i] = np.std(ydata)/np.sqrt(float(nydata))

                #bootstrap ydata to get medians of each bootstrap sample
                niter = 1000
                confint = 0.68                           #confidence interval
                with NumpyRNGContext(1):    #assures reproducibility of monte carlo
                    btmed = bootstrap(ydata,bootnum=niter,bootfunc=np.median)
                    #print("btmed = ", btmed)
                    #print("standard deviation = ",np.std(btmed))

                #compute confidence intervals of median
                sbtmed = sort(btmed)
                lowind = int(round((1 - confint) / 2 * niter,2))
                highind = int(round((1-((1 - confint) / 2))*niter,2))

                #compute errorbars relative to median
                ybinerrbtlow[i] = ybin[i] - sbtmed[lowind]
                ybinerrbthigh[i] = sbtmed[highind] - ybin[i]
                #print("low BT = ",sbtmed[lowind],"high BT = ",sbtmed[highind]," symm = ", (sbtmed[highind] - sbtmed[lowind])/2.)
                    
            else: 
                ybin[i]=0.
                ybinerr[i]=0.
                ybinerrbtlow[i] = 0.
                ybinerrbthigh[i] = 0.

                
        return xbin,ybin,ybinerr,ybinerrbtlow,ybinerrbthigh
            
    def binitbins(self,xmin,xmax,nbin,x,y):#use equally spaced bins
        #compute median values of a quantity for data binned by another quantity
        
        dx=float((xmax-xmin)/(nbin))                    #width of each bin
        xbin=np.arange(xmin,(xmax),dx)+dx/2.    #centers of each bin
        ybin=np.zeros(len(xbin),'d')                      #initialize y-values of each bin
        ybinerr=np.zeros(len(xbin),'d')                  #initialize yerror-values of each bin

        
        xbinnumb=np.array(len(x),'d')                  #give each bin a number
        x1=np.compress((x >= xmin) & (x <= xmax),x)
        y1=np.compress((x >= xmin) & (x <= xmax),y) 
        x=x1
        y=y1
        xbinnumb=((x-xmin)*nbin/(xmax-xmin))     #calculate x  bin number for each point 
        j=-1

        #iterate through bin number
        for i in range(len(xbin)):

            #find all data in that bin
            ydata=np.compress(abs(xbinnumb-float(i))<.5,y)
            nydata = len(ydata)
            
            #calculate median
            if nydata>0:
                #ybin[i]=np.average(ydata)
                ybin[i] = np.median(ydata)
                ybinerr[i] = np.std(ydata)/np.sqrt(float(nydata))                    
            else: 
                ybin[i]=0.
                ybinerr[i]=0.
                
        return xbin,ybin,ybinerr
            
    def musfr_mustar_ks(self, logmassmin=9.7,logmassmax=10.9,dlogmass=0.4,savefig=False, btcutflag=True):
        #make cumulative histograms of the SFR and Mstar surface
        #densities and perform a K-S and Anderson-Darling test of the
        #distributions in the different environments.
        
        minlsfrdense=-3
        maxlsfrdense=-0.0
        minlmstardense = 7.0
        maxlmstardense = 8.5

        #minlogmass = 8.8
        #maxlogmass = 11.2
        btcut = 0.3

        #the lower and upper mass limits
        #logmassmin = 9.6
        #logmassmax = 10.8
        #dlogmass = 0.3
        #logmassmin = 9.7
        #logmassmax = 10.9
        #dlogmass = 0.4
        nbin = round((logmassmax - logmassmin) / dlogmass)
        print(nbin,"bins of mass")

        #set up the figure and the subplots
        figure(figsize=(10,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        bothax=[]

        #SFR surface density
        self.logsfrdense = log10(0.5 * self.SFR_USE / (np.pi * self.mipssize**2))
        
        #stellar mass surface density using total half-light ratio
        self.optsize = self.s.SERSIC_TH50 * self.DA
        self.logmstardense = log10(0.5 * 10**(self.logstellarmass) / (np.pi * self.optsize**2))

        
        for i in range(0,nbin):
            minlogmass = logmassmin + float(i) * dlogmass
            maxlogmass = logmassmin + float(i+1) * dlogmass
            print("##########", minlogmass, maxlogmass)
            print("log mass range ", minlogmass, maxlogmass)
                       
            
            #SFR surface density
            plt.subplot(nbin,2,1 + i * 2)
            ax=plt.gca()
            
            #select members with B/T<btcut and in our mass range.
            if btcutflag:
                cflag = (self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass > minlogmass) & (self.logstellarmass < maxlogmass)
                eflag = (~self.membflag & self.sampleflag) & (self.gim2d.B_T_r < btcut) & (self.logstellarmass > minlogmass) & (self.logstellarmass < maxlogmass)
            else:
                cflag = (self.membflag & self.sampleflag) & (self.logstellarmass > minlogmass) & (self.logstellarmass < maxlogmass)
                eflag = (~self.membflag & self.sampleflag) & (self.logstellarmass > minlogmass) & (self.logstellarmass < maxlogmass)

            #plot cumulative histogram 
            limits=[minlsfrdense,maxlsfrdense,0.01,1.]
            plt.hist(self.logsfrdense[cflag], cumulative=True,normed=True,color='r',label='$Core$',lw=1.5,alpha=1,histtype='step',bins=len(self.logsfrdense[cflag]))
            plt.hist(self.logsfrdense[eflag], cumulative=True,normed=True,color='b',label='$External$',lw=1.5,alpha=1,histtype='step',bins=len(self.logsfrdense[cflag]))
            
            #K-S test
            print("SFR surface density")
            a,b=ks(self.logsfrdense[cflag],self.logsfrdense[eflag])
        
            #plt.gca().set_yscale('log')
            plt.axis(limits)
            bothax.append(ax)

            mrangestr = "log$(M_\star) = " + str(round(minlogmass,1))  + "-" + str(round(maxlogmass,1)) + "$"
            text(0.5,0.1,mrangestr,transform=ax.transAxes,horizontalalignment='left',fontsize=12)
            if (i==0):
                s1 = '$SF~Galaxies$'
                text(0.9,1.02,s1,transform=ax.transAxes,horizontalalignment='left',fontsize=20)
                text(0.5, 0.2, 'core', transform=ax.transAxes,horizontalalignment='left',fontsize=12, color='r')
                text(0.5, 0.3, 'external', transform=ax.transAxes,horizontalalignment='left',fontsize=12,color='b')
                if btcutflag:
                    s2 = '$B/T \ <  \  %.2f$'%(btcut)
                    text(0.9,1.18,s2,transform=ax.transAxes,horizontalalignment='left',fontsize=20)


            if(i<2):
                ax.set_xticklabels(([]))                

            if(i==2):
                plt.xlabel('$log_{10}(\Sigma_{SFR}/(M_\odot~yr^{-1}~kpc^{-2}))$')

            if(i==1):
                plt.ylabel('$N(<)/N_{tot}$')

            ##############################
            #Stellar Mass surface density
            plt.subplot(nbin,2,2 + i * 2)
            ax=plt.gca()

            #plot cumulative histogram 
            plt.hist(self.logmstardense[cflag], cumulative=True,normed=True,color='r',label='$Core$',lw=1.5,alpha=1,histtype='step',bins=len(self.logmstardense[cflag]))
            plt.hist(self.logmstardense[eflag], cumulative=True,normed=True,color='b',label='$External$',lw=1.5,alpha=1,histtype='step',bins=len(self.logmstardense[cflag]))
            
            #K-S test
            print("Stellar mass surface density")
            a,b=ks(self.logmstardense[cflag],self.logmstardense[eflag])

            limits=[minlmstardense,maxlmstardense,0.01,1.]

            #plt.gca().set_yscale('log')
            plt.axis(limits)
            bothax.append(ax)
            ax.set_yticklabels(([]))
            if(i<2):
                ax.set_xticklabels(([]))                

            if(i==2):
                plt.xlabel('$log_{10}(\Sigma_{\star}/(M_\odot~kpc^{-2}))$')

            #text(0.5,0.9,'$External$',transform=ax.transAxes,horizontalalignment='left',fontsize=20)

            #text(-0.02,-.2,'$R_e(24)/R_e(r)$',transform=ax.transAxes,horizontalalignment='center',fontsize=24)

            #c=colorbar(ax=bothax,fraction=.05,ticks=arange(minlogmass,maxlogmass,.1),format='%.1f')
            #c.ax.text(2.2,.5,'$log_{10}(M_*/M_\odot)$',rotation=-90,verticalalignment='center',fontsize=20)

        if savefig:
            plt.savefig(figuredir + 'musfr_mustar_cumhist.pdf')

    def MSfit(self):
        #fit M-S from own data. Use all galaxies above IR limits that
        #aren't AGN.  Also limit ourselves to things were we are
        #reasonably mass complete and to galaxies with SFR>SFR_Ms(Mstar)/X
        SFR_MSthresh = 0.1
        #fitflag = self.lirflag & ~self.agnflag & (self.logstellarmass > 9.5) & (self.SFR_USE >=  (.08e-9 * SFR_MSthresh)*10**self.logstellarmass)
        fitflag = self.sampleflag & (self.logstellarmass > 9.5) & (self.logstellarmass < 10.7) & ~self.membflag

        #popt, pcov = curve_fit(self.linefunc, self.logstellarmass[fitflag], log10(self.SFR_USE[fitflag]), p0=(1.0,-10.5), bounds=([0.,-12.],[3., -5.])
        p = np.polyfit(self.logstellarmass[fitflag], log10(self.SFR_USE[fitflag]), 1.)
        print("****************fit parameters",p)
        xmod=arange(8.5,12.,0.2)
        ymod=10**(p[0] * xmod + p[1])
        return xmod,ymod,p
    #plt.plot(xmod,ymod,'w-',lw=5)
    #plt.plot(xmod,ymod,'b-',lw=3)

    def sersicint(self, savefig=False):
        #make a plot of how the fractional total luminosity of a
        #sersic profile depends on the truncation radius, given in
        #units of Re

        #use
        #https://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
        #for analytical integral of Sersic function

        #set up figure
        figure(figsize=(5.5,5.5))
        bothax=[]
        ax=gca()
        limits=[0,7.0,0.0,1.0]
        plt.axis(limits)
        bothax.append(ax)
        plt.xlabel('$R_{cut}/R_e$',fontsize=18)
        plt.ylabel('$L(<R_{cut}) / L_{tot}$',fontsize=18)
        #the colors for the lines
        colstr = ['r-','b-','g-','k-']

        #sersic index                                               
        n = [1.0,2.0,3.0,4.0]
        #For purposes of calculation, assume Re is unity.  All radii
        #are therefore in fractions of Re
        re = 1.0
        #assume profile has central surface brightness of unity
        cent_sb = 1.0
        constterm =  2 * np.pi * cent_sb

        #For a range in cutoff radii, integrate the profiel
        #radial range in units of re
        rcutmin = 0.0 * re
        rcutmax = 7.0 * re
        drcut = 0.1 * re
        rcut = arange(rcutmin, rcutmax, drcut)
        dxint = 0.01 * re #dx for integration
        
        #loop through sersic indices
        iserc = 0
        for nserc in n:
            #constant in exponential
            bn = 1.999 * nserc - 0.327

            #analytical expression for total luminosity, ignoring the normalization
            ltot = nserc* scisp.gamma(2*nserc) * re**2 / (bn**(2*nserc)) *  constterm
            #integral of total luminosity out to the maximum radius
            x = arange(rcutmin, rcutmax, dxint)
            y = constterm * exp(-bn * (x/re)**(1/nserc)) * x
            #calculate integral
            ltotint = np.trapz(y,x)
            
            lcut = np.array([]) #the integral of the luminosity for a given cutoff radius

            #loop through cutoff radii
            for rc in rcut:
                #set up x and y arrays for integration
                x = arange(rcutmin, rc, dxint)
                y = constterm * exp(-bn * (x/re)**(1/nserc)) * x
                #calculate integral
                integral = np.trapz(y,x)
                lcut = np.append(lcut, integral)

            lrat = lcut / ltotint # the ratio of the cutoff luminosity to the total luminosity
        
            #now plot results
            plt.plot(rcut,lrat,colstr[iserc],lw=3,label='n='+str(nserc))
            iserc = iserc + 1

        #plt.title("Fractional change in total luminosity for cutoff radius")
        plt.legend(loc='lower right',numpoints=1,scatterpoints=1, markerscale=0.7,fontsize='x-small')

        if savefig:
            plt.savefig(figuredir + 'lum_sersic_rcut.pdf')

            
if __name__ == '__main__':
    homedir = os.environ['HOME']
    g = galaxies(homedir+'/github/LCS/')
    g.setSFR()
    

