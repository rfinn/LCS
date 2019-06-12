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
from astropy.stats import median_absolute_deviation as MAD
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

########################################
##### STATISTICS TO COMPARE CORE VS EXTERNAL
########################################

test_statistics = lambda x: (np.mean(x), np.var(x), MAD(x), st.skew(x), st.kurtosis(x))

###########################
##### START OF GALAXIES CLASS
###########################

class galaxies(lb.galaxies):
    def plotNUV24_vs_sizeratio(self):
        self.NUV24 = self.s.ABSMAG[:,1] - self.s.fcmag1
        plt.figure()
        plt.plot(self.sizeratio[self.sampleflag],self.NUV24[self.sampleflag],'b.')
        plt.xlabel('$R_{24}/R_d$', fontsize=24)
        plt.ylabel('$NUV - 24 $', fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    def plotn24_vs_R24(self):
        plt.figure()
        plt.plot(self.s.fcre1[self.sampleflag],self.s.fcnsersic1[self.sampleflag],'b.')
        plt.xlabel('$R_{24} \ (pixels)$', fontsize=24)
        plt.ylabel('$24\mu m \ Sersic \ Index $', fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        a = plt.axis()
        print(a)
        # add errorbars
        plt.errorbar(self.s.fcre1[self.sampleflag],self.s.fcnsersic1[self.sampleflag],xerr=self.s.fcre1err[self.sampleflag],yerr=self.s.fcnsersic1err[self.sampleflag],fmt='bo',label='all',ecolor='0.5',markersize=5,alpha=0.5)
        plt.plot(self.s.fcre1[self.sampleflag & self.nerrorflag],self.s.fcnsersic1[self.sampleflag & self.nerrorflag],'ro',markerfacecolor='None',markersize=10, label='Numerical Error')
        plt.axis(a)
        plt.legend()
        plt.subplots_adjust(bottom=.15)
        # denote galaxies with numerical

    def compare_SFR(self):
        plt.figure(figsize=(8,6))
        plt.subplots_adjust(hspace=.4,wspace=.3)
        plt.subplot(2,2,1)
        plot(np.log10(self.s.SFR_ZDIST/1.58),self.logSFR_NUV_ZDIST,'b.',alpha=.5)
        self.addSFRlabels('log10(SFR IR ZDIST)','log10(SFR NUV ZDIST)',0)
        
        #plt.figure(figsize=(8,6))
        plt.subplot(2,2,2)
        plot(np.log10(self.s.SFR_ZCLUST/1.58),self.logSFR_NUV_ZCLUST,'b.',alpha=.5)
        self.addSFRlabels('log10(SFR IR ZCLUST)','log10(SFR NUV ZCLUST)',0)
        
        #plt.figure(figsize=(8,6))
        plt.subplot(2,2,3)
        plot(np.log10(self.SFR_BEST),self.logSFR_NUV_BEST,'b.',alpha=.5)
        self.addSFRlabels('log10(SFR IR )','log10(SFR NUV BEST)',0)

        plt.subplot(2,2,4)
        plot(np.log10(self.SFR_BEST),(self.logSFR_NUV),'b.',alpha=.5)
        self.addSFRlabels('log10(SFR IR BEST)','log10(SFR NUV from ABSMAG)',0)
        ## plt.figure(figsize=(8,6))
        ## plot(np.log10(self.SFR_BEST),np.log10(self.s.SFR_ZDIST/1.58),'b.')
        ## plt.ylabel('log10(SFR BEST)')
        ## plt.xlabel('log10(SFR ZDIST)')
        ## x1,x2 = plt.xlim()
        ## xl = np.linspace(x1,x2,100)
        ## plt.plot(xl,xl,'k-')
        ## plt.plot(xl,xl-.3,'k--')
        ## plt.plot(xl,xl+.3,'k--')
        
        plt.figure(figsize=(6,8))
        plt.subplot(3,1,1)
        plt.subplots_adjust(hspace=.4)
        plot(np.log10(self.SFR_BEST),(self.logSFR_NUV),'b.')
        plt.text(-2.5,.2,'all',fontsize=12)
        self.addSFRlabels('log10(SFR IR BEST)','log10(SFR NUV from ABSMAG)',1)

        plt.subplot(3,1,2)
        plot(np.log10(self.SFR_BEST[self.membflag]),(self.logSFR_NUV[self.membflag]),'b.')
        plt.text(-2.5,.2,'core',fontsize=12)
        self.addSFRlabels('log10(SFR IR BEST)','log10(SFR NUV from ABSMAG)',2)

        plt.subplot(3,1,3)
        plot(np.log10(self.SFR_BEST[~self.membflag]),(self.logSFR_NUV[~self.membflag]),'b.')
        plt.text(-2.5,.2,'field',fontsize=12)
        self.addSFRlabels('$log_{10}(SFR\_IR  \ BEST)$','$log_{10}(SFR\_NUV \ from \ ABSMAG)$',3)

    def addSFRlabels(self,xlab,ylab,i):
        if i == 2:
            plt.ylabel(ylab,fontsize=12)
        if i == 3:
            plt.xlabel(xlab,fontsize=12)
        if i == 0:
            plt.ylabel(ylab,fontsize=12)
            plt.xlabel(xlab,fontsize=12)
        x1,x2 = plt.xlim()
        xl = np.linspace(x1,x2,100)
        plt.plot(xl,xl,'k-')
        plt.plot(xl,xl-.3,'k--')
        plt.plot(xl,xl+.3,'k--')
        plt.axis([-3.2,1.5,-3.2,1.5])
        
    def plotNUV24vsMstar(self):
        plt.figure()

        x = self.s.MSTAR_50
        y = self.NUV24


        flag = self.sampleflag & ~self.agnflag

        #plt.figure(figsize=(6,8))
        #plt.subplot(3,1,1)
        #plt.subplots_adjust(hspace=.4)
        #plot(x[self.lirflag],y[self.lirflag],'b.')
        #plt.title('all',fontsize=12)
        xmin,xmax = plt.xlim()
        ymin,ymax = plt.ylim()
        

        #plt.subplot(3,1,2)
        plot(x[self.membflag & flag],y[self.membflag & flag],'r.',alpha=.5)
        #plt.title('core',fontsize=12)
        #plt.axis([xmin,xmax,ymin,ymax])

        #plt.subplot(3,1,3)
        plot(x[~self.membflag & flag],y[~self.membflag & flag],'b.',alpha=.5)
        #plt.title('field',fontsize=12)
        #plt.axis([xmin,xmax,ymin,ymax])
        plt.ylabel('NUV - 24')
        plt.xlabel('$\log_{10}(M_\star/M_\odot)$')
    def plotelbaz(self):
        xe=arange(8.5,11.5,.1)
        xe=10.**xe
        ye=(.08e-9)*xe
        plot(log10(xe),(ye),'k-',lw=1,label='$Elbaz+2011$')
        plot(log10(xe),(2*ye),'k:',lw=1,label='$2 \ SFR_{MS}$')
        # using our own MS fit for field galaxies
        # use stellar mass between 9.5 and 10.5



    def plotSFRStellarmassSize(self,clustername=None,BTcutflag = False):
        figure(figsize=(8,8))
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        x_flags=[ self.irsampleflag & ~self.sampleflag & ~self.agnflag & self.membflag,
                  self.irsampleflag & ~self.sampleflag & ~self.agnflag & self.membflag,
                  self.irsampleflag & ~self.sampleflag & ~self.agnflag & ~self.membflag ,
                  self.irsampleflag & ~self.sampleflag & ~self.agnflag & ~self.membflag
                  ]
                 
        point_flags=[self.sampleflag & self.membflag,
                     self.sampleflag & self.membflag,
                     self.sampleflag & ~self.membflag,
                     self.sampleflag & ~self.membflag
                     ]
        if BTcutflag:
            for i in range(len(point_flags)):
                point_flags[i] = point_flags[i] & (self.s.B_T_r < 0.3)
        bothax=[]
        
        #y=(self.SFR_BEST)*1.58 # convert from salpeter to chabrier IMF according to Salim+07
        #y=10.**(self.logSFR_NUV_BEST)*1.58 # convert from salpeter to chabrier IMF according to Salim+07
        #y=(self.SFR_BEST)*1.58 # convert from salpeter to chabrier IMF according to Salim+07
        self.fitms()
        y=(self.logSFR_NUV_BEST)
        for i in range(len(x_flags)):
            plt.subplot(2,2,i+1)
            if (i == 0) | (i == 2):
                plt.plot(self.logstellarmass[x_flags[i]],y[x_flags[i]],'kx',markersize=8,label='No Fit')
                sp=plt.scatter(self.logstellarmass[point_flags[i]],y[point_flags[i]],c=self.sizeratio[point_flags[i]],vmin=0.3,vmax=1,s=40,label='GALFIT')
            if (i == 1) | (i == 3):
                flag = point_flags[i] & self.massflag
                xbin,ybin,ybinerr=binxycolor(self.logstellarmass[flag],y[flag],self.sizeratio[flag],6, use_median=True)
                plt.scatter(xbin,ybin,c=ybinerr,s=300,vmin=.3,vmax=1,marker='s',edgecolor='k')
            plt.axis([8,11.75,-1.5,1.5])
            plt.plot(self.msx,self.msy,'k-')
            #gca().set_yscale('log')
            a=plt.gca()
            bothax.append(a)
            axvline(x=minmass,c='k',ls='--')
            axhline(y=np.log10(.086/1.58),c='k',ls='--')
            #if i > 2:
            #    xlabel('$log_{10}(M_* (M_\odot)) $',fontsize=22)
            if i == 0:
                #a.set_xticklabels(([]))
                plt.text(0.1,0.9,'$Core$',transform=a.transAxes,horizontalalignment='left',fontsize=20)
                #plt.title('$ SF \ Galaxies $',fontsize=22)
            if i == 2:
                plt.ylabel('$SFR \ (M_\odot/yr)$',fontsize=24)
            if i == 2:
                text(0.1,0.9,'$External$',transform=a.transAxes,horizontalalignment='left',fontsize=20)
            if i ==3:
                text(-0.02,-.2,'$log_{10}(M_*/M_\odot)$',transform=a.transAxes,horizontalalignment='center',fontsize=24)
                
            i += 1
        c=colorbar(ax=bothax,fraction=.05)
        c.ax.text(2.2,.5,'$R_e(24)/R_e(r)$',rotation=-90,verticalalignment='center',fontsize=20)


        savefig(homedir+'research/LocalClusters/SamplePlots/SFRStellarmassSize.png')
        savefig(homedir+'research/LocalClusters/SamplePlots/SFRStellarmassSize.eps')

    def plotmsperpdist(self,allgals = False,normed=True, plotsingle=True):
        if plotsingle:
            plt.figure(figsize=(8,6))
        #plt.subplot(1,2,1)
        if allgals:
            sampleflag = self.sfsampleflag
        else:
            sampleflag = self.sampleflag
        myvar = self.msperdist
        mybins = np.linspace(min(myvar[sampleflag]),max(myvar[sampleflag]),20)
        plt.hist(myvar[self.membflag & sampleflag],bins=mybins,histtype='step',color='red',hatch='//',label='core',lw=2,normed=normed)
        #plt.subplot(1,2,2)
        plt.hist(myvar[~self.membflag & sampleflag],bins = mybins,histtype='step',color='blue',hatch='\\',label='external',lw=2,normed=normed)
        ks(myvar[self.membflag & sampleflag],myvar[~self.membflag & sampleflag])
        plt.legend()
        plt.xlabel('Perpendicular Distance from Main Sequence')
        if normed:
            plt.ylabel('Normalized Frequency')
        else:
            plt.ylabel('Frequency')
        plt.axvline(x=0,color='k')

        # calculate mean, skew, kurtosis
        myvarsub = myvar[self.membflag & sampleflag]
        print('Core : mean={:.3f}, MAD={:.3f}, skew={:.3f}, kurtosis={:.3f}'.format(np.mean(myvarsub),MAD(myvarsub), st.skew(myvarsub),st.kurtosis(myvarsub)))
        myvarsub = myvar[~self.membflag & sampleflag]
        print('External : mean={:.3f}, MAD={:.3f}, skew={:.3f}, kurtosis={:.3f}'.format(np.mean(myvarsub),MAD(myvarsub),st.skew(myvarsub),st.kurtosis(myvarsub)))

    def plotssfrmass(self,plotsingle=True):
        if plotsingle:
            plt.figure(figsize=(8,6))
        self.ssfr = self.logSFR_NUV_BEST - self.logstellarmass
        sample = self.sfsampleflag & (self.logstellarmass > 9.5) & ~self.membflag

        plt.plot(self.logstellarmass[sample], self.ssfr[sample],'bo',alpha=.5)
        
        self.ssfrmstarline = np.polyfit(self.logstellarmass[sample], self.ssfr[sample],1)
        xl = np.linspace(8,11,10)
        yl = np.polyval(self.ssfrmstarline,xl)
        plt.plot(xl,yl,'k-')
        

    def calcstats(self,allgals=True,nboot=100, percentile = 68.):
        if allgals:
            sampleflag = self.sfsampleflag
        else:
            sampleflag = self.sampleflag

        # calc mean, errormean, median_absolute_deviation, skew, kurtosis

        # SFR relative to main sequence
        # perpendicular distance from SF main sequence
        # offset in sSFR relative to best-fit sSFR, as a function of mass

        # errors - bootstrap resampling, calculate mean and 68% confidence interval
        #plt.figure(12,4)
        #plt.subplot(1,3,1)
        # hist of
        
        # keep seed of random generator constant, so numbers are the same each time
        with NumpyRNGContext(1):
            test_variables = [self.msdist, self.msperpdist, self.sSFRdist]
            names = ['MS DISTANCE', 'MS PERPENDICULAR DISTANCE', 'SSFR DISTANCE']
            for i in range(len(test_variables)):

                # core sample
                myvara = test_variables[i][self.membflag & sampleflag]
                test = bootstrap(myvara, bootnum=nboot, bootfunc = test_statistics)
                results = np.zeros((6,test.shape[1]),'f')
                results[1] = np.mean(test, axis=0)
                results[0] = scoreatpercentile(test, (50.-percentile/2.), axis=0)
                results[2] = scoreatpercentile(test, (50.+percentile/2.), axis=0)
    
                # external sample
                myvarb = test_variables[i][~self.membflag & sampleflag]
                test = bootstrap(myvarb, bootnum=nboot, bootfunc = test_statistics)
                results[4] = np.mean(test, axis=0)
                results[3] = scoreatpercentile(test, (50.-percentile/2.), axis=0)
                results[5] = scoreatpercentile(test, (50.+percentile/2.), axis=0)

                # print K-S test
                print('##################################')
                print(names[i])
                print('##################################\n')
                ks(myvara,myvarb)
                # save results
                if i == 0:
                    self.msdist_stats = results
                elif i == 1:
                    self.msperpdist_stats = results
                elif i == 2:
                    self.sSFRdist_stats = results

                print('##################################')
                print(names[i]+' STATS')
                print('##################################\n')
                print
                    
g = galaxies('/Users/rfinn/github/LCS/')

