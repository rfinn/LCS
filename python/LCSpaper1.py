#!/usr/bin/env python

'''
GOAL:
- this code contains all of the code to make figures for paper1


REQUIRED MODULES
- LCSbase.py



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

import argparse# here is min mass = 9.75

###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Run sextractor, scamp, and swarp to determine WCS solution and make mosaics')
parser.add_argument('--minmass', dest = 'minmass', default = 9., help = 'minimum stellar mass for sample.  default is log10(M*) > 7.9')
parser.add_argument('--figdir', dest = 'figdir', default = './', help = 'directory for saving figures.  default is current directory')
parser.add_argument('--diskonly', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')

args = parser.parse_args()

###########################
##### DEFINITIONS
###########################

USE_DISK_ONLY = np.bool(np.float(args.diskonly))#True # set to use disk effective radius to normalize 24um size
if USE_DISK_ONLY:
    print 'normalizing by radius of disk'
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
          'text.fontsize': 20,
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
figuredir = args.figdir
###########################
##### START OF GALAXIES CLASS
###########################

class galaxies(lb.galaxies):
    def plotsizedvdr(self,plotsingle=1,reonly=1,onlycoma=0,plotHI=0,plotbadfits=0,lowmass=0,himass=0,cluster=None,plothexbin=True,hexbinmax=40,scalepoint=0,clustername=None,blueflag=False,plotmembcut=True,colormin=.2,colormax=1,colorbydensity=False,plotoman=False,masscut=None,BTcut=None):
        # log10(chabrier) = log10(Salpeter) - .25 (SFR estimate)
        # log10(chabrier) = log10(diet Salpeter) - 0.1 (Stellar mass estimates)

        if plotsingle:
            plt.figure(figsize=(10,6))
            ax=plt.gca()
            plt.subplots_adjust(left=.1,bottom=.15,top=.9,right=.9)
            plt.ylabel('$ \Delta v/\sigma $',fontsize=26)
            plt.xlabel('$ \Delta R/R_{200}  $',fontsize=26)
            plt.legend(loc='upper left',numpoints=1)

        colors=self.sizeratio
        if colorbydensity:
            colors=np.log10(self.s.SIGMA_5)
            colormin=-1.5
            colormax=1.5
        cbticks=np.arange(colormin,colormax+.1,.1)
        if USE_DISK_ONLY:
            clabel=['$R_{24}/R_d$','$R_{iso}(24)/R_{iso}(r)$']
        else:
            clabel=['$R_e(24)/R_e(r)$','$R_{iso}(24)/R_{iso}(r)$']
        cmaps=['jet_r','jet_r']

        v1=[0.2,0.]
        v2=[1.2,2]
        nplot=1
        
        x=(self.s.DR_R200)
        y=abs(self.dv)
        flag=self.sampleflag #& self.dvflag
        if blueflag:
            flag=self.bluesampleflag & self.dvflag
        if clustername != None:
            flag = flag & (self.s.CLUSTER == clustername)
        if masscut != None:
            flag = flag & (self.logstellarmass < masscut)
        if BTcut != None:
            flag = flag & (self.gim2d.B_T_r < 0.3)
        if cluster != None:
            flag = flag & (self.s.CLUSTER == cluster)
        hexflag=self.dvflag
        if cluster != None:
            hexflag = hexflag & (self.s.CLUSTER == cluster)
        nofitflag = self.sfsampleflag & ~self.sampleflag & self.dvflag
        nofitflag = self.gim2dflag & (self.gim2d.B_T_r < .2) & self.sfsampleflag & ~self.sampleflag & self.dvflag 
        if cluster != None:
            nofitflag = nofitflag & (self.s.CLUSTER == cluster)
        if lowmass:
            flag = flag & (self.s.CLUSTER_LX < 1.)
            hexflag = hexflag & (self.s.CLUSTER_LX < 1.)
            nofitflag = nofitflag & (self.s.CLUSTER_LX < 1.)
        if himass:
            flag = flag & (self.s.CLUSTER_LX > 1.)
            hexflag = hexflag & (self.s.CLUSTER_LX > 1.)
            nofitflag = nofitflag & (self.s.CLUSTER_LX > 1.)
        if onlycoma:
            flag = flag & (self.s.CLUSTER == 'Coma')
        if plothexbin:
            sp=plt.hexbin(x[hexflag],y[hexflag],gridsize=(30,20),alpha=.7,extent=(0,5,0,10),cmap='gray_r',vmin=0,vmax=hexbinmax)
        plt.subplots_adjust(bottom=.15,left=.1,right=.95,top=.95,hspace=.02,wspace=.02)
        if plotmembcut:
            xl=np.array([-.2,1,1])
            yl=np.array([3,3,-0.1])
            plt.plot(xl,yl,'k-',lw=2)
        elif plotoman: # line to identify infall galaxies from Oman+2013
            xl=np.arange(0,2,.1)
            plt.plot(xl,-4./3.*xl+2,'k-',lw=3)
            #plt.plot(xl,-3./1.2*xl+3,'k-',lw=3)       
        else: # cut from Jaffe+2011
            xl=np.array([0.01,1.2])
            yl=np.array([1.5,0])
            plt.plot(xl,yl,'k-',lw=2)

        if reonly:
            nplot=1
        else:
            nplot=2
        if scalepoint:
            size=(self.ssfrms[flag]+2)*40
        else:
            size=60
        for i in range(nplot):
            if not(reonly):
                plt.subplot(1,2,nplot)
            nplot +=1
            if plotbadfits:
                plt.scatter(x[nofitflag],y[nofitflag],marker='x',color='k',s=40,edgecolors='k')#markersize=8,mec='r',mfc='None',label='No Fit')

            ax=plt.gca()
            if colorbydensity:
                sp=plt.scatter(x[flag],y[flag],c=colors[flag],s=size,cmap='jet',vmin=colormin,vmax=colormax,edgecolors=None,lw=0.)
            else:
                sp=plt.scatter(x[flag],y[flag],c=colors[flag],s=size,cmap='jet_r',vmin=colormin,vmax=colormax,edgecolors=None,lw=0.)
            plt.axis([-.1,4.5,-.1,5])
            if masscut != None:
                plt.axis([-.1,4.5,-.1,4])
            if i > 0:
                ax.set_yticklabels(([]))
            ax.tick_params(axis='both', which='major', labelsize=16)
            if plotsingle:
                cb=plt.colorbar(sp,fraction=0.08,label=clabel[i],ticks=cbticks)#cax=axins1,ticks=cbticks[i])
                #text(.95,.9,clabel[i],transform=ax.transAxes,horizontalalignment='right',fontsize=20)
            if plotHI:
                f=flag & self.HIflag
                plt.plot(x[f],y[f],'bs',mfc='None',mec='b',lw=2,markersize=20)

        if not(reonly):
            ax.text(0,-.1,'$ \Delta R/R_{200} $',fontsize=22,transform=ax.transAxes,horizontalalignment='center')
            ax.text(-1.3,.5,'$\Delta v/\sigma_v $',fontsize=22,transform=ax.transAxes,rotation=90,verticalalignment='center')

        if lowmass:
            figname=homedir+'research/LocalClusters/SamplePlots/sizedvdr-lowLx'
        elif himass:
            figname=homedir+'research/LocalClusters/SamplePlots/sizedvdr-hiLx'
        else:
            figname=homedir+'research/LocalClusters/SamplePlots/sizedvdr'
        if plotsingle:
            if masscut != None:
                plt.savefig(figuredir+'sizedvdr-lowmass-lowBT.eps')
            plt.savefig(figuredir+'fig4.pdf')
    def compare_cluster_exterior(self):
        plt.figure(figsize=plotsize_single)
        plt.subplots_adjust(bottom=.15,hspace=.4,top=.95)
        plt.subplot(2,2,1)
        self.compare_single((self.logstellarmass),baseflag=(self.sampleflag & ~self.agnflag),plotsingle=False,xlab='$ log_{10}(M_*/M_\odot) $',plotname='stellarmass')
        plt.legend(loc='upper left')
        plt.xticks(np.arange(9,12,.5))
        plt.xlim(8.9,11.15)
        #xlim(mstarmin,mstarmax)
        plt.subplot(2,2,2)
        self.compare_single(self.gim2d.B_T_r,baseflag=(self.sampleflag & ~self.agnflag),plotsingle=False,xlab='$GIM2D \ B/T $',plotname='BT')
        plt.xticks(np.arange(0,1.1,.2))
        plt.xlim(-.05,.85)
        plt.subplot(2,2,3)
        self.compare_single(self.s.ZDIST,baseflag=(self.sampleflag & ~self.agnflag),plotsingle=False,xlab='$ Redshift $',plotname='zdist')
        plt.xticks(np.arange(0.02,.055,.01))
        plt.xlim(.0146,.045)
        plt.subplot(2,2,4)
        #self.compare_single(self.s.SERSIC_TH50*self.da,baseflag=(self.sampleflag & ~self.agnflag),plotsingle=False,xlab='$R_e(r) \ (kpc)$',plotname='Rer')
        self.compare_single(self.gim2d.Rhlr,baseflag=(self.sampleflag & ~self.agnflag),plotsingle=False,xlab='$R_e(r) \ (kpc)$',plotname='Rer')
        #xticks(arange(2,20,2))
        #plt.xlim(2,20)
        plt.text(-1.5,1,'$Cumulative \ Distribution$',fontsize=22,transform=plt.gca().transAxes,rotation=90,verticalalignment='center')
        #plt.savefig(homedir+'research/LocalClusters/SamplePlots/cluster_exterior.png')
        #plt.savefig(homedir+'research/LocalClusters/SamplePlots/cluster_exterior.eps')
        plt.savefig(figuredir+'fig5.pdf')

    def compare_single(self,var,baseflag=None,plotsingle=True,xlab=None,plotname=None):
        if baseflag == None:
            f1 = self.sampleflag & self.membflag & ~self.agnflag
            f2 = self.sampleflag & ~self.membflag &self.dvflag & ~self.agnflag
        else:
            f1=baseflag & self.sampleflag & self.membflag & ~self.agnflag
            f2=baseflag & self.sampleflag & ~self.membflag  & ~self.agnflag
        xmin=min(var[baseflag])
        xmax=max(var[baseflag])
        #print 'xmin, xmax = ',xmin,xmax
        print 'KS test comparing members and exterior'
        (D,p)=ks(var[f1],var[f2])

        #t=anderson.anderson_ksamp([var[f1],var[f2]])

        #print '%%%%%%%%% ANDERSON  %%%%%%%%%%%'
        #print 'anderson statistic = ',t[0]
        #print 'critical values = ',t[1]
        #print 'p-value = ',t[2]
        if plotsingle:
            plt.figure()#figsize=(12,6))
            plt.title('Member vs. External ('+self.prefix+')')
            subplots_adjust(bottom=.15,left=.15)
            print 'hey'

        plt.xlabel(xlab,fontsize=18)
        #plt.ylabel('$Cumulative \ Distribution $',fontsize=20)
        plt.legend(loc='lower right')

        plt.hist(var[f1],bins=len(var[f1]),cumulative=True,histtype='step',normed=True,label='Core',range=(xmin,xmax),color='k')
            #print var[f2]
        plt.hist(var[f2],bins=len(var[f2]),cumulative=True,histtype='step',normed=True,label='External',range=(xmin,xmax),color='0.5')
        ylim(-.05,1.05)
        ax=gca()
        text(.9,.25,'$D = %4.2f$'%(D),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
        text(.9,.1,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=16)


        return D, p
    def plotRe24vsRe(self,plotsingle=1,sbcutobs=20.,prefix=None,usemyflag=0,myflag=None,showerr=0,logy=True,fixPA=False, usedr=False,colorflag=True):
        #print 'hi'
        if plotsingle:
            plt.figure(figsize=(10,8))
            ax=plt.gca()
            plt.xlabel('$ R_e(r) \ (arcsec)$',fontsize=20)
            plt.ylabel('$ R_e(24) \ (arcsec) $',fontsize=20)
            #legend(loc='upper left',numpoints=1)

        if usemyflag:
            flag=myflag
        else:
            flag=self.sampleflag & (self.sb_obs < sbcutobs)

        mflag=flag & self.membflag
        nfflag = flag & ~self.membflag & self.dvflag
        ffflag = flag & ~self.membflag & ~self.dvflag
        print 'flag = ',sum(mflag),sum(nfflag),sum(ffflag)
        x=(self.gim2d.Rhlr)
        if USE_DISK_ONLY:
            x=self.gim2d.Rd
        if fixPA:
            y=self.s.fcre1*mipspixelscale
            myerr=self.s.fcre1err*mipspixelscale
        else:
            y=self.s.fcre1*mipspixelscale
            myerr=self.s.fcre1err*mipspixelscale
        y=self.s.SUPER_RE1*mipspixelscale*self.DA
        myerr=self.s.SUPER_RE1ERR*mipspixelscale*self.DA
        if plotsingle:
            print 'not printing errorbars'
        else:
            plt.errorbar(x[flag],y[flag],yerr=myerr[flag],fmt=None,ecolor='k')
        mstarmin=9.3
        mstarmax=11

        color=self.logstellarmass
        cblabel='$log_{10}(M_*/M\odot) $'
        v1=mstarmin
        v2=mstarmax
        colormap=cm.jet
        if usedr:
            color=np.log10(sqrt(self.s.DR_R200**2 + self.s.DELTA_V**2))
            cblabel='$\Delta r/R_{200}$'
            cblabel='$log_{10}(\sqrt(\Delta r/R_{200}^2 + \Delta v/\sigma^2)$'
            v1=-.5
            v2=.7
            colormap=cm.jet_r

        if colorflag:
            plotcolors = ['r','b']
        else:
            plotcolors = ['k','0.5']
        plt.plot(x[mflag ],y[mflag],'ko',color=plotcolors[0],markersize=8,mec='k')
        plt.plot(x[nfflag ],y[nfflag],'ks',color=plotcolors[1],markersize=8,mec='k')
        plt.plot(x[ffflag ],y[ffflag],'ks',color=plotcolors[1],markersize=8,mec='k')
        uflag = flag & self.upperlimit
        print 'number of upper limits = ',sum(uflag)
        uplimits=np.array(zip(ones(sum(uflag)), zeros(sum(uflag))))
        plt.errorbar(x[uflag],y[uflag],yerr=uplimits.T, lolims=True, fmt='*',ecolor='k',color='k',markersize=12)
        if plotsingle:
            plt.colorbar(sp)
        self.addlines(logflag=logy)
        ax=plt.gca()
        plt.axis([.5,12,-.5,7.3])

    def addlines(self,logflag=True):
        xl=np.arange(0,100,.5)
        plt.plot(xl,xl,'k-')

        if logflag:
            ax=plt.gca()
            ax.set_yscale('log')
            ax.set_xscale('log')
        plt.axis([1,30.,1,30.])

    
    def plotsizehist(self, btcut = None,colorflag=True):
        figure(figsize=(6,6))
        plt.subplots_adjust(left=.15,bottom=.2,hspace=.1)
        axes=[]
        plt.subplot(2,1,1)

        axes.append(plt.gca())

        mybins=np.arange(0,2,.15)
        if btcut == None:
            flag = self.sampleflag
        else:
            flag = self.sampleflag & (self.gim2d.B_T_r < btcut)
        if colorflag:
            colors = ['r','b']
        else:
            colors = ['k','k']
        flags = [flag & self.membflag & ~self.agnflag,flag & ~self.membflag & ~self.agnflag]
        labels = ['$Core$','$External$']
        for i in range(len(colors)):
            plt.subplot(2,1,i+1)
            print 'median ratio for ',labels[i],' = ',np.median(self.sizeratio[flags[i]])
            hist(self.sizeratio[flags[i]],bins=mybins,histtype='stepfilled',color=colors[i],label=labels[i],lw=1.5,alpha=1)#,normed=True)
            plt.legend(loc='upper right')
            plt.axis([0,2,0,22])
            if i < 1:
                plt.xticks(([]))

        
        plt.text(-.2,1,'$N_{gal}$',transform=gca().transAxes,verticalalignment='center',rotation=90,fontsize=24)
        print 'comparing cluster and exterior SF galaxies'
        a,b=ks(self.sizeratio[flag & self.membflag & ~self.agnflag],self.sizeratio[flag & ~self.membflag & ~self.agnflag])
        print 
        
        plt.xlabel('$ R_{24}/R_d $')
        if btcut == None:
            #plt.ylim(0,20)
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblue.eps')
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblue.png')
            plt.savefig(figuredir+'fig11a.pdf')
            
        else:
            #plt.ylim(0,15)
            plt.subplot(2,1,1)
            plt.title('$ B/T < %2.1f \ Galaxies $'%(btcut),fontsize=20)
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblueBTcut.eps')
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblueBTcut.png')
            plt.savefig(figuredir+'fig11b.pdf')
    def plotsize3panel(self,logyscale=False,use_median=True,equal_pop_bins=True):
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=.12,bottom=.1,top=.9,wspace=.02,hspace=.4)


        nrow=3
        ncol=3        
        flags=[self.sampleflag, self.sampleflag & self.membflag, self.sampleflag & ~self.membflag]
        flags = flags & (self.s.SIGMA_5 > 0.)

        x=[self.gim2d.B_T_r,np.log10(self.s.SIGMA_5),self.logstellarmass]
        xbins = [np.linspace(0,.9,5),np.linspace(-.2,1.5,5),np.linspace(9,10.75,5)]
        xlabels=['$B/T$','$\log_{10}(\Sigma_5 \ (gal/Mpc^2))$','$\log_{10}(M_\star/M_\odot)$']
        colors=[self.logstellarmass,self.gim2d.B_T_r,self.gim2d.B_T_r]
        cblabel=['$\log(M_\star/M_\odot)$','$B/T$','$B/T$']
        cbticks=[np.arange(8.5,10.8,.4),np.arange(0,1,.2),np.arange(0,1,.2)]
        xticklabels=[np.arange(0,1,.2),np.arange(-1.2,2.2,1),np.arange(8.5,11.5,1)]
        xlims=[(-.05,.9),(-1.1,1.9),(8.3,11.2)]
        v1 = [8.5,0,0]
        v2 = [10.8,0.6,0.6]
        y=self.sizeratio
        yerror=self.sizeratioERR
        

        for i in range(len(x)):
            allax=[]
           
            for j in range(3):
                plt.subplot(nrow,ncol,3.*i+j+1)
                plt.errorbar(x[i][flags[j]],y[flags[j]],yerr=yerror[flags[j]],fmt=None,ecolor='.5',markerfacecolor='white',zorder=1,alpha=.5)
                sp=plt.scatter(x[i][flags[j]],y[flags[j]],c=colors[i][flags[j]],vmin=v1[i],vmax=v2[i],cmap='jet',s=40,label='GALFIT',lw=0,alpha=0.7,zorder=1,edgecolors='k')
                if j < 3:
                    (rho,p)=spearman_with_errors(x[i][flags[j]],y[flags[j]],yerror[flags[j]])
                    ax=plt.gca()
                    plt.text(.95,.9,r'$\rho = [%4.2f, %4.2f]$'%(np.percentile(rho,16),np.percentile(rho,84)),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
                    plt.text(.95,.8,'$p = [%5.4f, %5.4f]$'%(np.percentile(p,16),np.percentile(p,84)),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
                a=plt.gca()
                #plt.axis(limits)
                allax.append(a)
                if j > 0:
                    a.set_yticklabels(([]))
                if i == 0:
                    if j == 0:
                        plt.title('$All $',fontsize=24)
                    elif j == 1:
                        plt.title('$Core$',fontsize=24)
                        
                    elif j == 2:
                        plt.title('$External$',fontsize=24)
                if j == 1:
                    plt.xlabel(xlabels[i])
                if j == 0:
                    #plt.ylabel('$R_e(24)/Re(r)$')
                    plt.ylabel('$R_{24}/R_d$')
                xbin,ybin,ybinerr, colorbin = binxycolor(x[i][flags[j]],y[flags[j]],colors[i][flags[j]],nbin=5,bins=xbins[i],erry=True,equal_pop_bins=equal_pop_bins,use_median = use_median)
                plt.scatter(xbin,ybin,c=colorbin,s=180,vmin=v1[i],vmax=v2[i],cmap='jet',zorder=5,lw=2,edgecolors='k')
                plt.errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',alpha=0.7)
                if logyscale:
                    a.set_yscale('log')
                    ylim(.08,6)
                else:
                    ylim(-.1,3.3)
                    yticks((np.arange(0,4,1)))
                xticks(xticklabels[i])
                xlim(xlims[i])
                #ylim(-.1,2.8)
                if j == 2:
                    c = np.polyfit(xbin,ybin,1)
                    print 'xbin = ', xbin
                    print 'ybin = ', ybin
                    #c = np.polyfit(x[i][flags[j]],y[flags[j]],1)
                    xl=np.linspace(min(x[i][flags[j]]),max(x[i][flags[j]]),10)
                    yl = np.polyval(c,xl)
                    plt.plot(xl,yl,'k--',lw=2)
                    plt.subplot(nrow,ncol,3.*i+j)
                    xl=np.linspace(min(x[i][flags[j-1]]),max(x[i][flags[j-1]]),10)
                    yl = np.polyval(c,xl)
                    plt.plot(xl,yl,'k--',lw=2)
                    #print xbin,ybin,colorbin
                

        
            #if i == 2:
            #    #text(0.1,0.9,'$External$',transform=a.transAxes,horizontalalignment='left',fontsize=20)
            #    text(-2.3,1.7,'$R_e(24)/Re(r)$',transform=a.transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=26)


            c=colorbar(ax=allax,fraction=.02,ticks=cbticks[i])
            c.ax.text(6,.5,cblabel[i],rotation=-90,verticalalignment='center',fontsize=20)


        savefig(figuredir+'fig12.pdf')

    def plotsizestellarmass(self,plotsingle=True,btmax=None,btmin=None,equal_pop_bins=True,use_median=True,xbinmin=9.25,xbinmax=10.25,nbins=5):
        if plotsingle:
            plt.figure(figsize=(7,6))
            plt.subplots_adjust(bottom=.15,left=.15)
        flags = [self.sampleflag & self.membflag,self.sampleflag & ~self.membflag]
        if btmax != None:
            flags = flags & (self.gim2d.B_T_r < btmax)
        if btmin != None:
            flags = flags & (self.gim2d.B_T_r > btmin)
        colors = ['r','b']
        mybins = np.linspace(xbinmin,xbinmax,nbins+1)
        for i in range(len(flags)):
            #plot(self.logstellarmass[flags[i]],self.sizeratio[flags[i]],'ro',color=colors[i],alpha=0.5)
            plot(self.logstellarmass[flags[i]],self.sizeratio[flags[i]],'ro',color=colors[i],alpha=0.5)
            errorbar(self.logstellarmass[flags[i]],self.sizeratio[flags[i]],self.sizeratioERR[flags[i]],fmt=None,ecolor='0.5',alpha=0.5)
            flag = flags[i]
            #if btmax != None:
            #    flag = flag & (self.logstellarmass > 9.1) & (self.logstellarmass < 10.5)
            xbin,ybin,ybinerr,colorbin = binxycolor(self.logstellarmass[flag],self.sizeratio[flag],self.gim2d.B_T_r[flag],yweights=self.sizeratioERR[flag],yerr=True,nbin=nbins,equal_pop_bins=equal_pop_bins,use_median=use_median,bins=mybins)
            #print xbin
            plot(xbin[:-1],ybin[:-1],'ro',color=colors[i],markersize=16,mec='k',zorder=5)
            print ybinerr
            #scatter(xbin,ybin,s=200, c=colorbin,marker='^',vmin=0,vmax=0.6,cmap='jet')
            errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',alpha=1,lw=2,capsize=10)
        #colorbar(label='$B/T$')

        xlabel('$ \log_{10}(M_\star /M_\odot) $',fontsize=22)
        ylabel('$ R_{24}/R_d  $',fontsize=22)
        #rho,p=spearman(self.logstellarmass[flag],self.sizeratio[flag])
        #ax=plt.gca()
        #plt.text(.95,.9,r'$\rho = %4.2f$'%(rho),horizontalalignment='right',transform=ax.transAxes,fontsize=18)
        #plt.text(.95,.8,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=18)
        plt.legend(['$Core$','$<Core>$','$External$','$<External>$'],numpoints=1)
        s=''
        if btmax != None:
            s = '$B/T \ <  \  %.2f$'%(btmax)
        if btmin != None:
            s = '$B/T \ >  \  %.2f$'%(btmin)
        if (btmax != None) & (btmin != None):
            s = '$%.2f < B/T \ <  \  %.2f$'%(btmin,btmax)
        plt.title(s,fontsize=20)
        
        plt.axis([8.6,10.9,-.1,2.9])
        plt.savefig(figuredir+'fig13.pdf')
        
    def plotsizeHIfrac(self,sbcutobs=20.5,isoflag=0,r90flag=0,color_BT=False):
        plt.figure(figsize=plotsize_single)
        plt.subplots_adjust(bottom=.2,left=.15)
        plt.clf()
        flag = self.sampleflag & (self.HIflag) #& self.dvflag #& ~self.agnflag
        print 'number of galaxies = ',sum(flag)
        y=(self.sizeratio[flag & self.membflag])
        x=np.log10(self.s.HIMASS[flag & self.membflag])-self.logstellarmass[flag & self.membflag]
        print 'spearman for cluster galaxies only'
        t = spearman(x,y)
        if color_BT:
            pointcolor = self.gim2d.B_T_r
            v1=0
            v2=0.6
        else:
             pointcolor = self.logstellarmass
             v1=mstarmin
             v2=mstarmax
         #color=self.logstellarmass[flag]
        color=pointcolor[flag & self.membflag]
        sp=scatter(x,y,s=90,c=color,vmin=v1,vmax=v2,label='$Core$',cmap='jet',edgecolors='k')

        y=(self.sizeratio[flag & ~self.membflag])
        x=np.log10(self.s.HIMASS[flag & ~self.membflag])-self.logstellarmass[flag & ~self.membflag]
        print 'spearman for exterior galaxies only'
        t = spearman(x,y)

        color=pointcolor[flag & ~self.membflag]
        sp=scatter(x,y,s=90,c=color,vmin=v1,vmax=v2,marker='s',label='$External$',cmap='jet',edgecolor='k')
        y=(self.sizeratio[flag])
        x=np.log10(self.s.HIMASS[flag])-self.logstellarmass[flag]
        plt.legend(loc='upper left',scatterpoints=1)
        errorbar(x,y,self.sizeratioERR[flag],fmt=None,ecolor='.5',zorder=100)
        rho,p=spearman(x,y)

        ax=plt.gca()
        plt.text(.95,.9,r'$\rho = %4.2f$'%(rho),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
        plt.text(.95,.8,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
        print 'spearman for log(M*) < 10.41'
        rho,p=spearman(x[color < 10.41],y[color<10.41])
        cb = plt.colorbar(sp,fraction=.08,ticks=np.arange(8.5,11,.5))
        cb.ax.text(4.,.5,'$\log(M_\star/M_\odot)$',rotation=-90,verticalalignment='center',fontsize=20)
        #plt.ylabel(r'$ R_e(24)/R_e(r)$')
        plt.ylabel('$R_{24}/R_d$')
        plt.xlabel(r'$ \log_{10}(M_{HI}/M_*)$')

        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.axis([-1.8,1.6,0,2.5])
        plt.savefig(figuredir+'fig16a.pdf')
    def plotsizeHIdef(self,sbcutobs=20.5,isoflag=0,r90flag=0):
        figure(figsize=plotsize_single)
        plt.subplots_adjust(left=.15,bottom=.2)
        clf()
        flag = self.sampleflag & (self.HIflag) #& self.membflag #& self.dvflag
        print 'number of galaxies = ',sum(flag)
        y=(self.sizeratio[flag & self.membflag])
        x=(self.s.HIDef[flag & self.membflag])
        print 'spearman for cluster galaxies only'
        t = spearman(x,y)

        #color=self.logstellarmass[flag]
        #color=self.logstellarmass[flag & s.membflag]
        colors=self.logstellarmass
        color=colors[flag & self.membflag]
        sp=scatter(x,y,s=90,c=color,vmin=mstarmin,vmax=mstarmax,label='$Core$',cmap='jet',edgecolor='k')

        y=(self.sizeratio[flag & ~self.membflag])
        x=(self.s.HIDef[flag & ~self.membflag])
        print 'spearman for exterior galaxies only'
        t = spearman(x,y)

        #color=self.logstellarmass[flag]
        color=colors[flag & ~self.membflag]
        sp=scatter(x,y,s=90,c=color,vmin=8.5,vmax=10.8,marker='s',label='$External$',cmap='jet',edgecolor='k')
        y=(self.sizeratio[flag])
        x=(self.s.HIDef[flag])
        plt.legend(loc='upper left',scatterpoints=1)
        errorbar(x,y,self.sizeratioERR[flag],fmt=None,ecolor='.5',zorder=100)
        rho,p=spearman(x,y)
        ax=plt.gca()
        text(.95,.9,r'$\rho = %4.2f$'%(rho),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
        text(.95,.8,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
        print 'spearman for log(M*) < 10.41'
        rho,p=spearman(x[color < 10.41],y[color<10.41])
        cb = plt.colorbar(sp,fraction=.08,ticks=np.arange(8.5,11,.5))
        cb.ax.text(4.,.5,'$\log(M_\star/M_\odot)$',rotation=-90,verticalalignment='center',fontsize=20)

        plt.ylabel('$R_{24}/R_d$')
        plt.xlabel('$HI \ Deficiency$')#,fontsize=26)
        plt.axis([-.6,1.6,0,2.5])
        plt.savefig(figuredir+'fig16b.pdf')
    def plotNUVrsize(self):
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(left=.1,wspace=.01,bottom=.2,right=.9)

        BTmin = 0
        BTmax = 0.4
        flags = [self.sampleflag, self.sampleflag & self.membflag,self.sampleflag & ~self.membflag]
        labels = ['$All$','$Core$','$External$']
        allax=[]
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.scatter(self.sizeratio[flags[i]],self.NUVr[flags[i]],c=self.gim2d.B_T_r[flags[i]],s=60,cmap='jet',vmin=BTmin,vmax=BTmax,edgecolors='k')
            

            if i == 0:
                plt.ylabel('$NUV-r$',fontsize=24)
            else:
                plt.gca().set_yticklabels(([]))
            text(0.98,0.9,labels[i],transform=gca().transAxes,horizontalalignment='right',fontsize=20)
            (rho,p)=spearman_with_errors(self.NUVr[flags[i]],self.sizeratio[flags[i]],self.sizeratioERR[flags[i]])
              
            ax=plt.gca()
    
            plt.text(.05,.1,r'$\rho = [%4.2f, %4.2f]$'%(np.percentile(rho,16),np.percentile(rho,84)),horizontalalignment='left',transform=ax.transAxes,fontsize=12)
            plt.text(.05,.03,'$p = [%5.4f, %5.4f]$'%(np.percentile(p,16),np.percentile(p,84)),horizontalalignment='left',transform=ax.transAxes,fontsize=12)

            plt.axhline(y=4,ls='-',color='0.5')
            plt.axhline(y=4.5,ls='--',color='0.5')
            plt.axhline(y=3.5,ls='--',color='0.5')
            allax.append(plt.gca())
            plt.xticks(np.arange(0,4))
            plt.axis([-0.3,3.1,0,6.2])
        
        colorlabel='$B/T$'
        c=plt.colorbar(ax=allax,fraction=.02,ticks = np.arange(0,.5,.1))
        c.ax.text(3.5,.5,colorlabel,rotation=-90,verticalalignment='center',fontsize=20)
        plt.text(-.51,-.2,'$R_{24}/R_d $',transform=plt.gca().transAxes,fontsize=24,horizontalalignment='center')
        plt.savefig(figuredir+'fig17.pdf')

def plotsizevsMclallwhisker(sbcutobs=20,masscut=None,drcut=1.,blueflag=False,usetemp=False,useM500=False,usesigma=False,bwflag=True,btcut=None):

    plt.figure(figsize=(10,8))
    
    plt.subplots_adjust(hspace=.02,wspace=.02,bottom=.15,left=.15)
    i=0
    x1=[]
    y1=[]
    y2all=[]
    y3all=[]
    for cl in clusternamesbylx:
        
        flag = (g.s.CLUSTER == cl) & g.sampleflag  & g.membflag & ~g.agnflag
        if btcut != None:
            flag = flag & (g.gim2d.B_T_r < btcut)#& ~s.blueflag
        print 'number in ',cl,' = ',sum(flag)
        if masscut != None:
            flag=flag & (g.logstellarmass < masscut)
        if usetemp:
            x=float(clusterTx[cl])
        elif useM500:
            x=clusterXray[cl][1] # M500
        elif usesigma:
            x=log10(clustersigma[cl])
        else:
            x=log10(clusterLx[cl])+44
        y=(g.sizeratio[flag])

        y2=(g.size_ratio_corr[flag])
        BT=mean(g.gim2d.B_T_r[flag & g.gim2dflag])
        erry=std(g.sizeratioERR[flag])/sum(flag)
        #plot(x,median(y2),'k.',label='_nolegend_')
        if x > -99: #check for temp data, which is negative if not available
            print x, y
            if bwflag:
                plt.plot(x,median(y),'k.',color='k',marker=shapes[i],markersize=18,label=cl)
                bp = plt.boxplot([y],positions=[x],whis=99)
                plt.setp(bp['boxes'], color='black')
                plt.setp(bp['whiskers'], color='black')
                plt.setp(bp['fliers'], color='black', marker='+')
                plt.setp(bp['medians'], color='black')
            else:
                plt.plot(x,median(y),'k.',color=colors[i],marker=shapes[i],markersize=20,label=cl)                
                plt.boxplot([y],positions=[x],whis=99)

            x1.append(x)
            y1.append(median(y))
            y2all.append(median(y2))
            y3all.append(mean(y2))
        #errorbar(x,y,yerr=erry,fmt=None,ecolor=colors[i])
        #plot(x,BT,'b^',markersize=15)
    
        
        i+=1
    plt.legend(loc='upper right',numpoints=1,markerscale=.6)
    flag =  g.sampleflag  & ~g.membflag & ~g.agnflag #& ~s.dvflag
    exteriorvalue=mean(g.sizeratio[flag])
    errexteriorvalue=std(g.sizeratio[flag])/sqrt(1.*sum(flag))
    plt.axhline(y=exteriorvalue,color='0.5',ls='-')
    plt.axhline(y=exteriorvalue+errexteriorvalue,color='0.5',ls='--')
    plt.axhline(y=exteriorvalue-errexteriorvalue,color='0.5',ls='--')

    #print 'size corrected by B/A'
    #spearman(x1,y2all)
    #print y1
    #print y2all
    #print 'size corrected by B/A, mean'
    #spearman(x1,y3all)
    ax=plt.gca()
    #ax.set_xscale('log')

    #xl=arange(41,45,.1)
    #yl=-.3*(xl-43.)+.64
    #plot(xl,yl,'k--')
    if usetemp:
        plt.xlabel('$ T_X  (kev)$',fontsize = 28)
    else:
        plt.xlabel('$ log_{10}(L_X  \ erg \ s^{-1} )$',fontsize = 28)
    plt.ylabel('$R_{24}/R_d$',fontsize = 28)
    if usetemp:
        plt.xticks(np.arange(0,10.,1))
        plt.axis([-.05,10.5,0.,1.2])
        ax.tick_params(axis='both', which='major', labelsize=16)
    elif useM500:
        plt.axis([-.75,5.5,0.,1.2])
        ax.tick_params(axis='both', which='major', labelsize=16)
    elif usesigma:
        #axis([2,3.5,0.,1.2])
        ax.tick_params(axis='both', which='major', labelsize=16)
        #xticks(arange(2,4,.5))#,['','44','45'])
    else:
        plt.axis([42.5,45.5,-.1,2.8])
        plt.xticks(np.arange(43,46),['43','44','45'])
        ax.tick_params(axis='both', which='major', labelsize=16)

    plt.savefig(figuredir+'fig14.pdf')

def plotsigmaLx(bwflag=True):
    plt.figure(figsize=[7,6])

    plt.clf()
    plt.subplots_adjust(left=.16,bottom=.16,right=.95,top=.95,wspace=.3)
    i=0
    x=[]
    y=[]
    errm=[]
    errp=[]
    for cl in clusternamesbylx:
        if bwflag:
            plt.plot(clusterLx[cl],clusterbiweightscale[cl],'ko',marker=shapes[i],markersize=18,mfc=None,label=cl)
        else:
            plt.plot(clusterLx[cl],clusterbiweightscale[cl],'ko',color=colors[i],marker=shapes[i],markersize=16,label=cl)
        errm.append(clusterbiweightscale_errm[cl])
        errp.append(clusterbiweightscale_errp[cl])
        x.append(clusterLx[cl])
        y.append(clusterbiweightscale[cl])

        i += 1
    errm=array(errm)
    errp=array(errp)
    yerror=array(zip(errm, errp),'f')
    #print 'yerror = ',yerror
    errorbar(x,y,yerr=yerror.T,fmt=None,ecolor='k')
    # plot comparison sample
    mah = fits.getdata(homedir+'/github/LCS/tables/Mahdavi2001/systems.fits')
    # correct Lx to convert from H0=50 to H0=71 (divide by 1.96)
    # convert bolometric luminosity to L in 0.1-2.4 kev band, which is what I use in the figure
    # this conversion depends on temperature, ranges from 1.44 - 4.05; using 1.4 as a typical value
    # this also brings coma into agreement
    plt.plot(10.**(mah.logLXbol-44.)/1.96/1.4,10.**mah.logsigma,'k.',c='0.5',alpha=0.5)
        
    plt.gca().set_xscale('log')
    plt.xlabel('$L_X \ (10^{44} \  erg/s)$',fontsize=26)
    plt.ylabel('$\sigma \ (km/s) $',fontsize=26)
    plt.axis([.04,10,300,1100])
    leg=plt.legend(numpoints=1,loc='upper left',scatterpoints=1,markerscale=.6,borderpad=.2,labelspacing=.2,handletextpad=.2,prop={'size':14})
    gca().tick_params(axis='both', which='major', labelsize=16)

    plt.savefig(figuredir+'fig1.eps')

def plotpositionson24(plotsingle=0,plotcolorbar=1,plotnofit=0,useirsb=0):
    plt.figure(figsize=(10,8))
    plt.subplots_adjust(hspace=.02,wspace=.02,left=.12,bottom=.12,right=.85)
    i=1
    allax=[]
    for cl in clusternamesbylx:
        plt.subplot(3,3,i)
        infile=homedir+'/github/LCS/tables/clustertables/'+cl+'_NSAmastertable.fits'
        d=fits.getdata(infile)
        #print cl, i
        ra=g.s.RA-clusterRA[cl]
        dec=g.s.DEC-clusterDec[cl]
        r200=2.02*(clusterbiweightscale[cl])/1000./sqrt(OmegaL+OmegaM*(1.+clusterz[cl])**3)*H0/70. # in Mpc
        r200deg=r200*1000./(cosmo.angular_diameter_distance(clusterbiweightcenter[cl]/3.e5).value*Mpcrad_kpcarcsec)/3600.
        cir=Circle((0,0),radius=r200deg,color='None',ec='k')
        gca().add_patch(cir)

        flag=(g.s.CLUSTER == cl) & g.dvflag
        plt.hexbin(d.RA-clusterRA[cl],d.DEC-clusterDec[cl],cmap=cm.Greys,gridsize=40,vmin=0,vmax=10)
        if plotnofit:
            flag=g.sfsampleflag & ~g.sampleflag & g.dvflag & (g.s.CLUSTER == cl)
            plot(ra[flag],dec[flag],'rv',mec='r',mfc='None')

        flag=g.sampleflag & g.dvflag & (g.s.CLUSTER == cl)
        #print cl, len(ra[flag]),len(dec[flag]),len(s.s.SIZE_RATIO[flag])
        if useirsb:
            color=log10(g.sigma_ir)
            v1=7.6
            v2=10.5
            colormap=cm.jet
        else:
            color=g.s.SIZE_RATIO
            v1=.1
            v2=1
            colormap='jet_r'
        try:
            plt.scatter(ra[flag],dec[flag],s=30,c=color[flag],cmap=colormap,vmin=v1,vmax=v2,edgecolors='k')
        except ValueError:
            plt.scatter(ra[flag],dec[flag],s=30,c='k',cmap=cm.jet_r,vmin=.1,vmax=1,edgecolors='k')   
        


        ax=plt.gca()
        fsize=14
        t=cluster24Box[cl]
        drawbox([t[0]-clusterRA[cl],t[1]-clusterDec[cl],t[2],t[3],t[4]],'g-')
        ax=gca()
        ax.invert_xaxis()
        if plotsingle:
            xlabel('$ \Delta RA \ (deg) $',fontsize=22)
            ylabel('$ \Delta DEC \ (deg) $',fontsize=22)
            legend(numpoints=1,scatterpoints=1)

        cname='$'+cl+'$'
        text(.1,.8,cname,fontsize=18,transform=ax.transAxes,horizontalalignment='left')
        plt.axis([1.8,-1.8,-1.8,1.8])
        plt.xticks(np.arange(-1,2,1))
        plt.yticks(np.arange(-1,2,1))
        allax.append(ax)
        multiplotaxes(i)
        i+=1
    if plotcolorbar:
        c=colorbar(ax=allax,fraction=0.05)
        c.ax.text(2.2,.5,'$R_{24}/R_d$',rotation=-90,verticalalignment='center',fontsize=20)
    plt.text(-.5,-.28,'$\Delta RA \ (deg) $',fontsize=26,horizontalalignment='center',transform=ax.transAxes)
    plt.text(-2.4,1.5,'$\Delta Dec \ $',fontsize=26,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')


    #plt.savefig(homedir+'/research/LocalClusters/SamplePlots/positionson24.eps')
    #plt.savefig(homedir+'/research/LocalClusters/SamplePlots/positionson24.png')
    plt.savefig(figuredir+'fig3.pdf')

def plotRe24vsReall(sbcutobs=20,plotcolorbar=0,fixPA=False,logyflag=False,usedr=False):

    figure(figsize=(10,8))
    subplots_adjust(hspace=.02,wspace=.02,left=.15,bottom=.15,right=.9,top=.9)
    i=1
    allax=[]
    for cl in clusternamesbylx:
        plt.subplot(3,3,i)
        flag = (g.s.CLUSTER == cl) & g.sampleflag 
        g.plotRe24vsRe(plotsingle=0,usemyflag=1,myflag=flag,sbcutobs=sbcutobs,logy=logyflag,fixPA=fixPA,usedr=usedr)
        
            
        ax=plt.gca()
        cname='$'+cl+'$'
        plt.text(.9,.8,cname,fontsize=18,transform=ax.transAxes,horizontalalignment='right')
        allax.append(ax)
        multiplotaxes(i)
        i+=1
    if plotcolorbar:
        if usedr:
            cblabel = '$\Delta r/R_{200}$'
            cblabel='$log_{10}(\sqrt{(\Delta r/R_{200})^2 + (\Delta v/\sigma)^2})$'
        else:
            cblabel='$log_{10}(M_*/M\odot) $'

        plt.colorbar(ax=allax,fraction=0.08,label=cblabel)

    plt.text(-.5,-.3,'$R_d \ (kpc)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    plt.text(-2.4,1.5,'$R_{24} \ (kpc) $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'fig10.pdf')

def plotsizevscluster(masscut=None,btcut=None):

    clusters = ['Hercules','A1367','A2052','A2063']
    bigmomma = ['Coma']
    zflag = np.ones(len(g.sampleflag),'bool')
    if masscut != None:
        zflag = zflag & (g.logstellarmass < 10.)
    if btcut != None:
        zflag = zflag & (g.gim2d.B_T_r < btcut)
    
    btcut = .3
    flag = zflag & g.sampleflag  & g.membflag

    groupflag = flag & ((g.s.CLUSTER == 'MKW11') | (g.s.CLUSTER == 'MKW8') | (g.s.CLUSTER == 'AWM4') | (g.s.CLUSTER == 'NGC6107'))
    clusterflag = flag & ((g.s.CLUSTER == 'Hercules') | (g.s.CLUSTER == 'A1367') | (g.s.CLUSTER == 'A2052') | (g.s.CLUSTER == 'A2063'))
    bigmommaflag = flag & (g.s.CLUSTER == 'Coma')

    
    exteriorflag = zflag & g.sampleflag & (g.gim2d.B_T_r < btcut) & ~g.membflag & ~g.dvflag
    nearexteriorflag = zflag & g.sampleflag & (g.gim2d.B_T_r < btcut) & ~g.membflag & g.dvflag


    envs = [exteriorflag, nearexteriorflag,groupflag, clusterflag, bigmommaflag]
    plt.figure()
    
    ypoint = []
    y2 = []
    y2err=[]
    yerr = []
    for i in range(len(envs)):
        ypoint.append(np.median(g.sizeratio[envs[i]]))
        #ypoint.append(ws.weighted_mean(s.sizeratio[envs[i]],weights=1./s.sizeratioERR[envs[i]]))
        yerr.append(np.std(g.sizeratio[envs[i]])/np.sqrt(1.*np.sum(envs[i])))
        y2.append(np.median(g.gim2d.B_T_r[envs[i]]))
        #ypoint.append(ws.weighted_mean(s.sizeratio[envs[i]],weights=1./s.sizeratioERR[envs[i]]))
        y2err.append(np.std(g.gim2d.B_T_r[envs[i]])/np.sqrt(1.*np.sum(envs[i])))
        y=g.sizeratio[envs[i]]
        plt.plot(i,np.median(y),'ko',markersize=10)
        bp = boxplot([y],positions=[i],widths=[.3],whis=99)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='black', marker='+')
        plt.setp(bp['medians'], color='black')


    ax = plt.gca()
    plt.text(.95,.94,'$Far-External: \ \Delta v/\sigma > 3 $',fontsize=14,transform = ax.transAxes,horizontalalignment='right')
    plt.text(.95,.86,'$Near-External: \ \Delta v/\sigma < 3$',fontsize=14,transform = ax.transAxes,horizontalalignment='right')
    plt.text(.95,.78,'$Group: \ \sigma < 700 \ km/s$',fontsize=14,transform = ax.transAxes,horizontalalignment='right')
    plt.text(.95,.70,'$Cluster: \ \sigma > 700 \ km/s$',fontsize=14,transform = ax.transAxes,horizontalalignment='right')
    plt.xticks(np.arange(len(envs)),['$Field$', '$Near-Field$', '$Group$', '$Cluster$', '$Coma$'],fontsize=16)
    plt.xlim(-.3,len(envs)-.7)
    plt.ylim(-.1,2.95)
    #plt.legend()
    plt.ylabel('$R_{24}/R_d$')
    plt.xlabel('$Environment$')
    plt.subplots_adjust(bottom=.2,top=.9,left=.15,right=.92)
    #plt.subplots_adjust(bottom=.15)

    plt.savefig(figuredir+'fig15.pdf')
    
def paperTable1Paper1(sbcutobs=20,masscut=0):
    #clustersigma={'MKW11':361, 'MKW8':325., 'AWM4':500., 'A2063':660., 'A2052':562., 'NGC6107':500., 'Coma':1000., 'A1367':745., 'Hercules':689.}

    #clusterz={'MKW11':.022849,'MKW8':.027,'AWM4':.031755,'A2063':.034937,'A2052':.035491,'NGC6107':.030658,'Coma':.023,'A1367':.028,'Hercules':.037,'MKW10':.02054}
    #clusterbiweightcenter={'MKW11':6897,'MKW8':8045,'AWM4':9636,'A2063':10426,'A2052':10354,'NGC6107':9397,'Coma':7015,'A1367':6507,'Hercules':10941}

    #clusterbiweightcenter_errp={'MKW11':45,'MKW8':36,'AWM4':51,'A2063':63,'A2052':64,'NGC6107':48,'Coma':41,'A1367':48,'Hercules':48}
    
    #clusterbiweightcenter_errm={'MK
    
    #outfile=open(homedir+'/Dropbox/Research/MyPapers/LCSpaper1/Table1.tex','w')
    outfile=open(figuredir+'Table1.tex','w')
    outfile.write('\\begin{deluxetable*}{ccccc} \n')
    outfile.write('\\tablecaption{Cluster Properties and Galaxy Sample Sizes  \label{finalsample}} \n')
    #outfile.write('\\tablehead{\colhead{Cluster} &\colhead{Biweight Central Velocity} & \colhead{Lit.} & \colhead{Biweight Scale} & \colhead{Lit} & \colhead{N$_{spiral}$} & \colhead{N$_{spiral}$} } \n')#  % \\\\ & \colhead{(km/s)}  & \colhead{(km/s)} & \colhead{(km/s)}  & \colhead{(km/s)} & \colhead{Member} & \colhead{External}} \n')
    outfile.write('\\tablehead{\colhead{Cluster} &\colhead{Biweight Central Velocity}  & \colhead{Biweight Scale} & \colhead{N$_{gal}$} & \colhead{N$_{gal}$} \\\\ & \colhead{(km/s)}  & \colhead{(km/s)}  & Core & External } \n')
    outfile.write('\startdata \n')

    for cl in clusternamesbydistance:
        nmemb_spiral = sum((g.s.CLUSTER == cl) & g.sampleflag & g.membflag)
        nnearexterior_spiral = sum((g.s.CLUSTER == cl) & g.sampleflag & ~g.membflag & g.dvflag)
        nexterior_spiral = sum((g.s.CLUSTER == cl) & g.sampleflag & ~g.membflag & ~g.dvflag)
        exterior_spiral = sum((g.s.CLUSTER == cl) & g.sampleflag & ~g.membflag)

        #tableline='%s & %i$^{%+i}_{-%i}$ & %i & %i$^{+%i}_{-%i}$ & %i & %i & %i & %i \\\\ \n' %(cl, clusterbiweightcenter[cl],clusterbiweightcenter_errp[cl],clusterbiweightcenter_errm[cl],int(round(clusterz[cl]*3.e5)), clusterbiweightscale[cl],clusterbiweightscale_errp[cl],clusterbiweightscale_errm[cl],int(round(clustersigma[cl])),nmemb_spiral,nexterior_spiral)
        tableline='%s & %i$^{%+i}_{-%i}$  & %i$^{+%i}_{-%i}$  & %i & %i  \\\\ \n' %(cl, clusterbiweightcenter[cl],clusterbiweightcenter_errp[cl],clusterbiweightcenter_errm[cl], clusterbiweightscale[cl],clusterbiweightscale_errp[cl],clusterbiweightscale_errm[cl],nmemb_spiral,exterior_spiral)
        outfile.write(tableline)
    outfile.write('\enddata \n')
    outfile.write('\end{deluxetable*} \n')
    outfile.close()
def write_out_sizes():
    outfile = open(homedir+'/research/LocalClusters/catalogs/sizes.txt','w')
    size = g.sizeratio[g.sampleflag]
    sizerr = g.sizeratioERR[g.sampleflag]
    myflag = g.membflag[g.sampleflag]
    bt = g.gim2d.B_T_r[g.sampleflag]
    ra = g.s.RA[g.sampleflag]
    dec = g.s.DEC[g.sampleflag]
    outfile.write('#R24/Rd size_err  core_flag   B/T RA DEC \n')
    for i in range(len(size)):
        outfile.write('%6.2f  %6.2f  %i   %.2f %10.9e %10.9e \n'%(size[i],sizerr[i],myflag[i],bt[i],ra[i],dec[i]))
    outfile.close()

if __name__ == '__main__':
    homedir = os.environ['HOME']
    g = galaxies(homedir+'/github/LCS/')
    #plotsigmaLx() # Fig 1
    #plotpositionson24() # Fig 3
    #g.plotsizedvdr(plothexbin=True,plotmembcut=False,plotoman=True,plotbadfits=0,hexbinmax=40,colormin=.2,colormax=1.1) # Fig 4
    #g.compare_cluster_exterior() # Fig 5
    #plotRe24vsReall(logyflag=False) # Fig 10
    #g.plotsizehist() # Fig 11a
    #g.plotsizehist(btcut=.3) # Fig 11b
    #g.plotsize3panel(use_median=False,equal_pop_bins=True) # Fig 12
    #g.plotsizestellarmass(use_median=True,equal_pop_bins=False,btmax=0.3) # Fig 13
    #plotsizevsMclallwhisker(btcut=.3) # Fig 14
    #plotsizevscluster(btcut=.3) # Fig 15

    #g.plotsizeHIfrac() # Fig 16a
    #g.plotsizeHIdef() # Fig 16b
    #g.plotNUVrsize() # Fig 17

    
    

