#!/usr/bin/env python
'''
    Written by Rose A. Finn, probably sometime during the summer of 2013
    
    PURPOSE
      To fit a luminosity function to LCS and EDisCS 24um-detected galaxies.
    
    USEAGE
      From within ipython
        %run ~/svnRepository/pythonCode/LCSfit_lf.py
        l.fitlf(plotsingle=1)
        e.fitlf(plotsingle=1)
        showbothLF2()
        
    PROCEDURE
      The program uses mpfit to solve the least-squares fit a Schechter
      function to the completeness-corrected countsto the data.

    REQUIRED FILES
      LCS
      - LCS_all.fits

      EDisCS
      - ediscs_charyelbaz_lir.fits
    
    REQUIRED MODULES
      Python
        atpy
        numpy
        pylab
        
      Other
        LCScommon.py
        ediscsCommon.py
        fit_lf.py
        mpfit.py
        mpfitexpr.py

    OUTPUT
      lots of plots, that are written to figuredir

    UPDATES
    2015/02/24
    - updating to use LCS_all.fits instead of reading in files from each individual cluster
    - borrowing from code in LCSedicscomp.py
    - probably should just import that program instead of redoing the code here.

'''
import atpy
import numpy as np
import pylab as pl
import os
import fit_lf
import LCSediscscomp # import *
from LCScommon import *
figuredir= '/Users/rfinn/research/LCSEdiscsComp/'
ediscspath='/Users/rfinn/Dropbox/research-macbook/ediscsClusters/'

def getsfrfromlir(lir):
	sfr=lir*bellconv*Lsol
	return sfr

def yaxissfr(ax1):#when plotting lir, but on log scale
	y1, y2=ax1.get_ylim()
	#print y1,y2
	ax2=pl.twinx()
	ax2.set_ylim(getsfrfromlir(y1),getsfrfromlir(y2))
	#print 'y limit in SFR =',getsfrfromlir(y1),getsfrfromlir(y2)
	ax2.set_yscale('log')



def yaxissfrlog(ax1):#when plotting log(lir)
	y1, y2=ax1.get_ylim()
	ax2=pl.twinx()
	ax2.set_ylim(getsfrfromlir(10.**y1),getsfrfromlir(10.**y2))
	#ax2.set_yscale('log')

#def xaxissfrlog(ax1):#when plotting log(lir)
#	y1, y2=ax1.get_xlim()
#	ax2=twiny()
#	ax2.set_xlim(getsfrfromlir(10.**y1),getsfrfromlir(10.**y1))
#	ax2.set_xscale('log')


def xaxissfrlog(ax1,dy=0):
    x1,x2=ax1.get_xlim()
    ax2=pl.twiny()
    #xa=getsfrfromlir(10.**x1)
    #xb=getsfrfromlir(10.**x2)
    #print x1, x2
    #print xa, xb
    ax2.set_xlim(getsfrfromlir(10.**x1),getsfrfromlir(10.**x2))
    ax2.set_xscale('log')

def xaxissfrlog2(ax1,dy=0):#when plotting log(lir)
	locs, labels = pl.xticks()
	ax1=pl.gca()
	ticks=ax1.xaxis.get_major_ticks()
	a=ticks[0]
	xfontsize=a.label1.get_fontsize()
	y1, y2=ax1.get_ylim()
	xmin,xmax,ymin,ymax=ax1.axis()
	ylab=y2+0.03*(y2-y1)+dy
    #print locs
	for x in locs:
	    #if x < xmin: continue
	    #if x > xmax: continue
	    x2=getsfrfromlir(10.**x)
        print x2
        s='$%2.1e$'%(x2)
        pl.text(x,ylab,s,fontsize=xfontsize,horizontalalignment='center')#,transform=ax1.transAxes)





###############################################
##############      CLASSES       #############
###############################################

class lcluster(): #local cluster clusters
    def __init__(self,clustername):
        self.cz=LCScommon.clusterbiweightcenter[clustername]/3.e5
        self.csigma=LCScommon.clusterbiweightscale[clustername]
        self.cra=LCScommon.clusterRA[clustername]
        self.cdec=LCScommon.clusterDec[clustername]
        self.R200Mpc,b=r200(1.*self.csigma,self.cz)
        self.R200deg=b.value
        self.clustername=clustername
        self.membflag=((sqrt((l.lcs.RA-self.cra)**2 + (l.lcs.DEC-self.cdec)**2) < self.R200deg) & (abs(self.cz-l.lcs.ZDIST)*3.e5 < 3.*self.csigma) & (l.lcs.CLUSTER == clustername))
        self.f80MJysr=LCScommon.clusterf80MJysr[clustername]
        self.irflag= self.membflag & (l.lcs.FLUX24 > 0.) & (l.snr24>3) #& (~l.lcs.AGNKAUFF)

    def fitlf(self,plotsingle=1,nbins=6):
        if plotsingle:
            figure()
        comatest(self.logL,self.irweight,sigmaflag=1,sigmaweight=np.ones(len(self.irweight),'f'),nbins=nbins,plotsingle=plotsingle)
        pl.text(0.1,.9,self.prefix,horizontalalignment='left',transform=pl.gca().transAxes)
        pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR}) $',fontsize=20)
        if plotsingle:
            pl.xlabel('$ log_{10}(L_{IR}/L_\odot) $',fontsize=20)
            outfile=figuredir+self.prefix+'_LF_no_sigma_weighting.png'
            pl.savefig(outfile)
            outfile=figuredir+self.prefix+'_LF_no_sigma_weighting.eps'
            pl.savefig(outfile)

class ecluster(): #ediscs clusters#individual EDisCS cluster
    def __init__(self,clustername):
        self.cluster_fullname=ediscsCommon.cluster_fullname[clustername]
        self.membflag=(e.lir.CLUSTER_FULLNAME == self.cluster_fullname)

        self.irflag= self.membflag & (e.lir.FLUX24 > 0.)
        self.sumsfr=sum(e.lir.SFR_ZCLUST[self.irflag]/e.lir.MIPS_WEIGHT[self.irflag]/e.lir.SPEC_WEIGHT[self.irflag])
        self.sumssfr=self.sumsfr/sum(10.**l.logstellarmass[self.irflag]/e.lir.SPEC_WEIGHT[self.irflag])
        self.nmemb=sum(self.membflag/e.lir.SPEC_WEIGHT)
        self.sfr=e.lir.SFR_ZCLUST[self.irflag]
        self.logstellarmass=e.logstellarmass[self.irflag]
        self.cz=mean(e.lir.CLUSTER_Z[self.membflag])
        try:
            self.cra=ediscsCommon.racenter[clustername]
            self.cdec=ediscsCommon.deccenter[clustername]
        except:
            self.cra=mean(e.spec.RA[self.membflag])
            self.cdec=mean(e.spec.DEC[self.membflag])
        try:
            self.csigma=ediscsCommon.sigma[clustername]
        except:
            print 'no velocity dispersion for ',clustername
            print 'setting sigma = 400 km/s'
            self.clsigma=400.
        self.R200Mpc,b=r200(self.csigma,self.cz)
        self.R200deg=b.value

class lcs(LCSediscscomp.lcs): # entire LCS catalog
    def __init__(self):
        LCSediscscomp.lcs.__init__(self)

    def fitlf2(self,plotsingle=1,nbins=6):
        if plotsingle:
            pl.figure()
        flag=self.irflag & self.membflag
        comatest(self.lcs.LIR_ZCLUST[flag],self.lcs.MIPS_WEIGHT[flag],sigmaflag=0,sigmaweight=1.,nbins=nbins,plotsingle=plotsingle)
        pl.text(0.1,.9,self.prefix,horizontalalignment='left',transform=pl.gca().transAxes)
        pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR}) $',fontsize=20)
        if plotsingle:
            pl.xlabel('$ log_{10}(L_{IR}/L_\odot) $',fontsize=20)
            outfile=figuredir+'LCS_LF_no_sigma_weighting.png'
            pl.savefig(outfile)
            outfile=figuredir+'LCS_LF_no_sigma_weighting.eps'
            pl.savefig(outfile)

    def fitlf(self,nbins=8,xmin=8,xmax=10.5,plotsingle=0):
        #pl.close('all')
        if plotsingle:
            pl.figure(figsize=(8,6))
            pl.subplots_adjust(bottom=.2,left=.1,top=.85,right=.9)
            pl.hist(np.log10(self.lcs.LIR_ZCLUST[self.irflag & self.membflag]))

            pl.xlabel('$ log_{10}(L_{IR}) $',fontsize=20)
            pl.ylabel('$ N_{gal} $',fontsize=20)
            pl.title('All Clusters')
            outfile=figuredir+'LCS_LIR_hist.png'
            pl.savefig(outfile)

            outfile=figuredir+'LCS_LIR_hist.eps'
            pl.savefig(outfile)

            pl.figure(figsize=(10,7))
        flag=self.irflag & self.membflag
        comatest(np.log10(self.lcs.LIR_ZCLUST[flag]),self.lcs.MIPS_WEIGHT[flag],sigmaflag=0,sigmaweight=1./(self.lcs.CLUSTER_SIGMA[flag]/1000.)**0,nbins=nbins,plotsingle=plotsingle,xmin=8.,xmax=10.5)

        if plotsingle:
            ax=pl.gca()
            pl.subplots_adjust(bottom=.15,left=.15,top=.85,right=.9)
            pl.text(.5,1.10,'$ SFR \ (M_{\odot}/yr) $',fontsize=20,transform=ax.transAxes,horizontalalignment='center')
            pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR}) $',fontsize=20)
            xaxissfrlog(ax,dy=120)
            pl.legend(numpoints=1,loc='lower left',prop={'size':14})
            outfile=figuredir+'LCS_LF_no_sigma_weighting.png'
            pl.savefig(outfile)
class ediscs(LCSediscscomp.ediscs): # entire EDisCS catalog
    def __init__(self):
        LCSediscscomp.ediscs.__init__(self)

    def fitlf(self,plotsingle=1,nbins=6,xmin=10.5,xmax=11.9,lowzflag=0,hizflag=0):
        if plotsingle:
            pl.figure(figsize=(8,6))
            pl.subplots_adjust(bottom=.15,left=.15,top=.85,right=.9)
            pl.hist(np.log10(self.lir.LIR_ZCLUST[self.irflag & self.lir.CLUSTER_MEMBER]))

            pl.xlabel('$ log_{10}(L_{IR}) $',fontsize=20)
            pl.ylabel('$ N_{gal} $',fontsize=20)
            pl.title('All Clusters')
            outfile=figuredir+'ediscs_LIR_hist.png'
            pl.savefig(outfile)

            outfile=figuredir+'ediscs_LIR_hist.eps'
            pl.savefig(outfile)

        flag=self.irflag & self.lir.CLUSTER_MEMBER
        if lowzflag:
            flag = flag & (self.lir.Z < .6)
        if hizflag:
            flag = flag & (self.lir.Z > .6)
        weight=self.lir.MIPS_WEIGHT*self.lir.SPEC_WEIGHT#*self.lir.GEOM_WEIGHT
        print 'min, max log(L_IR) = ',min(np.log10(self.lir.LIR_ZCLUST[flag])),max(np.log10(self.lir.LIR_ZCLUST[flag]))
        ediscstest(np.log10(self.lir.LIR_ZCLUST[flag]),weight[flag],sigmaflag=1,sigmaweight=np.ones(sum(flag),'f'),nbins=nbins,plotsingle=plotsingle,xmin=xmin,xmax=xmax)

        if plotsingle:
            ax=gca()
            pl.subplots_adjust(bottom=.15,left=.15,top=.85,right=.9)
            #pl.text(0.1,.9,'EDisCS',horizontalalignment='left',transform=ax.transAxes)
            pl.text(.5,1.10,'$ SFR \ (M_{\odot}/yr) $',fontsize=20,transform=ax.transAxes,horizontalalignment='center')
            pl.xlabel('$ log_{10}(L_{IR}/L_\odot) $',fontsize=20)
            pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR}) $',fontsize=20)

            xaxissfrlog(ax,dy=120)
            outfile=figuredir+'ediscs_LF_no_sigma_weighting.png'
            pl.savefig(outfile)
            outfile=figuredir+'ediscs_LF_no_sigma_weighting.eps'
            pl.savefig(outfile)

###############################################
######## FUNCTIONS THAT CALL CLASSES  #########
###############################################

def comatest(logL,irweight,sigmaflag=0,sigmaweight=None,nbins=None,plotsingle=1,xmin=8.5,xmax=11):

    # Bai et al 2009 LF parameters
    alpha = -1.41
    logLstar = 10.7
    phistar = 8.5
    t=fit_lf.fit_schechter(logL,1./irweight,phistar,logLstar,alpha,nbin=nbins,xmin=xmin,xmax=xmax,sigmaflag=sigmaflag,sweight=sigmaweight,plotsingle=plotsingle, prefix='LCS ')
    return t
    

def ediscstest(logL,irweight,sigmaflag=0,sigmaweight=None,nbins=None,plotsingle=1,xmin=10.5,xmax=11.75):

    # Bai et al 2009 alpha parameters
    alpha = -1.41
    logLstar = 11.2
    phistar = 11.
    t=fit_lf.fit_schechter(logL,1./irweight,phistar,logLstar,alpha,nbin=nbins,xmin=xmin,xmax=xmax,sigmaflag=sigmaflag,sweight=sigmaweight,plotsingle=plotsingle,baiflag=0,prefix='EDisCS ')
    return t

def fiteachlf(nbins=6):
    pl.figure(figsize=(15,10))
    i=1
    noyticks=[2,3,5,6,8,9]
    pl.subplots_adjust(hspace=.02,wspace=0.02)
    for cl in mylocalclusters:
        pl.subplot(3,3,i)
        cl.fitlf(plotsingle=0,nbins=nbins)
        multiplotaxes(i)
        i += 1
        pl.axis([7.2,11.2,.5,300])
        pl.legend(prop={'size':8},numpoints=1,loc='upper right')
        pl.xlabel('')
        pl.ylabel('')
    pl.text(-.55,-.2,'$ log_{10}(L_{IR}/L_\odot) $',transform=pl.gca().transAxes,fontsize=22,horizontalalignment='center')
    pl.text(-2.3,1.5,'$ \Phi(L_{IR}/L_\odot)/d log_{10}(L_{IR}) $',transform=pl.gca().transAxes,rotation=90,fontsize=22,verticalalignment='center')
    pl.savefig(homedir+'research/LuminosityFunction/LCS_LIR_subplots.png')
    pl.savefig(homedir+'research/LuminosityFunction/LCS_LIR_subplots.eps')

    



    # scale by area
    #    comatest(ldat.LOG10_LIR,ldat.MIPS_WEIGHT,sigmaflag=1,sigmaweight=1./(ldat.CLUSTER_SIGMA/1000.)**2,nbins=8)
    #pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR})/\sigma_{CL}^2 $',fontsize=20)
    #title('All Clusters')
    #outfile=homedir+'research/LuminosityFunction/LCS_LF_sigma2_weighting.png'
    #pl.savefig(outfile)

    #comatest(ldat.LOG10_LIR,ldat.MIPS_WEIGHT,sigmaflag=1,sigmaweight=1./(ldat.CLUSTER_SIGMA/1000.)**3,nbins=8)
    #pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR})/\sigma_{CL}^3 $',fontsize=20)
    #title('All Clusters')
    #outfile=homedir+'research/LuminosityFunction/LCS_LF_sigma3_weighting.png'
    #pl.savefig(outfile)
    #outfile=homedir+'research/LuminosityFunction/LCS_LF_sigma3_weighting.eps'
    #pl.savefig(outfile)


def edifitlf():
    #pl.close('all')
    ediLmin=10.5
    ediLmax=11.9
    edinbin=6


    figure()
    pl.hist(edi.logL)

    pl.xlabel('$ log_{10}(L_{IR}) $',fontsize=20)
    pl.ylabel('$ N_{gal} $',fontsize=20)
    title('EDisCS Clusters')
    outfile=homedir+'research/LuminosityFunction/EDISCS_LIR_hist.png'
    pl.savefig(outfile)

    outfile=homedir+'research/LuminosityFunction/EDISCS_LIR_hist.eps'
    pl.savefig(outfile)


    editest(edi.logL,edi.irweight,sigmaflag=1,sigmaweight=1./(edi.cluster_sigma/1000.)**0,nbins=edinbin,xmin=ediLmin,xmax=ediLmax)
    pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR}) $',fontsize=20)
    outfile=homedir+'research/LuminosityFunction/EDISCS_LF_no_sigma_weighting.png'
    title('All Clusters')
    pl.savefig(outfile)


    # scale by area
    editest(edi.logL,edi.irweight,sigmaflag=1,sigmaweight=1./(edi.cluster_sigma/1000.)**2,nbins=edinbin,xmin=ediLmin,xmax=ediLmax)
    pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR})/\sigma_{CL}^2 $',fontsize=20)
    title('All Clusters')
    outfile=homedir+'research/LuminosityFunction/EDISCS_LF_sigma2_weighting.png'
    pl.savefig(outfile)

    editest(edi.logL,edi.irweight,sigmaflag=1,sigmaweight=1./(edi.cluster_sigma/1000.)**3,nbins=edinbin,xmin=ediLmin,xmax=ediLmax)
    pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR})/\sigma_{CL}^3 $',fontsize=20)
    title('All Clusters')
    outfile=homedir+'research/LuminosityFunction/EDISCS_LF_sigma3_weighting.png'
    pl.savefig(outfile)
    outfile=homedir+'research/LuminosityFunction/EDISCS_LF_sigma3_weighting.eps'
    pl.savefig(outfile)



def showbothLF(): 

    ediLmin=10.5
    ediLmax=11.9
    edinbin=6

    pl.figure(figsize=(8,6))
    pl.subplots_adjust(bottom=.15,top=.15,right=.05)
    comatest(ldat.LOG10_LIR,ldat.MIPS_WEIGHT,sigmaflag=1,sigmaweight=1./(ldat.CLUSTER_SIGMA/1000.)**0,nbins=8,plotsingle=0)
    editest(edi.logL,edi.irweight,sigmaflag=1,sigmaweight=1./(edi.cluster_sigma/1000.)**0,nbins=edinbin,xmin=ediLmin,xmax=ediLmax,plotsingle=0,lowzflag=1)
    editest(edi.logL,edi.irweight,sigmaflag=1,sigmaweight=1./(edi.cluster_sigma/1000.)**0,nbins=edinbin,xmin=ediLmin,xmax=ediLmax,plotsingle=0,hizflag=1)
    
    pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR}) $',fontsize=22)
    pl.xlabel('$ log_{10}(L_{IR}/L_\odot) $',fontsize=22)

    ax=pl.gca()
    pl.text(.5,1.05,'$ SFR \ (M_{\odot}/yr) $',fontsize=22,transform=ax.transAxes,horizontalalignment='center')
    xaxissfrlog(ax,dy=100)
    pl.legend(numpoints=1,loc='lower left',prop={'size':12})
    outfile=figuredir+'LCS_EDISCS_LF_no_sigma_weighting.png'
    #title('All Clusters')
    pl.savefig(outfile)
    outfile=figuredir+'LCS_EDISCS_LF_no_sigma_weighting.eps'
    #title('All Clusters')
    pl.savefig(outfile)

def showbothLF2(): 

    ediLmin=10.5
    ediLmax=11.9
    ediLminlz=10.
    ediLmaxlz=11.
    edinbin=6

    pl.figure(figsize=(9,6))
    pl.subplots_adjust(bottom=.15,top=.85,left=.15,right=.95)
    l.fitlf(nbins=8,plotsingle=0,xmin=8,xmax=11)
    #editest(edi.logLlz,edi.irweightlz,sigmaflag=1,sigmaweight=1./(edi.cluster_sigmalz/1000.)**0,nbins=edinbin,xmin=ediLminlz,xmax=ediLmaxlz,plotsingle=0)
    #editest(edi.logLhz,edi.irweighthz,sigmaflag=1,sigmaweight=1./(edi.cluster_sigmahz/1000.)**0,nbins=edinbin,xmin=ediLmin,xmax=ediLmax,plotsingle=0)
    #e.fitlf(nbins=edinbin-2,xmin=ediLmin,xmax=ediLmax,plotsingle=0,lowzflag=1)
    #e.fitlf(nbins=edinbin-2,xmin=ediLmin,xmax=ediLmax,plotsingle=0,hizflag=1)
    e.fitlf(nbins=edinbin,xmin=ediLmin,xmax=ediLmax,plotsingle=0)
    pl.ylabel('$ \Phi(L_{IR})/dlog_{10}(L_{IR}) $',fontsize=22)
    pl.xlabel('$ log_{10}(L_{IR}/L_\odot) $',fontsize=22)

    ax=pl.gca()
    pl.text(.5,1.1,'$ SFR \ (M_{\odot}/yr) $',fontsize=22,transform=ax.transAxes,horizontalalignment='center')
    xaxissfrlog(ax,dy=350)
    pl.legend(numpoints=1,loc='lower left',prop={'size':12})
    outfile=figuredir+'LCS_EDISCS_LF_no_sigma_weighting.png'
    #title('All Clusters')
    pl.savefig(outfile)
    outfile=figuredir+'LCS_EDISCS_LF_no_sigma_weighting.eps'
    #title('All Clusters')
    pl.savefig(outfile)

def showbothLIRhist(): 


    figure(figsize=(10,7))
    pl.hist(ldat.LOG10_LIR,label='LCS Raw Counts',histtype='step',hatch='/',color='b')
    pl.hist(edi.logL,label='EDisCS Raw Counts',histtype='step',hatch='\\',color='r')

    pl.xlabel('$ log_{10}(L_{IR}) $',fontsize=20)
    pl.ylabel('$ N_{gal} $',fontsize=20)

    ax=pl.gca()
    pl.text(.5,1.05,'$ SFR \ (M_{\odot}/yr) $',fontsize=22,transform=ax.transAxes,horizontalalignment='center')
    xaxissfrlog(ax,dy=-3)

    pl.legend()
    outfile=homedir+'research/LuminosityFunction/LCS_EDISCS_LIR_hist.png'
    pl.savefig(outfile)

def plotediscszhist():
    figure(figsize=(10,7))
    pl.hist(edi.cluster_redshift,bins=len(edi.cluster_redshift),cumulative=True,histtype='stepfilled')
    pl.xlabel('$ Redshift $',fontsize=20)
    pl.axhline(y=len(edi.cluster_redshift)/2., ls='--',color='k')
    pl.axvline(x=.58, ls='--',color='k')
    pl.axis([.39,.81,0,len(edi.cluster_redshift)+2])
    pl.text(.42,120,'$ average \ {z} = 0.58 $',fontsize=20)
    pl.ylabel('$ {Cumulative \ Distribution} $',fontsize=20)
    pl.title('$ Redshift \ Distribution \ of \ EDisCS \ Sample $',fontsize=24)
    outfile=homedir+'research/LuminosityFunction/EDISCS_z_hist.png'
    pl.savefig(outfile)

        
###############################################
##############      MAIN          #############
###############################################

if __name__ == "__main__":
    l=lcs()
    e=ediscs()
    ## # initiate Local Clusters
    ## mkw11=lcluster('MKW11')
    ## mkw8=lcluster('MKW8')
    ## awm4=lcluster('AWM4')
    ## a2052=lcluster('A2052')
    ## a2063=lcluster('A2063')
    ## ngc6107=lcluster('NGC6107')
    ## coma=lcluster('Coma')
    ## herc=lcluster('Hercules')
    ## a1367=lcluster('A1367')

    ## # initiate EDisCS Clusters
    ## # 28 groups + clusters
    ## # should be 26 according to Milvang-Jensen+2008
    ## # we have data for 24
    ## cl1018=ecluster('cl1018')
    ## cl1037=ecluster('cl1037')
    ## cl1037a=ecluster('cl1037a') # group, according to MJ+2008
    ## cl1040=ecluster('cl1040')
    ## #cl1040a=ecluster('cl1040a') # not a group
    ## #cl1040b=ecluster('cl1040b') # not a group
    ## cl105411=ecluster('cl105411')
    ## #cl105411a=ecluster('cl105411a') # not a group
    ## cl105412=ecluster('cl105412')
    ## #cl105412a=ecluster('cl105412a') # not a group
    ## cl1059=ecluster('cl1059')
    ## cl1103=ecluster('cl1103')
    ## cl1103a=ecluster('cl1103a') # group, according to MJ+2008
    ## cl1103b=ecluster('cl1103b') # group, according to MJ+2008
    ## cl1138=ecluster('cl1138')
    ## cl1138a=ecluster('cl1138a') # group, according to MJ+2008
    ## cl1202=ecluster('cl1202')
    ## cl1216=ecluster('cl1216')
    ## cl1227=ecluster('cl1227')
    ## cl1227a=ecluster('cl1227a') # group, according to MJ+2008
    ## cl1232=ecluster('cl1232')
    ## cl1301=ecluster('cl1301')
    ## cl1301a=ecluster('cl1301a') # group, according to MJ+2008
    ## cl1353=ecluster('cl1353')
    ## cl1354=ecluster('cl1354')
    ## cl1354a=ecluster('cl1354a') # group, according to MJ+2008
    ## cl1411=ecluster('cl1411')
    ## cl1420=ecluster('cl1420')

    ## # store instances of each class in a list so that I can easily loop over all instances when making plots of full samples
    ## mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc6107,coma,herc,a1367]
    ## myediscsclusters=[cl1018,cl1037,cl1037a,cl1040,cl105411,cl105412,cl1059,cl1103,cl1103a,cl1103b,cl1138,cl1138a,cl1202,cl1216,cl1227,cl1227a,cl1232,cl1301,cl1301a,cl1353,cl1354,cl1354a,cl1411,cl1420]





#### old code
## def mergedata():
##     logL_all=[]
##     irweight_all=[]
##     sigma_all=[]
##     for cl in clustersbymass:
##         logL_all = logL_all + cl.logL.tolist()
##         irweight_all = irweight_all + cl.irweight.tolist()
##         sigma_all = sigma_all + cl.sigma.tolist()
##         print len(logL_all),len(irweight_all),len(sigma_all)
##     logL_all=np.array(logL_all,'d')
##     irweight_all=np.array(irweight_all,'d')
##     sigma_all = np.array(sigma_all,'d')
##     print len(logL_all),len(irweight_all),len(sigma_all)
##     ldat=atpy.Table()
##     ldat.add_column('LOG10_LIR',logL_all)
##     ldat.add_column('MIPS_WEIGHT',irweight_all)
##     ldat.add_column('CLUSTER_SIGMA',sigma_all)
##     outfile=homedir+'research/LuminosityFunction/LCS_LIR_all.fits'
##     if os.path.exists(outfile):
##         os.remove(outfile)

##     ldat.write(homedir+'research/LuminosityFunction/LCS_LIR_all.fits')
