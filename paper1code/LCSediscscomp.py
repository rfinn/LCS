#!/usr/bin/env python
'''
    PURPOSE
      To compare the integrated SFR of LCS and EDisCS clusters
      

    CALLING SEQUENCE

    INPUT PARAMETERS

    OUTPUT
    
    PROCEDURE
      EDisCS: uses ediscs spec sample only
      LCS: uses all galaxies that fall on 24um image
      
    REQUIRED PYTHON MODULES
      astropy
      atpy
      pylab
      matplotlib
      
    ADDITIONAL REQUIRED MODULES
      LCScommon.py
      ediscsCommon.py
      
    NOTES


    UPDATES

      2015/02/21: updated by Rose Finn to use John M's spec files
      and my LCS_all.fits table as input rather than indivual mastertables.

'''

from pylab import *
import LCScommon 
import ediscsCommon 
import atpy
from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(H0=LCScommon.H0,Om0=LCScommon.OmegaM)
import mystuff as my
#import LCSReadmasterBaseNSA as lcs
#import ediscsreadspitzercatalogs as edi
from matplotlib import rc
rc('font',family='serif')
#rc('serif',family='Times, Computer Modern Roman')
rc('mathtext',fontset='cm')
rc('mathtext',default='regular')
rc('xtick',labelsize='large')
rc('ytick',labelsize='large')
rc('figure',figsize=[8,6])
#rc('text',usetex=True)
edimarkersize=12
lcsmarkersize=12
masslimit=10.2 #log10(mass-min)
nsigma=3.
#directory where figures will be saved
figuredir= '/Users/rfinn/research/LCSEdiscsComp/'
ediscspath='/Users/rfinn/Dropbox/research-macbook/ediscsClusters/'

# conversion from LIR to SFR used in Chary & Elbaz code
# SFR_Sanders[igal] = 1.7217e-10*LIR_Sanders[igal]
# should use this when converting axis to sfr in LF program

def SFRfromLIR(LIR):
    SFR=1.7217e-10*LIR
    return SFR

def LIRfromSFR(SFR):
    LIR=SFR/1.7217e-10
    return LIR
    
def r200(sigma,z):
    r200Mpc=2.02*(sigma)/1000./sqrt(LCScommon.OmegaL+LCScommon.OmegaM*(1.+z)**3)/(LCScommon.H0/70.) # in Mpc
    #print r200Mpc
    # angular_diameter_distance returns physical scale in Mpc corresponding to 1 radian
    r200Mpc=1.2*r200Mpc # based on LCS dv vs dr plots
    r200deg=r200Mpc/cosmo.angular_diameter_distance(z)*(180./pi)
    return r200Mpc, r200deg
def Mcl(sigma,z):
    Mcl = 1.4e15*(sigma/1000.)**3/sqrt(LCScommon.OmegaL+LCScommon.OmegaM*(1.+z)**3)/(LCScommon.H0/100.)
    return Mcl

def ediscsmultiplotaxes(i):
    keepyticks=[1,5,9,13,17,21]
    keepxticks=[21,22,23,24]
    ax=gca()
    if i not in keepyticks:
        ax.set_yticklabels(([]))
    if i not in keepxticks:
        ax.set_xticklabels(([]))

class lcs(): # entire LCS catalog
    def __init__(self):
        infile='/Users/rfinn/research/LocalClusters/NSAmastertables/LCS_all.fits'
        self.lcs=atpy.Table(infile)
        # log(M*) = -0.68+0.70*(g-i)+logLi (Eqn. 8 in Taylor+11 MNRAS 418, 158).  
        self.logLi=(-1*(self.lcs.ABSMAG[:,5]-LCScommon.SolarMag['i'])/2.5)
        self.logstellarmass=-0.68 +  0.70*(self.lcs.ABSMAG[:,3]-self.lcs.ABSMAG[:,5]) + self.logLi
        self.logstellarmass=1.15+0.70*(self.lcs.ABSMAG[:,3]-self.lcs.ABSMAG[:,5]) -0.4*(self.lcs.ABSMAG[:,5]+ 5.*log10(LCScommon.h))
        self.snr24=abs(self.lcs.FLUX24/self.lcs.FLUX24ERR)
        ur=self.lcs.ABSMAG[:,2]-self.lcs.ABSMAG[:,4]
        xl=arange(8.5,12.5,.1)
        yl=.25*(xl-10)+2.35
        #plot(xl,.25*(xl-10)+2.35-.2,'k--')
        #self.redflag=(ur > 2.3)
        #self.greenflag=(ur > 1.8) & (ur < 2.3)
        #self.blueflag=(ur < (.25*(self.logstellarmass-10.)+2.35-.2))
        self.spiralflag = (self.lcs.p_cs > 0.7)
        self.blueflag=(ur < (.25*(self.logstellarmass-10.)+2.35-.2))
        self.redflag=(ur > (.25*(self.logstellarmass-10.)+2.35-.2))
        self.membflag=(self.lcs.DR_R200 < 1) & (self.lcs.DELTA_V < nsigma)
        self.irflag=  (self.lcs.FLUX24 > 0.) & (log10(self.lcs.LIR_ZDIST) > 8.7) #& (~l.lcs.AGNKAUFF)
        self.agnflag=self.membflag & self.lcs.AGNKAUFF
    def getsffrac(self,logLIR,logM):
        nscaled=self.membflag/self.lcs.MIPS_WEIGHT
        sfflag=self.membflag & (self.lcs.LIR_ZCLUST > 10.**logLIR) & (self.logstellarmass > logM)
        allflag=self.membflag  & (self.logstellarmass > logM)
        self.irfrac=sum(nscaled[sfflag])/sum(self.membflag[allflag])
    def plotcolormass(self,plottitle=1):
        #color=self.sdssumag-self.sdssrmag
        #mag=self.sdssrmag
        color=self.lcs.ABSMAG[:,2]-self.lcs.ABSMAG[:,4] # u-r
        #mag=l.lcs.ABSMAG[:,5]
        mag=self.logstellarmass
        plot(mag[self.membflag],color[self.membflag],'kd',markersize=1,color='0.1',mfc='None',mec='0.7',label='Spec Memb')
        #hexbin(mag[self.membflag],color[self.membflag],cmap=cm.Greys,gridsize=[200,50],vmin=0,vmax=7)
        scatter(mag[self.irflag & self.membflag],color[self.irflag & self.membflag],color='r',s=(l.lcs.SFR_ZCLUST[self.irflag & self.membflag])*10,alpha=1,label='SFRx10')
        plot(mag[self.irflag & self.membflag & self.lcs.AGNKAUFF],color[self.irflag & self.membflag & self.lcs.AGNKAUFF],'r*',mec='k', label='$AGN$')
        ymin=-.5
        ymax=4

        axis([8.8,12.4,ymin,ymax])
        yticks(arange(int(ymin),int(ymax)+1,1))
        xticks(arange(9,13,1))
        ## x1,x2=xlim()
        ## xl=arange(x1+.5,x2,1)
        ## plot(xl,.25*(xl-10)+2.35,'k-')
        ## plot(xl,.25*(xl-10)+2.35-.2,'k--')

class ediscs(): # entire ediscs spec catalogs
    def __init__(self):
        infile=ediscspath+'ediscs_charyelbaz_lir.fits'
        self.lir=atpy.Table(infile)
        infile=ediscspath+'ediscs_photometry.v23.fits'
        self.phot=atpy.Table(infile)
        infile=ediscspath+'ediscs_kcorrect_v4.0.topcat.fits'
        self.kcorrect=atpy.Table(infile)
        infile=ediscspath+'ediscs_spec1d_info.fits'
        self.spec=atpy.Table(infile)
        
        self.logLi=(-1*(self.kcorrect.UGRIZ_ABSMAG_00[:,3]-LCScommon.SolarMag['i'])/2.5)
        self.logstellarmass=-0.68 +  0.70*(self.kcorrect.UGRIZ_ABSMAG_00[:,1]-self.kcorrect.UGRIZ_ABSMAG_00[:,3]) + self.logLi
        self.logstellarmass=1.15+0.70*(self.kcorrect.UGRIZ_ABSMAG_00[:,1]-self.kcorrect.UGRIZ_ABSMAG_00[:,3]) -0.4*(self.kcorrect.UGRIZ_ABSMAG_00[:,3] + 5.*log10(LCScommon.h))

        self.spiralflag = (self.phot.TYPE > 0) & (self.phot.TYPE < 9)
        self.irflag=  (self.lir.FLUX24 > 0.)
        self.irflag = self.lir.MATCHFLAG24

        #R200deg=zeros(len(self.irflag),'f')
        #membflag=self.spec.CLUSTER_Z > 0.
        #R200deg[membflag] = r200(self.spec.CLUSTER_SIGMA[membflag],self.spec.CLUSTER_Z[membflag])
        #self.membflag=(e.lir.CLUSTER_FULLNAME == self.cluster_fullname) &(sqrt((e.spec.RA-self.cra)**2+(e.spec.DEC-self.cdec)**2) < self.R200deg)
    def storemembers(self):
        self.membflag=np.zeros(len(self.irflag),'bool')
        for cl in myediscsclusters:
            self.membflag[cl.membflag]=np.ones(sum(cl.membflag),'bool')
    def getsffrac(self,logLIR,logM):
        irscaled=self.membflag/e.lir.MIPS_WEIGHT/e.lir.SPEC_WEIGHT
        
        allscaled=self.membflag/e.lir.SPEC_WEIGHT
        sfflag=self.membflag & (self.lir.LIR_ZCLUST > 10.**logLIR) & (self.logstellarmass > logM)
        allflag=self.membflag  & (self.logstellarmass > logM)

        self.irfrac=sum(irscaled[sfflag])/sum(allscaled[allflag])
    def plotcolormass(self,plottitle=1):
        color=self.kcorrect.UGRIZ_ABSMAG_00[:,0]-self.kcorrect.UGRIZ_ABSMAG_00[:,2] # u-r
        mag=self.logstellarmass
        #plot(mag[self.membflag],color[self.membflag],'k.',color='0.3',alpha=.4, label='Spec Memb')
        plot(mag[self.membflag],color[self.membflag],'kd',markersize=1,color='0.1',mfc='None',mec='0.7',label='Spec Memb')
        #hexbin(mag[self.membflag],color[self.membflag],cmap=cm.Greys,gridsize=[100,50],vmin=0,vmax=7)
        scatter(mag[self.irflag & self.membflag],color[self.irflag & self.membflag],color='r',s=(self.lir.SFR_ZCLUST[self.irflag])*2,alpha=1,label = 'SFRx2')
        

        #plot(mag[self.irflag],color[self.irflag ],'r.')
        #plot(mag[self.irflag&self.agn3 & self.On24ImageFlag],color[self.irflag&self.agn3 & self.On24ImageFlag],'b*',markersize=10)
        ymin=-.5
        ymax=4
        xmin=8.8
        xmax=12.5
        axis([8.8,12.4,ymin,ymax])
        yticks(arange(int(ymin),int(ymax)+1,1))
        xticks(arange(9,13,1))
        ## x1,x2=xlim()
        ## xl=arange(x1+.5,x2,1)
        ## plot(xl,.25*(xl-10)+2.35-.3,'k-')
        ## plot(xl,.25*(xl-10)+2.35-.5,'k--')


class lcluster(): #local cluster clusters
    def __init__(self,clustername):
        self.cz=LCScommon.clusterbiweightcenter[clustername]/3.e5
        self.csigma=LCScommon.clusterbiweightscale[clustername]
        self.cra=LCScommon.clusterRA[clustername]
        self.cdec=LCScommon.clusterDec[clustername]
        self.R200Mpc,b=r200(1.*self.csigma,self.cz)
        self.R200deg=b.value
        self.clustername=clustername
        self.mcl = Mcl(self.csigma,self.cz)
        print clustername, ' R200 (deg) = ',self.R200deg,self.R200Mpc,self.cra,self.cdec
        self.membflag=((sqrt((l.lcs.RA-self.cra)**2 + (l.lcs.DEC-self.cdec)**2) < self.R200deg) & (abs(self.cz-l.lcs.ZDIST)*3.e5 < nsigma*self.csigma) & (l.lcs.CLUSTER == clustername))
        self.dvflag=( (abs(self.cz-l.lcs.ZDIST)*3.e5 < nsigma*self.csigma) & (l.lcs.CLUSTER == clustername))
        self.f80MJysr=LCScommon.clusterf80MJysr[clustername]
        self.irflag=  (l.lcs.FLUX24 > 0.) & (log10(l.lcs.LIR_ZDIST)> 8.7) #& (~l.lcs.AGNKAUFF)
        self.sumsfr=sum(l.lcs.SFR_ZCLUST[self.irflag & self.membflag]/l.lcs.MIPS_WEIGHT[self.irflag & self.membflag])
        self.nmemb=sum(self.membflag)
        self.sfr=l.lcs.SFR_ZCLUST[self.irflag & self.membflag]
        self.sumssfr=self.sumsfr/sum(10.**l.logstellarmass[self.irflag])
        self.agnflag=self.membflag & l.lcs.AGNKAUFF


        #
        # need to calculate stellar mass
        #
        
        #self.irflag = self.apexflag & self.sdssflag & self.membflag
        #self.massflag=(log10(self.stellarmass) < masslimit)
        #flag=self.irflag  & self.massflag
        #self.sumsfr=sum(self.SFR24[flag])
        #self.sumstellarmass=sum(self.stellarmass[flag])
        #self.nmemb=sum(self.membflag & self.On24ImageFlag & self.massflag)
        #self.sumssfr=self.sumsfr/self.sumstellarmass
        #self.sumssfrerr=sqrt(self.sumsfrerr**2+(self.sumssfr**2*self.sumstellarmasserr**2))/self.sumstellarmass
    def plotSFRStellarmass(self,spiralflag=False):
        lcsmarkersize=8
        #plot(self.stellarmass[self.irflag],self.SFR24[self.irflag],'k^',markersize=lcsmarkersize)
        if spiralflag:
            baseflag = self.irflag & l.spiralflag & self.membflag
        else:
            baseflag = self.irflag & self.membflag
        #plot(l.logstellarmass[baseflag ],l.lcs.SFR_ZCLUST[baseflag],'r^',mec='r',markersize=lcsmarkersize)
        if (self.clustername.find('MKW11')>-1):
            lab1='_nolegend_'
            lab2='$AGN$'
            lab3='$AGN-Kew$'
        else:
            lab1='_nolegend_'
            lab2='_nolegend_'
            lab3='_nolegend_'
        plot(l.logstellarmass[baseflag ],l.lcs.SFR_ZCLUST[baseflag ],'r.',mec='r',markersize=2,label=lab1)
        #plot(l.logstellarmass[baseflag & l.greenflag],l.lcs.SFR_ZCLUST[baseflag & l.greenflag],'g^',mec='g',markersize=lcsmarkersize)
        #plot(l.logstellarmass[baseflag & l.blueflag],l.lcs.SFR_ZCLUST[baseflag & l.blueflag],'r.',mec='r',markersize=lcsmarkersize)
        agnflag=l.lcs.AGNKAUFF
        plot(l.logstellarmass[baseflag & agnflag],l.lcs.SFR_ZCLUST[baseflag & agnflag],'c*',mec='r',markersize=8,label=lab2)
        agnflag=l.lcs.AGNKEWLEY
        plot(l.logstellarmass[baseflag & agnflag],l.lcs.SFR_ZCLUST[baseflag & agnflag],'k*',mec='r',markersize=8,label=lab3)
        #ax=gca()
        #ax.set_xscale('log')

    def plotsSFRz(self):
        plot(self.cz,self.sumssfr,'ro',markersize=lcsmarkersize)

    def plotIntSFRsigma(self):
        plot(self.csigma,self.sumsfr,'ro',markersize=lcsmarkersize)

    def plotIntSFRMclMcl(self):
        y=self.sumsfr/self.mcl*1.e14
        plot(self.mcl/1.e14,y,'ro',markersize=lcsmarkersize)
        return y
    def plotIntSFRMclRedshift(self):
        plot(self.cz,self.sumsfr/self.mcl*1.e14,'ro',markersize=lcsmarkersize)

    def plotIntSFRNmembRedshift(self):
        plot(self.cz,self.sumsfr/self.nmemb,'ro',markersize=lcsmarkersize)

    def plotIntSFRNmembMcl(self):
        y=self.sumsfr/self.nmemb
        plot(self.mcl/1.e14,y,'ro',markersize=lcsmarkersize)
        return y
    def plotsSFRMcl(self):
        y=self.sumssfr
        plot(self.mcl/1.e14,y,'ro',markersize=lcsmarkersize)
        return self.mcl/1.e14,y
    def plotcolormag(self):
        #color=self.sdssumag-self.sdssrmag
        #mag=self.sdssrmag
        color=l.lcs.ABSMAG[:,2]-l.lcs.ABSMAG[:,4] # u-r
        mag=l.lcs.ABSMAG[:,5]
        #mag=l.logstellarmass
        plot(mag[self.membflag],color[self.membflag],'k.',alpha=.4)
        scatter(mag[self.irflag & self.membflag],color[self.irflag & self.membflag],color='r',s=(l.lcs.SFR_ZCLUST[self.irflag & self.membflag])*10,alpha=0.8)

        #plot(mag[self.irflag],color[self.irflag ],'r.')
        #plot(mag[self.irflag&self.agn3 & self.On24ImageFlag],color[self.irflag&self.agn3 & self.On24ImageFlag],'b*',markersize=10)
        ymin=-.5
        ymax=4
        xmin=-24.5
        xmax=-14
        axis([xmin,xmax,ymin,ymax])
        yticks(arange(int(ymin),int(ymax)+1,1))
        xticks(arange(int(xmin),int(xmax)+1,2))
        ax=gca()
        text(.5,.85,self.clustername,horizontalalignment='center',transform=ax.transAxes)
    def plotcolormass(self,plottitle=1):
        #color=self.sdssumag-self.sdssrmag
        #mag=self.sdssrmag
        color=l.lcs.ABSMAG[:,2]-l.lcs.ABSMAG[:,4] # u-r
        #mag=l.lcs.ABSMAG[:,5]
        mag=l.logstellarmass
        plot(mag[self.membflag],color[self.membflag],'k.',alpha=.4)
        scatter(mag[self.irflag & self.membflag],color[self.irflag & self.membflag],color='r',s=(l.lcs.SFR_ZCLUST[self.irflag & self.membflag])*10,alpha=0.8)
        

        #plot(mag[self.irflag],color[self.irflag ],'r.')
        #plot(mag[self.irflag&self.agn3 & self.On24ImageFlag],color[self.irflag&self.agn3 & self.On24ImageFlag],'b*',markersize=10)
        ymin=-.5
        ymax=4

        axis([8.8,12.4,ymin,ymax])
        yticks(arange(int(ymin),int(ymax)+1,1))
        xticks(arange(9,13,1))
        #yticks(arange(ymin,ymax+1,1))
        if plottitle:
            ax=gca()
            text(.5,.85,self.clustername,horizontalalignment='center',transform=ax.transAxes)

    def plotpositions(self,plotsingle=1):
        if plotsingle:
            figure()
        dx=(self.cra-self.cra)
        dy=(self.cdec-self.cdec)
        cir=Circle((dx,dy),radius=self.R200deg,color='None',ec='0.3',alpha=0.1)
        gca().add_patch(cir)

        RA=l.lcs.RA
        DEC=l.lcs.DEC
        flag=self.membflag# & self.on24imageflag
        dx=(RA[flag] - self.cra)
        dy=(DEC[flag]-self.cdec)
        plot(dx,dy,color='0.5',marker='o',ls='None',markersize=2)
        mflag=self.dvflag# & self.on24imageflag& self.drflag
        dx=(RA[mflag] - self.cra)
        dy=(DEC[mflag]-self.cdec)
        plot(dx,dy,'k.',markersize=2)
        mflag=self.irflag & self.dvflag & ~l.lcs.agnflag #& self.drflag
        dx=(RA[mflag] - self.cra)
        dy=(DEC[mflag]-self.cdec)
        plot(dx,dy,'ro',markersize=4)
        #show R200
        #axis('equal')
        axis([-1.3,1.3,-1.3,1.3])
        
        ax=gca()
        text(.05,.85,'$'+self.clustername+'$',transform=ax.transAxes,horizontalalignment='left')

        
class ecluster(): #ediscs clusters
    def __init__(self,clustername):
        self.cluster_fullname=ediscsCommon.cluster_fullname[clustername]
        self.dvflag=(e.lir.CLUSTER_FULLNAME == self.cluster_fullname)
        self.cz=mean(e.lir.CLUSTER_Z[self.dvflag])
        try:
            self.cra=ediscsCommon.racenter[clustername]
            self.cdec=ediscsCommon.deccenter[clustername]
        except:
            self.cra=mean(e.spec.RA[self.dvflag])
            self.cdec=mean(e.spec.DEC[self.dvflag])
        try:
            self.csigma=ediscsCommon.sigma[clustername]
        except:
            print 'no velocity dispersion for ',clustername
            print 'setting sigma = 400 km/s'
            self.clsigma=400.
        self.R200Mpc,b=r200(self.csigma,self.cz)
        self.R200deg=b.value
        self.mcl = Mcl(self.csigma,self.cz)

        self.membflag=(e.lir.CLUSTER_FULLNAME == self.cluster_fullname) &(sqrt((e.spec.RA-self.cra)**2+(e.spec.DEC-self.cdec)**2) < self.R200deg)

        self.irflag= self.dvflag & (e.lir.FLUX24 > 0.)
        self.sumsfr=sum(e.lir.SFR_ZCLUST[self.irflag & self.membflag]/e.lir.MIPS_WEIGHT[self.irflag& self.membflag]/e.lir.SPEC_WEIGHT[self.irflag& self.membflag])
        self.sumssfr=self.sumsfr/sum(10.**l.logstellarmass[self.membflag]/e.lir.SPEC_WEIGHT[self.membflag])
        self.nmemb=sum(self.membflag/e.lir.SPEC_WEIGHT)
        self.sfr=e.lir.SFR_ZCLUST[self.irflag & self.membflag]
        self.logstellarmass=e.logstellarmass[self.membflag]


    def plotzhist(self):
        figure()
        hist(e.lir.Z[self.membflag])
        xlabel('$ Redshift $')
        ylabel('$ N_{gal}$')
        
    def plotSFRStellarmass(self,spiralflag=False):
        edimarkersize=8
        if spiralflag:
            flag=self.irflag & e.spiralflag
        else:
            flag = self.irflag
        if (self.cluster_fullname.find('1216')>-1):
            lab1='_nolegend_'
            lab2='$AGN$'
        else:
            lab1='_nolegend_'
            lab2='_nolegend_'

        plot(e.logstellarmass[flag],e.lir.SFR_ZCLUST[flag],'b.',label=lab1,markersize=2)#,markersize=edimarkersize)
        #plot(self.logstellarmass[self.irflag & self.redflag],self.SFRir[self.irflag & self.redflag],'r.',markersize=edimarkersize)

    def plotsSFRz(self):
        plot(self.cz,self.sumssfr,'bo',markersize=edimarkersize)
        
    def plotIntSFRsigma(self):
        plot(self.csigma,self.sumsfr,'bo',markersize=edimarkersize)

    def plotIntSFRMclMcl(self):
        y=self.sumsfr/self.mcl*1.e14
        plot(self.mcl/1.e14,y,'bo',markersize=edimarkersize)
        return y
    def plotIntSFRMclRedshift(self):
        plot(self.cz,self.sumsfr/self.mcl*1.e14,'bo',markersize=edimarkersize)
    
    def plotIntSFRNmembRedshift(self):
        y=self.sumsfr/self.nmemb
        plot(self.cz,y,'bo',markersize=edimarkersize)

    def plotIntSFRNmembMcl(self):
        y=self.sumsfr/self.nmemb
        plot(self.mcl/1.e14,y,'bo',markersize=edimarkersize)
        return y
    def plotsSFRMcl(self):
        y=self.sumssfr
        plot(self.mcl/1.e14,y,'bo',markersize=edimarkersize)
        return self.mcl/1.e14,y
    def plotpositions(self,plotsingle=1):
        if plotsingle:
            figure()
        dx=(self.cra-self.cra)*60
        dy=(self.cdec-self.cdec)*60
        cir=Circle((dx,dy),radius=self.R200deg*60,color='None',ec='0.3',alpha=0.1)
        gca().add_patch(cir)

        RA=e.spec.RA
        DEC=e.spec.DEC
        mflag=self.dvflag# & self.on24imageflag& self.drflag
        dx=(RA[mflag] - self.cra)*60
        dy=(DEC[mflag]-self.cdec)*60
        plot(dx,dy,'k.',markersize=2)
        mflag=self.irflag & self.dvflag #& self.drflag
        dx=(RA[mflag] - self.cra)*60
        dy=(DEC[mflag]-self.cdec)*60
        plot(dx,dy,'ro',markersize=4)
        #show R200
        #axis('equal')
        axis([-5,5,-5,5])
        
        ax=gca()
        text(.05,.85,'$'+self.cluster_fullname+'$',transform=ax.transAxes,horizontalalignment='left')
    def plotcolormag(self):
        #color=self.sdssumag-self.sdssrmag
        #mag=self.sdssrmag
        color=e.kcorrect.UGRIZ_ABSMAG_00[:,0]-e.kcorrect.UGRIZ_ABSMAG_00[:,2] # u-r
        mag=e.kcorrect.UGRIZ_ABSMAG_00[:,3]
        #mag=l.logstellarmass
        plot(mag[self.membflag],color[self.membflag],'k.',alpha=.4)
        scatter(mag[self.irflag],color[self.irflag],color='r',s=(e.lir.SFR_ZCLUST[self.irflag])*2,alpha=0.8)

        #plot(mag[self.irflag],color[self.irflag ],'r.')
        #plot(mag[self.irflag&self.agn3 & self.On24ImageFlag],color[self.irflag&self.agn3 & self.On24ImageFlag],'b*',markersize=10)
        ymin=-.5
        ymax=4
        xmin=-24.5
        xmax=-14
        axis([xmin,xmax,ymin,ymax])
        yticks(arange(int(ymin),int(ymax)+1,1))
        xticks(arange(int(xmin),int(xmax)+1,2))
        ax=gca()
        text(.5,.85,self.cluster_fullname,horizontalalignment='center',transform=ax.transAxes)
    def plotcolormass(self,plottitle=1):
        color=e.kcorrect.UGRIZ_ABSMAG_00[:,0]-e.kcorrect.UGRIZ_ABSMAG_00[:,2] # u-r
        mag=e.logstellarmass
        plot(mag[self.membflag],color[self.membflag],'k.',alpha=.4)
        scatter(mag[self.irflag & self.membflag],color[self.irflag & self.membflag],color='r',s=(e.lir.SFR_ZCLUST[self.irflag])*2,alpha=0.8)
        

        #plot(mag[self.irflag],color[self.irflag ],'r.')
        #plot(mag[self.irflag&self.agn3 & self.On24ImageFlag],color[self.irflag&self.agn3 & self.On24ImageFlag],'b*',markersize=10)
        ymin=-.5
        ymax=4
        xmin=8.8
        xmax=12.5
        axis([8.8,12.4,ymin,ymax])
        yticks(arange(int(ymin),int(ymax)+1,1))
        xticks(arange(9,13,1))
        #yticks(arange(ymin,ymax+1,1))
        if plottitle:
            ax=gca()
            text(.5,.85,self.cluster_fullname,horizontalalignment='center',transform=ax.transAxes)


def plotbothSFRStellarmass(spiralflag=False,plotfield=False):
    figure(figsize=(6,4))
    subplots_adjust(bottom=.2,left=.18)
    i=0
    for cl in mylocalclusters:
        cl.plotSFRStellarmass(spiralflag=spiralflag)
    for cl in myediscsclusters:
        cl.plotSFRStellarmass(spiralflag=spiralflag)

    if spiralflag:
        xbin,ybin,ybinerr=my.binit(l.logstellarmass[l.spiralflag & l.membflag & l.irflag],l.lcs.SFR_ZDIST[l.spiralflag & l.membflag & l.irflag],7)
        plot(xbin,ybin,'ro',markersize=10,label='$LCS$')
        errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='r')

        xbin,ybin,ybinerr=my.binit(e.logstellarmass[e.spiralflag & e.lir.CLUSTER_MEMBER & e.irflag], e.lir.SFR_ZCLUST[e.spiralflag & e.lir.CLUSTER_MEMBER & e.irflag],7)
        plot(xbin,ybin,'bo',markersize=10,label='$EDisCS$')
        errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b')
    else:
        xbin,ybin,ybinerr=my.binit(l.logstellarmass[l.membflag & l.irflag],l.lcs.SFR_ZDIST[l.membflag & l.irflag],7)
        plot(xbin,ybin,'ro',markersize=10,label='$LCS$')
        errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='r')

        xbin,ybin,ybinerr=my.binit(e.logstellarmass[ e.lir.CLUSTER_MEMBER & e.irflag], e.lir.SFR_ZCLUST[e.lir.CLUSTER_MEMBER & e.irflag],7)
        plot(xbin,ybin,'bo',markersize=10,label='$EDisCS$')
        errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b')
    legend(loc='upper left',numpoints=1,prop={'size':12})
    if plotfield:
        xbin,ybin,ybinerr=my.binit(l.logstellarmass[l.spiralflag & ~l.membflag & l.irflag],l.lcs.SFR_ZDIST[l.spiralflag & ~l.membflag & l.irflag],7)
        plot(xbin,ybin,'ro',mfc='None',markersize=10,label='LCS-field')
        errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='r')

        xbin,ybin,ybinerr=my.binit(e.logstellarmass[e.spiralflag & ~e.lir.CLUSTER_MEMBER & e.irflag], e.lir.SFR_ZSPEC[e.spiralflag & ~e.lir.CLUSTER_MEMBER & e.irflag],7)
        plot(xbin,ybin,'bo',mfc='None',mec='b',markersize=10,label='EDisCS-field')
        errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b')

    ax=gca()
    xl=arange(9,11.7,.1)
    #xl=10.**xl

    #yl=(10.**xl/12.85e10)
    #plot((xl),yl,'k-')
    textdx=.1
    ## #text((xl[0])-textdx,yl[0],'$\mathrm{z=0}$',fontsize=16,horizontalalignment='right')
    ## yl2=(10.**xl)/16.e9
    ## if spiralflag:
    ##     plot(xl,yl2,'k--')
    ##     text(xl[0]-textdx,yl2[0],'$\mathrm{0.25x}$',fontsize=14,horizontalalignment='right')
    ## yl2=4*yl2
    ## if spiralflag:
    ##     plot(xl,yl2,'k:')
    ## #text(xl[0]-textdx,yl2[0],'$\mathrm{50x}$',fontsize=16,horizontalalignment='right')
    ## # plot slope of one line from Elbaz et al 2011
    ## ye=(10.**xl/.08e9)
    # plot SF Main Sequence from Elbaz et al 2011
    xe=arange(9.,11.5,.1)
    xe=10.**xe
    ye=(.08e-9)*xe
    plot(log10(xe),(ye),'k-',lw=1,label='Elbaz+2011')
    
    text(8.3,ye[0]*2,'$\mathrm{Elbaz \ z=0}$',color='k',fontsize=14,horizontalalignment='left')
    
    ye=(.269e-9)*xe
    plot(log10(xe),(ye),'k--',lw=1,label='Elbaz+2011')
    
    text(8.3,ye[0]*2,'$\mathrm{Elbaz \ z=0.6}$',color='k',fontsize=14,horizontalalignment='left')

    #ax.set_xscale('log')
    ax.set_yscale('log')
    axis([8.25,12.25,.01,200])
    xlabel('$\mathrm{log_{10}(Stellar \ Mass /M_\odot)}$', fontsize=18)
    ylabel('$\mathrm{SFR_{IR}\  (M_\odot \ yr^{-1})}$',fontsize=18)
    if spiralflag:
        savefig(figuredir+'SFRStellarmassspirals.eps')
        savefig(figuredir+'SFRStellarmassspirals.png')
    else:
        savefig(figuredir+'SFRStellarmass.eps')
        savefig(figuredir+'SFRStellarmass.png')

def plotbothIntSFRsigma():
    figure(figsize=[6,4])
    subplots_adjust(bottom=.2,left=.18)

    for cl in mylocalclusters:
        cl.plotIntSFRsigma()
    for cl in myediscsclusters:
        cl.plotIntSFRsigma()
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlabel('$\mathrm{ \sigma \ (km/s) }$', fontsize=20)
    ylabel('$\mathrm{SFR_{IR}\  (M_\odot \ yr^{-1})}$',fontsize=20)
    x=arange(200.,1500.,100)
    y=(x/200.)**3
    plot(x,y)
    #plotevolutionlines()
    savefig(figuredir+'IntSFRsigma.eps')


def plotevolutionlines2(ysdss,yediscs):
    '''
    an updated version of the evolution lines as inferred from weschler et al.

    yvalues is a length-3 array,  first value corresponds to yvalue for sdss

    note, ac values greater than 0.7 are estimated
    '''
    
    ysdss=array(ysdss,'f')
    yediscs=array(yediscs,'f')
    print ysdss
    print yediscs
    ac=array([.53,.68,.72,.8,.9])
    halomass=array([1.e14,3.e14,5.e14,10e14,20e14])
    ac=array([.6,.7,.75,.8,.9])
    halomass=array([2.74,6.17,10.97,17.14,24.68])#in 1.e14 Msun
    redshift=array([0,.6])
    yvalues=array([median(ysdss),median(yediscs)],'f')
    for i in range(len(halomass)):
        mofz=halomass[i]*exp(-2*ac[i]*(redshift))
        plot(mofz,yvalues,'0.5')

def plotevolutionlines3():
    '''
    the plot the lines horizontally on Mcl vs z plot
    
    an updated version of the evolution lines as inferred from weschler et al.

    yvalues is a length-3 array,  first value corresponds to yvalue for sdss

    note, ac values greater than 0.7 are estimated
    '''
    
    ac=array([.53,.68,.72,.8,.9])
    halomass=array([1.e14,3.e14,5.e14,10e14,20e14])
    ac=array([.6,.7,.75,.8,.9])
    halomass=array([2.74,6.17,10.97,17.14,24.68])#in 1.e14 Msun
    redshift=arange(0,.9,.01)
    for i in range(len(halomass)):
        mofz=halomass[i]*exp(-2*ac[i]*(redshift))
        plot(redshift,mofz*1.e14,'0.5')

def plotevolutionlines():
    zp=10
    c1=log10(10.)
    c2=1.
    zc=.75
    x2=400.
    ac=0.6
    x1=x2*(exp(-2.*ac*zc)*sqrt(.7+.3*(1.+zc)**3.))**(1./3.)
    x1=log10(x1)
    x2=log10(x2)
    y1=zp*(x1-3)+c1
    y2=zp*(x2-3)+c2
    xl=array([x1,x2],'f')
    yl=array([y1,y2],'f')
    print x2, "evol = ",10.**(y1-y2)
    #y=(x/400)**3 + y0
    plot(10.**xl,10.**yl)

    x2=600.
    ac=0.7
    x1=x2*(exp(-2.*ac*zc)*sqrt(.7+.3*(1.+zc)**3.))**(1./3.)
    x1=log10(x1)
    x2=log10(x2)
    y1=zp*(x1-3)+c1
    y2=zp*(x2-3)+c2
    xl=array([x1,x2],'f')
    yl=array([y1,y2],'f')
    #y=(x/400)**3 + y0
    print x2, "evol = ",10.**(y1-y2),xl,yl
    plot(xl,yl)

    x2=800.
    ac=0.75

    x1=x2*(exp(-2.*ac*zc)*sqrt(.7+.3*(1.+zc)**3.))**(1./3.)
    x1=log10(x1)
    x2=log10(x2)
    y1=zp*(x1-3)+c1
    y2=zp*(x2-3)+c2
    xl=array([x1,x2],'f')
    yl=array([y1,y2],'f')
    #y=(x/400)**3 + y0
    print x2, "evol = ",10.**(y1-y2)
    plot(xl,yl)

    x2=1000.
    ac=0.8

    x1=x2*(exp(-2.*ac*zc)*sqrt(.7+.3*(1.+zc)**3.))**(1./3.)
    x1=log10(x1)
    x2=log10(x2)
    y1=zp*(x1-3)+c1
    y2=zp*(x2-3)+c2
    xl=array([x1,x2],'f')
    yl=array([y1,y2],'f')
    #y=(x/400)**3 + y0
    print x2, "evol = ",10.**(y1-y2)
    plot(xl,yl)

    x2=1200.
    ac=0.9
    x1=x2*(exp(-2.*ac*zc)*sqrt(.7+.3*(1.+zc)**3.))**(1./3.)
    x1=log10(x1)
    x2=log10(x2)
    y1=zp*(x1-3)+c1
    y2=zp*(x2-3)+c2
    xl=array([x1,x2],'f')
    yl=array([y1,y2],'f')
    #y=(x/400)**3 + y0
    print x2, "evol = ",10.**(y1-y2)
    plot(xl,yl)

def plotbothsSFRz(plotsingle=1):
    fsize=15
    if plotsingle:
        figure(figsize=(6,4))
        subplots_adjust(bottom=.2,left=.18)
        fsize=18

    for cl in mylocalclusters:
        cl.plotsSFRz()
    for cl in myediscsclusters:
        cl.plotsSFRz()
    ax=gca()
    ax.set_yscale('log')
    axis([-.02,1,1.e-13,2.e-9])
    ylabel(r'$\mathrm{ \Sigma SFR/\Sigma Stellar \  Mass \  (yr^{-1})}$', fontsize=fsize)
    legend(loc='lower right',numpoints=1,scatterpoints=1,prop={'size':12})
    if plotsingle:
        xlabel(r'$\mathrm{Redshift}$',fontsize=fsize)
        legend(loc='lower right',numpoints=1,scatterpoints=1,prop={'size':12})
        savefig(figuredir+'sSFRz.eps')

    
def plotbothIntSFRMclRedshift(plotsingle=1):
    fsize=15
    if plotsingle:
        figure(figsize=(6,4))
        subplots_adjust(bottom=.2,left=.18)
        fsize=18
    for cl in mylocalclusters:
        cl.plotIntSFRMclRedshift()
    for cl in myediscsclusters:
        cl.plotIntSFRMclRedshift()
    ax=gca()
    ax.set_yscale('log')
    axis([-0.02,1,1.5,1000.])
    legend(loc='lower right',numpoints=1,scatterpoints=1,prop={'size':12})
    ylabel('$\mathrm{\Sigma SFR_{IR}/M_{cl} \ (10^{-14}\ yr^{-1}) }$',fontsize=fsize)
    if plotsingle:
        xlabel('$\mathrm{Redshift}$', fontsize=fsize)
        savefig(figuredir+'IntSFRMclRedshift.eps')
def plotbothIntSFRNmembRedshift(plotsingle=1):
    fsize=15
    if plotsingle:
        figure(figsize=(6,4))
        subplots_adjust(bottom=.2,left=.18)
        fsize=18

    for cl in mylocalclusters:
        cl.plotIntSFRNmembRedshift()
    for cl in myediscsclusters:
        cl.plotIntSFRNmembRedshift()
    ax=gca()
    ax.set_yscale('log')
    axis([-0.02,1,.03,50.])
    ylabel('$\mathrm{\Sigma SFR_{IR}/N_{gal} \ (M_\odot \ yr^{-1})}$',fontsize=fsize)
    if plotsingle:
        xlabel('$\mathrm{Redshift} $', fontsize=fsize)
        savefig(figuredir+'IntSFRNmembRedshift.eps')
def plotredshiftevolution():
    figure(figsize=(6,9))
    subplots_adjust(left=.18,hspace=0.05)
    subplot(3,1,1)
    plotbothIntSFRMclRedshift(plotsingle=0)
    gca().set_xticklabels(([]))
    subplot(3,1,2)
    plotbothIntSFRNmembRedshift(plotsingle=0)
    gca().set_xticklabels(([]))
    subplot(3,1,3)
    plotbothsSFRz(plotsingle=0)
    xlabel('$\mathrm{Redshift} $', fontsize=15)
    savefig(figuredir+'redshiftevolution.eps')
    savefig(figuredir+'redshiftevolution.png')
def plotmasstrends():
    figure(figsize=(6,9))
    subplots_adjust(left=.18,hspace=0.05)
    subplot(3,1,1)
    plotbothIntSFRMclMcl(plotsingle=0)
    gca().set_xticklabels(([]))
    subplot(3,1,2)
    plotbothIntSFRNmembMcl(plotsingle=0)
    gca().set_xticklabels(([]))
    subplot(3,1,3)
    plotbothsSFRMcl(plotsingle=0)
    savefig(figuredir+'masstrends.eps')
    savefig(figuredir+'masstrends.png')

def plotbothIntSFRMclMcl(plotsingle=1):
    ysdss=[]
    fsize=15
    if plotsingle:
        figure(figsize=[6,4])
        subplots_adjust(bottom=.2,left=.18)
        fsize=18

    for cl in mylocalclusters:
        y=cl.plotIntSFRMclMcl()
        ysdss.append(str(y))
    yediscs=[]
    for cl in myediscsclusters:
        y=cl.plotIntSFRMclMcl()
        #print y
        yediscs.append(str(y))
    ax=gca()
    plotevolutionlines2(ysdss,yediscs)
    ysdss=array(ysdss,'f')
    yediscs=array(yediscs,'f')
    print 'median evolution Sum SFR/Mcl ',median(ysdss),median(yediscs),median(yediscs)/median(ysdss)
    print 'mean evolution ',mean(ysdss),mean(yediscs),mean(yediscs)/mean(ysdss)
    xl=array([1,30.])
    plot(xl,median(ysdss)*ones(len(xl)),'k')
    xl=array([.7,20.])
    plot(xl,median(yediscs)*ones(len(xl)),'k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    text(.95,median(ysdss),'$z=0$',horizontalalignment='right',fontsize=14)
    text(.65,median(yediscs),'$z=0.6$',horizontalalignment='right',fontsize=14)

    axis([.3,40,1.5,1000])

    #ylabel('$\mathrm{ SFR_{IR}\ /M_{cl} \ (M_\odot \ yr^{-1}/ 10^{14} \ M_\odot) }$',fontsize=20)
    ylabel('$\mathrm{ \Sigma SFR_{IR}\ /M_{cl} \ (10^{-14} \ yr^{-1}) }$',fontsize=fsize)
    if plotsingle:
        xlabel('$\mathrm{ M_{cl}/(10^{14} \ M_\odot}) $', fontsize=fsize)
        savefig(figuredir+'IntSFRMclMcl.eps')

def plotbothIntSFRNmembMcl(plotsingle=1):
    fsize=15
    if plotsingle:
        figure(figsize=(6,4))
        subplots_adjust(bottom=.2,left=.18)
        fsize=18
    #figure()
    ysdss=[]
    for cl in mylocalclusters:
        y=cl.plotIntSFRNmembMcl()
        ysdss.append(y)
    yediscs=[]
    for cl in myediscsclusters:
        y=cl.plotIntSFRNmembMcl()
        yediscs.append(y)
    ax=gca()
    ysdss=array(ysdss,'f')
    yediscs=array(yediscs,'f')
    plotevolutionlines2(ysdss,yediscs)
    print 'median evolution of Sum SFR/Nmemb =  ',median(ysdss),median(yediscs),median(yediscs)/median(ysdss)
    print 'mean evolution of Sum SFR/Nmemb = ',mean(ysdss),mean(yediscs),mean(yediscs)/mean(ysdss)
    xl=array([1,30.])
    plot(xl,median(ysdss)*ones(len(xl)),'k')
    xl=array([.8,20.])
    plot(xl,median(yediscs)*ones(len(xl)),'k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    axis([.3,40,.05,20])
    text(.95,median(ysdss),'$z=0$',horizontalalignment='right',fontsize=14)
    text(.75,median(yediscs),'$z=0.6$',horizontalalignment='right',fontsize=14)

    ylabel('$\mathrm{\Sigma SFR/N_{gal} \ (M_\odot \ yr^{-1})}$',fontsize=fsize)
    if plotsingle:
        xlabel('$\mathrm{ M_{cl}/ (10^{14} \ M_\odot)}$', fontsize=fsize)
        savefig(figuredir+'IntSFRNmembMcl.eps')
def plotbothsSFRMcl(plotsingle=1):
    fsize=15
    if plotsingle:
        figure(figsize=(6,4))
        subplots_adjust(bottom=.2,left=.18)
        fsize=18
    #figure()
    ysdss=[]
    xsdss=[]
    for cl in mylocalclusters:
        x,y=cl.plotsSFRMcl()
        ysdss.append(y)
        xsdss.append(x)
    yediscs=[]
    xediscs=[]
    for cl in myediscsclusters:
        if cl.cz < 0.:
            continue
        x,y=cl.plotsSFRMcl()
        yediscs.append(y)
        xediscs.append(x)
    ax=gca()
    #xbin,ybin,ybinerr=my.binit(xediscs,yediscs,3)
    #plot(xbin,ybin,'ko',markersize=10,label='_nolegend_')
    #errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k')

    ysdss=array(ysdss,'f')
    yediscs=array(yediscs,'f')
    plotevolutionlines2(ysdss,yediscs)
    print 'median evolution of Sum SFR/Sum M* =  ',median(ysdss),median(yediscs),median(yediscs)/median(ysdss)
    print 'mean evolution of Sum SFR/Sum M* = ',mean(ysdss),mean(yediscs),mean(yediscs)/mean(ysdss)
    xl=array([1,30.])
    plot(xl,median(ysdss)*ones(len(xl)),'k')
    xl=array([.8,20.])
    plot(xl,median(yediscs)*ones(len(xl)),'k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    axis([.3,40,1.e-13,2.e-9])
    text(.95,median(ysdss),'$z=0$',horizontalalignment='right',fontsize=14)
    text(.75,median(yediscs),'$z=0.6$',horizontalalignment='right',fontsize=14)

    ylabel('$\mathrm{\Sigma SFR/\Sigma M_* \ ( \ yr^{-1})}$',fontsize=fsize)
    xlabel('$\mathrm{ M_{cl}/ (10^{14} \ M_\odot)}$', fontsize=fsize)
    if plotsingle:

        savefig(figuredir+'IntsSFRMcl.eps')
def plothistforDJ():
    t=cl1040.SFRir[cl1040.irflag].tolist()+cl105411.SFRir[cl105411.irflag].tolist()+cl105412.SFRir[cl105412.irflag].tolist()+cl1216.SFRir[cl1216.irflag].tolist()+cl1227.SFRir[cl1227.irflag].tolist()+cl1232.SFRir[cl1232.irflag].tolist()+cl1354.SFRir[cl1354.irflag].tolist()
    figure()
    mybins=arange(0,150,10)
    hist(t,bins=mybins)
    axvline(x=50,color='r',ls='--')
    axis([0,135.,0.,65.])
    xlabel('$\mathrm{SFR \ (M_\odot/yr)}$',fontsize=20)
    ylabel('$\mathrm{N_{gal}}$',fontsize=20)

def plotlcscolormag(plotmass=False,subplots=True):

    if subplots:
        figure(figsize=(8,6))
        subplots_adjust(bottom=.15,left=.12,wspace=.05,hspace=.05)
        titleflag=1
    else:
        figure(figsize=(8,6))
        subplots_adjust(bottom=.15,left=.1)
        titleflag=0

    i=0
    allclusters=mylocalclusters
    if subplots:
        fstring='LCS-subplots-'
    else:
        fstring='LCS-onepanel-'

    for cl in allclusters:
        i=i+1
        if subplots:
            subplot(3,3,i)
        if plotmass:
            cl.plotcolormass(plottitle=titleflag)
        else:
            cl.plotcolormag()
        if subplots:
            LCScommon.multiplotaxes(i)
        x1,x2=xlim()
        xl=arange(x1+.5,x2,1)
        plot(xl,.25*(xl-10)+2.35,'k-')
        plot(xl,.25*(xl-10)+2.35-.2,'k--')

    ax=gca()
    if subplots:
        text(-2.4,1.5,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    else:
        ylabel('$u-r$',fontsize=22)
        #legend(('all','','','SFRx10'),numpoints=1)
    if plotmass:
        if subplots:
            text(-.5,-.4,'$log_{10}(M_*/M_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
        else:
            xlabel('$log_{10}(M_*/M_\odot)$',fontsize=22)
        savefig(figuredir+fstring+'colormass.eps')
    else:
        if subplots:
            text(-.5,-.4,'$M_i$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
        else:
            xlabel('$M_i$',fontsize=22)
        savefig(figuredir+fstring+'colormag.eps')
def plotediscscolormag(plotmass=False,subplots=True):
    if subplots:
        figure(figsize=(8,12))
        subplots_adjust(wspace=.05,hspace=.05)
    else:
        figure(figsize=(8,6))
        subplots_adjust(bottom=.15,left=.1)
    i=0
    allclusters=myediscsclusters
    if subplots:
        fstring='ediscs-subplots-'
        titleflag=1
    else:
        fstring='ediscs-onepanel-'
        titleflag=0
    for cl in allclusters:
        i=i+1
        if subplots:
            subplot(6,4,i)
        if plotmass:
            cl.plotcolormass(plottitle=titleflag)
        else:
            cl.plotcolormag()
        if subplots:
            ediscsmultiplotaxes(i)
        x1,x2=xlim()
        xl=arange(x1+.5,x2,1)
        plot(xl,.25*(xl-10)+2.35-.3,'k-')
        plot(xl,.25*(xl-10)+2.35-.5,'k--')
    ax=gca()
    if subplots:
        text(-3.5,3.1,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    else:
        ylabel('$u-r$',fontsize=22)
        #legend(('all','','','SFRx2'),numpoints=1)
    if plotmass:
        if subplots:
            text(-1.2,-.4,'$log_{10}(M_*/M_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
        else:
            xlabel('$log_{10}(M_*/M_\odot)$',fontsize=22)
        savefig(figuredir+fstring+'colormass.eps')
    else:
        if subplots:
            text(-1.2,-.4,'$M_i$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
        else:
            xlabel('$M_i$',fontsize=22)
        savefig(figuredir+fstring+'colormag.eps')


def plotbothcolormass():

    figure(figsize=(6,6))
    subplots_adjust(bottom=.15,left=.12,hspace=.01,right=.9,top=.9)
    fstring='LCS-ediscs-onepanel-'
    subplot(2,1,2)
    l.plotcolormass()
    legend(loc='lower right',numpoints=1,scatterpoints=1,prop={'size':12})
    x1,x2=xlim()
    xl=arange(x1+.5,x2,1)
    plot(xl,.25*(xl-10)+2.35,'k-')
    plot(xl,.25*(xl-10)+2.35-.2,'k--')
    ylabel('$u-r$',fontsize=22)
    xlabel('$log_{10}(M_*/M_\odot)$',fontsize=22)
    text(.05,.85,'$LCS$',fontsize=20,transform=gca().transAxes,horizontalalignment='left')
    subplot(2,1,1)
    e.plotcolormass()    
    legend(loc='lower right',numpoints=1,scatterpoints=1,prop={'size':12})
    #legend(('all','','','SFRx10'),numpoints=1)
    x1,x2=xlim()
    xl=arange(x1+.5,x2,1)
    plot(xl,.25*(xl-10)+2.35-.3,'k-')
    plot(xl,.25*(xl-10)+2.35-.5,'k--')
    text(.05,.85,'$EDisCS$',fontsize=20,transform=gca().transAxes,horizontalalignment='left')
    gca().set_xticklabels(([]))
    ylabel('$u-r$',fontsize=22)

    savefig(figuredir+fstring+'colormass.eps')

def plotediscspositions():
    figure(figsize=[8,12])
    subplots_adjust(wspace=.05,hspace=.05)
    i=1
    for cl in myediscsclusters:
        subplot(6,4,i)
        cl.plotpositions(plotsingle=False)
        ax=gca()
        ediscsmultiplotaxes(i)
        i += 1
    text(-1.2,-.4,'$ \Delta RA \ (arcmin) $',fontsize=20,transform=ax.transAxes,horizontalalignment='center')

    text(-3.6,3.1,'$ \Delta Dec \ (arcmin) $',fontsize=20,transform=ax.transAxes,verticalalignment='center',rotation=90)
    savefig(figuredir+'Ediscspositions.eps')
    savefig(figuredir+'Ediscspositions.png')

def plotLCSpositions():
    figure(figsize=[8,8])
    subplots_adjust(wspace=.05,hspace=.05)
    i=1
    for cl in mylocalclusters:
        subplot(3,3,i)
        cl.plotpositions(plotsingle=False)
        ax=gca()
        LCScommon.multiplotaxes(i)
        i += 1
    text(-.6,-.3,'$ \Delta RA \ (deg) $',fontsize=20,transform=ax.transAxes,horizontalalignment='center')

    text(-2.5,1.5,'$ \Delta Dec \ (deg) $',fontsize=20,transform=ax.transAxes,verticalalignment='center',rotation=90)
    savefig(figuredir+'LCSpositions.eps')
    savefig(figuredir+'LCSpositions.png')
def plotbothMclRedshift():
    figure(figsize=[6,4])
    subplots_adjust(bottom=.2,left=.18)
    i=0
    for cl in mylocalclusters:
        if i == 0:
            plot(cl.cz,cl.mcl,'ro',markersize=10,label='$LCS$')
        else:
            plot(cl.cz,cl.mcl,'ro',markersize=10,label='_nolegend_')
        i += 1
    i=0
    for cl in myediscsclusters:
        if i == 0:
            plot(cl.cz,cl.mcl,'bo',markersize=10,label='$EDisCS$')
        else:
            plot(cl.cz,cl.mcl,'bo',markersize=10,label='_nolegend_')
        i += 1
    axis([0,1,1.e13,5.e15])
    gca().set_yscale('log')

    legend(loc='lower left',numpoints=1,prop={'size':12})
    plotevolutionlines3()
    xlabel('$Redshift $',fontsize=18)
    ylabel('$M_{cl} \ (M_\odot) $',fontsize=18)
    savefig(figuredir+'LCS-ediscs-MclRedshift.eps')
    savefig(figuredir+'LCS-ediscs-MclRedshift.png')
def plotbothIntSFRMcl():
    figure(figsize=[6,4])
    subplots_adjust(bottom=.2,left=.18)
    i=0
    for cl in mylocalclusters:
        if i == 0:
            plot(cl.mcl,cl.sumsfr,'ro',markersize=10,label='$LCS$')
        else:
            plot(cl.mcl,cl.sumsfr,'ro',markersize=10,label='_nolegend_')
        i += 1
    i=0
    for cl in myediscsclusters:
        if i == 0:
            plot(cl.mcl,cl.sumsfr,'bo',markersize=10,label='$EDisCS$')
            #scatter(cl.mcl,cl.sumsfr,s=80,c=cl.cz,vmin=0,vmax=1,label='$EDisCS$',cmap='jet_r')
        else:
            plot(cl.mcl,cl.sumsfr,'bo',markersize=10,label='_nolegend_')
            #sp=scatter(cl.mcl,cl.sumsfr,s=80,c=cl.cz,vmin=0,vmax=1,label='_nolegend_',cmap='jet_r')
        i += 1
    #axis([0,1,1.e13,5.e15])
    #colorbar(sp,fraction=.08)
    gca().set_xscale('log')
    gca().set_yscale('log')
    xl=arange(13.5,15.5,.01)
    xl=10.**xl
    yl=xl/2.5e13
    plot(xl,yl,'k-')
    plot(xl,17.*yl,'k--')

    legend(loc='upper left',numpoints=1,scatterpoints=1,prop={'size':12})
    
    #plotevolutionlines3()
    ylabel('$\sum SFR \ (M_\odot/yr) $',fontsize=18)
    xlabel('$M_{cl} \ (M_\odot) $',fontsize=18)
    savefig(figuredir+'LCS-ediscs-IntSFRMcl.eps')
    savefig(figuredir+'LCS-ediscs-IntSFRMcl.png')
    
def fitevolsSFRz():
    x=[]
    y=[]
    for cl in (mylocalclusters+myediscsclusters):
        y.append(cl.sumsfr/cl.sumstellarmass)
        x.append(cl.cz)
    x=array(x,'f')
    y=array(y,'f')
    yerr=array(yerr,'f')
        
def fitpowerlaw(x,y,err,pinit):
    fitfunc= lambda p, x: p[0] + p[1]*x
    errfunc = lambda p, x, y, err : (y-fitfunc(p,x))/err
    out=optimize.leastsq(errfunc,pinit,args=(x,y,err),full_output=1)
    pffinal=out[0]
    print pfinal
    covar=out[1]
    print covar


def compare24flux():
    figure()
    clf()
    subplots_adjust(bottom=.15,left=.15)
    x=l.lcs.mag1
    flag = (x > 10) & (x < 17)
    plot(x,23.9-2.5*log10(l.lcs.APEXFLUX),'ro',label='APEX',markersize=9)
    diff=x-(23.9-2.5*log10(l.lcs.APEXFLUX))
    print 'APEX mean, std = ',mean(diff[flag]),std(diff[flag])
    
    plot(x,23.9-2.5*log10(l.lcs.FLUX24),'bo',label='FLUX_BEST',markersize=8)
    diff=x-(23.9-2.5*log10(l.lcs.FLUX24))
    print 'FLUX_BEST mean, std = ',mean(diff[flag]),std(diff[flag])

    plot(x,23.9-2.5*log10(l.lcs.FLUX_AUTO*141.086),'go',label='FLUX_AUTO',markersize=4)
    diff=x-(23.9-2.5*log10(l.lcs.FLUX_AUTO*141.086))
    print 'FLUX_AUTO mean, std = ',mean(diff[flag]),std(diff[flag])
    print 'sources with > .5 mag difference = '
    flag=(diff < -1.) & (x < 16) & (x > 10)
    diff2=diff[flag]
    diff_index=arange(len(diff))[flag]
    for i in diff_index:
        print  l.lcs.CLUSTER[i],l.lcs.NSAID[i], l.lcs.p_cs[i]
    xl=arange(10,22)
    plot(xl,xl,'k--')
    legend(numpoints=1,loc='lower right')
    axis([10,21,10,21])
    ylabel('Various SE and Apex Fluxes',fontsize=20)
    xlabel('Model Mag from galfit',fontsize=20)

##### MAIN   ####

if __name__ == "__main__":
    l=lcs()
    e=ediscs()

    # initiate Local Clusters
    mkw11=lcluster('MKW11')
    mkw8=lcluster('MKW8')
    awm4=lcluster('AWM4')
    a2052=lcluster('A2052')
    a2063=lcluster('A2063')
    ngc6107=lcluster('NGC6107')
    coma=lcluster('Coma')
    herc=lcluster('Hercules')
    a1367=lcluster('A1367')

    # initiate EDisCS Clusters
    # 28 groups + clusters
    # should be 26 according to Milvang-Jensen+2008
    # we have data for 24
    cl1018=ecluster('cl1018')
    cl1037=ecluster('cl1037')
    cl1037a=ecluster('cl1037a') # group, according to MJ+2008
    cl1040=ecluster('cl1040')
    #cl1040a=ecluster('cl1040a')
    #cl1040b=ecluster('cl1040b')
    cl105411=ecluster('cl105411')
    #cl105411a=ecluster('cl105411a')
    cl105412=ecluster('cl105412')
    #cl105412a=ecluster('cl105412a')
    cl1059=ecluster('cl1059')
    cl1103=ecluster('cl1103')
    cl1103a=ecluster('cl1103a') # group, according to MJ+2008
    cl1103b=ecluster('cl1103b') # group, according to MJ+2008
    cl1138=ecluster('cl1138')
    cl1138a=ecluster('cl1138a') # group, according to MJ+2008
    cl1202=ecluster('cl1202')
    cl1216=ecluster('cl1216')
    cl1227=ecluster('cl1227')
    cl1227a=ecluster('cl1227a') # group, according to MJ+2008
    cl1232=ecluster('cl1232')
    cl1301=ecluster('cl1301')
    cl1301a=ecluster('cl1301a') # group, according to MJ+2008
    cl1353=ecluster('cl1353')
    cl1354=ecluster('cl1354')
    cl1354a=ecluster('cl1354a') # group, according to MJ+2008
    cl1411=ecluster('cl1411')
    cl1420=ecluster('cl1420')

    # store instances of each class in a list so that I can easily loop over all instances when making plots of full samples
    #mylocalclusters=[lcluster('MKW11'),lcluster('MKW8'),lcluster('AWM4'),lcluster('A2052'),lcluster('A2063'),lcluster('NGC6107'),lcluster('Coma'),lcluster('Hercules'),lcluster('A1367')]
    mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc6107,coma,herc,a1367]
    #myediscsclusters=[cl1018,cl1037,cl1037a,cl1040,cl1040a,cl1040b,cl105411,cl105411a,cl105412,cl105412a,cl1059,cl1103,cl1103a,cl1103b,cl1138,cl1138a,cl1202,cl1216,cl1227,cl1227a,cl1232,cl1301,cl1301a,cl1353,cl1354,cl1354a,cl1411,cl1420]
    myediscsclusters=[cl1018,cl1037,cl1037a,cl1040,cl105411,cl105412,cl1059,cl1103,cl1103a,cl1103b,cl1138,cl1138a,cl1202,cl1216,cl1227,cl1227a,cl1232,cl1301,cl1301a,cl1353,cl1354,cl1354a,cl1411,cl1420]

    e.storemembers()
def allplots():
    #plotediscspositions()
    #plotLCSpositions()
    #plotbothMclRedshift()
    #plotbothIntSFRMcl()
    #plotbothIntSFRMclMcl()
    plotbothIntSFRMclRedshift()
    #plotbothIntSFRNmembRedshift()
    #plotbothsSFRz()
    #plotbothSFRStellarmass()
    #plotbothSFRStellarmass(spiralflag=1)
    #plotbothcolormass()

    plotbothIntSFRNmembMcl()


    # not needed
    #plotbothIntSFRsigma()
    #plotlcscolormag(plotmass=True,subplots=True)
    #plotlcscolormag(plotmass=True,subplots=False)
    #plotediscscolormag(plotmass=True,subplots=True)
    #plotediscscolormag(plotmass=True,subplots=False)



#########  old code
## class eclusterold(edi.baseCluster): #ediscs clusters
##     def __init__(self,clustername):
##         edi.baseCluster.__init__(self,clustername)
##         self.irflag = self.matchflag24 & self.on24imageflag & self.supermembflag & self.drflag
##         self.sumssfr=sum(self.SFRir[self.irflag])/sum(self.stellarmass[self.irflag])
##         self.sumsfr=sum(self.SFRir[self.irflag])
##         self.sumstellarmass=sum(self.stellarmass[self.irflag])
##         self.nmemb=sum(self.supermembflag & self.on24imageflag & self.drflag)

##     def plotSFRStellarmass(self):
##         edimarkersize=8
##         plot(self.stellarmass[self.irflag],self.SFRir[self.irflag],'b.',markersize=edimarkersize)
##         plot(self.stellarmass[self.irflag & self.redflag],self.SFRir[self.irflag & self.redflag],'r.',markersize=edimarkersize)
##     def plotsSFRz(self):
##         plot(self.cz,self.sumssfr,'bo',markersize=edimarkersize)
        
##     def plotIntSFRsigma(self):
##         sfr=sum(self.SFRir[self.irflag])
##         plot(self.csigma,sfr,'bo',markersize=edimarkersize)

##     def plotIntSFRMclMcl(self):
##         y=self.sumsfr/self.mcl*1.e14
##         plot(self.mcl/1.e14,y,'bo',markersize=edimarkersize)
##         return y
##     def plotIntSFRMclRedshift(self):
##         plot(self.cz,self.sumsfr/self.mcl*1.e14,'bo',markersize=edimarkersize)
    
##     def plotIntSFRNmembRedshift(self):
##         y=self.sumsfr/self.nmemb
##         plot(self.cz,y,'bo',markersize=edimarkersize)

##     def plotIntSFRNmembMcl(self):
##         y=self.sumsfr/self.nmemb
##         plot(self.mcl/1.e14,y,'bo',markersize=edimarkersize)
##         return y
##     def plotpositions(self):
##         #figure()
##         flag=self.membflag & self.on24imageflag
##         dx=(self.ra[flag] - self.cra)*60
##         dy=(self.dec[flag]-self.cdec)*60
##         plot(dx,dy,color='0.5',marker='.',ls='None',markersize=2)
##         mflag=self.membflag & self.on24imageflag& self.drflag
##         dx=(self.ra[mflag] - self.cra)*60
##         dy=(self.dec[mflag]-self.cdec)*60
##         plot(dx,dy,'k.',markersize=2)
##         mflag=self.irflag & self.drflag
##         dx=(self.ra[mflag] - self.cra)*60
##         dy=(self.dec[mflag]-self.cdec)*60
##         plot(dx,dy,'r.',markersize=2)
##         #show R200
##         dx=(self.cra-self.cra)*60
##         dy=(self.cdec-self.cdec)*60
##         cir=Circle((dx,dy),radius=self.r200deg*60,color='0.5',ec='0.8',alpha=0.3)
##         gca().add_patch(cir)
##         axis([-5,5,-5,5])
##         title(self.fullname)

## class lclusterold(lcs.baseClusterNSA): #local cluster clusters
##     def __init__(self,clustername):
##         lcs.baseClusterNSA.__init__(self,clustername)
##         self.irflag = self.apexflag & self.sdssflag & self.membflag
##         self.massflag=(log10(self.stellarmass) < masslimit)
##         flag=self.irflag  & self.massflag
##         self.sumsfr=sum(self.SFR24[flag])
##         self.sumstellarmass=sum(self.stellarmass[flag])
##         self.nmemb=sum(self.membflag & self.On24ImageFlag & self.massflag)
##         self.sumssfr=self.sumsfr/self.sumstellarmass
##         #self.sumssfrerr=sqrt(self.sumsfrerr**2+(self.sumssfr**2*self.sumstellarmasserr**2))/self.sumstellarmass
##     def plotSFRStellarmass(self):
##         lcsmarkersize=8
##         #plot(self.stellarmass[self.irflag],self.SFR24[self.irflag],'k^',markersize=lcsmarkersize)
##         plot(self.stellarmass[self.irflag & self.redflag],self.SFR24[self.irflag & self.redflag],'r^',mec='r',markersize=lcsmarkersize)
##         plot(self.stellarmass[self.irflag & self.greenflag],self.SFR24[self.irflag & self.greenflag],'g^',mec='g',markersize=lcsmarkersize)
##         plot(self.stellarmass[self.irflag & self.blueflag],self.SFR24[self.irflag & self.blueflag],'b^',mec='b',markersize=lcsmarkersize)
##         ax=gca()
##         ax.set_xscale('log')

##     def plotsSFRz(self):
##         plot(self.cz,self.sumssfr,'k^',markersize=lcsmarkersize)

##     def plotIntSFRsigma(self):
##         plot(self.csigma,self.sumsfr,'k^',markersize=lcsmarkersize)

##     def plotIntSFRMclMcl(self):
##         y=self.sumsfr/self.mcl*1.e14
##         plot(self.mcl/1.e14,y,'k^',markersize=lcsmarkersize)
##         return y
##     def plotIntSFRMclRedshift(self):
##         plot(self.cz,self.sumsfr/self.mcl*1.e14,'k^',markersize=lcsmarkersize)

##     def plotIntSFRNmembRedshift(self):
##         plot(self.cz,self.sumsfr/self.nmemb,'k^',markersize=lcsmarkersize)

##     def plotIntSFRNmembMcl(self):
##         y=self.sumsfr/self.nmemb
##         plot(self.mcl/1.e14,y,'k^',markersize=lcsmarkersize)
##         return y
##     def plotcolormag(self):
##         color=self.sdssumag-self.sdssrmag
##         mag=self.sdssrmag
##         scatter(mag[self.irflag],color[self.irflag],color='r',s=(self.SFR24[self.irflag])*50,alpha=0.5)
##         flag=self.sdssflag & self.On24ImageFlag
##         plot(mag[flag],color[flag],'k.')
##         plot(mag[self.irflag & self.On24ImageFlag],color[self.irflag & self.On24ImageFlag],'r.')
##         plot(mag[self.irflag&self.agn3 & self.On24ImageFlag],color[self.irflag&self.agn3 & self.On24ImageFlag],'b*',markersize=10)
##         ymin=0
##         ymax=4

##         axis([12,18,ymin,ymax])
##         yticks(arange(ymin,ymax+1,1))
##         title(self.prefix)
