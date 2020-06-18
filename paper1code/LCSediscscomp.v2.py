#!/usr/bin/env python

from pylab import *
from LCScommon import *
from ediscsCommon import *
import LCSReadmasterBaseNSA as lcs
import ediscsreadspitzercatalogs as edi
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
#directory where figures will be saved
figuredir= '/Users/rfinn/research/LCSEdiscsComp/'
class lclusterspec
class lcluster(lcs.baseClusterNSA): #local cluster clusters
    def __init__(self,clustername):
        lcs.baseClusterNSA.__init__(self,clustername)
        self.irflag = self.apexflag & self.sdssflag & self.membflag
        self.massflag=(log10(self.stellarmass) < masslimit)
        flag=self.irflag  & self.massflag
        self.sumsfr=sum(self.SFR24[flag])
        self.sumstellarmass=sum(self.stellarmass[flag])
        self.nmemb=sum(self.membflag & self.On24ImageFlag & self.massflag)
        self.sumssfr=self.sumsfr/self.sumstellarmass
        #self.sumssfrerr=sqrt(self.sumsfrerr**2+(self.sumssfr**2*self.sumstellarmasserr**2))/self.sumstellarmass
    def plotSFRStellarmass(self):
        lcsmarkersize=8
        #plot(self.stellarmass[self.irflag],self.SFR24[self.irflag],'k^',markersize=lcsmarkersize)
        plot(self.stellarmass[self.irflag & self.redflag],self.SFR24[self.irflag & self.redflag],'r^',mec='r',markersize=lcsmarkersize)
        plot(self.stellarmass[self.irflag & self.greenflag],self.SFR24[self.irflag & self.greenflag],'g^',mec='g',markersize=lcsmarkersize)
        plot(self.stellarmass[self.irflag & self.blueflag],self.SFR24[self.irflag & self.blueflag],'b^',mec='b',markersize=lcsmarkersize)
        ax=gca()
        ax.set_xscale('log')

    def plotsSFRz(self):
        plot(self.cz,self.sumssfr,'k^',markersize=lcsmarkersize)

    def plotIntSFRsigma(self):
        plot(self.csigma,self.sumsfr,'k^',markersize=lcsmarkersize)

    def plotIntSFRMclMcl(self):
        y=self.sumsfr/self.mcl*1.e14
        plot(self.mcl/1.e14,y,'k^',markersize=lcsmarkersize)
        return y
    def plotIntSFRMclRedshift(self):
        plot(self.cz,self.sumsfr/self.mcl*1.e14,'k^',markersize=lcsmarkersize)

    def plotIntSFRNmembRedshift(self):
        plot(self.cz,self.sumsfr/self.nmemb,'k^',markersize=lcsmarkersize)

    def plotIntSFRNmembMcl(self):
        y=self.sumsfr/self.nmemb
        plot(self.mcl/1.e14,y,'k^',markersize=lcsmarkersize)
        return y
    def plotcolormag(self):
        color=self.sdssumag-self.sdssrmag
        mag=self.sdssrmag
        scatter(mag[self.irflag],color[self.irflag],color='r',s=(self.SFR24[self.irflag])*50,alpha=0.5)
        flag=self.sdssflag & self.On24ImageFlag
        plot(mag[flag],color[flag],'k.')
        plot(mag[self.irflag & self.On24ImageFlag],color[self.irflag & self.On24ImageFlag],'r.')
        plot(mag[self.irflag&self.agn3 & self.On24ImageFlag],color[self.irflag&self.agn3 & self.On24ImageFlag],'b*',markersize=10)
        ymin=0
        ymax=4

        axis([12,18,ymin,ymax])
        yticks(arange(ymin,ymax+1,1))
        title(self.prefix)

class ecluster(edi.baseCluster): #ediscs clusters
    def __init__(self,clustername):
        edi.baseCluster.__init__(self,clustername)
        self.irflag = self.matchflag24 & self.on24imageflag & self.supermembflag & self.drflag
        self.sumssfr=sum(self.SFRir[self.irflag])/sum(self.stellarmass[self.irflag])
        self.sumsfr=sum(self.SFRir[self.irflag])
        self.sumstellarmass=sum(self.stellarmass[self.irflag])
        self.nmemb=sum(self.supermembflag & self.on24imageflag & self.drflag)

    def plotSFRStellarmass(self):
        edimarkersize=8
        plot(self.stellarmass[self.irflag],self.SFRir[self.irflag],'b.',markersize=edimarkersize)
        plot(self.stellarmass[self.irflag & self.redflag],self.SFRir[self.irflag & self.redflag],'r.',markersize=edimarkersize)
    def plotsSFRz(self):
        plot(self.cz,self.sumssfr,'bo',markersize=edimarkersize)
        
    def plotIntSFRsigma(self):
        sfr=sum(self.SFRir[self.irflag])
        plot(self.csigma,sfr,'bo',markersize=edimarkersize)

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
    def plotpositions(self):
        #figure()
        flag=self.membflag & self.on24imageflag
        dx=(self.ra[flag] - self.cra)*60
        dy=(self.dec[flag]-self.cdec)*60
        plot(dx,dy,color='0.5',marker='.',ls='None',markersize=2)
        mflag=self.membflag & self.on24imageflag& self.drflag
        dx=(self.ra[mflag] - self.cra)*60
        dy=(self.dec[mflag]-self.cdec)*60
        plot(dx,dy,'k.',markersize=2)
        mflag=self.irflag & self.drflag
        dx=(self.ra[mflag] - self.cra)*60
        dy=(self.dec[mflag]-self.cdec)*60
        plot(dx,dy,'r.',markersize=2)
        #show R200
        dx=(self.cra-self.cra)*60
        dy=(self.cdec-self.cdec)*60
        cir=Circle((dx,dy),radius=self.r200deg*60,color='0.5',ec='0.8',alpha=0.3)
        gca().add_patch(cir)
        axis([-5,5,-5,5])
        title(self.fullname)
    
def plotbothSFRStellarmass():
    figure()
    for cl in mylocalclusters:
        cl.plotSFRStellarmass()
    for cl in myediscsclusters:
        cl.plotSFRStellarmass()
    ax=gca()
    xl=arange(9,11.7,.1)
    xl=10.**xl

    yl=(xl/12.85e10)
    plot(xl,yl,'k-')
    textdx=2.e7
    text(xl[0]-textdx,yl[0],'$\mathrm{z=0}$',fontsize=16,horizontalalignment='right')
    yl2=10*yl
    plot(xl,yl2,'k--')
    text(xl[0]-textdx,yl2[0],'$\mathrm{10x}$',fontsize=16,horizontalalignment='right')
    yl2=50*yl
    plot(xl,yl2,'k:')
    text(xl[0]-textdx,yl2[0],'$\mathrm{50x}$',fontsize=16,horizontalalignment='right')
    # plot slope of one line from Elbaz et al 2011
    ye=(xl/4.e9)
    plot(xl,ye,color='c',ls='-')
    text(xl[0]-textdx,ye[0],'$\mathrm{Elbaz \ z=0}$',color='c',fontsize=16,horizontalalignment='right')

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    xlabel('$\mathrm{Stellar \ Mass \ (M_\odot)}$', fontsize=20)
    ylabel('$\mathrm{SFR_{IR}\  (M_\odot \ yr^{-1})}$',fontsize=20)
    savefig(figuredir+'SFRStellarmass.eps')

def plotbothsSFRz():
    figure()
    for cl in mylocalclusters:
        cl.plotsSFRz()
    for cl in myediscsclusters:
        cl.plotsSFRz()
    ax=gca()
    ax.set_yscale('log')
    ylabel(r'$\mathrm{ \Sigma SFR/\Sigma Stellar \  Mass \  (M_\odot)}$', fontsize=20)
    xlabel(r'$\mathrm{Redshift}$',fontsize=20)
    savefig(figuredir+'sSFRz.eps')

def plotbothIntSFRsigma():
    figure()
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

def plotbothIntSFRMclMcl():
    figure()
    ysdss=[]
    subplots_adjust(bottom=.15,left=.15)
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
    print 'median evolution ',median(ysdss),median(yediscs),median(yediscs)/median(ysdss)
    print 'mean evolution ',mean(ysdss),mean(yediscs),mean(yediscs)/mean(ysdss)
    xl=array([2,30.])
    plot(xl,median(ysdss)*ones(len(xl)),'k')
    xl=array([1,20.])
    plot(xl,median(yediscs)*ones(len(xl)),'k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    text(1.8,median(ysdss),'$z=0$',horizontalalignment='right',fontsize=14)
    text(.95,median(yediscs),'$z=0.6$',horizontalalignment='right',fontsize=14)

    axis([.3,50,.01,300])
    xlabel('$\mathrm{ M_{cl}/(10^{14} \ M_\odot}) $', fontsize=20)
    ylabel('$\mathrm{ SFR_{IR}\ /M_{cl} \ (M_\odot \ yr^{-1}/ 10^{14} \ M_\odot) }$',fontsize=20)
    savefig(figuredir+'IntSFRMclMcl.eps')

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

def plotbothIntSFRMclRedshift():
    figure()
    subplots_adjust(bottom=.15,left=.15)
    for cl in mylocalclusters:
        cl.plotIntSFRMclRedshift()
    for cl in myediscsclusters:
        cl.plotIntSFRMclRedshift()
    ax=gca()
    ax.set_yscale('log')
    xlabel('$\mathrm{Redshift}$', fontsize=20)
    ylabel('$\mathrm{\Sigma SFR_{IR}/M_{cl} \ (M_\odot \ yr^{-1} / 10^{14} \ M_\odot) }$',fontsize=20)
    savefig(figuredir+'IntSFRMclRedshift.eps')
def plotbothIntSFRNmembRedshift():
    figure()
    for cl in mylocalclusters:
        cl.plotIntSFRNmembRedshift()
    for cl in myediscsclusters:
        cl.plotIntSFRNmembRedshift()
    ax=gca()
    ax.set_yscale('log')
    xlabel('$\mathrm{Redshift} $', fontsize=20)
    ylabel('$\mathrm{\Sigma SFR_{IR}/N_{gal} \ (M_\odot \ yr^{-1})}$',fontsize=20)
    savefig(figuredir+'IntSFRNmembRedshift.eps')

def plotbothIntSFRNmembMcl():
    figure()
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
    xl=array([2,30.])
    plot(xl,median(ysdss)*ones(len(xl)),'k')
    xl=array([1,20.])
    plot(xl,median(yediscs)*ones(len(xl)),'k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    axis([.3,50,.01,10])
    text(1.8,median(ysdss),'$z=0$',horizontalalignment='right',fontsize=14)
    text(.95,median(yediscs),'$z=0.6$',horizontalalignment='right',fontsize=14)
    xlabel('$\mathrm{ M_{cl}/ (10^{14} \ M_\odot)}$', fontsize=20)
    ylabel('$\mathrm{\Sigma SFR/N_{gal} \ (M_\odot \ yr^{-1})}$',fontsize=20)
    savefig(figuredir+'IntSFRNmembMcl.eps')
def plothistforDJ():
    t=cl1040.SFRir[cl1040.irflag].tolist()+cl105411.SFRir[cl105411.irflag].tolist()+cl105412.SFRir[cl105412.irflag].tolist()+cl1216.SFRir[cl1216.irflag].tolist()+cl1227.SFRir[cl1227.irflag].tolist()+cl1232.SFRir[cl1232.irflag].tolist()+cl1354.SFRir[cl1354.irflag].tolist()
    figure()
    mybins=arange(0,150,10)
    hist(t,bins=mybins)
    axvline(x=50,color='r',ls='--')
    axis([0,135.,0.,65.])
    xlabel('$\mathrm{SFR \ (M_\odot/yr)}$',fontsize=20)
    ylabel('$\mathrm{N_{gal}}$',fontsize=20)

def plotlcscolormag():
    figure(figsize=(15,12))
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i=i+1
        subplot(3,3,i)
        cl.plotcolormag()
    ax=gca()
    text(-.75,-.35,'$r$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.8,1.9,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'LCScolormag.eps')


def plotediscspositions():
    figure(figsize=[12,12])
    subplots_adjust(wspace=.25,hspace=.35)
    i=1
    for cl in myediscsclusters:
        subplot(4,4,i)
        cl.plotpositions()
        i += 1
    savefig(figuredir+'Ediscspositions.eps')

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
##### MAIN   ####

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
cl1018=ecluster('cl1018')
cl1037=ecluster('cl1037')
cl1040=ecluster('cl1040')
cl1059=ecluster('cl1059')
#cl1103=ecluster('cl1103')
cl1138=ecluster('cl1138')
cl1202=ecluster('cl1202')
cl1216=ecluster('cl1216')
cl1227=ecluster('cl1227')
cl1232=ecluster('cl1232')
cl1301=ecluster('cl1301')
cl1353=ecluster('cl1353')
cl1354=ecluster('cl1354')
cl1411=ecluster('cl1411')
cl1420=ecluster('cl1420')
cl105411=ecluster('cl105411')
cl105412=ecluster('cl105412')

# store instances of each class in a list so that I can easily loop over all instances when making plots of full samples
#mylocalclusters=[lcluster('MKW11'),lcluster('MKW8'),lcluster('AWM4'),lcluster('A2052'),lcluster('A2063'),lcluster('NGC6107'),lcluster('Coma'),lcluster('Hercules'),lcluster('A1367')]
mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc6107,coma,herc,a1367]
myediscsclusters=[cl1018,cl1037,cl1040,cl1059,cl1138,cl1202,cl1216,cl1227,cl1232,cl1301,cl1353,cl1354,cl1411,cl1420,cl105411,cl105412]


def allplots():
    plotbothSFRStellarmass()
    plotbothsSFRz()
    plotbothIntSFRsigma()
    plotbothIntSFRMclMcl()
    plotbothIntSFRMclRedshift()
    plotbothIntSFRNmembRedshift()
    plotbothIntSFRNmembMcl()
    #plotediscspositions()
