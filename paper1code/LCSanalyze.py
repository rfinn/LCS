#!/usr/bin/env python
import pyfits
from LCScommon import *
from pylab import *
import os
import mystuff as my
import aplpy
#from LCSReadmasterBaseWithProfileFits import *
from LCSReadmasterBase import *
import matplotlib.cm as cm
import chary_elbaz_24um as chary
Lsol=3.826e33#normalize by solar luminosity
bellconv=9.8e-11#converts Lir (in L_sun) to SFR/yr
bellconv=4.5e-44#Kenn 98 conversion fro erg/s to SFR/yr

#these correpond to area w/more uniform covereage
MKW824um=array([220.16377,3.4883817,1.3137727,2.5,12.7456],'f')
MKW1124um=array([202.36305,11.746882,1.2454248,2.9,206.4],'f')
NGC24um=array([244.30994,34.933704,1.2865442,2.5,321.317],'f')

fieldColor='0.7'
mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

figuredir=homedir+'Dropbox/Research/LocalClusters/SamplePlots/'
def percentdiff(a,b):
    pdiff=abs(a-b)/((a+b)/2.)*100.
    return pdiff
def drawbox(data,style):#feed in center x,y,dx,dy,rotation E of N
    #xcoords of unrotated box, going around CCW
    xl=array([data[0]-0.5*data[2],data[0]+0.5*data[2],data[0]+0.5*data[2],data[0]-0.5*data[2],data[0]-0.5*data[2]],'d')
    yl=array([data[1]-0.5*data[3],data[1]-0.5*data[3],data[1]+0.5*data[3],data[1]+0.5*data[3],data[1]-0.5*data[3] ],'d')

    xl=array([-0.5*data[2],+0.5*data[2],+0.5*data[2],-0.5*data[2],-0.5*data[2]],'d')
    yl=array([-0.5*data[3],-0.5*data[3],+0.5*data[3],+0.5*data[3],-0.5*data[3] ],'d')

    ang=data[4]*pi/180.*-1.#convert rotation to radians
    #rotate coordinates
    xp=cos(ang)*xl-sin(ang)*yl
    yp=sin(ang)*xl+cos(ang)*yl

    #put back on absolute scale
    xp=data[0]+xp
    yp=data[1]+yp
    #draw rotated box
    plot(xp,yp,style)


class cluster(baseCluster):
    def __init__(self,clustername):
        baseCluster.__init__(self,clustername)
#Get current path so program can tell if this is being run on Becky or Rose's computer

        mypath=os.getcwd()

        if mypath.find('Users') > -1:
            print "Running on Rose's mac pro"
            infile='/Users/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'
        elif mypath.find('home') > -1:
            print "Running on coma"
            infile=homedir+'research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'

        #infile=homedir+'LocalClusters/MasterTables/'+clustername+'mastertable.fits'

##        tb=pyfits.open(infile)
##        tbdata=tb[1].data
##        tb.close()


##        self.spiralFlag=tbdata.field('SPIRALFLAG')
##        self.r0SDSS = tbdata.field('R0SDSS')
##        self.r30SDSS = tbdata.field('R30SDSS')
##        self.r50SDSS = tbdata.field('R50SDSS')
##        self.r90SDSS = tbdata.field('R90SDSS')
##        self.skySDSS = tbdata.field('SKYSDSS')
##        self.r30EncFluxSDSS = tbdata.field('R30ENCFLUXSDSS')
##        self.r90EncFluxSDSS = tbdata.field('R90ENCFLUXSDSS')
##        self.MaxEncFluxSDSS = tbdata.field('MAXENCFLUXSDSS')

##        self.SDSSF30 = tbdata.field('SDSSF30')
##        self.SDSSF50 = tbdata.field('SDSSF50')
##        self.SDSSF90 = tbdata.field('SDSSF90')

##        self.r0F24 = tbdata.field('R0F24')
##        self.r30F24 = tbdata.field('R30F24')
##        self.r90F24 = tbdata.field('R90F24')
##        self.skyF24 = tbdata.field('SKYF24')
##        self.r30EncFluxF24 = tbdata.field('R30ENCFLUXF24')
##        self.r90EncFluxF24 = tbdata.field('R90ENCFLUXF24')
##        self.MaxEncFluxF24 = tbdata.field('MAXENCFLUXF24')

##        self.mipsF30 = tbdata.field('MIPSF30')
##        self.mipsF50 = tbdata.field('MIPSF50')
##        self.mipsF90 = tbdata.field('MIPSF90')

##        #end of master table!

##        self.sSFR50=self.mipsF50/self.SDSSF50
##        self.sSFR5090=(self.mipsF90-self.mipsF50)/(self.SDSSF90 -self.SDSSF50)
##        self.ratio0=self.r0F24/self.r0SDSS        
##        self.ratio30=self.r30F24/self.r30SDSS

##        self.ratio90=self.r90F24/self.r90SDSS
##        self.ratio30EncFlux=self.r30EncFluxF24/self.r30EncFluxSDSS
##        self.ratio90EncFlux=self.r90EncFluxF24/self.r90EncFluxSDSS

##        cutoff=40.
##        self.pd30SDSS=percentdiff(self.r30SDSS,self.r30EncFluxSDSS)
##        self.ratio30FlagSDSS= (self.pd30SDSS < cutoff)
##        self.ratio90FlagSDSS= ((percentdiff(self.r90SDSS,self.r90EncFluxSDSS)) < cutoff)
##        self.ratio30FlagF24= ((percentdiff(self.r30F24,self.r30EncFluxF24)) < cutoff)
##        self.pd90F24=(percentdiff(self.r90F24,self.r90EncFluxF24))
##        self.ratio90FlagF24= (self.pd90F24 < cutoff)
##        self.ratio30Flag = self.ratio30FlagSDSS & self.ratio30FlagF24
##        self.ratio90Flag = self.ratio90FlagSDSS & self.ratio90FlagF24

        self.mipssnrflag = self.mipssnr > 6.
#        self.plotratio90Flag=self.mipssnrflag & self.ratio90FlagF24 & self.spiralFlag & self.apexflag #& ~self.agn2 
#        self.plotratio30Flag=self.mipssnrflag & self.ratio30FlagF24 & self.spiralFlag  & self.apexflag #& ~self.agn2        print self.clustername
        
#        print self.prefix,": number of spirals for radial (90) analysis = ",sum(self.plotratio90Flag)
#        print self.prefix,": number of spirals for radial (30) analysis = ",sum(self.plotratio30Flag)

	#self.L24=self.L24/Lsol
	#self.L24err=self.L24err/Lsol

    def readGalfitResults(self):
        self.galflag=zeros([len(self.ra),3],'i')
        self.galflag24=zeros([len(self.ra),3],'i')
        self.galflag_too_faint24=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.galflag_too_faint=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.galflag_stop=zeros(len(self.ra),'i') # if fitting was stopped, either b/c too faint or results were not reasonable
        self.galflag_stop24=zeros(len(self.ra),'i') # 

        self.galfit_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/'
        self.galfit_sdssR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_sdssR90=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_mipsR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_mipsR90=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        results_file=self.galfit_dir+self.prefix+'-galfitResults-spirals-sdss.dat'
        results_file24=self.galfit_dir+self.prefix+'-galfitResults-spirals-24.dat'

        infile=open(results_file24,'r')
        i=0
        for line in infile:
            t=line.split()
            i=int(t[0])
            index,On24ImageFlag,self.galflag24[i][0],self.galflag24[i][1],self.galflag24[i][2],self.galflag_stop24[i],self.galflag_too_faint24[i],self.galfit_mipsR50[i][0],self.galfit_mipsR50[i][1],self.galfit_mipsR50[i][2],self.galfit_mipsR90[i][0],self.galfit_mipsR90[i][1],self.galfit_mipsR90[i][2]=t
        infile.close()

        infile=open(results_file,'r')
        for line in infile:
            t=line.split()
            i=int(t[0])
            index,On24ImageFlag,self.galflag[i][0],self.galflag[i][1],self.galflag[i][2],self.galflag_stop[i],self.galflag_too_faint[i],self.galfit_sdssR50[i][0],self.galfit_sdssR50[i][1],self.galfit_sdssR50[i][2],self.galfit_sdssR90[i][0],self.galfit_sdssR90[i][1],self.galfit_sdssR90[i][2]=t

        infile.close()

        self.galfit_sdssR50=self.galfit_sdssR50*sdsspixelscale
        self.galfit_sdssR90=self.galfit_sdssR90*sdsspixelscale
        self.galfit_mipsR50=self.galfit_mipsR50*mipspixelscale
        self.galfit_mipsR90=self.galfit_mipsR90*mipspixelscale

    def plotGalfitR90(self):
        keepflag=(self.On24ImageFlag & self.spiralFlag)
        figure()
        x=self.galfit_sdssR90[keepflag,0]
        y=self.galfit_mipsR90[keepflag,0]
        #plot(x,y,'ko',label='1 Comp Fit')
        x=self.galfit_sdssR90[keepflag,1]
        y=self.galfit_mipsR90[keepflag,1]
        plot(x,y,'go',label='2 Comp Fit')
        x=self.galfit_sdssR90[(keepflag & self.membflag),1]
        y=self.galfit_mipsR90[(keepflag & self.membflag),0]
        plot(x,y,'g^',label='2 Comp Fit Memb',markersize=12)
        x=self.galfit_sdssR90[keepflag,2]
        y=self.galfit_mipsR90[keepflag,2]
        #plot(x,y,'ro',label='3 Comp Fit')
        legend(loc='upper right', numpoints=1)
        xlabel('SDSS R90 (arcsec)')
        ylabel('MIPS R90 (arcsec)')
        xl=arange(51)
        plot(xl,xl,'k--')
    def plotGalfitRvsHIDef(self):
        keepflag=(self.On24ImageFlag & self.spiralFlag)
        figure()
        y1=self.galfit_sdssR50[keepflag,0]
        y2=self.galfit_mipsR50[keepflag,0]
        y=y1-y2
        x=self.HIDef[keepflag]
        #plot(x,y,'ko',label='1 Comp Fit')
        y1=self.galfit_sdssR50[keepflag,1]
        y2=self.galfit_mipsR50[keepflag,1]
        y=y1-y2
        plot(x,y,'go',label='2 Comp Fit')
        y1=self.galfit_sdssR50[keepflag,2]
        y2=self.galfit_mipsR50[keepflag,2]
        y=y1-y2
        #plot(x,y,'ro',label='3 Comp Fit')
        
        y1=self.galfit_sdssR90[keepflag,1]
        y2=self.galfit_mipsR90[keepflag,1]
        y=y1-y2
        plot(x,y,'go',label='2 Comp R90')


        x=self.HIDef[keepflag & self.membflag]
        y1=self.galfit_sdssR90[keepflag & self.membflag,1]
        y2=self.galfit_mipsR90[keepflag & self.membflag,1]
        y=y1-y2
        plot(x,y,'g^',label='2 Comp R90',markersize=12)

        legend(loc='lower right', numpoints=1)
        xlabel('HIDef')
        ylabel('SDSS R50-MIPS R50')
        axhline(y=0,ls='--',color='k')
    def plotGalfitR50(self):
        keepflag=(self.On24ImageFlag & self.spiralFlag)
        figure()
        x=self.galfit_sdssR50[keepflag,0]
        y=self.galfit_mipsR50[keepflag,0]
        #plot(x,y,'ko',label='1 Comp Fit')
        x=self.galfit_sdssR50[keepflag,1]
        y=self.galfit_mipsR50[keepflag,1]
        plot(x,y,'go',label='2 Comp Fit')
        x=self.galfit_sdssR50[(keepflag & self.membflag),1]
        y=self.galfit_mipsR50[(keepflag & self.membflag),1]
        plot(x,y,'g^',label='2 Comp Fit Memb',markersize=12)

        x=self.galfit_sdssR50[keepflag,2]
        y=self.galfit_mipsR50[keepflag,2]
        #plot(x,y,'ro',label='3 Comp Fit')
        legend(loc='upper right', numpoints=1)
        xlabel('SDSS R50')
        ylabel('MIPS R50')
        xl=arange(20)
        plot(xl,xl,'k--')
    def compareRad24(self):
        figure()
        plot(self.r30F24,self.r30EncFluxF24,'bo')
        plot(self.r90F24,self.r90EncFluxF24,'ro')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        plot(xl,xl,'k-')
        plot(xl,1.3*xl,'k:')
        plot(xl,.7*xl,'k:')
        plot(self.r30F24[self.mipssnrflag],self.r30EncFluxF24[self.mipssnrflag],'b^',markersize=12)
        plot(self.r90F24[self.mipssnrflag],self.r90EncFluxF24[self.mipssnrflag],'r^',markersize=12)

    def compareRadSDSS(self):
        figure()
        plot(self.r30SDSS,self.r30EncFluxSDSS,'bo')
        plot(self.r90SDSS,self.r90EncFluxSDSS,'ro')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        plot(xl,xl,'k-')
        plot(xl,1.3*xl,'k:')
        plot(xl,.7*xl,'k:')

    def compareR90(self):
        #figure()
        flag=self.ellipseflag & self.spiralFlag & (self.r50SDSS > 6.) & self.membflag
        plot(self.r50SDSS[flag],self.r50F24[flag],'bo')
        plot(self.r90SDSS[flag],self.r50F24[flag],'ro')
        flag=self.ellipseflag & self.spiralFlag & (self.r50SDSS > 6.) & ~self.membflag
        plot(self.r50SDSS[flag],self.r50F24[flag],'bo',markeredgecolor='b',markerfacecolor='None')
        plot(self.r90SDSS[flag],self.r50F24[flag],'ro',markeredgecolor='r',markerfacecolor='None')
        axis([0,40,0,40])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        plot(xl,xl,'k-')
        plot(xl,1.3*xl,'k:')
        plot(xl,.7*xl,'k:')
        axis([0,40,0,40])

        title(self.clustername)

    def comparediffR90localdens(self):
        #figure()
        flag=self.ellipseflag & self.spiralFlag & (self.r50SDSS > 6.) & self.membflag
        plot(self.localdens[flag],self.r50SDSS[flag]-self.r50F24[flag],'bo')
        plot(self.localdens[flag],self.r90SDSS[flag]-self.r50F24[flag],'ro')
        flag=self.ellipseflag & self.spiralFlag & (self.r50SDSS > 6.) & ~self.membflag
        plot(self.localdens[flag],self.r50SDSS[flag]-self.r50F24[flag],'bo',markeredgecolor='b',markerfacecolor='None')
        plot(self.localdens[flag],self.r90SDSS[flag]-self.r50F24[flag],'ro',markeredgecolor='r',markerfacecolor='None')
        axis([1,1000,-25,30])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        #plot(xl,xl,'k-')
        #plot(xl,1.3*xl,'k:')
        #plot(xl,.7*xl,'k:')
        #axis([0,40,0,40])
        axhline(0,color='k')
        ax=gca()
        ax.set_xscale('log')
        axis([1,1000,-25,30])
        title(self.clustername)

    def plotagn(self):
        figure()
        clf()
        plot((self.logn2halpha),(self.logo3hbeta),'k.', label='_nolabel_')

        plot(self.logn2halpha[self.agn2],self.logo3hbeta[self.agn2],'co',markersize=12, label='_nolabel_')
        plot(self.logn2halpha[self.agn1],self.logo3hbeta[self.agn1],'go',markersize=8, label='_nolabel_')
        plot(self.logn2halpha[self.agn3],self.logo3hbeta[self.agn3],'ro',markersize=4, label='_nolabel_')

        #draw AGN diagnostic lines
        x=arange(-3,1,.01)
        y=(.61/(x-.47)+1.19)
        #Kewley
        plot(x,y,'c',label='Kewley & Dopita 2002')
        y =(.61/(x-.05)+1.3)#Kauffman 2003?
        plot(x,y,'g',label='Kauffmann et al. 2003')
        y = ((-30.787+(1.1358*x)+((.27297)*(x)**2))*tanh(5.7409*x))-31.093 #Stasinska 2006	    
        plot(x,y,'r',label='Stasinska et al. 2006')
        axis([-3,1.,-2,2])
        xlabel(r'$\log_{10}(NII/H\alpha)$',fontsize=20)
        ylabel(r'$\log_{10}(OIII/H\beta)$',fontsize=20)
        legend()

    def plotpositions(self):

        #figure()
        #clf()

    #draw footprint of mips data, if applicable
        
        #if self.clustername.find('MKW8') > -1:
        #    drawbox(MKW824um,'r-')
        #if self.clustername.find('MKW11') > -1:
        #    drawbox(MKW1124um,'r-')
        #if self.clustername.find('NGC6107') > -1:
        #    drawbox(NGC24um,'r-')


        #scatter(ra[flag],dec[flag],s=(20-agcmag10[flag]/10)*20+20,color='.8')
        #plot(ra[flag],dec[flag],'k.')

        plot(self.ra[self.sdssflag],self.dec[self.sdssflag],'k.', alpha=0.5,markersize=4,label='SDSS')
        plot(self.ra[self.HIflag],self.dec[self.HIflag],'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='HI')
        plot(self.ra[self.spiralFlag],self.dec[self.spiralFlag],'g^',markerfacecolor='None',markeredgecolor='g',markersize=12,label='Spiral')

        plot(self.ra[self.apexflag],self.dec[self.apexflag],'ro', markerfacecolor='r',markeredgecolor='b',markersize=4,label='24um')
            #plot(ra[flag],dec[flag],'k.')
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x


        #legend(loc='upper right',numpoints=1)


        title(self.clustername,fontsize=14)
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        axis('equal')
        drawbox(cluster24Box[self.clustername],'g-')
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax+1,2,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(round(ymin),ymax+1,2,'i'),fontsize=10)
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.png'

        #savefig(s)

    def plotpositionson24(self,racenter=0,deccenter=0):
        cir=Circle((self.clusterra-racenter,self.clusterdec-deccenter),radius=self.r200deg,color='0.9',ec='0.2',alpha=0.2)
        gca().add_patch(cir)

        baseflag = self.dvflag & self.On24ImageFlag
        flag=self.sdssflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'k.', alpha=0.5,markersize=2,label='SDSS')
        flag = self.HIflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='HI')
        flag=self.spiralFlag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'g^',markerfacecolor='None',markeredgecolor='g',markersize=8,label='Spiral')
        flag=self.agnflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'c*',mec='0.5',label='AGN',markersize=4)
        flag=self.apexflag & ~self.agnflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'ro', markerfacecolor='r',markeredgecolor='r',markersize=1,label='_nolabel_')
        scatter(self.ra[flag]-racenter,self.dec[flag]-deccenter,s=50*self.SFR24[flag],color='r',alpha=0.5,label='24um')
            #plot(ra[flag],dec[flag],'k.')
            #plot(array([self.clusterra])-racenter,array([self.clusterdec])-deccenter,'kx',markersize=15,lw=8)#mark cluster center with a red x



        #legend(loc='upper left',numpoints=1,scatterpoints=1)


        title(self.clustername,fontsize=12)
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #axis('equal')
        axis('equal')

        #xmin,xmax=xlim()
        #print xmin,xmax
        #x1=round(xmin)
        #x2=round(xmax)
        #xt=arange(x1,x2+1)
        #xticks(xt)
        #ymin,ymax=ylim()
        #yticks(arange(round(ymin),ymax,'i'),fontsize=10)

        t=cluster24Box[self.clustername]
        drawbox([t[0]-racenter,t[1]-deccenter,t[2],t[3],t[4]],'g-')


    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.png'

        #savefig(s)

    def plotpositionson24WithRatio(self):
        #plot(self.ra[self.sdssflag & self.On24ImageFlag],self.dec[self.sdssflag& self.On24ImageFlag],'k.', alpha=0.5,markersize=2,label='SDSS')
        plotflag=self.spiralFlag & self.ellipseflag & ~self.agn1 #self.mipssnrflag & (~self.agn2) #& velFlag
        #plotflag=self.spiralFlag & self.ellipseflag & ~self.agn1 #self.mipssnrflag & (~self.agn2) #& velFlag
        #print 'Number of spirals to plot = ',sum(plotflag)
        plot(self.ra[plotflag],self.dec[plotflag],'wo',markerfacecolor='0.2',markeredgecolor='w',markersize=4,label='Spiral')
        #axis('equal')
        drawbox(cluster24Box[self.clustername],'g-')
        a,b=xlim()
        xticks(arange(round(a),b,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(round(ymin),ymax,1,'i'),fontsize=10)
        ra=self.ra[self.sdssflag & self.On24ImageFlag]
        dec=self.dec[self.sdssflag & self.On24ImageFlag]
	dx=0.15
	dy=dx
        ramin,ramax=xlim()
        decmin,decmax=ylim()
	x=arange(ramin,(ramax+dx),dx)
	y=arange(decmin,(decmax+dx),dy)

	z=zeros([len(y),len(x)],'f')
	ylength=len(y)
	xlength=len(x)
	for i in range(len(ra)):
	    xbin=int(round((ra[i]-x[0])/dx))
	    ybin=int(round((dec[i]-y[0])/dy))
	    if (xbin > -1) & (xbin < xlength):
		if (ybin > -1) & (ybin < ylength):
		    z[ybin,xbin] += 1		  
	ncon=8
        cmin=1.
        cmax=24.
        t=arange(log10(cmin),log10(cmax),(log10(cmax)-log10(cmin))/(1.*ncon))
        
        contours=10.**t
        contours=arange(1.,25.,2)
	#contour(x,y,z,contours,linewidth=1)
        #contour(x,y,z,ncon,linewidth=1)
        #print z
	contourf(x,y,z,contours,alpha=.8)#cmap=cm.Greys)

        scale=4.e5*(self.sSFR90-self.sSFR50)
        print scale[plotflag]
        scatter(self.ra[plotflag],self.dec[plotflag],s=(scale[plotflag]),c='r',alpha=0.7)
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x

        title(self.clustername,fontsize=12)
        #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #axis('equal')
        #yticks(arange(round(ymin),ymax+1,1,'i'),fontsize=10)
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.png'

        #savefig(s)

        

    def plotrelativepositionson24(self):
        plot(self.ra[self.sdssflag & self.On24ImageFlag]-self.clusterra,self.dec[self.sdssflag& self.On24ImageFlag]-self.clusterdec,'k.', alpha=0.5,markersize=4,label='SDSS')
        plot(self.ra[self.HIflag & self.On24ImageFlag]-self.clusterra,self.dec[self.HIflag & self.On24ImageFlag]-self.clusterdec,'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='HI')

        plot(self.ra[self.apexflag]-self.clusterra,self.dec[self.apexflag]-self.clusterdec,'ro', markerfacecolor='r',markeredgecolor='b',markersize=4,label='24um')
        plot(array([0]),array([0]),'kx',markersize=15,lw=8,label='_nolegend_')#mark cluster center with a red x

        plot(self.ra[self.agnflag & self.On24ImageFlag],self.dec[self.agnflag & self.On24ImageFlag],'c*',label='AGN',markersize=4)
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x

        #if self.prefix.find('Herc') > -1:
        #    legend(loc='upper left',numpoints=1,scatterpoints=1,prop=10)

        drawbox(cluster24Box[self.clustername],'g-')

        cir=Circle((self.clusterra,self.clusterdec),radius=self.r200deg,color='0.5',ec='k',alpha=0.3)
        gca().add_patch(cir)



        title(self.clustername,fontsize=12)
        axis('equal')


        axis([-1.5,1.5,-2.,2.])
        xmin,xmax=xlim()
        xticks(arange(-1,2,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(-2,3,1,'i'),fontsize=10)

    def plotcolormag(self):
        #color=self.sdssumag-self.sdssrmag
        color=self.sdssgmag-self.sdssrmag
        mag=self.sdssMr
        flag=self.sdssflag & self.On24ImageFlag & self.dvflag
        plot(mag[flag],color[flag],'k.',label='SDSS',markersize=3)
        flag=self.irflag & ~self.agnflag & self.dvflag
        scatter(mag[flag],color[flag],color='r',s=(self.SFR24[flag])*100,alpha=0.5,label='24um')
        #hexbin(mag[flag],color[flag],cmap=cm.jet,gridsize=20)
        flag= self.On24ImageFlag & self.dvflag & self.HIflag
        plot(mag[flag],color[flag],'bo',markerfacecolor='None',mec='b',markersize=8,label='HI')

        flag= self.On24ImageFlag & self.dvflag & self.spiralFlag
        plot(mag[flag],color[flag],'g^',markerfacecolor='None',mec='g',markersize=10,label='Spiral')

        #plot(mag[self.irflag & self.On24ImageFlag],color[self.irflag & self.On24ImageFlag],'r.')
        plot(mag[self.irflag&self.agnflag & self.On24ImageFlag],color[self.irflag&self.agnflag & self.On24ImageFlag],'c*',mec='c',markersize=4,label='AGN')
        ymin=-0.1
        ymax=1.2
        if self.prefix.find('11') > -1:
            leg=legend(numpoints=1,loc='lower left',scatterpoints=1,markerscale=0.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
            for t in leg.get_texts():
                t.set_fontsize('small')
        axis([-23.2,-16,ymin,ymax])
        dy=.2
        #yticks(arange(ymin,ymax+dy,dy))
        title(self.prefix)



    def plotdvdr(self):
        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        dv=(self.supervopt)#-self.biweightvel)#/self.biweightscale

        membflag=(dr/self.r200deg < 1) & (abs(dv) < 3.*self.biweightscale)

        plot(dr[self.sdssflag & self.On24ImageFlag],dv[self.sdssflag& self.On24ImageFlag],'k.', alpha=0.5,markersize=6,label='SDSS')
        plot(dr[self.sdssflag & self.On24ImageFlag & self.agnflag],dv[self.sdssflag& self.On24ImageFlag & self.agnflag],'c*',mec='c', alpha=0.5,markersize=6,label='AGN')
        plotflag=self.spiralFlag & self.ellipseflag#self.mipssnrflag & (~self.agn2) & self.ratio90FlagF24
        #print 'Number of spirals to plot = ',sum(self.plotratio90Flag)
        #plot(dr[self.spiralFlag & ~self.agn2 & ~self.apexflag],dv[self.spiralFlag & ~self.agn2 & ~self.apexflag],'bo',markerfacecolor='b',markeredgecolor='g',markersize=6,label='Spiral',alpha=0.7)
        #plot(dr[self.spiralFlag & self.agn1],dv[self.spiralFlag & self.agn1],'x',markeredgecolor='r',markersize=8,label='AGN')
        scale=(50*(self.SFR24))
        #print scale[plotflag]
        #print scale[plotflag]
        #plot(dr[plotflag],dv[plotflag],'bo',markersize=6)
        flag=self.apexflag & ~self.agnflag
        scatter(dr[flag],dv[flag],s=scale[flag],c='r',alpha=0.4)
        #plot(dr[flag],dv[flag],'r.')

        
        ymin=3500
        ymax=14000
        axis([0,2,ymin,ymax])
        xmin,xmax=xlim()
        #xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)
        #yticks(arange(4000,ymax,4000,'i'),fontsize=10)
        title(self.clustername,fontsize=12)
        axhline(self.biweightvel,ls='-',color='k')
        axhline(self.biweightvel+3*self.biweightscale,ls='--',color='k')
        axhline(self.biweightvel-3*self.biweightscale,ls='--',color='k')
        axvline(self.r200,ls=':',color='k')

    def plotveldr(self):

        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        dv=(self.supervopt-self.biweightvel)

        membflag=(dr/self.r200deg < 1) & (abs(dv) < 3.*self.biweightscale)

        plot(dr,self.supervopt,'k.',markersize=3)
        plot(dr[membflag],self.supervopt[membflag],'bo',markersize=6)
        ymin=3500
        ymax=14000
        axis([0,3,ymin,ymax])
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)

        yticks(arange(4000,ymax,4000,'i'),fontsize=10)
        title(self.clustername,fontsize=12)
        axhline(self.biweightvel,ls='-',color='r')
        axhline(self.biweightvel+3*self.biweightscale,ls='--',color='r')
        axhline(self.biweightvel-3*self.biweightscale,ls='--',color='r')


    def plotvelhist(self):
        figure()
        bins=30
        x1=self.allvelocity
        (yhist,xhist,patches)=hist(x1,bins)
        xhist=xhist[0:len(xhist)-1]+0.5*(xhist[1]-xhist[0])
        mymean= average(x1)
        mystd=std(x1)
        print mystd
        norm=max(yhist)
        xmin=3000
        xmax=15000
        xplot=arange(xmin,xmax,50)
        y1=norm*exp(-((xplot -self.clustervel)**2)/(2*self.clustersigma**2))
        plot(xplot,y1,'r-')
        xlabel('Recession Velocity ')
        xlim(xmin,xmax)

        axvline(self.clustervel,ymin=0,ymax=60,color='r')



    def plotF24hist(self):
        mybins=arange(0,5000,100)
	y=hist(self.mipsflux[self.apexflag],bins=mybins,histtype='step')
	ngal=y[0]
	x=y[1]
	xbin=zeros(len(ngal))
	for i in range(len(xbin)):
            xbin[i]=0.5*(x[i]+x[i+1])
	#clf()                                                                                       
        self.xbin=xbin
        self.ngal=ngal
	plot(xbin,ngal,'ro')
	errorbar(xbin,ngal,sqrt(ngal),fmt=None)
        axis([0,2000,0,20])
        title(self.clustername)
	ax=gca()
	#ax.set_yscale('log')
	#ax.set_xscale('log')
        #xlabel('24um Flux')

    def plotL24hist(self):
        #figure(1)
        
	y=hist(log10(self.L24[self.apexflag]),bins=10,histtype='stepfilled')
	ax=gca()
	#ax.set_yscale('log')
        axis([6.5,11.,1,22])
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,2,'i'),fontsize=10)
    def plotSFR24hist(self):
        #figure(1)
        flag=self.apexflag & ~self.agnflag & self.membflag
	y=hist(log10(self.SFR24[flag]),bins=8,histtype='step',color='k')
	ax=gca()
	#ax.set_yscale('log')
        axis([-2.5,1.5,0,20])
        xmin,xmax=xlim()
        xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotHImasshist(self):
        #figure(1)
        flag= self.HIflag & self.dvflag
        mybins=arange(8,11,.2)
	y=hist(log10(self.HImass[flag]),bins=mybins,histtype='step',color='k')
	ax=gca()
	#ax.set_yscale('log')
        axis([8,11,0,55])
        xmin,xmax=xlim()
        xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotHIDefhist(self):
        #figure(1)
        flag= self.HIflag & self.sdssflag
        #mybins=arange(8,11,.2)
	y=hist((self.HIDef[flag]),bins=10,histtype='step',color='k')

        flag= self.HIflag &  self.sdssflag
        axvline(x=0,ls='--',color='r')
        #mybins=arange(8,11,.2)
	#y=hist((self.HIDef[flag]),bins=20,histtype='step',color='b')
        ax=gca()
	#ax.set_yscale('log')
        #axis([8,11,0,55])
        xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        axis([-2,2,0,ymax])
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotDiamHImass(self):
        flag= self.HIflag 
        # log10 HI mass on y axis 
	y=log10(self.HImass)
        # some measure of D25 on x axis
        x=self.logD25kpc
        # plot only those galaxies w/HI detection
        flag= self.HIflag & self.sdssflag
        plot(x[flag],y[flag],'.')
        # plot toribio et al relation
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        ax=gca()
        xmin,xmax=xlim()
        xline=arange(xmin,xmax+.1,.1)
        yline=8.72+1.25*(xline-log10(2.))
        plot(xline,yline,'r-')
        xlabel('$\mathrm{log_{10}(Diameter \ (D_{25}/kpc))}$',fontsize=16)
        ylabel('$\mathrm{log_{10}(M_{HI}/M_\odot)}$',fontsize=16)
        title(self.clustername)

    def plotMrHImass(self):
        flag= self.HIflag 
        # log10 HI mass on y axis 
	y=log10(self.HImass)
        # some measure of D25 on x axis
        x=self.sdssMr 
        # plot only those galaxies w/HI detection
        flag= self.HIflag & self.sdssflag
        plot(x[flag],y[flag],'.')
        # plot toribio et al relation
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        ax=gca()
        xmin,xmax=xlim()
        xline=arange(xmin,xmax+.1,.1)
        yline=6.44-0.18*xline
        plot(xline,yline,'r-')
        xlabel('$\mathrm{M_r}$',fontsize=16)
        ylabel('$\mathrm{log_{10}(M_{HI}/M_\odot)}$',fontsize=16)
        title(self.clustername)

    def plotgrHImass(self):
        # log10 HI mass on y axis 
	y=log10(self.HImass)
        # sdss deredened g-r color
        x=self.sdssgmag-self.sdssrmag
        # plot only those galaxies w/HI detection
        flag= self.HIflag & self.sdssflag
        plot(x[flag],y[flag],'.')
        # plot toribio et al relation
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        ax=gca()
        xmin,xmax=xlim()
        xline=arange(xmin,xmax+.1,.1)
        yline=8.84+1.81*xline
        plot(xline,yline,'r-')
        xlabel('$\mathrm{g-r \ magnitude}$',fontsize=16)
        ylabel('$\mathrm{log_{10}(M_{HI}/M_\odot)}$',fontsize=16)
        title(self.clustername)
    def plotratioHImass(self,colors,shapes):
        #figure(1)
        #plotflag=self.apexflag & self.HIflag & ~self.agn2 & self.spiralFlag
        #plotflag=self.apexflag  & ~self.agn2 & self.spiralFlag
        x=self.HImass[self.plotratio90Flag]/self.stellarmass[self.plotratio90Flag]
        y=self.ratio90[self.plotratio90Flag]
        plot(x,y,'ko',color=colors,marker=shapes)
	#ax=gca()
	#ax.set_yscale('log')
        #axis([-2.5,1.5,0,22])
        #xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        #ymin,ymax=ylim()
        #title(self.clustername)
        return x.tolist(),y.tolist()
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotSFR24stellmass(self,colors,shapes):
        #figure(1)
        #plotflag=self.apexflag & self.HIflag & ~self.agn2 & self.spiralFlag
        #plotflag=self.apexflag  & ~self.agn2 & self.spiralFlag

        #flag=self.ellipseflag & self.spiralFlag & self.membflag & ~self.agnflag
        #flag=self.ellipseflag & self.membflag & ~self.agnflag
        flag=self.apexflag & self.membflag & ~self.agnflag
        print self.prefix,' numb of IR members = ',sum(flag)
        x=self.stellarmass[flag]
        y=self.SFR24[flag]
        yerr=self.SFR24err[flag]
        plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=8)
        errorbar(x,y,yerr,fmt=None,color=colors,label='_nolegend_')

        #flag=self.ellipseflag & self.spiralFlag & ~self.membflag & ~self.agn1
        flag=self.apexflag  & ~self.membflag & ~self.agn1
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        x1=self.stellarmass[flag]
        y1=self.SFR24[flag]
        yerr=self.SFR24err[flag]
        if self.prefix.find('MKW11')> -1:
            clabel=self.prefix+' Field'

        else:
            clabel='_nolegend_'
            #colors='0.3'
        #plot(x1,y1,'k.',markeredgecolor=colors,marker=shapes,markerfacecolor=colors,label=clabel,markersize=10,lw=3,alpha=0.2)
        plot(x1,y1,'k.',markeredgecolor=fieldColor,marker=shapes,markerfacecolor=fieldColor,label=clabel,markersize=10,lw=3,alpha=0.2)

	#ax=gca()
	#ax.set_yscale('log')
        #axis([-2.5,1.5,0,22])
        #xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        #ymin,ymax=ylim()
        #title(self.clustername)
        return x.tolist(),y.tolist(),x1.tolist(),y1.tolist()
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotSFR24HIperArea(self,colors,shapes):
        #figure(1)
        #plotflag=self.apexflag & self.HIflag & ~self.agn2 & self.spiralFlag
        #plotflag=self.apexflag  & ~self.agn2 & self.spiralFlag

        #flag=self.ellipseflag & self.spiralFlag & self.membflag & ~self.agnflag
        #flag=self.ellipseflag & self.membflag & ~self.agnflag
        flag=self.apexflag & self.dvflag & ~self.agnflag
        x=self.HImass[flag]/(self.sdssPetroR90r[flag]**2)
        y=self.SFR24[flag]
        yerr=self.SFR24err[flag]
        plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=8)
        errorbar(x,y,yerr,fmt=None,color=colors,label='_nolegend_')

        #flag=self.ellipseflag & self.spiralFlag & ~self.membflag & ~self.agn1
        flag=self.apexflag  & ~self.membflag & ~self.agnflag
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        x1=self.HImass[flag]/(self.sdssPetroR90r[flag]**2)
        y1=self.SFR24[flag]
        yerr=self.SFR24err[flag]
        if self.prefix.find('MKW11')> -1:
            clabel=self.prefix+' Field'

        else:
            clabel='_nolegend_'
            #colors='0.3'
        #plot(x1,y1,'k.',markeredgecolor=colors,marker=shapes,markerfacecolor=colors,label=clabel,markersize=10,lw=3,alpha=0.2)
        plot(x1,y1,'k.',markeredgecolor=fieldColor,marker=shapes,markerfacecolor=fieldColor,label=clabel,markersize=10,lw=3,alpha=0.2)

        return x.tolist(),y.tolist(),x1.tolist(),y1.tolist()

    def checkmorph(self):#print out files that our class and burstein type disagree or no burstein type
        flag=(self.morph==3)
        self.summerSpirals=self.agcnumber[flag]
        bflag=((self.bsteintype>=120)&(self.bsteintype<183))|((self.bsteintype>=300)&(self.bsteintype<400)) 
        self.bflag=bflag
        funnyflag= ((~(flag) & bflag) | (flag & ~bflag)) & self.morphflag & (self.bsteintype > 0)
        self.funnySpirals=self.agcnumber[funnyflag]
        self.ourtype=self.morph[funnyflag]
        self.theirtype=self.bsteintype[funnyflag]
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.funnySpirals.dat'
        outfile=open(s,'w')
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.funnySpirals'
        outfile2=open(s,'w')
        for i in range(len(self.funnySpirals)):
            if  self.prefix in 'Coma Hercules A1367':
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss.fits \n'
            else:
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss-g.fits \n'
            outfile.write(name)
            name=str(self.funnySpirals[i])+' '+str(self.ourtype[i])+' '+str(self.theirtype[i])+' \n'
            outfile2.write(name)
        outfile.close()
        outfile2.close()


        nobflag= ((self.bsteintype == 0) & (self.On24ImageFlag))
        self.funnySpirals=self.agcnumber[nobflag]
        self.ourtype=self.morph[nobflag]
        self.theirtype=self.bsteintype[nobflag]
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noBsteinSpirals.dat'
        outfile=open(s,'w')
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noBsteinSpirals'
        outfile2=open(s,'w')
        for i in range(len(self.funnySpirals)):
            if  self.prefix in 'Coma Hercules A1367':
                
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss.fits \n'
            else:
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss-g.fits \n'
            outfile.write(name)
            name=str(self.funnySpirals[i])+' '+str(self.ourtype[i])+' '+str(self.theirtype[i])+' \n'
            outfile2.write(name)
        outfile.close()
        outfile2.close()

        missmorphflag= ((self.morphflag == 0) & (self.On24ImageFlag))
        self.funnySpirals=self.agcnumber[missmorphflag]
        self.ourtype=self.morph[missmorphflag]
        self.theirtype=self.bsteintype[missmorphflag]
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noMorph.dat'
        outfile=open(s,'w')
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noMorph'
        outfile2=open(s,'w')
        for i in range(len(self.funnySpirals)):
            if  self.prefix in 'Coma Hercules A1367':
                
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss.fits \n'
            else:
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss-g.fits \n'
            outfile.write(name)
            name=str(self.funnySpirals[i])+' '+str(self.ourtype[i])+' '+str(self.theirtype[i])+' \n'
            outfile2.write(name)
        outfile.close()
        outfile2.close()

    def plotsSFRLocaldens(self,colors,shapes): 
        plotflag=self.ellipseflag & self.spiralFlag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        sSFR=self.SuperSFR24[plotflag]/self.stellarmass[plotflag]
        plot(self.localdens[plotflag],sSFR,'ko',marker=shapes,color=colors,label=self.prefix,markersize=10)
        x=self.localdens[plotflag].tolist()
        y=sSFR.tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralFlag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.localdens[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        print 'min local dens = ',min(self.localdens[plotflag])
        #x=x+self.localdens[plotflag].tolist()
        #y=y+self.sSFR[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y

    def plotsSFRHIDef(self,colors,shapes): 
        plotflag=self.On24ImageFlag #& ~self.agnflag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        sSFR=self.SuperSFR24[plotflag]/self.stellarmass[plotflag]*1.e9
        x=self.HIDef[plotflag]
        plot(x,sSFR,'ko',marker=shapes,color=colors,label=self.prefix,markersize=10)
        x=x.tolist()
        y=sSFR.tolist()
        return x,y

    def plotsSFRratioLocaldens(self,colors,shapes): 
        plotflag=self.ellipseflag & self.spiralFlag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        
        ry=self.sSFR90-self.sSFR50
        scatter(self.localdens[plotflag],ry[plotflag],marker=shapes,color=colors)
        x=self.localdens[plotflag].tolist()
        y=ry[plotflag].tolist()
        #print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralFlag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.localdens[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        #print 'min local dens = ',min(self.localdens[plotflag])
        #x=x+self.localdens[plotflag].tolist()
        #y=y+self.sSFR[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y

    def plotsSFRLocaldens2(self,colors,shapes): 
        plotflag=self.ellipseflag & self.spiralFlag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        
        scatter(self.localdens[plotflag],self.sSFR[plotflag],marker=shapes,color=colors)
        x=self.localdens[plotflag].tolist()
        y=self.sSFR[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralFlag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.localdens[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        print 'min local dens = ',min(self.localdens[plotflag])
        x=x+self.localdens[plotflag].tolist()
        y=y+self.sSFR[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y

    def plotRatiosDr(self,colors,shapes): 
        plotflag=self.plotratio90Flag
        plotflag=self.ellipseflag & self.spiraflag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        ry=self.sSFR90-self.sSFR50
        scatter(self.dr[plotflag],y[plotflag],marker=shapes,color=colors)
        x=self.dr[plotflag].tolist()
        y=ry[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralFlag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.dr[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        #x=x+self.dr[plotflag].tolist()
        #y=y+self.ratio90[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y
    def plotRatiosDr200(self,colors,shapes): 
        #plotflag=self.plotratio90Flag
        plotflag=self.ellipseflag & self.spiralFlag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        ry=self.sSFR90-self.sSFR50
        scatter(self.drR200[plotflag],ry[plotflag],marker=shapes,color=colors,label=self.prefix)
        x=self.drR200[plotflag].tolist()
        y=ry[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralFlag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.drR200[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        #x=x+self.drR200[plotflag].tolist()
        #y=y+self.ratio90[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y
    def plotRatiosStellarMass(self,colors,shapes): 
        plotflag=self.plotratio90Flag
        scatter(self.stellarmass[plotflag],self.ratio90[plotflag],marker=shapes,color=colors)
        x=self.stellarmass[plotflag].tolist()
        y=self.ratio90[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        plotflag=(self.spiralFlag & self.On24ImageFlag & ~self.apexflag)
        scatter(self.stellarmass[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        print 'min local dens = ',min(self.localdens[plotflag])
        x=x+self.stellarmass[plotflag].tolist()
        y=y+self.ratio90[plotflag].tolist()
        return x,y
    def plotsSFR(self):
        #figure()
        title(self.clustername)
        #sSFR90=(self.mipsF90-self.mipsF50)/(self.SDSSF90-self.SDSSF50)
        #sSFR50=sSFR50[self.ellipseflag]
        #sSFR90=sSFR90[self.ellipseflag]

        flag=self.ellipseflag
        plot(self.sSFR50[flag],self.sSFR90[flag],'k.',label='All')
        errorbar(self.sSFR50[flag],self.sSFR90[flag],xerr=self.sSFR50err[flag],yerr=self.sSFR90err[flag],fmt=None,label='no_label')
        m,b=polyfit(self.sSFR50[flag],self.sSFR90[flag],1)
        print m,b
        #scatter(sSFR50[flag],sSFR90[flag],s=1.*(self.localdens[flag]),c='c',alpha=0.7,label='$\Sigma$')
        flag=self.ellipseflag & self.spiralFlag
        plot(self.sSFR50[flag],self.sSFR90[flag],'bo',markeredgecolor='b',markerfacecolor='None',label='Field Spirals',markersize=8)
        flag=self.ellipseflag & self.spiralFlag& self.membflag
        plot(self.sSFR50[flag],self.sSFR90[flag],'b^',markersize=10,label='Cluster Spirals')
        flag=self.ellipseflag & self.HIflag
        if sum(flag) > 1:
            scatter(self.sSFR50[flag],self.sSFR90[flag],s=1.*(self.flux100[flag]),c='r',alpha=0.7,label='HI')
        xlabel('$F_{24}/F_r(r<R_{50})$',fontsize=16)
        ylabel('$F_{24}/F_r(r<R_{90})$',fontsize=16)
        legend(loc='upper left',numpoints=1,scatterpoints=1,markerscale=1)
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.001)
        plot(xl,xl,'k-')
        plot(xl,m*xl,'g')

    def sSFRnew(self):
        figure()
        title(self.clustername)
        #sSFR90=(self.mipsF90-self.mipsF50)/(self.SDSSF90-self.SDSSF50)
        #sSFR50=sSFR50[self.ellipseflag]
        #sSFR90=sSFR90[self.ellipseflag]

        flag=self.ellipseflag
        plot(self.sSFR50[flag],self.sSFR5090[flag],'k.',label='All')
        errorbar(self.sSFR50[flag],self.sSFR5090[flag],xerr=self.sSFR50err[flag],yerr=self.sSFR90err[flag],fmt=None,label='no_label')
        m,b=polyfit(self.sSFR50[flag],self.sSFR5090[flag],1)
        print m,b
        #scatter(sSFR50[flag],sSFR90[flag],s=1.*(self.localdens[flag]),c='c',alpha=0.7,label='$\Sigma$')
        flag=self.ellipseflag & self.spiralFlag
        plot(self.sSFR50[flag],self.sSFR5090[flag],'bo',markeredgecolor='b',markerfacecolor='None',label='Field Spirals',markersize=8)
        flag=self.ellipseflag & self.spiralFlag& self.membflag
        plot(self.sSFR50[flag],self.sSFR5090[flag],'b^',markersize=10,label='Cluster Spirals')
        flag=self.ellipseflag & self.HIflag
        if sum(flag) > 1:
            scatter(self.sSFR50[flag],self.sSFR5090[flag],s=1.*(self.flux100[flag]),c='r',alpha=0.7,label='HI')
        xlabel('$F_{24}/F_r(r<R_{50})$',fontsize=16)
        ylabel('$F_{24}/F_r(R_{50} <r<R_{90})$',fontsize=16)
        legend(loc='upper left',numpoints=1,scatterpoints=1,markerscale=1)
        axis([-.01,.02,-.01,.02])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.001)
        plot(xl,xl,'k-')
        plot(xl,m*xl,'g')
        axis([-.01,.02,-.01,.02])
        ax=gca()
        ax.set_yscale('log')
        ax.set_xscale('log')

    def integratedHI(self):
        flag=self.membflag & self.HIflag
        self.sumHIflux=sum(self.flux100[flag]/100.)
        print 'sum of HI flux = ',self.sumHIflux
        #scaled to z=0.4
        #ratio=self.dL

    def optiroffset(self):#plot X-ray contours w/opt-ir distances overlaid
        self.optirdist=sqrt((self.sdssra-self.mipsra)**2+(self.sdssdec-self.mipsdec)**2)
        maxdist=6./3600 #6 arcsec
        flag=self.apexflag & (self.optirdist < maxdist) #& self.On24ImageFlag 
        #figure()
        rimage='/home/share/research/LocalClusters/ColorImages/'+self.prefix+'/'+self.prefix+'-r.fits'
        ximage='/home/share/research/LocalClusters/ROSATimages/'+self.prefix+'.RASS.3.75x3.75.fits'
        #mylevels=[2.6,3.6,4.,5,7]
        gc=aplpy.FITSFigure(rimage)
        level=xraycontourlevels[self.prefix]
        gc.show_contour(ximage,layer='marker_set_1',smooth=4,colors='black',levels=level)
        gc.show_markers(self.sdssra[flag],self.sdssdec[flag],facecolor='red',marker='o',edgecolor='None',s=self.optirdist[flag]*3600*10)
        return gc
    def dalevschary(self):
        figure(figsize=[10,10])
        flag=self.apexflag & self.sdssflag
        redshift=self.cz*ones(len(flag))  # convert recession velocity to redshift
        ceLir,ceSFR=chary.chary_elbaz_24um(redshift[flag],self.mipsflux[flag])
        subplot(2,1,1)
        plot(self.mipsflux[flag],self.Lir[flag],'bo',self.mipsflux[flag],ceLir,'ro')
        ylabel('IR Luminosity')
        title('Comparison of Lir from Dale & Helou and Chary & Elbaz')
        axvline(x=671,ls='--')
        legend(['Dale & Helou','Chary & Elbaz','lower limit of C&E templates'],numpoints=1,loc='upper left')
        gca().set_yscale('log')
        gca().set_xscale('log')

        subplot(2,1,2)
        a=self.Lir[flag]
        b=ceLir
        percentdiff=(b-a)/(.5*(a+b))*100
        plot(self.mipsflux[flag],percentdiff,'go')
        axvline(x=671,ls='--')
        axhline(y=0,ls='-',color='k')
        gca().set_xscale('log')
        xlabel('MIPS flux (micro-Jy)')
        ylabel('% Difference')
        # compare SFRs
        figure()
        subplot(2,1,1)
        plot(self.mipsflux[flag],self.SFR24[flag],'bo',self.mipsflux[flag],ceSFR,'ro')
        ylabel('IR SFR')
        title('Comparison of IR SFR from Dale & Helou and Chary & Elbaz')
        axvline(x=671,ls='--')
        legend(['Dale & Helou','Chary & Elbaz','lower limit of C&E templates'],numpoints=1,loc='upper left')
        gca().set_yscale('log')
        gca().set_xscale('log')
        xlabel('MIPS-24 Flux (micro-Jy)')

        subplot(2,1,2)
        a=self.SFR24[flag]
        b=ceSFR
        percentdiff=(b-a)/(.5*(a+b))*100
        plot(self.stellarmass[flag],a,'bo',self.stellarmass[flag],b,'ro')
        # plot slope of one line from Elbaz et al 2011
        xl=arange(9,11.7,.1)
        xl=10.**xl
        ye=(xl/4.e9)
        textdx=2.e7
        plot(xl,ye,color='c',ls='-')
        text(xl[0]-textdx,ye[0],'$\mathrm{Elbaz \ z=0}$',color='c',fontsize=16,horizontalalignment='right')

        gca().set_xscale('log')
        gca().set_yscale('log')
        xlabel('Stellar Mass')
        ylabel('IR SFR')
        

#def sSFRlocaldens(self):
        
def plotF24histall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotF24hist()
    ax=gca()
    text(-.75,-.35,'$log_{10}(F_{24} \  (\mu Jy))$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotF24histAll.png')

def plotSFR24histall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotSFR24hist()
    ax=gca()
    text(-.75,-.35,'$log_{10}(SFR_{24} \  (M_\odot/yr))$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotSFR24histAll.png')

def plotL24histall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        cl.plotL24hist()
    ax=gca()
    text(-.75,-.35,'$log_{10}(L_{24}/L_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotL24histAll.png')


def plotHImasshistall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        cl.plotHImasshist()
    ax=gca()
    text(-.75,-.35,'$HI \ Mass$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'HImasshistAll.png')

def plotHIDefhistall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        try:
            cl.plotHIDefhist()
        except:
            title(cl.prefix)
            continue
    ax=gca()
    text(-.75,-.35,'$HI \ Deficiency$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'HIDefhistAll.png')

def plotpositionsall():
    figure(figsize=(15,10))
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotpositions()
    ax=gca()
    text(-.75,-.35,'RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotPositionsAll.png')


def plotpositionson24all():
    figure(figsize=[12,12])
    clf()
    subplots_adjust(wspace=.2,hspace=.2)
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        print cl.prefix, cl.clusterra,cl.clusterdec
        #cl.plotpositionson24WithRatio()
        #cl.plotpositionson24()
        cl.plotpositionson24(cl.clusterra,cl.clusterdec)
        #cl.plotrelativepositionson24()
    ax=gca()
    text(-.75,-.35,'RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotPositionsOn24All.eps')

def plotrelativepositionson24all():
    figure(figsize=[9,9])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotrelativepositionson24()
    leg=legend(numpoints=1)#,fontsize=12)
    for t in leg.get_texts():
        t.set_fontsize('small')
    ax=gca()
    text(-.75,-.35,'$\Delta$RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$\Delta$Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotRelativePositionsOn24All.eps')

def plotveldrall():
    figure(figsize=[9,6])
    clf()
    subplots_adjust(wspace=.35,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotveldr()
    ax=gca()
    text(-.75,-.35,'dr (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-3.1,1.9,'$V_r$ (km/s)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotVeldrAll.png')

def plotdvdron24all():
    figure(figsize=[12,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.25)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        print cl.prefix
        cl.plotdvdron24()
    ax=gca()
    text(-.75,-.35,'$\Delta$r (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.9,1.9,'$V_r$ (km/s)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotdvdrAllOn24.eps')

def checkmorphall():
    for cl in mylocalclusters:
        cl.checkmorph()

def getSpirals():
    figure()
    clf()
    i=0
    for cl in mylocalclusters:
        i += 1
        cl.getFilesForProfileFitting()

def plotallsSFRratioLocaldens():
    figure(figsize=[15,10])
    clf()
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRratioLocaldens(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    xlabel('$Local \  Density $',fontsize=18)
    ylabel('$sSFR_{90} - sSFR_{50} $',fontsize=18)
    f=figuredir+'sSFRLocaldens.png'
    savefig(f)

def plotallsSFRHIDef():
    figure(figsize=[15,10])
    clf()
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRHIDef(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    axvline(x=0,ls='--',color='k')
    axhline(y=0.2,ls='--',color='b')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_yscale('log')
    xlabel('$HI \ Deficiency $',fontsize=18)
    ylabel('$sSFR \ (M_\odot \ yr^{-1} / 10^9 M_\odot )$',fontsize=18)
    legend(numpoints=1)
    f=figuredir+'sSFRHIDef.png'
    savefig(f)

def plotallRatiosLocaldens():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRLocaldens2(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlabel('$Local \  Density $',fontsize=18)
    ylabel('$SFR/M_* (M_\odot/yr/M_\odot)$',fontsize=18)
    f=figuredir+'sSFR2Localdens.png'
    savefig(f)

def plotallsSFRLocaldens():
    figure(figsize=[15,10])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRLocaldens(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    legend(numpoints=1,loc='upper right')
    xbin,ybin,ybinerr=my.binit(xall,yall,9)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    #axis([.9,1000,-.05e-10,.85e-9])
    xlabel('$Local \  Density $',fontsize=20)
    ylabel('$SFR/M_* \ (M_\odot/yr/M_\odot)$',fontsize=20)
    f=figuredir+'sSFRLocaldens.png'
    savefig(f)

def plotallRatiosDr():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotRatiosDr(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    axis([.9,1700,-.5,8])
    xlabel('$Local \  Density $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosLocaldens.png'
    savefig(f)
def plotallRatiosDr():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotRatiosDr(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,9)
    plot(xbin,ybin,'k:')
    print yall
    ax=gca()
    axis([-.1,2.,-.5,7])
    xlabel('$\Delta r \ (deg)$',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figurdir+'RatiosDr.png'
    savefig(f)

def plotallRatiosDr200():
    figure(figsize=[15,10])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotRatiosDr200(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    legend(loc='upper right',scatterpoints=1)
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    ax=gca()
    xlabel('$\Delta r / R_{200} $',fontsize=18)
    ylabel('$sSFR_{90} - sSFR_{50}$',fontsize=18)
    f=figuredir+'RatiosDr200.png'
    savefig(f)


def plotallRatiosSize():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=awm4
        if i == 3:
            cl=mkw8
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = herc
        if i == 8:
            cl = a1367
        if i == 9:
            cl = coma
        #subplot(3,3,i)
        x,y=cl.plotRatiosStellarMass(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,7)
    #plot(xbin,ybin,'k--')
    ax=gca()
    ax.set_xscale('log')
    axis([1.e9,2.e12,-.5,8])
    xlabel('$Stellar \ Mass (M_\odot) $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosStellarMass.png'
    savefig(f)

def plotallRatiosStellarMass():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=awm4
        if i == 3:
            cl=mkw8
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = herc
        if i == 8:
            cl = a1367
        if i == 9:
            cl = coma
        #subplot(3,3,i)
        x,y=cl.plotRatiosStellarMass(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,7)
    plot(xbin,ybin,'k--')
    ax=gca()
    ax.set_xscale('log')
    axis([1.e9,2.e12,-.5,8])
    xlabel('$Stellar \ Mass (M_\odot) $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosStellarMass.png'
    savefig(f)

def plotallRatiosHIStellarMass():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','+','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotratioHImass(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,7)
    plot(xbin,ybin,'k--')
    ax=gca()
    xlabel('$HI \ Mass / Stellar \ Mass  $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosHIStellarMass.png'
    savefig(f)

def plotSFR24StellarMassall():
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    #colors=['k','k','k','k','k','k','k','k','k']
    shapes=['o','^','s','*','p','d','v','h','<']
    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y,x1,y1=cl.plotSFR24stellmass(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
        x1all=x1all+x1#field galaxies
        y1all=y1all+y1
    legend(loc='upper left',numpoints=1)#labels=clusternames)
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k')
    print xbin,ybin,ybinerr
    #xbin,ybin,ybinerr=my.biniterr(xall,yall,5)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    print xbin,ybin,ybinerr
    plot(xbin,ybin,'k--')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k', lw=3)
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    #axis([8.e8,2e12,.002,70])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    xlabel('$Stellar \ Mass \ (M_\odot)$',fontsize=20)
    ylabel('$SFR \ (M_\odot/yr)$',fontsize=20)
    f=figuredir+'SFR24StellarMass.eps'
    savefig(f)

def plotSFR24HIperAreaall():
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    colors=['k','k','k','k','k','k','k','k','k']
    shapes=['o','^','s','*','p','d','v','h','<']
    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y,x1,y1=cl.plotSFR24HIperArea(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
        x1all=x1all+x1#field galaxies
        y1all=y1all+y1
    legend(loc='upper right',numpoints=1)#labels=clusternames)
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k')
    print xbin,ybin,ybinerr
    #xbin,ybin,ybinerr=my.biniterr(xall,yall,5)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    print xbin,ybin,ybinerr
    plot(xbin,ybin,'k--')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k', lw=3)
    ax=gca()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #axis([8.e8,2e12,.002,70])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    xlabel('$HI \ Mass \ / Area (M_\odot/arcsec^2)$',fontsize=20)
    ylabel('$SFR \ (M_\odot/yr)$',fontsize=20)
    f=figuredir+'SFR24HIperArea.eps'
    savefig(f)


def plotsSFRall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotsSFR()
    ax=gca()
    text(-.75,-.35,'$F_{24}/F_r(r<R_{50})$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$F_{24}/F_r(r<R_{90})$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotsSFRAll.png')


def plotcompareR90all():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.compareR90()
    ax=gca()
    text(-.75,-.35,'$r-band \ radius$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,r'$24 \ radius$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(figuredir+'PlotsCompareR90All.png')

def plotcomparediffR90localdensall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.comparediffR90localdens()
    ax=gca()
    text(-.75,-.35,'$r-band \ radius$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,r'$24 \ radius$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(figuredir+'PlotsCompareR90All.png')

def plotabellclusters():
    figure(figsize=(12,10))
    a2052.plotpositionson24()
    legend(numpoints=1,scatterpoints=1)
    a2063.plotpositionson24()
    axis([228,232,5,11])

def plotoptirdistXray():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    for cl in mylocalclusters:
        #subplot(3,3,i)
        cl.optiroffset()
        figfile=figuredir+cl.prefix+'OptIRDistXrayContour.png'
        savefig(figfile)
    #ax=gca()
    #text(-.75,-.35,'$r-band \ radius$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    #subplot(3,3,4)
    #text(-2.8,1.9,r'$24 \ radius$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)

def plotlcscolormag():
    figure(figsize=(15,12))
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i=i+1
        subplot(3,3,i)
        cl.plotcolormag()
    ax=gca()
    text(-.75,-.35,'$M_r$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.8,1.9,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
    savefig(figuredir+'LCScolormag.eps')

def plotHImassRelations():
    figure(figsize=(12,10))
    subplots_adjust(wspace=.25,hspace=.35)
    clf()
    subplot(2,2,1)
    for cl in mylocalclusters:
        flag=cl.HIflag
        plot(cl.sdssvopt[flag]/1000,cl.HImass[flag],marker='.',ls='None',label=cl.prefix)
    xlabel('$\mathrm{Recession \ Velocity / (1000 \ km/s)}$',fontsize=16)
    ylabel('$\mathrm{HI \ mass \ (M_\odot)}$',fontsize=16)
    title('$\mathrm{HI mass vs redshift}$')
    ax=gca()
    ax.set_yscale('log')
    #legend(numpoints=1)
    subplot(2,2,2)
    for cl in mylocalclusters:
        cl.plotDiamHImass()
    title('$\mathrm{HI \ mass \ vs \ Diameter}$')
    #compare different mass estimates from Toribio
    subplot(2,2,3)
    for cl in mylocalclusters:
        cl.plotMrHImass()
    title('$\mathrm{HI mass vs Abs r Mag')
    subplot(2,2,4)
    for cl in mylocalclusters:
        cl.plotgrHImass()
    title('HI mass vs g-r color')
    
mkw11=cluster('MKW11')

#mkw11.loadProfileFits('MKW11')
mkw8=cluster('MKW8')
awm4=cluster('AWM4')
a2052=cluster('A2052')
a2063=cluster('A2063')
ngc=cluster('NGC6107')
coma=cluster('Coma')
herc=cluster('Hercules')
a1367=cluster('A1367')

mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc,coma,herc,a1367]
#mylocalclusters=[mkw11,mkw8,awm4]

#plotpositionsall()
#plotpositionson24all()
#plotveldrall()
#getSpirals()
#mkw11.fitprofiles()

#plotallRatiosLocaldens()


