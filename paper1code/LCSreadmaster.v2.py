#!/usr/bin/env python
import pyfits
from LCScommon import *
from pylab import *
import os
import mystuff as my

#these correpond to area w/more uniform covereage
MKW824um=array([220.16377,3.4883817,1.3137727,2.5,12.7456],'f')
MKW1124um=array([202.36305,11.746882,1.2454248,2.9,206.4],'f')
NGC24um=array([244.30994,34.933704,1.2865442,2.5,321.317],'f')


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

def calcC90(x,y,yerr,fluxencl):#radius("),intensity,error,
    #find average using last 3 pts, which should approximate sky
    sky=mean(y[len(y)-3:len(y)])
    #subtract sky from y
    sy=y-sky
    #multiply y by r**2 to account for increasing area
    toty=y*(x**2)
    #sum r=0-36arcsec to get total flux
    totygood=toty[0:len(y)-3]
    rgood=x[0:len(y)-3]
    totflux=sum(totygood)
    #start summing from 39arcsec inward, until it reaches 10% of total flux
    sum10=0
    tenpercent=.1*totflux
    thirty=.3*totflux
    for i in range(len(totygood)):
        index=len(totygood)-1-i
        sum10 += totygood[index]
        #print sky,sum10,tenpercent,totflux
        if sum10 > tenpercent:
            r90=rgood[index]
            break

    #calculate r30
    sum30=0
    for i in range(len(totygood)):
        sum30 += totygood[i]
        if sum30 > thirty:
            r30=rgood[i]
            break

    ##calculate r90 using enclosed flux array

    #find max of array
    maxEncFlux=max(fluxencl)
    indexMax=where((fluxencl == maxEncFlux))
    #break index out of array and into a plain integer
    indexMax=indexMax[0]
    #use index of max of array and move inward until value is .9 max
    transitionFlux=0.9*maxEncFlux
    for i in range(indexMax):
        index=indexMax-i
        if fluxencl[index] < transitionFlux:
            r90FromEncFlux=x[index]
            break
    #calculate r30FromEncFlux
    transitionFlux=0.3*maxEncFlux
    for i in range(indexMax):
        index=i
        if fluxencl[index] > transitionFlux:
            r30FromEncFlux=x[index]
            break
    #return radius at C90 (the radius that encloses 90% of the light) and sky
    try:
        return r90,sky,r90FromEncFlux,maxEncFlux,r30,r30FromEncFlux
    except UnboundLocalError:
        print "Warning: Could not find R90"
        try:
            return 0,sky,r90FromEncFlux,maxEncFlux,r30,r30FromEncFlux
        except UnboundLocalError:
            print "Warning: Could not find R30 From Enc Flux"
            return 0,sky,r90FromEncFlux,maxEncFlux,r30,0

class cluster:
    def __init__(self,clustername):
#Get current path so program can tell if this is being run on Becky or Rose's computer
	self.prefix=clustername
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]
	self.biweightvel=clustercbi[self.prefix]
	self.biweightscale=clustersbi[self.prefix]
	self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. # in Mpc
        self.r200deg=self.r200*1000./my.DA(self.cz,h)/3600.

        mypath=os.getcwd()
        if mypath.find('Users') > -1:
            print "Running on Rose's mac pro"
            infile='/Users/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.fits'
            homedir='/Users/rfinn/'
        elif mypath.find('home') > -1:
            print "Running on coma"
            infile='/home/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.fits'
            homedir='/home/rfinn/'

        #infile='/home/rfinn/LocalClusters/MasterTables/'+clustername+'mastertable.fits'
        tb=pyfits.open(infile)
        tbdata=tb[1].data
        tb.close()
        self.agcflag=tbdata.field('AGCflag')
        self.HIflag=tbdata.field('HIFLAG')
        self.sdssflag=tbdata.field('SDSSflag')
        self.sdssphotflag=tbdata.field('SDSSphotflag')
        self.mpaflag=tbdata.field('MPAFLAG')
        self.apexflag=tbdata.field('APEXFLAG')
        self.sexsdssflag=tbdata.field('SEXSDSSflag')
        self.sex24flag=tbdata.field('SEX24FLAG')
        self.agcvoptflag=tbdata.field('AGCVOPTFLAG')

        self.agcnumber=tbdata.field('AGCNUMBER')
        self.raagc=tbdata.field('AGCRA')
        self.decagc=tbdata.field('AGCDEC')
        self.a100=tbdata.field('A100')
        self.b100=tbdata.field('B100')
        self.mag10=tbdata.field('MAG10')
        self.posang=tbdata.field('POSANG')
        self.bsteintype=tbdata.field('BSTEINTYPE')
        self.vopt=tbdata.field('VOPT')
        self.verr=tbdata.field('VERR')
        self.vsource=tbdata.field('VSOURCE')
        self.flux100=tbdata.field('FLUX100')
        self.rms100=tbdata.field('RMS100')
        self.v21=tbdata.field('V21')
        self.width=tbdata.field('WIDTH')
        self.widtherr=tbdata.field('WIDTHERR')
        #sdss info
        self.sdssra=tbdata.field('SDSSRA')
        self.sdssdec=tbdata.field('SDSSDEC')
        self.sdssphotra=tbdata.field('SDSSphotRA')
        self.sdssphotdec=tbdata.field('SDSSphotDEC')
        self.sdssu=tbdata.field('SDSSU')
        self.sdssg=tbdata.field('SDSSG')
        self.sdssr=tbdata.field('SDSSR')
        self.sdssi=tbdata.field('SDSSI')
        self.sdssz=tbdata.field('SDSSZ')
        self.sdssspecz=tbdata.field('SDSSSPECZ')
        self.sdssvopt=tbdata.field('SDSSVOPT')
        self.sdsshaew=tbdata.field('SDSSHAEW')
        self.sdsshaewerr=tbdata.field('SDSSHAEWERR')
        self.sdssplate=tbdata.field('SDSSPLATE')
        self.sdssfiberid=tbdata.field('SDSSFIBERID')
        self.sdsstile=tbdata.field('SDSSTILE')
        self.mpahalpha=tbdata.field('MPAHALPHA')
        self.mpahbeta=tbdata.field('MPAHBETA')
        self.mpao3=tbdata.field('MPAOIII')
        self.mpan2=tbdata.field('MPANII')
        #sextractor info
        self.numberser=tbdata.field('NUMBERSER')
        self.ximageser=tbdata.field('XIMAGESER')
        self.yimageser=tbdata.field('YIMAGESER')
        self.xminimageser=tbdata.field('XMINIMAGESER')
        self.xmaximageser=tbdata.field('XMAXIMAGESER')
        self.yminimageser=tbdata.field('YMINIMAGESER')
        self.raser=tbdata.field('RASER')
        self.decser=tbdata.field('DECSER')
        self.fluxisoser=tbdata.field('FLUXISOSER')
        self.fluxerrisoser=tbdata.field('FLUXERRISOSER')
        self.magisoser=tbdata.field('MAGISOSER')
        self.magerrisoser=tbdata.field('MAGERRISOSER')
        self.fluxautoser=tbdata.field('FLUXAUTOSER')
        self.fluxerrautoser=tbdata.field('FLUXERRAUTOSER')
        self.magautoser=tbdata.field('MAGAUTOSER')
        self.magerrautoser=tbdata.field('MAGERRAUTOSER')
        self.fluxpetroser=tbdata.field('FLUXPETROSER')
        self.fluxerrpetroser=tbdata.field('FLUXERRPETROSER')
        self.magpetroser=tbdata.field('MAGPETROSER')
        self.magerrpetroser=tbdata.field('MAGERRPETROSER')
        self.kronradser=tbdata.field('KRONRADSER')#kron radius
        self.petroradser=tbdata.field('PETRORADSER')#petrosian radius
        self.fluxradser=tbdata.field('FLUXRADSER')#1/2 light radius
        self.isoareaser=tbdata.field('ISOAREASER')
        self.aworldser=tbdata.field('AWORLDSER')
        self.bworldser=tbdata.field('BWORLDSER')
        self.thetaser=tbdata.field('THETASER')
        self.errthetaser=tbdata.field('ERRTHETASER')
        self.thetaj2000ser=tbdata.field('THETAJ2000SER')
        self.errthetaj2000ser=tbdata.field('ERRTHETAJ2000SER')
        self.elongser=tbdata.field('ELONGATIONSER')
        self.elliptser=tbdata.field('ELLIPTICITYSER')
        self.fwhmser=tbdata.field('FWHMSER')
        self.flagsser=tbdata.field('FLAGSSER')
        self.classstarser=tbdata.field('CLASSSTARSER')
        #SEXTRACTOR  output 24 micron data
        self.numberse24=tbdata.field('NUMBERSE24')
        self.ximagese24=tbdata.field('XIMAGESE24')
        self.yimagese24=tbdata.field('YIMAGESE24')
        self.xminimagese24=tbdata.field('XMINIMAGESE24')
        self.xmaximagese24=tbdata.field('XMAXIMAGESE24')
        self.xminimagese24=tbdata.field('YMINIMAGESE24')
        self.rase24=tbdata.field('RASE24')
        self.decse24=tbdata.field('DECSE24')
        self.fluxisose24=tbdata.field('FLUXISOSE24')
        self.fluxerrisose24=tbdata.field('FLUXERRISOSE24')
        self.magisose24=tbdata.field('MAGISOSE24')
        self.magerrisose24=tbdata.field('MAGERRISOSE24')
        self.fluxautose24=tbdata.field('FLUXAUTOSE24')
        self.fluxerrautose24=tbdata.field('FLUXERRAUTOSE24')
        self.magautose24=tbdata.field('MAGAUTOSE24')
        self.magerrautose24=tbdata.field('MAGERRAUTOSE24')
        self.fluxpetrose24=tbdata.field('FLUXPETROSE24')
        self.fluxerrpetrose24=tbdata.field('FLUXERRPETROSE24')
        self.magpetrose24=tbdata.field('MAGPETROSE24')
        self.magerrpetrose24=tbdata.field('MAGERRPETROSE24')
        self.kronradse24=tbdata.field('KRONRADSE24')
        self.petroradse24=tbdata.field('PETRORADSE24')
        self.fluxradse24=tbdata.field('FLUXRADSE24')
        self.isoarease24=tbdata.field('ISOAREASE24')
        self.aworldse24=tbdata.field('AWORLDSE24')
        self.bworldse24=tbdata.field('BWORLDSE24')
        self.thetase24=tbdata.field('THETASE24')
        self.errthetase24=tbdata.field('ERRTHETASE24')
        self.thetaj2000se24=tbdata.field('THETAJ2000SE24')
        self.errthetaj2000se24=tbdata.field('ERRTHETAJ2000SE24')
        self.elongse24=tbdata.field('ELONGATIONSE24')
        self.elliptse24=tbdata.field('ELLIPTICITYSE24')
        self.fwhmse24=tbdata.field('FWHMSE24')
        self.flagsse24=tbdata.field('FLAGSSE24')
        self.classstarse24=tbdata.field('CLASSSTARSE24')
        self.f24dist=self.fluxautose24[self.sex24flag]
        #apex output
        self.mipsra=tbdata.field('MIPSRA')
        self.mipsdec=tbdata.field('MIPSDEC')
        self.mipsflux=tbdata.field('MIPSFLUX')
        self.mipsfluxerr=tbdata.field('MIPSFLUXERR')
        self.mipssnr=tbdata.field('MIPSSNR')
        self.mipsdeblend=tbdata.field('MIPSDEBLEND')
        self.mipsfluxap1=tbdata.field('MIPSFLUXAP1')
        self.mipsfluxap1err=tbdata.field('MIPSFLUXAP1ERR')
        self.mipsfluxap2=tbdata.field('MIPSFLUXAP2')
        self.mipsfluxap2err=tbdata.field('MIPSFLUXAP2ERR')
        self.mipsfluxap3=tbdata.field('MIPSFLUXAP3')
        self.mipsfluxap4err=tbdata.field('MIPSFLUXAP3ERR')
                        

        self.On24ImageFlag=tbdata.field('On24ImageFlag')
        self.supervopt=tbdata.field('SUPERVOPT')
        self.ra=tbdata.field('SUPERRA')
        self.dec=tbdata.field('SUPERDEC')



        self.stellarmass=tbdata.field('STELLARMASS')

        self.sdssMu=tbdata.field('SDSSMU')
	self.sdssLu=tbdata.field('SDSSLU')
        self.sdssMg=tbdata.field('SDSSMG')
        self.sdssLg=tbdata.field('SDSSLG')
        self.sdssMr=tbdata.field('SDSSMR')
        self.sdssLr=tbdata.field('SDSSLR')
        self.sdssMi=tbdata.field('SDSSMI')
        self.sdssLi=tbdata.field('SDSSLI')
        self.sdssMz=tbdata.field('SDSSMZ')
        self.sdssLz=tbdata.field('SDSSLZ')
        self.membflag =tbdata.field('MEMBFLAG')
        self.morphflag =tbdata.field('MORPHFLAG')
        self.morph =tbdata.field('MORPH')
        self.disturb =tbdata.field('DISTURB')
        self.localdens =tbdata.field('LOCALDENS')
        self.agn1 =tbdata.field('AGNKAUFF')
        self.agn2 =tbdata.field('AGNKEWLEY')
        self.agn3 =tbdata.field('AGNSTASIN')
        self.logn2halpha=log10(self.mpan2/self.mpahalpha)
        self.logo3hbeta=log10(self.mpao3/self.mpahbeta)
        self.ellipseflag24 =tbdata.field('ELLIPSEFLAG24')
        self.ellipseflagsdss =tbdata.field('ELLIPSEFLAGSDSS')
        self.ellipseflag =tbdata.field('ELLIPSEFLAG')

        # galaxy zoo fields
        self.galzooflag =tbdata.field('GALZOOFLAG')
        self.galzoonvote =tbdata.field('GALZOONVOTE')
        self.galzoopel =tbdata.field('GALZOOPEL')
        self.galzoopcw =tbdata.field('GALZOOPCW')
        self.galzoopacw =tbdata.field('GALZOOPACW')
        self.galzoopedge =tbdata.field('GALZOOPEDGE')
        self.galzoopdk =tbdata.field('GALZOOPDK')
        self.galzoopmg =tbdata.field('GALZOOPMG')
        self.galzoopcs =tbdata.field('GALZOOPCS')
        self.galzoopeldebiased =tbdata.field('GALZOOPELDEBIASED')
        self.galzoopcsdebiased =tbdata.field('GALZOOPCSDEBIASED')
        self.galzoospiral =tbdata.field('GALZOOSPIRAL')
        self.galzooelliptical =tbdata.field('GALZOOELLIPTICAL')
        self.galzoouncertain =tbdata.field('GALZOOUNCERTAIN')

        #new SDSS fields that quantify radial extent of galaxy
        self.sdssIsoAr =tbdata.field('SDSSISOAR')
        self.sdssIsoBr =tbdata.field('SDSSISOBR')
        self.sdssIsoPhir =tbdata.field('SDSSISOPHIR')
        self.sdssIsoPhirErr =tbdata.field('SDSSISOPHIERRR')
        self.sdssExpRadr =tbdata.field('SDSSEXPRADR')
        self.sdssExpABr =tbdata.field('SDSSEXPABR')
        self.sdssExpABrErr =tbdata.field('SDSSEXPABRERR')
        self.sdssExpPhir =tbdata.field('SDSSEXPPHIR')
        self.sdssExpPhirErr =tbdata.field('SDSSEXPPHIERRR')

        self.sdssumag=tbdata.field('SDSSDEREDU')#de-redened magnitudes
        self.sdssgmag=tbdata.field('SDSDEREDSG')
        self.sdssrmag=tbdata.field('SDSSDEREDR')
        self.sdssimag=tbdata.field('SDSSDEREDI')
        self.sdsszmag=tbdata.field('SDSSDEREDZ')

        
        #end of master table!
        #self.spiralFlag=self.On24ImageFlag & self.galzooflag & self.ellipseflag & (self.galzoopcsdebiased > 0.6)
        self.spiralFlag=self.On24ImageFlag & self.galzooflag & self.ellipseflag & self.galzoospiral
        self.clustername=clustername
        self.clusterra=clusterRA[clustername]
        self.clusterdec=clusterDec[clustername]
        self.dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        self.drR200=self.dr/self.r200deg
        self.clustervel=clustervel[clustername]
        self.clustersigma=clustersigma[clustername]
        self.clustervmin=self.clustervel-3.*self.clustersigma
        self.clustervmax=self.clustervel+3.*self.clustersigma

        self.dist=sqrt((self.clusterra-self.ra)**2 + (self.clusterdec-self.dec)**2)
        self.flagHI = (self.flux100 > 0.)
        self.flagmemb = ((self.vopt > self.clustervmin) & (self.vopt < self.clustervmax)) |  ((self.v21 > self.clustervmin) & (self.v21 < self.clustervmax))

        self.allvelocity=3.e5*self.sdssspecz
        for i in range(len(self.allvelocity)):
            if self.sdssflag[i] < 1:
                if self.v21[i] > 0:
                    self.allvelocity[i]=self.v21[i]
                else:
                    self.allvelocity[i]=self.vopt[i]


        self.nmemb=len(self.dist[self.membflag & self.On24ImageFlag])
        self.nfield=len(self.dist[self.On24ImageFlag])-self.nmemb
        print self.clustername,": ","N members = ",self.nmemb," N field = ",self.nfield
        print ' N spirals = ',sum(self.spiralFlag),' Nspiral members = ',sum(self.spiralFlag&self.membflag)

    def plotagn(self):
        figure()
        clf()
        plot(self.logn2halpha,self.logo3hbeta,'k.')

        plot(self.logn2halpha[self.agn2],self.logo3hbeta[self.agn2],'co',markersize=12)
        plot(self.logn2halpha[self.agn1],self.logo3hbeta[self.agn1],'go',markersize=8)
        plot(self.logn2halpha[self.agn3],self.logo3hbeta[self.agn3],'ro',markersize=4)

        #draw AGN diagnostic lines
        x=arange(-3,1,.01)
        y=(.61/(x-.47)+1.19)#Kewley
        plot(x,y,'c')
        y =(.61/(x-.05)+1.3)#Kauffman 2003?
        plot(x,y,'g')
        y = ((-30.787+(1.1358*x)+((.27297)*(x)**2))*tanh(5.7409*x))-31.093 #Stasinska 2006	    
        plot(x,y,'r')
        axis([-3,1.,-2,2])

    def getFilesForProfileFitting(self):
        outfile1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.24.dat'
        outfile2=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.sdss.dat'
        outfile3=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.24.dat'
        outfile4=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.sdss.dat'
        agcSpiral=self.agcnumber[self.spiralFlag]
        out1=open(outfile1,'w')
        out2=open(outfile2,'w')
        out3=open(outfile3,'w')
        out4=open(outfile4,'w')
        cutoutpath=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'
        for i in range(len(agcSpiral)):
            outim=cutoutpath+self.prefix+'-'+str(agcSpiral[i])+'-cutout-sdss.fits \n'
	    outim24=cutoutpath+self.prefix+'-'+str(agcSpiral[i])+'-cutout-24-rot.fits \n'
            out1.write(outim24)
            out2.write(outim)
            outfile=homedir+'research/LocalClusters/EllipseTables/'+self.prefix+'/'+self.prefix+'-'+str(agcSpiral[i])+'-cutout-sdss.dat \n'
            outfile24=homedir+'research/LocalClusters/EllipseTables/'+self.prefix+'/'+self.prefix+'-'+str(agcSpiral[i])+'-cutout-24-rot.dat \n'
            out3.write(outfile24)
            out4.write(outfile)
        out1.close()
        out2.close()
        out3.close()
        out4.close()
                 
                

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

        plot(self.ra[self.apexflag],self.dec[self.apexflag],'ro', markerfacecolor='r',markeredgecolor='b',markersize=4,label='24um')
            #plot(ra[flag],dec[flag],'k.')
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x


        #legend(loc='upper right',numpoints=1)


        title(self.clustername,fontsize=12)
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        axis('equal')
        drawbox(cluster24Box[self.clustername],'g-')
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax+1,2,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(round(ymin),ymax+1,2,'i'),fontsize=10)
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.eps'

        #savefig(s)

    def plotpositionson24(self):

        plot(self.ra[self.sdssflag & self.On24ImageFlag],self.dec[self.sdssflag& self.On24ImageFlag],'k.', alpha=0.5,markersize=4,label='SDSS')
        plot(self.ra[self.HIflag & self.On24ImageFlag],self.dec[self.HIflag & self.On24ImageFlag],'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='HI')

        plot(self.ra[self.apexflag],self.dec[self.apexflag],'ro', markerfacecolor='r',markeredgecolor='b',markersize=4,label='24um')
            #plot(ra[flag],dec[flag],'k.')
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x


        #legend(loc='upper right',numpoints=1)


        title(self.clustername,fontsize=12)
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #axis('equal')
        drawbox(cluster24Box[self.clustername],'g-')
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(round(ymin),ymax,1,'i'),fontsize=10)
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.eps'

        #savefig(s)

        

    def plotrelativepositionson24(self):

        plot(self.ra[self.sdssflag & self.On24ImageFlag]-self.clusterra,self.dec[self.sdssflag& self.On24ImageFlag]-self.clusterdec,'k.', alpha=0.5,markersize=4,label='SDSS')
        plot(self.ra[self.HIflag & self.On24ImageFlag]-self.clusterra,self.dec[self.HIflag & self.On24ImageFlag]-self.clusterdec,'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='HI')

        plot(self.ra[self.apexflag]-self.clusterra,self.dec[self.apexflag]-self.clusterdec,'ro', markerfacecolor='r',markeredgecolor='b',markersize=4,label='24um')
            #plot(ra[flag],dec[flag],'k.')
        plot(array([0]),array([0]),'kx',markersize=15,lw=8,label='_nolegend_')#mark cluster center with a red x


        #legend(loc='upper right',numpoints=1)


        title(self.clustername,fontsize=12)
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #axis('equal')
        drawbox(cluster24Box[self.clustername]-array([self.clusterra,self.clusterdec,0,0,0]),'g-')
        axis([-1.5,1.5,-2.,2.])
        xmin,xmax=xlim()
        xticks(arange(-1,2,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(-2,3,1,'i'),fontsize=10)
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.eps'

        #savefig(s)

    def plotveldron24(self):

        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        dv=self.supervopt-self.biweightvel

        membflag=(dr/self.r200deg < 1) & (abs(dv) < 3.*self.biweightscale)

        plot(dr[self.On24ImageFlag],self.supervopt[self.On24ImageFlag],'k.',markersize=3)
        plot(dr[membflag & self.On24ImageFlag],self.supervopt[membflag & self.On24ImageFlag],'bo',markersize=4)

        ymin=3500
        ymax=14000
        axis([0,2,ymin,ymax])
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)

        yticks(arange(4000,ymax,4000,'i'),fontsize=10)
        title(self.clustername,fontsize=12)
        axhline(self.biweightvel,ls='-',color='r')
        axhline(self.biweightvel+3*self.biweightscale,ls='--',color='r')
        axhline(self.biweightvel-3*self.biweightscale,ls='--',color='r')


    def plotveldr(self):

        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        dv=self.supervopt-self.biweightvel

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



    def plotlf(self):
        figure(1)

	y=hist(self.f24dist,histtype='step')
	ngal=y[0]
	x=y[1]
	xbin=zeros(len(ngal))
	for i in range(len(xbin)):
            xbin[i]=0.5*(x[i]+x[i+1])
	#clf()                                                                                       
        self.xbin=xbin
        self.ngal=ngal
        figure(2)
	plot(xbin,ngal,'ro')
	errorbar(xbin,ngal,sqrt(ngal))
	ax=gca()
	ax.set_yscale('log')
	ax.set_xscale('log')
        xlabel('24um Flux')
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


    def fitprofiles(self):
        #get list from LCSreadmaster.py
        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.sdss.dat'
        infile1=open(inf1,'r')
        sfiles=[]
        for line in infile1:
            t=line.rstrip()
            sfiles.append(t)
            #print t
        infile1.close()

        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.24.dat'
        infile1=open(inf1,'r')
        s24files=[]
        for line in infile1:
            t=line.rstrip()
            s24files.append(t)
            #print t
        infile1.close()

        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.sdss.dat'
        infile1=open(inf1,'r')
        simages=[]
        for line in infile1:
            t=line.rstrip()
            simages.append(t)
            #print t
        infile1.close()

        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.24.dat'
        infile1=open(inf1,'r')
        s24images=[]
        for line in infile1:
            t=line.rstrip()
            s24images.append(t)
            #print t
        infile1.close()

        pscale24=2.45#arcsec per pixel

        pscalesdss=1.#arcsec per pixel
        
        nrow=2
        ncol=4
        xticksize=10
        yticksize=10
        #for i in range(0,len(sfiles),nrow):
        ngal=0
        ngaltot=1.*len(sfiles)

        ratio=ngaltot/((nrow*ncol)/8.)
        npage=round(ratio)
        if ratio > npage:
            npage += 1
        npage=ngaltot
        #print "Ngal = ",ngaltot
        #print "Npage = ",npage
        redshift=(self.supervopt[self.spiralFlag]-self.clustervel)/self.clustersigma
        member=self.flagmemb[self.spiralFlag]
        dr=self.drR200[self.spiralFlag]

        index=arange(len(self.spiralFlag))
        spiralIndex=index[self.spiralFlag]
        vminsdss=-400
        vmaxsdss=100
        vmin24=-2.1
        vmax24=.5
        self.r0SDSS=zeros(len(spiralIndex),'f')#scale length from exponential fit
        self.r30SDSS=zeros(len(spiralIndex),'f')
        self.r90SDSS=zeros(len(spiralIndex),'f')
        self.skySDSS=zeros(len(spiralIndex),'f')
        self.r30EncFluxSDSS=zeros(len(spiralIndex),'f')
        self.r90EncFluxSDSS=zeros(len(spiralIndex),'f')
        self.MaxEncFluxSDSS=zeros(len(spiralIndex),'f')
        #same array for 24um
        self.r0F24=zeros(len(spiralIndex),'f')#scale length from exponential fit
        self.r30F24=zeros(len(spiralIndex),'f')
        self.r90F24=zeros(len(spiralIndex),'f')
        self.skyF24=zeros(len(spiralIndex),'f')
        self.r30EncFluxF24=zeros(len(spiralIndex),'f')
        self.r90EncFluxF24=zeros(len(spiralIndex),'f')
        self.MaxEncFluxF24=zeros(len(spiralIndex),'f')
        for i in range(int(npage)):
            figure(figsize=(15,5))
            subplots_adjust(left=0.05, right=.95,bottom=.1,top=0.9,wspace=0.4,hspace=0.6)
            clf()
            print 'ngal = ',ngal
            for j in range(0,ncol*nrow,8):
                t=sfiles[ngal]
                t1=t.split('/')
                t2=t1[len(t1)-1].split('-')
                if len(t2) > 4:
                    galname='-'+t2[2]
                else:
                    galname=t2[1]

                t=s24files[ngal]
                t1=t.split('/')
                t2=t1[len(t1)-1].split('-')
                if len(t2) > 5:
                    galname24='-'+t2[2]
                else:
                    galname24=t2[1]


                subplot(nrow,ncol,j+1)#sdss image
                fits=pyfits.open(simages[ngal])
                im=fits[0].data.copy()
                
                fits.close()
                axis('equal')
                imshow(-1.*(im),interpolation='nearest',origin='upper',vmin=vminsdss,vmax=vmaxsdss,cmap=cm.Greys)                           
                ax=gca()
                ax.set_yticklabels(([]))
                ax.set_xticklabels(([]))
                size='100\"'
                #text(.92, .5, size, horizontalalignment='center', verticalalignment='center',rotation=270, transform=ax.transAxes,fontsize=10)


                #text(.1, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)

                #text(.1, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                s='$'+self.prefix+': \ '+galname+'$'
                title(s,fontsize=12)
                s='$ r-band$'
                ylabel(s,fontsize=14)
                xlabel(r'$100 X 100 \ arcsec^2$',fontsize=10)
                #ylabel('$100 \ arcsec$')
##                subplot(nrow,ncol,j+2)#sdss masked image
##                fits=pyfits.open(simages[ngal])
##                im=fits[0].data.copy()
##                fits.close()
##                axis('equal')
##                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax,cmap=cm.Greys)
##                ax=gca()
##                ax.set_yticklabels(([]))
##                ax.set_xticklabels(([]))
##                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                
                subplot(nrow,ncol,j+3)#sdss profile
                edat=load(sfiles[ngal],usecols=[1,2,3,21,40])
                x=edat[:,0]
                y=edat[:,1]
                yerr=edat[:,2]
                tflux=edat[:,3]
                sarea=edat[:,4]
                plot(x,y,'b.')
                errorbar(x,y,yerr,fmt=None)
                r90,sky,r90EncFlux,MaxEncFlux,r30,r30EncFlux=calcC90(x,y,yerr,tflux)

                self.r30SDSS[ngal]=r30
                self.r90SDSS[ngal]=r90
                self.skySDSS[ngal]=sky
                self.r30EncFluxSDSS[ngal]=r30EncFlux
                self.r90EncFluxSDSS[ngal]=r90EncFlux
                self.MaxEncFluxSDSS[ngal]=MaxEncFlux

                axhline(sky,color='k',ls=':')
                axvline(r90,color='k',ls='--')
                axvline(r30,color='c',ls='--')
                xlabel('$r \ (arcsec)$')
                ylabel('$I(r)$')
                s='$sky = %5.2f, \ r30 = %5.1f, \ r90 = %5.1f$'%(sky,r30,r90)
                title(s,fontsize=10)
                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)

                #text(.1, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                

                subplot(nrow,ncol,j+2)
                plot(x,tflux,'b.')
                #plot(x,y*sarea,'r.')
                axhline(MaxEncFlux,color='k',ls=':')
                axvline(r90EncFlux,color='k',ls='--')
                axvline(r30EncFlux,color='c',ls='--')
                s='$max(F_{enc}) = %5.2e, \ r30 = %5.1f, \ r90 = %5.1f$'%(MaxEncFlux,r30EncFlux,r90EncFlux)
                title(s,fontsize=10)
                xlabel('$r \ (arcsec)$')
                ylabel('$\Sigma Flux(<r)$')

                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)
                xlim(0,50)

                subplot(nrow,ncol,j+4)#sdss ln profile with fit
                xfit=(x[y>5])
                yfit=log(y[y>5])
                m,b=polyfit(xfit,yfit,1)
                plot(xfit,yfit-b,'r^')
                xlabel('$r (arcsec)$')
                ylabel(r'$ln(I(r))-ln(I_0)$')
                #gradient, intercept, r_value, p_value, std_err = stats.linregress(xfit,yfit)

                plot(xfit,m*xfit,'g')
                axvline(-1./m,color='k',ls='--')
                axhline(-1,color='k',ls=':')
                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)
                s='$R_0 = %5.1f$'%(-1./m)
                title(s, fontsize=10)
                self.r0SDSS[ngal]=-1./m

                
                subplot(nrow,ncol,j+5)#sdss image
                fits=pyfits.open(s24images[ngal])
                im=fits[0].data.copy()
                fits.close()
                axis('equal')
                imshow(-1.*(im),interpolation='nearest',origin='upper',vmin=vmin24,vmax=vmax24,cmap=cm.Greys)
                ax=gca()
                ax.set_yticklabels(([]))
                ax.set_xticklabels(([]))
                #text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                ylabel('$24 \ \mu m$',fontsize=14)
                s1='$\Delta v/\sigma = %5.2f, \ \Delta r/R_{200} = %5.2f$'%(redshift[ngal],dr[ngal])
                title(s1,fontsize=10)
                xlabel(r'$100 X 100 \ arcsec^2$',fontsize=10)

##                subplot(nrow,ncol,j+6)#sdss masked image
##                fits=pyfits.open(s24images[ngal])
##                im=fits[0].data.copy()
##                fits.close()
##                axis('equal')
##                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax,cmap=cm.Greys)
##                ax=gca()
##                ax.set_yticklabels(([]))
##                ax.set_xticklabels(([]))
##                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                
                subplot(nrow,ncol,j+7)#sdss profile
                edat=load(s24files[ngal],usecols=[1,2,3,21])
                x=edat[:,0]
                y=edat[:,1]
                yerr=edat[:,2]
                tflux=edat[:,3]
                plot(x*pscale24,y,'b.')
                errorbar(x*pscale24,y,yerr,fmt=None)
                r90,sky,r90EncFlux,MaxEncFlux,r30,r30EncFlux=calcC90(x*pscale24,y,yerr,tflux)
                self.r30F24[ngal]=r30
                self.r90F24[ngal]=r90
                self.skyF24[ngal]=sky
                self.r30EncFluxF24[ngal]=r30EncFlux
                self.r90EncFluxF24[ngal]=r90EncFlux
                self.MaxEncFluxF24[ngal]=MaxEncFlux

                xlabel('$r \ (arcsec)$')
                ylabel('$I(r)$')
                s='$sky = %5.2f, \ r30 = %5.1f, \ r90 = %5.1f$'%(sky,r30,r90)
                title(s,fontsize=10)
                axhline(sky,color='k',ls=':')
                axvline(r90,color='k',ls='--')
                axvline(r30,color='c',ls='--')
                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)
                xlim(0,50)

                subplot(nrow,ncol,j+6)
                plot(x*pscale24,tflux,'b.')
                axhline(MaxEncFlux,color='k',ls=':')
                axvline(r90EncFlux,color='k',ls='--')
                axvline(r30EncFlux,color='c',ls='--')
                xlabel('$r \ (arcsec)$')
                ylabel('$\Sigma Flux(<r)$')
                s='$max(F_{enc}) = %5.2e, \ r30 = %5.1f, \ r90 = %5.1f$'%(MaxEncFlux,r30EncFlux,r90EncFlux)
                title(s,fontsize=10)
                xlim(0,50)
                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)


                subplot(nrow,ncol,j+8)#sdss ln profile with fit
                xfit=(x[y>.05])
                yfit=log(y[y>.05])
                if len(xfit) > 1:
                    m,b=polyfit(xfit*pscale24,yfit,1)
                    plot(xfit*pscale24,m*xfit*pscale24,'g')
                    plot(xfit*pscale24,yfit-b,'r^')
                else:
                    plot(xfit*pscale24,yfit,'r^')
                xlabel('$r (arcsec)$')
                ylabel(r'$ln(I(r))-ln(I_0)$')
                #gradient, intercept, r_value, p_value, std_err = stats.linregress(xfit,yfit)
                axvline(-1./m,color='k',ls='--')
                axhline(-1,color='k',ls=':')
                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)
                s='$R_0 = %5.1f$'%(-1./m)
                title(s, fontsize=10)
                self.r0F24[ngal]=-1./m

                ngal += 1
                if ngal >= ngaltot:
                    figname=self.prefix+'Profiles'+str(galname)+'.png'
                    savefig(figname)
                    break
            figname=self.prefix+'Profiles'+str(galname)+'.png'
            savefig(figname)

mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

mkw11=cluster('MKW11')
mkw8=cluster('MKW8')
awm4=cluster('AWM4')
a2052=cluster('A2052')
a2063=cluster('A2063')
ngc=cluster('NGC6107')
coma=cluster('Coma')
herc=cluster('Hercules')
a1367=cluster('A1367')
def plotpositionsall():
    figure()
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        subplot(3,3,i)
        cl.plotpositions()
    ax=gca()
    text(-.75,-.35,'RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(homedir+'research/LocalClusters/SamplePlots/PlotPositionsAll.eps')



def plotpositionson24all():
    figure()
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        subplot(3,3,i)
        cl.plotpositionson24()
    ax=gca()
    text(-.75,-.35,'RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(homedir+'research/LocalClusters/SamplePlots/PlotPositionsOn24All.eps')


def plotrelativepositionson24all():
    figure(figsize=[9,9])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        subplot(3,3,i)
        cl.plotrelativepositionson24()
    leg=legend(numpoints=1)#,fontsize=12)
    for t in leg.get_texts():
        t.set_fontsize('small')
    ax=gca()
    text(-.75,-.35,'$\Delta$RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$\Delta$Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(homedir+'research/LocalClusters/SamplePlots/PlotRelativePositionsOn24All.eps')


def plotveldrall():
    figure(figsize=[9,6])
    clf()
    subplots_adjust(wspace=.35,hspace=.35)
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        subplot(3,3,i)
        cl.plotveldr()
    ax=gca()
    text(-.75,-.35,'dr (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-3.1,1.9,'$V_r$ (km/s)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(homedir+'research/LocalClusters/SamplePlots/PlotVeldrAll.eps')


def plotveldron24all():
    figure(figsize=[9,9])
    clf()
    subplots_adjust(wspace=.35,hspace=.35)
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        subplot(3,3,i)
        cl.plotveldron24()
    ax=gca()
    text(-.75,-.35,'$\Delta$r (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-3.1,1.9,'$V_r$ (km/s)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(homedir+'research/LocalClusters/SamplePlots/PlotVeldrAllOn24.eps')


def checkmorphall():
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        cl.checkmorph()

def getSpirals():
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        cl.getFilesForProfileFitting()

#plotpositionsall()
#plotpositionson24all()
#plotveldrall()
#getSpirals()
#mkw11.fitprofiles()
