#!/usr/bin/env python
import pyfits
from LCScommon import *
from pylab import *
import os
import mystuff as my


class baseCluster:
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

        self.cdMpc=self.biweightvel/H0
        self.cdcm=self.cdMpc*3.e24


        mypath=os.getcwd()
        if mypath.find('Users') > -1:
            print "Running on Rose's mac pro"
            infile='/Users/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'
            homedir='/Users/rfinn/'
        elif mypath.find('home') > -1:
            print "Running on coma"
            infile='/home/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'
            homedir='/home/rfinn/'
        self.cutoutpath=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'
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
        self.sdssgmag=tbdata.field('SDSSDEREDG')
        self.sdssrmag=tbdata.field('SDSSDEREDR')
        self.sdssimag=tbdata.field('SDSSDEREDI')
        self.sdsszmag=tbdata.field('SDSSDEREDZ')

        
        #end of master table!

        #from mastertable WithProfileFits

        self.spiralFlag=tbdata.field('SPIRALFLAG')
        self.r0SDSS = tbdata.field('R0SDSS')
        self.r30SDSS = tbdata.field('R30SDSS')
        self.r50SDSS = tbdata.field('R50SDSS')
        self.r90SDSS = tbdata.field('R90SDSS')
        self.skySDSS = tbdata.field('SKYSDSS')
        self.r30EncFluxSDSS = tbdata.field('R30ENCFLUXSDSS')
        self.r90EncFluxSDSS = tbdata.field('R90ENCFLUXSDSS')
        self.MaxEncFluxSDSS = tbdata.field('MAXENCFLUXSDSS')

        self.SDSSF30 = tbdata.field('SDSSF30')
        self.SDSSF50 = tbdata.field('SDSSF50')
        self.SDSSF90 = tbdata.field('SDSSF90')

        self.r0F24 = tbdata.field('R0F24')
        self.r30F24 = tbdata.field('R30F24')
        self.r50F24 = tbdata.field('R50F24')
        self.r90F24 = tbdata.field('R90F24')
        self.skyF24 = tbdata.field('SKYF24')
        self.r30EncFluxF24 = tbdata.field('R30ENCFLUXF24')
        self.r90EncFluxF24 = tbdata.field('R90ENCFLUXF24')
        self.MaxEncFluxF24 = tbdata.field('MAXENCFLUXF24')

        self.mipsF30 = tbdata.field('MIPSF30')
        self.mipsF50 = tbdata.field('MIPSF50')
        self.mipsF90 = tbdata.field('MIPSF90')

        self.SDSSEllipseEllip=tbdata.field('SDSSEllipseEllip')
        self.SDSSEllipseEllipErr=tbdata.field('SDSSEllipseEllipErr')
        self.SDSSEllipsePhi=tbdata.field('SDSSEllipsePhi')
        self.SDSSEllipsePhiErr=tbdata.field('SDSSEllipsePhiErr')


        self.imagearea50=pi*(self.r50SDSS/pscalesdss)**2*(1-self.SDSSEllipseEllip)
        self.imagearea90=pi*(self.r90SDSS/pscalesdss)**2*(1-self.SDSSEllipseEllip)
        self.mipsimagearea50=pi*(self.r50SDSS/pscale24)**2*(1-self.SDSSEllipseEllip)
        self.mipsimagearea90=pi*(self.r90SDSS/pscale24)**2*(1-self.SDSSEllipseEllip)

        self.mipsF30err = tbdata.field('mipsF30err')
        self.mipsF50err = tbdata.field('mipsF50err')#/sqrt(mipsimagearea50)
        self.mipsF90err = tbdata.field('mipsF90err')#/sqrt(mipsimagearea90)

        self.SDSSF30err = tbdata.field('sdssF30err')
        self.SDSSF50err = tbdata.field('sdssF50err')#/sqrt(imagearea50)
        self.SDSSF90err = tbdata.field('sdssF90err')#/sqrt(imagearea90)




        #end of master table!
        #redefining noise estimates until I can deal w/this properly
        self.sdsssnr=self.fluxautoser/self.fluxerrautoser

#        self.mipsF50err = self.mipsF50/(self.mipssnr*sqrt(self.mipsF50/self.fluxautose24))
#        self.mipsF90err = self.mipsF90/(self.mipssnr*sqrt(self.mipsF90/self.fluxautose24))

#        self.SDSSF50err = self.SDSSF50/(self.sdsssnr*sqrt(self.SDSSF50/self.fluxautoser))
#        self.SDSSF90err = self.SDSSF90/(self.sdsssnr*sqrt(self.SDSSF90/self.fluxautoser))
        #self.mipsF50err = self.mipsF50/(self.mipssnr)
        #self.mipsF90err = self.mipsF90/(self.mipssnr)

        #self.SDSSF50err = self.SDSSF50/(self.sdsssnr)
        #self.SDSSF90err = self.SDSSF90/(self.sdsssnr)
        

        #self.spiralFlag=self.On24ImageFlag & self.galzooflag & self.ellipseflag & (self.galzoopcsdebiased > 0.6)
        #self.spiralFlag=self.On24ImageFlag & self.galzooflag & self.ellipseflag & self.galzoospiral
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

        self.dv=abs(self.supervopt-self.biweightvel)/self.biweightscale


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
        self.agcdict=dict((a,b) for a,b in zip(self.agcnumber,arange(len(self.agcnumber))))



        self.sSFR50=self.mipsF50/self.SDSSF50
        a=self.mipsF50
        b=self.SDSSF50
        aerr=self.mipsF50err
        berr=self.SDSSF50err
        self.sSFR50err=sqrt((aerr/b)**2+(a*berr/b**2)**2)
        self.sSFR90=self.mipsF90/self.SDSSF90

        a=self.mipsF90
        b=self.SDSSF90
        aerr=self.mipsF90err
        berr=self.SDSSF90err
        self.sSFR90err=sqrt((aerr/b)**2+(a*berr/b**2)**2)

        self.sSFR5090=(self.mipsF90-self.mipsF50)/(self.SDSSF90 -self.SDSSF50)

        a=(self.mipsF90-self.mipsF50)
        b=(self.SDSSF90-self.SDSSF50)
        aerr=sqrt((self.mipsF50err)**2+(self.mipsF90err)**2)
        berr=sqrt(self.SDSSF50err**2+self.SDSSF90err**2)
        self.sSFR5090err=sqrt((aerr/b)**2+(a*berr/b**2)**2)

        self.ratio0=self.r0F24/self.r0SDSS        
        self.ratio30=self.r30F24/self.r30SDSS

        self.ratio90=self.r90F24/self.r90SDSS
        self.ratio30EncFlux=self.r30EncFluxF24/self.r30EncFluxSDSS
        self.ratio90EncFlux=self.r90EncFluxF24/self.r90EncFluxSDSS



        #self.L24=zeros(len(self.mipsflux),'d')
        #self.L24err=zeros(len(self.mipsflux),'d')

        ## these should be added to mastertable
        conv=4*pi*(self.cdcm**2)*1.e-6*1.e-23*(3.e8/24.e-6)/Lsol
        print 'conversion from F24 to L24 = ',conv
        self.L24=self.mipsflux*conv


        self.L24err=self.mipsfluxerr*conv
        self.SFR24=self.L24*8.*4.5e-44*Lsol#approx conv from papovich
        self.SFR24err=self.L24err*8.*4.5e-44*Lsol#approx conv from papovich

        self.SFR24se=(self.fluxautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
        self.SFR24seerr=(self.fluxerrautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
        self.snr24se=abs(self.SFR24se/self.SFR24seerr)

        self.superSFR24=self.SFR24*self.apexflag+self.SFR24se*(~self.apexflag&self.sex24flag)
        self.superSFR24err=self.SFR24err*self.apexflag+self.SFR24seerr*(~self.apexflag&self.sex24flag)
        self.sSFR=self.SFR24/self.stellarmass
        
        self.HImass=2.356e5*self.flux100/100.*self.cdMpc**2

        #DEF = log(M_HI(Dopt,T)) - log(M_HI)
        #self.HIdef=
