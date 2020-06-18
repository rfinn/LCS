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
        self.csigma=self.biweightscale
        self.mcl=my.clusterMass(self.csigma,self.cz,h)
        self.AngDistance=my.DA(self.cz,h)
        mypath=os.getcwd()
        if mypath.find('Users') > -1:
            print "Running on Rose's mac pro"
            infile='/Users/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.fits'
            homedir='/Users/rfinn/'
        elif mypath.find('home') > -1:
            print "Running on coma"
            infile='/home/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.fits'
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

        self.sdssmag=tbdata.field('SDSSMAG')
        self.sdssu=self.sdssmag[:,0]
        self.sdssg=self.sdssmag[:,1]
        self.sdssr=self.sdssmag[:,2]
        self.sdssi=self.sdssmag[:,3]
        self.sdssz=self.sdssmag[:,4]
        self.sdssmagerr=tbdata.field('SDSSMAGERR')
        self.sdssuerr=self.sdssmagerr[:,0]
        self.sdssgerr=self.sdssmagerr[:,1]
        self.sdssrerr=self.sdssmagerr[:,2]
        self.sdssierr=self.sdssmagerr[:,3]
        self.sdsszerr=self.sdssmagerr[:,4]

        
                
        self.sdssspecz=tbdata.field('SDSSSPECZ')
        self.sdssvopt=tbdata.field('SDSSVOPT')
        self.sdsshaew=tbdata.field('SDSSHAEW')
        self.sdsshaewerr=tbdata.field('SDSSHAEWERR')
        self.sdssplate=tbdata.field('SDSSPLATE')
        self.sdssfiberid=tbdata.field('SDSSFIBERID')
        self.sdsstile=tbdata.field('SDSSTILE')
        self.sdssrun=tbdata.field('SDSSRUN')
        self.sdssrerun=tbdata.field('SDSSRERUN')
        self.sdsscamcol=tbdata.field('SDSSCAMCOL')
        self.sdssfield=tbdata.field('SDSSFIELD')
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
        self.stellarmass_cl=tbdata.field('STELLARMASS_CL')

        self.sdssabsmag=tbdata.field('SDSSABSMAG')
        self.sdssMu=self.sdssabsmag[:,0]
        self.sdssMg=self.sdssabsmag[:,1]
        self.sdssMr=self.sdssabsmag[:,2]
        self.sdssMi=self.sdssabsmag[:,3]
        self.sdssMz=self.sdssabsmag[:,4]


        self.sdsslum=tbdata.field('SDSSLUM')
        self.sdssLu=self.sdsslum[:,0]
        self.sdssLg=self.sdsslum[:,1]
        self.sdssLr=self.sdsslum[:,2]
        self.sdssLi=self.sdsslum[:,3]
        self.sdssLz=self.sdsslum[:,4]

        self.sdssabsmag_cl=tbdata.field('SDSSABSMAG_CL')
        self.sdssMu=self.sdssabsmag_cl[:,0]
        self.sdssMg=self.sdssabsmag_cl[:,1]
        self.sdssMr=self.sdssabsmag_cl[:,2]
        self.sdssMi=self.sdssabsmag_cl[:,3]
        self.sdssMz=self.sdssabsmag_cl[:,4]

        self.sdsslum_cl=tbdata.field('SDSSLUM_CL')
        self.sdssLu_cl=self.sdsslum_cl[:,0]
        self.sdssLg_cl=self.sdsslum_cl[:,1]
        self.sdssLr_cl=self.sdsslum_cl[:,2]
        self.sdssLi_cl=self.sdsslum_cl[:,3]
        self.sdssLz_cl=self.sdsslum_cl[:,4]

        self.sdsscolc=tbdata.field('SDSSCOLC')
        self.sdssrowc=tbdata.field('SDSSROWC')

        self.membflag =tbdata.field('MEMBFLAG')
        self.morphflag =tbdata.field('MORPHFLAG')
        self.morph =tbdata.field('MORPH')
        self.disturb =tbdata.field('DISTURB')
        self.localdens =tbdata.field('LOCALDENS')
        self.agn1 =tbdata.field('AGNKAUFF')
        self.agn2 =tbdata.field('AGNKEWLEY')
        self.agn3 =tbdata.field('AGNSTASIN')
        self.n2halpha=(self.mpan2/self.mpahalpha)
        self.o3hbeta=(self.mpao3/self.mpahbeta)
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

        self.sdssPetroMag=tbdata.field('SDSSPETROMAG')
        self.sdssPetroMagr=self.sdssPetroMag[:,2]

        self.sdssPetroRad=tbdata.field('SDSSPETRORAD')
        self.sdssPetroRadr=self.sdssPetroRad[:,2]

        self.sdssPetroR50=tbdata.field('SDSSPETROR50')
        self.sdssPetroR50r=self.sdssPetroR50[:,2]

        self.sdssPetroR90=tbdata.field('SDSSPETROR90')
        self.sdssPetroR90r=self.sdssPetroR90[:,2]

        #de-redened magnitudes
        self.sdssdered=tbdata.field('SDSSDERED')
        self.sdssumag=self.sdssdered[:,0]
        self.sdssgmag=self.sdssdered[:,1]
        self.sdssrmag=self.sdssdered[:,2]
        self.sdssimag=self.sdssdered[:,3]
        self.sdsszmag=self.sdssdered[:,4]

        # other Lum and SFR
        self.HImass=tbdata.field('HIMASS')
        self.L24=tbdata.field('L24')
        self.L24err=tbdata.field('L24ERR')
        self.Lir=tbdata.field('LIR')
        self.Lirerr=tbdata.field('LIRERR')
        self.SFR24=tbdata.field('SFR24')
        self.SFR24err=tbdata.field('SFR24ERR')
        self.SuperSFR24=tbdata.field('SUPERSFR24')
        self.SuperSFR24err=tbdata.field('SUPERSFR24ERR')

        self.HImass_cl=tbdata.field('HIMASS_CL')
        self.L24_cl=tbdata.field('L24_CL')
        self.L24err_cl=tbdata.field('L24ERR_CL')
        self.Lir_cl=tbdata.field('LIR_CL')
        self.Lirerr_cl=tbdata.field('LIRERR_CL')
        self.SFR24_cl=tbdata.field('SFR24_CL')
        self.SFR24err_cl=tbdata.field('SFR24ERR_CL')
        self.SuperSFR24_cl=tbdata.field('SUPERSFR24_CL')
        self.SuperSFR24err_cl=tbdata.field('SUPERSFR24ERR_CL')

        
        #define red, green and blue galaxies
        ur=self.sdssumag-self.sdssrmag
        self.redflag=(ur > 2.3)
        self.greenflag=(ur > 1.8) & (ur < 2.3)
        self.blueflag=(ur<1.8)
        #end of master table!
        #self.spiralFlag=self.On24ImageFlag & self.galzooflag & self.ellipseflag & (self.galzoopcsdebiased > 0.6)
        self.spiralFlag=self.galzooflag & self.galzoospiral
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
        print ' N spirals on 24um image = ',sum(self.spiralFlag & self.On24ImageFlag),' Nspiral members = ',sum(self.spiralFlag&self.membflag & self.On24ImageFlag)
        print ' N galaxies on 24um image = ',sum(self.On24ImageFlag),' Nspiral members = ',sum(self.membflag & self.On24ImageFlag)
        self.agcdict=dict((a,b) for a,b in zip(self.agcnumber,arange(len(self.agcnumber))))




        #self.L24=zeros(len(self.mipsflux),'d')
        #self.L24err=zeros(len(self.mipsflux),'d')

        # calculate HI deficiency using Toribio et al 2011 results
        # their relation is
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        # and
        # log D_25 = log D_25(obs) + beta log(b/a), where beta = 0.35 in r-band
        # NOTE: SDSS isophotal radii are given in pixels!!!!
        a=self.sdssIsoAr
        b=self.sdssIsoBr
        # convert from arcsec to kpc with self.AngDistance (which is in units of kpc/arcsec)
        # multiply by 2 to convert from radius to diameter
        # multiply by sdss pixel scale (0.39) b/c isophotal radii are given in pixels
        self.D25obskpc=2.*self.sdssIsoAr*sdsspixelscale*self.AngDistance
        # apply correction from toribio et al 2011 
        self.logD25kpc=log10(self.D25obskpc) + 0.35*log10(b/a)
        # use toribio et al relation to predict the expected HI mass, including factor of 2 correction
        self.HImassExpected = 10.**(8.72 + 1.25*(self.logD25kpc-log10(2.)))
        self.HImassExpFromMr=10.**(6.44-0.18*self.sdssMr)
        self.HImassExpFromgr=10.**(8.84+1.81*(self.sdssgmag-self.sdssrmag))
        # calculate deficiency as log expected - log observed
        self.HIDef = log10(self.HImassExpected) - log10(self.HImass)
        self.myHIDef = log10(self.HImassExpected -self.HImass)
        self.agnflag=self.agn1#use Kauffmann et al 2003 cut
        self.irflag = self.apexflag  & self.membflag
        #set flag fot galaxies with dv < 3 sigma
        self.dvflag = self.dv < 3.
