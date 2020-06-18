#!/usr/bin/env python
#import pyfits
from LCScommon import *
from pylab import *
import os
from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii
import mystuff as my

import chary_elbaz_24um as chary

mingalaxysize_kpc=my.DA(clusterbiweightcenter['Hercules']/3.e5,h)*mingalaxysize
class baseClusterNSA:
    def __init__(self,clustername):
        #Get current path so program can tell if this is being run on Becky or Rose's computer
        self.prefix=clustername
        self.john_prefix=john_prefix[self.prefix]
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]
        self.biweightvel=clusterbiweightcenter[self.prefix]
        self.biweightscale=clusterbiweightscale[self.prefix]
        self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. # in Mpc
        self.r200deg=self.r200*1000./my.DA(self.biweightvel/3.e5,h)/3600.

        self.cdMpc=self.biweightvel/H0
        self.cdcm=self.cdMpc*3.e24
        self.csigma=self.biweightscale
        self.mcl=my.clusterMass(self.biweightscale,self.biweightvel/3.e5,h)
        self.cLx=clusterLx[self.prefix]
        self.mingalaxysize=mingalaxysize_kpc/my.DA(self.biweightvel/3.e5,h)
        #infile=homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSAmastertable.fits'
        infile=homedir+'research/LocalClusters/NSAmastertables/NSAwithAGC/'+clustername+'_NSAmastertable_topcat.fits'
        self.cutoutpath=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'
        #infile='/home/rfinn/LocalClusters/MasterTables/'+clustername+'mastertable.fits'
        print infile
        self.n=fits.getdata(infile)

        infile=homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSA.fits'
        self.nsa=fits.getdata(infile)

        # read in WISE catalogs
        infile=homedir+'research/LocalClusters/NSAmastertables/WISETables/'+clustername+'_WISE_NSA.fits'
        self.wise=fits.getdata(infile)

        # read in JM Stellar Mass catalogs
        infile=homedir+'research/LocalClusters/NSAmastertables/JMstellarmass/'+clustername+'_JM_MSTAR.fits'
        self.jmass=fits.getdata(infile)

        # read in Galaxy Zoo catalogs
        infile=homedir+'research/LocalClusters/NSAmastertables/GalaxyZooTables/'+clustername+'_GalaxyZoo.fits'
        self.zoo=fits.getdata(infile)
        #self.zooflag=self.zoo.zoo_match_flag | self.zoo.zoo_phot_match_flag

        # read in GIM2D catalogs
        infile=homedir+'research/LocalClusters/NSAmastertables/SimardGIM2D/'+clustername+'_GIM2D.fits'
        self.gim2d=fits.getdata(infile)
        # fields with _1 are from bulge+disk; fields with _2 are from pure sersic fit

        infile=homedir+'research/LocalClusters/NSAmastertables/Sex24Tables/'+clustername+'_sex24.fits'
        self.sex24=fits.getdata(infile)

        infile=homedir+'research/LocalClusters/NSAmastertables/AGCTables/'+clustername+'_AGC.fits'
        self.agc=fits.getdata(infile)

        sersicparam24_file=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_24.fits'
        self.galfit24=fits.getdata(sersicparam24_file)
        
        infile=homedir+'research/LocalClusters/NSAmastertables/LocalDensityTables/'+clustername+'_localdensity.fits'
        self.ld=fits.getdata(infile)


        infile=homedir+'research/LocalClusters/NSAmastertables/CharyElbazTables/'+clustername+'_ce_lir.fits'
        self.ce=fits.getdata(infile)
        self.snrse=abs(self.ce.FLUX24/self.ce.FLUX24ERR)

        # read in John's Lir catalogs - for his members only
        infile=homedir+'research/LocalClusters/NSAmastertables/MoustakasLir/'+self.john_prefix+'_lir.fits'
        self.lir=fits.getdata(infile)

        self.memb_id=self.lir.NSAID

        # calculating magnitudes from fluxes provided from NSA 
        # 
        # m = 22.5 - 2.5 log_10 (flux_nanomaggies)
        # from http://www.sdss3.org/dr8/algorithms/magnitudes.php#nmgy
        self.nsamag=22.5-2.5*log10(self.nsa.NMGY)

 
        self.sdssLr=10.**(-1*(self.nsa.ABSMAG[:,4]-SolarMag['r'])/2.5)
        a,b=bellgr['r']
        self.stellarmass=10.**(-1.*a+b*(self.nsa.ABSMAG[:,3]-self.nsa.ABSMAG[:,4]))*self.sdssLr # uses g-r color
        #self.stellarmass=self.n.STELLARMASS
        #self.stellarmass_cl=self.n.STELLARMASS_CL


        
        self.sdssLu=10.**(-1*(self.nsa.ABSMAG[:,2]-SolarMag['u'])/2.5)
        
        self.sdssLg=10.**(-1*(self.nsa.ABSMAG[:,3]-SolarMag['g'])/2.5)
        
        self.sdssLi=10.**(-1*(self.nsa.ABSMAG[:,5]-SolarMag['i'])/2.5)
        
        self.sdssLz=10.**(-1*(self.nsa.ABSMAG[:,6]-SolarMag['z'])/2.5)
        

        self.AngDistanceCl=my.DA(self.cz,h)
        self.AngDistanceAll=zeros(len(self.nsa.ZDIST),'f')
        for i in range(len(self.nsa.ZDIST)):
            self.AngDistanceAll[i]=my.DA(self.nsa.ZDIST[i],h)

        # FOR BACKWARDS COMPATABILITY WITH EXISTING CODE

        # data from AGC-based mastertables
        self.agcflag=self.agc.AGCMATCHFLAG
        self.HIflag=(self.agc.FLUX100 > 0)
        self.sdssflag=self.n.SDSSflag
        self.sdssphotflag=self.n.SDSSphotflag
        self.mpaflag=self.n.MPAFLAG
        self.apexflag=self.n.APEXflag
        self.sexsdssflag=self.n.SEXSDSSflag
        self.sex24flag=self.n.SEX24FLAG
        self.agcvoptflag=(self.agc.VOPT > 0)

        self.agcnumber=self.agc.AGCNUMBER
        self.raagc=self.agc.RA
        self.decagc=self.agc.DEC
        self.a100=self.agc.A100
        self.b100=self.agc.B100
        self.mag10=self.agc.MAG10
        self.posang=self.agc.POSANG
        self.bsteintype=self.agc.BSTEINTYPE
        self.vopt=self.agc.VOPT
        self.verr=self.agc.VERR
        self.vsource=self.agc.VSOURCE
        self.flux100=self.agc.FLUX100
        self.rms100=self.agc.RMS100
        self.v21=self.agc.V21
        self.width=self.agc.WIDTH
        self.widtherr=self.agc.WIDTHERR
        #sdss info
        self.sdssra=self.n.SDSSRA
        self.sdssdec=self.n.SDSSDEC
        self.sdssphotra=self.n.SDSSphotRA
        self.sdssphotdec=self.n.SDSSphotDEC

        self.sdssmag=self.n.SDSSMAG
        self.sdssu=self.sdssmag[:,0]
        self.sdssg=self.sdssmag[:,1]
        self.sdssr=self.sdssmag[:,2]
        self.sdssi=self.sdssmag[:,3]
        self.sdssz=self.sdssmag[:,4]
        self.sdssmagerr=self.n.SDSSMAGERR
        self.sdssuerr=self.sdssmagerr[:,0]
        self.sdssgerr=self.sdssmagerr[:,1]
        self.sdssrerr=self.sdssmagerr[:,2]
        self.sdssierr=self.sdssmagerr[:,3]
        self.sdsszerr=self.sdssmagerr[:,4]

        
                
        self.sdssspecz=self.n.SDSSSPECZ
        self.sdssvopt=self.n.SDSSVOPT
        self.sdsshaew=self.n.SDSSHAEW
        self.sdsshaewerr=self.n.SDSSHAEWERR
        self.sdssplate=self.n.SDSSPLATE
        self.sdssfiberid=self.n.SDSSFIBERID
        self.sdsstile=self.n.SDSSTILE
        self.sdssrun=self.n.SDSSRUN
        self.sdssrerun=self.n.SDSSRERUN
        self.sdsscamcol=self.n.SDSSCAMCOL
        self.sdssfield=self.n.SDSSFIELD
        self.mpahalpha=self.n.MPAHALPHA
        self.mpahbeta=self.n.MPAHBETA
        self.mpao3=self.n.MPAOIII
        self.mpan2=self.n.MPANII
        #sextractor info
        self.numberser=self.n.NUMBERSER
        self.ximageser=self.n.XIMAGESER
        self.yimageser=self.n.YIMAGESER
        self.xminimageser=self.n.XMINIMAGESER
        self.xmaximageser=self.n.XMAXIMAGESER
        self.yminimageser=self.n.YMINIMAGESER
        self.raser=self.n.RASER
        self.decser=self.n.DECSER
        self.fluxisoser=self.n.FLUXISOSER
        self.fluxerrisoser=self.n.FLUXERRISOSER
        self.magisoser=self.n.MAGISOSER
        self.magerrisoser=self.n.MAGERRISOSER
        self.fluxautoser=self.n.FLUXAUTOSER
        self.fluxerrautoser=self.n.FLUXERRAUTOSER
        self.magautoser=self.n.MAGAUTOSER
        self.magerrautoser=self.n.MAGERRAUTOSER
        self.fluxpetroser=self.n.FLUXPETROSER
        self.fluxerrpetroser=self.n.FLUXERRPETROSER
        self.magpetroser=self.n.MAGPETROSER
        self.magerrpetroser=self.n.MAGERRPETROSER
        self.kronradser=self.n.KRONRADSER#kron radius
        self.petroradser=self.n.PETRORADSER#petrosian radius
        self.fluxradser=self.n.FLUXRADSER#1/2 light radius
        self.isoareaser=self.n.ISOAREASER
        self.aworldser=self.n.AWORLDSER
        self.bworldser=self.n.BWORLDSER
        self.thetaser=self.n.THETASER
        self.errthetaser=self.n.ERRTHETASER
        self.thetaj2000ser=self.n.THETAJ2000SER
        self.errthetaj2000ser=self.n.ERRTHETAJ2000SER
        self.elongser=self.n.ELONGATIONSER
        self.elliptser=self.n.ELLIPTICITYSER
        self.fwhmser=self.n.FWHMSER
        self.flagsser=self.n.FLAGSSER
        self.classstarser=self.n.CLASSSTARSER
        #SEXTRACTOR  output 24 micron data
        self.numberse24=self.n.NUMBERSE24
        self.ximagese24=self.n.XIMAGESE24
        self.yimagese24=self.n.YIMAGESE24
        self.xminimagese24=self.n.XMINIMAGESE24
        self.xmaximagese24=self.n.XMAXIMAGESE24
        self.xminimagese24=self.n.YMINIMAGESE24
        self.rase24=self.n.RASE24
        self.decse24=self.n.DECSE24
        self.fluxisose24=self.n.FLUXISOSE24
        self.fluxerrisose24=self.n.FLUXERRISOSE24
        self.magisose24=self.n.MAGISOSE24
        self.magerrisose24=self.n.MAGERRISOSE24
        self.fluxautose24=self.n.FLUXAUTOSE24
        self.fluxerrautose24=self.n.FLUXERRAUTOSE24
        self.magautose24=self.n.MAGAUTOSE24
        self.magerrautose24=self.n.MAGERRAUTOSE24
        self.fluxpetrose24=self.n.FLUXPETROSE24
        self.fluxerrpetrose24=self.n.FLUXERRPETROSE24
        self.magpetrose24=self.n.MAGPETROSE24
        self.magerrpetrose24=self.n.MAGERRPETROSE24
        self.kronradse24=self.n.KRONRADSE24
        self.petroradse24=self.n.PETRORADSE24
        self.fluxradse24=self.n.FLUXRADSE24
        self.isoarease24=self.n.ISOAREASE24
        self.aworldse24=self.n.AWORLDSE24
        self.bworldse24=self.n.BWORLDSE24
        self.thetase24=self.n.THETASE24
        self.errthetase24=self.n.ERRTHETASE24
        self.thetaj2000se24=self.n.THETAJ2000SE24
        self.errthetaj2000se24=self.n.ERRTHETAJ2000SE24
        self.elongse24=self.n.ELONGATIONSE24
        self.elliptse24=self.n.ELLIPTICITYSE24
        self.fwhmse24=self.n.FWHMSE24
        self.flagsse24=self.n.FLAGSSE24
        self.classstarse24=self.n.CLASSSTARSE24
        self.f24dist=self.fluxautose24[self.sex24flag]
        #apex output
        self.mipsra=self.n.MIPSRA
        self.mipsdec=self.n.MIPSDEC
        self.mipsflux=self.n.MIPSFLUX
        self.mipsfluxerr=self.n.MIPSFLUXERR
        self.mipssnr=self.n.MIPSSNR
        self.mipsdeblend=self.n.MIPSDEBLEND
        self.mipsfluxap1=self.n.MIPSFLUXAP1
        self.mipsfluxap1err=self.n.MIPSFLUXAP1ERR
        self.mipsfluxap2=self.n.MIPSFLUXAP2
        self.mipsfluxap2err=self.n.MIPSFLUXAP2ERR
        self.mipsfluxap3=self.n.MIPSFLUXAP3
        self.mipsfluxap4err=self.n.MIPSFLUXAP3ERR
                        

        self.On24ImageFlag_AGC=self.n['On24ImageFlag_2']
        self.On24ImageFlag=self.n['On24ImageFlag_1']
        self.supervopt=self.n.SUPERVOPT
        self.ra=self.n.RA
        self.dec=self.n.DEC

        self.sdssabsmag=self.n.SDSSABSMAG
        self.sdssMu=self.sdssabsmag[:,0]
        self.sdssMg=self.sdssabsmag[:,1]
        self.sdssMr=self.sdssabsmag[:,2]
        self.sdssMi=self.sdssabsmag[:,3]
        self.sdssMz=self.sdssabsmag[:,4]


        self.sdsslum=self.n.SDSSLUM
        self.sdssLu=self.sdsslum[:,0]
        self.sdssLg=self.sdsslum[:,1]
        self.sdssLr=self.sdsslum[:,2]
        self.sdssLi=self.sdsslum[:,3]
        self.sdssLz=self.sdsslum[:,4]

        self.sdssabsmag_cl=self.n.SDSSABSMAG_CL
        self.sdssMu=self.sdssabsmag_cl[:,0]
        self.sdssMg=self.sdssabsmag_cl[:,1]
        self.sdssMr=self.sdssabsmag_cl[:,2]
        self.sdssMi=self.sdssabsmag_cl[:,3]
        self.sdssMz=self.sdssabsmag_cl[:,4]

        self.sdsslum_cl=self.n.SDSSLUM_CL
        self.sdssLu_cl=self.sdsslum_cl[:,0]
        self.sdssLg_cl=self.sdsslum_cl[:,1]
        self.sdssLr_cl=self.sdsslum_cl[:,2]
        self.sdssLi_cl=self.sdsslum_cl[:,3]
        self.sdssLz_cl=self.sdsslum_cl[:,4]

        self.sdsscolc=self.n.SDSSCOLC
        self.sdssrowc=self.n.SDSSROWC

        self.membflag =self.n.MEMBFLAG
        self.morphflag =self.n.MORPHFLAG
        self.morph =self.n.MORPH
        self.disturb =self.n.DISTURB
        self.agn1 =self.n.AGNKAUFF 
        self.agn2 =self.n.AGNKEWLEY
        self.agn3 =self.n.AGNSTASIN
        self.n2halpha=(self.mpan2/self.mpahalpha)
        self.o3hbeta=(self.mpao3/self.mpahbeta)
        self.logn2halpha=log10(self.mpan2/self.mpahalpha)
        self.logo3hbeta=log10(self.mpao3/self.mpahbeta)
        self.ellipseflag24 =self.n.ELLIPSEFLAG24
        self.ellipseflagsdss =self.n.ELLIPSEFLAGSDSS
        self.ellipseflag =self.n.ELLIPSEFLAG


        #new SDSS fields that quantify radial extent of galaxy
        self.sdssIsoAr =self.n.SDSSISOAR
        self.sdssIsoBr =self.n.SDSSISOBR
        self.sdssIsoPhir =self.n.SDSSISOPHIR
        self.sdssIsoPhirErr =self.n.SDSSISOPHIERRR
        self.sdssExpRadr =self.n.SDSSEXPRADR
        self.sdssExpABr =self.n.SDSSEXPABR
        self.sdssExpABrErr =self.n.SDSSEXPABRERR
        self.sdssExpPhir =self.n.SDSSEXPPHIR
        self.sdssExpPhirErr =self.n.SDSSEXPPHIERRR

        self.sdssPetroMag=self.n.SDSSPETROMAG
        self.sdssPetroMagr=self.sdssPetroMag[:,2]

        self.sdssPetroRad=self.n.SDSSPETRORAD
        self.sdssPetroRadr=self.sdssPetroRad[:,2]

        self.sdssPetroR50=self.n.SDSSPETROR50
        self.sdssPetroR50r=self.sdssPetroR50[:,2]

        self.sdssPetroR90=self.n.SDSSPETROR90
        self.sdssPetroR90r=self.sdssPetroR90[:,2]

        #de-redened magnitudes
        self.sdssdered=self.n.SDSSDERED
        self.sdssumag=self.sdssdered[:,0]
        self.sdssgmag=self.sdssdered[:,1]
        self.sdssrmag=self.sdssdered[:,2]
        self.sdssimag=self.sdssdered[:,3]
        self.sdsszmag=self.sdssdered[:,4]


        conv=4*pi*(self.cdcm**2)*1.e-6*1.e-23*(3.e8/24.e-6)/Lsol
        print 'conversion from F24 to L24 = ',conv
        self.L24_cl=array(self.sex24.FLUX_BEST*conv*mipsconv_MJysr_to_uJy,'d')
        #self.L24_cl=array(t,'d')
        self.L24err_cl=self.sex24.FLUXERR_BEST*conv*mipsconv_MJysr_to_uJy
        self.Lir_cl=self.L24_cl*8.#approx conv from papovich
        self.Lirerr_cl=self.L24err_cl*8.#approx conv from papovich
        self.SFR24_cl=self.Lir_cl*4.5e-44*Lsol#approx conv from papovich
        self.SFR24err_cl=self.Lirerr_cl*4.5e-44*Lsol#approx conv from papovich

        #self.SFR24se_cl=(fluxautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
        #self.SFR24seerr_cl=(fluxerrautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
        #self.snr24se=abs(self.SFR24se_cl/self.SFR24seerr_cl)
        #self.superSFR24_cl=self.SFR24_cl*apexflag+self.SFR24se_cl*(~apexflag&sex24flag)
        #self.superSFR24err_cl=self.SFR24err_cl*apexflag+self.SFR24seerr_cl*(~apexflag&sex24flag)
        #self.sSFR_cl=self.SFR24_cl/self.stellarmassr_cl	    
            
        # other Lum and SFR
        self.HImass=self.n.HIMASS
        self.L24=self.n.L24
        self.L24err=self.n.L24ERR
        self.Lir=self.n.LIR
        self.Lirerr=self.n.LIRERR
        self.SFR24=self.n.SFR24
        self.SFR24err=self.n.SFR24ERR
        self.SuperSFR24=self.n.SUPERSFR24
        self.SuperSFR24err=self.n.SUPERSFR24ERR

        self.HImass_cl=self.n.HIMASS_CL
        #self.L24_cl=self.n.L24_CL
        #self.L24err_cl=self.n.L24ERR_CL
        #self.Lir_cl=self.n.LIR_CL
        #self.Lirerr_cl=self.n.LIRERR_CL
        #self.SFR24_cl=self.n.SFR24_CL
        #self.SFR24err_cl=self.n.SFR24ERR_CL
        #self.SuperSFR24_cl=self.n.SUPERSFR24_CL
        #self.SuperSFR24err_cl=self.n.SUPERSFR24ERR_CL

        
        #define red, green and blue galaxies
        ur=self.n.ABSMAG[:,2]-self.n.ABSMAG[:,4]
        self.redflag=(ur > 2.3)
        self.greenflag=(ur > 1.8) & (ur < 2.3)
        self.blueflag=(ur<1.8)
        #end of master table!
        #self.spiralFlag=self.On24ImageFlag & self.galzooflag & self.ellipseflag & (self.galzoopcsdebiased > 0.6)

        self.clustername=clustername
        self.clusterra=clusterRA[clustername]
        self.clusterdec=clusterDec[clustername]
        self.dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        self.drR200=self.dr/self.r200deg
        self.clustervel=clusterbiweightcenter[clustername]
        self.clustersigma=clusterbiweightscale[clustername]
        self.clustervmin=self.clustervel-3.*self.clustersigma
        self.clustervmax=self.clustervel+3.*self.clustersigma

        self.dist=sqrt((self.clusterra-self.ra)**2 + (self.clusterdec-self.dec)**2)
        self.flagHI = (self.flux100 > 0.)
        self.vopt=self.n.Z*3.e5

        self.dv=abs(self.vopt-self.biweightvel)/self.biweightscale

        #self.dvflag = ((self.vopt > self.clustervmin) & (self.vopt < self.clustervmax))
        self.dvflag = (self.dv < 3.)

        self.dist3d=sqrt(self.drR200**2 + (self.dv)**2)

        self.membflag=self.dvflag & (self.drR200 < 1)

        self.nmemb=len(self.dist[self.membflag & self.On24ImageFlag])
        self.nfield=len(self.dist[self.On24ImageFlag])-self.nmemb

        self.agcdict=dict((a,b) for a,b in zip(self.agcnumber,arange(len(self.agcnumber))))
        self.nsadict=dict((a,b) for a,b in zip(self.n.NSAID,arange(len(self.n.NSAID))))


        self.spiralflag=(self.zoo.p_cs > 0.6)
        self.ellipticalflag=(self.zoo.p_el > 0.7)

        # set spiral flag for galaxies that are clearly spiral but have zoo.p_cs < .6
        slist=spiral_nozoo[self.prefix]
        for i in slist:
            self.spiralflag[self.nsadict[i]]=1
        elist=elliptical_nozoo[self.prefix]
        for i in elist:
            self.ellipticalflag[self.nsadict[i]]=1

        print self.clustername,": ","N members = ",self.nmemb," N field = ",self.nfield
        #print ' N spirals = ',sum(self.spiralFlag),' Nspiral members = ',sum(self.spiralFlag&self.membflag)
        print ' N spirals on 24um image = ',sum(self.spiralflag & self.On24ImageFlag),' Nspiral members = ',sum(self.spiralflag&self.membflag & self.On24ImageFlag)
        print ' N galaxies on 24um image = ',sum(self.On24ImageFlag),' N members on 24um image = ',sum(self.membflag & self.On24ImageFlag)



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
        self.D25obskpc=2.*self.sdssIsoAr*sdsspixelscale*self.AngDistanceCl
        self.D25obskpcall=2.*self.sdssIsoAr*sdsspixelscale*self.AngDistanceAll
        # apply correction from toribio et al 2011 
        self.logD25kpc=log10(self.D25obskpc) + 0.35*log10(b/a)
        self.logD25kpcall=log10(self.D25obskpcall) + 0.35*log10(b/a)
        # use toribio et al relation to predict the expected HI mass, including factor of 2 correction
        self.HImassExpected = 10.**(8.72 + 1.25*(self.logD25kpc-log10(2.)))
        self.HImassExpectedall = 10.**(8.72 + 1.25*(self.logD25kpcall-log10(2.)))
        self.HImassExpFromMr=10.**(6.44-0.18*self.sdssMr)
        self.HImassExpFromgr=10.**(8.84+1.81*(self.sdssgmag-self.sdssrmag))
        # calculate deficiency as log expected - log observed
        self.HIDef = log10(self.HImassExpected) - log10(self.HImass)
        self.HIDef_zdist = log10(self.HImassExpectedall) - log10(self.HImass)
        self.myHIDef = log10(self.HImassExpected -self.HImass)

        # HI upper limit for clusters observed in alpha 40 sample (MKW11, A2052, A2063)
        # M(HI lim) = 2.36e5 * d^2 (W50 * 20)^(1/2) sigma_rms * S/N lim
        # W50 = W50,0 * sin(i)
        # cos^2(i) = ((b/a)^2 - q^2)/(1-q^2)
        # W50,0 = 100 km/s
        # sigma rms = 10 km/s
        # q = 0.32 for Sa
        # q = 0.23 for Sab
        # q = 0.18 for Sb-Sc  (from Gavazzi et al 2013)
        # assign upper limits for each galaxy based on its B/A
        W50=70.
        sigma_rms=2.1e-3
        snlim=6.5
        self.clHImass_ulim= 2.36e5*(self.cdMpc)**2*sqrt(W50*20.)*sigma_rms*snlim
        self.HIDef_ulim=log10(self.HImassExpected)-log10(self.clHImass_ulim)


        self.agnflag=self.agn1#use Kauffmann et al 2003 cut
        #self.agnflag=self.agn2#use Kewley
        self.irflag = self.apexflag  & self.membflag
        #set flag fot galaxies with dv < 3 sigma

        writesdssradec=0
        if writesdssradec:
            # extra function for writing out RA and Dec to upload into SDSS query page
            # http://skyserver.sdss3.org/dr9/en/tools/search/IQS.asp
            offset=30./60.
            outfile=homedir+'research/LocalClusters/NSAmastertables/RADecfiles/'+self.prefix+'_RADEC.csv'
            out1=open(outfile,'w')
            out1.write('ra, dec, sep \n')
            for i in range(len(self.n.RA)):
                out1.write('%12.8f, %12.8f, %12.8f \n'%(self.n.RA[i],self.n.DEC[i],offset))
            out1.close

        # calculate the angle between major axis and ellipse center
        # call is clusterphi (see page in ResearchNotes.nb entitles 'Angle between galaxy's major axis and cluster center')
        # in notes I call the angle psi, but a little too painful for a variable
        # if cluster_phi = 90 - galaxy falling face-on into cluster
        # if cluster_phi = 0  - galaxy falling edge-on into cluster
        self.delta_dec=self.dec-self.cdec
        self.delta_ra=self.ra-self.cra
        r=sqrt(self.delta_dec**2+self.delta_ra**2)
        self.theta=arctan(abs(self.delta_dec)/abs(self.delta_ra))*180./pi
        self.cluster_phi=zeros(len(self.ra),'f')
        #for i in range(len(self.ra)):
        #    if (self.delta_dec[i]*self.delta_ra[i] > 1.):
        #        self.cluster_phi[i] = abs(90.- self.n.SERSIC_PHI[i] + self.theta[i])
        #    else:
        #        self.cluster_phi[i] = abs(self.n.SERSIC_PHI[i] - 90. + self.theta[i])
        phi=self.n.SERSIC_PHI/180.*pi # in radians
        theta=self.theta/180.*pi # convert back to radians
        self.cluster_phi=arccos(abs(sin(phi)*-1.*cos(theta)+cos(phi)*sin(theta)))*180./pi
                                

        self.readsnr24NSA()

        self.snr24flag=self.snr24 > 1.5


        # define a blue flag based on NUV-r vs mass plot

        color=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]
        self.NUVr_color=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]
        #mag=log10(self.stellarmass)

        #self.blueflag = color < (.6*(mag-9)+3.3)

        # define a blue flag based on NUV-r vs Mr plot
        mag=self.n.ABSMAG[:,4]

        #yb=1.73-0.17*xl-2.
        self.blueflag = color < (1.73-0.17*mag-2.)
        self.redflag = color > (1.73-0.17*mag-.5)
        self.greenflag= (color > (1.73-0.17*mag-2.)) & (color < (1.73-0.17*mag-.5))

        # 80% completeness
        self.f80=clusterf80MJysr[self.prefix]*mipsconv_MJysr_to_uJy # in micro Jy
        print self.biweightvel/3.e5,self.f80
        self.Lir80,self.SFRir80=chary.chary_elbaz_24um(array([self.biweightvel/3.e5],'f'),array([self.f80],'f'))

        # 
        best_distance=self.membflag * self.cdMpc + ~self.membflag*(self.n.ZDIST*3.e5/H0)
        self.M24=self.sex24.MAG_BEST-25-5.*log10(best_distance)
        self.M24zdist=self.sex24.MAG_BEST-25-5.*log10(self.n.ZDIST*3.e5/H0)

    def readsnr24NSA(self):
        infile=homedir+'research/LocalClusters/NSAmastertables/SNR24/'+self.prefix+'_snr24NSA.dat'
        snrdat=ascii.read(infile)
        self.f24NSA=snrdat['col1']
        self.f24NSAerr=snrdat['col2']
        self.snr24=snrdat['col3']

