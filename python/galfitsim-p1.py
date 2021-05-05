#!/usr/bin/env python


"""
    PURPOSE: 
      to understand how well galfit recovers galaxy parameters as a function of galaxy brightness
      
    PROCEDURE
      create range of galaxies using galfit
      add galaxies to blank regions on MIPS images
      run galfit to see how we are able to recover galaxy parameters

    CALLING SEQUENCE

      to run galfit on the 24um images:

      mkw11.run_just_get_images24(make_mask=1,review_mask=1)
      mkw11.rungalfit_first_time_24()
      mkw11.write_galfit_sersic_parameters_24()

      To run galfit on the sdss das images:

      mkw11.just_get_images()
      mkw11.rungalfit_first_time_sdss()
      mkw11.write_galfit_sersic_parameters_sdss()


      to run galfit on one 24um image
      %run ~/svnRepository/pythonCode/LCSrungalfit.py
      mkw11=Cluster('MKW11')
      set_ntimes(1)
      mkw11.get_images24(mkw11.nsadict[NSAID],0)

    REQUIRED PYTHON MODULES
        scipy
        pylab
        atpy
        
    ADDITIONAL REQUIRED MODULES
        astrofuncs.py

    NOTES
      2013-05-01 - implementing simulations to test reliability of 24um parameters
      2014-04-13 - changing program to measure SNR with sextractor, as I now do for real images

      2014-05-13 - changing to expand sampling of flux vs surface brightness parameter space
                 - want to be able to quantify SB and flux limits for which galfit can still:
                   (1) run successfully, and (2) recover input parameter
                 - will increase max value of Re to 40 pix and max magnitude to 16.

      2014-05-17 - updating to run galfit first w/out convolution, then w/convolution 

      2016-01-02 - updating to run galfit with fixed BA and PA
      
"""

from pylab import *
import os
import shutil
from scipy.interpolate import interp1d
#from scipy.optimize import leastsq
import scipy
#import urllib
import numpy as np
import glob
from pyraf import iraf
import aplpy
#import numpy as np
import pyfits
import ds9
import atpy

from LCSReadmasterBaseNSA import *

from LCScommon import *
save_ds9_fig=0

# from individual bcd image
GAIN=5 # e-/SN conversion
FLUXCONV=.0447 # DN/s to BUNIT
EXPT=2.62 # expt per scan

# model parameters
minmodelre_pixels = .5 # min re of model in pixels
maxmodelre_pixels = 10. # min re of model in pixels
minmodelmag=11.5
maxmodelmag=16.


max_cutout_radius=20. # in arcsec, cutout will be +/- 3 times this size

# trying to make sim galaxies bigger to see if sersic index is better reproduced.
min_cutout_size=80. # in arcsec
max_cutout_size=120.# in arcsec

sbcut={'MKW11':19.8, 'MKW8':20, 'AWM4':20, 'A2063':19.8, 'A2052':20., 'NGC6107':19.8, 'Coma':20.5, 'A1367':20., 'Hercules':19.7}
sbcut={'MKW11':20., 'MKW8':20, 'AWM4':20, 'A2063':20., 'A2052':20., 'NGC6107':20., 'Coma':20., 'A1367':20., 'Hercules':20.}

try:
    mypath=os.getcwd()
    if mypath.find('Users') > -1:
        print("Running on Rose's mac pro")
        homedir='/Users/rfinn/'
    elif mypath.find('home') > -1:
        print("Running on coma")
        homedir='/home/rfinn/'
except:
    homedir='/Users/rfinn/'
npoints=200
dxstar=15


iraf.digiphot()
iraf.daophot()


def getpositionsLCS(ximage,yimage,im,coverage_map,npoints):
    dmin=15
    iraf.imgets(image=im,param='naxis1')#get RA of image
    t=float(iraf.imgets.value)
    xmax=t
    xcenter=t/2.
    iraf.imgets(image=im,param='naxis2')#get RA of image
    t=float(iraf.imgets.value)
    ymax=t
    ycenter=t/2.
    xpos=[]
    ypos=[]
    d = zeros(len(ximage),'f')
    #print len(d)
    i=0
    covimage=pyfits.open(coverage_map)
    cov_data=(covimage[0].data)
    covimage.close()

    while i < npoints:
        xtemp=np.random.uniform(0,1)*xmax
        ytemp=np.random.uniform(0,1)*ymax
        # check coverage map to make sure point is on science area
        # coverage map values > 6 are ok
        if cov_data[ytemp,xtemp] > 10:
	    d = sqrt((ximage-xtemp)**2+(yimage-ytemp)**2)
            if (min(d) > dmin):       
                xpos.append(xtemp)
                ypos.append(ytemp)
                i=i+1
                #print 'found a good place to measure sky!',i,npoints
    xpos=array(xpos,'f')
    ypos=array(ypos,'f')
    return xpos,ypos


class Cluster(baseClusterNSA):
    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)
        self.galflag=zeros([len(self.ra),3],'i')
        self.galflag24=zeros([len(self.ra),3],'i')
        self.galflag_too_faint24=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.galflag_too_faint=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.galflag_stop=zeros(len(self.ra),'i') # if fitting was stopped, either b/c too faint or results were not reasonable
        self.galflag_stop24=zeros(len(self.ra),'i') # 
        self.working_dir_sdss=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'
        self.working_dir_nsa=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'
        self.working_dir_mips=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        self.galfit_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/'
        self.diskflag=zeros([len(self.ra),6],'i') # flag for 1comp-1 2comp-1 2comp-2 3comp-1 3comp-2 3comp-3
        self.diskflag24=zeros([len(self.ra),6],'i') # flag for 1comp-1 2comp-1 2comp-2 3comp-1 3comp-2 3comp-3
        self.galfit_sdssR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_sdssR90=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_mipsR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_mipsR90=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits

        self.xcenter_cutout=zeros(len(self.ra),'f')
        self.ycenter_cutout=zeros(len(self.ra),'f') 
        self.xcenter_cutout24=zeros(len(self.ra),'f')
        self.ycenter_cutout24=zeros(len(self.ra),'f') 
        ##############################################
        # MAG ZP
        ##############################################
        # define mag zp for 24um image scans
        flux_zp_AB = 3631. # in Jy
        flux_zp_Vega = 7.17 # in Jy
        flux_zp=flux_zp_AB
        
        # conversion from image units of MJ/sr to micro-Jy (1 sq arcsec = 2.3504e-11 sr)
        conv_MJysr_uJy = 23.5045*(2.45**2)
        self.magzp24=2.5*log10(flux_zp*1.e6/conv_MJysr_uJy)


        self.mipsprf=atpy.Table(homedir+'research/LocalClusters/GalfitAnalysis/MIPSpsf/mips24_prf.dat',type='ascii')


        #self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        #self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_unc.fits'


        #self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-scaled24.fits'
        #self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-scalednoise.fits'
        #self.zp24offset=5.

        self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        #self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_std.fits'
        self.mosaic24unc=homedir+'research/LocalClusters/MIPS/rawdata/'+self.prefix+'/FullMosaic/mosaic_noise.fits'
        self.mosaic24cov=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_cov.fits'
        self.zp24offset=0.
        #self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/PRF.fits'
        #self.psf_oversampling=4


        #self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_star_byhand.fits'
        # copied above file to the following to see if filename was too long
        
        if self.prefix.find('Herc') > -1:
            self.psf_image='/Users/rfinn/research/LocalClusters/PRF/mips24_prf_mosaic_2.45_4x.fits'
            self.psf_oversampling=4
        else:
            self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_starv2.fits'        
            self.psf_oversampling=1

        self.numberofstars=numberofstars[self.prefix]
        #if self.prefix.find('Herc') > -1:
        #    self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'psf_starv2.fits'
        #self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_star.fits'
        #self.psf_oversampling=1

        #self.psf_image='/Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits'
        #self.psf_oversampling=4



        #npoints=sum(self.spiralflag & self.On24ImageFlag)
        #xsim,ysim=getpositionsLCS(self.sex24.X_IMAGE,self.sex24.Y_IMAGE,self.mosaic24,self.mosaic24cov,npoints)
        #self.xsim=zeros(len(self.ra))
        #self.ysim=zeros(len(self.ra))
        #self.xsim[self.spiralflag & self.On24ImageFlag] = xsim
        #self.ysim[self.spiralflag & self.On24ImageFlag] = ysim

    def write_galfit_sersic_parameters_24(self):
        usepyfits=1
        nmax=npoints
        narray=npoints

        self.galfit_xc24=zeros((narray,2),'f')
        self.galfit_yc24=zeros((narray,2),'f')
        self.galfit_mag24=zeros((narray,2),'f')
        self.galfit_Re24=zeros((narray,2),'f')
        self.galfit_Nsersic24=zeros((narray,2),'f')
        self.galfit_axisratio24=zeros((narray,2),'f')
        self.galfit_PA24=zeros((narray,2),'f')
        self.galfit_flag=zeros(narray,'bool')
        numerical_error_flag24=zeros(narray,'bool')

        self.galfit_ncxc24=zeros((narray,2),'f')
        self.galfit_ncyc24=zeros((narray,2),'f')
        self.galfit_ncmag24=zeros((narray,2),'f')
        self.galfit_ncRe24=zeros((narray,2),'f')
        self.galfit_ncNsersic24=zeros((narray,2),'f')
        self.galfit_ncaxisratio24=zeros((narray,2),'f')
        self.galfit_ncPA24=zeros((narray,2),'f')
        self.galfit_ncflag=zeros(narray,'bool')
        ncnumerical_error_flag24=zeros(narray,'bool')
# fit parameters when asymmetry is turned on
        self.galfit_axc24=zeros((narray,2),'f')
        self.galfit_ayc24=zeros((narray,2),'f')
        self.galfit_amag24=zeros((narray,2),'f')
        self.galfit_aRe24=zeros((narray,2),'f')
        self.galfit_aNsersic24=zeros((narray,2),'f')
        self.galfit_aaxisratio24=zeros((narray,2),'f')
        self.galfit_aPA24=zeros((narray,2),'f')
        self.galfit_aF1=zeros((narray,2),'f')
        self.galfit_aF1PA=zeros((narray,2),'f')
        anumerical_error_flag24=zeros(narray,'bool')
        f24=zeros(narray,'f')
        f24err=zeros(narray,'f')
        snr24=zeros(narray,'f')
        isoarea=zeros(narray,'f')
        for i in range(nmax):
            counter='%02i'%(i)
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+self.prefix+'-'+counter+'-24-1Comp-galfit-out.fits'
            output_image24_noconv=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+self.prefix+'-'+counter+'-24-1Comp-noconv-galfit-out.fits'

            if os.path.exists(output_image24_noconv):
                xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,ncnumerical_error_flag24[i]=parse_galfit_1comp(output_image24_noconv+'[2]')
                self.galfit_ncxc24[i]=xc[0],xc[1]
                self.galfit_ncyc24[i]=yc[0],yc[1]
                self.galfit_ncmag24[i]=mag[0],mag[1]
                self.galfit_ncNsersic24[i]=Nsersic[0],Nsersic[1]
                self.galfit_ncRe24[i]=Re[0],Re[1]
                self.galfit_ncaxisratio24[i]=axis_ratio[0],axis_ratio[1]
                self.galfit_ncPA24[i]=PA[0],PA[1]
                if xc[1] > .001:
                    self.galfit_ncflag[i] = 1

            if os.path.exists(output_image24):
                xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24[i]=parse_galfit_1comp(output_image24+'[2]')
                self.galfit_xc24[i]=xc[0],xc[1]
                self.galfit_yc24[i]=yc[0],yc[1]
                self.galfit_mag24[i]=mag[0],mag[1]
                self.galfit_Nsersic24[i]=Nsersic[0],Nsersic[1]
                self.galfit_Re24[i]=Re[0],Re[1]
                self.galfit_axisratio24[i]=axis_ratio[0],axis_ratio[1]
                self.galfit_PA24[i]=PA[0],PA[1]
                if xc[1] > .001:
                    self.galfit_flag[i] = 1
                

            # get fit parameters when asymmetry parameter is turned on
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+self.prefix+'-'+counter+'-24-1Comp-galfit-out-asym.fits'

            if os.path.exists(output_image24):
                xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,F1,F1PA,anumerical_error_flag24[i]=parse_galfit_1comp(output_image24+'[2]',asymflag=1)
                self.galfit_axc24[i]=xc[0],xc[1]
                self.galfit_ayc24[i]=yc[0],yc[1]
                self.galfit_amag24[i]=mag[0],mag[1]
                self.galfit_aNsersic24[i]=Nsersic[0],Nsersic[1]
                self.galfit_aRe24[i]=Re[0],Re[1]
                self.galfit_aaxisratio24[i]=axis_ratio[0],axis_ratio[1]
                self.galfit_aPA24[i]=PA[0],PA[1]
                self.galfit_aF1[i]=F1[0],F1[1]
                self.galfit_aF1PA[i]=F1PA[0],F1PA[1]

            print(i)
            #snrindex=self.snr24dict[i]
            #f24[i]=self.f24[snrindex]
            #f24err[i]=self.f24err[snrindex]
            #snr24[i]=self.snr24[snrindex]
            #snr24[i]=self.seiso_snr24[snrindex]
            #snrindex=self.snr24dict[i]
            try:
                f24[i]=self.sef24[i]
                f24err[i]=self.sef24err[i]
            #snr24[i]=self.snr24[snrindex]
                snr24[i]=self.seiso_snr24[i]
                isoarea[i]=self.seiso_area[i]
            except:
                print('no SE data - oh well!')

        output24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_GalfitSersicParam_24_sim.fits'
        if usepyfits:
            print('using pyfits')

            gtab24=atpy.Table()
            c1 = pyfits.Column(name='modelmag', array=self.model_mag,format='E')
            c2 = pyfits.Column(name='ximage', array=self.xsim,format='E')
            c3 = pyfits.Column(name='yimage', array=self.ysim,format='E')
            c2a = pyfits.Column(name='modelxc', array=self.model_xc,format='E')
            c3a = pyfits.Column(name='modelyc', array=self.model_yc,format='E')
            c4 = pyfits.Column(name='modelre', array=self.model_re,format='E')
            c5 = pyfits.Column(name='modelpa', array=self.model_PA,format='E')
            c6 = pyfits.Column(name='modelba', array=self.model_BA,format='E')
            c7 = pyfits.Column(name='modeln', array=self.model_n,format='E')
            c3c = pyfits.Column(name='inputmag', array=self.inmag,format='E')
            c4c = pyfits.Column(name='inputre', array=self.inre,format='E')
            c5c = pyfits.Column(name='inputpa', array=self.inpa,format='E')
            c6c = pyfits.Column(name='inputba', array=self.inba,format='E')
            c7c = pyfits.Column(name='inputn', array=self.model_n,format='E')
            c7a = pyfits.Column(name='galfit_flag', array=self.galfit_ncflag,format='L')
            c7b = pyfits.Column(name='cgalfit_flag', array=self.galfit_flag,format='L')
            
            c8 = pyfits.Column(name='xc1', array=self.galfit_ncxc24[:,0],format='E')
            c9 = pyfits.Column(name='xc1err', array=self.galfit_ncxc24[:,1],format='E')
            c10 = pyfits.Column(name='yc1', array=self.galfit_ncyc24[:,0],format='E')
            c11 = pyfits.Column(name='yc1err', array=self.galfit_ncyc24[:,1],format='E')
            c12 = pyfits.Column(name='mag1', array=self.galfit_ncmag24[:,0],format='E')
            c13 = pyfits.Column(name='mag1err', array=self.galfit_ncmag24[:,1],format='E')
            c14 = pyfits.Column(name='re1', array=self.galfit_ncRe24[:,0],format='E')
            c15 = pyfits.Column(name='re1err', array=self.galfit_ncRe24[:,1],format='E')
            c16 = pyfits.Column(name='nsersic1', array=self.galfit_ncNsersic24[:,0],format='E')
            c17 = pyfits.Column(name='nsersic1err', array=self.galfit_ncNsersic24[:,1],format='E')
            c18 = pyfits.Column(name='axisratio1', array=self.galfit_ncaxisratio24[:,0],format='E')
            c19 = pyfits.Column(name='axisratio1err', array=self.galfit_ncaxisratio24[:,1],format='E')
            c20 = pyfits.Column(name='pa1', array=self.galfit_ncPA24[:,0],format='E')
            c21 = pyfits.Column(name='pa1err', array=self.galfit_ncPA24[:,1],format='E')
            c22 = pyfits.Column(name='numerical_error_flag24', array=ncnumerical_error_flag24,format='E')

            cc8 = pyfits.Column(name='cxc1', array=self.galfit_xc24[:,0],format='E')
            cc9 = pyfits.Column(name='cxc1err', array=self.galfit_xc24[:,1],format='E')
            cc10 = pyfits.Column(name='cyc1', array=self.galfit_yc24[:,0],format='E')
            cc11 = pyfits.Column(name='cyc1err', array=self.galfit_yc24[:,1],format='E')
            cc12 = pyfits.Column(name='cmag1', array=self.galfit_mag24[:,0],format='E')
            cc13 = pyfits.Column(name='cmag1err', array=self.galfit_mag24[:,1],format='E')
            cc14 = pyfits.Column(name='cre1', array=self.galfit_Re24[:,0],format='E')
            cc15 = pyfits.Column(name='cre1err', array=self.galfit_Re24[:,1],format='E')
            cc16 = pyfits.Column(name='cnsersic1', array=self.galfit_Nsersic24[:,0],format='E')
            cc17 = pyfits.Column(name='cnsersic1err', array=self.galfit_Nsersic24[:,1],format='E')
            cc18 = pyfits.Column(name='caxisratio1', array=self.galfit_axisratio24[:,0],format='E')
            cc19 = pyfits.Column(name='caxisratio1err', array=self.galfit_axisratio24[:,1],format='E')
            cc20 = pyfits.Column(name='cpa1', array=self.galfit_PA24[:,0],format='E')
            cc21 = pyfits.Column(name='cpa1err', array=self.galfit_PA24[:,1],format='E')
            cc22 = pyfits.Column(name='cnumerical_error_flag24', array=numerical_error_flag24,format='E')

            c23 = pyfits.Column(name='asym_xc1', array=self.galfit_axc24[:,0],format='E')
            c24 = pyfits.Column(name='asym_xc1err', array=self.galfit_axc24[:,1],format='E')
            c25 = pyfits.Column(name='asym_yc1', array=self.galfit_ayc24[:,0],format='E')
            c26 = pyfits.Column(name='asym_yc1err', array=self.galfit_ayc24[:,1],format='E')
            c27 = pyfits.Column(name='asym_mag1', array=self.galfit_amag24[:,0],format='E')
            c28 = pyfits.Column(name='asym_mag1err', array=self.galfit_amag24[:,1],format='E')
            c29 = pyfits.Column(name='asym_re1', array=self.galfit_aRe24[:,0],format='E')
            c30 = pyfits.Column(name='asym_re1err', array=self.galfit_aRe24[:,1],format='E')
            c31 = pyfits.Column(name='asym_nsersic1', array=self.galfit_aNsersic24[:,0],format='E')
            c32 = pyfits.Column(name='asym_nsersic1err', array=self.galfit_aNsersic24[:,1],format='E')
            c33 = pyfits.Column(name='asym_axisratio1', array=self.galfit_aaxisratio24[:,0],format='E')
            c34 = pyfits.Column(name='asym_axisratio1err', array=self.galfit_aaxisratio24[:,1],format='E')
            c35 = pyfits.Column(name='asym_pa1', array=self.galfit_aPA24[:,0],format='E')
            c36 = pyfits.Column(name='asym_pa1err', array=self.galfit_aPA24[:,1],format='E')
            c37 = pyfits.Column(name='asym_f1', array=self.galfit_aF1[:,0],format='E')
            c38 = pyfits.Column(name='asym_f1err', array=self.galfit_aF1[:,0],format='E')
            c39 = pyfits.Column(name='asym_f1pa', array=self.galfit_aF1PA[:,0],format='E')
            c40 = pyfits.Column(name='asym_f1paerr', array=self.galfit_aF1PA[:,0],format='E')
            c41 = pyfits.Column(name='asym_numerical_error_flag24', array=anumerical_error_flag24,format='E')
            c42 = pyfits.Column(name='f24', array=f24,format='E')
            c43 = pyfits.Column(name='f24err', array=f24err,format='E')
            c44 = pyfits.Column(name='snr24', array=snr24,format='E')
            c45 = pyfits.Column(name='mu', array=self.galfit_mag24[:,0] + 2.5*log10(pi*(self.galfit_Re24[:,0]**2)),format='E')
            c45 = pyfits.Column(name='isoarea', array=isoarea,format='E')

            mastertb=pyfits.new_table([c1,c2,c3,c2a,c3a,c4,c5,c6,c7,c3c,c4c,c5c,c6c,c7c,c7a,c7b,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,cc8,cc9,cc10,cc11,cc12,cc13,cc14,cc15,cc16,cc17,cc18,cc19,cc20,cc21,cc22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45])

            mastertb.writeto(output24,clobber='yes')

            
        else:

            gtab24=atpy.Table()
            gtab24.add_column('modelmag',self.model_mag)
            gtab24.add_column('ximage',self.xsim)
            gtab24.add_column('yimage',self.ysim)
            gtab24.add_column('modelre',self.model_re)
            gtab24.add_column('modelpa',self.model_PA)
            gtab24.add_column('modelba',self.model_BA)
            gtab24.add_column('modeln',self.model_n)
            
            gtab24.add_column('xc1',self.galfit_xc24[:,0])
            gtab24.add_column('xc1err',self.galfit_xc24[:,1])
            gtab24.add_column('yc1',self.galfit_yc24[:,0])
            gtab24.add_column('yc1err',self.galfit_yc24[:,1])
            gtab24.add_column('mag1',self.galfit_mag24[:,0])
            gtab24.add_column('mag1err',self.galfit_mag24[:,1])
            gtab24.add_column('re1',self.galfit_Re24[:,0])
            gtab24.add_column('re1err',self.galfit_Re24[:,1])
            gtab24.add_column('nsersic1',self.galfit_Nsersic24[:,0])
            gtab24.add_column('nsersic1err',self.galfit_Nsersic24[:,1])
            gtab24.add_column('axisratio1',self.galfit_axisratio24[:,0])
            gtab24.add_column('axisratio1err',self.galfit_axisratio24[:,1])
            gtab24.add_column('pa1',self.galfit_PA24[:,0])
            gtab24.add_column('pa1err',self.galfit_PA24[:,1])
            gtab24.add_column('numerical_error_flag24',numerical_error_flag24)

            gtab24.add_column('asym_xc1',self.galfit_axc24[:,0])
            gtab24.add_column('asym_xc1err',self.galfit_axc24[:,1])
            gtab24.add_column('asym_yc1',self.galfit_ayc24[:,0])
            gtab24.add_column('asym_yc1err',self.galfit_ayc24[:,1])
            gtab24.add_column('asym_mag1',self.galfit_amag24[:,0])
            gtab24.add_column('asym_mag1err',self.galfit_amag24[:,1])
            gtab24.add_column('asym_re1',self.galfit_aRe24[:,0])
            gtab24.add_column('asym_re1err',self.galfit_aRe24[:,1])
            gtab24.add_column('asym_nsersic1',self.galfit_aNsersic24[:,0])
            gtab24.add_column('asym_nsersic1err',self.galfit_aNsersic24[:,1])
            gtab24.add_column('asym_axisratio1',self.galfit_aaxisratio24[:,0])
            gtab24.add_column('asym_axisratio1err',self.galfit_aaxisratio24[:,1])
            gtab24.add_column('asym_pa1',self.galfit_aPA24[:,0])
            gtab24.add_column('asym_pa1err',self.galfit_aPA24[:,1])
            gtab24.add_column('asym_f1',self.galfit_aF1[:,0])
            gtab24.add_column('asym_f1err',self.galfit_aF1[:,0])
            gtab24.add_column('asym_f1pa',self.galfit_aF1PA[:,0])
            gtab24.add_column('asym_f1paerr',self.galfit_aF1PA[:,0])
            gtab24.add_column('asym_numerical_error_flag24',anumerical_error_flag24)
            gtab24.add_column('f24',f24)
            gtab24.add_column('f24err',f24err)
            gtab24.add_column('snr24',snr24)
            gtab24.add_column('mu',self.galfit_mag24[:,0] - 2.5*log10(pi*(self.galfit_Re24[:,0]**2)))
            print('output file = ',output24)
            if os.path.exists(output24):
                os.remove(output24)
                gtab24.write(output24,type='fits')
        return

    def write_galfit_sersic_parameters_stars_24(self):

        files=glob.glob('*star*24-1Comp-galfit-out.fits')
        narray=len(files)
        print('number of files = ',narray)

        self.galname=zeros(narray,'i')
        self.galfit_xc24=zeros((narray,2),'f')
        self.galfit_yc24=zeros((narray,2),'f')
        self.galfit_mag24=zeros((narray,2),'f')
        self.galfit_Re24=zeros((narray,2),'f')
        self.galfit_Nsersic24=zeros((narray,2),'f')
        self.galfit_axisratio24=zeros((narray,2),'f')
        self.galfit_PA24=zeros((narray,2),'f')
        numerical_error_flag24=zeros(narray,'bool')
        for i in range(narray):
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+files[i]
            t=files[i].split('-')
            star=t[1]
            self.galname[i]=star[4:]
            if os.path.exists(output_image24):
                xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24[i]=parse_galfit_1comp(output_image24+'[2]')
                #print i, self.galfit_xc24.shape,narray
                self.galfit_xc24[i]=xc[0],xc[1]
                self.galfit_yc24[i]=yc[0],yc[1]
                self.galfit_mag24[i]=mag[0],mag[1]
                self.galfit_Nsersic24[i]=Nsersic[0],Nsersic[1]
                self.galfit_Re24[i]=Re[0],Re[1]
                self.galfit_axisratio24[i]=axis_ratio[0],axis_ratio[1]
                self.galfit_PA24[i]=PA[0],PA[1]            
        output24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_stars_GalfitSersicParam_24_sim.fits'
        gtab24=atpy.Table()
        gtab24.add_column('starID',self.galname)
        gtab24.add_column('modelmag',self.se.MAG_BEST[self.starflag])
        gtab24.add_column('modelre',self.se.FLUX_RADIUS[self.starflag][:,0])
        gtab24.add_column('modelpa',self.se.THETA_IMAGE[self.starflag])
        gtab24.add_column('modelba',self.se.B_IMAGE[self.starflag]/self.se.A_IMAGE[self.starflag])
        #print narray, 0.5*ones(narray)
        gtab24.add_column('modeln',0.5*ones(narray))
        #print self.galfit_xc24
        #print self.galfit_xc24[:,0]
        gtab24.add_column('xc1',self.galfit_xc24[:,0])
        gtab24.add_column('xc1err',self.galfit_xc24[:,1])
        gtab24.add_column('yc1',self.galfit_yc24[:,0])
        gtab24.add_column('yc1err',self.galfit_yc24[:,1])
        gtab24.add_column('mag1',self.galfit_mag24[:,0])
        gtab24.add_column('mag1err',self.galfit_mag24[:,1])
        gtab24.add_column('re1',self.galfit_Re24[:,0])
        gtab24.add_column('re1err',self.galfit_Re24[:,1])
        gtab24.add_column('nsersic1',self.galfit_Nsersic24[:,0])
        gtab24.add_column('nsersic1err',self.galfit_Nsersic24[:,1])
        gtab24.add_column('axisratio1',self.galfit_axisratio24[:,0])
        gtab24.add_column('axisratio1err',self.galfit_axisratio24[:,1])
        gtab24.add_column('pa1',self.galfit_PA24[:,0])
        gtab24.add_column('pa1err',self.galfit_PA24[:,1])
        gtab24.add_column('numerical_error_flag24',numerical_error_flag24)

        if os.path.exists(output24):
            os.remove(output24)
        gtab24.write(output24)
        return


    def plotstarprofiles(self):
        self.readsimresults(starflag=1)
        files=glob.glob(homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+'*star*24-1Comp-galfit-out.fits')
        working_dir24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        nrow=2
        ncol=3
        xticksize=10
        yticksize=10
        
        ymin=.1
        dy=.8
        dx=.25

        for i in range(len(self.sim.starID)):
        #for i in range(1):
            subcomp_image24=files[i]
            subcomp_image24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+self.prefix+'-star'+str(self.sim.starID[i])+'-24-1Comp-galfit-out.fits'
            vminsdss=-1
            vmaxsdss=1
            vminsdssapl=-100
            vmaxsdssapl=400
        
            vmin24=-40
            vmax24=10.
            subplots_adjust(left=0.05, right=.95,bottom=.1,top=0.9,wspace=0.4,hspace=0.6)
        
            fig=figure(figsize=(8,2.5))
        
            clf()
            fig=gcf()
            try:
                subf=aplpy.FITSFigure(subcomp_image24,hdu=1,figure=fig,subplot=[.08,ymin,dx,dy])
            except IOError:
                return
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
            subf.tick_labels.hide()
            t=subcomp_image24.split('/')
            t2=t[-1].split('-24')
            galname=t2[0]
            print(galname)
            s='$'+galname+'$'
            subf.add_label(.5,1.05,s,fontsize=14,relative=True)
            
        
            subf.tick_labels.set_font(size='x-small')

                
            subf=aplpy.FITSFigure(subcomp_image24,hdu=2,figure=fig,subplot=[.38,ymin,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
            subf.tick_labels.set_font(size='x-small')
            if self.sim.numerical_error_flag24[i]:
                s='$ Sersic \ param**: \ m=%5.1f,Re=%5.1f$'%(self.sim.mag1[i],self.sim.re1[i]*mipspixelscale)
            else:
                s='$ Sersic \ param: \ m=%5.1f,Re=%5.1f$'%(self.sim.mag1[i],self.sim.re1[i]*mipspixelscale)
            subf.add_label(.9,.9,s,fontsize=12,relative=True,color='yellow',horizontalalignment='right',family='serif',weight='medium')

            s='$ n=%5.1f,B/A=%4.2f,PA=%5.1f $'%(self.sim.nsersic1[i],self.sim.axisratio1[i],self.sim.pa1[i])
            subf.add_label(.9,.82,s,fontsize=12,relative=True,color='yellow',horizontalalignment='right')

            s='$ SE \ param: \ m= %5.1f,Re=%5.1f$'%((self.sim.modelmag[i]),self.sim.modelre[i]*mipspixelscale)
            subf.add_label(.88,.18,s,fontsize=12,relative=True,color='yellow',horizontalalignment='right')

            s='$B/A=%5.2f, PA=%5.1f $'%(1-self.sim.modelba[i],self.sim.modelpa[i])
            subf.add_label(.88,.1,s,fontsize=12,relative=True,color='yellow',horizontalalignment='right')

            subf.add_label(.5,1.05,'Model',fontsize=14,relative=True)
            subf=aplpy.FITSFigure(subcomp_image24,hdu=3,figure=fig,subplot=[.68,ymin,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
            subf.tick_labels.set_font(size='x-small')
            subf.add_label(.5,1.05,'Residual',fontsize=14,relative=True)
            figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+str(galname)+'-galfitResults.eps'
            savefig(figname)

            figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+str(galname)+'-galfitResults.png'
            savefig(figname)

    def runSEstars(self):
	#cp sextractor files to this director
	os.system('cp '+homedir+'research/LocalClusters/sextractor/default.param .')
	os.system('cp '+homedir+'research/LocalClusters/sextractor/default.nnw .')
	s='sex '+self.mosaic24+' -c '+homedir+'research/LocalClusters/sextractor/default.sex.24um.stars -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+self.mosaic24unc+' -CATALOG_NAME '+self.prefix+'-stars-test.fits -CATALOG_TYPE FITS_1.0'
	os.system(s)
    def readSEstars(self):
        fname=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+self.prefix+'-stars-test.fits'
        self.se=atpy.Table(fname)
        #self.starflag=(self.se.MAG_AUTO < 15) & (self.se.MAG_AUTO > 10) & (self.se.CLASS_STAR > 0.7)
        self.starflag=zeros(len(self.se.MAG_AUTO),'bool')
        for i in range(len(self.xstar)):
            	imatch, matchflag,nmatch=findnearest(self.xstar[i],self.ystar[i],self.se.X_IMAGE,self.se.Y_IMAGE,3)
                if matchflag:
                    self.starflag[imatch]=1
        print('number of stars used to make PSF = ',len(self.xstar))
        print('number of stars matched to SE cat = ',sum(self.starflag))
        if interactive:
            t=input('hit any key to continue \n')
    def readstarfile(self):
        starlist='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'-starlist.tbl'
        stars=open(starlist,'r')
        xs=[]
        ys=[]
        for line in stars:
            if line.find('\\') > -1:
                continue
            if line.find('|') > -1:
                continue
            t=line.split()
            xs.append(float(t[8]))
            ys.append(float(t[10]))
        self.xstar=array(xs,'f')
        self.ystar=array(ys,'f')

    def plotSEclass(self):
        figure()
        plot(self.se.MAG_AUTO,self.se.CLASS_STAR,'k.')
        xlabel('MAG_AUTO')
        ylabel('CLASS_STAR')
        axis([9,18,-.05,1.05])

    def runSE(self):
	#cp sextractor files to this director
	os.system('cp '+homedir+'research/LocalClusters/sextractor/default.param .')
	os.system('cp '+homedir+'research/LocalClusters/sextractor/default.nnw .')
	s='sex '+self.mosaic24+' -c '+homedir+'research/LocalClusters/sextractor/default.sex.24um.galfitsource -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+self.mosaic24unc+' -CATALOG_NAME '+self.prefix+'test.fits -CATALOG_TYPE FITS_1.0'
	os.system(s)
        os.rename(self.prefix+'test.fits',homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'SEgalfitsource.fits')
    def readSE(self):
        #fname=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'+self.prefix+'test.fits'
        fname=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'SEgalfitsource.fits'
        self.se=atpy.Table(fname)
        #self.starflag=(self.se.MAG_AUTO < 15) & (self.se.MAG_AUTO > 10) & (self.se.CLASS_STAR > 0.7)

    def get_images24(self,i,getimages, keepimages=1,starflag=0):
        quitflag=0
        # getimages:
        # set = 0 to not get images (if they already exist)
        # set = 1 if sdss images are needed
        #
        # this subroutine gets a cutouot from the mips mosaic,
        #    parses relevant input for galfit,
        #    and runs galfit three times.
        #        #
        # keepimages = 1 to keep MIPS cutout images
        # keepimages = 0 to delete MIPS cutout images (better if disk space is tight)

        ##############################################
        # MAG ZP
        ##############################################
        # define mag zp for 24um image scans
        flux_zp_AB = 3631. # in Jy
        flux_zp_Vega = 7.17 # in Jy
        flux_zp=flux_zp_AB
        
        # conversion from image units of MJ/sr to micro-Jy (1 sq arcsec = 2.3504e-11 sr)
        conv_MJysr_uJy = 23.5045*(2.45**2)
        magzp=2.5*log10(flux_zp*1.e6/conv_MJysr_uJy)

        ##############################################
        # SOME SETUP COMMANDS
        ##############################################
        # open ds9 display
        d=ds9.ds9()
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        d.set('cd '+working_dir)
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')




        # convert arrays to numbers
        if starflag:
            col=self.se.X_IMAGE[i]
            row=self.se.Y_IMAGE[i]
        else:
            col=self.xsim[i]
            row=self.ysim[i]


        #####################################################
        # GET CUTOUT OF GALAXY AND UNCERTAINTY IMAGE
        #####################################################
        # !! SHOULD UPDATE THIS TO USE THE ASTLIB LIBRARY !!
        #####################################################

        if starflag:
            dx=10
            xmin=col-dxstar
            xmax=col+dxstar
            ymin=row-dxstar
            ymax=row+dxstar
        else:
            if self.model_petro90[i] > (min_cutout_size/6.):
                PETROTH90_pixels =self.model_petro90[i]/mipspixelscale
                if self.model_petro90[i]> (max_cutout_size/6.):
                    PETROTH90_pixels =max_cutout_size/6./mipspixelscale

            else:
                PETROTH90_pixels =(min_cutout_size/6.)/mipspixelscale

            # make a cutout of the galaxy (easier than trying to run sextractor on full mosaic
            # need x and y pixel values of the galaxy
            
            xmin=col-3*PETROTH90_pixels
            xmax=col+3*PETROTH90_pixels
            ymin=row-3*PETROTH90_pixels
            ymax=row+3*PETROTH90_pixels

        # get image dimensions of 24um mosaic
        iraf.imgets(image=self.mosaic24,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=self.mosaic24,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)


        
        # check that cutout region is not outside bounds of image

        if (xmax > image_xmax):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_xmax,xmax))
            print(self.prefix, i, ': setting xmax to max x of image')
            xmax=image_xmax

        if (ymax > image_ymax):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_ymax,ymax))
            print(self.prefix, i, ': setting ymax to max y of image')
            ymax=image_ymax

        if (xmin < 1):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f)'%(image_xmax,xmin))
            print(self.prefix, i, ': setting xmin to 1')
            xmin=1

        if (ymin < 1):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f)'%(image_xmax,ymin))
            print(self.prefix, i, ': setting ymin to 1')
            ymin=1
        counter='%02i'%(i)
        if starflag:
            sex_image=self.prefix+'-star'+counter+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-star'+counter+'-'+'galfit-cutout-unc24.fits'
        else:
            sex_image=self.prefix+'-'+counter+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-'+counter+'-'+'galfit-cutout-unc24.fits'
            cov_image=self.prefix+'-'+counter+'-'+'galfit-cutout-cov24.fits'


        try:
            iraf.imgets(image=working_dir+sex_image,param='naxis1')#get RA of image
        except:
            iraf.imgets(image=working_dir+sex_image+'[1]',param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        try:
            iraf.imgets(image=working_dir+sex_image,param='naxis2')#get RA of image
        except:
            iraf.imgets(image=working_dir+sex_image+'[1]',param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)


        xcenter=0.5*(image_xmax)
        ycenter=0.5*(image_ymax)

        print('image dimensions: (x,y) = ',image_xmax, image_ymax)
        ##############################################
        # GATHER GALFIT INPUT
        ##############################################
        input_image=sex_image
        sigma_image=unc_image
        if starflag:
            mask_image=working_dir+self.prefix+'-star'+counter+'-'+'galfit-mask24.fits'
        else:
            mask_image=working_dir+self.prefix+'-'+counter+'-'+'galfit-mask24.fits'

        # use SSC PSF for making sim galaxies
        #psf_image_for_sim='/Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits'
        #psf_image='/Users/rfinn/research/LocalClusters/Images/'+self.prefix+'/24umWCS/PRF.fits'
        #psf_oversampling_for_sim=4

        # choose one of LCS PSFs at random to use when making sim galaxies
        cname=clusternames[np.random.randint(low=0,high=len(clusternames))]
        psf_image_for_sim='/Users/rfinn/research/LocalClusters/PRF/'+cname+'/'+cname+'psf_star.fits'
        psf_oversampling_for_sim=1

        # choose one of stars on the image
        
        if self.numberofstars < 2.:
            starnumber='00'
        else:
            starnumber='%02i'%(np.random.randint(low=0,high=self.numberofstars))
        psf_image_for_sim='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/star-'+starnumber+'.fits'
        psf_oversampling_for_sim=1


        #psf_image='/Users/rfinn/research/LocalClusters/Images/MKW11/24umWCS/PSFestimate_13Feb2012_sky.fits'
        #s='cp /Users/rfinn/research/LocalClusters/GalfitAnalysis/MKW11/24um/measurePSF/newpsf2.fits .'
        #os.system(s)
        #psf_image='newpsf2.fits'
        #psf_oversampling=1

        psf_image=self.psf_image
        psf_oversampling=self.psf_oversampling


        #psf_image='/Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits'
        #psf_image='/Users/rfinn/research/LocalClusters/Images/'+self.prefix+'/24umWCS/PRF.fits'
        #psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/PRF.fits'
        #psf_oversampling=4

        # mask_image # already defined

        xobj =xcenter
        yobj =ycenter
        #mag_total=22.5-2.5*log10(self.n.NMGY[i,4]) + self.n.EXTINCTION[i,4] #magzp-2.5*log10(self.f24NSA[i])
        if starflag:
            mag_total=self.se.MAG_BEST[i]
            Re=2.5
            axis_ratio=.9
            PA=5
            rad=2.5
        else:
            mag_total=self.model_mag[i]
            Re =self.model_re[i]/mipspixelscale
            if self.model_re[i] > max_cutout_radius:
                Re=max_cutout_radius/mipspixelscale/2.
            axis_ratio =self.model_BA[i]
            PA =self.model_PA[i]
            rad=self.model_re[i]/mipspixelscale
            
        xminfit=1
        #xmaxfit=xmax-xmin
        xmaxfit=image_xmax
        yminfit=1
        #ymaxfit=ymax-ymin
        ymaxfit=image_ymax
        
        print(xmin,xmax,xminfit,xmaxfit)
        print(ymin,ymax,yminfit,ymaxfit)
        convolution_size=31
        if xmaxfit < convolution_size:
            convolution_size=xmaxfit
        convolution_size=min(xmaxfit,ymaxfit)
        magzp=magzp
        pscale=mipspixelscale   # arcsec/pix
	sky=0

        if starflag:
            galname=working_dir+self.prefix+'-star'+counter+'-24'
        else:
            galname=working_dir+self.prefix+'-'+counter+'-24'


        ##############################################
        # GENERATE GALFIT MODEL IMAGE
        ##############################################
        #print galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky
        #q=raw_input('hit any key to continue')
        if starflag:
            print('not generating model for star')
            #t=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,simflag=1,NSAsersic=1,nsersic=1,constrflag=0)
        else:
            
            t=rungalfit_3times(galname,input_image,sigma_image,psf_image_for_sim,psf_oversampling_for_sim,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,simflag=1,NSAsersic=1,nsersic=self.model_n[i],constrflag=0,fitall=0)

            #!!! COME BACK HERE AND CONTINUE CHECKING
            print(galname+'-1Comp-galfit-out.fits',galname+'-1Comp-'+'model.fits',input_image)
            outim=galname+'-1Comp-galfit-out.fits'
            sim_gal=self.prefix+'-'+counter+'-model.fits'
            # add image (extension 1) plus model (extension 2) to create model image w/noise
            iraf.imarith(galname+'-1Comp-galfit-out.fits[1]','+',galname+'-1Comp-galfit-out.fits[2]',sim_gal)
            
            # make noise of galaxy model
            os.system('rm temp*.fits')
            # multiply image by exptime x gain x coverage map
            scale=GAIN*EXPT*FLUXCONV
            print('cov_image = ',cov_image)
            comb_unc_image=self.prefix+'-'+counter+'-'+'galfit-cutout-unc-poisson24.fits'
            oldway=1

            if oldway:
                #scale=GAIN*EXPT
                iraf.imarith(operand1=galname+'-1Comp-galfit-out.fits[2]',op='*',operand2=scale,result='temp2.fits')

                try:
                    iraf.imarith(operand1='temp2.fits',op='*',operand2=cov_image,result='temp3.fits')
                except:
                    iraf.imarith(operand1='temp2.fits',op='*',operand2=cov_image+'[1]',result='temp3.fits')

                # take sqrt
                iraf.imfunction(input='temp3.fits',output='temp4.fits',function='sqrt')
                # divide by exptime x gain x coverage map

                fudgefactor=1
                iraf.imarith(operand1='temp4.fits',op='/',operand2=scale*fudgefactor,result='temp5.fits')
                try:
                    iraf.imarith(operand1='temp5.fits',op='/',operand2=cov_image,result='temp6.fits')
                except:
                    iraf.imarith(operand1='temp5.fits',op='/',operand2=cov_image+'[1]',result='temp6.fits')

                # should we add the sum of the squared images and then take the sqrt?


                # combine poisson noise from model w/noise image


                iraf.imfunction(input='temp6.fits',output='temp7.fits',function='square')
                try:
                    iraf.imfunction(input=sigma_image,output='temp8.fits',function='square')
                except:
                    iraf.imfunction(input=sigma_image+'[1]',output='temp8.fits',function='square')
                iraf.imarith(operand1='temp7.fits',op='+',operand2='temp8.fits',result='temp9.fits')
                try:
                    iraf.imfunction(input='temp9.fits',output=comb_unc_image,function='sqrt')
                except:
                    iraf.imfunction(input='temp9.fits',output=comb_unc_image+'[1]',function='sqrt')
            else:


                iraf.imarith(operand1=galname+'-1Comp-galfit-out.fits[2]',op='*',operand2=scale,result='temp2.fits')
                iraf.imarith(operand1='temp2.fits',op='/',operand2=cov_image,result='temp3.fits')

                # not adding the _noise.fits cutout to see if it improves recovered sersic index
                #iraf.imfunction(input='temp3.fits',output=comb_unc_image,function='sqrt')

                iraf.imfunction(input=sigma_image,output='temp4.fits',function='square')
                iraf.imarith(operand1='temp3.fits',op='+',operand2='temp4.fits',result='temp5.fits')


                # remove the following line once testing of sersic index is done
                #shutil.copy(sigma_image,comb_unc_image)




        ##############################################
        # RUN GALFIT  TIMES
        ##############################################


        # run galfit  times
        if starflag:
            t=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,NSAsersic=1,nsersic=1.5,constrflag=0)

        else:
            randomr=np.random.uniform(low=.8*rad,high=1.2*rad)
            bamin=axis_ratio-.2
            if bamin < .1:
                bamin=.1
            bamax=axis_ratio+.2
            if bamax > 1:
                bamax=1
            self.inmag[i]=mag_total+np.random.uniform(low=-.2,high=.2)
            self.inre[i]=randomr
            self.inba[i]=axis_ratio#np.random.uniform(low=bamin,high=bamax)
            self.inpa[i]=PA#+np.random.uniform(low=-20,high=20)
            self.insersicn[i]=np.random.uniform(low=.8*self.model_n[i],high=1.2*self.model_n[i])

            # skipping this because it no longer relates to how I'm analyzing real images
            #t=rungalfit_3times(galname,sim_gal,comb_unc_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,self.inmag[i],self.inre[i],self.inba[i],self.inpa[i],sky,constrflag=0,fixnsersic=0,fitall=0,convolutionflag=0,NSAsersic=1,nsersic=self.insersicn[i])

            try: # allow for times when galfit will fail b/c sb or flux is too low
                # run first w/out convolution

                t=rungalfit_3times(galname,sim_gal,comb_unc_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,self.inmag[i],self.inre[i],self.inba[i],self.inpa[i],sky,constrflag=0,fixnsersic=0,fitall=0,NSAsersic=1,nsersic=self.insersicn[i],fixba=1,fixpa=1)

                # check sersic index
                # if greater than 5, run again holding n fixed and equal to 5

                print('***********************************************')
                print('Checking sersic index for ', galname)
                output_image=galname+'-1Comp-galfit-out.fits'
                txc,tyc,tmag,tRe,Nsersic,taxis_ratio,tPA,tsky,tnumerical_error_flag24=parse_galfit_1comp(output_image+'[2]')
                print('Nsersic = ',Nsersic)
                if float(Nsersic[0]) > 6:
                    print('rerunning galfit with index held fixed at 6')
                    t=rungalfit_3times(galname,sim_gal,comb_unc_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total+np.random.uniform(low=-.2,high=.2),rad+np.random.uniform(low=-3,high=3),axis_ratio+np.random.uniform(low=-.2,high=.2),PA+np.random.uniform(low=-20,high=20),sky,constrflag=0,fixnsersic=1,fixnsersicvalue=6,fitall=0)

                #print '***********************************************'
                #print 'Running with Fourier mode 1 turned on', galname
                #output_image=galname+'-1Comp-galfit-out.fits'
                #txc,tyc,tmag,tRe,Nsersic,taxis_ratio,tPA,tsky,tnumerical_error_flag24=parse_galfit_1comp(output_image+'[2]')
                #t=rungalfit_3times(galname,sim_gal,comb_unc_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,txc[0],tyc[0],tmag[0],tRe[0],taxis_ratio[0],tPA[0],sky,constrflag=0,fixnsersic=0,fitall=1,asymflag=1)

            except:
                quitflag=1

        return quitflag

    def just_get_images24(self,i, keepimages=1,make_mask_flag=0,review_mask_flag=0,starflag=0,simflag=0):
        # getimages:
        # set = 0 to not get images (if they already exist)
        # set = 1 if sdss images are needed
        #
        # this subroutine gets a cutouot from the mips mosaic,
        #    parses relevant input for galfit,
        #    and runs galfit three times.
        #        #
        # keepimages = 1 to keep MIPS cutout images
        # keepimages = 0 to delete MIPS cutout images (better if disk space is tight)
        quitflag=0
        ##############################################
        # MAG ZP
        ##############################################
        # define mag zp for 24um image scans
        flux_zp_AB = 3631. # in Jy
        flux_zp_Vega = 7.17 # in Jy
        flux_zp=flux_zp_AB
        
        # conversion from image units of MJ/sr to micro-Jy (1 sq arcsec = 2.3504e-11 sr)
        conv_MJysr_uJy = 23.5045*(2.45**2)
        magzp=2.5*log10(flux_zp*1.e6/conv_MJysr_uJy)



        ##############################################
        # SOME SETUP COMMANDS
        ##############################################
        # open ds9 display
        d=ds9.ds9()
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        d.set('cd '+working_dir)
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')


        if starflag:
            col=self.se.X_IMAGE[i]
            row=self.se.Y_IMAGE[i]
        else:
            col=self.xsim[i]
            row=self.ysim[i]


        ##############################################
        # GET CUTOUT OF GALAXY AND UNCERTAINTY IMAGE
        ##############################################
        if starflag:

            xmin=col-dxstar
            xmax=col+dxstar
            ymin=row-dxstar
            ymax=row+dxstar
        else:
            if self.model_re[i] > (min_cutout_size/6.):
                PETROTH90_pixels =self.model_petro90[i]/mipspixelscale
                if self.model_petro90[i]> (max_cutout_size/6.):
                    PETROTH90_pixels =max_cutout_size/6./mipspixelscale

            else:
                PETROTH90_pixels =(min_cutout_size/6.)/mipspixelscale

            # make a cutout of the galaxy (easier than trying to run sextractor on full mosaic
            # need x and y pixel values of the galaxy
            xmin=col-3*PETROTH90_pixels
            xmax=col+3*PETROTH90_pixels
            ymin=row-3*PETROTH90_pixels
            ymax=row+3*PETROTH90_pixels

        # get image dimensions of 24um mosaic

        iraf.imgets(image=self.mosaic24,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=self.mosaic24,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)
        
        # check that cutout region is not outside bounds of image

        if (xmax > image_xmax):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_xmax,xmax))
            print(self.prefix, i, ': setting xmax to max x of image')
            xmax=image_xmax

        if (ymax > image_ymax):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_ymax,ymax))
            print(self.prefix, i, ': setting ymax to max y of image')
            ymax=image_ymax

        if (xmin < 1):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f)'%(image_xmax,xmin))
            print(self.prefix, i, ': setting xmin to 1')
            xmin=1

        if (ymin < 1):
            print('WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f)'%(image_xmax,ymin))
            print(self.prefix, i, ': setting ymin to 1')
            ymin=1

        counter='%02i'%(i)
        if starflag:
            sex_image=self.prefix+'-star'+counter+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-star'+counter+'-'+'galfit-cutout-unc24.fits'
            cov_image=self.prefix+'-star'+counter+'-'+'galfit-cutout-cov24.fits'
        else:
            sex_image=self.prefix+'-'+counter+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-'+counter+'-'+'galfit-cutout-unc24.fits'
            cov_image=self.prefix+'-'+counter+'-'+'galfit-cutout-cov24.fits'

        try:
            s=self.mosaic24+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+sex_image)
            print(s)
            s=self.mosaic24unc+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+unc_image)
            s=self.mosaic24cov+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+cov_image)

        except:
            print('Warning:  Problem creating cutout image, probably b/c it already exists')
            print('   going to delete existing cutout image and try imcopy again')
            iraf.imdel(sex_image)
            iraf.imdel(unc_image)
            iraf.imdel(cov_image)
            s=self.mosaic24+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+sex_image)
            s=self.mosaic24unc+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+unc_image)
            s=self.mosaic24cov+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+cov_image)


        ##############################################
        # RUN SEXTRACTOR AND MAKE OBJECT MASK
        ##############################################

        # run sextractor to generate a list of objects in the image and generate 'segmentation image'
        os.system('cp ~/research/LocalClusters/sextractor/default.sex.24um.galfit .')
        os.system('cp ~/research/LocalClusters/sextractor/default.param .')
        os.system('cp ~/research/LocalClusters/sextractor/default.conv .')
        os.system('cp ~/research/LocalClusters/sextractor/default.nnw .')
        os.system('sex %s -c default.sex.24um.galfit'%(sex_image))
        # convert segmentation image to object mask by replacing the object ID of target galaxy with zeros
        #   parse sextractor output to get x,y coords of objects        
        sexout=atpy.Table('test.cat',type='ascii')
        sexnumber=sexout['col1']
        xsex=sexout['col2']
        ysex=sexout['col3']
        xcenter=1.*(xmax-xmin)/2.
        ycenter=1.*(ymax-ymin)/2.
        self.model_xc[i]=xcenter
        self.model_yc[i]=ycenter
        dist=sqrt((ycenter-ysex)**2+(xcenter-xsex)**2)

        #   find object ID
        objIndex=where(dist == min(dist))
        objNumber=sexnumber[objIndex]
        objNumber=objNumber[0] # not sure why above line returns a list
        print('object number = ',objNumber)
        if make_mask_flag:
            counter='%02i'%(i)
            if starflag:
                mask_image=working_dir+self.prefix+'-star'+counter+'-'+'galfit-mask24.fits'
            else:
                mask_image=working_dir+self.prefix+'-'+counter+'-'+'galfit-mask24.fits'
            if os.path.exists(mask_image):
                iraf.imdel(mask_image)
            try:
                iraf.imcopy('segmentation.fits[1]',mask_image)
            except:
                iraf.imcopy('segmentation.fits',mask_image)

        #   use iraf imexam to replace object ID values with zero
        iraf.imreplace(mask_image,value=0,lower=objNumber-.5,upper=objNumber+.5)

        while review_mask_flag:
            d.set('frame delete all')
            s='file new '+homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits[1]'
            try:
                d.set(s)
                d.set('zoom to fit')
            except:
                print("couldn't access: ",s)
            s='file new '+sex_image
            d.set(s)
            d.set('zoom to fit')
            s='file new '+mask_image
            d.set(s)
            d.set('zoom to fit')
            flag=input('edit the mask? \n')
            if flag.find('n') > -1:
                review_mask_flag = 0
            elif flag.find('y') > -1:
                editflag=int(input('enter 0 to subtract an object, 1 to add an object'))
                if editflag == 0:
                    objid=float(input('enter object id to remove from mask'))
                    iraf.imreplace(mask_image,value=0,lower=objid-.5,upper=objid+.5)
                elif editflag == 1:
                    print('entering imedit')
                    a,b=runimedit(mask_image)
            elif flag.find('q') > -1:
                quitflag=1
                return quitflag

        # get best x,y positions from sextractor
        #    - subtract 1 b/c sextractor catalog starts at one, but python is zero-indexed
        xcenter=xsex[objNumber-1]
        ycenter=ysex[objNumber-1]

        return quitflag


    def measuresnr24(self,starflag=0):
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'
        os.chdir(working_dir)
        if starflag:
            files=glob.glob('*star*cutout24.fits')
            
        else:
            files=glob.glob('*model.fits')
        self.f24=zeros(len(files),'d')
        self.f24err=zeros(len(files),'d')
        self.snr24=zeros(len(files),'d')
        fileindex=[]
        for i in range(len(files)):
            print(i,files[i])
            inimage=files[i]
            if starflag:
                print('hi')
                t=inimage.split('-')
                r=t[1]
                s=r.split('star')
                ind=s[1]
                fileindex.append(int(ind))
                prefix=t[0]+'-'+t[1]+'-'
                noise=prefix+'galfit-cutout-unc24.fits'
            else:
                t=inimage.split('model')
                r=t[0].split('-')
                fileindex.append(int(r[1]))
                prefix=t[0]
                noise=prefix+'galfit-cutout-unc-poisson24.fits'
            iraf.imgets(image=working_dir+inimage,param='naxis1')#get RA of image
            image_xmax=float(iraf.imgets.value)
            iraf.imgets(image=working_dir+inimage,param='naxis2')#get RA of image
            image_ymax=float(iraf.imgets.value)
            xc=image_xmax*.5
            yc=image_ymax*.5
            print(i, inimage,image_xmax,image_ymax,xc,yc)
            outfile=open(working_dir+'output_coords','w')
            outfile.write('%5.2f %5.2f \n'%(int(round(xc)),int(round(yc))))
            outfile.close()

            skyfile=working_dir+self.prefix+"_sky"
            sky = open(skyfile,'w')
            #sky.write("%5.2f %5.2f 0.0 0.0 0.0 1 0 \n"%(int(round(xc)),int(round(yc))))
            sky.write('0.0\n')
            sky.close()
            aps = open(working_dir+"apertures",'w')
            aps.write("2.6")
            aps.close()

            datfile=working_dir+t[0]+"phot.dat"
            if os.path.exists(datfile):
                os.remove(datfile)
            print(inimage,'output_coords',datfile,skyfile)
            iraf.digiphot.daophot.phot(image=working_dir+inimage,coords=working_dir+'output_coords',output=datfile,calgorithm='none',skyfile=skyfile,salgori="file",aperture=working_dir+"apertures",interactive="no",verify='no',verbose='no',wcsin='logical',wcsout='logical')

            input=open(datfile,'r')
            j=0

            for line in input:
                if line.find('#') > -1: #skip lines with '#' in them
                    continue
                if line.find(self.prefix) > -1: #skip lines with '#' in them
                    j=0
                    continue
                j=j+1
                if (j > 3):
                #print j, line
                
                    t = line.split()
                    try:
                        self.f24[i]=float(t[1])
                    except ValueError:
                        self.f24[i]=0
            input.close()


            datfile=prefix+"phot_unc.dat"
            if os.path.exists(datfile):
                os.remove(datfile)
            iraf.digiphot.daophot.phot(image=working_dir+noise,coords=working_dir+'output_coords',output=datfile,calgorithm='none',skyfile=skyfile,salgori="file",aperture=working_dir+"apertures",interactive="no",verify='no',verbose='no',wcsin='logical',wcsout='logical')
            #iraf.digiphot.daophot.phot(image=noise,coords='output_coords',output=datfile,calgorithm='none',skyfile=skyfile,salgori="file",aperture="apertures",interactive="no",verify='no',verbose='yes')
        ##iraf.digiphot.apphot.phot(image,coords="noisecoords.dat",output="noise.dat",calgorithm='none',skyfile="sky",salgori="file",aperture="apertures",interactive="no",verbose='yes')


            input=open(datfile,'r')
            j=0

            for line in input:
                if line.find('#') > -1: #skip lines with '#' in them
                    continue
                if line.find('mosaic') > -1: #skip lines with '#' in them
                    j=0
                    continue
                j=j+1
                if (j > 3):
                    #print j, line
                    t = line.split()
                    try:
                        self.f24err[i]=float(t[1])
                    except ValueError:
                        self.f24err[i]=0
            input.close()
        self.snr24=abs(self.f24/self.f24err)
        fileindex=array(fileindex,'i')
        self.snr24dict=dict((a,b) for a,b in zip(fileindex,arange(len(files))))



    def measure_SEsnr24(self,starflag=0,interactive_flag=0):
        if interactive_flag:
            d=ds9.ds9()
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'
        os.chdir(working_dir)
        if starflag:
            files=glob.glob('*star*cutout24.fits')
            
        else:
            files=glob.glob('*model.fits')
            self.sef24=zeros(npoints,'d')
            self.sef24err=zeros(npoints,'d')
            self.sebest_snr24=zeros(npoints,'d')
            self.seauto_snr24=zeros(npoints,'d')
            self.seiso_snr24=zeros(npoints,'d')

            self.seiso_area=zeros(npoints,'d')
        fileindex=[]
        t=files[0].split('model')
        for i in range(npoints):
        #for i in range(10):
            print(i,files[i])
            inimage=files[i]
            fnumber='%02i'%(i)
            if starflag:
                print('hi')
                t=inimage.split('-')
                r=t[1]
                s=r.split('star')
                ind=s[1]
                fileindex.append(int(ind))
                prefix=t[0]+'-'+t[1]+'-'
                noise=prefix+'galfit-cutout-unc24.fits'
            else:
                t=inimage.split('model')
                r=t[0].split('-')
                fileindex.append(i)
                inimage=self.prefix+'-'+fnumber+'-model.fits'
                prefix=self.prefix+'-'+fnumber+'-'
                noise=prefix+'galfit-cutout-unc-poisson24.fits'

            if not(os.path.exists(working_dir+inimage)):
                print('could not find ',working_dir+inimage)
                continue
            iraf.imgets(image=working_dir+inimage,param='naxis1')#get x dimension of image
            image_xmax=float(iraf.imgets.value)
                
            iraf.imgets(image=working_dir+inimage,param='naxis2')#get y dimension of image
            image_ymax=float(iraf.imgets.value)
            xc=image_xmax*.5
            yc=image_ymax*.5
            print(i, inimage,image_xmax,image_ymax,xc,yc)
            outfile=open(working_dir+'output_coords','w')
            outfile.write('%5.2f %5.2f \n'%(int(round(xc)),int(round(yc))))
            outfile.close()

            # RUN SEXTRACTOR
            os.system('cp '+homedir+'research/LocalClusters/sextractor/default.param .')
            os.system('cp '+homedir+'research/LocalClusters/sextractor/default.nnw .')
            s='sex '+inimage+' -c '+homedir+'research/LocalClusters/sextractor/default.sex.24um.galfit -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+noise+' -CATALOG_NAME '+prefix+'test.fits -CATALOG_TYPE FITS_1.0'
            os.system(s)
            # DISPLAY SEGMENTATION IMAGE TO MAKE SURE THE IMAGE ISN'T SPLIT INTO MANY
            if interactive_flag:
                mask_image='segmentation.fits'
                #iraf.imcopy('segmentation.fits[1]',mask_image)
            
                d.set('frame delete all')
                s='file new '+working_dir+'/'+inimage
                d.set(s)
                d.set('zoom to fit')
                s='file new '+working_dir+'/segmentation.fits'
                d.set(s)
                d.set('zoom to fit')
                input('press any key when ready to continue\n')
            # read in SE table to get x,y for sources
            fname=prefix+'test.fits'
            try:
                se=atpy.Table(fname)
                print('found ',len(se.X_IMAGE),' sources on the field of ',prefix)
                distance=sqrt((se.X_IMAGE-xc)**2+(se.Y_IMAGE-yc)**2)
                if min(distance) < 3.:
                    matchindex2=where(distance == min(distance))
                    matchindex=matchindex2[0]
                    print('matchindex',matchindex,matchindex2)
                    # FIND SOURCE AT CENTER
                    # WRITE FLUX AND FLUX_ERR
                    self.sef24[i]=float(se.FLUX_BEST[matchindex])
                    self.sef24err[i]=float(se.FLUXERR_BEST[matchindex])
                    self.sebest_snr24[i]=abs(self.sef24[i]/self.sef24err[i])
                    self.seauto_snr24[i]=abs(float(se.FLUX_AUTO[matchindex])/float(se.FLUXERR_AUTO[matchindex]))
                    self.seiso_snr24[i]=abs(float(se.FLUX_ISO[matchindex])/float(se.FLUXERR_ISO[matchindex]))
                    self.seiso_area[i]=float(se.ISOAREA_IMAGE[matchindex])

                else:
                    print("NO SOURCES W/IN 3 PIXELS OF OBJECT CENTER")
                        # write_galfit_sersic(galfit_input,objnumber,profile,se.X_IMAGE[k],se.Y_IMAGE[k],se.MAG_BEST[k],se.FLUX_RADIUS[k,0],2,se.B_IMAGE[k]/se.A_IMAGE[k],se.THETA_IMAGE[k])
            except AttributeError:
                print('WARNING: no sources detected in image!')


        fileindex=array(fileindex,'i')
        self.sesnr24dict=dict((a,b) for a,b in zip(fileindex,arange(len(files))))


    def compare_snr():
        figure()
        plot(self.sim.modelmag,self.sim.snr24,'ro')
        plot(self.sim.modelmag,self.sebest_snr24,'bo',label='FLUX_BEST')
        plot(self.sim.modelmag,self.seauto_snr24,'co',label='FLUX_AUTO')
        plot(self.sim.modelmag,self.seiso_snr24,'go',label='FLUX_ISO')

    def rungalfit_sim_24(self,startindex=0,runSE=0):
        # to run galfit once for all galaxies
        # assumes that you have already run run_just_get_images24()
        global interactive
        interactive=0
        set_ntimes(1)

        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'

        if os.path.exists(working_dir):
            os.chdir(working_dir)
        else:
            s='mkdir -p '+working_dir
            os.system(s)
            os.chdir(working_dir)

        if runSE:
            print('Running SE')
            self.runSE()
        print('reading SE catalog')
        self.readSE()
        #self.model_mag=np.random.uniform(low=minmag,high=maxmag,size=len(self.ra))
        #self.model_re=np.random.uniform(low=2,high=25,size=len(self.ra))

        print('getting positions for artificial sources')
        self.xsim,self.ysim=getpositionsLCS(self.se.X_IMAGE,self.se.Y_IMAGE,self.mosaic24,self.mosaic24cov,npoints)
        figure()
        plot(self.xsim,self.ysim,'bo')
        scatter(self.se.X_IMAGE,self.se.Y_IMAGE,marker='o',s=10,color='0.5',alpha=0.5)
        savefig(homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'simgalpositions.eps')
        close('all')
        #raw_input('hit any key to continue')
        #simpos=atpy.Table()
        #simpos.add_column('X',self.xsim)
        #simpos.add_column('Y',self.ysim)
        #s=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_simgal.fits'
        #simpos.write(s)
        print('setting up model parameters')
        self.model_mag=np.random.uniform(low=minmodelmag,high=maxmodelmag,size=npoints)

        
        # select Re, B/A and PA from the parent sample of all spirals
        spiral_index=where(self.spiralflag)
        spiral_index=spiral_index[0]
        modindex=np.random.randint(low=0,high=len(spiral_index)-1,size=npoints)
        self.model_re=self.n.SERSIC_TH50[spiral_index[modindex]]
        self.model_n=self.n.SERSIC_N[spiral_index[modindex]]
        self.model_petro90=self.n.PETROTH90[spiral_index[modindex]]
        #self.model_mag=22.5-2.5*log10(self.n.NMGY[modindex,4]) + self.n.EXTINCTION[modindex,4] - 2. # mag 2 mags brighter
        self.model_PA=self.n.SERSIC_PHI[spiral_index[modindex]]
        self.model_BA=self.n.SERSIC_BA[spiral_index[modindex]]


        # OR uncomment next 3 lines to select parameters from a uniform range
        # re and PETRO90 should be in arcsec
        self.model_re=np.random.uniform(low=minmodelre_pixels,high=maxmodelre_pixels,size=npoints)*mipspixelscale
        self.model_n=np.random.uniform(low=.5,high=4.2,size=npoints)
        self.model_petro90=2.*self.model_re # arcsec
        self.model_xc=zeros(npoints,'f')
        self.model_yc=zeros(npoints,'f')

        self.inmag=zeros(npoints,'f')
        self.inre=zeros(npoints,'f')
        self.inba=zeros(npoints,'f')
        self.inpa=zeros(npoints,'f')
        self.insersicn=zeros(npoints,'f')
        self.rungalfit_sim_part2(startindex=startindex)
        
    def rungalfit_sim_part2(self,startindex=0):
        print('starting to run galfit')
        for i in range(startindex,npoints):
            quitflag=self.just_get_images24(i,make_mask_flag=1)
            if quitflag:
                return
            quitflag=self.get_images24(i,0)
            if quitflag:
                print('galfit did not complete for ', i,' mag = ',self.model_mag[i],' Re(24) = ',self.model_re[i])
        self.measure_SEsnr24()
        self.write_galfit_sersic_parameters_24()

    def rungalfit_stars_24(self,startindex=0):
        # to run galfit once for all galaxies
        # assumes that you have already run run_just_get_images24()
        global interactive
        interactive=0
        
        set_ntimes(1)

        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/'
        d=ds9.ds9()
        if os.path.exists(working_dir):
            os.chdir(working_dir)
        else:
            s='mkdir -p '+working_dir
            os.system(s)
            os.chdir(working_dir)


        self.readstarfile()
        self.runSEstars()
        self.readSEstars()
        for i in range(len(self.se.X_IMAGE)):
            if self.starflag[i]:
                quitflag=self.just_get_images24(i,make_mask_flag=1,starflag=1)
                if quitflag:
                    return
                quitflag=self.get_images24(i,0,starflag=1)
                if quitflag:
                    return
                
                s='file new '+self.psf_image
                d.set(s)
                d.set('zoom to fit')
                if interactive:
                    t=input('press any key to continue, x to quit \n')
                    if t.find('x') > -1:
                         return
        self.measuresnr24(starflag=1)
        self.write_galfit_sersic_parameters_stars_24()
    def reviewstars(self):
        self.readsimresults(starflag=1)
        self.plotresults(starflag=1)

class simresults:
    def __init__(self,clustername):
        self.prefix=clustername
        self.sbcutobs=sbcut[self.prefix]
        ##############################################
        # MAG ZP
        ##############################################
        # define mag zp for 24um image scans
        flux_zp_AB = 3631. # in Jy
        flux_zp_Vega = 7.17 # in Jy
        flux_zp=flux_zp_AB
        # conversion from image units of MJ/sr to micro-Jy (1 sq arcsec = 2.3504e-11 sr)
        conv_MJysr_uJy = 23.5045*(2.45**2)
        self.magzp24=2.5*log10(flux_zp*1.e6/conv_MJysr_uJy)
        self.readsimresults()

    def readsimresults(self,starflag=0):
        if starflag:
            output24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_stars_GalfitSersicParam_24_sim.fits'
        else:
            output24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_GalfitSersicParam_24_sim.fits'
        self.sim=atpy.Table(output24)
        self.sb=self.sim.modelmag + 2.5*log10(pi*(self.sim.modelre**2)*self.sim.modelba)
        self.sb_obs=self.sim.mag1 + 2.5*log10(pi*((self.sim.re1*mipspixelscale)**2)*self.sim.axisratio1)
        self.csb_obs=self.sim.cmag1 + 2.5*log10(pi*((self.sim.cre1*mipspixelscale)**2)*self.sim.caxisratio1)
        self.dr=sqrt((self.sim.yc1-self.sim.modelyc)**2+(self.sim.xc1-self.sim.modelxc)**2)
        self.cdr=(self.sim.cxc1-self.sim.modelxc)**2
        self.cdr=sqrt((self.sim.cyc1-self.sim.modelyc)**2)#+(self.sim.cxc1-self.sim.modelxc)**2)

        self.statflag = (self.sim.modelmag + 2.5*log10(pi*(self.sim.modelre**2))) < sbcut
        self.statflagobs = (self.sb_obs < self.sbcutobs) & (self.dr < 5) & self.sim.galfit_flag #& (self.sim.numerical_error_flag24 < .1)
        self.cstatflagobs = (self.csb_obs < self.sbcutobs) & (self.cdr < 5) & self.sim.cgalfit_flag & (self.sim.cnumerical_error_flag24 < .1)



    def review(self,usesnrflag=0):
        self.readsimresults()
        self.plotsample()
        self.plotresultsv2(usesb=1)
        #self.plotresults(usesnr=usesnrflag)


    def reviewimages(self):
        self.readsimresults()
        d=ds9.ds9()
        for i in range(len(self.sim.modelre)):
            if abs(self.sim.modeln[i] - self.sim.nsersic1[i]) > 1:
                print('galaxy number ',i)
                print('-----------------------------------------------')
                print('var     mod      fit (no conv)   fit (w/conv)')
                print('-----------------------------------------------')
                print('n   = %5.1f, %5.1f+/-%3.2f, %5.1f+/-%3.2f'%(self.sim.modeln[i],self.sim.nsersic1[i],self.sim.nsersic1err[i], self.sim.cnsersic1[i],self.sim.cnsersic1err[i]))
                print('Re  = %5.1f, %5.1f+/-%3.2f, %5.1f+/-%3.2f'%(self.sim.modelre[i]/mipspixelscale,self.sim.re1[i],self.sim.re1err[i], self.sim.cre1[i],self.sim.cre1err[i]))
                print('mag = %5.1f, %5.1f+/-%3.2f, %5.1f+/-%3.2f'%(self.sim.modelmag[i],self.sim.mag1[i],self.sim.mag1err[i], self.sim.cmag1[i],self.sim.cmag1err[i]))
                print('B/A = %5.1f, %5.1f+/-%3.2f, %5.1f+/-%3.2f'%(self.sim.modelba[i],self.sim.axisratio1[i],self.sim.axisratio1err[i], self.sim.caxisratio1[i],self.sim.caxisratio1err[i]))
                print('phi = %5.1f, %5.1f+/-%3.2f, %5.1f+/-%3.2f'%(self.sim.modelpa[i],self.sim.pa1[i],self.sim.pa1err[i], self.sim.cpa1[i],self.sim.cpa1err[i]))
                print('snr = %5.1f, %5.1f+/-%3.2f'%(self.sim.snr24[i],self.sim.f24[i],self.sim.f24err[i]))
                print('nerr = ',self.sim.numerical_error_flag24[i])

                
                d.set('frame delete all')
                counter='%02i'%(i)
                s='file new '+self.prefix+'-'+counter+'-24-1Comp-galfit-out.fits[1]'
                d.set(s)
                s='file new '+self.prefix+'-'+counter+'-galfit-cutout-unc-poisson24.fits'
                d.set(s)
                s='file new '+self.prefix+'-'+counter+'-galfit-mask24.fits'
                d.set(s)
                s='file new '+self.prefix+'-'+counter+'-24-1Comp-noconv-galfit-out.fits[2]'
                d.set(s)
                s='file new '+self.prefix+'-'+counter+'-24-1Comp-noconv-galfit-out.fits[3]'
                d.set(s)


                s='file new '+self.prefix+'-'+counter+'-24-1Comp-galfit-out.fits[1]'
                d.set(s)
                s='file new '+self.prefix+'-'+counter+'-24-1Comp-galfit-out.fits[2]'
                d.set(s)
                s='file new '+self.prefix+'-'+counter+'-24-1Comp-galfit-out.fits[3]'
                d.set(s)

                try:
                    s='file new '+self.prefix+'-'+counter+'-24-1Comp-subcomps.fits[3]'
                    d.set(s)
                except:
                    print("WARNING:  couldn't load ",self.prefix+'-'+counter+'-24-1Comp-subcomps.fits[3]')
                d.set('tile')
                for k in range(1,9):
                    s='frame '+str(k)
                    d.set(s)
                    d.set('scale log')
                    d.set('zoom to fit')
                string=input('hit any key when ready to continue (q to quit) \n')
                if string.find('q') > -1:
                    quitflag=1
                    break



    def plotresults(self,usesb=0,starflag=0,usesersic=0,usesnr=0,asymflag=0,snrcut=4,usere=0,minre=2.,convflag=0):

        fitfunc = lambda p, x: p[0] + p[1] * x
        #errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
        errfunc = lambda p, x, y: (y - fitfunc(p, x)) 

        flag=self.sim.modelmag < 18.
        if usesb:
            colorscale=self.sim.modelmag + 2.5*log10(pi*(self.sim.modelre**2))
            v1=13
            v2=25
        elif usesersic:
            colorscale=abs(self.sim.modeln - self.sim.nsersic1)
            v1=0
            v2=6
        elif usere:
            colorscale=self.sim.modelre
            v1=3
            v2=30
            #flag=self.sim.modelre < minre
        elif usesnr:
            colorscale=self.sim.snr24
            v1=1
            v2=10
            flag=(self.sim.snr24 > snrcut) & (self.sim.re1 > minre) & (self.sim.numerical_error_flag24 < .1)
        else:
            colorscale=self.sim.modelmag
            v1=minmodelmag
            v2=maxmodelmag
            if starflag:
                v1=11
                v2=15

        figure(figsize=(12,8))
        subplots_adjust(left=.1,right=.95,bottom=.15,top=.95,wspace=.25,hspace=.25)

        xl=['R_e input (pixel)','Sersic n input','B/A input','mag input','input mag','mag/Re^2' ]
        #yl=['R_e out- R_e input/R_e input','Sersic n out','B/A out','mag out']
        #yl=['R_e out- R_e input/R_e input','Sersic n_out - n_in','B/A out','mag_out - mag_in']
        #yl=['R_e out- R_e input/R_e input','Sersic n_out','B/A out','mag_out - mag_in']
        yl=['R_e out','Sersic n_out','B/A out','mag_out - mag_in','dr (pixel)','dr (pixel)']
        x=[self.sim.modelre/mipspixelscale,self.sim.modeln,self.sim.modelba,self.sim.modelmag,self.sim.modelmag,self.sim.modelmag + 2.5*log10(pi*(self.sim.modelre**2))]
        if asymflag:
            y=[(self.sim.asym_re1-self.sim.modelre/mipspixelscale)/(self.sim.modelre/mipspixelscale),(self.sim.asym_nsersic1),self.sim.asym_axisratio1,(self.sim.asym_mag1-self.sim.modelmag)]
        #erry=[self.sim.re1err/(self.sim.modelre/mipspixelscale),self.sim.nsersic1err,self.sim.axisratio1err,self.sim.mag1err]
            erry=[self.sim.asym_re1err/(self.sim.modelre/mipspixelscale),self.sim.asym_nsersic1err/self.sim.modeln,self.sim.asym_axisratio1err,self.sim.asym_mag1err/self.sim.modelmag]

        else:
            print('got here!')
            if convflag:
                y=[(self.sim.cre1),(self.sim.cnsersic1),self.sim.caxisratio1,(self.sim.cmag1-self.sim.modelmag),self.cdr,sqrt((self.sim.cyc1-self.sim.modelyc)**2+(self.sim.cxc1-self.sim.modelxc)**2)]
                erry=[self.sim.re1err,self.sim.nsersic1err,self.sim.axisratio1err,self.sim.mag1err/self.sim.modelmag,self.sim.xc1err,self.sim.yc1err]
            else:
                y=[(self.sim.re1),(self.sim.nsersic1),self.sim.axisratio1,(self.sim.mag1-self.sim.modelmag),self.dr,sqrt((self.sim.yc1-self.sim.modelyc)**2+(self.sim.xc1-self.sim.modelxc)**2)]
                erry=[self.sim.re1err,self.sim.nsersic1err,self.sim.axisratio1err,self.sim.mag1err/self.sim.modelmag,self.sim.xc1err,self.sim.yc1err]

        xmin=[.5,0,-0.05,8.]
        xmax=[50,7,1.05,18.]
        #ymin=[-.5,0,-0.05,-.5]
        #ymax=[.5,6,1.0,.5]
        ymin=[.5,0,-0.05,-.5]
        ymax=[50,6,1.0,.5]
        if starflag:

            xmin=[.1,0,-0.05,8.]
            xmax=[3,2,1.05,16.]
            ymin=[0.,0,-0.05,-.5]
            ymax=[4,8,1.05,.5]
            xl=['SE Flux Radius','Gaussian ref','SE B/A','SE MAG_BEST']
        for i in range(6):
            subplot(2,3,i+1)
            xp=x[i]
            yp=y[i]
            ye=erry[i]
            if usesnr:
                try:
                    plot(xp[~flag],yp[~flag],'k.')
                except:
                    print('WARNING: problem plotting points w/snr below cut')
            sp=scatter(xp[flag],yp[flag],marker='o',s=ones(len(self.sim.re1[flag]))*40,c=colorscale[flag],vmin=v1,vmax=v2)
            #plot(xp[flag],yp[flag],'bo')
            #errorbar(xp[flag],yp[flag],yerr=ye[flag],fmt=None)#,ecolor='k')
            if i < 4:
                axis([xmin[i],xmax[i],ymin[i],ymax[i]])
            else:
                gca().set_yscale('log')
            panels=[0,1,2]
            if i in panels:
                x1,x2=xlim()
                xline=arange(x1,x2,.05)
                plot(xline,xline,'k--')
                pdiff=(yp[flag]-xp[flag])/xp[flag]
                s='ave per cent err = %5.2f +/- %5.2f'%(mean(pdiff),std(pdiff))
                print(s)
                text(.1,.9,s,horizontalalignment='left',transform=gca().transAxes)

            else:

                axhline(y=0,color='k',ls='--')
                #errorbar(xp[flag],yp[flag],yerr=ye[flag],fmt=None)#,ecolor='k')

                s='ave diff = %5.2f +/- %5.2f'%(mean(yp[flag]),std(yp[flag]))
                print(s)
                text(.1,.9,s,horizontalalignment='left',transform=gca().transAxes)

            xlabel(xl[i])
            ylabel(yl[i])
            if i == 1:
                #axhline(y=0,color='k',ls='-')
                #axhline(y=2,color='k',ls='--')
                #axhline(y=-2,color='k',ls='--')
                plot(xline,xline-2,'k:')
                plot(xline,xline+2,'k:')
            if i == 0:
                title(self.prefix)
                if starflag:
                    print('analyzing stars :)')
                    title(self.prefix+' Stars')
                else:
                    pinit = [.5, 0.5]
                    snrflag=self.sim.snr24>snrcut
                    #print xp[snrflag]
                    #print yp[snrflag]
                    yer=ye[snrflag]
                    for i in range(len(yer)):
                        if yer[i] == 0:
                            yer[i]=99

                    #print yer
                    #figure(10)
                    #clf()
                    #plot(xp[snrflag],yp[snrflag],'bo')
                    #p1,success = scipy.optimize.leastsq(errfunc, pinit,args=(xp[snrflag], yp[snrflag]))
                    #print 'p1 = ',p1,p1[0]
                    #yfit=fitfunc(p1,xl)
                    #print yfit
                    
                    #plot(xl, yfit, "r-") # Plot of the data and the fit
                    #plot(xline,.8*xline,'k:')
                    plot(xline,.5*xline,'k:')

                    gca().set_xscale('log')
                    gca().set_yscale('log')

        colorbar(sp)
        if starflag:

            fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'galfitstars.eps'
        else:
            fname=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'SNR'+str(snrcut)+'galfitsim.eps'
        savefig(fname)

    def plotresultsv2(self,usesb=1,starflag=0,usesersic=0,usesnr=0,asymflag=0,snrcut=4,usere=0,minre=2.,sbcut=22.,sbcutobs=21.,convflag=0):
    
        fitfunc = lambda p, x: p[0] + p[1] * x
        #errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
        errfunc = lambda p, x, y: (y - fitfunc(p, x)) 

        flag=(self.sim.modelmag+2.5*log10(pi*self.sim.modelre**2)) < sbcut
        statflag = (self.sim.modelmag + 2.5*log10(pi*(self.sim.modelre**2))) < sbcut

        statflagobs = (self.sb_obs < sbcutobs) & (self.sim.cnumerical_error_flag24 < .1)
        if convflag:
            statflagobs = (self.csb_obs < sbcutobs) & (self.sim.cnumerical_error_flag24 < .1)
        flag=self.sim.modelmag < 18.
        if usesb:
            colorscale=self.sb 
            v1=13
            v2=25
        elif usesersic:
            colorscale=abs(self.sim.modeln - self.sim.cnsersic1)
            v1=0
            v2=6
        elif usere:
            colorscale=self.sim.modelre
            v1=3
            v2=30
            #flag=self.sim.modelre < minre
        elif usesnr:
            colorscale=self.sim.snr24
            v1=1
            v2=10
            flag=(self.sim.snr24 > snrcut) & (self.sim.cre1 > minre) & (self.sim.cnumerical_error_flag24 < .1)
        else:
            colorscale=self.sim.modelmag
            v1=minmodelmag
            v2=maxmodelmag
            if starflag:
                v1=11
                v2=15

        figure(figsize=(10,12))
        subplots_adjust(left=.1,right=.95,bottom=.1,top=.95,wspace=.2,hspace=.05)

        xl=['Mag input','model mag/Re^2','measured mag/Re^2']

        #yl=['R_e out- R_e input/R_e input','Sersic n out','B/A out','mag out']
        #yl=['R_e out- R_e input/R_e input','Sersic n_out - n_in','B/A out','mag_out - mag_in']
        #yl=['R_e out- R_e input/R_e input','Sersic n_out','B/A out','mag_out - mag_in']
        yl=['R_e out - in','Sersic n out - in','B/A out - in','mag out - in','dr (pixel)','dr (pixel)']
        x=[self.sim.modelmag,self.sb,self.sb_obs ]
        if asymflag:
            y=[(self.sim.asym_re1-self.sim.modelre/mipspixelscale)/(self.sim.modelre/mipspixelscale),(self.sim.asym_nsersic1),self.sim.asym_axisratio1,(self.sim.asym_mag1-self.sim.modelmag)]
        #erry=[self.sim.re1err/(self.sim.modelre/mipspixelscale),self.sim.nsersic1err,self.sim.axisratio1err,self.sim.mag1err]
            erry=[self.sim.asym_re1err/(self.sim.modelre/mipspixelscale),self.sim.asym_nsersic1err/self.sim.modeln,self.sim.asym_axisratio1err,self.sim.asym_mag1err/self.sim.modelmag]

        else:
            print('got here!')
            y=[(self.sim.cre1-self.sim.modelre/mipspixelscale),(self.sim.cnsersic1-self.sim.modeln),self.sim.caxisratio1-self.sim.modelba,(self.sim.cmag1-self.sim.modelmag),sqrt((self.sim.cyc1-self.sim.modelyc)**2+(self.sim.cxc1-self.sim.modelxc)**2)]
            ynorm=[self.sim.modelre/mipspixelscale,self.sim.modeln,self.sim.modelba,self.sim.modelmag,ones(len(self.sim.xc1),'f')]
            #erry=[self.sim.re1err,self.sim.nsersic1err,self.sim.axisratio1err,self.sim.mag1err/self.sim.modelmag,self.sim.xc1err,self.sim.yc1err]
            if convflag:
                y=[(self.sim.cre1-self.sim.modelre/mipspixelscale),(self.sim.cnsersic1-self.sim.modeln),self.sim.caxisratio1-self.sim.modelba,(self.sim.cmag1-self.sim.modelmag),sqrt((self.sim.cyc1-self.sim.modelyc)**2+(self.sim.cxc1-self.sim.modelxc)**2)]


        xmin=[.5,0,-0.05,8.]
        xmax=[50,7,1.05,18.]
        #ymin=[-.5,0,-0.05,-.5]
        #ymax=[.5,6,1.0,.5]
        ymin=[-12,0,-0.05,-.5]
        ymax=[30,6,1.0,.5]
        panels=[0,1,2]
        for i in range(5):
            for j in range(3):
                nsubplot=2*i+j+1
                subplot(5,3,3*i+j+1)
                xp=x[j]
                yp=y[i]
                if usesnr:
                    try:
                        plot(xp[~flag],yp[~flag],'k.')
                    except:
                        print('WARNING: problem plotting points w/snr below cut')
                sp=scatter(xp[flag],yp[flag],marker='o',s=ones(len(self.sim.re1[flag]))*40,c=colorscale[flag],vmin=v1,vmax=v2)
            #plot(xp[flag],yp[flag],'bo')
            #errorbar(xp[flag],yp[flag],yerr=ye[flag],fmt=None)#,ecolor='k')
            #    if i < 4:
            #       axis([xmin[i],xmax[i],ymin[i],ymax[i]])
            #   if i >:
            #       gca().set_yscale('log')

                x1,x2=xlim()
                axhline(y=0,color='k',ls='--')
                if i < 4:
                    #print j
                    pdiff=abs(yp[statflag])/array(ynorm[i][statflag],'f')
                    if j == 2:
                        pdiff=abs(yp[statflagobs])/array(ynorm[i][statflagobs],'f')
                        #print pdiff
                    s='ave per cent err = %5.2f +/- %5.2f'%(mean(pdiff),std(pdiff))
                    s='err = %4.2f +/- %4.2f'%(mean(pdiff),std(pdiff))
                    #print s
                    text(.1,.9,s,horizontalalignment='left',transform=gca().transAxes,fontsize=10)
                if i == 4:
                    xlabel(xl[j])
                else:
                    gca().set_xticklabels(([]))
                if j == 0:
                    ylabel(yl[i])
                if j == 1:
                    axvline(x=sbcut,ls=':',color='k')
                if j == 2:
                    axvline(x=sbcutobs,ls=':',color='k')
                if i == 1:
                #axhline(y=0,color='k',ls='-')
                    axhline(y=2,color='k',ls=':')
                    axhline(y=-2,color='k',ls=':')
                if (i+j == 0):
                    title(self.prefix)
                if i < 1:
                    ylim(ymin[0],ymax[0])
        #colorbar(sp)
        fname=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'SB'+'galfitsim.eps'
        savefig(fname)
    def plotsizesb(self,sbcut=22.,sbcutobs=21.,plotsingle=1,usesb=1,deltaflag=0):
        if plotsingle:
            figure()
            xlabel('Observed SB')
            ylabel('Re(out)/Re(in)')
        baseflag=(self.cdr < 5) & self.sim.cgalfit_flag & (self.sim.cnumerical_error_flag24 < .1)
        if usesb:
            #print 'using sb'
            colorscale=self.csb_obs 
            v1=sbmin
            v2=sbmax
            flag=self.cstatflagobs
        else:
            flag=self.statflag
            colorscale=self.sb 
            v1=13
            v2=25

        if deltaflag:
            size=(self.sim.cre1*mipspixelscale-self.sim.modelre)/self.sim.modelre
        else:
            size=self.sim.re1*mipspixelscale/self.sim.modelre
            csize=self.sim.cre1*mipspixelscale/self.sim.modelre
        flag=self.cstatflagobs
        errcsize=self.sim.cre1err*mipspixelscale/self.sim.modelre
        errorbar(self.csb_obs[flag],csize[flag],errcsize[flag],fmt=None,ecolor='0.5')
        #sp=scatter(self.csb_obs[flag],csize[flag],marker='^',s=ones(len(self.sim.re1[flag]))*50,c=colorscale[flag],vmin=v1,vmax=v2)
        #sp=scatter(self.csb_obs[baseflag],csize[baseflag],marker='^',s=50*ones(len(csize[baseflag])),c=colorscale[baseflag],vmin=v1,vmax=v2)
        plot(self.csb_obs[baseflag],csize[baseflag],'ko')#,s=50*ones(len(csize[baseflag])),c=colorscale[baseflag],vmin=v1,vmax=v2)

        #plot(self.csb_obs[flag],size[flag],'k.')
        if deltaflag:
            #axis([16,23,-1,1])
            axhline(y=0,color='k')
            axhline(y=-.5,color='k',ls='--')        
        else:
            #axis([16,23,0,2])
            axhline(y=1,color='k')
            axhline(y=.5,color='k',ls='--')
            axvline(x=self.sbcutobs,color='k',ls='--')
        axis([13.3,22.2,.2,5])
        xticks(arange(14,24,2))
        gca().set_yscale('log')
        text(.6,.85,self.prefix,transform=gca().transAxes,fontsize=14)
        s='ave=%5.2f+/-%5.2f'%(mean(csize[flag]),std(csize[flag]))
        text(.1,.15,s,transform=gca().transAxes,fontsize=10)

    def plotsizeRe(self,sbcut=22.,sbcutobs=21.,plotsingle=1,usesb=1,deltaflag=0,convflag=1):
        if plotsingle:
            figure()
            xlabel('Model Re (arcsec)')
            ylabel('Re(out)/Re(in)')
        cy=self.sim.cre1*mipspixelscale
        cyerr=self.sim.cre1err*mipspixelscale
        y=self.sim.re1*mipspixelscale
        yerr=self.sim.re1err*mipspixelscale
        if usesb:
            colorscale=self.csb_obs 
            v1=13
            v2=25
            flag=self.cstatflagobs
        else:
            flag=self.statflag
            colorscale=self.sb 
            v1=13
            v2=25

        if deltaflag:
            csize=(cy-self.sim.modelre)/self.sim.modelre
            size=(y-self.sim.modelre)/self.sim.modelre
        else:
            csize=cy/self.sim.modelre
            size=y/self.sim.modelre
        errsize=cyerr/self.sim.modelre
        errorbar(self.sim.modelre[self.cstatflagobs],csize[self.cstatflagobs],yerr=errsize[self.cstatflagobs],fmt=None,ecolor='0.5')

        #sp=scatter(self.sim.modelre[flag],size[flag],marker='.',s=ones(len(self.sim.re1[flag]))*40,c=colorscale[flag],vmin=v1,vmax=v2)
        #plot(self.sim.modelre[flag],size[flag],'k.')#marker='.',s=ones(len(self.sim.re1[flag]))*40,c=colorscale[flag],vmin=v1,vmax=v2)
        print('Number w/successful fits = ',sum(self.cstatflagobs))
        #sp=scatter(self.sim.modelre[self.cstatflagobs],csize[self.cstatflagobs],marker='^',s=ones(len(self.sim.re1[self.cstatflagobs]))*40,c=colorscale[self.cstatflagobs],vmin=v1,vmax=v2)
        plot(self.sim.modelre[self.cstatflagobs],csize[self.cstatflagobs],'ko')#
        #errorbar(self.sim.modelre[flag],size[flag],errsize[flag],fmt=None)

        if deltaflag:
            axis([1,60,-1,1])
            axhline(y=0,color='k')
            axhline(y=-.5,color='k',ls='--')        
        else:
            axis([1,60,0.2,5])
            axhline(y=1,color='k')
            #axhline(y=.5,color='k',ls='--')
            axvline(x=mipspixelscale,color='k',ls=':')

        gca().set_xscale('log')
        gca().set_yscale('log')
        text(.6,.85,self.prefix,transform=gca().transAxes,fontsize=14)
        s='ave=%5.2f+/-%5.2f'%(mean(csize[self.cstatflagobs]),std(csize[self.cstatflagobs]))
        text(.1,.15,s,transform=gca().transAxes,fontsize=10)
    def plotsample(self):
        figure(figsize=(10,8))
        subplots_adjust(hspace=.3)
        subplot(2,2,1)
        scatter(self.sim.modelmag,self.sim.cmodelre,s=40,c=self.sim.cmodeln,vmin=.5,vmax=4)
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by model sersic index',fontsize=12)
        subplot(2,2,2)
        scatter(self.sim.modelmag[self.sim.cgalfit_flag],self.sim.cmodelre[self.sim.cgalfit_flag],s=40,c=self.sim.snr24[self.sim.cgalfit_flag],vmin=1,vmax=10)
        plot(self.sim.modelmag[~self.sim.cgalfit_flag],self.sim.cmodelre[~self.sim.cgalfit_flag],'kx')
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by SNR',fontsize=12)
        subplot(2,2,3)
        colorscale=self.sb
        scatter(self.sim.modelmag,self.sim.cmodelre,s=40,c=colorscale,vmin=13,vmax=25)
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by model SB',fontsize=12)

        subplot(2,2,4)
        #colorscale=self.sim.f24/self.sim.isoarea
        colorscale=self.csb_obs
        scatter(self.sim.modelmag,self.sim.cmodelre,s=40,c=colorscale,vmin=13,vmax=25)
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by measured SB',fontsize=12)
        savefig(homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_sample.eps')




def write_galfit_image_param(outfile,input_image,output_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,convflag=1,constraintflag=1,fitallflag=0):
    outfile.write('# IMAGE PARAMETERS\n')
    outfile.write('A) '+input_image+'              # Input data image (FITS file)\n')
    outfile.write('B) '+output_image+'       # Name for the output image\n')
    outfile.write('C) %s                # Sigma image name (made from data if blank or "none") \n'%(sigma_image))
    if convflag:
        outfile.write('D) '+psf_image+'     # Input PSF image and (optional) diffusion kernel\n')
    outfile.write('E) %i                   # PSF oversampling factor relative to data\n'%(psf_oversampling))
    if fitallflag:
        outfile.write('F)            # Pixel mask (ASCII file or FITS file with non-0 values)\n')
    else:
        outfile.write('F) '+mask_image+'           # Pixel mask (ASCII file or FITS file with non-0 values)\n')
    if constraintflag:
        outfile.write('G) /Users/rfinn/research/LocalClusters/GalfitAnalysis/sersic.constraint        # Parameter constraint file (ASCII)\n')
    outfile.write('H) '+str(int(round(xminfit)))+' '+str(int(round(xmaxfit)))+' '+str(int(round(yminfit)))+' '+str(int(round(ymaxfit)))+'     # Image region to fit (xmin xmax ymin ymax)\n')
    if convflag:
        outfile.write('I) '+str(int(round(convolution_size)))+' '+str(int(round(convolution_size)))+'             # Size of convolution box (x y)\n')
    outfile.write('J) %5.2f              # Magnitude photometric zeropoint \n'%(magzp))
    outfile.write('K) %6.5f   %6.5f         # Plate scale (dx dy)  [arcsec/pix]\n'%(pscale,pscale))
    outfile.write('O) regular                # Display type (regular, curses, both)\n')
    outfile.write('P) 0                   # Create output image only? (1=yes; 0=optimize) \n')
    outfile.write('S) 0                   # Modify/create objects interactively?\n')


def write_galfit_sersic(outfile,objnumber,profile,xobj,yobj,mag,rad,sersic_exp,axis_ratio,pa,fixsersic=0,asymmetry=0,fixpa=0,fixba=0):
    print('%%%%%%%%%%%%%%%%%%')
    print('fixba = ',fixba)
    outfile.write(' \n')
    outfile.write('# Object number: %i \n'%(objnumber))
    outfile.write(' 0) %s             # Object type \n'%(profile))
    outfile.write(' 1) %8.1f  %8.1f 1 1  # position x, y        [pixel] \n'%(xobj,yobj))
    outfile.write(' 3) %5.2f      1       # total magnitude     \n'%(mag))
    outfile.write(' 4) %8.2f       1       #     R_e              [Pixels] \n'%(rad))
    if fixsersic:
        outfile.write(' 5) %5.2f       0       # Sersic exponent (deVauc=4, expdisk=1)   \n'%(sersic_exp))
    else:
        outfile.write(' 5) %5.2f       1       # Sersic exponent (deVauc=4, expdisk=1)   \n'%(sersic_exp))
    if fixba:
        outfile.write(' 9) %5.2f       0       # axis ratio (b/a)    \n'%(axis_ratio))
    else:
        outfile.write(' 9) %5.2f       1       # axis ratio (b/a)    \n'%(axis_ratio))
    if fixpa:
        outfile.write('10) %5.2f       0       # position angle (PA)  [Degrees: Up=0, Left=90] \n'%(pa))
    else:
        outfile.write('10) %5.2f       1       # position angle (PA)  [Degrees: Up=0, Left=90] \n'%(pa))
    if asymmetry:
        outfile.write('F1) 0.0001 0.00   1  1     # azim. Fourier mode 1, amplitude & phase angle \n')
    outfile.write(" Z) 0                  # Output option (0 = residual, 1 = Don't subtract)  \n")

def write_galfit_sky(outfile,objnumber,sky):    
    outfile.write(' \n')
    outfile.write('# Object number: %i \n'%(objnumber))
    outfile.write(' 0) sky             # Object type \n')
    outfile.write(' 1) %8.1f   1  # sky background at center of fitting region [ADUs] \n'%(sky))
    outfile.write(' 2) 0      0       # dsky/dx (sky gradient in x)    \n')
    outfile.write(' 3) 0      0       # dsky/dy (sky gradient in y) \n')
    outfile.write(" Z) 0                  # Output option (0 = residual, 1 = Don't subtract)  \n")

def parse_galfit_1comp(galfit_outimage,asymflag=0):
    numerical_error_flag=0
    if asymflag:
        header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_SKY','1_F1','1_F1PA']
    else:
        header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_SKY']
    fit_parameters=[]
    working_dir=os.getcwd()+'/'
    for hkey in header_keywords:
        iraf.imgets(image=galfit_outimage,param=hkey)
        s=iraf.imgets.value
        #print hkey,t
        if s.find('[') > -1:
            t=s.rstrip(']')
            q=t.replace('[','')
            values=(float(q),0.)# fit and error
        else:
            t=s.split('+/-')
            try:
                values=(float(t[0]),float(t[1]))# fit and error
            except ValueError:
                # look for * in the string, which indicates numerical problem
                if t[0].find('*') > -1:
                    numerical_error_flag=1
                    t[0]=t[0].replace('*','')
                    t[1]=t[1].replace('*','')
                    values=(float(t[0]),float(t[1]))# fit and error
                
        fit_parameters.append(values)
    #iraf.imgets(image=galfit_outimage,param='1_YC')
    #t=iraf.imgets.value.split('+/-')
    #y_fit=(float(t[0]),float(t[1]))# fit and error
    #iraf.imgets(image=galfit_outimage,param='1_MAG')
    #t=iraf.imgets.value.split('+/-')
    #mag_fit=(float(t[0]),float(t[1]))# fit and error
    #iraf.imgets(image=galfit_outimage,param='1_RE')
    #t=iraf.imgets.value.split('+/-')
    #Re_fit=(float(t[0]),float(t[1]))# fit and error
    #iraf.imgets(image=galfit_outimage,param='1_N')
    #t=iraf.imgets.value.split('+/-')
    #Nsersic_fit=(float(t[0]),float(t[1]))# fit and error
    #iraf.imgets(image=galfit_outimage,param='1_AR')
    #t=iraf.imgets.value.split('+/-')
    #axis_ratio_fit=(float(t[0]),float(t[1]))# fit and error
    #iraf.imgets(image=galfit_outimage,param='1_PA')
    #t=iraf.imgets.value.split('+/-')
    #pa_fit=(float(t[0]),float(t[1]))# fit and error
    ## get best-fit sky values
    #iraf.imgets(image=galfit_outimage,param='2_SKY')
    #t=iraf.imgets.value.split('+/-')
    #sky_fit=(float(t[0]),float(t[1]))# fit and error

    #return x_fit,y_fit,mag_fit,Re_fit,Nsersic_fit,axis_ratio_fit,pa_fit,sky_fit
    fit_parameters.append(numerical_error_flag)
    return fit_parameters

def parse_galfit_2comp(galfit_outimage):
    header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_XC','2_YC','2_MAG','2_RE','2_N','2_AR','2_PA','3_SKY']
    fit_parameters=[]
    numerical_error_flag=0
    numerical_error_flag2=0
    working_dir=os.getcwd()+'/'
    for hkey in header_keywords:
        iraf.imgets(image=galfit_outimage,param=hkey)
        t=iraf.imgets.value.split('+/-')
        try:
            values=(float(t[0]),float(t[1]))# fit and error
        except ValueError:
            # look for * in the string, which indicates numerical problem
            if t[0].find('*') > -1:
                if hkey.find('1') > -1:
                    numerical_error_flag=1
                if hkey.find('1') > -1:
                    numerical_error_flag2=1
            t[0]=t[0].replace('*','')
            t[1]=t[1].replace('*','')
            values=(float(t[0]),float(t[1]))# fit and error
        fit_parameters.append(values)
    fit_parameters.append(numerical_error_flag)
    fit_parameters.append(numerical_error_flag2)
    return fit_parameters

def parse_galfit_2compold(galfit_outimage):
    working_dir=os.getcwd()+'/'
    iraf.imgets(image=galfit_outimage,param='1_XC')
    t=iraf.imgets.value.split('+/-')
    x_fit1=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='1_YC')
    t=iraf.imgets.value.split('+/-')
    y_fit1=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='1_MAG')
    t=iraf.imgets.value.split('+/-')
    mag_fit1=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='1_RE')
    t=iraf.imgets.value.split('+/-')
    Re_fit1=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='1_N')
    t=iraf.imgets.value.split('+/-')
    Nsersic_fit1=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='1_AR')
    t=iraf.imgets.value.split('+/-')
    axis_ratio_fit1=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='1_PA')
    t=iraf.imgets.value.split('+/-')
    pa_fit1=(float(t[0]),float(t[1]))# fit and error
    
    iraf.imgets(image=galfit_outimage,param='2_XC')
    t=iraf.imgets.value.split('+/-')
    x_fit2=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='2_YC')
    t=iraf.imgets.value.split('+/-')
    y_fit2=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='2_MAG')
    t=iraf.imgets.value.split('+/-')
    mag_fit2=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='2_RE')
    t=iraf.imgets.value.split('+/-')
    Re_fit2=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='2_N')
    t=iraf.imgets.value.split('+/-')
    Nsersic_fit2=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='2_AR')
    t=iraf.imgets.value.split('+/-')
    axis_ratio_fit2=(float(t[0]),float(t[1]))# fit and error
    iraf.imgets(image=galfit_outimage,param='2_PA')
    t=iraf.imgets.value.split('+/-')
    pa_fit2=(float(t[0]),float(t[1]))# fit and error
    
    # get best-fit sky values
    iraf.imgets(image=galfit_outimage,param='3_SKY')
    t=iraf.imgets.value.split('+/-')
    sky_fit=(float(t[0]),float(t[1]))# fit and error
    return x_fit1,y_fit1,mag_fit1,Re_fit1,Nsersic_fit1,axis_ratio_fit1,pa_fit1,x_fit2,y_fit2,mag_fit2,Re_fit2,Nsersic_fit2,axis_ratio_fit2,pa_fit2,sky_fit


def rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,convolutionflag=1,simflag=0,NSAsersic=0,nsersic=2,constrflag=1,fixnsersic=0,fixnsersicvalue=5,fitall=0,asymflag=0,fixba=0,fixpa=0):
    
    # FLAGS & parameters:
    #
    # simflag
    # 
    # NSAsersic  - if set, use NSAsersic as input sersic index
    # nsersic - see above
    #
    # constrflag
    #
    # fixnsersic - if set, set sersic index to fixnsersicvalue
    # fixnsersicvalue - see above
    # 
    # fitall - perform a simultaneous fit on objects with distance greater than 10 pixels
    #
    # asymflag - add fourier mode to fit
    

    initial_rad=rad
    profile = 'sersic'
    if NSAsersic:
        sersic_exp=nsersic
    elif fixnsersic:
        sersic_exp=fixnsersicvalue
    else:
        sersic_exp =2
    # open ds9 display
    d=ds9.ds9()
    galflag=zeros(3,'i')
    quitflag=0
    stop_gal_flag=0
    too_faint_flag=0
    working_dir=os.getcwd()+'/'
    d.set('cd '+working_dir)
    for j in loops:

        if asymflag:
            output_image=galname+'-'+str(j+1)+'Comp-galfit-out-asym.fits'
        else:
            output_image=galname+'-'+str(j+1)+'Comp-galfit-out.fits'
        if (convolutionflag < .1):
            output_image=galname+'-'+str(j+1)+'Comp-noconv-galfit-out.fits'
        # create galfit input file
        if simflag:
            galfile=galname+'galfit.sim.input.'+str(j+1)+'Comp'
        else:
            if convolutionflag:
                galfile=galname+'galfit.input.'+str(j+1)+'Comp'
            else:
                galfile=galname+'galfit.input.'+str(j+1)+'Comp-noconv'
        galfit_input=open(galfile,'w')
        
        write_galfit_image_param(galfit_input,input_image,output_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,convflag=convolutionflag,constraintflag=constrflag,fitallflag=fitall)
        
        if j == 0:
            
            objnumber=1
            write_galfit_sersic(galfit_input,objnumber,profile,xobj,yobj,mag_total,rad,sersic_exp,axis_ratio,PA,fixsersic=fixnsersic,asymmetry=asymflag,fixba=fixba,fixpa=fixpa)
            objnumber=2
            write_galfit_sky(galfit_input,objnumber,sky)    
            


        elif j == 1:
            # read in output from round 1
            # get values from header of image[2]
            galfit_outimage=galname+'-'+'1Comp-galfit-out.fits[2]'
            
            (x_fit,y_fit,mag_fit,Re_fit,Nsersic_fit,axis_ratio_fit,pa_fit,sky_fit,numerical_error_flag)=parse_galfit_1comp(galfit_outimage)
            
            
            #############################################################################
            #
            #    second time fits 2 sersic profiles
            #    - use output from first pass (mag_0 and Re_0)
            #    - put half luminosity (output mag_0 - 0.75) in output from first fit
            #    - add second sersic profile with half luminosity (output mag_0 - 0.75)
            #    - make size of second component 1/2 - 3/4 of size of first (Re = 0.5 Re)
            #    - display results in ds9
            # 
            #############################################################################
            rad1=Re_fit[0]

            if rad1 < 2:
                rad1=initial_rad

            #    - if n > 4 for subcomponents, set it back to n=1-2 at
            nsers1=Nsersic_fit[0]
            if nsers1 > 4:
                nsers1=2
            elif nsers1 < .1:
                nsers1=2
            objnumber=1
            mag=mag_fit[0]+0.75
            write_galfit_sersic(galfit_input,objnumber,profile,x_fit[0],y_fit[0],mag,rad,nsers1,axis_ratio_fit[0],pa_fit[0])
            objnumber=2
            rad=0.5*rad1
            write_galfit_sersic(galfit_input,objnumber,profile,x_fit[0],y_fit[0],mag,rad,2,axis_ratio_fit[0],pa_fit[0])
            objnumber=3
            write_galfit_sky(galfit_input,objnumber,sky_fit[0])    
            print('wrote input file for 2comp fit')
        elif j == 2:
            #############################################################################
            #
            #    third time - fit 3 sersic profiles
            #    - try using mean of 2-component fit for the 3rd component
            #      * split the flux 3 ways
            #    - if 2 components have similar values,
            #      * use 0.5-0.75 of the other components
            #    - if n > 4 for subcomponents, set it back to n=1-2 at
            #    - if size is tool small (<1 pixel), reset to a bigger value
            #
            #############################################################################
            # read in output from round 2
            
            galfit_outimage=galname+'-'+'2Comp-galfit-out.fits[2]'
            x_fit1,y_fit1,mag_fit1,Re_fit1,Nsersic_fit1,axis_ratio_fit1,pa_fit1,x_fit2,y_fit2,mag_fit2,Re_fit2,Nsersic_fit2,axis_ratio_fit2,pa_fit2,sky_fit=parse_galfit_2comp(galfit_outimage)

            flux_tot=10.**(-1.*mag_fit1[0])+10.**(-1.*mag_fit2[0])
            aveflux=flux_tot
            mag=-2.5*log10((10.**(-1.*mag_fit1[0]/2.5)+10.**(-1.*mag_fit2[0])/2.5)/3.)
            print('HHHHHEEEEEYYYYY')
            print(mag, mag_fit1[0],mag_fit2[0])
            # check fit parameters for numerical problems
            #
            #    - if size is tool small (<1 pixel), reset to a bigger value
            rad1=Re_fit1[0]
            rad2=Re_fit2[0]
            if rad1 < 2:
                rad1=initial_rad
            if rad2 < 2:
                rad2=initial_rad
            rad3=0.5*(rad1+rad2)

            #    - if n > 4 for subcomponents, set it back to n=1-2 at
            nsers1=Nsersic_fit1[0]
            if nsers1 > 4:
                nsers1=2
            elif nsers1 < .1:
                nsers1=2
            nsers2=Nsersic_fit2[0]
            if nsers2 > 4:
                nsers2=2
            elif nsers1 < .1:
                nsers1=2
            nsers3=2
            # write out galfit input file w/3 components to fit
                
            objnumber=1
            write_galfit_sersic(galfit_input,objnumber,profile,x_fit1[0],y_fit1[0],mag,rad1,nsers1,axis_ratio_fit1[0],pa_fit1[0])
            objnumber=2
            write_galfit_sersic(galfit_input,objnumber,profile,xobj,yobj,mag,rad2,nsers2,axis_ratio_fit2[0],pa_fit2[0])
            objnumber=3
            pa=PA
            axis_ratio=0.5*(axis_ratio_fit1[0]+axis_ratio_fit2[0])
            #xobj=0.5*(x_fit1[0]+x_fit2[0])
            #yobj=0.5*(y_fit1[0]+y_fit2[0])
            write_galfit_sersic(galfit_input,objnumber,profile,xobj,yobj,mag,rad3,nsers3,axis_ratio,pa)
            objnumber=4
            write_galfit_sky(galfit_input,objnumber,sky)    

        if fitall:
            os.system('cp '+homedir+'research/LocalClusters/sextractor/default.param .')
            os.system('cp '+homedir+'research/LocalClusters/sextractor/default.nnw .')
            s='sex '+input_image+' -c '+homedir+'research/LocalClusters/sextractor/default.sex.24um.galfitsource -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+sigma_image+' -CATALOG_NAME '+galname+'test.fits -CATALOG_TYPE FITS_1.0'
            os.system(s)
            # read in SE table to get x,y for sources
            fname=galname+'test.fits'
            try:
                se=atpy.Table(fname)
                print('found ',len(se.X_IMAGE),' sources on the field of ',galname)
                nearbyobjflag=sqrt((se.X_IMAGE-xobj)**2+(se.Y_IMAGE-yobj)**2) > 10.
                for k in range(len(se.X_IMAGE)):
                    if nearbyobjflag[k]:
                        objnumer=objnumber+1
                        write_galfit_sersic(galfit_input,objnumber,profile,se.X_IMAGE[k],se.Y_IMAGE[k],se.MAG_BEST[k],se.FLUX_RADIUS[k,0],2,se.B_IMAGE[k]/se.A_IMAGE[k],se.THETA_IMAGE[k])
            except AttributeError:
                print('WARNING: no sources detected in image!')
        galfit_input.close()
        if simflag:
            s = 'galfit -o2 '+galfile
            os.system(s)
            return
            # run galfit
        s = 'galfit '+galfile
        print('run the following: ',s)
        try:
            errno=os.system(s)
            print('errno = ',errno)
            if errno == 25600:
                print('Error running galfit ', errno)
                galflag[j]=0
            else:
                galflag[j]=1
        except:
            print('problem running galfit for ',j+1,' comp fit for ',image_id)
            galflag[j]=0
            if interactive:
                flag=input('press any key to continue, q to quit; s to stop running galfit on this galaxy')
                if flag.find('q') > -1:
                    quitflag=1
                elif string.find('s') > -1:
                    stop_gal_flag=1
                    flag=input('is galaxy too faint? (y or n)')
                    if flag.find('y') > -1:
                        too_faint_flag=1
                    return galflag[0],galflag[1],galflag[2],quitflag,stop_gal_flag,too_faint_flag
                continue

        
        image_id=galname+'-'
        galfit_log=image_id+str(j+1)+'Comp-fit.log'
        s='cp fit.log '+galfit_log
        os.system(s)
        galfit_out=image_id+str(j+1)+'Comp'+'-galfit.01'
        s='mv galfit.01 '+galfit_out
        try:
            os.rename('galfit.01',galfit_out)
        except:
            print("appears like galfit did not complete")
            galflag[j]=0

        if galflag[j]:
            # create image of subcomponents
            subcomp_image=image_id+str(j+1)+'Comp'+'-subcomps.fits'
            s='galfit -o3 '+galfit_out
            os.system(s)
            os.rename('subcomps.fits',subcomp_image)
            #    - display results (like ds9 multiextension data cube -> use xpa)
            #
            d.set('frame delete all')
            print('file to display = ',output_image)
            s='file new multiframe '+output_image
            print(s)
            d.set(s)
            d.set('frame delete 1')
            for k in range(2,5):
                s='frame '+str(k)
                d.set(s)
                #d.set('scale linear')
                d.set('zoom to fit')
                print(k)
                if k == 2:
                    d.set('regions command {text 30 10 #text="Image" font="times 18 bold" color="red"}')
                if k == 3:
                    d.set('regions command {text 30 10 #text="Model" font="times 18 bold" color="red"}')
                if k == 4:
                    d.set('regions command {text 30 10 #text="Residual" font="times 18 bold" color="red"}')
            d.set('frame match wcs')
            if j == 0:
                galfit_outimage=galname+'-'+'1Comp-galfit-out.fits[2]'
                (x_fit,y_fit,mag_fit,Re_fit,Nsersic_fit,axis_ratio_fit,pa_fit,sky_fit,numerical_error_flag)=parse_galfit_1comp(galfit_outimage)
                #d.set('frame 7')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit[0],Re_fit[0],Nsersic_fit[0],axis_ratio_fit[0],pa_fit[0])
                print(s)
                #d.set(s)
            if j == 1:
                galfit_outimage=galname+'-'+'2Comp-galfit-out.fits[2]'
                x_fit1,y_fit1,mag_fit1,Re_fit1,Nsersic_fit1,axis_ratio_fit1,pa_fit1,x_fit2,y_fit2,mag_fit2,Re_fit2,Nsersic_fit2,axis_ratio_fit2,pa_fit2,sky_fit=parse_galfit_2comp(galfit_outimage)
                #d.set('frame 7')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit1[0],Re_fit1[0],Nsersic_fit1[0],axis_ratio_fit1[0],pa_fit1[0])
                print(s)
                #d.set(s)
                #d.set('frame 8')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit2[0],Re_fit2[0],Nsersic_fit2[0],axis_ratio_fit2[0],pa_fit2[0])
                #d.set(s)
            #raw_input('hit any key when ready to view subcomponent images \n')
            print('file to display = ',subcomp_image)
            s='file new multiframe '+subcomp_image
            d.set(s)

            if j == 0:
                endframe=8
            if j == 1:
                endframe=9
            if j == 2:
                endframe=10
            s='frame delete '+str(endframe)
            #d.set(s)
            d.set('frame delete 5')
            d.set('frame delete 6')
            d.set('frame 7')
            d.set('file '+mask_image)

            for k in range(2,endframe):
                if k == 5:
                    continue
                if k == 6:
                    continue
                s='frame '+str(k)
                d.set(s)
                d.set('scale log')
                d.set('zoom to fit')
            d.set('frame match wcs')
            if save_ds9_fig:
                d.set('saveimage png image.png')
                img_name=image_id+str(j+1)+'Comp.png'
                os.rename('image.png',img_name)

            if interactive:
                string=input('hit any key when ready to continue (q to quit; s to stop running galfit for this galaxy) \n')
                if string.find('q') > -1:
                    quitflag=1
                    break
                elif string.find('s') > -1:
                    print('gal and quitflags = ',galflag[0],galflag[1],galflag[2],quitflag)
                    stop_gal_flag=1
                    flag=input('is galaxy too faint? (y or n)')
                    if flag.find('y') > -1:
                        too_faint_flag=1
                    return galflag[0],galflag[1],galflag[2],quitflag,stop_gal_flag,too_faint_flag
        else:
            print("not diplaying images b/c galfit didn't finish")
            print("not going to try to fit anything else w/this galaxy (if there is anymore...)")
            return galflag[0],galflag[1],galflag[2],quitflag
    return galflag[0],galflag[1],galflag[2],quitflag,stop_gal_flag,too_faint_flag


#    ntimes=3 parameter
#    - give the user the option to run galfit 1, 2, or 3 times
#        if ntimes == 3, run 3 times
#  
#        elif ntimes == 2, run 2 times
#
#        elif ntimes == 1, run once
#
#        elif ntimes= -3, run one time, assumes this is the 3rd time through
#
#        elif ntimes= -2, run one time, assumes this is the second time through
#


def set_ntimes(ntimes):
    global loops
    if ntimes == 3:
        loops=arange(3)
    elif ntimes == 2:
        loops=arange(2)
    elif ntimes == 1:
        loops=arange(1)
    elif ntimes == -3:
        loops=array([2],'i')
    elif ntimes == -2:
        loops=array([1],'i')
    else:
        print("Didn't recognize ntimes")
        return


try:
    cluster=sys.argv[1]
    doall=0
except:
    doall=1
if not(doall):
    cl=Cluster(cluster)
    cl.rungalfit_sim_24(runSE=1)

    #cl.readsimresults()
    
    #cl.review(usesnrflag=1)
    #cl.plotresults(usesnr=1,snrcut=4)
    #cl.plotresultsv2(usesb=1)
    #cl.rungalfit_stars_24()
    #cl.reviewstars()
    #cl.plotstarprofiles()
else:
    print('no cluster name given')
    print('----------------------')
    print('loading all clusters')
    #mkw11=Cluster('MKW11')
    #coma=Cluster('Coma')
    #a2052=Cluster('A2052')
    #a2063=Cluster('A2063')
    #awm4=Cluster('AWM4')
    #ngc=Cluster('NGC6107')
    #mkw8=Cluster('MKW8')
    #a1367=Cluster('A1367')
    #herc=Cluster('Hercules')
    mkw11=simresults('MKW11')
    coma=simresults('Coma')
    a2052=simresults('A2052')
    a2063=simresults('A2063')
    awm4=simresults('AWM4')
    ngc=simresults('NGC6107')
    mkw8=simresults('MKW8')
    a1367=simresults('A1367')
    herc=simresults('Hercules')

    mylocalclusters=[mkw11,mkw8,awm4,ngc,a2052,a2063,a1367,herc,coma]
def writeresults():

    for cl in clusternames:
        print(cl)
        c=Cluster(cl)
        print(c.prefix)
        c.write_galfit_sersic_parameters_24()

def plotallresults():
    for cl in mylocalclusters:
        #print cl.prefix
        #cl.readsimresults()
        cl.plotsample()
        cl.plotresultsv2(usesb=1)
def plotsizesball(usesb=1,deltaflag=0,sbcutobs=20.5,sbcut=21.):
    i=1
    figure(figsize=(10,8))
    subplots_adjust(hspace=.02,wspace=.02)
    for cl in mylocalclusters:
        subplot(3,3,i)
        cl.plotsizesb(plotsingle=0,usesb=usesb,deltaflag=deltaflag,sbcutobs=sbcutobs,sbcut=sbcut)
        #title(cl.prefix)
        multiplotaxes(i)
        i+= 1
        
    ax=gca()
    text(-.5,-.25,'$ Observed \ SB$ ', fontsize=20,transform=ax.transAxes,horizontalalignment='center')
    text(-2.35,1.9,'$ R_e(out)/R_e(in) $ ', fontsize=20,transform=ax.transAxes,rotation=90)
    savefig(homedir+'/research/LocalClusters/SamplePlots/galfitsim_sizesball.eps')
    savefig(homedir+'/research/LocalClusters/SamplePlots/galfitsim_sizesball.png')

def plotsizeReall(usesb=1,deltaflag=0,sbcutobs=21):
    i=1
    figure(figsize=(10,8))
    subplots_adjust(hspace=.02,wspace=.02)
    for cl in mylocalclusters:
        subplot(3,3,i)
        cl.plotsizeRe(plotsingle=0,usesb=usesb,deltaflag=deltaflag,sbcutobs=sbcutobs)
        multiplotaxes(i)
        #title(cl.prefix)
        i+= 1
        
    ax=gca()
    text(-.7,-.25,'$ R_e(in) \ (arcsec)$ ', fontsize=20,transform=ax.transAxes,horizontalalignment='center')
    text(-2.3,1.9,'$ R_e(out)/R_e(in) $ ', fontsize=20,transform=ax.transAxes,rotation=90)
    savefig(homedir+'/research/LocalClusters/SamplePlots/galfitsim_sizeReall.eps')
    savefig(homedir+'/research/LocalClusters/SamplePlots/galfitsim_sizeReall.png')

'''
    def readsimresults(self,starflag=0):
        if starflag:
            output24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_stars_GalfitSersicParam_24_sim.fits'
        else:
            output24=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_GalfitSersicParam_24_sim.fits'
        self.sim=atpy.Table(output24)
        self.sb=self.sim.modelmag + 2.5*log10(pi*(self.sim.modelre**2)*self.sim.modelba)
        self.csb_obs=self.sim.cmag1 + 2.5*log10(pi*((self.sim.cre1*mipspixelscale)**2)*self.sim.caxisratio1)
        self.sb_obs=self.sim.mag1 + 2.5*log10(pi*((self.sim.re1*mipspixelscale)**2)*self.sim.axisratio1)
        self.dr=sqrt((self.sim.yc1-self.sim.modelyc)**2+(self.sim.xc1-self.sim.modelxc)**2)
        #self.cdr=sqrt((self.sim.cyc1-self.sim.modelyc)**2+(self.sim.cxc1-self.sim.modelxc)**2)
    def plotsample(self):
        figure(figsize=(10,8))
        subplots_adjust(hspace=.3)
        subplot(2,2,1)
        sp=scatter(self.sim.modelmag[self.sim.galfit_flag],self.sim.modelre[self.sim.galfit_flag],s=40,c=self.sim.modeln[self.sim.galfit_flag],vmin=1,vmax=2,alpha=0.5)
        colorbar(sp)
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by model sersic index',fontsize=12)
        subplot(2,2,2)
        scatter(self.sim.modelmag[self.sim.galfit_flag],self.sim.modelre[self.sim.galfit_flag],s=40,c=self.sim.snr24[self.sim.galfit_flag],vmin=1,vmax=10)
        plot(self.sim.modelmag[~self.sim.galfit_flag],self.sim.modelre[~self.sim.galfit_flag],'kx')
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by SNR',fontsize=12)
        subplot(2,2,3)
        colorscale=self.sb
        scatter(self.sim.modelmag,self.sim.modelre,s=40,c=colorscale,vmin=13,vmax=25)
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by model SB',fontsize=12)

        subplot(2,2,4)
        #colorscale=self.sim.f24/self.sim.isoarea
        colorscale=self.sb_obs
        scatter(self.sim.modelmag,self.sim.modelre,s=40,c=colorscale,vmin=13,vmax=25)
        xlabel('Model mag')
        ylabel('Model R_e')
        title('colored by measured SB',fontsize=12)
        savefig(homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_sample.eps')

'''
