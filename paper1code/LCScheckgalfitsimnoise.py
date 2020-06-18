#!/usr/bin/env python


"""
    PURPOSE: 
      create range of galaxies using galfit
      add galaxies to blank regions on MIPS images
      run galfit to see how we are able to recover galaxy parameters
      
    PROCEDURE
      create range of galaxies using galfit
      add galaxies to blank regions on MIPS images
      run galfit to see how we are able to recover galaxy parameters


    CALLING SEQUENCE
      cd ~/research/LocalClusters/GalfitAnalysis/MKW11/NSA
      ipython -pylab
      %run ~/svnRepository/pythonCode/LCSrungalfit.py
      mkw11=Cluster('MKW11')
      mkw11.run_just_get_imagesNSA()
      mkw11.rungalfit_first_time_nsa()
      mkw11.write_galfit_sersic_parameters_NSA()

      mkw11.rungalfit_second_time_nsa()
      mkw11.write_galfit_sersic_parameters_2comp_NSA()

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
"""

from pylab import *
import os
from scipy.interpolate import interp1d
#from scipy.optimize import leastsq
import scipy
#import urllib
import numpy as np
import glob
from pyraf import iraf
#import numpy as np
import pyfits
import ds9
import atpy

from LCSReadmasterBaseNSA import *

# from individual bcd image
GAIN=5 # e-/SN conversion
FLUXCONV=.0447 # DN/s to BUNIT
EXPT=2.62 # expt per scan


min_cutout_size=100. # in arcsec
max_cutout_radius=20. # in arcsec, cutout will be +/- 4 times this size
mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

npoints=100
minmag=11.
maxmag=15.5
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

        self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_star.fits'
        self.psf_oversampling=1

        #self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_star_byhand.fits'
        #self.psf_oversampling=1

        #self.psf_image='/Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits'
        #self.psf_oversampling=4



        #npoints=sum(self.spiralflag & self.On24ImageFlag)
        #xsim,ysim=getpositionsLCS(self.sex24.X_IMAGE,self.sex24.Y_IMAGE,self.mosaic24,self.mosaic24cov,npoints)
        #self.xsim=zeros(len(self.ra))
        #self.ysim=zeros(len(self.ra))
        #self.xsim[self.spiralflag & self.On24ImageFlag] = xsim
        #self.ysim[self.spiralflag & self.On24ImageFlag] = ysim
    def check_noise(self):
        objcut=-1

        fig = figure(figsize=(10,6))
        subplots_adjust(hspace=.3,wspace=0.2,left=0.1,right=0.95)

        ax1 = subplot2grid((2, 3), (0, 0),colspan=2,rowspan=2)
        ax2 = subplot2grid((2, 3), (0, 2))
        ax3 = subplot2grid((2, 3), (1, 2))
        #ax4 = subplot2grid((3, 3), (1, 2), rowspan=2)


        # cutouts
        fname=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/*cutout24.fits'
        uname=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'/24um/*cutout-unc-poisson24.fits'
        infiles=glob.glob(fname)
        ufiles=glob.glob(uname)
        cutim=[]
        cutnoise=[]
        for i in range(len(infiles)):


            print infiles[i]
            print ufiles[i]
            cutout=pyfits.open(infiles[i])
            cdat=cutout[0].data
            print 'size of data = ',cdat.shape
            cutout.close()


            # get x and y dimensions of image

            iraf.imgets(image=infiles[i],param='naxis1')#get RA of image
            image_xmax=float(iraf.imgets.value)
            iraf.imgets(image=infiles[i],param='naxis2')#get RA of image
            image_ymax=float(iraf.imgets.value)
            image_ymax,image_xmax=cdat.shape
            print  'cutout dimension = ',image_xmax,image_ymax

            # keep pixels within r=10 of the center (source objects only)
            keepflag=zeros((image_ymax,image_xmax),'bool')

            # I'm sure there is a better way to do this...
            for k in range(image_ymax):
                for j in range(image_xmax):
                    d =  ((k-image_ymax/2.)**2+ (j-image_xmax/2.)**2)
                    if d < 100.:
                        #print k,j,k-image_ymax/2.,j-image_xmax/2.,d
                        keepflag[k,j]=1
            

            keepflag=ones((image_ymax,image_xmax),'bool')
            cdata=cdat[keepflag].reshape(-1)
            ucutout=pyfits.open(ufiles[i])
            ucdat=ucutout[0].data
            ucutout.close()
            print 'size of data = ',cdat.shape
            udata=ucdat[keepflag].reshape(-1)
            flag=cdata > objcut
            print 'sum of flag = ',sum(flag),len(cdata)

            #print cdata
            #flag=ones(len(cdata),'bool')
            if i == 0:
                clabel='Sim'
            else:
                clabel='_nolegend_'
            try:
                ax1.plot(cdata[flag],udata[flag],'b.',label=clabel,alpha=0.1)
                #print cdata[flag]
            except:
                print 'problem with scatter plot!'
            cutim=cutim+cdata[flag].tolist()
            cutnoise=cutnoise+udata[flag].tolist()


        print 'checking noise'
        mosaicimage=pyfits.open(self.mosaic24)
        mos_data=(mosaicimage[0].data)


        mosaicimage.close()


        # keep pixels within r=10 of the center (source objects only)
        image_ymax,image_xmax=mos_data.shape
        keepflag=zeros((image_ymax,image_xmax),'bool')
        for k in range(len(self.ra)):
            if self.sex24.MATCHFLAG24[k]:
                xobj=int(round(self.sex24.X_IMAGE[k]))
                yobj=int(round(self.sex24.Y_IMAGE[k]))
                #print keepflag[yobj-10:yobj+10,xobj-10:xobj+10]
                #print keepflag[yobj-10:yobj+10,xobj-10:xobj+10].shape
                keepflag[yobj-10:yobj+10,xobj-10:xobj+10]=ones((20,20),'bool')
                

        #keepflag=ones((image_ymax,image_xmax),'bool')
        self.mpix=mos_data[keepflag].reshape(-1)

        uncimage=pyfits.open(self.mosaic24unc)
        unc_data=(uncimage[0].data)
        uncimage.close()
        self.upix=unc_data[keepflag].reshape(-1)

        covimage=pyfits.open(self.mosaic24cov)
        cov_data=(covimage[0].data)
        covimage.close()
        self.cpix=cov_data[keepflag].reshape(-1)
        keepflag=(self.cpix > 10) & (self.mpix > objcut)


        # alternate approach is to keep only pixels w/in r=10 of NSA sources

        



        #ax1.plot(self.mpix[keepflag]/FLUXCONV*EXPT,(self.upix[keepflag]/FLUXCONV)**2*sqrt(self.cpix[keepflag]),'k.',label='Real',alpha=0.1)
        ax1.plot(self.mpix[keepflag],self.upix[keepflag],'k.',label='Real',alpha=0.1)
        #hexbin(self.mpix[keepflag],self.upix[keepflag],gridsize=200,cmap=cm.Greys)
        #hexbin(self.ra[flag]-racenter,self.dec[flag]-deccenter,cmap=cm.Greys,gridsize=15,vmin=1,vmax=15,extent=(-1.2,1.2,-1.2,1.2))
        ibins=arange(.2,10,.1)
        nbins=arange(.01,2,.02)
        ax2.hist(self.upix[keepflag],bins=nbins,histtype='step',color='k',cumulative=True,normed=True)
        ax3.hist(self.mpix[keepflag],bins=ibins,histtype='step',color='k',cumulative=True,normed=True)

        #ax1.set_xscale('log')
        #ax1.set_yscale('log')
        #ax1.axis([-2,200,0,600])
        #xl=arange(100)
        #ax1.plot(xl,.25*xl+.085,'b--')
        #ax1.plot(xl,6*xl+50,'r--')
        ax1.set_xlabel('(Image Counts(BUNIT))/FLUXCONV*EXPT')
        ax1.set_ylabel('(Noise Counts(BUNIT)/FLUXCONV)^2*sqrt(COVERAGE)')
        ax1.legend(loc='upper left',numpoints=1)
        ax1.set_title(self.prefix+': Comparing real and simulated noise')
        ax2.hist(array(cutnoise,'f'),bins=nbins,histtype='step',color='b',cumulative=True,normed=True)
        ax3.hist(array(cutim,'f'),bins=ibins,histtype='step',color='b',cumulative=True,normed=True)
        ax2.axis([0.009,2.05,0,1.05])
        ax2.set_title('Noise Counts',fontsize=12)
        ax3.axis([0.2,10.5,0,1.05])
        ax3.set_title('Image Counts',fontsize=12)
        #ax2.set_xscale('log')
        #ax3.set_xscale('log')
        # print hist of noise values
        savefig(homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'simnoise.png')
    def write_galfit_sersic_parameters_24(self):

        nmax=npoints
        narray=npoints

        self.galfit_xc24=zeros((narray,2),'f')
        self.galfit_yc24=zeros((narray,2),'f')
        self.galfit_mag24=zeros((narray,2),'f')
        self.galfit_Re24=zeros((narray,2),'f')
        self.galfit_Nsersic24=zeros((narray,2),'f')
        self.galfit_axisratio24=zeros((narray,2),'f')
        self.galfit_PA24=zeros((narray,2),'f')
        numerical_error_flag24=zeros(narray,'bool')
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

    def write_galfit_sersic_parameters_stars_24(self):

        files=glob.glob('*star*24-1Comp-galfit-out.fits')
        narray=len(files)


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


        ##############################################
        # GET CUTOUT OF GALAXY AND UNCERTAINTY IMAGE
        ##############################################
        if starflag:
            dx=10
            xmin=col-dxstar
            xmax=col+dxstar
            ymin=row-dxstar
            ymax=row+dxstar
        else:
            if self.model_petro90[i] > (min_cutout_size/8.):
                PETROTH90_pixels =self.model_petro90[i]/mipspixelscale
                if self.model_petro90[i]> (max_cutout_radius):
                    PETROTH90_pixels =max_cutout_radius/mipspixelscale

            else:
                PETROTH90_pixels =(min_cutout_size/8.)/mipspixelscale

            # make a cutout of the galaxy (easier than trying to run sextractor on full mosaic
            # need x and y pixel values of the galaxy
            
            xmin=col-4*PETROTH90_pixels
            xmax=col+4*PETROTH90_pixels
            ymin=row-4*PETROTH90_pixels
            ymax=row+4*PETROTH90_pixels

        # get image dimensions of 24um mosaic
        iraf.imgets(image=self.mosaic24,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=self.mosaic24,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)


        
        # check that cutout region is not outside bounds of image

        if (xmax > image_xmax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_xmax,xmax)
            print self.prefix, i, ': setting xmax to max x of image'
            xmax=image_xmax

        if (ymax > image_ymax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_ymax,ymax)
            print self.prefix, i, ': setting ymax to max y of image'
            ymax=image_ymax

        if (xmin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f)'%(image_xmax,xmin)
            print self.prefix, i, ': setting xmin to 1'
            xmin=1

        if (ymin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f)'%(image_xmax,ymin)
            print self.prefix, i, ': setting ymin to 1'
            ymin=1
        
        if starflag:
            sex_image=self.prefix+'-star'+str(i)+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-star'+str(i)+'-'+'galfit-cutout-unc24.fits'
        else:
            sex_image=self.prefix+'-'+str(i)+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-'+str(i)+'-'+'galfit-cutout-unc24.fits'
            cov_image=self.prefix+'-'+str(i)+'-'+'galfit-cutout-cov24.fits'


        iraf.imgets(image=working_dir+sex_image,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=working_dir+sex_image,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)


        xcenter=0.5*(image_xmax)
        ycenter=0.5*(image_ymax)


        ##############################################
        # GATHER GALFIT INPUT
        ##############################################
        input_image=sex_image
        sigma_image=unc_image
        if starflag:
            mask_image=working_dir+self.prefix+'-star'+str(i)+'-'+'galfit-mask24.fits'
        else:
            mask_image=working_dir+self.prefix+'-'+str(i)+'-'+'galfit-mask24.fits'
        #psf_image='/Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits'
        #psf_image='/Users/rfinn/research/LocalClusters/Images/'+self.prefix+'/24umWCS/PRF.fits'
        #psf_oversampling=4
        #psf_image='/Users/rfinn/research/LocalClusters/Images/MKW11/24umWCS/PSFestimate_13Feb2012_sky.fits'
        #s='cp /Users/rfinn/research/LocalClusters/GalfitAnalysis/MKW11/24um/measurePSF/newpsf2.fits .'
        #os.system(s)
        #psf_image='newpsf2.fits'
        #psf_oversampling=1

        psf_image=self.psf_image
        psf_oversampling=self.psf_oversampling

        # mask_image # already defined

        xobj =xcenter
        yobj =ycenter
        #mag_total=22.5-2.5*log10(self.n.NMGY[i,4]) + self.n.EXTINCTION[i,4] #magzp-2.5*log10(self.f24NSA[i])
        if starflag:
            mag_total=self.se.MAG_BEST[i]
            Re=2.5
            axis_ratio=1
            PA=0
            rad=2.
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
        
        print xmin,xmax,xminfit,xmaxfit
        print ymin,ymax,yminfit,ymaxfit
        convolution_size=100
        if xmaxfit < 100:
            convolution_size=xmaxfit

        magzp=magzp
        pscale=mipspixelscale   # arcsec/pix
	sky=0

        if starflag:
            galname=working_dir+self.prefix+'-star'+str(i)+'-24'
        else:
            galname=working_dir+self.prefix+'-'+str(i)+'-24'


        ##############################################
        # GENERATE GALFIT MODEL IMAGE
        ##############################################
        #print galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky
        #q=raw_input('hit any key to continue')
        if starflag:
            print 'not generating model for star'
            #t=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,simflag=1,NSAsersic=1,nsersic=1,constrflag=0)
        else:
            
            t=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,simflag=1,NSAsersic=1,nsersic=self.n.SERSIC_N[i],constrflag=0,fitall=0)
            print galname+'-1Comp-galfit-out.fits',galname+'-1Comp-'+'model.fits',input_image
            outim=galname+'-1Comp-galfit-out.fits'
            sim_gal=self.prefix+'-'+str(i)+'-model.fits'
            iraf.imarith(galname+'-1Comp-galfit-out.fits[1]','+',galname+'-1Comp-galfit-out.fits[2]',sim_gal)
            
            # make noise of galaxy model
            os.system('rm temp*.fits')
            # multiply image by exptime x gain x coverage map
            scale=GAIN*EXPT/FLUXCONV
            #scale=1
            iraf.imarith(operand1=galname+'-1Comp-galfit-out.fits[2]',op='*',operand2=scale,result='temp1.fits')
            print 'cov_image = ',cov_image
            iraf.imarith(operand1='temp1.fits',op='*',operand2=cov_image,result='temp2.fits')

            # take sqrt
            iraf.imfunction(input='temp2.fits',output='temp3.fits',function='sqrt')
            # divide by exptime x gain x coverage map

            iraf.imarith(operand1='temp3.fits',op='/',operand2=scale,result='temp4.fits')
            iraf.imarith(operand1='temp4.fits',op='/',operand2=cov_image,result='temp5.fits')

            # combine poisson noise from model w/noise image
            comb_unc_image=self.prefix+'-'+str(i)+'-'+'galfit-cutout-unc-poisson24.fits'
            iraf.imarith(operand1=sigma_image,op='+',operand2='temp5.fits',result=comb_unc_image)




        ##############################################
        # RUN GALFIT  TIMES
        ##############################################


        # run galfit  times
        if starflag:
            t=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,NSAsersic=1,nsersic=1,constrflag=0)
        else:

            rmin=rad-3
            if rmin < 1:
                rmin=1
            randomr=np.random.uniform(low=rmin,high=rad+3)
            bamin=axis_ratio-.2
            if bamin < .1:
                bamin=.1
            bamax=axis_ratio+.2
            if bamax > 1:
                bamax=1
            t=rungalfit_3times(galname,sim_gal,comb_unc_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total+np.random.uniform(low=-.2,high=.2),randomr,np.random.uniform(low=bamin,high=bamax),PA+np.random.uniform(low=-20,high=20),sky,constrflag=0,fixnsersic=0,fitall=1)

            # check sersic index
            # if greater than 5, run again holding n fixed and equal to 5

            print '***********************************************'
            print 'Checking sersic index for ', galname
            output_image=galname+'-1Comp-galfit-out.fits'
            txc,tyc,tmag,tRe,Nsersic,taxis_ratio,tPA,tsky,tnumerical_error_flag24=parse_galfit_1comp(output_image+'[2]')
            print 'Nsersic = ',Nsersic
            if float(Nsersic[0]) > 5:
                print 'rerunning galfit with index held fixed at 5'
                t=rungalfit_3times(galname,sim_gal,comb_unc_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total+np.random.uniform(low=-.2,high=.2),rad+np.random.uniform(low=-3,high=3),axis_ratio+np.random.uniform(low=-.2,high=.2),PA+np.random.uniform(low=-20,high=20),sky,constrflag=0,fixnsersic=1,fixnsersicvalue=5,fitall=1)

            print '***********************************************'
            print 'Running with Fourier mode 1 turned on', galname
            output_image=galname+'-1Comp-galfit-out.fits'
            txc,tyc,tmag,tRe,Nsersic,taxis_ratio,tPA,tsky,tnumerical_error_flag24=parse_galfit_1comp(output_image+'[2]')
            t=rungalfit_3times(galname,sim_gal,comb_unc_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,txc[0],tyc[0],tmag[0],tRe[0],taxis_ratio[0],tPA[0],sky,constrflag=0,fixnsersic=0,fitall=1,asymflag=1)

            #except:
            #    print 'Error checking sersic index for ',galname
            #    t=raw_input('hit any key to continue \n')


            # if other objects nearby, fit all objects simultaneously



            # add first fourier mode to measure asymmetry


        return quitflag

    def just_get_images24(self,i, keepimages=1,make_mask_flag=0,review_mask_flag=0,starflag=0):
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
            if self.model_re[i] > (min_cutout_size/8.):
                PETROTH90_pixels =self.model_petro90[i]/mipspixelscale
                if self.model_petro90[i]> (max_cutout_radius):
                    PETROTH90_pixels =max_cutout_radius/mipspixelscale

            else:
                PETROTH90_pixels =(min_cutout_size/8.)/mipspixelscale

            # make a cutout of the galaxy (easier than trying to run sextractor on full mosaic
            # need x and y pixel values of the galaxy
            xmin=col-4*PETROTH90_pixels
            xmax=col+4*PETROTH90_pixels
            ymin=row-4*PETROTH90_pixels
            ymax=row+4*PETROTH90_pixels

        # get image dimensions of 24um mosaic

        iraf.imgets(image=self.mosaic24,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=self.mosaic24,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)
        
        # check that cutout region is not outside bounds of image

        if (xmax > image_xmax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_xmax,xmax)
            print self.prefix, i, ': setting xmax to max x of image'
            xmax=image_xmax

        if (ymax > image_ymax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_ymax,ymax)
            print self.prefix, i, ': setting ymax to max y of image'
            ymax=image_ymax

        if (xmin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f)'%(image_xmax,xmin)
            print self.prefix, i, ': setting xmin to 1'
            xmin=1

        if (ymin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f)'%(image_xmax,ymin)
            print self.prefix, i, ': setting ymin to 1'
            ymin=1
        
        if starflag:
            sex_image=self.prefix+'-star'+str(i)+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-star'+str(i)+'-'+'galfit-cutout-unc24.fits'
            cov_image=self.prefix+'-star'+str(i)+'-'+'galfit-cutout-cov24.fits'
        else:
            sex_image=self.prefix+'-'+str(i)+'-'+'galfit-cutout24.fits'
            unc_image=self.prefix+'-'+str(i)+'-'+'galfit-cutout-unc24.fits'
            cov_image=self.prefix+'-'+str(i)+'-'+'galfit-cutout-cov24.fits'

        try:
            s=self.mosaic24+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+sex_image)
            s=self.mosaic24unc+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+unc_image)
            s=self.mosaic24cov+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+cov_image)

        except:
            print 'Warning:  Problem creating cutout image, probably b/c it already exists'
            print '   going to delete existing cutout image and try imcopy again'
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
        dist=sqrt((ycenter-ysex)**2+(xcenter-xsex)**2)

        #   find object ID
        objIndex=where(dist == min(dist))
        objNumber=sexnumber[objIndex]
        objNumber=objNumber[0] # not sure why above line returns a list
        print 'object number = ',objNumber
        if make_mask_flag:
            if starflag:
                mask_image=working_dir+self.prefix+'-star'+str(i)+'-'+'galfit-mask24.fits'
            else:
                mask_image=working_dir+self.prefix+'-'+str(i)+'-'+'galfit-mask24.fits'
            if os.path.exists(mask_image):
                iraf.imdel(mask_image)
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
                print "couldn't access: ",s
            s='file new '+sex_image
            d.set(s)
            d.set('zoom to fit')
            s='file new '+mask_image
            d.set(s)
            d.set('zoom to fit')
            flag=raw_input('edit the mask? \n')
            if flag.find('n') > -1:
                review_mask_flag = 0
            elif flag.find('y') > -1:
                editflag=int(raw_input('enter 0 to subtract an object, 1 to add an object'))
                if editflag == 0:
                    objid=float(raw_input('enter object id to remove from mask'))
                    iraf.imreplace(mask_image,value=0,lower=objid-.5,upper=objid+.5)
                elif editflag == 1:
                    print 'entering imedit'
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
            print i,files[i]
            inimage=files[i]
            if starflag:
                print 'hi'
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
            print i, inimage,image_xmax,image_ymax,xc,yc
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
            print inimage,'output_coords',datfile,skyfile
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
            self.runSE()
        print 'reading SE catalog'
        self.readSE()
        #self.model_mag=np.random.uniform(low=minmag,high=maxmag,size=len(self.ra))
        #self.model_re=np.random.uniform(low=2,high=25,size=len(self.ra))

        print 'getting positions for artificial sources'
        self.xsim,self.ysim=getpositionsLCS(self.se.X_IMAGE,self.se.Y_IMAGE,self.mosaic24,self.mosaic24cov,npoints)
        figure()
        plot(self.xsim,self.ysim,'bo')
        scatter(self.se.X_IMAGE,self.se.Y_IMAGE,marker='o',s=10,color='0.5',alpha=0.5)
        savefig(homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'simgalpositions.eps')
        #simpos=atpy.Table()
        #simpos.add_column('X',self.xsim)
        #simpos.add_column('Y',self.ysim)
        #s=homedir+'research/LocalClusters/GalfitAnalysis/sim/'+self.prefix+'_simgal.fits'
        #simpos.write(s)
        print 'setting up model parameters'
        self.model_mag=np.random.uniform(low=minmag,high=maxmag,size=npoints)

        # select Re, B/A and PA from the parent sample of all spirals
        spiral_index=where(self.spiralflag)
        spiral_index=spiral_index[0]
        modindex=np.random.randint(low=0,high=len(spiral_index)-1,size=npoints)
        self.model_re=self.n.SERSIC_TH50[spiral_index[modindex]]
        self.model_petro90=self.n.PETROTH90[spiral_index[modindex]]
        self.model_PA=self.n.SERSIC_PHI[spiral_index[modindex]]
        self.model_BA=self.n.SERSIC_BA[spiral_index[modindex]]
        self.model_n=self.n.SERSIC_N[spiral_index[modindex]]
        #self.model_mag=22.5-2.5*log10(self.n.NMGY[modindex,4]) + self.n.EXTINCTION[modindex,4] - 2. # mag 2 mags brighter

        print 'starting to run galfit'
        for i in range(startindex,npoints):
            quitflag=self.just_get_images24(i,make_mask_flag=1)
            if quitflag:
                return
            quitflag=self.get_images24(i,0)
            if quitflag:
                return
        self.measuresnr24()
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
                t=raw_input('press any key to continue, x to quit \n')
                if t.find('x') > -1:
                    return
        self.measuresnr24(starflag=1)
        self.write_galfit_sersic_parameters_stars_24()
    def review(self,usesnrflag=0):
        self.readsimresults()
        self.plotresults(usesnr=usesnrflag)
    def reviewstars(self):
        self.readsimresults(starflag=1)
        self.plotresults(starflag=1)

    def reviewimages(self):
        self.readsimresults()
        d=ds9.ds9()
        for i in range(len(self.sim.modelre)):
            if abs(self.sim.modeln[i] - self.sim.nsersic1[i]) > 2:
                print 'galaxy number ',i
                print '-------------------------'
                print 'var     mod      fit'
                print '-------------------------'
                print 'n   = %5.1f, %5.1f+/-%3.2f'%(self.sim.modeln[i],self.sim.nsersic1[i],self.sim.nsersic1err[i])
                print 'Re  = %5.1f, %5.1f+/-%3.2f'%(self.sim.modelre[i]/mipspixelscale,self.sim.re1[i],self.sim.re1err[i])
                print 'mag = %5.1f, %5.1f+/-%3.2f'%(self.sim.modelmag[i],self.sim.mag1[i],self.sim.mag1err[i])
                print 'B/A = %5.1f, %5.1f+/-%3.2f'%(self.sim.modelba[i],self.sim.axisratio1[i],self.sim.axisratio1err[i])
                print 'phi = %5.1f, %5.1f+/-%3.2f'%(self.sim.modelpa[i],self.sim.pa1[i],self.sim.pa1err[i])
                print 'snr = %5.1f, %5.1f+/-%3.2f'%(self.sim.snr24[i],self.sim.f24[i],self.sim.f24err[i])
                print 'nerr = ',self.sim.numerical_error_flag24[i]

                
                d.set('frame delete all')
                s='file new '+self.prefix+'-'+str(i)+'-24-1Comp-galfit-out.fits[1]'
                d.set(s)
                s='file new '+self.prefix+'-'+str(i)+'-galfit-cutout-unc-poisson24.fits'
                d.set(s)
                s='file new '+self.prefix+'-'+str(i)+'-galfit-mask24.fits'
                d.set(s)
                s='file new '+self.prefix+'-'+str(i)+'-24-1Comp-galfit-out.fits[2]'
                d.set(s)
                s='file new '+self.prefix+'-'+str(i)+'-24-1Comp-galfit-out.fits[3]'
                d.set(s)
                try:
                    s='file new '+self.prefix+'-'+str(i)+'-24-1Comp-subcomps.fits[3]'
                    d.set(s)
                except:
                    print "WARNING:  couldn't load ",self.prefix+'-'+str(i)+'-24-1Comp-subcomps.fits[3]'
                d.set('tile')
                for k in range(1,7):
                    s='frame '+str(k)
                    d.set(s)
                    d.set('scale log')
                    d.set('zoom to fit')
                string=raw_input('hit any key when ready to continue (q to quit) \n')
                if string.find('q') > -1:
                    quitflag=1
                    break


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
    outfile.write('O) both                # Display type (regular, curses, both)\n')
    outfile.write('P) 0                   # Create output image only? (1=yes; 0=optimize) \n')
    outfile.write('S) 0                   # Modify/create objects interactively?\n')


def write_galfit_sersic(outfile,objnumber,profile,xobj,yobj,mag,rad,sersic_exp,axis_ratio,pa,fixsersic=0,asymmetry=0):
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
    outfile.write(' 9) %5.2f       1       # axis ratio (b/a)    \n'%(axis_ratio))
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
            values=(5,0.)# fit and error
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


def rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,convolutionflag=1,simflag=0,NSAsersic=0,nsersic=2,constrflag=1,fixnsersic=0,fixnsersicvalue=5,fitall=0,asymflag=0):
    
    # loop through galfit 3 times...
    #    first time fits on sersic profile
    #    - display results (like ds9 multiextension data cube -> use xpa)
    #
    #    second time fits 2 sersic profiles
    #    - use output from first pass (mag_0 and Re_0)
    #    - put half luminosity (output mag_0 - 0.75) in output from first fit
    #    - add second sersic profile with half luminosity (output mag_0 - 0.75)
    #    - make size of second component 1/2 - 3/4 of size of first (Re = 0.5 Re)
    #    - display results in ds9
    # 
    #    third time - fit 3 sersic profiles
    #    - try using mean of 2-component fit for the 3rd component
    #      * split the flux 3 ways
    #    - if 2 components have similar values,
    #      * use 0.5-0.75 of the other components
    #    - if n > 4 for subcomponents, set it back to n=1-2 at
    #    - if size is tool small (<1 pixel), reset to a bigger value
    #

            # run sextractor to detect sources (different detection threshold than mask)

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
        # create galfit input file
        galfile=galname+'galfit.input.'+str(j+1)+'Comp'
        galfit_input=open(galfile,'w')
        
        write_galfit_image_param(galfit_input,input_image,output_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,convflag=convolutionflag,constraintflag=constrflag,fitallflag=fitall)
        
        if j == 0:
            # gather object parameters from sextractor output
            
            # loop through galfit 3 times...
            #    first time fits on sersic profile
            
            objnumber=1
            write_galfit_sersic(galfit_input,objnumber,profile,xobj,yobj,mag_total,rad,sersic_exp,axis_ratio,PA,fixsersic=fixnsersic,asymmetry=asymflag)
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
            print 'wrote input file for 2comp fit'
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
            print 'HHHHHEEEEEYYYYY'
            print mag, mag_fit1[0],mag_fit2[0]
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
                print 'found ',len(se.X_IMAGE),' sources on the field of ',galname
                nearbyobjflag=sqrt((se.X_IMAGE-xobj)**2+(se.Y_IMAGE-yobj)**2) > 10.
                for k in range(len(se.X_IMAGE)):
                    if nearbyobjflag[k]:
                        objnumer=objnumber+1
                        write_galfit_sersic(galfit_input,objnumber,profile,se.X_IMAGE[k],se.Y_IMAGE[k],se.MAG_BEST[k],se.FLUX_RADIUS[k,0],2,se.B_IMAGE[k]/se.A_IMAGE[k],se.THETA_IMAGE[k])
            except AttributeError:
                print 'WARNING: no sources detected in image!'
        galfit_input.close()
        if simflag:
            s = 'galfit -o2 '+galfile
            os.system(s)
            return
            # run galfit
        s = 'galfit '+galfile
        print 'run the following: ',s
        try:
            errno=os.system(s)
            print 'errno = ',errno
            if errno == 25600:
                print 'Error running galfit ', errno
                galflag[j]=0
            else:
                galflag[j]=1
        except:
            print 'problem running galfit for ',j+1,' comp fit for ',image_id
            galflag[j]=0
            if interactive:
                flag=raw_input('press any key to continue, q to quit; s to stop running galfit on this galaxy')
                if flag.find('q') > -1:
                    quitflag=1
                elif string.find('s') > -1:
                    stop_gal_flag=1
                    flag=raw_input('is galaxy too faint? (y or n)')
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
            print "appears like galfit did not complete"
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
            print 'file to display = ',output_image
            s='file new multiframe '+output_image
            print s
            d.set(s)
            d.set('frame delete 1')
            for k in range(2,5):
                s='frame '+str(k)
                d.set(s)
                #d.set('scale linear')
                d.set('zoom to fit')
                print k
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
                print s
                #d.set(s)
            if j == 1:
                galfit_outimage=galname+'-'+'2Comp-galfit-out.fits[2]'
                x_fit1,y_fit1,mag_fit1,Re_fit1,Nsersic_fit1,axis_ratio_fit1,pa_fit1,x_fit2,y_fit2,mag_fit2,Re_fit2,Nsersic_fit2,axis_ratio_fit2,pa_fit2,sky_fit=parse_galfit_2comp(galfit_outimage)
                #d.set('frame 7')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit1[0],Re_fit1[0],Nsersic_fit1[0],axis_ratio_fit1[0],pa_fit1[0])
                print s
                #d.set(s)
                #d.set('frame 8')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit2[0],Re_fit2[0],Nsersic_fit2[0],axis_ratio_fit2[0],pa_fit2[0])
                #d.set(s)
            #raw_input('hit any key when ready to view subcomponent images \n')
            print 'file to display = ',subcomp_image
            s='file new multiframe '+subcomp_image
            d.set(s)

            if j == 0:
                endframe=8
            if j == 1:
                endframe=9
            if j == 2:
                endframe=10
            s='frame delete '+str(endframe)
            d.set(s)
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
            d.set('saveimage png image.png')
            img_name=image_id+str(j+1)+'Comp.png'
            os.rename('image.png',img_name)

            if interactive:
                string=raw_input('hit any key when ready to continue (q to quit; s to stop running galfit for this galaxy) \n')
                if string.find('q') > -1:
                    quitflag=1
                    break
                elif string.find('s') > -1:
                    print 'gal and quitflags = ',galflag[0],galflag[1],galflag[2],quitflag
                    stop_gal_flag=1
                    flag=raw_input('is galaxy too faint? (y or n)')
                    if flag.find('y') > -1:
                        too_faint_flag=1
                    return galflag[0],galflag[1],galflag[2],quitflag,stop_gal_flag,too_faint_flag
        else:
            print "not diplaying images b/c galfit didn't finish"
            print "not going to try to fit anything else w/this galaxy (if there is anymore...)"
            return galflag[0],galflag[1],galflag[2],quitflag
    return galflag[0],galflag[1],galflag[2],quitflag,stop_gal_flag,too_faint_flag

mkw11=Cluster('MKW11')
#coma=Cluster('Coma')
#a2052=Cluster('A2052')
#a2063=Cluster('A2063')
#awm4=Cluster('AWM4')
#ngc=Cluster('NGC6107')
#mkw8=Cluster('MKW8')
#a1367=Cluster('A1367')
#herc=Cluster('Hercules')

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
        print "Didn't recognize ntimes"
        return


