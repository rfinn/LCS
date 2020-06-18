#!/usr/bin/env python

import os
import atpy
from pyraf import iraf
from pylab import *
from LCScommon import *
mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

# from individual bcd image
GAIN=5 # e-/SN conversion
FLUXCONV=.0447 # DN/s to BUNIT
EXPT=2.62 # expt per scan


iraf.imutil()
iraf.images()
iraf.imfilter()

class cluster:
    def __init__(self,clustername):
	self.prefix=clustername

	self.imagepath24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'
	self.sdssimagepath=homedir+'research/LocalClusters/Images/'+self.prefix+'/SDSS/'
        self.sex_image=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        self.unc_image=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_unc.fits'

        self.cov_image=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_cov.fits'
    
        #infile=homedir+'research/LocalClusters/NSAmastertables/NSAwithAGC/'+clustername+'_NSAmastertable_topcat.fits'

        #self.n=atpy.Table(infile)


    def mk24noiseimage(self):
	os.chdir(self.imagepath24)

        # remove temp images if they exist
        os.system('rm temp*.fits')

        # multiply image by exptime x gain x coverage map
        scale=FLUXCONV*GAIN*EXPT
        iraf.imarith(operand1=self.sex_image,op='*',operand2=scale,result='temp1.fits')
        iraf.imarith(operand1='temp1.fits',op='*',operand2=self.cov_image,result='temp2.fits')
        
        # smooth image using iraf.images.imfilter.gauss
        iraf.gauss(input='temp2.fits',output='temp3.fits',sigma=2,nsigma=6)

        # take sqrt
        iraf.imfunction(input='temp3.fits',output='temp4.fits',function='sqrt')
        # divide by exptime x gain x coverage map

        iraf.imarith(operand1='temp4.fits',op='/',operand2=scale,result='temp5.fits')
    
        # mutliply image and sigma image by 100
        s=self.prefix+'-scalednoise.fits'
        iraf.imarith(operand1='temp5.fits',op='*',operand2=100,result=s)
        s=self.prefix+'-scaled24.fits'
        iraf.imarith(operand1=self.sex_image,op='*',operand2=100,result=s)
        # adjust zp - make it fainter by 5 mag


mkw11=cluster('MKW11')
#for cname in clusternames:
#    cl=cluster(cname)
#    cl.mk24noiseimage()


#   1 NUMBER          Running object number
#   2 X_IMAGE         Object position along x                         [pixel]
#   3 Y_IMAGE         Object position along y                         [pixel]
#   4 XMIN_IMAGE      Minimum x-coordinate among detected pixels      [pixel]
#   5 YMIN_IMAGE      Minimum y-coordinate among detected pixels      [pixel]
#   6 XMAX_IMAGE      Maximum x-coordinate among detected pixels      [pixel]
#   7 YMAX_IMAGE      Maximum y-coordinate among detected pixels      [pixel]
#   8 ALPHA_J2000     Right ascension of barycenter (J2000)           [deg]
#   9 DELTA_J2000     Declination of barycenter (J2000)               [deg]
#  10 FLUX_ISO        Isophotal flux                                  [count]
#  11 FLUXERR_ISO     RMS error for isophotal flux                    [count]
#  12 MAG_ISO         Isophotal magnitude                             [mag]
#  13 MAGERR_ISO      RMS error for isophotal magnitude               [mag]
#  14 FLUX_ISOCOR     Corrected isophotal flux                        [count]
#  15 FLUXERR_ISOCOR  RMS error for corrected isophotal flux          [count]
#  16 MAG_ISOCOR      Corrected isophotal magnitude                   [mag]
#  17 MAGERR_ISOCOR   RMS error for corrected isophotal magnitude     [mag]
#  18 FLUX_APER       Flux vector within fixed circular aperture(s)   [count]
#  21 FLUXERR_APER    RMS error vector for aperture flux(es)          [count]
#  24 MAG_APER        Fixed aperture magnitude vector                 [mag]
#  27 MAGERR_APER     RMS error vector for fixed aperture mag.        [mag]
#  30 FLUX_AUTO       Flux within a Kron-like elliptical aperture     [count]
#  31 FLUXERR_AUTO    RMS error for AUTO flux                         [count]
#  32 MAG_AUTO        Kron-like elliptical aperture magnitude         [mag]
#  33 MAGERR_AUTO     RMS error for AUTO magnitude                    [mag]
#  34 FLUX_BEST       Best of FLUX_AUTO and FLUX_ISOCOR               [count]
#  35 FLUXERR_BEST    RMS error for BEST flux                         [count]
#  36 MAG_BEST        Best of MAG_AUTO and MAG_ISOCOR                 [mag]
#  37 MAGERR_BEST     RMS error for MAG_BEST                          [mag]
#  38 KRON_RADIUS     Kron apertures in units of A or B
#  39 PETRO_RADIUS    Petrosian apertures in units of A or B
#  40 FLUX_PETRO      Flux within a Petrosian-like elliptical apertur [count]
#  41 FLUXERR_PETRO   RMS error for PETROsian flux                    [count]
#  42 MAG_PETRO       Petrosian-like elliptical aperture magnitude    [mag]
#  43 MAGERR_PETRO    RMS error for PETROsian magnitude               [mag]
#  44 FLUX_RADIUS     Fraction-of-light radii                         [pixel]
#  47 BACKGROUND      Background at centroid position                 [count]
#  48 THRESHOLD       Detection threshold above background            [count]
#  49 MU_THRESHOLD    Detection threshold above background            [mag * arcsec**(-2)]
#  50 FLUX_MAX        Peak flux above background                      [count]
#  51 MU_MAX          Peak surface brightness above background        [mag * arcsec**(-2)]
#  52 ISOAREA_IMAGE   Isophotal area above Analysis threshold         [pixel**2]
#  53 ISOAREA_WORLD   Isophotal area above Analysis threshold         [deg**2]
#  54 A_IMAGE         Profile RMS along major axis                    [pixel]
#  55 B_IMAGE         Profile RMS along minor axis                    [pixel]
#  56 A_WORLD         Profile RMS along major axis (world units)      [deg]
#  57 B_WORLD         Profile RMS along minor axis (world units)      [deg]
#  58 THETA_IMAGE     Position angle (CCW/x)                          [deg]
#  59 ERRTHETA_IMAGE  Error ellipse position angle (CCW/x)            [deg]
#  60 THETA_WORLD     Position angle (CCW/world-x)                    [deg]
#  61 ERRTHETA_WORLD  Error ellipse pos. angle (CCW/world-x)          [deg]
#  62 THETA_J2000     Position angle (east of north) (J2000)          [deg]
#  63 ERRTHETA_J2000  J2000 error ellipse pos. angle (east of north)  [deg]
#  64 ELONGATION      A_IMAGE/B_IMAGE
#  65 ELLIPTICITY     1 - B_IMAGE/A_IMAGE
#  66 FWHM_IMAGE      FWHM assuming a gaussian core                   [pixel]
#  67 FWHM_WORLD      FWHM assuming a gaussian core                   [deg]
#  68 FLAGS           Extraction flags
#  69 CLASS_STAR      S/G classifier output
