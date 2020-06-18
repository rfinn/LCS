#!/usr/bin/env python


"""
    PURPOSE: 
      run galfit on sdss and mips images of LCS cluster members and save results in a fits table
      
    PROCEDURE

      get sdss image of each galaxy
        reconstruct psf or use other available psf image
      run galfit on these images
        run both 1 and 2 comp fits
      for each fit, identify the disk component
      measure R50 and R90 for the disk model
      save results in a fits table
      repeat for 24um image

      program can run on sdss images from archive or NASA-Sloan atlas

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

      to run on 24um images with PA and BA held fixed at SDSS values: (2015/02/03)

      mkw11.run_just_get_images24(make_mask=1,review_mask=1) # if you haven't made cutouts yet
      mkw11.rungalfit_first_time_24(fitBAflag=False,fitPAflag=False)
      mkw11.write_galfit_sersic_parameters_24()

      to run on 24um images with PA and BA held fixed at SDSS values, and no convolution: (2015/08/30)

      mkw11.rungalfit_first_time_24(fitBAflag=False,fitPAflag=False,convflag=False)
      mkw11.write_galfit_sersic_parameters_24()


      to run galfit on one 24um image
      %run ~/svnRepository/pythonCode/LCSrungalfit.py
      mkw11=Cluster('MKW11')
      set_ntimes(1)
      ngc.just_get_images24(ngc.nsadict[43784],make_mask_flag=1, review_mask_flag=1)
      mkw11.get_images24(mkw11.nsadict[NSAID],0)

      mkw11.keep_BA_fixed(mkw11.nsadict[NSAID],0)

      THIS IS OLD BUT I'M KEEPING FOR COMPLETENESS
      
      To run galfit on the sdss das images:

      mkw11.just_get_images()
      mkw11.rungalfit_first_time_sdss()
      mkw11.write_galfit_sersic_parameters_sdss()

    REQUIRED PYTHON MODULES
        scipy
        pylab
        atpy
        
    ADDITIONAL REQUIRED MODULES
        astrofuncs.py

    NOTES
      updating on 07-Jan-2013 to use NSA mastertables

      2013-03-04 - implementing 2-comp fit for MKW11 r-band images

      2015-02-03 - updating to run galfit on 24um images holding PA and B/A
      fixed at NSA values
"""

print 'got here'
from pylab import *
import os
from scipy.interpolate import interp1d
import urllib
import numpy as np

from pyraf import iraf
import pyfits
import ds9
import atpy
from astropy.io import fits
from LCSReadmasterBaseNSA import *
from astropy.wcs import WCS
from wcsaxes import WCSAxes
import argparse


parser = argparse.ArgumentParser(description ='Run galfit')
#parser.add_argument('--s', dest = 's', default = False, action = 'store_true', help = 'Run sextractor to create obje#ct catalogs')
#parser.add_argument('--c', dest = 'c', default = False, action = 'store_true', help = 'Run scamp')
parser.add_argument('--w', dest = 'web', default = False, action = 'store_true', help = 'Display NSA cutouts on web browser')
#parser.add_argument('--l', dest = 'l', default = False, help = 'List of images to input to swarp')
args = parser.parse_args()
if args.web:
    import webbrowser
    new=2

#iraf.stsdas()
#iraf.analysis()
#iraf.isophote()
#iraf.tables()
#iraf.ttools()
#iraf.digiphot()
#iraf.apphot()
#iraf.daophot()
#iraf.imfilter()

#min_cutout_size=80. # in arcsec
min_cutout_radius=15. # in arcsec
max_cutout_radius=30. # in arcsec

mult_petro90=3 # multiple of PETRO90 to use radius of galaxy cutout

mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'


def toggle(var):
    try:
        if var == 0:
            var = 1
        elif var == 1:
            var = 0
        return var
    except:
        print 'problem with toggle.  Variable should be 0 or 1, but var = ',var

def normalize_mask(mask_image):
    os.rename(mask_image,'temp.fits')
    fdulist = fits.open('temp.fits',mode='update')
    t=fdulist[0].data
    # replace object ID values with zero
    #f = (t == objNumber)
    #replace_values = np.zeros(t.shape)
    #t=t*(~f) + replace_values*f
    # set all bad values to 1
    t[t> .1]=t[t > .1]/t[t > .1]
    fdulist.flush()
    # write out mask image
    if os.path.isfile(mask_image):
        os.remove(mask_image)
    hdu=fits.PrimaryHDU(t)
    hdu.writeto(mask_image)

    fdulist.close()
    os.remove('temp.fits')

def runimedit(mfile,nframe=1):
    continueWithProgram=1
    continueWithObject=1
    repeatflag=1
    while (repeatflag > 0.1):

        #iraf.display(mfile,frame=nframe, fill='yes')

        print mfile
        print 'Running imedit to mask out other sources in the field:'
        print 'Enter b to mask out a circular region'
        print 'Enter a to mark the corners of a rectangle'
        print 'Enter q when finished'
        print 'running imedit ',mfile
        outfile='tmp.fits'
        if os.path.exists(outfile):
            os.remove(outfile)
        
        iraf.imedit(mfile,output=outfile)

        flag=str(raw_input('Are you happy with the editing?  n=no x=quit y (or any other key) = yes '))
        flag=str(flag)
        print 'this is what I think you typed ',flag
        if flag.find('n') > -1:
            flag2=str(raw_input('What is wrong?  r=redo masking, o=nearby object, p=partial image, x=quit '))
            if flag2.find('r') > -1:
                s='rm '+outfile
                os.system(s)
                repeatflag=1
                print 'i think repeatflag = 1 ', repeatflag
            elif flag2.find('o') > -1:
                s='rm '+outfile
                os.system(s)

                s='mv '+mfile+' NearbyObjects/'
                os.system(s)
                continueWithObject=0
                return continueWithProgram,continueWithObject
            elif flag2.find('p') > -1:
                s='rm '+outfile
                os.system(s)

                s='mv '+mfile+' PartialImages/'
                os.system(s)
                continueWithObject=0
                return continueWithProgram,continueWithObject
            elif flag2.find('x') > -1:
                continueWithProgram=0
                repeatflag=0
                print 'i think repeatflag = 0', repeatflag
                return continueWithProgram,continueWithObject
            else: 
                repeatflag=0

        elif flag.find('x') > -1:
            print 'i think you want to exit'
            continueWithProgram=0
            repeatflag=0
            return continueWithProgram,continueWithObject
        else:
            repeatflag=0
        os.system('mv '+mfile+' '+outfile)
    return continueWithProgram,continueWithObject


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

        self.read_sdss_csv()
        self.read_snr24_NSA()

        self.mipsprf=atpy.Table(homedir+'research/LocalClusters/GalfitAnalysis/MIPSpsf/mips24_prf.dat',type='ascii')


        #self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'



        #self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-scaled24.fits'
        #self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-scalednoise.fits'
        #self.zp24offset=5.

        self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        #self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_std.fits'
        #self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_unc.fits'
        # the image below seems to produce bad fits.  going to retry w/_std.fits
        self.mosaic24unc=homedir+'research/LocalClusters/MIPS/rawdata/'+self.prefix+'/FullMosaic/mosaic_noise.fits'
        #self.zp24offset=0.
        #self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/PRF.fits'
        #self.psf_oversampling=4


        self.mosaic24cov=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_cov.fits'
        self.zp24offset=0.

        #self.psf_image=homedir+'research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_star.fits'
        #self.psf_oversampling=1

        #self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_star_byhand.fits'
        # copied above file to the following to see if filename was too long
        self.psf_image='/Users/rfinn/research/LocalClusters/PRF/'+self.prefix+'/'+self.prefix+'psf_starv2.fits'
        self.psf_oversampling=1
        if (self.prefix.find('Herc') > -1):# | (self.prefix.find('A2052') > -1):
            self.psf_image='/Users/rfinn/research/LocalClusters/PRF/mips24_prf_mosaic_2.45_4x.fits'
            self.psf_oversampling=4

        self.ur=self.n.ABSMAG[:,2]-self.n.ABSMAG[:,4]
        self.redflag=(self.ur > 2.3)
        self.greenflag=(self.ur > 1.8) & (self.ur < 2.3)
        self.blueflag=(self.ur<1.8)
        self.NUVr=self.n.ABSMAG[:,1] - self.n.ABSMAG[:,4]
        self.blueflag2=self.NUVr < 4.1

        # add galaxies with blue u-r colors but no galex data
        self.blue_nogalex = (self.n.NMGY[:,1] == 0.) & (self.blueflag)
        self.blueflag2[self.blue_nogalex] = np.ones(sum(self.blue_nogalex))



        # setting self.snrse < snr24cut to run galfit on all spirals
        #self.analyze_mips=self.On24ImageFlag & (self.snrse < snr24cut) & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag
        self.analyze_mips=self.On24ImageFlag & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag & (self.snrse > 1)
        self.analyze_mips=self.On24ImageFlag & (self.n.SERSIC_TH50 > mipspixelscale) &  (self.n.SERSIC_TH50 < 20.) & (self.snrse > 2) & (self.blueflag2) & ~self.agnflag
        for id in coma_badobjects:
            try:
                self.analyze_mips[self.nsadict[id]]=0
            except KeyError:
                print 'could not find ',id,' in coma'
        #& (self.sex24.MATCHFLAG24)
        #self.psf_image='/Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits'
        #self.psf_oversampling=4

    def read_sdss_csv(self):
        infile=homedir+'research/LocalClusters/NSAmastertables/SDSSTables/'+self.prefix+'_SDSS_dr7.csv'
        scat=atpy.Table(infile,type='ascii',data_start=1)
        self.sdss_run=scat.col3
        self.sdss_rerun=scat.col4
        self.sdss_camcol=scat.col5
        self.sdss_field=scat.col6
        self.sdss_rowc=scat.col17
        self.sdss_colc=scat.col15
        self.sdss_r = scat.col10
    def read_snr24_NSA(self):
        infile=homedir+'research/LocalClusters/NSAmastertables/SNR24/'+self.prefix+'_snr24NSA.dat'
        snrdat=atpy.Table(infile,type='ascii')
        self.f24NSA=snrdat['col1']
        self.f24NSAerr=snrdat['col2']
        self.snr24=snrdat['col3']




    def run_one(self,i, makemask=True,set_size=False,new_size=None):
        if makemask:
            self.just_get_images24(i,make_mask_flag=1,review_mask_flag=1,set_size=set_size)
        set_ntimes(1)
        self.keep_BA_fixed(i,0)
        #self.get_images24(i,0)

    def rungalfit(self,getimages=1,sdssflag=1,mipsflag=1,continue_flag=0,use_nsa=1,start_index=0,fitBA=1,fitPA=1,convflag=1):
        print '#################'
        print 'just inside rungalfit'
        print 'conv, fitBA, fitPA flag = ',convflag, fitBA, fitPA
        # wrapper function to run galfit analysis on the sdss and mips images
        #
        # get images = 1 to get SDSS image or create 24um cutout
        # get images = 0 - can use this if all the images already exist
        #
        # sdssflag = 1 to run galfit analysis on the sdss images
        #
        # mipsflag = 1 to run galfit analysis on the 24um images
    
        # wrapper for get_images
        # calls get_images for a series of galaxies
        # for now, just use images that are on 24 micron image
        #start_index=0
        index=arange(start_index,len(self.ra))
        on24index=index[self.On24ImageFlag[index]]
        #i=self.agcdict[230351]
        #i=self.agcdict[230369]
        #i=self.agcdict[-99991]
        if use_nsa:
            results_file=self.galfit_dir+self.prefix+'-galfitResults-NSA.dat'
        else:
            results_file=self.galfit_dir+self.prefix+'-galfitResults-sdss.dat'
        

        results_file24=self.galfit_dir+self.prefix+'-galfitResults-24.dat'
        print continue_flag
        if continue_flag:
            print 'HEY, continue flag = 1!'
        if continue_flag:
            if sdssflag:
                print 'got in continue_flag, sdssflag loop'
                if os.path.exists(results_file):
                    print 'found file w/previous results'
                    infile=open(results_file,'r')
                    i=0
                    for line in infile:
                        t=line.split()
                        index,On24ImageFlag,self.galflag[i][0],self.galflag[i][1],self.galflag[i][2],self.galflag_stop[i],self.galflag_too_faint[i],self.galfit_sdssR50[i][0],self.galfit_sdssR50[i][1],self.galfit_sdssR50[i][2],self.galfit_sdssR90[i][0],self.galfit_sdssR90[i][1],self.galfit_sdssR90[i][2]=t
                        i += 1
                    infile.close()
                    start_index=i
                    
            if mipsflag:

                if os.path.exists(results_file24):
                    infile=open(results_file24,'r')
                    i=0
                    for line in infile:
                        t=line.split()
                        index,On24ImageFlag,self.galflag24[i][0],self.galflag24[i][1],self.galflag24[i][2],self.galflag_stop24[i],self.galflag_too_faint24[i],self.galfit_mipsR50[i][0],self.galfit_mipsR50[i][1],self.galfit_mipsR50[i][2],self.galfit_mipsR90[i][0],self.galfit_mipsR90[i][1],self.galfit_mipsR90[i][2]=t
                        i += 1
                    infile.close()

                    if sdssflag:
                        if i < start_index:
                            start_index=i
                    else:
                        start_index=i
        else:
            if sdssflag:
                try:
                    os.remove(results_file)
                except OSError:
                    print 'looks like this is the first time running sdss for ',self.prefix

            if mipsflag:
                try:
                    os.remove(results_file24)
                except OSError:
                    print 'looks like this is the first time running mips for ',self.prefix
        print 'here is what I think the start_index is: ',start_index
        ngal=0
        for i in range(start_index,len(self.ra)):
            #if (self.On24ImageFlag[i] & self.spiralFlag[i]):
            # running for all galaxies, so removing spiral flag
            if (self.On24ImageFlag[i]):
                #if ngal < start_index:
                #    ngal += 1
                #    continue
                if sdssflag:
                    print 'running galfit analysis for SDSS image of ',i,self.prefix,'-',self.n.NSAID[i]
                    if use_nsa:
                        quitflag=self.get_imagesNSA(i,getimages)
                    else:
                        quitflag=self.get_images(i,getimages)
                    if quitflag:
                        self.write_galflags_SDSS()
                        return
                    if self.galflag_stop[i] < .1:
                        
                        print 'running display results for SDSS image of ',i,self.prefix,'-',self.n.NSAID[i]
                        quitflag,self.galflag_stop[i]=self.display_galfit_results(i,band=0,nsaflag=use_nsa)
                        if quitflag:
                            self.write_galflags_SDSS()
                            return
                        print 'running call_measure_disk for SDSS image of ',i,self.prefix,'-',self.n.NSAID[i]
                        if self.galflag_stop[i] < .1:
                            self.call_measure_disk(i,wave_band=0,nsaflag=use_nsa)
                            self.call_measure_radius(i,wave_band=0,nsaflag=use_nsa)
                    # add line to output file
                    outfile=open(results_file,'a')
                    output_string=' %i %i %i %i %i %i %i %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f \n'%(i,self.On24ImageFlag[i],self.galflag[i][0],self.galflag[i][1],self.galflag[i][2],self.galflag_stop[i],self.galflag_too_faint[i],self.galfit_sdssR50[i][0],self.galfit_sdssR50[i][1],self.galfit_sdssR50[i][2],self.galfit_sdssR90[i][0],self.galfit_sdssR90[i][1],self.galfit_sdssR90[i][2])
                    outfile.write(output_string)
                    outfile.close()
                if mipsflag & (self.analyze_mips[i]):# & ~self.ellipticalflag[i]:
                    print '*******************'
                    print '*******************'
                    print '*******************'
                    print ''
                    print 'running galfit analysis for MIPS image of ',self.prefix,'-',self.n.NSAID[i]
                    print 'snr24 = ',self.snr24[i]
                    if fitBA:
                        quitflag=self.get_images24(i,getimages)
                    else:
                        # check if output image already exists
                        if convflag:
                            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'
                        else:
                            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-fixedBA-noconv-1Comp-galfit-out.fits'
                        if os.path.exists(output_image24):
                            print '\n output image already exists \n'
                            print output_image24
                            flag=str(raw_input('Run galfit again?  y (or any key)=yes n=no x=quit \n'))
                            if flag.find('n') > -1:
                                print 'I think you said no'
                                continue
                            elif flag.find('x') > -1:
                                print 'I think you want to quit'
                                return
                            print 'running galfit again - hope that is what you want!'
                        print output_image24
                        print 'convflag, fitBA, fitPA = ',convflag, fitBA, fitPA
                        quitflag=self.keep_BA_fixed(i,convflag=convflag)
                    if quitflag:
                        self.write_galflags_24()
                        return
                    if self.galflag_stop24[i] < .1:
                        quitflag,self.galflag_stop24[i]=self.display_galfit_results(i,band=1)
                        if quitflag:
                            self.write_galflags_24()
                            return
                        #if self.galflag_stop24[i] < .1:
                        #    self.call_measure_disk(i,wave_band=1,nsaflag=0)

                        #    try:
                        #        self.call_measure_radius(i,wave_band=1,nsaflag=0)
                        #    except:
                        #        print 'error running call_measure_radius()'

                    # add line to output file
                    outfile=open(results_file24,'a')
                    output_string=' %i %i %i %i %i %i %i %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f \n'%(i,self.On24ImageFlag[i],self.galflag24[i][0],self.galflag24[i][1],self.galflag24[i][2],self.galflag_stop24[i],self.galflag_too_faint24[i],self.galfit_mipsR50[i][0],self.galfit_mipsR50[i][1],self.galfit_mipsR50[i][2],self.galfit_mipsR90[i][0],self.galfit_mipsR90[i][1],self.galfit_mipsR90[i][2])
                    outfile.write(output_string)
                    outfile.close()
            
        if sdssflag:
            self.write_galflags_SDSS()
        if mipsflag:
            self.write_galflags_24()

            
    def write_galflags_SDSS(self):
        outfile=self.prefix+'-galfitSDSSFlags.dat'
        out1=open(outfile,'w')
        for i in range(len(self.ra)):
            out1.write('%i %i %i %i %i %i\n'%(i,self.On24ImageFlag[i],self.galflag[i,0],self.galflag[i,1],self.galflag[i,2],self.galflag_too_faint[i]))
        out1.close()
    def write_galflags_24(self):
        outfile=self.prefix+'-galfit24Flags.dat'
        out1=open(outfile,'w')
        for i in range(len(self.ra)):
            out1.write('%i %i %i %i %i %i\n'%(i,self.On24ImageFlag[i],self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],self.galflag_too_faint24[i]))
        out1.close()
    def read_galflags_sdss(self):
        infile=self.prefix+'-galfitSDSSFlags.dat'
        dat=atpy.Table(infile,type='ascii')
        self.galflag[:,0]=dat['col3']
        self.galflag[:,1]=dat['col4']
        self.galflag[:,2]=dat['col5']
        self.galflag_too_faint=dat['col6']

    def read_galflags_24(self):
        infile=self.prefix+'-galfit24Flags.dat'
        dat=atpy.Table(infile,type='ascii')
        self.galflag24[:,0]=dat['col3']
        self.galflag24[:,1]=dat['col4']
        self.galflag24[:,2]=dat['col5']
        self.galflag_too_faint24=dat['col6']



    def write_galfit_sersic_parameters_sdss(self):
        self.galfit_xc=zeros((len(self.ra),2),'f')
        self.galfit_yc=zeros((len(self.ra),2),'f')
        self.galfit_mag=zeros((len(self.ra),2),'f')
        self.galfit_Re=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic=zeros((len(self.ra),2),'f')
        self.galfit_axisratio=zeros((len(self.ra),2),'f')
        self.galfit_PA=zeros((len(self.ra),2),'f')
        numerical_error_flag=zeros(len(self.ra),'bool')
        self.galfit_R50=zeros((len(self.ra)),'f')
        self.galfit_R90=zeros((len(self.ra)),'f')
        self.R50_image=zeros((len(self.ra)),'f')
        self.R90_image=zeros((len(self.ra)),'f')

        for i in range(len(self.ra)):
            if self.On24ImageFlag[i]:
                output_image=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'+self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits'
                model1_image=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'+self.prefix+'-'+str(self.n.NSAID[i])+'-diskimage-1Comp.fits'
                imagesdss=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout.fits'


                try:
                    iraf.imgets(image=model1_image,param='LCS_R50')
                    self.galfit_R50[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50 from ',imagesdss
                try:
                    iraf.imgets(image=model1_image,param='LCS_R90')
                    self.galfit_R90[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90 from ',imagesdss
                try:
                    iraf.imgets(image=imagesdss,param='LCS_R50_IMAGE')
                    self.R50_image[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50_IMAGE from ',imagesdss
                try:
                    iraf.imgets(image=imagesdss,param='LCS_R90_IMAGE')
                    self.R90_image[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90_IMAGE from ',imagesdss


                if os.path.exists(output_image):
                    xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag[i],chi2nu=parse_galfit_1comp(output_image+'[2]')
                    self.galfit_xc[i]=xc[0],xc[1]
                    self.galfit_yc[i]=yc[0],yc[1]
                    self.galfit_mag[i]=mag[0],mag[1]
                    self.galfit_Re[i]=Re[0],Re[1]
                    self.galfit_Nsersic[i]=Nsersic[0],Nsersic[1]
                    self.galfit_axisratio[i]=axis_ratio[0],axis_ratio[1]
                    self.galfit_PA[i]=PA[0],PA[1]

        output=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_SDSS.fits'
        gtab=atpy.Table()
        gtab.add_column('xc1',self.galfit_xc[:,0])
        gtab.add_column('xc1err',self.galfit_xc[:,1])
        gtab.add_column('yc1',self.galfit_yc[:,0])
        gtab.add_column('yc1err',self.galfit_yc[:,0])
        gtab.add_column('mag1',self.galfit_mag[:,0])
        gtab.add_column('mag1err',self.galfit_mag[:,1])
        gtab.add_column('re1',self.galfit_Re[:,0])
        gtab.add_column('re1err',self.galfit_Re[:,1])
        gtab.add_column('nsersic1',self.galfit_Nsersic[:,0])
        gtab.add_column('nsersic1err',self.galfit_Nsersic[:,1])
        gtab.add_column('axisratio1',self.galfit_axisratio[:,0])
        gtab.add_column('axisratio1err',self.galfit_axisratio[:,1])
        gtab.add_column('pa1',self.galfit_PA[:,0])
        gtab.add_column('pa1err',self.galfit_PA[:,1])
        gtab.add_column('numerical_error_flag',numerical_error_flag)
        gtab.add_column('R50',self.galfit_R50)
        gtab.add_column('R90',self.galfit_R90)
        gtab.add_column('R50_rawimage',self.R50_image)
        gtab.add_column('R90_rawimage',self.R90_image)
        if os.path.exists(output):
            os.remove(output)

        gtab.write(output)
        return

    def write_galfit_sersic_parameters_24(self):
        self.galfit_xc24=zeros((len(self.ra),2),'f')
        self.galfit_yc24=zeros((len(self.ra),2),'f')
        self.galfit_mag24=zeros((len(self.ra),2),'f')
        self.galfit_Re24=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic24=zeros((len(self.ra),2),'f')
        self.galfit_axisratio24=zeros((len(self.ra),2),'f')
        self.galfit_PA24=zeros((len(self.ra),2),'f')
        numerical_error_flag24=zeros(len(self.ra),'bool')
        chi2nu=zeros(len(self.ra),'f')

        self.galfit_xc24c=zeros((len(self.ra),2),'f')
        self.galfit_yc24c=zeros((len(self.ra),2),'f')
        self.galfit_mag24c=zeros((len(self.ra),2),'f')
        self.galfit_Re24c=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic24c=zeros((len(self.ra),2),'f')
        self.galfit_axisratio24c=zeros((len(self.ra),2),'f')
        self.galfit_PA24c=zeros((len(self.ra),2),'f')
        numerical_error_flag24c=zeros(len(self.ra),'bool')
        chi2nuc=zeros(len(self.ra),'f')
        self.galfit_R5024=zeros((len(self.ra)),'f')
        self.galfit_R9024=zeros((len(self.ra)),'f')
        self.R50_image24=zeros((len(self.ra)),'f')
        self.R90_image24=zeros((len(self.ra)),'f')
        # fixed BA and PA, no conv
        self.galfit_xc24f=zeros((len(self.ra),2),'f')
        self.galfit_yc24f=zeros((len(self.ra),2),'f')
        self.galfit_mag24f=zeros((len(self.ra),2),'f')
        self.galfit_Re24f=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic24f=zeros((len(self.ra),2),'f')
        self.galfit_axisratio24f=zeros((len(self.ra),2),'f')
        self.galfit_PA24f=zeros((len(self.ra),2),'f')
        numerical_error_flag24f=zeros(len(self.ra),'bool')
        chi2nuf=zeros(len(self.ra),'f')
        # fixed BA and PA, with conv
        self.galfit_xc24fc=zeros((len(self.ra),2),'f')
        self.galfit_yc24fc=zeros((len(self.ra),2),'f')
        self.galfit_mag24fc=zeros((len(self.ra),2),'f')
        self.galfit_Re24fc=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic24fc=zeros((len(self.ra),2),'f')
        self.galfit_axisratio24fc=zeros((len(self.ra),2),'f')
        self.galfit_PA24fc=zeros((len(self.ra),2),'f')
        numerical_error_flag24fc=zeros(len(self.ra),'bool')
        chi2nufc=zeros(len(self.ra),'f')
        #flag=(self.On24ImageFlag & self.spiralflag & (self.n.SERSIC_TH50> mipspixelscale))
        flag=(self.On24ImageFlag)
        

        for i in range(len(self.ra)):
            #if (self.On24ImageFlag[i] & self.spiralflag[i] & (self.n.SERSIC_TH50[i] > mipspixelscale)):
            if flag[i]:
                #print output_image
                print i
                output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-noconv-1Comp-galfit-out.fits'
                model1_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-noconv-24-diskimage-1Comp.fits'
                image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits'

                #get parameters from 24um images
                try:
                    iraf.imgets(image=model1_image24,param='LCS_R50')
                    self.galfit_R5024[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50 from ',model1_image24
                try:
                    iraf.imgets(image=model1_image24,param='LCS_R90')
                    self.galfit_R9024[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90 from ',model1_image24
                try:
                    iraf.imgets(image=image24,param='LCS_R50_IMAGE')
                    self.R50_image24[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50_IMAGE from ',image24
                try:
                    iraf.imgets(image=image24,param='LCS_R90_IMAGE')
                    self.R90_image24[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90_IMAGE from ',image24


                if os.path.exists(output_image24):
                    xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24[i],chi2nu[i]=parse_galfit_1comp(output_image24+'[2]')
                    self.galfit_xc24[i]=xc[0],xc[1]
                    self.galfit_yc24[i]=yc[0],yc[1]
                    self.galfit_mag24[i]=mag[0],mag[1]
                    self.galfit_Nsersic24[i]=Nsersic[0],Nsersic[1]
                    self.galfit_Re24[i]=Re[0],Re[1]
                    self.galfit_axisratio24[i]=axis_ratio[0],axis_ratio[1]
                    self.galfit_PA24[i]=PA[0],PA[1]

        for i in range(len(self.ra)):
            if flag[i]:
                #print output_image
                print i
                output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-conv-1Comp-galfit-out.fits'
                model1_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-conv-24-diskimage-1Comp.fits'
                image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits'

                #get parameters from 24um images
                try:
                    iraf.imgets(image=model1_image24,param='LCS_R50')
                    self.galfit_R5024[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50 from ',model1_image24
                try:
                    iraf.imgets(image=model1_image24,param='LCS_R90')
                    self.galfit_R9024[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90 from ',model1_image24
                try:
                    iraf.imgets(image=image24,param='LCS_R50_IMAGE')
                    self.R50_image24[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50_IMAGE from ',image24
                try:
                    iraf.imgets(image=image24,param='LCS_R90_IMAGE')
                    self.R90_image24[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90_IMAGE from ',image24


                if os.path.exists(output_image24):
                    xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24c[i],chi2nuc[i]=parse_galfit_1comp(output_image24+'[2]')
                    self.galfit_xc24c[i]=xc[0],xc[1]
                    self.galfit_yc24c[i]=yc[0],yc[1]
                    self.galfit_mag24c[i]=mag[0],mag[1]
                    self.galfit_Nsersic24c[i]=Nsersic[0],Nsersic[1]
                    self.galfit_Re24c[i]=Re[0],Re[1]
                    self.galfit_axisratio24c[i]=axis_ratio[0],axis_ratio[1]
                    self.galfit_PA24c[i]=PA[0],PA[1]

        for i in range(len(self.ra)):
            if flag[i]:
                #print output_image
                print i
                output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'

                #get parameters from 24um images
                if os.path.exists(output_image24):
                    xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24fc[i],chi2nufc[i]=parse_galfit_1comp(output_image24+'[2]')
                    self.galfit_xc24fc[i]=xc[0],xc[1]
                    self.galfit_yc24fc[i]=yc[0],yc[1]
                    self.galfit_mag24fc[i]=mag[0],mag[1]
                    self.galfit_Nsersic24fc[i]=Nsersic[0],Nsersic[1]
                    self.galfit_Re24fc[i]=Re[0],Re[1]
                    self.galfit_axisratio24fc[i]=axis_ratio[0],axis_ratio[1]
                    self.galfit_PA24fc[i]=PA[0],PA[1]
        for i in range(len(self.ra)):
            if flag[i]:
                #print output_image
                print i
                output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-fixedBA-noconv-1Comp-galfit-out.fits'

                #get parameters from 24um images
                if os.path.exists(output_image24):
                    xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24f[i],chi2nuf[i]=parse_galfit_1comp(output_image24+'[2]')
                    self.galfit_xc24f[i]=xc[0],xc[1]
                    self.galfit_yc24f[i]=yc[0],yc[1]
                    self.galfit_mag24f[i]=mag[0],mag[1]
                    self.galfit_Nsersic24f[i]=Nsersic[0],Nsersic[1]
                    self.galfit_Re24f[i]=Re[0],Re[1]
                    self.galfit_axisratio24f[i]=axis_ratio[0],axis_ratio[1]
                    self.galfit_PA24f[i]=PA[0],PA[1]

        output24=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_24.fits'
        #gtab24=atpy.Table()
        col0a = fits.Column(name='NSAID',array=self.n.NSAID,format='J')
        col0b = fits.Column(name='MIPS_FLAG',array=self.analyze_mips,format='L')
        col1 = fits.Column(name='xc1',array=self.galfit_xc24[:,0],format='E')
        col2 = fits.Column(name='xc1err',array=self.galfit_xc24[:,1],format='E')
        col3 = fits.Column(name='yc1',array=self.galfit_yc24[:,0],format='E')
        col4 = fits.Column(name='yc1err',array=self.galfit_yc24[:,1],format='E')
        col5 = fits.Column(name='mag1',array=self.galfit_mag24[:,0],format='E')
        col6 = fits.Column(name='mag1err',array=self.galfit_mag24[:,1],format='E')
        col7 = fits.Column(name='re1',array=self.galfit_Re24[:,0],format='E')
        col8 = fits.Column(name='re1err',array=self.galfit_Re24[:,1],format='E')
        col9 = fits.Column(name='nsersic1',array=self.galfit_Nsersic24[:,0],format='E')
        col10 = fits.Column(name='nsersic1err',array=self.galfit_Nsersic24[:,1],format='E')
        col11 = fits.Column(name='axisratio1',array=self.galfit_axisratio24[:,0],format='E')
        col12 = fits.Column(name='axisratio1err',array=self.galfit_axisratio24[:,1],format='E')
        col13 = fits.Column(name='pa1',array=self.galfit_PA24[:,0],format='E')
        col14 = fits.Column(name='pa1err',array=self.galfit_PA24[:,1],format='E')
        col15 = fits.Column(name='numerical_error_flag24',array=numerical_error_flag24,format='L')
        col16 = fits.Column(name='chi2nu',array=chi2nu,format='E')
        col17 = fits.Column(name='cxc1',array=self.galfit_xc24c[:,0],format='E')
        col18 = fits.Column(name='cxc1err',array=self.galfit_xc24c[:,1],format='E')
        col19 = fits.Column(name='cyc1',array=self.galfit_yc24c[:,0],format='E')
        col20 = fits.Column(name='cyc1err',array=self.galfit_yc24c[:,1],format='E')
        col21 = fits.Column(name='cmag1',array=self.galfit_mag24c[:,0],format='E')
        col22 = fits.Column(name='cmag1err',array=self.galfit_mag24c[:,1],format='E')
        col23 = fits.Column(name='cre1',array=self.galfit_Re24c[:,0],format='E')
        col24 = fits.Column(name='cre1err',array=self.galfit_Re24c[:,1],format='E')
        col25 = fits.Column(name='cnsersic1',array=self.galfit_Nsersic24c[:,0],format='E')
        col26 = fits.Column(name='cnsersic1err',array=self.galfit_Nsersic24c[:,1],format='E')
        col27 = fits.Column(name='caxisratio1',array=self.galfit_axisratio24c[:,0],format='E')
        col28 = fits.Column(name='caxisratio1err',array=self.galfit_axisratio24c[:,1],format='E')
        col29 = fits.Column(name='cpa1',array=self.galfit_PA24c[:,0],format='E')
        col30 = fits.Column(name='cpa1err',array=self.galfit_PA24c[:,1],format='E')
        col31 = fits.Column(name='cnumerical_error_flag24',array=numerical_error_flag24c,format='L')
        col32 = fits.Column(name='cchi2nu',array=chi2nuc,format='E')
        col17f = fits.Column(name='fxc1',array=self.galfit_xc24f[:,0],format='E')
        col18f = fits.Column(name='fxc1err',array=self.galfit_xc24f[:,1],format='E')
        col19f = fits.Column(name='fyc1',array=self.galfit_yc24f[:,0],format='E')
        col20f = fits.Column(name='fyc1err',array=self.galfit_yc24f[:,1],format='E')
        col21f = fits.Column(name='fmag1',array=self.galfit_mag24f[:,0],format='E')
        col22f = fits.Column(name='fmag1err',array=self.galfit_mag24f[:,1],format='E')
        col23f = fits.Column(name='fre1',array=self.galfit_Re24f[:,0],format='E')
        col24f = fits.Column(name='fre1err',array=self.galfit_Re24f[:,1],format='E')
        col25f = fits.Column(name='fnsersic1',array=self.galfit_Nsersic24f[:,0],format='E')
        col26f = fits.Column(name='fnsersic1err',array=self.galfit_Nsersic24f[:,1],format='E')
        col27f = fits.Column(name='faxisratio1',array=self.galfit_axisratio24f[:,0],format='E')
        col28f = fits.Column(name='faxisratio1err',array=self.galfit_axisratio24f[:,1],format='E')
        col29f = fits.Column(name='fpa1',array=self.galfit_PA24f[:,0],format='E')
        col30f = fits.Column(name='fpa1err',array=self.galfit_PA24f[:,1],format='E')
        col31f = fits.Column(name='fnumerical_error_flag24',array=numerical_error_flag24f,format='L')
        col32f = fits.Column(name='fchi2nu',array=chi2nuf,format='E')
        
        col33f = fits.Column(name='fcxc1',array=self.galfit_xc24fc[:,0],format='E')
        col34f = fits.Column(name='fcxc1err',array=self.galfit_xc24fc[:,1],format='E')
        col35f = fits.Column(name='fcyc1',array=self.galfit_yc24fc[:,0],format='E')
        col36f = fits.Column(name='fcyc1err',array=self.galfit_yc24fc[:,1],format='E')
        col37f = fits.Column(name='fcmag1',array=self.galfit_mag24fc[:,0],format='E')
        col38f = fits.Column(name='fcmag1err',array=self.galfit_mag24fc[:,1],format='E')
        col39f = fits.Column(name='fcre1',array=self.galfit_Re24fc[:,0],format='E')
        col40f = fits.Column(name='fcre1err',array=self.galfit_Re24fc[:,1],format='E')
        col41f = fits.Column(name='fcnsersic1',array=self.galfit_Nsersic24fc[:,0],format='E')
        col42f = fits.Column(name='fcnsersic1err',array=self.galfit_Nsersic24fc[:,1],format='E')
        col43f = fits.Column(name='fcaxisratio1',array=self.galfit_axisratio24fc[:,0],format='E')
        col44f = fits.Column(name='fcaxisratio1err',array=self.galfit_axisratio24fc[:,1],format='E')
        col45f = fits.Column(name='fcpa1',array=self.galfit_PA24fc[:,0],format='E')
        col46f = fits.Column(name='fcpa1err',array=self.galfit_PA24fc[:,1],format='E')
        col47f = fits.Column(name='fcnumerical_error_flag24',array=numerical_error_flag24fc,format='L')
        col48f = fits.Column(name='fcchi2nu',array=chi2nufc,format='E')
        col33 = fits.Column(name='R50',array=self.galfit_R5024,format='E')
        col34 = fits.Column(name='R90',array=self.galfit_R9024,format='E')
        col35 = fits.Column(name='R50_rawimage',array=self.R50_image24,format='E')
        col36 = fits.Column(name='R90_rawimage',array=self.R90_image24,format='E')

        if os.path.exists(output24):
            os.remove(output24)
        print output24
        #gtab24.write(output24)
        cols=fits.ColDefs([col0a,col0b,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col17f,col18f,col19f,col20f,col21f,col22f,col23f,col24f,col25f,col26f,col27f,col28f,col29f,col30f,col31f,col32f,col33f,col34f,col35f,col36f,col37f,col38f,col39f,col40f,col41f,col42f,col43f,col44f,col45f,col46f,col47f,col48f,col33,col34,col35,col36])
        tbhdu=fits.new_table(cols)
        #hdu=tbhdu.header
        #thdulist=fits.HDUList([hdu,tbhdu])
        #thdulist.writeto(output24)
        tbhdu.writeto(output24)
        return

    def write_galfit_sersic_parameters_NSA(self):
        self.galfit_xc=zeros((len(self.ra),2),'f')
        self.galfit_yc=zeros((len(self.ra),2),'f')
        self.galfit_mag=zeros((len(self.ra),2),'f')
        self.galfit_Re=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic=zeros((len(self.ra),2),'f')
        self.galfit_axisratio=zeros((len(self.ra),2),'f')
        self.galfit_PA=zeros((len(self.ra),2),'f')
        numerical_error_flag=zeros(len(self.ra),'bool')
        self.galfit_R50=zeros((len(self.ra)),'f')
        self.galfit_R90=zeros((len(self.ra)),'f')
        self.R50_image=zeros((len(self.ra)),'f')
        self.R90_image=zeros((len(self.ra)),'f')


        for i in range(len(self.ra)):
            if self.On24ImageFlag[i]:
                output_image=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits'
                model1_image=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-diskimage-1Comp.fits'
                imagesdss=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout.fits'
                #print output_image
                try:
                    iraf.imgets(image=model1_image,param='LCS_R50')
                    self.galfit_R50[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50 from ',model1_image
                try:
                    iraf.imgets(image=model1_image,param='LCS_R90')
                    self.galfit_R90[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90 from ',model1_image
                try:
                    iraf.imgets(image=imagesdss,param='LCS_R50_IMAGE')
                    self.R50_image[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50_IMAGE from ',imagesdss
                try:
                    iraf.imgets(image=imagesdss,param='LCS_R90_IMAGE')
                    self.R90_image[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90_IMAGE from ',imagesdss

                if os.path.exists(output_image):
                    xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag[i],chi2nu=parse_galfit_1comp(output_image+'[2]')
                    self.galfit_xc[i]=xc[0],xc[1]
                    self.galfit_yc[i]=yc[0],yc[1]
                    self.galfit_mag[i]=mag[0],mag[1]
                    self.galfit_Re[i]=Re[0],Re[1]
                    self.galfit_Nsersic[i]=Nsersic[0],Nsersic[1]
                    self.galfit_axisratio[i]=axis_ratio[0],axis_ratio[1]
                    self.galfit_PA[i]=PA[0],PA[1]

        output=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_NSA.fits'
        gtab=atpy.Table()
        gtab.add_column('xc1',self.galfit_xc[:,0])
        gtab.add_column('xc1err',self.galfit_xc[:,1])
        gtab.add_column('yc1',self.galfit_yc[:,0])
        gtab.add_column('yc1err',self.galfit_yc[:,0])
        gtab.add_column('mag1',self.galfit_mag[:,0])
        gtab.add_column('mag1err',self.galfit_mag[:,1])
        gtab.add_column('re1',self.galfit_Re[:,0])
        gtab.add_column('re1err',self.galfit_Re[:,1])
        gtab.add_column('nsersic1',self.galfit_Nsersic[:,0])
        gtab.add_column('nsersic1err',self.galfit_Nsersic[:,1])
        gtab.add_column('axisratio1',self.galfit_axisratio[:,0])
        gtab.add_column('axisratio1err',self.galfit_axisratio[:,1])
        gtab.add_column('pa1',self.galfit_PA[:,0])
        gtab.add_column('pa1err',self.galfit_PA[:,1])
        gtab.add_column('numerical_error_flag',numerical_error_flag)
        gtab.add_column('R50',self.galfit_R50)
        gtab.add_column('R90',self.galfit_R90)
        gtab.add_column('R50_rawimage',self.R50_image)
        gtab.add_column('R90_rawimage',self.R90_image)
        if os.path.exists(output):
            os.remove(output)

        gtab.write(output)


    def write_galfit_sersic_parameters_2comp_NSA(self):
        self.galfit_xc=zeros((len(self.ra),2),'f')
        self.galfit_yc=zeros((len(self.ra),2),'f')
        self.galfit_mag=zeros((len(self.ra),2),'f')
        self.galfit_Re=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic=zeros((len(self.ra),2),'f')
        self.galfit_axisratio=zeros((len(self.ra),2),'f')
        self.galfit_PA=zeros((len(self.ra),2),'f')
        numerical_error_flag=zeros(len(self.ra),'bool')
        self.galfit_R50=zeros((len(self.ra)),'f')
        self.galfit_R90=zeros((len(self.ra)),'f')
        self.R50_image=zeros((len(self.ra)),'f')
        self.R90_image=zeros((len(self.ra)),'f')

        self.galfit_xc2=zeros((len(self.ra),2),'f')
        self.galfit_yc2=zeros((len(self.ra),2),'f')
        self.galfit_mag2=zeros((len(self.ra),2),'f')
        self.galfit_Re2=zeros((len(self.ra),2),'f')
        self.galfit_Nsersic2=zeros((len(self.ra),2),'f')
        self.galfit_axisratio2=zeros((len(self.ra),2),'f')
        self.galfit_PA2=zeros((len(self.ra),2),'f')
        numerical_error_flag2=zeros(len(self.ra),'bool')



        for i in range(len(self.ra)):
            if self.On24ImageFlag[i]:
                output_image=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-2Comp-galfit-out.fits'
                model1_image=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-diskimage-2Comp.fits'
                imagesdss=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout.fits'
                #print output_image
                try:
                    iraf.imgets(image=model1_image,param='LCS_R50')
                    self.galfit_R50[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50 from ',model1_image
                try:
                    iraf.imgets(image=model1_image,param='LCS_R90')
                    self.galfit_R90[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90 from ',model1_image
                try:
                    iraf.imgets(image=imagesdss,param='LCS_R50_IMAGE')
                    self.R50_image[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R50_IMAGE from ',imagesdss
                try:
                    iraf.imgets(image=imagesdss,param='LCS_R90_IMAGE')
                    self.R90_image[i]=float(iraf.imgets.value)
                except:
                    print 'problem getting LCS_R90_IMAGE from ',imagesdss

                if os.path.exists(output_image):
                    xc,yc,mag,Re,Nsersic,axis_ratio,PA,xc2,yc2,mag2,Re2,Nsersic2,axis_ratio2,PA2,sky,numerical_error_flag[i],numerical_error_flag2[i]=parse_galfit_2comp(output_image+'[2]')
                    self.galfit_xc[i]=xc[0],xc[1]
                    self.galfit_yc[i]=yc[0],yc[1]
                    self.galfit_mag[i]=mag[0],mag[1]
                    self.galfit_Re[i]=Re[0],Re[1]
                    self.galfit_Nsersic[i]=Nsersic[0],Nsersic[1]
                    self.galfit_axisratio[i]=axis_ratio[0],axis_ratio[1]
                    self.galfit_PA[i]=PA[0],PA[1]
                    # second component
                    self.galfit_xc2[i]=xc2[0],xc2[1]
                    self.galfit_yc2[i]=yc2[0],yc2[1]
                    self.galfit_mag2[i]=mag2[0],mag2[1]
                    self.galfit_Re2[i]=Re2[0],Re2[1]
                    self.galfit_Nsersic2[i]=Nsersic2[0],Nsersic2[1]
                    self.galfit_axisratio2[i]=axis_ratio2[0],axis_ratio2[1]
                    self.galfit_PA2[i]=PA2[0],PA2[1]

        output=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_2comp_NSA.fits'
        gtab=atpy.Table()
        gtab.add_column('xc1',self.galfit_xc[:,0])
        gtab.add_column('xc1err',self.galfit_xc[:,1])
        gtab.add_column('yc1',self.galfit_yc[:,0])
        gtab.add_column('yc1err',self.galfit_yc[:,0])
        gtab.add_column('mag1',self.galfit_mag[:,0])
        gtab.add_column('mag1err',self.galfit_mag[:,1])
        gtab.add_column('re1',self.galfit_Re[:,0])
        gtab.add_column('re1err',self.galfit_Re[:,1])
        gtab.add_column('nsersic1',self.galfit_Nsersic[:,0])
        gtab.add_column('nsersic1err',self.galfit_Nsersic[:,1])
        gtab.add_column('axisratio1',self.galfit_axisratio[:,0])
        gtab.add_column('axisratio1err',self.galfit_axisratio[:,1])
        gtab.add_column('pa1',self.galfit_PA[:,0])
        gtab.add_column('pa1err',self.galfit_PA[:,1])
        gtab.add_column('numerical_error_flag',numerical_error_flag)

        gtab.add_column('xc2',self.galfit_xc2[:,0])
        gtab.add_column('xc2err',self.galfit_xc2[:,1])
        gtab.add_column('yc2',self.galfit_yc2[:,0])
        gtab.add_column('yc2err',self.galfit_yc2[:,0])
        gtab.add_column('mag2',self.galfit_mag2[:,0])
        gtab.add_column('mag2err',self.galfit_mag2[:,1])
        gtab.add_column('re2',self.galfit_Re2[:,0])
        gtab.add_column('re2err',self.galfit_Re2[:,1])
        gtab.add_column('nsersic2',self.galfit_Nsersic2[:,0])
        gtab.add_column('nsersic2err',self.galfit_Nsersic2[:,1])
        gtab.add_column('axisratio2',self.galfit_axisratio2[:,0])
        gtab.add_column('axisratio2err',self.galfit_axisratio2[:,1])
        gtab.add_column('pa2',self.galfit_PA2[:,0])
        gtab.add_column('pa2err',self.galfit_PA2[:,1])
        gtab.add_column('numerical_error_flag2',numerical_error_flag2)
        gtab.add_column('R50',self.galfit_R50)
        gtab.add_column('R90',self.galfit_R90)
        gtab.add_column('R50_rawimage',self.R50_image)
        gtab.add_column('R90_rawimage',self.R90_image)
        if os.path.exists(output):
            os.remove(output)

        gtab.write(output)

    def display_galfit_results(self,i,band=0,ncomp=3,nsaflag=0):
        # one comp fit
        # band = 0 for sdss
        # band = 1 for mips

        galname=os.getcwd()+'/'+self.prefix+'-'+str(self.n.NSAID[i])
        model_mag=[]
        if band == 0:
            print 'running for sdss galaxies!'
            #self.readgalflags()
            galflag=self.galflag
            diskflag=self.diskflag
            if nsaflag:
                working_dir=self.working_dir_nsa
            else:
                working_dir=self.working_dir_sdss
        if band == 1:
            print 'running for mips galaxies!'
            #self.read_galflags_24()
            galflag = self.galflag24
            galname=galname+'-24'
            working_dir=self.working_dir_mips
            diskflag=self.diskflag24
        os.chdir(working_dir)
        
        quitflag=0
        galflag_stop=0
        d=ds9.ds9()
        d.set('cd '+os.getcwd())
        # don't need to do this for the 1-Component fit
        for j in loops:
            component_flag=0
            model_mag=[]
            #print i,j,galflag[i,j]
            output_image=galname+'-'+str(j+1)+'Comp-galfit-out.fits'
            print output_image
            if os.path.exists(output_image): #if galflag[i,j]:
                image_id=galname+'-'
                galfit_log=image_id+str(j+1)+'Comp-fit.log'
                galfit_out=image_id+str(j+1)+'Comp'+'-galfit.01'

                subcomp_image=image_id+str(j+1)+'Comp'+'-subcomps.fits'
                #    - display results (like ds9 multiextension data cube -> use xpa)
                #
                infile=open(galfit_log,'r')
                galfit_results=infile.readlines()
                infile.close()
                d.set('frame delete all')
                print 'file to display = ',output_image
                s='file new multiframe '+output_image
                #print s
                d.set(s)
                d.set('frame delete 1')
                for k in range(2,5):
                    s='frame '+str(k)
                    d.set(s)
                    d.set('scale log')
                    d.set('zoom to fit')
                    print k
                    if k == 2:
                        d.set('regions command {text 30 10 #text="Image" font="times 18 bold" color="red"}')
                    if k == 3:
                        d.set('regions command {text 30 10 #text="Model" font="times 18 bold" color="red"}')
                    if k == 4:
                        d.set('regions command {text 30 10 #text="Residual" font="times 18 bold" color="red"}')
                    #raw_input('hit any key when ready to view subcomponent images \n')
                print 'file to display = ',subcomp_image
                s='file new multiframe '+subcomp_image
                d.set(s)

                if j == 0:
                    endframe=8
                    for m in range(15):
                        print galfit_results[m].rstrip()
                elif j == 1:
                    endframe=9
                    if len(loops) < 1.5:
                        startindex=0
                        endindex=15
                    else:
                        startindex=15
                        endindex=32
                    for m in range(startindex,endindex):
                        print galfit_results[m].rstrip()
                elif j == 2:
                    endframe=10
                    for m in range(32,len(galfit_results)-1):
                        print galfit_results[m].rstrip()
                s='frame delete '+str(endframe)
                d.set(s)
                d.set('frame delete 5')
                d.set('frame delete 6')

                for k in range(2,endframe):
                    if k == 5:
                        continue
                    elif k == 6:
                        continue
                    elif k == 8:
                        d.set('regions command {text 30 10 #text="Components" font="times 18 bold" color="red"}')

                    if k == endframe:
                        d.set('regions command {text 30 10 #text="Components" font="times 18 bold" color="red"}')
                    s='frame '+str(k)
                    d.set(s)
                    d.set('scale log')
                    d.set('zoom to fit')
                #if interactive:
                #    string=raw_input('hit any key when ready to continue (q to quit; s to stop running galfit for this galaxy) \n')
                #    if string.find('q') > -1:
                #        quitflag = 1
                #        return quitflag,galflag_stop
                #    elif string.find('s') > -1:
                #        galflag_stop=1
                #        return quitflag,galflag_stop
                for k in range(j+1):
                    # for one component fit, assume everything is part of the disk
                    if j == 0:
                        compflag = 1
                    else:
                        s='is component %i part of the disk? (1=yes, 0=no) \n'%(k+1)
                        compflag=raw_input(s)
                        if compflag.find('q') > -1:
                            quitflag=1
                            return quitflag
            
                        try:
                            compflag=int(compflag)
                        except:
                            print "couldn't read your input.  let's try again."

                            compflag=raw_input(s)
                            if string.find('q') > -1:
                                quitflag=1
                                return quitflag
                            try:
                                compflag=int(compflag)
                            except:
                                quitflag=1
                                return quitflag
                    if compflag == 1:
                        if j == 0:
                            ncomp=j+k
                            model_mag.append(float(galfit_results[7][35:41]))
                            
                        elif j == 1:
                            ncomp=j+k
                            if len(loops) < 1.5:
                                startindex=0
                            else:
                                startindex=15
                            if k == 0:
                                model_mag.append(float(galfit_results[startindex+7][35:41]))
                            elif k == 1:
                                model_mag.append(float(galfit_results[startindex+9][35:41]))
                        elif j == 2:
                            ncomp=j+k+1
                            if k == 0:
                                model_mag.append(float(galfit_results[32+7][35:41]))
                            elif k == 1:
                                model_mag.append(float(galfit_results[32+9][35:41]))
                            elif k == 2:
                                model_mag.append(float(galfit_results[32+11][35:41]))

                        diskflag[i,ncomp]=compflag
                print 'here is what I have for diskflag[i] = ',diskflag[i]
                # make disk image
                diskimage=galname+'-diskimage-'+str(j+1)+'Comp.fits'
                if os.path.exists(diskimage):
                    os.remove(diskimage)
                if (j == 0) & (diskflag[i,0]):
                    input=subcomp_image+'[2]'
                    iraf.imcopy(input,diskimage)
                    component_flag = 1
                # 2 component fit
                if (j == 1):
                    if diskflag[i,1]:
                        input=subcomp_image+'[2]'
                        iraf.imcopy(input,diskimage)
                        component_flag = 1
                    if diskflag[i,2]:
                        input=subcomp_image+'[3]'
                        if os.path.exists(diskimage):
                            os.system('rm '+working_dir+'temp.fits')
                            print 'adding second component of 2-comp fit model'
                            iraf.imarith(input,"+",diskimage,working_dir+'temp.fits')
                            os.rename(working_dir+'temp.fits',diskimage)
                        else:
                            iraf.imcopy(input,diskimage)
                        component_flag += 2
                # 3 component fit
                if (j == 2): 
                    if diskflag[i,3]:
                        input=subcomp_image+'[2]'
                        iraf.imcopy(input,diskimage)
                        component_flag=1
                    if diskflag[i,4]:                        
                        input=subcomp_image+'[3]'
                        if os.path.exists(diskimage):
                            os.system('rm '+working_dir+'temp.fits')
                            print 'adding second component of 3-comp fit model'
                            iraf.imarith(input,"+",diskimage,working_dir+'temp.fits')
                            os.rename(working_dir+'temp.fits',diskimage)
                        else:
                            iraf.imcopy(input,diskimage)
                        component_flag += 2
                    if diskflag[i,5]:
                        input=subcomp_image+'[4]'
                        if os.path.exists(diskimage):
                            print 'adding third component of 3-comp fit model'
                            os.system('rm '+working_dir+'temp.fits')
                            iraf.imarith(input,"+",diskimage,working_dir+'temp.fits')
                            os.rename(working_dir+'temp.fits',diskimage)
                        else:
                            iraf.imcopy(input,diskimage)
                        component_flag += 4
                # calculate total mag corresponding to the sum of the disk subcomponents
                iraf.imgets(image=subcomp_image+'[2]',param='MAGZPT')
                mzp=float(iraf.imgets.value)
                model_mag=array(model_mag,'d')
                print 'model_mag = ',model_mag
                fluxes=10.**(-1*(model_mag-mzp)/2.5)
                ftot=sum(fluxes)
                mtot=mzp-2.5*log10(ftot)
                print 'total mag = ',mtot
                # add total mag to image header
                iraf.hedit(images=diskimage,fields='MAG_TOTAL',value=mtot,add='yes',verify='no')
                iraf.hedit(images=diskimage,fields='COMPONENT_FLAG',value=component_flag,add='yes',verify='no')
        return quitflag,galflag_stop

    def call_measure_disk(self,i,wave_band=0,raw_data=0,nsaflag=1):
        # gather necessary input for measure_disk
        #galname=os.getcwd()+'/'+self.prefix+'-'+str(self.n.NSAID[i])

        # if raw_data, run ellipse on the actual image rather than the galfit model
        working_dir=os.getcwd()+'/'
        print working_dir
        galname=self.prefix+'-'+str(self.n.NSAID[i])
        model_mag=[]
        if wave_band == 0:
            print 'running call_measure_disk for sdss galaxies!'
            #self.readgalflags()
            galflag=self.galflag
            if nsaflag:
                working_dir=self.working_dir_nsa
            else:
                working_dir=self.working_dir_sdss
            minr=2
        if wave_band == 1:
            print 'running call_measure_disk for mips galaxies!'
            #self.read_galflags_24()
            galflag = self.galflag24
            galname=galname+'-24'
            working_dir=self.working_dir_mips
            minr=1
        os.chdir(working_dir)
        # get basic fit parameters from 1-Component fit
        output_image=galname+'-1Comp-galfit-out.fits'
        iraf.imgets(image=working_dir+output_image+'[2]',param='MAGZPT')
        mag_zp=float(iraf.imgets.value)

        # get basic fit parameters from 1-Component fit
        #output_image=galname+'-1Comp-galfit-out.fits'
        xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag,chi2nu=parse_galfit_1comp(os.getcwd()+'/'+output_image+'[2]')
        print 'output from parse_galfit_1comp in measure_disk:'
        print xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag
        xc=xc[0]
        yc=yc[0]
        if (wave_band == 0) & nsaflag:
            print 'xc,yc = ',xc,yc,self.xcenter_cutout[i],self.ycenter_cutout[i]
            xc=self.xcenter_cutout[i]
            yc=self.ycenter_cutout[i]
        mag=mag[0]
        Re=Re[0]
        axis_ratio=axis_ratio[0]
        PA=PA[0]
        iellip=1-axis_ratio
        if iellip < .05:
            iellip = .05

        initialr = 4*Re
        if initialr < 5:
            initialr = 5
        if initialr > 100:
            initialr = 50


        d=ds9.ds9()
        d.set('frame new')
        d.set('single')
        d.set('zoom to fit')
        if raw_data:
            if wave_band == 1:
                disk_image=self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits'
                mask_image=self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-mask24.fits'
            iraf.imgets(image=working_dir+disk_image,param='naxis1')
            maxr=float(iraf.imgets.value)
            if initialr > maxr:
                initialr = 0.25*maxr
            if wave_band == 1:
                initialr=5
            self.measure_disk(i,disk_image,mask_image,xc,yc,PA,iellip,initialr,minr,maxr,mag_zp,band=wave_band)
            disk_image=galname+'-diskimage-1Comp.fits'
            raw_image=self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits'
            if os.path.exists(working_dir+disk_image):#if galflag[i,j]:

                iraf.imgets(image=working_dir+disk_image,param='MAG_TOTAL')
                mag_tot=float(iraf.imgets.value)

                iraf.hedit(images=raw_image,fields='MAG_TOTAL',value=mag_tot,add='yes',verify='no')

        else:
            for j in loops:
                disk_image=galname+'-diskimage-'+str(j+1)+'Comp.fits'
                if os.path.exists(working_dir+disk_image):#if galflag[i,j]:

                    iraf.imgets(image=working_dir+disk_image,param='MAG_TOTAL')
                    mag_tot=float(iraf.imgets.value)
                    iraf.imgets(image=working_dir+disk_image,param='naxis1')
                    maxr=float(iraf.imgets.value)
                    if initialr > maxr:
                        initialr = 0.5*maxr
                    print i,disk_image,mag_tot,xc,yc,PA,iellip,initialr,minr,maxr,mag_zp
                    self.measure_disk(i,disk_image,'',xc,yc,PA,iellip,initialr,minr,maxr,mag_zp,band=wave_band)


    def measure_disk(self,i,image,mask_image,xcenter,ycenter,ipa,iellip,initialr,minr,maxr,zp,band=0,nframe=1,myradius=15,keepfixed=0,einteractive=0):
        myradius=initialr
        recentervalue='yes'
        recentervalue='no'
        interactivevalue='no'
        # run ellipse on disk image
        # interpolate where enclosed flux is 50% and 90% of total (see elbaz code for interpolating w/discrete data)
        # store results in an array
        # galfit_sdssR50[Nx3] array
        #
        # band = 0 for sdss
        # band = 1 for 2um

        print '\n Running ellipse on ',image,'\n'
        print 'current working directory = ',os.getcwd()
        working_dir = os.getcwd()+'/'
        mfile=image

        #if working_dir.find('SDSS') > -1:
        #    xcenter_cutout=self.xcenter_cutout
        #    ycenter_cutout=self.ycenter_cutout
        #elif working_dir.find('24') > -1:
        #    xcenter_cutout=self.xcenter_cutout24
        #    ycenter_cutout=self.ycenter_cutout24
        xcenter_cutout=xcenter
        ycenter_cutout=ycenter
        #run ellipse
        t=mfile.split('.')
        efile=t[0]+'.tab'
        efile_ascii=t[0]+'.dat'
        imprefix=t[0]
        print 'Running ellipse to fit isophotes to galaxy:',mfile
        if os.path.exists(efile):
            os.remove(efile)
        if keepfixed:
            print "keeping input parameters fixed"
            print image,efile,xcenter_cutout,ycenter_cutout,initialr,minr,maxr,ipa,iellip
            if band == 0:
                iraf.ellipse(input=image,output=efile,dqf=mask_image,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='yes',recenter=recentervalue,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='yes',ellip=iellip,hellip='yes',interactive='yes',step=0.1,linear='no')
            else:
                try:
                    iraf.ellipse(input=image,output=efile,dqf=mask_image,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='yes',recenter=0,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='yes',ellip=iellip,hellip='yes',interactive='yes',step=0.1,linear='no')
                except:
                    image=image+'[1]'
                    iraf.ellipse(input=image,output=efile,dqf=mask_image,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='yes',recenter=0,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='yes',ellip=iellip,hellip='yes',interactive='yes',step=0.1,linear='no')
                
        else:
            print "First pass, letting PA and e vary"
            print image,efile,xcenter_cutout,ycenter_cutout,initialr,minr,maxr,ipa,iellip
            print 'just printed iellip'
            try:
                iraf.ellipse(input=image,output=efile,dqf=mask_image,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='yes',recenter=0,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no',interactive=einteractive,step=0.1,linear='no')
            except:
                image=image+'[1]'
                iraf.ellipse(input=image,output=efile,dqf=mask_image,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='yes',recenter=0,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no',interactive=einteractive,step=0.1,linear='no')

        print 'Displaying isophotes from first pass.  Hit q in DS9 window to quit'
        iraf.isoexam(table=efile)
        try:
            os.system('rm tp.00*')
        except:
            print 'no tp.00? files to remove'
        iraf.isoimap(image,table=efile)
        if os.path.exists(working_dir+'junk.txt'):
            os.remove(working_dir+'junk.txt')
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout=working_dir+'junk.txt')
        s="awk '{print $2, $7, $9, $11, $13}' < "+working_dir+"junk.txt > "+working_dir+"junk2.txt"
        os.system(s)
        #run ellipse a second time, keeping PA and ellip fixed
        #allow user to adjust the radius where PA and ellip are measured
        if os.path.exists(efile_ascii):
            os.remove(efile_ascii)
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout=efile_ascii)
        
    def measure_disk_interactive(self,i,image,mask_image,xcenter,ycenter,ipa,iellip,initialr,minr,maxr,zp,band=0,nframe=1,myradius=15,keepfixed=0):
        #run ellipse a second time, keeping PA and ellip fixed
        #allow user to adjust the radius where PA and ellip are measured
        repeatflag=1
        while (repeatflag > 0.1):
            infile=open(working_dir+'junk2.txt','r')
            for line in infile:
                t=line.split()
                if float(t[0]) > myradius:
                    newellip=float(t[1])
                    if newellip < .05: # min value that ellipse can handle
                        newellip=.05
                    newPA=float(t[2])
                    if newPA < -90:
                        newPA=newPA+180
                    elif newPA > 90:
                        newPA = newPA-180
                    
                    newxcenter=float(t[3])
                    newycenter=float(t[4])
                    break
            s='rm '+efile
            os.system(s)
            print 'newxcenter,newycenter = ',newxcenter,newycenter
            iraf.ellipse(input=image,output=efile,dqf=mask_image,x0=newxcenter,y0=newycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=newellip,hellip='yes',recenter=recentervalue,interactive=interactivevalue)

            print 'Displaying isophotes from second pass using r = ',myradius
            print 'Hit q in the DS9 window to quit'
            iraf.isoexam(table=efile)
            #iraf.isoimage(image,table=efile)
            flag=str(raw_input('Are you happy with the fit?  y (or any key)=yes n=no x=quit'))
            flag=str(flag)
            print 'this is what I think you typed ',flag
            if flag.find('n') > -1:
                flag2=str(raw_input('What is the problem? r=set new radius, c=set new center, i=enter interactive mode, x=quit '))
                flag2=str(flag2)
                if flag2.find('r') > -1:
                    myr=input('Enter new radius to use ')
                    myradius=float(myr)
                    s='rm '+efile
                    os.system(s)
                    repeatflag=1
                if flag2.find('i') > -1:
                    interactivevalue='yes'
                    s='rm '+efile
                    os.system(s)
                    repeatflag=1
                elif flag2.find('c') > -1:
                    repeatflag=1
                    newxc=input('Enter new x center to use ')
                    newxcenter=float(newxc)
                    newyc=input('Enter new y center to use ')
                    newyccenter=float(newyc)
                    s='rm '+efile
                    os.system(s)
                    repeatflag=1
                    recentervalue='no'
                elif flag2.find('x') > -1:
                    repeatflag=0
                    return
            elif flag.find('x') > -1:
                repeatflag=0
                print 'i think repeatflag = 0', repeatflag
                return
            else:
                repeatflag=0
                print 'i think repeatflag = 0 ', repeatflag

        # convert ellipse table from binary to ascii
        if os.path.exists(efile_ascii):
            os.remove(efile_ascii)
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout=efile_ascii)


    def call_measure_disk_useNSA(self,i,wave_band=0,raw_data=0,nsaflag=1):
        # gather necessary input for measure_disk
        #galname=os.getcwd()+'/'+self.prefix+'-'+str(self.n.NSAID[i])

        # if raw_data, run ellipse on the actual image rather than the galfit model
        working_dir=os.getcwd()+'/'
        print working_dir
        galname=self.prefix+'-'+str(self.n.NSAID[i])
        model_mag=[]
        if wave_band == 0:
            print 'running call_measure_disk for sdss galaxies!'
            #self.readgalflags()
            galflag=self.galflag
            if nsaflag:
                working_dir=self.working_dir_nsa
            else:
                working_dir=self.working_dir_sdss
            minr=2
        if wave_band == 1:
            print 'running call_measure_disk for mips galaxies!'
            #self.read_galflags_24()
            galflag = self.galflag24
            galname=galname+'-24'
            working_dir=self.working_dir_mips
            minr=1
        os.chdir(working_dir)
        # get basic fit parameters from 1-Component fit

        flux_zp_AB = 3631. # in Jy
        flux_zp_Vega = 7.17 # in Jy
        flux_zp=flux_zp_AB
        
        # conversion from image units of MJ/sr to micro-Jy (1 sq arcsec = 2.3504e-11 sr)
        conv_MJysr_uJy = 23.5045*(2.45**2)
        self.magzp24=2.5*log10(flux_zp*1.e6/conv_MJysr_uJy)

        mag_zp=self.magzp24
        
        # get center of 24um image
        mipsimage=self.working_dir_mips+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits'
        rimage=self.working_dir_nsa+self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits'

        if wave_band == 0:
            inputimage=rimage
        else:
            inputimage=mipsimage

        print inputimage
        hdu=fits.open(inputimage)
        hdat=hdu[0].data.copy()
        n2,n1=hdat.shape
        hdu.close()

        #w= WCS(inputimage)
        #xc,yc = w.wcs_world2pix(self.n.RA[i],self.n.DEC[i],1)
        #print 'center x,y = ',xc,yc
        xc=n1/2.
        yc=n2/2.
        # get basic fit parameters from 1-Component fit of r-band image
        Re=self.n.SERSIC_TH50[i]/mipspixelscale
        axis_ratio=self.n.SERSIC_BA[i]
        PA=self.n.SERSIC_PHI[i]
        if PA < -90:
            PA=PA+180
        elif PA > 90:
            PA = PA-180

        iellip=1-axis_ratio
        if iellip < .05:
            iellip = .05
        initialr = 4*Re
        if initialr < 5:
            initialr = 5
        if initialr > 100:
            initialr = 50



        try:
            d.set('frame delete all')
        except:
            d=ds9.ds9()

        d.set('frame 1')
        d.set('single')
        d.set('frame 2')
        s='file '+rimage+'[1]'
        try:
            d.set(s)
        except:
            print 'trouble loading ',rimage

        if wave_band == 1:
            disk_image=self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits'
            mask_image=self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-mask24.fits'
            if iellip > .6:
                iellip = .6
        else:

            multext_disk_image=self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits'
            disk_image=self.prefix+'-'+str(self.n.NSAID[i])+'-cutout.fits'
            if os.path.isfile(disk_image):
                os.remove(disk_image)
            iraf.imcopy(multext_disk_image+'[1]', disk_image)
            
            full_mask_image=self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-mask.fits'
            mask_image=self.prefix+'-'+str(self.n.NSAID[i])+'-mask.fits'
            inputfile=self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit.01'
            print inputfile
            os.system('grep H '+inputfile+' |grep xmin > junk')
            dat=np.loadtxt('junk',usecols=(1,2,3,4))
            imsection='[%i:%i,%i:%i]'%(dat[0],dat[1],dat[2],dat[3])
            if os.path.isfile(mask_image):
                os.remove(mask_image)
            print full_mask_image+imsection, mask_image
            s=full_mask_image+imsection
            iraf.imcopy(s,mask_image)
            bad_masks=['70685','18127','79416']
            if disk_image.find('70685') > -1:
                mask_image='none'
            if disk_image.find('18127') > -1:
                mask_image='none'
            if disk_image.find('79416') > -1:
                mask_image='none'
            if disk_image.find('166042') > -1:
                mask_image='none'
            if disk_image.find('43712') > -1:
                mask_image='none'
            if disk_image.find('43857') > -1:
                mask_image='none'
            if disk_image.find('69538') > -1:
                mask_image='none'
            if disk_image.find('113482') > -1:
                mask_image='none'
            if disk_image.find('140139') > -1:
                mask_image='none'
            if disk_image.find('104232') > -1:
                mask_image='none'
            if disk_image.find('142781') > -1:
                mask_image='none'
        maxr=max(n1,n2)/2.
        if initialr > maxr:
            initialr = 0.25*maxr
        if wave_band == 1:
            initialr=5
        print 'image dimensions = ',n1,n2
        print xc,yc,iellip,PA,initialr,minr,maxr,mag_zp
        if self.n.NSAID[i] == 166134:
            maxr=25.
        if wave_band == 1:
            print 'iellip = ',iellip
            self.measure_disk(i,disk_image,mask_image,xc,yc,PA,0.05,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=1,einteractive=0)
            os.rename(disk_image.split('.fits')[0]+'.dat',disk_image.split('.fits')[0]+'.e05.dat')
            #t=raw_input('fit PA and ellip? (y=yes, other to skip)')
            #if t.find('y') > -1:
            #    print 'fitting PA and ellip!'
            #    self.measure_disk(i,disk_image,mask_image,xc,yc,PA,iellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=0,einteractive=0)
            #    os.rename(disk_image.split('.fits')[0]+'.dat',disk_image.split('.fits')[0]+'.epfit.dat')
            #else:
            #    print 'not fitting PA and ellip'
        print 'iellip = ',iellip
        if (wave_band == 1) & (self.n.NSAID[i] == 166134):
            maxr=25.
        if (wave_band == 1) & (self.n.NSAID[i] == 166167):
            maxr=18.

        self.measure_disk(i,disk_image,mask_image,xc,yc,PA,iellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=1)
        #t=raw_input('enter x to quit, i to rerun in interactive mode, any other key to continue\n')
        t='a'
        if t.find('x') > -1:
            return 1
        if t.find('i') > -1:
            self.measure_disk(i,disk_image,mask_image,xc,yc,PA,iellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=1,einteractive=1)
        #skipping this for now by settin wave_band == 8 (which doesn't exist)
        if wave_band == 8: #
            # make convolved image
            iraf.imfilter.gauss(input=disk_image,output='c'+disk_image,sigma=5.9)
            # run ellipse on convolved image
            if iellip > .6:
                iellip = .6
            self.measure_disk(i,'c'+disk_image,mask_image,xc,yc,PA,iellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=1,einteractive=1)



    def call_measure_radius(self,i,wave_band=0,raw_data=0,nsaflag=1):
        # gather necessary input for measure_disk
        #galname=os.getcwd()+'/'+self.prefix+'-'+str(self.n.NSAID[i])
        working_dir=os.getcwd()+'/'
        print working_dir
        galname=self.prefix+'-'+str(self.n.NSAID[i])
        model_mag=[]
        if wave_band == 0:
            print 'running call_measure_radius for sdss galaxies!'
            #self.readgalflags()
            galflag=self.galflag
            if nsaflag:
                working_dir=self.working_dir_nsa
            else:
                working_dir=self.working_dir_sdss
            minr=2
        if wave_band == 1:
            print 'running call_measure_radius for mips galaxies!'
            #self.read_galflags_24()
            galflag = self.galflag24
            galname=galname+'-24'
            working_dir=self.working_dir_mips
            minr=1
        os.chdir(working_dir)
        # get basic fit parameters from 1-Component fit
        output_image=galname+'-1Comp-galfit-out.fits'
        iraf.imgets(image=working_dir+output_image+'[2]',param='MAGZPT')
        mag_zp=float(iraf.imgets.value)

        # get basic fit parameters from 1-Component fit

        xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag,chi2nu=parse_galfit_1comp(os.getcwd()+'/'+output_image+'[2]')
        xc=xc[0]
        yc=yc[0]
        mag=mag[0]
        Re=Re[0]
        axis_ratio=axis_ratio[0]
        PA=PA[0]
        iellip=1-axis_ratio
        if iellip < .05:
            iellip = .05
        initialr = 4*Re
        if initialr < 5:
            initialr = 5


        d=ds9.ds9()
        d.set('frame new')
        d.set('single')
        d.set('zoom to fit')
        for j in loops:
            disk_image=galname+'-diskimage-'+str(j+1)+'Comp.fits'
            if os.path.exists(working_dir+disk_image):#if galflag[i,j]:

                iraf.imgets(image=working_dir+disk_image,param='MAG_TOTAL')
                mag_tot=float(iraf.imgets.value)
                iraf.imgets(image=working_dir+disk_image,param='naxis1')
                maxr=float(iraf.imgets.value)

                R50,R90=self.measure_radius(i,disk_image,wave_band,mag_zp)
                print 'R50, R90 = ',R50,R90
                if wave_band == 0:
                    self.galfit_sdssR50[i,j]=R50
                    self.galfit_sdssR90[i,j]=R90
                elif wave_band == 1:
                    self.galfit_mipsR50[i,j]=R50
                    self.galfit_mipsR90[i,j]=R90


 
    def measure_radius(self,i,image,band,zp,raw_data=0):
        working_dir = os.getcwd()+'/'
        # construct ellipse table names
        t=image.split('.')
        efile=t[0]+'.tab'
        efile_ascii=t[0]+'.dat'
        imprefix=t[0]
        
        # get total mag and flux zp from the image header
        iraf.imgets(image=image,param='MAG_TOTAL') 
	mag_total=float(iraf.imgets.value)
        # calculate total flux
        flux_tot=10.**(-1.*(mag_total-zp)/2.5)

        if raw_data:
            flux_tot=self.mipsflux[i]/146.
            
        if band == 0:
            flux_tot = flux_tot*53.9075
            pixelscale=sdsspixelscale
            if os.getcwd().find('NSA') > -1:
                flux_tot=10.**(-1.*(mag_total-zp)/2.5)
        else:
            pixelscale=mipspixelscale
        ellipse_output=atpy.Table(efile_ascii,type='ascii')

        # read in sma and enclosed flux from ellipse output
        sma=ellipse_output['col2']
        enc_flux=ellipse_output['col22']
        print zp, mag_total, flux_tot, max(enc_flux)
        figure()
        plot(sma,enc_flux)
        s=image+' (SNR24=%5.1f)'%(self.snr24[i])
        title(s)
        xlabel('semi-major axis (pixels)')
        ylabel('Enclosed Flux (ADU or something like that)')
        axhline(y=0.5*flux_tot,ls='--',color='k',label='_nolegend_')
        axhline(y=0.9*flux_tot,ls=':',color='k',label='_nolegend_')

        # fit enclosed flux vs semi-major axis
        profile_func=interp1d(enc_flux,sma)

        # measure R50 (where enclosed flux = 50% total, R90
        try:
            R50=profile_func(0.5*flux_tot)
        except ValueError:
            print '\n ',image[0],': F50 is outside interpolation range!!! \n'
            R50=0
        iraf.hedit(images=image,fields='LCS_R50',value=R50,add='yes',verify='no')
        try:
            R90=profile_func(0.9*flux_tot)
        except ValueError:
            print '\n ',image[0],': F90 is outside interpolation range!!! \n'
            R90=0
        iraf.hedit(images=image,fields='LCS_R90',value=R90,add='yes',verify='no')
        axvline(x=R50,ls='--',color='r',label='LCS R50')
        axvline(x=R90,ls=':',color='r',label='LCS R90')
        #axvline(x=self.sdssPetroR50r[i]/sdsspixelscale,ls='--',color='k',label='SDSS R50')
        #axvline(x=self.sdssPetroR90r[i]/sdsspixelscale,ls=':',color='k',label='SDSS R90')
        axvline(x=self.n.PETROTH50[i]/pixelscale,ls='--',color='k',label='SDSS R50')
        axvline(x=self.n.PETROTH90[i]/pixelscale,ls=':',color='k',label='SDSS R90')
        #axvline(x=self.sdssIsoAr[i],ls=':',color='c',label='SDSS ISO')
        legend(loc='lower right')
        t=image.split('.')
        fname=working_dir+t[0]+'-radialPlot.png'
        savefig(fname)
        close('all')
        return R50,R90

    def measure_radius_rawdata(self,i,band):
        if band == 1:
            image=self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits' 
            output_image=self.prefix+'-'+str(self.n.NSAID[i])+'-24-1Comp-galfit-out.fits'
        xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag,chi2nu=parse_galfit_1comp(os.getcwd()+'/'+output_image+'[2]')
        working_dir = os.getcwd()+'/'
        # construct ellipse table names
        t=image.split('.')
        efile=t[0]+'.tab'
        efile_ascii=t[0]+'.dat'
        imprefix=t[0]
        
        ellipse_output=atpy.Table(efile_ascii,type='ascii')

        # read in sma and enclosed flux from ellipse output
        sma=ellipse_output['col2']
        inten=ellipse_output['col3']
        enc_flux=ellipse_output['col22']
        index=arange(len(sma))

        
        ## left off here.  need to find radius where flux meets sky, then sum flux w/in that
        # find radius where inten drops below sky
        iend=min(index[inten < (sky[0]+2.*abs(sky[1]))])-1
        flux_tot=enc_flux[iend]
        print flux_tot, max(enc_flux)
        figure(figsize=(8,10))
        subplot(2,1,2)
        plot(sma,enc_flux)
        xlabel('semi-major axis (pixels)')
        ylabel('Enclosed Flux (ADU or something like that)')
        axhline(y=0.5*flux_tot,ls='--',color='k',label='_nolegend_')
        axhline(y=0.9*flux_tot,ls=':',color='k',label='_nolegend_')
        axvline(x=sma[iend],ls='--',color='c',label='_nolegend_')
        # fit enclosed flux vs semi-major axis
        profile_func=interp1d(enc_flux[0:iend+1],sma[0:iend+1])

        # measure R50 (where enclosed flux = 50% total, R90
        try:
            R50=profile_func(0.5*flux_tot)
        except ValueError:
            print '\n ',image[0],': F50 is outside interpolation range!!! \n'
            R50=0
        iraf.hedit(images=image,fields='LCS_R50_IMAGE',value=R50,add='yes',verify='no')
        try:
            R90=profile_func(0.9*flux_tot)
        except ValueError:
            print '\n ',image[0],': F90 is outside interpolation range!!! \n'
            #print 0.9*flux_tot, enc_flux[0:iend],sma[0:iend]
            R90=0
        iraf.hedit(images=image,fields='LCS_R90_IMAGE',value=R90,add='yes',verify='no')
        axvline(x=R50,ls='--',color='r',label='LCS R50')
        axvline(x=R90,ls=':',color='r',label='LCS R90')
        #axvline(x=self.sdssPetroR50r[i]/sdsspixelscale,ls='--',color='k',label='SDSS R50')
        #axvline(x=self.sdssPetroR90r[i]/sdsspixelscale,ls=':',color='k',label='SDSS R90')
        axvline(x=self.n.PETROTH50[i]/mipspixelscale,ls='--',color='k',label='SDSS R50')
        axvline(x=self.n.PETROTH90[i]/mipspixelscale,ls=':',color='k',label='SDSS R90')
        #axvline(x=self.sdssIsoAr[i],ls=':',color='c',label='SDSS ISO')
        legend(loc='lower right')
        subplot(2,1,1)
        plot(sma,inten)
        mipssma=self.mipsprf['col2']/4.
        mipsinten=self.mipsprf['col3']
        index=arange(len(mipssma))
        iscale=min(index[mipssma > min(sma)])-1
        scale=inten[0]/mipsinten[iscale]
        plot(mipssma,mipsinten*scale,color='0.5',label='MIPS PRF')
        xlabel('semi-major axis (pixels)')
        ylabel('Intensity (ADU/pixel or something like that)')
        s=image+' (SNR24=%5.1f)'%(self.snr24[i])
        title(s)

        axhline(y=sky[0],ls='-',color='k',label='_nolegend_')
        axhline(y=sky[0]+2.*sky[1],ls='--',color='k',label='_nolegend_')
        axhline(y=sky[0]-2.*sky[1],ls='--',color='k',label='_nolegend_')
        axvline(x=sma[iend],ls='--',color='c',label='F(total)')

        axvline(x=R50,ls='--',color='r',label='LCS R50')
        axvline(x=R90,ls=':',color='r',label='LCS R90')

        axvline(x=self.n.PETROTH50[i]/mipspixelscale,ls='--',color='k',label='SDSS R50')
        axvline(x=self.n.PETROTH90[i]/mipspixelscale,ls=':',color='k',label='SDSS R90')
        legend(loc='upper right')
        t=image.split('.')
        fname=working_dir+t[0]+'-radialPlot.png'
        
        savefig(fname)
        return R50,R90
   
    def get_images(self,i,getimages=0, keepimages=1):
        # getimages:
        # set = 0 to not get images (if they already exist)
        # set = 1 if sdss images are needed
        #
        # this subroutine gets sdss images,
        #    parses relevant input for galfit,
        #    and runs galfit three times.
        #
        # band = 0 for sdss
        # band = 1 for 2um
        #
        # keepimages = 1 to keep sdss DAS images
        # keepimages = 0 to delete sdss DAS images (better if disk space is tight)

        getimages=0
        # open ds9 display
        d=ds9.ds9()
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        d.set('cd '+os.getcwd())
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')
        print i
        #field=mfield[i]
        #run=mrun[i]
        #rerun=mrerun[i]
        #camcol=mcamcol[i]

        ######################
        #
        # Get SDSS Images
        #
        ######################
            
        field=self.sdss_field[i]
        run=self.sdss_run[i]
        rerun=self.sdss_rerun[i]
        camcol=self.sdss_camcol[i]
        row=self.sdss_rowc[i]
        #row=row[2]
        col=self.sdss_colc[i]
        #col=col[2]
        print 'row, col = ',row, col
        tmp_run='%06d'%(run)
        tmp_field='%04d'%(field)
        sdss_path_c = "imaging/"+str(run)+"/"+str(rerun)+"/corr/"+str(camcol)+"/"
        sdss_corr_name_g = "fpC-"+tmp_run+"-g"+str(camcol)+"-"+tmp_field+".fit"
        sdss_corr_name_r = "fpC-"+tmp_run+"-r"+str(camcol)+"-"+tmp_field+".fit"
        url="http://das.sdss.org/"+sdss_path_c+sdss_corr_name_g+'.gz'

        # updating program to use drC file instead of fpC image
        # looks like rowc and colc are the positions on the drC file
        #
        #s='wget --no-http-keep-alive -S http://das.sdss.org/'+sdss_path_c+sdss_corr_name_r+'.gz'
        #print url
        #if getimages:
        #    urllib.urlretrieve(url,filename=sdss_corr_name_r+'.gz')
        #    os.system('gunzip %s'%(sdss_corr_name_r+'.gz'))

        # allsky=tsfield.sky[0]
        # sky=allsky[2] - this is in maggies/arcsec^2
        # get drC file -
        #   The DAS can generate "header supplemented" versions of the fpC files, called drC files.
        #   These are the same as the fpC files, except have newly generated header values for a number
        #   of parameters, including FLUX20 and SKY, derived from the data in the tsField file. (The
        #   DAS generates these "on the fly" from the fpC and tsField files on disk, so you need to use
        #   the http interface rather than the rsync interface to get them.)

        #   e.g. http://das.sdss.org/www/cgi-bin/drC?RUN=3805&RERUN=41&CAMCOL=3&FIELD=96&FILTER=r

        #iraf.imgets(image=sdss_corr_name_r,param='SKY')#get RA of image
	#sky=float(iraf.imgets.value)
        sdss_drC_name_r = "drC-"+tmp_run+"-r"+str(camcol)+"-"+str(rerun)+"-"+tmp_field+".fits"
        drCurl="http://das.sdss.org/www/cgi-bin/drC?RUN=%s&RERUN=%s&CAMCOL=%s&FIELD=%s&FILTER=r"%(str(run),str(rerun),str(camcol),str(field))
        print sdss_drC_name_r
        print drCurl
        if getimages:
            urllib.urlretrieve(drCurl,filename=sdss_drC_name_r)
        #   photometric zeropoint from fpC image header, keyword = FLUX20, then convert to zp
        iraf.imgets(image=working_dir+sdss_drC_name_r,param='FLUX20')
        flux20=float(iraf.imgets.value)
        iraf.imgets(image=working_dir+sdss_drC_name_r,param='EXPTIME')
        exptime=float(iraf.imgets.value)
        iraf.imgets(image=working_dir+sdss_drC_name_r,param='ZP')
        magzp=float(iraf.imgets.value) - 2.5*log10(exptime)
        iraf.imgets(image=working_dir+sdss_drC_name_r,param='SKY')
        sky=float(iraf.imgets.value)


        # display sdss image
        #d.set('file %s'%(sdss_drC_name_r))
        # get psField file to use when reconstructing the PSF
        sdss_path_o = "imaging/"+str(run)+"/"+str(rerun)+"/objcs/"+str(camcol)+"/"
        sdss_ps_name_r = "psField-"+tmp_run+"-"+str(camcol)+"-"+tmp_field+".fit"

        psurl="http://das.sdss.org/"+sdss_path_o+sdss_ps_name_r
        print psurl
        if getimages:
            urllib.urlretrieve(psurl,filename=sdss_ps_name_r)
        #http://das.sdss.org/imaging/3805/41/objcs/3/psField-003805-3-0096.fit

        # construct PSF, given row, col in image
        #
        # read_PSF psField.... .fit 2 row col output.fit

        psf_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'SDSS-PSF.fits'
        sky_sub_psf_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'SDSS-PSF-SKY.fits'
        s='read_PSF '+sdss_ps_name_r+' 2 '+str(row)+' '+str(col)+' '+psf_image
        os.system(s)
        # subtract 1000 ADU pedestal from psf image
        try:
            iraf.imarith(psf_image,'-','1000.',sky_sub_psf_image)
        except:
            print 'Warning:  Problem creating bias-subtracted PSF image, probably b/c it already exists'
            print '   going to delete bias-subtracted PSF image and try imarith again'
            iraf.imdel(sky_sub_psf_image)
            iraf.imarith(psf_image,'-','1000.',sky_sub_psf_image)


        # get image parameters (GAIN, dark_variance) from tsfield file
            
        sdss_path_calib = "imaging/"+str(run)+"/"+str(rerun)+"/calibChunks/"+str(camcol)+"/"
        sdss_ts_name_r = "tsField-"+tmp_run+"-"+str(camcol)+"-"+str(rerun)+"-"+tmp_field+".fit"
        tsurl="http://das.sdss.org/"+sdss_path_calib+sdss_ts_name_r
        print tsurl
        if getimages:
            urllib.urlretrieve(tsurl,filename=sdss_ts_name_r)
        tsfield=atpy.Table(working_dir+sdss_ts_name_r)
        gain=tsfield.gain[0]
        gain_r=gain[2]
        psfwidth=tsfield.psf_width[0]
        psfwidth_r=psfwidth[2]
        darknoise=tsfield.dark_variance[0]
        darknoise_r=darknoise[2]


        magzp=20.+2.5*log10(flux20/exptime)


        ######################
        #
        # End of getting SDSS Images
        #
        ######################
        if self.rotate_sdssdrC:
            # get rotation from the image header
            # rotate image so N is up, E to left
            iraf.rotate(sdss_drC_name_r,'temp.fits',rotation=90)
            os.rename('temp.fits',sdss_drC_name_r)
            # rotate the col,row for new rotated image
            col=self.sdss_rowc[i]
            row=self.sdss_colc[i]

        # subtract soft bias from image (1000 added to all DN)
        bias_sub_image="drC-"+tmp_run+"-r"+str(camcol)+"-"+tmp_field+"BiasSub.fit"
        try:
            iraf.imarith(sdss_drC_name_r,'-','1000.',bias_sub_image)
        except:
            print 'Warning:  Problem creating bias-subtracted image, probably b/c it already exists'
            print '   going to delete bias-subtracted image and try imarith again'
            iraf.imdel(bias_sub_image)
            iraf.imarith(sdss_drC_name_r,'-','1000.',bias_sub_image)
        # add rdnoise and gain to image header
        iraf.hedit(images=bias_sub_image,fields='RDNOISE',value=darknoise_r,add='yes',verify='no')
        iraf.hedit(images=bias_sub_image,fields='GAIN',value=gain_r,add='yes',verify='no')

        # convert sdssPetroR90r to pixels using sdss plate scale (0.396127"/pix)
        # if R90 < 10 arcsec, use 10 arcsec instead of R90
        if self.n.PETROTH90[i] > (min_cutout_radius):
            PETROTH90_pixels =self.n.PETROTH90[i]/sdsspixelscale
        else:
            PETROTH90_pixels =min_cutout_radius/sdsspixelscale
        sex_image=bias_sub_image

        # run sextractor to generate a list of objects in the image
        # generate 'segmentation image'
        os.system('cp ~/research/LocalClusters/sextractor/default.sex.sdss.galfit .')
        os.system('cp ~/research/LocalClusters/sextractor/default.param .')
        os.system('cp ~/research/LocalClusters/sextractor/default.conv .')
        os.system('cp ~/research/LocalClusters/sextractor/default.nnw .')
        os.system('sex %s -c default.sex.sdss.galfit'%(sex_image))
        # convert segmentation image to object mask by replacing the object ID of target galaxy with zeros
        #   parse sextractor output to get x,y coords of objects        
        sexout=atpy.Table(working_dir+'test.cat',type='ascii')
        sexnumber=sexout['col1']
        xsex=sexout['col2']
        ysex=sexout['col3']
        dist=sqrt((row-ysex)**2+(col-xsex)**2)

        #   find object ID

        objIndex=where(dist == min(dist))
        objNumber=sexnumber[objIndex]
        objNumber=objNumber[0] # not sure why above line returns a list
        print 'object number = ',objNumber
        mask_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-mask.fits'
        os.rename(working_dir+'segmentation.fits',working_dir+mask_image)

        #   use iraf imexam to replace object ID values with zero
        iraf.imreplace(working_dir+mask_image,value=0,lower=objNumber-.5,upper=objNumber+.5)

        # get best x,y positions from sextractor
        #    - subtract 1 b/c sextractor catalog starts at one, but python is zero-indexed
        print "row, col, xsex, ysex = ",row, col, xsex[objIndex],ysex[objIndex]
        xcenter=xsex[objNumber-1]
        ycenter=ysex[objNumber-1]
        # define fit area in terms of the petrosian R90
        xminfit=xcenter-mult_petro90*PETROTH90_pixels
        xmaxfit=xcenter+mult_petro90*PETROTH90_pixels 
        yminfit=ycenter-mult_petro90*PETROTH90_pixels 
        ymaxfit=ycenter+mult_petro90*PETROTH90_pixels 

        print 'xcenter, ycenter = ',xcenter,ycenter
        print 'petrosian R90 in pixels = ',PETROTH90_pixels
        print 'fitting region = [%5.2f, %5.2f, %5.2f, %5.2f]'%(xminfit,xmaxfit,yminfit,ymaxfit)

        # check for boundary issues
        #   row+4*PetroR90, col+4*PetroR90
        #   row-4*PetroR90, col-4*PetroR90

        iraf.imgets(image=working_dir+sdss_drC_name_r,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=working_dir+sdss_drC_name_r,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)

        if (xmaxfit > image_xmax):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f) \n'%(image_xmax,xmaxfit)
            print self.prefix, self.n.NSAID[i], ': setting xmaxfit to max x of image'
            xmaxfit=image_xmax

        if (ymaxfit > image_ymax):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f) \n'%(image_ymax,ymaxfit)
            print self.prefix, self.n.NSAID[i], ': setting ymaxfit to max y of image'
            ymaxfit=image_ymax

        if (xminfit < 1):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f) \n'%(image_xmax,xminfit)
            print self.prefix, self.n.NSAID[i], ': setting xminfit to 1'
            xminfit=1

        if (yminfit < 1):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f) \n'%(image_xmax,yminfit)
            print self.prefix, self.n.NSAID[i], ': setting yminfit to 1'
            yminfit=1

        self.xcenter_cutout[i]=(xcenter - xminfit +1)
        self.ycenter_cutout[i]=(ycenter - yminfit +1)



        # gather image parameters needed for galfit input
        #   approx sky value from fpC image header, keyword = SKY

        input_image=bias_sub_image
        sigma_image='none'
        psf_image=sky_sub_psf_image
        psf_oversampling=1
        # mask_image=mask_image
        convolution_size=100
        print '\n flux20, exptime, magzp = ',flux20, exptime, magzp,'\n'
        pscale=sdsspixelscale   # arcsec/pix
        xobj =xcenter
        yobj =ycenter
        mag_total =self.sdss_r[i]
        Re =self.n.SERSIC_TH50[i]/sdsspixelscale
        sersic_exp =1
        sersic_bulge=4
        axis_ratio =self.n.SERSIC_BA[i]
        PA =self.n.SERSIC_PHI[i]
        rad=self.n.PETROTH50[i]/sdsspixelscale
        if self.n.PETROTH50[i] > max_cutout_radius:
            rad=max_cutout_radius/sdsspixelscale/2.

        # run galfit 3 times
        galname=working_dir+self.prefix+'-'+str(self.n.NSAID[i])
        self.galflag[i,0],self.galflag[i,1],self.galflag[i,2],quitflag,self.galflag_stop[i],self.galflag_too_faint[i]=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky)


        # check for other galaxies on this image
        nearby_gal = sqrt(1.*(run-self.sdssrun)**2+(rerun-self.sdssrerun)**2+(field-self.sdssfield)**2+(camcol-self.sdsscamcol)**2)
        print 'SDSS: number of galaxies to analyze in this image = ',len(nearby_gal[nearby_gal < 1.])
        return quitflag


    def get_imagesNSA(self,i,getimages=0, keepimages=1):
        # getimages:
        # set = 0 to not get images (if they already exist)
        # set = 1 if sdss images are needed
        #
        # this subroutine gets sdss images,
        #    parses relevant input for galfit,
        #    and runs galfit three times.
        #
        # band = 0 for sdss
        # band = 1 for 2um
        #
        # keepimages = 1 to keep sdss DAS images
        # keepimages = 0 to delete sdss DAS images (better if disk space is tight)

        getimages=0
        # open ds9 display
        d=ds9.ds9()
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        d.set('cd '+os.getcwd())
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')
        print i
            
        row=self.n.YCEN[i] 

        col=self.n.XCEN[i]



        local_filename_child=str(self.n.IAUNAME[i])+'-'+str(self.n.PID[i])+'-atlas-'+str(self.n.AID[i])+'.fits'
        nsa_parent=str(self.n.IAUNAME[i])+'-parent-'+str(self.n.PID[i])+'.fits'
        nsa_ivar=str(self.n.IAUNAME[i])+'-ivar-'+str(self.n.PID[i])+'.fits'

        # extract r-band images from multi-extension images
        nsa_parent_rband=self.prefix+'-'+str(self.n.NSAID[i])+'-parent-r.fits'
        if os.path.exists(nsa_parent_rband):
            os.remove(nsa_parent_rband)
        iraf.imcopy(nsa_parent+'[2]',nsa_parent_rband)

        nsa_ivar_rband=self.prefix+'-'+str(self.n.NSAID[i])+'-ivar-r.fits'
        if os.path.exists(nsa_ivar_rband):
            os.remove(nsa_ivar_rband)
        iraf.imcopy(nsa_ivar+'[2]',nsa_ivar_rband)

        nsa_psf=str(self.n.IAUNAME[i])+'-r-bpsf.fits'   


        magzp=22.5
        sky=0

        # use NSA psf file
        psf_image=nsa_psf
        # convert sdssPetroR90r to pixels using sdss plate scale (0.396127"/pix)
        # if R90 < 10 arcsec, use min cutout size of 100 arcsec instead of R90
        if self.n.PETROTH90[i] > min_cutout_radius:
            PETROTH90_pixels =self.n.PETROTH90[i]/sdsspixelscale
            if self.n.PETROTH90[i] > (max_cutout_radius):
                PETROTH90_pixels =max_cutout_radius/sdsspixelscale
        else:
            PETROTH90_pixels =min_cutout_radius/sdsspixelscale
        sex_image=nsa_parent_rband
        # left off here - need to create noise image from ivar
        image=pyfits.open(nsa_ivar_rband)
        image_data=(image[0].data)
        image.close()
        # set any negative values to zero
        image_data[np.where(image_data < 0)]=0
        sigma_data=sqrt(1./(image_data + (image_data == 0))*(image_data != 0))
        # write out sigma pixel values to a fits image
        hdu=pyfits.PrimaryHDU(sigma_data)
        sigma_image=str(self.n.IAUNAME[i])+'-sigma-'+str(self.n.PID[i])+'.fits'
        if os.path.exists(sigma_image):
            os.remove(sigma_image)
        hdu.writeto(sigma_image)

        

        # run sextractor to generate a list of objects in the image
        # generate 'segmentation image'
        os.system('cp ~/research/LocalClusters/sextractor/default.sex.sdss.galfit .')
        os.system('cp ~/research/LocalClusters/sextractor/default.param .')
        os.system('cp ~/research/LocalClusters/sextractor/default.conv .')
        os.system('cp ~/research/LocalClusters/sextractor/default.nnw .')
        os.system('sex %s -c default.sex.sdss.galfit'%(sex_image))
        # convert segmentation image to object mask by replacing the object ID of target galaxy with zeros
        #   parse sextractor output to get x,y coords of objects        
        sexout=atpy.Table(working_dir+'test.cat',type='ascii')
        sexnumber=sexout['col1']
        xsex=sexout['col2']
        ysex=sexout['col3']
        dist=sqrt((row-ysex)**2+(col-xsex)**2)

        #   find object ID

        objIndex=where(dist == min(dist))
        objNumber=sexnumber[objIndex]
        objNumber=objNumber[0] # not sure why above line returns a list
        print 'object number = ',objNumber
        mask_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-mask.fits'
        if os.path.exists(mask_image):
            os.remove(mask_image)
        iraf.imcopy(working_dir+'segmentation.fits',working_dir+mask_image)

        #   use iraf imreplace to replace object ID values with zero
        iraf.imreplace(working_dir+mask_image,value=0,lower=objNumber-.5,upper=objNumber+.5)

        # get best x,y positions from sextractor
        #    - subtract 1 b/c sextractor catalog starts at one, but python is zero-indexed
        print "row, col, xsex, ysex = ",row, col, xsex[objIndex],ysex[objIndex]
        xcenter=xsex[objIndex]
        ycenter=ysex[objIndex]
        # define fit area in terms of the petrosian R90
        xminfit=xcenter-mult_petro90*PETROTH90_pixels
        xmaxfit=xcenter+mult_petro90*PETROTH90_pixels 
        yminfit=ycenter-mult_petro90*PETROTH90_pixels 
        ymaxfit=ycenter+mult_petro90*PETROTH90_pixels 


        print 'xcenter, ycenter = ',xcenter,ycenter
        print 'petrosian R90 in pixels = ',PETROTH90_pixels
        print 'fitting region = [%5.2f, %5.2f, %5.2f, %5.2f]'%(xminfit,xmaxfit,yminfit,ymaxfit)

        # check for boundary issues
        #   row+4*PetroR90, col+4*PetroR90
        #   row-4*PetroR90, col-4*PetroR90

        iraf.imgets(image=working_dir+sex_image,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=working_dir+sex_image,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)

        if (xmaxfit > image_xmax):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f) \n'%(image_xmax,xmaxfit)
            print self.prefix, self.n.NSAID[i], ': setting xmaxfit to max x of image'
            xmaxfit=image_xmax

        if (ymaxfit > image_ymax):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f) \n'%(image_ymax,ymaxfit)
            print self.prefix, self.n.NSAID[i], ': setting ymaxfit to max y of image'
            ymaxfit=image_ymax

        if (xminfit < 1):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f) \n'%(image_xmax,xminfit)
            print self.prefix, self.n.NSAID[i], ': setting xminfit to 1'
            xminfit=1

        if (yminfit < 1):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f) \n'%(image_xmax,yminfit)
            print self.prefix, self.n.NSAID[i], ': setting yminfit to 1'
            yminfit=1

        self.xcenter_cutout[i]=(xcenter - xminfit +1)
        self.ycenter_cutout[i]=(ycenter - yminfit +1)



        input_image=sex_image
        sigma_image=sigma_image
        psf_image=nsa_psf
        psf_oversampling=1
        # mask_image=mask_image
        convolution_size=100
        pscale=sdsspixelscale   # arcsec/pix
        xobj =xcenter
        yobj =ycenter
        mag_total =self.sdss_r[i]
        Re =self.n.SERSIC_TH50[i]/sdsspixelscale
        sersic_exp =1
        sersic_bulge=4
        axis_ratio =self.n.SERSIC_BA[i]
        PA =self.n.SERSIC_PHI[i]
        rad=self.n.PETROTH50[i]/sdsspixelscale
        if rad < 1:
            if Re > 5:
                rad=Re
        sky=0
        
        # should update to fit sersic profiles to any other sources w/in 4*petro_r90

        # run galfit 3 times
        galname=working_dir+self.prefix+'-'+str(self.n.NSAID[i])
        self.galflag[i,0],self.galflag[i,1],self.galflag[i,2],quitflag,self.galflag_stop[i],self.galflag_too_faint[i]=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,convolutionflag=1)


        # check for other galaxies on this image
        #nearby_gal = sqrt(1.*(run-self.sdssrun)**2+(rerun-self.sdssrerun)**2+(field-self.sdssfield)**2+(camcol-self.sdsscamcol)**2)
        #print 'SDSS: number of galaxies to analyze in this image = ',len(nearby_gal[nearby_gal < 1.])
        return quitflag

    def run_just_get_images(self):
        j=0
        for i in range(len(self.ra)):
            print i,self.On24ImageFlag[i],j
            if self.On24ImageFlag[i]:
                j += 1
                self.just_get_images(i)

    def just_get_images(self,i):
        # getimages:
        # set = 0 to not get images (if they already exist)
        # set = 1 if sdss images are needed
        #
        # this subroutine gets sdss images,
        #    parses relevant input for galfit,
        #    and runs galfit three times.
        #
        # band = 0 for sdss
        # band = 1 for 2um
        #
        # keepimages = 1 to keep sdss DAS images
        # keepimages = 0 to delete sdss DAS images (better if disk space is tight)
        getimages=1
        # open ds9 display
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)

        ######################
        #
        # Get SDSS Images
        #
        ######################
            
        field=self.sdss_field[i]
        run=self.sdss_run[i]
        rerun=self.sdss_rerun[i]
        camcol=self.sdss_camcol[i]
        row=self.sdss_rowc[i]
        #row=row[2]
        col=self.sdss_colc[i]
        #col=col[2]
        print 'row, col = ',row, col
        tmp_run='%06d'%(run)
        tmp_field='%04d'%(field)
        sdss_path_c = "imaging/"+str(run)+"/"+str(rerun)+"/corr/"+str(camcol)+"/"
        sdss_corr_name_g = "fpC-"+tmp_run+"-g"+str(camcol)+"-"+tmp_field+".fit"
        sdss_corr_name_r = "fpC-"+tmp_run+"-r"+str(camcol)+"-"+tmp_field+".fit"
        url="http://das.sdss.org/"+sdss_path_c+sdss_corr_name_g+'.gz'

        # updating program to use drC file instead of fpC image
        # looks like rowc and colc are the positions on the drC file
        #
        #s='wget --no-http-keep-alive -S http://das.sdss.org/'+sdss_path_c+sdss_corr_name_r+'.gz'
        #print s
        #print url
        #if getimages:
        #    urllib.urlretrieve(url,filename=sdss_corr_name_r+'.gz')
        #    os.system('gunzip %s'%(sdss_corr_name_r+'.gz'))

        # allsky=tsfield.sky[0]
        # sky=allsky[2] - this is in maggies/arcsec^2
        # get drC file -
        #   The DAS can generate "header supplemented" versions of the fpC files, called drC files.
        #   These are the same as the fpC files, except have newly generated header values for a number
        #   of parameters, including FLUX20 and SKY, derived from the data in the tsField file. (The
        #   DAS generates these "on the fly" from the fpC and tsField files on disk, so you need to use
        #   the http interface rather than the rsync interface to get them.)

        #   e.g. http://das.sdss.org/www/cgi-bin/drC?RUN=3805&RERUN=41&CAMCOL=3&FIELD=96&FILTER=r

        #iraf.imgets(image=sdss_corr_name_r,param='SKY')#get RA of image
	#sky=float(iraf.imgets.value)
        sdss_drC_name_r = "drC-"+tmp_run+"-r"+str(camcol)+"-"+str(rerun)+"-"+tmp_field+".fits"
        drCurl="http://das.sdss.org/www/cgi-bin/drC?RUN=%s&RERUN=%s&CAMCOL=%s&FIELD=%s&FILTER=r"%(str(run),str(rerun),str(camcol),str(field))
        print drCurl
        if getimages:
            urllib.urlretrieve(drCurl,filename=sdss_drC_name_r)
        #   photometric zeropoint from fpC image header, keyword = FLUX20, then convert to zp

        # get psField file to use when reconstructing the PSF
        sdss_path_o = "imaging/"+str(run)+"/"+str(rerun)+"/objcs/"+str(camcol)+"/"
        sdss_ps_name_r = "psField-"+tmp_run+"-"+str(camcol)+"-"+tmp_field+".fit"

        psurl="http://das.sdss.org/"+sdss_path_o+sdss_ps_name_r
        print psurl
        if getimages:
            urllib.urlretrieve(psurl,filename=sdss_ps_name_r)
        #http://das.sdss.org/imaging/3805/41/objcs/3/psField-003805-3-0096.fit

        # get image parameters (GAIN, dark_variance) from tsfield file
            
        sdss_path_calib = "imaging/"+str(run)+"/"+str(rerun)+"/calibChunks/"+str(camcol)+"/"
        sdss_ts_name_r = "tsField-"+tmp_run+"-"+str(camcol)+"-"+str(rerun)+"-"+tmp_field+".fit"
        tsurl="http://das.sdss.org/"+sdss_path_calib+sdss_ts_name_r
        print tsurl
        if getimages:
            urllib.urlretrieve(tsurl,filename=sdss_ts_name_r)

        ######################
        #
        # End of getting SDSS Images
        #
        ######################

    def run_just_get_imagesNSA(self,start_index=0):
        # provide an index if you want to restart
        j=0
        for i in range(start_index,len(self.ra)):
            print i,self.On24ImageFlag[i],j
            if self.On24ImageFlag[i]:
                j += 1
                self.just_get_imagesNSA(i)


    def just_get_imagesNSA(self,i):
        getimages=1
        # open ds9 display
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)

        ######################
        #
        # Get SDSS Images
        #
        ######################
            
        row=self.sdss_rowc[i]
        #row=row[2]
        col=self.sdss_colc[i]
        #col=col[2]
        print 'row, col = ',row, col
        url_parent="http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/"+str(self.n.SUBDIR[i])+'/atlases/'+str(self.n.PID[i])+'/'+str(self.n.IAUNAME[i])+'-parent-'+str(self.n.PID[i])+'.fits.gz'
        url_child="http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/"+str(self.n.SUBDIR[i])+'/atlases/'+str(self.n.PID[i])+'/'+str(self.n.IAUNAME[i])+'-'+str(self.n.PID[i])+'-atlas-'+str(self.n.AID[i])+'.fits.gz'
        url_psf="http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/"+str(self.n.SUBDIR[i])+'/'+str(self.n.IAUNAME[i])+'-r-bpsf.fits.gz'
        url_ivar="http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/"+str(self.n.SUBDIR[i])+'/atlases/'+str(self.n.PID[i])+'/'+str(self.n.IAUNAME[i])+'-ivar-'+str(self.n.PID[i])+'.fits.gz'
        local_filename_child=str(self.n.IAUNAME[i])+'-'+str(self.n.PID[i])+'-atlas-'+str(self.n.AID[i])+'.fits.gz'
        local_filename_parent=str(self.n.IAUNAME[i])+'-parent-'+str(self.n.PID[i])+'.fits.gz'
        local_filename_ivar=str(self.n.IAUNAME[i])+'-ivar-'+str(self.n.PID[i])+'.fits.gz'
        local_filename_psf=str(self.n.IAUNAME[i])+'-r-bpsf.fits.gz'                              
        if getimages:
            print 'getting ',url_parent
            urllib.urlretrieve(url_parent,filename=local_filename_parent)
            print 'getting ',url_child
            urllib.urlretrieve(url_child,filename=local_filename_child)
            print 'getting ',url_ivar
            urllib.urlretrieve(url_ivar,filename=local_filename_ivar)
            print 'getting ',url_psf
            urllib.urlretrieve(url_psf,filename=local_filename_psf)


        # unzip compressed images
        os.system('gunzip -f *.gz')




    def get_images24(self,i,getimages, keepimages=1,convflag=1,fitBAflag=1,fitPAflag=1):
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
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        d.set('cd '+working_dir)
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')




        ra=self.ra[i]
        dec=self.dec[i]
        s='echo %f %f > '%(ra,dec)
        t=s+working_dir+'incoords'
        os.system(t)
        print 'deleting outcoords if it exists'
        output_coords=working_dir+'outcoords'
        if os.path.exists(output_coords):
            os.remove(output_coords)
        input_coords=working_dir+'incoords'
        print input_coords,os.getcwd()
        iraf.imcoords.wcsctran(image=self.mosaic24,input=input_coords,output=output_coords,inwcs='world',outwcs='logical',verbose='no')
        coordout=atpy.Table(output_coords,type='ascii')
        col=coordout['col1']
        row=coordout['col2']
        # convert arrays to numbers
        col=col[0]
        row=row[0]


        sex_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout24.fits'
        unc_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout-unc24.fits'

        # commenting the following block b/c just_get_images24 now makes the images



        try:
            iraf.imgets(image=sex_image,param='naxis1')#get RA of image
            xmaxfit=float(iraf.imgets.value)
            iraf.imgets(image=sex_image,param='naxis2')#get RA of image
            ymaxfit=float(iraf.imgets.value)
        except:
            iraf.imgets(image=sex_image+'[0]',param='naxis1')#get RA of image
            xmaxfit=float(iraf.imgets.value)
            iraf.imgets(image=sex_image+'[0]',param='naxis2')#get RA of image
            ymaxfit=float(iraf.imgets.value)
            print xmaxfit,ymaxfit

        xminfit=1
        yminfit=1


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
        xcenter=1.*(xmaxfit)/2.
        ycenter=1.*(ymaxfit)/2.
        print 'xcenter, ycenter = ',xcenter,ycenter
        dist=sqrt((ycenter-ysex)**2+(xcenter-xsex)**2)

        #   find object ID
        objIndex=where(dist == min(dist))
        print 'objIndex = ',objIndex, dist
        objNumber=sexnumber[objIndex]
        objNumber=objNumber[0] # not sure why above line returns a list
        print 'object number = ',objNumber
        mask_image=working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-mask24.fits'
        #if os.path.exists(mask_image):
        #    iraf.imdel(mask_image)
        #iraf.imcopy('segmentation.fits',mask_image)

        #   use iraf imexam to replace object ID values with zero
        #iraf.imreplace(mask_image,value=0,lower=objNumber-.5,upper=objNumber+.5)

        # get best x,y positions from sextractor
        #    - subtract 1 b/c sextractor catalog starts at one, but python is zero-indexed
        #xcenter=xsex[objIndex]
        #ycenter=ysex[objIndex]

        self.xcenter_cutout24[i]=(xcenter)
        self.ycenter_cutout24[i]=(ycenter)


        ##############################################
        # GATHER GALFIT INPUT
        ##############################################
        input_image=sex_image
        sigma_image=unc_image
        psf_image=self.psf_image
        psf_oversampling=self.psf_oversampling


        # mask_image # already defined

        xobj =xcenter
        yobj =ycenter
        mag_total=magzp-2.5*log10(self.f24NSA[i])
        Re =self.n.PETROTH50[i]/mipspixelscale
        if self.n.PETROTH50[i] > max_cutout_radius:
            Re=max_cutout_radius/mipspixelscale/2.
        axis_ratio =self.n.SERSIC_BA[i]
        PA =self.n.SERSIC_PHI[i]
        rad=self.n.SERSIC_TH50[i]/mipspixelscale
        nsersic=self.n.SERSIC_N[i]

        
        convolution_size=100
        if xmaxfit < 100:
            convolution_size=xmaxfit

        magzp=magzp
        pscale=mipspixelscale   # arcsec/pix
	sky=0


        ##############################################
        # DISPLAY IMAGE, NOISE AND MASK
        ##############################################

        d=ds9.ds9()
        
        d.set('frame delete all')
                
        s='file new '+input_image
        d.set(s)
        s='file new '+sigma_image
        d.set(s)
        s='file new '+mask_image
        d.set(s)
        s='file new '+psf_image
        d.set(s)
        d.set('tile')
        for k in range(1,4):
            s='frame '+str(k)
            d.set(s)
            d.set('scale log')
            d.set('zoom to fit')

        print 'elliptical = ',self.ellipticalflag[i]
        print 'spiral = ',self.spiralflag[i]
        print 'agn = ',self.agnflag[i]
        string=raw_input('hit any key when ready to continue (q to quit) \n')
        if string.find('q') > -1:
            quitflag=1
            return


        ##############################################
        # RUN iGALFIT 
        ##############################################
        galname=working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-24'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print galname
        print xminfit,xmaxfit,yminfit,ymaxfit,convolution_size

        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        convflag=0
        
        self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=igalfit(galname+'-noconv',input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,nsersic,axis_ratio,PA,sky,constrflag=0,fixnsersic=0,fitall=0,convolution=convflag,fitPA=fitPAflag,fitBA=fitBAflag)
        
        output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-noconv-1Comp-galfit-out.fits'
        xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24,chi2nu=parse_galfit_1comp(output_image24+'[2]')

        # use fit w/out convolution as starting point for fit w/convolution
        # self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=runigalfit(galname+'-conv',input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,constrflag=0,fixnsersic=0,fitall=0,convolution=convflag)
        convflag=1
        self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=igalfit(galname+'-conv',input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xc[0],yc[0],mag[0],Re[0],Nsersic[0],axis_ratio[0],PA[0],sky[0],nsersic_flag=1,nsersic_value=Nsersic[0],constrflag=0,fixnsersic=0,fitall=0,convolution=convflag,fitPA=fitPAflag,fitBA=fitBAflag)

        ask_flag=1
        fit_2comp_flag=0
        while ask_flag:
            t=raw_input('Fit 2 Component Model? y/n \n')
            if t.find('y') > -1:
                ask_flag=0
                fit_2comp_flag=1
            elif t.find('n') > -1:
                ask_flag=0
        if fit_2comp_flag:
            # run 2 comp fit
            set_ntimes(-2) 

#        convflag=0
#        self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=runigalfit(galname+'-noconv',input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,constrflag=0,fixnsersic=0,fitall=0,convolution=convflag)
        
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-conv-1Comp-galfit-out.fits'
            xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24,chi2nu=parse_galfit_1comp(output_image24+'[2]')

            convflag=1
            self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=runigalfit(galname+'-conv',input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xc[0],yc[0],mag[0],Re[0],axis_ratio[0],PA[0],sky[0],nsersic_flag=1,nsersic_value=Nsersic[0],constrflag=0,fixnsersic=0,fitall=0,convolution=convflag)
            set_ntimes(1)


        #check sersic index
        #if greater than 5, run again holding n fixed and equal to 5

        #print '***********************************************'
        #print 'Checking sersic index for ', galname
        #output_image=galname+'-conv-1Comp-galfit-out.fits'
        #txc,tyc,tmag,tRe,Nsersic,taxis_ratio,tPA,tsky,tnumerical_error_flag24,tchi2nu=parse_galfit_1comp(output_image)
        #print 'Nsersic = ',Nsersic
        #if float(Nsersic[0]) > 6:
        #    print 'rerunning galfit with index held fixed at 5'
        #    self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=runigalfit(galname+'-conv',input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xc[0],yc[0],mag[0],Re[0],axis_ratio[0],PA[0],sky[0],nsersic_flag=1,nsersic_value=Nsersic[0],constrflag=0,fixnsersic=1,fixnsersicvalue=6,fitall=0,convolution=convflag)
        return quitflag


    def keep_BA_fixed(self,i,fitBAflag=0,fitPAflag=0,convflag=1):
        quitflag=0
        print '###############'
        print 'just inside keep_BA_fixed'
        print 'fitBAflag, fitPAflag, convflag = ',fitBAflag, fitPAflag, convflag
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
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        d.set('cd '+working_dir)
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')




        ra=self.ra[i]
        dec=self.dec[i]
        s='echo %f %f > '%(ra,dec)
        t=s+working_dir+'incoords'
        os.system(t)
        print 'deleting outcoords if it exists'
        output_coords=working_dir+'outcoords'
        if os.path.exists(output_coords):
            os.remove(output_coords)
        input_coords=working_dir+'incoords'
        print input_coords,os.getcwd()
        iraf.imcoords.wcsctran(image=self.mosaic24,input=input_coords,output=output_coords,inwcs='world',outwcs='logical',verbose='no')
        coordout=atpy.Table(output_coords,type='ascii')
        col=coordout['col1']
        row=coordout['col2']
        # convert arrays to numbers
        col=col[0]
        row=row[0]


        sex_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout24.fits'
        unc_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout-unc24.fits'

        try:
            iraf.imgets(image=sex_image,param='naxis1')#get RA of image
            xmaxfit=float(iraf.imgets.value)
            iraf.imgets(image=sex_image,param='naxis2')#get RA of image
            ymaxfit=float(iraf.imgets.value)
        except:
            iraf.imgets(image=sex_image+'[0]',param='naxis1')#get RA of image
            xmaxfit=float(iraf.imgets.value)
            iraf.imgets(image=sex_image+'[0]',param='naxis2')#get RA of image
            ymaxfit=float(iraf.imgets.value)
            print xmaxfit,ymaxfit

        xminfit=1
        yminfit=1


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
        xcenter=1.*(xmaxfit)/2.
        ycenter=1.*(ymaxfit)/2.
        print 'xcenter, ycenter = ',xcenter,ycenter
        dist=sqrt((ycenter-ysex)**2+(xcenter-xsex)**2)

        #   find object ID
        objIndex=where(dist == min(dist))
        print 'objIndex = ',objIndex, dist
        objNumber=sexnumber[objIndex]
        objNumber=objNumber[0] # not sure why above line returns a list
        print 'object number = ',objNumber
        mask_image=working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-mask24.fits'

        self.xcenter_cutout24[i]=(xcenter)
        self.ycenter_cutout24[i]=(ycenter)


        ##############################################
        # GATHER GALFIT INPUT
        ##############################################
        input_image=sex_image
        sigma_image=unc_image
        psf_image=self.psf_image
        psf_oversampling=self.psf_oversampling


        # mask_image # already defined

        xobj =xcenter
        yobj =ycenter
        mag_total=magzp-2.5*log10(self.f24NSA[i])
        Re =self.n.PETROTH50[i]/mipspixelscale
        if self.n.PETROTH50[i] > max_cutout_radius:
            Re=max_cutout_radius/mipspixelscale/2.
        axis_ratio =self.n.SERSIC_BA[i]
        PA =self.n.SERSIC_PHI[i]
        rad=self.n.SERSIC_TH50[i]/mipspixelscale
        nsersic=self.n.SERSIC_N[i]

        
        convolution_size=100
        if xmaxfit < 100:
            convolution_size=xmaxfit

        magzp=magzp
        pscale=mipspixelscale   # arcsec/pix
	sky=0


        ##############################################
        # DISPLAY IMAGE, NOISE AND MASK
        ##############################################

        d=ds9.ds9()
        
        d.set('frame delete all')
                
        s='file new '+input_image
        d.set(s)
        s='file new '+sigma_image
        d.set(s)
        s='file new '+mask_image
        d.set(s)
        s='file new '+psf_image
        d.set(s)
        d.set('tile')
        for k in range(1,4):
            s='frame '+str(k)
            d.set(s)
            d.set('scale log')
            d.set('zoom to fit')

        print 'elliptical = ',self.ellipticalflag[i]
        print 'spiral = ',self.spiralflag[i]
        print 'agn = ',self.agnflag[i]
        string=raw_input('hit any key when ready to continue (q to quit) \n')
        if string.find('q') > -1:
            quitflag=1
            return


        ##############################################
        # RUN iGALFIT 
        ##############################################
        galname=working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-24'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print galname
        print xminfit,xmaxfit,yminfit,ymaxfit,convolution_size

        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        #convflag=1
        if (convflag & fitBAflag):
            outimage=galname+'-conv'
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-conv-1Comp-galfit-out.fits'
        elif (convflag & ~fitBAflag):
            outimage=galname+'-fixedBA'
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'
        elif (~convflag & fitBAflag):
            outimage=galname+'-noconv'
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'
        elif (~convflag & ~fitBAflag):
            outimage=galname+'-fixedBA-noconv'
            output_image24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'+self.prefix+'-'+str(self.n.NSAID[i])+'-24-fixedBA-noconv-1Comp-galfit-out.fits'


        print 'just before calling igalfit'
        print 'fitBA = ',fitBAflag
        self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=igalfit(outimage,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp+self.zp24offset,pscale,xobj,yobj,mag_total,rad,nsersic,axis_ratio,PA,sky,constrflag=0,fixnsersic=0,fitall=0,convolution=convflag,fitPA=fitPAflag,fitBA=fitBAflag)
        

        xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky,numerical_error_flag24,chi2nu=parse_galfit_1comp(output_image24+'[2]')
        #ask_flag=1
        #fit_2comp_flag=0
        #while ask_flag:
        #    t=raw_input('Fit 2 Component Model? y/n \n')
        #    if t.find('y') > -1:
        #        ask_flag=0
        #        fit_2comp_flag=1
        #    elif t.find('n') > -1:
        #        ask_flag=0
        return quitflag


    def just_get_images24(self,i, keepimages=1,make_mask_flag=0,review_mask_flag=0, set_size=None):
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
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        d.set('cd '+working_dir)
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')

        magzp=magzp+5.

        if args.web:
            webbrowser.open('http://www.nsatlas.org/getAtlas.html?search=nsaid&nsaID='+str(self.n.NSAID[i])+'&submit_form=Submit',new=new)
        # check to see if output images exist
        if os.path.exists(working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-mask24.fits') & os.path.exists(working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout24.fits'):
            print '\n image and mask already exist \n'
            print '   galaxy zoo p_cs = ',self.zoo.p_cs[i]
            print i, working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout24.fits \n'
            flag=str(raw_input('Make cutout and mask again?  y (or any key)=yes n=no \n'))
            if flag.find('n') > -1:
                print 'I think you said no'
                return
        print 'making mask and cutout - hope that is what you want!'

        ra=self.ra[i]
        dec=self.dec[i]
        s='echo %f %f > '%(ra,dec)
        t=s+working_dir+'incoords'
        os.system(t)
        print 'deleting outcoords if it exists'
        output_coords=working_dir+'outcoords'
        if os.path.exists(output_coords):
            os.remove(output_coords)
        input_coords=working_dir+'incoords'
        #print input_coords,os.getcwd()
        iraf.imcoords.wcsctran(image=self.mosaic24,input=input_coords,output=output_coords,inwcs='world',outwcs='logical',verbose='no')
        coordout=atpy.Table(output_coords,type='ascii')
        col=coordout['col1']
        row=coordout['col2']
        # convert arrays to numbers
        col=col[0]
        row=row[0]


        ##############################################
        # GET CUTOUT OF GALAXY AND UNCERTAINTY IMAGE
        ##############################################
        radius=self.n.PETROTH90[i]
        #radius=2.*self.n.SERSIC_TH50[i]
        print '\n PETROTH90, min rad, max rad = ',self.n.PETROTH90[i],min_cutout_radius,max_cutout_radius
        if radius > min_cutout_radius:
            PETROTH90_pixels =radius/mipspixelscale
            if radius > (max_cutout_radius):
                print '\n scaling back cutout radius from', self.n.PETROTH90[i],' to ',max_cutout_radius,'\n'
                PETROTH90_pixels =max_cutout_radius/mipspixelscale

        else:
            PETROTH90_pixels =(min_cutout_radius)/mipspixelscale

        # make a cutout of the galaxy (easier than trying to run sextractor on full mosaic
        # need x and y pixel values of the galaxy
        if set_size == None:
            print 'using petro 90 to set cutout size'
            xmin=col-mult_petro90*PETROTH90_pixels
            xmax=col+mult_petro90*PETROTH90_pixels
            ymin=row-mult_petro90*PETROTH90_pixels
            ymax=row+mult_petro90*PETROTH90_pixels

        else:
            print 'setting size manually'
            xmin=col-set_size/mipspixelscale
            xmax=col+set_size/mipspixelscale
            ymin=row-set_size/mipspixelscale
            ymax=row+set_size/mipspixelscale

        # get image dimensions of 24um mosaic

        iraf.imgets(image=self.mosaic24,param='naxis1')#get RA of image
        image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=self.mosaic24,param='naxis2')#get RA of image
        image_ymax=float(iraf.imgets.value)
        
        # check that cutout region is not outside bounds of image

        if (xmax > image_xmax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_xmax,xmax)
            print self.prefix, self.n.NSAID[i], ': setting xmax to max x of image'
            xmax=image_xmax

        if (ymax > image_ymax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_ymax,ymax)
            print self.prefix, self.n.NSAID[i], ': setting ymax to max y of image'
            ymax=image_ymax

        if (xmin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f)'%(image_xmax,xmin)
            print self.prefix, self.n.NSAID[i], ': setting xmin to 1'
            xmin=1

        if (ymin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f)'%(image_xmax,ymin)
            print self.prefix, self.n.NSAID[i], ': setting ymin to 1'
            ymin=1
        
        sex_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout24.fits'
        unc_image=self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-cutout-unc24.fits'

        if os.path.isfile(sex_image):
            os.remove(sex_image)
        if os.path.isfile(unc_image):
            os.remove(unc_image)
        s=self.mosaic24+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
        iraf.imcopy(s,working_dir+sex_image)
        s=self.mosaic24unc+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
        iraf.imcopy(s,working_dir+unc_image)


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
            print working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-mask24.fits'
            mask_image=working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-'+'galfit-mask24.fits'

            print mask_image
            if os.path.exists(mask_image):
                iraf.imdel(mask_image)
            try:
                iraf.imcopy('segmentation.fits',mask_image)
            except:
                iraf.imcopy('segmentation.fits[1]',mask_image)


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

        self.xcenter_cutout24[i]=(xcenter)
        self.ycenter_cutout24[i]=(ycenter)

        normalize_mask(mask_image)
        return quitflag


    def rungalfit_first_time_sdss(self):
        # to run galfit once for all galaxies
        # assumes that you have already run JustGetImages()
        global interactive_flag
        interactive_flag=0
        set_ntimes(1)
        # ellipse gets confused when switching directories, so it's best to run all sdss first,
        # then switch directories and run 24um
        self.rungalfit(getimages=1,sdssflag=1,mipsflag=0,continue_flag=0)
    def rungalfit_first_time_nsa(self,startindex=0):
        # to run galfit once for all galaxies
        # assumes that you have already run self.run_just_get_imagesNSA()
        global interactive_flag
        interactive_flag=0
        set_ntimes(1)
        # ellipse gets confused when switching directories, so it's best to run all sdss first,
        # then switch directories and run 24um
        self.rungalfit(getimages=1,sdssflag=1,mipsflag=0,continue_flag=0,use_nsa=1,start_index=startindex)

    def rungalfit_second_time_nsa(self,startindex=0):
        # to run galfit once for all galaxies
        # assumes that you have already run self.run_just_get_imagesNSA()
        global interactive_flag
        interactive_flag=0
        set_ntimes(-2)
        # ellipse gets confused when switching directories, so it's best to run all sdss first,
        # then switch directories and run 24um
        self.rungalfit(getimages=1,sdssflag=1,mipsflag=0,continue_flag=0,use_nsa=1,start_index=startindex)
       
    def rungalfit_second_time_sdss(self):
        # to run galfit once for all galaxies
        # assumes that you have already run JustGetImages()
        global interactive_flag
        interactive_flag=1
        set_ntimes(-2)
        # ellipse gets confused when switching directories, so it's best to run all sdss first,
        # then switch directories and run 24um
        self.rungalfit(getimages=0,sdssflag=1,mipsflag=0,continue_flag=0)
        
    def rungalfit_first_time_24(self,startindex=0,fitBAflag=True,fitPAflag=True,convflag=True,intflag=True):
        # to run galfit once for all galaxies
        # assumes that you have already run run_just_get_images24()
        global interactive_flag
        interactive_flag=intflag
        set_ntimes(1)
        # link 
        # ellipse gets confused when switching directories, so it's best to run all sdss first,
        # then switch directories and run 24um
        self.rungalfit(getimages=1,sdssflag=0,mipsflag=1,continue_flag=0,use_nsa=0,start_index=startindex,fitBA=fitBAflag,fitPA=fitPAflag,convflag=convflag)

    def runellipse_first_time(self,startindex=0,waveband=1):
        # to run galfit once for all galaxies
        # assumes that you have already run run_just_get_images24()
        global interactive_flag
        interactive_flag=0
        set_ntimes(1)
        # link 
        # ellipse gets confused when switching directories, so it's best to run all sdss first,
        # then switch directories and run 24um
        for i in range(startindex,len(self.n.RA)):
            if self.n.NSAID[i] == 72480:
                continue
            if self.analyze_mips[i]:
                try:
                    quitflag=self.call_measure_disk_useNSA(i,wave_band=waveband,raw_data=1,nsaflag=1)
                except IOError:
                    self.just_get_images24(i,make_mask_flag=1,review_mask_flag=1)
                    quitflag=self.call_measure_disk_useNSA(i,wave_band=waveband,raw_data=1,nsaflag=1)
                if quitflag:
                    return

    def run_just_get_images24(self,startindex=0,make_mask=0,review_mask=0):
        for i in range(startindex,len(self.ra)):
            if self.analyze_mips[i]:
                #run ellipse and measure radius on image (rather than galfit model)
                quitflag=self.just_get_images24(i,review_mask_flag=review_mask,make_mask_flag=make_mask)
                if quitflag:
                    return
        # run rungalfit_first_time_24 next
        # then fit_rawimages_24 
    def fit_rawimages_24(self,startindex=0):
        # to run galfit once for all galaxies
        # assumes that you have already run just_get_images()

        global interactive_flag
        interactive_Flag=1
        
        set_ntimes(1)
        # ellipse gets confused when switching directories, so it's best to run all sdss first,
        # then switch directories and run 24um
        for i in range(startindex,len(self.ra)):
            if self.On24ImageFlag[i]  & (self.snr24[i]> snr24cut):
                #run ellipse and measure radius on image (rather than galfit model)
                self.call_measure_disk(i,wave_band=1,raw_data=1)
                self.measure_radius_rawdata(i,1)

        # next run write_galfit_sersic_parameters()



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


def write_galfit_sersic(outfile,objnumber,profile,xobj,yobj,mag,rad,sersic_exp,axis_ratio,pa,fixsersic=0,asymmetry=0,fitmag=1,fitcenter=1,fitrad=1,fitBA=1,fitPA=1,fitn=1):
    print '##############'
    print 'fit BA'
    print '##############'
    outfile.write(' \n')
    outfile.write('# Object number: %i \n'%(objnumber))
    outfile.write(' 0) %s             # Object type \n'%(profile))
    outfile.write(' 1) %8.1f  %8.1f %i %i  # position x, y        [pixel] \n'%(xobj,yobj,int(fitcenter),int(fitcenter)))
    outfile.write(' 3) %5.2f      %i       # total magnitude     \n'%(mag,fitmag))
    outfile.write(' 4) %8.2f       %i       #     R_e              [Pixels] \n'%(rad,fitrad))
    if fixsersic:
        outfile.write(' 5) %5.2f       0       # Sersic exponent (deVauc=4, expdisk=1)   \n'%(sersic_exp))
    else:
        outfile.write(' 5) %5.2f       %i       # Sersic exponent (deVauc=4, expdisk=1)   \n'%(sersic_exp,int(fitn)))
    outfile.write(' 9) %5.2f       %i       # axis ratio (b/a)    \n'%(axis_ratio,int(fitBA)))
    outfile.write('10) %5.2f       %i       # position angle (PA)  [Degrees: Up=0, Left=90] \n'%(pa,int(fitPA)))
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

def parse_galfit_1comp(galfit_outimage,asymflag=0,ncomp=1):
    numerical_error_flag=0
    if asymflag:
        header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_SKY','1_F1','1_F1PA','CHI2NU']
    else:
        header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_SKY','CHI2NU']
    if ncomp == 2:
        header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_XC','2_YC','2_MAG','2_RE','2_N','2_AR','2_PA','3_SKY','CHI2NU']
    fit_parameters=[]
    working_dir=os.getcwd()+'/'
    for hkey in header_keywords:
        iraf.imgets(image=galfit_outimage,param=hkey)
        s=iraf.imgets.value
        #print hkey,t
        if s.find('[') > -1:
            s=s.replace('[','')
            s=s.replace(']','')
            t=s.split('+/-')
            values=(float(t[0]),0.)# fit and error
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
            except IndexError: # for CHI2NU
                chi2nu=float(t[0])
                continue
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
    fit_parameters.append(chi2nu)
    #print len(fit_parameters),fit_parameters
    return fit_parameters


def parse_galfit_2comp(galfit_outimage):
    header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_XC','2_YC','2_MAG','2_RE','2_N','2_AR','2_PA','3_SKY','CHI2NU']
    fit_parameters=parse_galfit_1comp(galfit_outimage,ncomp=2)
#    fit_parameters=[]
#    numerical_error_flag=0
#    numerical_error_flag2=0
#    working_dir=os.getcwd()+'/'
#    for hkey in header_keywords:
#        iraf.imgets(image=galfit_outimage,param=hkey)
#        t=iraf.imgets.value.split('+/-')
#        try:
#            values=(float(t[0]),float(t[1]))# fit and error
#        except ValueError:
#            # look for * in the string, which indicates numerical problem
#            if t[0].find('*') > -1:
#                if hkey.find('1') > -1:
#                    numerical_error_flag=1
#                if hkey.find('1') > -1:
#                    numerical_error_flag2=1
#            t[0]=t[0].replace('*','')
#            t[1]=t[1].replace('*','')
#            values=(float(t[0]),float(t[1]))# fit and error
#        fit_parameters.append(values)
#    fit_parameters.append(numerical_error_flag)
#    fit_parameters.append(numerical_error_flag2)
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



def rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,convolutionflag=1,simflag=0,NSAsersic=0,nsersic=2,constrflag=1,fixnsersic=0,fixnsersicvalue=5,fitall=0,asymflag=0,fitmag=1,fitcenter=1,fitrad=1,fitBA=1,fitPA=1,fitn=1):
    
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
        sersic_exp =1.5
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
        if simflag:
            galfile=galname+'galfit.sim.input.'+str(j+1)+'Comp'
        else:
            galfile=galname+'galfit.input.'+str(j+1)+'Comp'
        galfit_input=open(galfile,'w')
        
        write_galfit_image_param(galfit_input,input_image,output_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,convflag=convolutionflag,constraintflag=constrflag,fitallflag=fitall)
        
        if j == 0:
            
            objnumber=1
            write_galfit_sersic(galfit_input,objnumber,profile,xobj,yobj,mag_total,rad,sersic_exp,axis_ratio,PA,fixsersic=fixnsersic,asymmetry=asymflag,fitmag=1,fitcenter=1,fitrad=1,fitBA=fitBA,fitPA=fitBA,fitn=1)
            objnumber=2
            write_galfit_sky(galfit_input,objnumber,sky)    
            


        elif j == 1:
            # read in output from round 1
            # get values from header of image[2]
            galfit_outimage=galname+'-'+'1Comp-galfit-out.fits[2]'
            
            (x_fit,y_fit,mag_fit,Re_fit,Nsersic_fit,axis_ratio_fit,pa_fit,sky_fit,numerical_error_flag,chi2nu)=parse_galfit_1comp(galfit_outimage)
            
            
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
                nearbyobjflag=sqrt((se.X_IMAGE-xobj)**2+(se.Y_IMAGE-yobj)**2) > 8.
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
                (x_fit,y_fit,mag_fit,Re_fit,Nsersic_fit,axis_ratio_fit,pa_fit,sky_fit,numerical_error_flag,chi2nu)=parse_galfit_1comp(galfit_outimage)
                #d.set('frame 7')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit[0],Re_fit[0],Nsersic_fit[0],axis_ratio_fit[0],pa_fit[0])
                print s
                #d.set(s)
            if j == 1:
                galfit_outimage=galname+'-'+'2Comp-galfit-out.fits[2]'
                #x_fit1,y_fit1,mag_fit1,Re_fit1,Nsersic_fit1,axis_ratio_fit1,pa_fit1,x_fit2,y_fit2,mag_fit2,Re_fit2,Nsersic_fit2,axis_ratio_fit2,pa_fit2,sky_fit,chi2nu=parse_galfit_2comp(galfit_outimage)
                t=parse_galfit_2comp(galfit_outimage)
                #d.set('frame 7')
                #s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit1[0],Re_fit1[0],Nsersic_fit1[0],axis_ratio_fit1[0],pa_fit1[0])
                print s
                #d.set(s)
                #d.set('frame 8')
                #s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit2[0],Re_fit2[0],Nsersic_fit2[0],axis_ratio_fit2[0],pa_fit2[0])
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
            #s='frame delete '+str(endframe)
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
            return galflag[0],galflag[1],galflag[2],quitflag,stop_gal_flag,too_faint_flag
    return galflag[0],galflag[1],galflag[2],quitflag,stop_gal_flag,too_faint_flag


def runigalfit(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,constrflag=0,fixnsersic=0,fitall=0,convolution=1,nsersic_flag=0,nsersic_value=2,fitmag=1,fitrad=1,fitn=1,fitBA=1,fitPA=1):

    
    (galflag24_0,galflag24_1,galflag24_2,quitflag,galflag_stop24,galflag_too_faint24)=[0,0,0,0,0,0]

    repeat_flag=True
    fitall_value=0
    constr_value=0
    fixnsersic_flag=0
    fixnsersic_value=5
    #nsersic_value=1.5
    #nsersic_flag=0
    xo=xobj
    yo=yobj
    ro=rad
    ao=axis_ratio
    po=PA
    nsersico=nsersic_value
    nfit=1
    magfit=1
    centerfit=1
    radfit=1
    BAfit=0
    PAfit=0
    while repeat_flag:
        #print 'at beginning of loop, repeat_flag = ',repeat_flag
        galflag24_0,galflag24_1,galflag24_2,quitflag,galflag_stop24,galflag_too_faint24=rungalfit_3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,constrflag=constr_value,fixnsersic=fixnsersic_flag,fixnsersicvalue=fixnsersic_value,fitall=fitall_value,convolutionflag=convolution,NSAsersic=nsersic_flag,nsersic=nsersic_value,fitmag=magfit,fitcenter=centerfit,fitrad=radfit,fitBA=BAfit,fitPA=PAfit,fitn=nfit)
        flag=str(raw_input('Are you happy with the fit?  n=no x=quit y (or any other key) = yes \n'))
        flag=str(flag)
        print 'this is what I think you typed ',flag
        #t=raw_input('type anything')
        edit_flag=1
        if flag.find('n') > -1:
            while edit_flag:

                print 'CURRENT INPUTS: \n mag = %5.2f %i \n Re = %5.2f %i \n n = %5.2f %i\n B/A = %5.2f %i \n PA = %5.2f %i \n fitall = %i'%(mag_total,fitmag,rad,fitrad,nsersic_value,fitn,axis_ratio,fitBA,PA,fitBA,fitall_value)
                flag2=str(raw_input('What is wrong?\n n = adjust sersic \n r = reset Re \n o=nearby object (toggle fitall) \n b = B/A \n p = PA \n  m = mag \n c = recenter \n f = hold values fixed \n a = add asymmetry parameter \n R = reset to original values \n g = go (run galfit) \n x=quit \n '))

                if flag2.find('n') > -1:
                    n=float(raw_input('sersic exponent = '))
                    nsersic_flag=1
                    nsersic_value=n

                elif flag2.find('r') > -1:
                    rad=float(raw_input('Re = '))

                elif flag2.find('b') > -1:
                    axis_ratio=float(raw_input('B/A = '))

                elif flag2.find('p') > -1:
                    PA=float(raw_input('PA = '))

                elif flag2.find('m') > -1:
                    mag_total=float(raw_input('mag = '))

                elif flag2.find('o') > -1:
                    if fitall_value == 0:
                        fitall_value = 1
                    elif fitall_value == 1:
                        fitall_value = 0

                elif flag2.find('c') > -1:
                    xobj=float(raw_input('xc = '))
                    yobj=float(raw_input('yc = '))


                elif flag2.find('f') > -1:
                    flag3=g.print_fix_menu()
                    if flag3.find('n') > -1:
                        g.fix_n()
                    elif flag3.find('r') > -1:
                        g.fix_rad()
                    elif flag3.find('p') > -1:
                        g.fix_PA()
                    elif flag3.find('c') > -1:
                        g.fix_center()
                    elif flag3.find('f') > -1:
                        g.add_constraint_file()
                elif flag2.find('R') > -1:
                    xobj=xo
                    yobj=yo
                    PA=po
                    axis_ratio=ao
                    rad=ro
                    nsersic_value=nsersico
                elif flag2.find('g') > -1:
                    edit_flag=0

                
                elif flag2.find('x') > -1:
                    quitflag=0
                    edit_flag=0
                    repeat_flag=False
                #else: 
                #    repeat_flag=False
        elif flag.find('x') > -1:
            print 'i think you want to exit'
            repeat_flag=False
            edit_flag=False
            print 'repeat_flag = ',repeat_flag
        elif flag.find('y') >-1:
            repeat_flag=False
            edit_flag=False
            
        else:
            print 'not sure what you entered, so setting repeat_flag = False'
            repeat_flag=False
            edit_flag=False
        #repeat_flag=False
            print 'at end of loop, repeat_flag = ',repeat_flag
    #repeat_flag=False
    return galflag24_0,galflag24_1,galflag24_2,quitflag,galflag_stop24,galflag_too_faint24

def igalfit(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,nsersic,axis_ratio,PA,sky,constrflag=0,fixnsersic=0,fitall=0,convolution=True,nsersic_flag=0,nsersic_value=2,fitmag=1,fitrad=1,fitn=1,fitBA=1,fitPA=1):


    print '%%%%%%%%%%%%%%%%%'
    print 'inside galfit'
    print xminfit,xmaxfit,yminfit,ymaxfit,convolution_size
    print 'psf_image = ',psf_image
    print '%%%%%%%%%%%%%%%%%'
    print 'conv, fitBA,fitPA = ',convolution,fitBA,fitPA
    (galflag24_0,galflag24_1,galflag24_2,quitflag,galflag_stop24,galflag_too_faint24)=[0,0,0,0,0,0]

    repeat_flag=True
    fitall_value=0
    constr_value=0
    fixnsersic_flag=0
    fixnsersic_value=5
    #nsersic_value=1.5
    #nsersic_flag=0
    xo=xobj
    yo=yobj
    ro=rad
    ao=axis_ratio
    po=PA
    nsersico=nsersic
    nfit=1
    magfit=1
    centerfit=1
    radfit=1
    BAfit=fitBA
    PAfit=fitPA
    g=galfit(galname=galname,image=input_image,sigma_image=sigma_image,psf_image=psf_image,psf_oversampling=psf_oversampling,mask_image=mask_image,xminfit=xminfit,yminfit=yminfit,xmaxfit=xmaxfit,ymaxfit=ymaxfit,convolution_size=convolution_size,magzp=magzp,pscale=pscale,convflag=convolution,constraintflag=constr_value,fitallflag=fitall_value,ncomp=1)
    g.set_sersic_params(xobj=xobj,yobj=yobj,mag=mag_total,rad=rad,nsersic=nsersic,BA=axis_ratio,PA=PA,fitmag=magfit,fitcenter=centerfit,fitrad=radfit,fitBA=BAfit,fitPA=PAfit,fitn=nfit,first_time=1)
    g.set_sky(sky)

    while repeat_flag:
        #print 'at beginning of loop, repeat_flag = ',repeat_flag
        g.run_galfit()
        flag=str(raw_input('Are you happy with the fit?  n=no x=quit y (or any other key) = yes \n'))
        flag=str(flag)
        print 'this is what I think you typed ',flag
        #t=raw_input('type anything')
        edit_flag=1
        if flag.find('n') > -1:
            while edit_flag:
                g.print_params()
                #print 'ellipticalflag = ',self.ellipticalflag[i]
                #print 'CURRENT INPUTS: \n mag = %5.2f %i \n Re = %5.2f %i \n n = %5.2f %i\n B/A = %5.2f %i \n PA = %5.2f %i \n fitall = %i'%(mag_total,fitmag,rad,fitrad,nsersic_value,fitn,axis_ratio,fitBA,PA,fitBA,fitall_value)
                flag2=g.edit_params_menu()

                if flag2.find('n') > -1:
                    g.set_n()

                elif flag2.find('r') > -1:
                    g.set_r()

                elif flag2.find('b') > -1:
                    g.set_BA()

                elif flag2.find('p') > -1:
                    g.set_PA()
                
                elif flag2.find('m') > -1:
                    g.set_mag()
                
                elif flag2.find('o') > -1:
                    g.toggle_fitall()
                elif flag2.find('c') > -1:
                    g.set_center()
                elif flag2.find('a')> -1:
                    g.toggle_asymmetry()
                elif flag2.find('f') > -1:
                    flag3=g.print_fix_menu()
                    if flag3.find('n') > -1:
                        g.fix_n()
                    elif flag3.find('r') > -1:
                        g.fix_rad()
                    elif flag3.find('p') > -1:
                        g.fix_PA()
                    elif flag3.find('b') > -1:
                        g.fix_BA()
                    elif flag3.find('c') > -1:
                        g.fix_center()
                    elif flag3.find('f') > -1:
                        g.add_constraint_file()

                elif flag2.find('R') > -1:
                    g.reset_sersic_params()
                elif flag2.find('g') > -1:
                    edit_flag=0

                elif flag2.find('x') > -1:
                    quitflag=0
                    edit_flag=0
                    repeat_flag=False
                #else: 
                #    repeat_flag=False
        elif flag.find('x') > -1:
            print 'i think you want to exit'
            repeat_flag=False
            edit_flag=False
            quitflag=1
            print 'repeat_flag = ',repeat_flag
        elif flag.find('y') >-1:
            repeat_flag=False
            edit_flag=False
            
        else:
            print 'not sure what you entered, so setting repeat_flag = False'
            repeat_flag=False
            edit_flag=False

    return galflag24_0,galflag24_1,galflag24_2,quitflag,galflag_stop24,galflag_too_faint24

#for cl in mylocalclusters:
#    print cl.prefix,': Number of galaxies to analyze = ',sum(cl.analyze_mips)

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


# here is the calling sequence
# cl.justGetSDSSImages()
# cl.rungalfit() which calls:
#    self.get_images() - this gets psf image and runs galfit035
#    self.display_galfit_results() # user selects which components are assoc w/disk
#    self.call_measure_disk() # runs ellipse fitting
#    self.call_measure_radius() # fits R50 and R90 to profile



# to run galfit second time for SDSS galaxies
# cl.justGetSDSSImages()
# cl.rungalfit(getimages=1,sdssflag=1,mipsflag=0,continue_flag=0,interact=0)

# to run galfit second time for SDSS galaxies
# cl.justGetSDSSImages()
# cl.rungalfit(getimages=1,sdssflag=1,mipsflag=0,continue_flag=0,interact=1)

class galfit:
    def __init__(self,galname=None,image=None,sigma_image=None,psf_image=None,psf_oversampling=None,mask_image=None,xminfit=None,yminfit=None,xmaxfit=None,ymaxfit=None,convolution_size=None,magzp=None,pscale=None,convflag=1,constraintflag=1,fitallflag=0,ncomp=1):
        self.galname=galname
        self.image=image

        self.sigma_image=sigma_image
        self.psf_image=psf_image
        self.psf_oversampling=psf_oversampling
        self.mask_image=mask_image
        self.xminfit=xminfit
        self.yminfit=yminfit
        self.xmaxfit=xmaxfit
        self.ymaxfit=ymaxfit
        self.convolution_size=convolution_size
        self.magzp=magzp
        self.pscale=pscale
        self.convflag=convflag
        self.constraintflag=constraintflag
        self.fitallflag=fitallflag
        self.ncomp=ncomp
        self.asymmetry=0

        print '***%%%%%%%%%%%%%%%%%'
        print 'inside galfit class'
        print xminfit,xmaxfit,yminfit,ymaxfit,convolution_size
        print self.xminfit,self.xmaxfit,self.yminfit,self.ymaxfit,self.convolution_size
        print 'psf_image = ',psf_image
        print 'self.fitall = ',self.fitallflag
        print '***%%%%%%%%%%%%%%%%%'

        

    def create_output_names(self):
        if self.asymmetry:
            output_image=self.galname+'-'+str(self.ncomp)+'Comp-galfit-out-asym.fits'
        else:
            output_image=self.galname+'-'+str(self.ncomp)+'Comp-galfit-out.fits'

        self.output_image=output_image
        # create galfit input file
        self.galfile=self.galname+'galfit.input.'+str(self.ncomp)+'Comp'


    def open_galfit_input(self):
        self.galfit_input=open(self.galfile,'w')


    def write_image_params(self):#,input_image,output_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,convflag=1,constraintflag=1,fitallflag=0):
        self.galfit_input.write('# IMAGE PARAMETERS\n')
        self.galfit_input.write('A) '+self.image+'              # Input data image (FITS file)\n')
        self.galfit_input.write('B) '+self.output_image+'       # Name for the output image\n')
        self.galfit_input.write('C) %s                # Sigma image name (made from data if blank or "none") \n'%(self.sigma_image))
        if self.convflag:
            self.galfit_input.write('D) '+self.psf_image+'     # Input PSF image and (optional) diffusion kernel\n')
            self.galfit_input.write('E) %i                   # PSF oversampling factor relative to data\n'%(self.psf_oversampling))
        if self.fitallflag:
            self.galfit_input.write('F)            # Pixel mask (ASCII file or FITS file with non-0 values)\n')
        else:
            self.galfit_input.write('F) '+self.mask_image+'           # Pixel mask (ASCII file or FITS file with non-0 values)\n')
        if self.constraintflag:
            self.galfit_input.write('G) /Users/rfinn/research/LocalClusters/GalfitAnalysis/sersic.constraint        # Parameter constraint file (ASCII)\n')
        self.galfit_input.write('H) '+str(int(round(self.xminfit)))+' '+str(int(round(self.xmaxfit)))+' '+str(int(round(self.yminfit)))+' '+str(int(round(self.ymaxfit)))+'     # Image region to fit (xmin xmax ymin ymax)\n')
        if self.convflag:
            self.galfit_input.write('I) '+str(int(round(self.convolution_size)))+' '+str(int(round(self.convolution_size)))+'             # Size of convolution box (x y)\n')
        self.galfit_input.write('J) %5.2f              # Magnitude photometric zeropoint \n'%(self.magzp))
        self.galfit_input.write('K) %6.5f   %6.5f         # Plate scale (dx dy)  [arcsec/pix]\n'%(self.pscale,self.pscale))
        self.galfit_input.write('O) regular                # Display type (regular, curses, both)\n')
        self.galfit_input.write('P) 0                   # Create output image only? (1=yes; 0=optimize) \n')
        self.galfit_input.write('S) 0                   # Modify/create objects interactively?\n')


    def set_sersic_params(self,xobj=None,yobj=None,mag=None,rad=None,nsersic=None,BA=None,PA=None,fitmag=1,fitcenter=1,fitrad=1,fitBA=1,fitPA=1,fitn=1,first_time=0):
        self.xobj=xobj
        self.yobj=yobj
        self.mag=mag
        self.rad=rad
        self.nsersic=nsersic
        self.BA=BA
        self.PA=PA
        self.fitmag=fitmag
        self.fitn=fitn
        self.fitcenter=fitcenter
        self.fitrad=fitrad
        self.fitBA=fitBA
        self.fitPA=fitPA


        if first_time:
            self.xobj0=xobj
            self.yobj0=yobj
            self.mag0=mag
            self.rad0=rad
            self.nsersic0=nsersic
            self.BA0=BA
            self.PA0=PA
            self.fitmag0=fitmag
            self.fitn0=fitn
            self.fitcenter0=fitcenter
            self.fitrad0=fitrad
            self.fitBA0=fitBA
            self.fitPA0=fitPA
            self.asymmetry0=self.asymmetry

    def reset_sersic_params(self):
        self.xobj=self.xobj0
        self.yobj=self.yobj0
        self.mag=self.mag0
        self.rad=self.rad0
        self.nsersic=self.nsersic0
        self.BA=self.BA0
        self.PA=self.PA0
        self.fitmag=self.fitmag0
        self.fitn=self.fitn0
        self.fitcenter=self.fitcenter0
        self.fitrad=self.fitrad0
        self.fitBA=self.fitBA0
        self.fitPA=self.fitPA0
        self.asymmetry=self.asymmetry0
        
    def set_sky(self,sky):
        self.sky=sky

    def write_sersic(self,objnumber,profile):
        self.galfit_input.write(' \n')
        self.galfit_input.write('# Object number: %i \n'%(objnumber))
        self.galfit_input.write(' 0) %s             # Object type \n'%(profile))
        self.galfit_input.write(' 1) %8.1f  %8.1f %i %i  # position x, y        [pixel] \n'%(self.xobj,self.yobj,int(self.fitcenter),int(self.fitcenter)))
        self.galfit_input.write(' 3) %5.2f      %i       # total magnitude     \n'%(self.mag,self.fitmag))
        self.galfit_input.write(' 4) %8.2f       %i       #     R_e              [Pixels] \n'%(self.rad,self.fitrad))
        print 'sersic n, fitsersicn = ',self.nsersic,self.fitn
        self.galfit_input.write(' 5) %5.2f       %i       # Sersic exponent (deVauc=4, expdisk=1)   \n'%(self.nsersic,int(self.fitn)))
        print 'BA, fitBA = ',self.BA,self.fitBA
        self.galfit_input.write(' 9) %5.2f       %i       # axis ratio (b/a)    \n'%(self.BA,int(self.fitBA)))
        self.galfit_input.write('10) %5.2f       %i       # position angle (PA)  [Degrees: Up=0, Left=90] \n'%(self.PA,int(self.fitPA)))
        if self.asymmetry:
            self.galfit_input.write('F1) 0.0001 0.00   1  1     # azim. Fourier mode 1, amplitude & phase angle \n')
        self.galfit_input.write(" Z) 0                  # Output option (0 = residual, 1 = Don't subtract)  \n")

    def write_sky(self,objnumber):    
        self.galfit_input.write(' \n')
        self.galfit_input.write('# Object number: %i \n'%(objnumber))
        self.galfit_input.write(' 0) sky             # Object type \n')
        self.galfit_input.write(' 1) %8.1f   1  # sky background at center of fitting region [ADUs] \n'%(self.sky))
        self.galfit_input.write(' 2) 0      0       # dsky/dx (sky gradient in x)    \n')
        self.galfit_input.write(' 3) 0      0       # dsky/dy (sky gradient in y) \n')
        self.galfit_input.write(" Z) 0                  # Output option (0 = residual, 1 = Don't subtract)  \n")


    def run_galfit(self):
        #print 'self.fitall = ',self.fitall
        self.create_output_names()
        self.open_galfit_input()
        self.write_image_params()
        #print 'self.fitall = ',self.fitall
        self.write_sersic(1,'sersic')
        #print 'self.fitall = ',self.fitall
        self.write_sky(2)
        #print 'self.fitall = ',self.fitall
        if (self.fitallflag):
            print '%%%%%%%%%%%%%% HEY %%%%%%%%%%%%%'
            print 'I think fitall is true, just sayin...'
            self.fitall()
        self.close_input_file()
        #print 'self.fitall = ',self.fitall
        s = 'galfit '+self.galfile
        print 'run the following: ',s

        try:
            errno=os.system(s)
            self.galfit_flag=1
        except:
            print "PROBLEM RUNNING GALFIT!!!!"
            self.galfit_flag=0
            return

        image_id=self.galname+'-'
        self.galfit_log=image_id+str(self.ncomp)+'Comp-fit.log'
        s='cp fit.log '+self.galfit_log
        os.system(s)
        self.galfit_out=image_id+str(self.ncomp)+'Comp'+'-galfit.01'
        s='mv galfit.01 '+self.galfit_out
        try:
            os.rename('galfit.01',self.galfit_out)
        except:
            print "appears like galfit did not complete"
            #galflag[j]=0
            self.galfit_flag=0
            return
        self.display_results()

    def display_results(self):
        print '%%%%%%%%%%%%%%%%%%'
        print 'inside display_results'
        print 'self.galfit_flag = ',self.galfit_flag
        if (self.galfit_flag < 0.1):
            print 'GALFIT did not complete - can not display results'
            return

        subcomp_image=self.galname+'-'+str(self.ncomp)+'Comp'+'-subcomps.fits'
        s='galfit -o3 '+self.galfit_out
        os.system(s)
        os.rename('subcomps.fits',subcomp_image)
        #    - display results (like ds9 multiextension data cube -> use xpa)
        #

        try:
            d.set('frame delete all')
        except NameError:
            d=ds9.ds9()
            d.set('frame delete all')
        #print 'file to display = ',self.output_image
        s='file new multiframe '+self.output_image
        #print s

        d.set(s)
        d.set('frame delete 1')
        for k in range(2,5):
            s='frame '+str(k)
            d.set(s)
            d.set('zoom to fit')
            #print k
            if k == 2:
                d.set('regions command {text 30 10 #text="Image" font="times 18 bold" color="red"}')
            if k == 3:
                d.set('regions command {text 30 10 #text="Model" font="times 18 bold" color="red"}')
            if k == 4:
                d.set('regions command {text 30 10 #text="Residual" font="times 18 bold" color="red"}')
        d.set('frame match wcs')

        galfit_outimage=self.galname+'-'+'1Comp-galfit-out.fits[2]'
        self.print_galfit_results(galfit_outimage)
        print 'file to display = ',subcomp_image
        s='file new multiframe '+subcomp_image
        d.set(s)

        if self.ncomp == 1:
            endframe=8
        if self.ncomp == 2:
            endframe=9
        if self.ncomp == 3:
            endframe=10
        s='frame delete '+str(endframe)
        print s
        try:
            d.set(s)
        except ValueError:
            print "couldn't execute the following ds9 command : ",s
        try:
            d.set('frame delete 5')
        except:
            print "couldn't delete frame 5"
        try:
            d.set('frame delete 6')
        except:
            print "couldn't delete frame 6"
        d.set('frame 7')
        d.set('file '+self.mask_image)

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
        #d.set('saveimage png image.png')
        #img_name=self.galname+'-'+str(self.ncomp)+'Comp.png'
        #os.rename('image.png',img_name)


    def fitall(self,mindistance=8):
        os.system('cp '+homedir+'research/LocalClusters/sextractor/default.param .')
        os.system('cp '+homedir+'research/LocalClusters/sextractor/default.nnw .')
        s='sex '+self.image+'[1] -c '+homedir+'research/LocalClusters/sextractor/default.sex.24um.galfitsource -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+self.sigma_image+' -CATALOG_NAME '+self.galname+'test.cat -CATALOG_TYPE ASCII_HEAD'
        os.system(s)
        # read in SE table to get x,y for sources
        #fname=self.galname+'test.fits'
        fname=self.galname+'test.cat'
        print 'FITALL CATALOG NAME = ',fname
        objnumber=2
        profile='sersic'
        try:
            se=atpy.Table(fname,type='ascii')
            print 'found ',len(se.X_IMAGE),' sources on the field of ',self.galname
            nearbyobjflag=sqrt((se.X_IMAGE-self.xobj)**2+(se.Y_IMAGE-self.yobj)**2) > mindistance
            for k in range(len(se.X_IMAGE)):
                if nearbyobjflag[k]:
                    objnumer=objnumber+1
                    self.add_simple_sersic_object(objnumber,profile,se.X_IMAGE[k],se.Y_IMAGE[k],se.MAG_BEST[k],se.FLUX_RADIUS[k,0],2,se.B_IMAGE[k]/se.A_IMAGE[k],se.THETA_IMAGE[k])
        except AttributeError:
            print 'WARNING: no sources detected in image!'
        raw_input=('hit any key to continue \n')

    def add_simple_sersic_object(self,objnumber,profile,x,y,mag,rad,nsersic,BA,PA):
        self.galfit_input.write(' \n')
        self.galfit_input.write('# Object number: %i \n'%(objnumber))
        self.galfit_input.write(' 0) %s             # Object type \n'%(profile))
        self.galfit_input.write(' 1) %8.1f  %8.1f 1 1  # position x, y        [pixel] \n'%(x,y))
        self.galfit_input.write(' 3) %5.2f      1       # total magnitude     \n'%(mag))
        self.galfit_input.write(' 4) %8.2f       1       #     R_e              [Pixels] \n'%(rad))
        self.galfit_input.write(' 5) %5.2f       1       # Sersic exponent (deVauc=4, expdisk=1)   \n'%(nsersic))
        self.galfit_input.write(' 9) %5.2f       1       # axis ratio (b/a)    \n'%(BA))
        self.galfit_input.write('10) %5.2f       1       # position angle (PA)  [Degrees: Up=0, Left=90] \n'%(PA))
        self.galfit_input.write(" Z) 0                  # Output option (0 = residual, 1 = Don't subtract)  \n")
                                

    def close_input_file(self):
        self.galfit_input.close()

    def print_params(self):
        print 'CURRENT INPUTS: \n mag = %5.2f %i \n Re = %5.2f %i \n n = %5.2f %i\n B/A = %5.2f %i \n PA = %5.2f %i \n fitall = %i \n fitcenter = %i \n'%(self.mag,self.fitmag,self.rad,self.fitrad,self.nsersic,self.fitn,self.BA,self.fitBA,self.PA,self.fitPA,self.fitallflag,self.fitcenter)
                                
    def print_galfit_results(self,image):            
        if self.asymmetry:
            header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_SKY','1_F1','1_F1PA','ERROR','CHI2NU']
        else:
            header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_SKY','ERROR','CHI2NU']
        if self.ncomp == 2:
            header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_XC','2_YC','2_MAG','2_RE','2_N','2_AR','2_PA','3_SKY','ERROR','CHI2NU']

        t=parse_galfit_1comp(image)
        for i in range(len(header_keywords)):
            try:
                print '%6s : %5.2f +/- %5.2f'%(header_keywords[i],t[i][0],t[i][1])
            except:
                print '%6s : %5.2f'%(header_keywords[i],t[i])
    def edit_params_menu(self):
        flag=str(raw_input('What is wrong?\n n = adjust sersic \n r = reset Re \n o = nearby object (toggle fitall) \n b = B/A \n p = PA \n m = mag \n c = recenter \n f = hold values fixed \n a = toggle asymmetry parameter \n R = reset to original values \n g = go (run galfit) \n x=quit \n '))
        return flag

    def set_n(self):
        n=float(raw_input('sersic exponent = '))
        self.nsersic=n

    def set_r(self):
        r=float(raw_input('radius = '))
        self.rad=r

    def set_BA(self):
        r=float(raw_input('BA = '))
        self.BA=r

    def set_PA(self):
        r=float(raw_input('PA = '))
        self.PA=r

    def set_mag(self):
        r=float(raw_input('mag = '))
        self.mag=r

    def set_center(self):
        r=float(raw_input('xc = '))
        self.xobj=r
        r=float(raw_input('yc = '))
        self.yobj=r

    def toggle_fitall(self):
        self.fitallflag=toggle(self.fitallflag)

    def toggle_asymmetry(self):
        self.asymmetry=toggle(self.asymmetry)

    def print_fix_menu(self):
        self.print_params()
        flag3=str(raw_input('What do you want to hold fixed/toggle?\n n = fix sersic index \n r = fix Re \n b = fix B/A \n p = PA \n c = center \n f = use constraint file \n R = reset to original values \n g = go (run galfit) \n x=quit \n '))
        return flag3

    def fix_n(self):
        n=float(raw_input('sersic exponent = '))
        self.nsersic=n
        self.fitn=toggle(self.fitn)

    def fix_rad(self):
        self.fitrad=toggle(self.fitrad)

    def fix_BA(self):
        self.fitBA=toggle(self.fitBA)
        print self.fitBA, self.BA
    def fix_PA(self):
        self.fitPA=toggle(self.fitPA)

    def fix_center(self):
        self.fitcenter=toggle(self.fitcenter)

    def add_constraint_file(self):
        self.constraintflag=toggle(self.constraintflag)
        

print 'got to the end!'

if __name__ == '__main__':
    print 'loading clusters if you so desire'
    #mkw11=Cluster('MKW11')
    #mkw8=Cluster('MKW8')
    #awm4=Cluster('AWM4')
    #a2063=Cluster('A2063')
    #a2052=Cluster('A2052')
    #ngc=Cluster('NGC6107')
    #a1367=Cluster('A1367')
    #herc=Cluster('Hercules')
    #coma=Cluster('Coma')
    #mylocalclusters=[mkw11,mkw8,awm4,ngc,a2052,a2063,a1367,herc,coma]
    #for cl in mylocalclusters:
    #    cl.write_galfit_sersic_parameters_24()
