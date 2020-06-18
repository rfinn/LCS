#!/usr/bin/env python
"""
    GOAL
      - make disk images from galfit output of 1, 2, and 3-comp fits
      - measure R50 and R90 for each image 

    PROCEDURE
      if galfit is able to complete the N-component model
      (1) identify disk components
	    display images of components
	    display fit parameters
	    query user to select components
      (2) add disk components together to create cluster-AGC#-diskimage-NCompFit-band.fits
      (3) run ellipse on disk image
      (4) fit curve of growth + find R50, R90
	    determine f_model from galfit mag
	    fit curve of growth and find when F_enclosed = 0.5 * f_model (R50) 
		  			      F_enclosed = 0.9 * f_model (R90)

      (5) create line-matched table that contains

            index On24ImageFlag galFlag[Nx3] sdssR50[Nx3] sdssR90[Nx3] galFlag24[Nx3] mipsR50[Nx3] mipsR90[Nx3]
    
"""

from pylab import *
import os, pyfits,ds9
from LCSReadmasterBase import *
import urllib, atpy
from pyraf import iraf
min_cutout_size=100. # in arcsec
mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

    #getimages=1
class cluster(baseCluster):
    def __init__(self,clustername):
        baseCluster.__init__(self,clustername)
        self.galflag=zeros([len(self.ra),3],'i')
        self.galflag24=zeros([len(self.ra),3],'i')
        self.galflag_too_faint24=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.galflag_too_faint=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.diskflag=zeros([len(self.ra),6],'i') # flag for 1comp-1 2comp-1 2comp-2 3comp-1 3comp-2 3comp-3
    def rungalfit(self,getimages=1,sdssflag=1,mipsflag=1,start_index=0):
        index=arange(start_index,len(self.ra))
        on24index=index[self.On24ImageFlag[index]]
        if sdssflag:
            for i in on24index:
                print 'running galfit analysis for SDSS image of ',self.prefix,'-',self.agcnumber[i]
                quitflag=self.getImages(i,getimages)
                if quitflag:
                    self.writegalflags24()
                    return
            self.writegalflagsSDSS()
        if mipsflag:
            for i in on24index:
                print 'running galfit analysis for MIPS image of ',self.prefix,'-',self.agcnumber[i]
                quitflag=self.getImages24(i,getimages)
                if quitflag:
                    self.writegalflags24()
                    return
            self.writegalflags24()
            
    def readgalflagsSDSS(self):
        infile=self.prefix+'-galfitSDSSFlags.dat'
        dat=atpy.Table(infile,type='ascii')
        self.galflag[:,0]=dat['col3']
        self.galflag[:,1]=dat['col4']
        self.galflag[:,2]=dat['col5']
        self.galflag_too_faint=dat['col6']

    def readgalflags24(self):
        infile=self.prefix+'-galfit24Flags.dat'
        dat=atpy.Table(infile,type='ascii')
        self.galflag24[:,0]=dat['col3']
        self.galflag24[:,1]=dat['col4']
        self.galflag24[:,2]=dat['col5']
        self.galflag_too_faint24=dat['col6']

    def displaygalfitresults(self,galname,i):
        # one comp fit
        quitflag=0
        for j in range(3):
            if self.gaflag[i,j]:
                image_id=galname+'-'
                galfit_log=image_id+str(j+1)+'Comp-fit.log'
                galfit_out=image_id+str(j+1)+'Comp'+'-galfit.01'
                output_image=galname+'-'+str(j+1)+'Comp-galfit-out.fits'
                subcomp_image=image_id+str(j+1)+'Comp'+'-subcomps.fits'
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
                elif j == 1:
                    endframe=9
                elif j == 2:
                    endframe=10
                s='frame delete '+str(endframe)
                d.set(s)
                d.set('frame delete 5')

                for k in range(2,endframe):
                    if k == 5:
                        continue
                    s='frame '+str(k)
                    d.set(s)
                    d.set('scale log')
                    d.set('zoom to fit')
                string=raw_input('hit any key when ready to continue (q to quit; s to stop running galfit for this galaxy) \n')
                for k in range(j+1):
                    s='is component %i part of the disk? (1=yes, 0=no)'%(k)
                    compflag=raw_input(s)
                    if string.find('q') > -1:
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
                    if j == 0:
                        ncomp=j+k
                    elif j == 1:
                        ncomp=j+k
                    elif j == 2:
                        ncomp=j+k+1
                    self.diskflag[i,ncomp]=compflag
                # make disk image
                diskimage=galname+'diskimage-'+(j+1)+'Comp.fits'
                if (j == 0) & (self.diskflag[i,0]):
                    input=subcomp_image+'[2]'
                    iraf.imcopy(input,diskimage)
                if (j == 1):
                    if self.diskflag[i,1]:
                        input=subcomp_image+'[2]'
                        iraf.imcopy(input,diskimage)
                    if self.diskflag[i,2]:
                        input=subcomp_image+'[3]'
                        iraf.imarith(input,"+",diskimage,'temp.fits')
                        os.rename('temp.fits',diskimage)
                if (j == 2):
                    if self.diskflag[i,3]:
                        input=subcomp_image+'[2]'
                        iraf.imcopy(input,diskimage)
                    if self.diskflag[i,4]:
                        input=subcomp_image+'[3]'
                        try:
                            iraf.imcopy(input,diskimage)
                        except:
                            iraf.imarith(input,"+",diskimage,'temp.fits')
                            os.rename('temp.fits',diskimage)
                    if self.diskflag[i,5]:
                        input=subcomp_image+'[4]'
                        try:
                            iraf.imcopy(input,diskimage)
                        except:
                            iraf.imarith(input,"+",diskimage,'temp.fits')
                            os.rename('temp.fits',diskimage)
                        
                        
                
        return quitflag
    def parsegalfit(self):
        # read all output from galfit and save in three arrays
        return
    def reviewgalfit(self):
        # review galfit results, select appropriate model (2 or 3 comp fit), identify disk component(s), make disk image
        return
    def measuredisk(self):
        # run ellipse on disk image, interpolate where enclosed flux is 50% and 90% of total (see elbaz code for interpolating w/discrete data)
        return
    
    def getImages(self,i,getimages, keepimages=1):
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

        # open ds9 display
        d=ds9.ds9()
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
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
            
        field=self.sdssfield[i]
        run=self.sdssrun[i]
        rerun=self.sdssrerun[i]
        camcol=self.sdsscamcol[i]
        row=self.sdssrowc[i]
        row=row[2]
        col=self.sdsscolc[i]
        col=col[2]
        tmp_run='%06d'%(run)
        tmp_field='%04d'%(field)
        sdss_path_c = "imaging/"+str(run)+"/"+str(rerun)+"/corr/"+str(camcol)+"/"
        sdss_corr_name_g = "fpC-"+tmp_run+"-g"+str(camcol)+"-"+tmp_field+".fit"
        sdss_corr_name_r = "fpC-"+tmp_run+"-r"+str(camcol)+"-"+tmp_field+".fit"
        url="http://das.sdss.org/"+sdss_path_c+sdss_corr_name_g+'.gz'
        
        s='wget --no-http-keep-alive -S http://das.sdss.org/'+sdss_path_c+sdss_corr_name_r+'.gz'
        print url
        if getimages:
            urllib.urlretrieve(url,filename=sdss_corr_name_r+'.gz')
            os.system('gunzip %s'%(sdss_corr_name_r+'.gz'))

        # display sdss image
        d.set('file %s'%(sdss_corr_name_r))
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

        psf_image=self.prefix+'-'+str(self.agcnumber[i])+'-'+'SDSS-PSF.fits'
        sky_sub_psf_image=self.prefix+'-'+str(self.agcnumber[i])+'-'+'SDSS-PSF-SKY.fits'
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
        tsfield=atpy.Table(sdss_ts_name_r)
        gain=tsfield.gain[0]
        gain_r=gain[2]
        psfwidth=tsfield.psf_width[0]
        psfwidth_r=psfwidth[2]
        darknoise=tsfield.dark_variance[0]
        darknoise_r=darknoise[2]

        ######################
        #
        # End of getting SDSS Images
        #
        ######################

        # subtract soft bias from image (1000 added to all DN)
        bias_sub_image="fpC-"+tmp_run+"-r"+str(camcol)+"-"+tmp_field+"BiasSub.fit"
        try:
            iraf.imarith(sdss_corr_name_r,'-','1000.',bias_sub_image)
        except:
            print 'Warning:  Problem creating bias-subtracted image, probably b/c it already exists'
            print '   going to delete bias-subtracted image and try imarith again'
            iraf.imdel(bias_sub_image)
            iraf.imarith(sdss_corr_name_r,'-','1000.',bias_sub_image)
        # add rdnoise and gain to image header
        iraf.hedit(images=bias_sub_image,fields='RDNOISE',value=darknoise_r,add='yes',verify='no')
        iraf.hedit(images=bias_sub_image,fields='GAIN',value=gain_r,add='yes',verify='no')

        # convert sdssPetroR90r to pixels using sdss plate scale (0.396127"/pix)
        # if R90 < 10 arcsec, use 10 arcsec instead of R90
        if self.sdssPetroR90r[i] > (min_cutout_size/8.):
            sdssPetroR90r_pixels =self.sdssPetroR90r[i]/sdsspixelscale
        else:
            sdssPetroR90r_pixels =(min_cutout_size/8.)/sdsspixelscale
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
        sexout=atpy.Table('test.cat',type='ascii')
        sexnumber=sexout['col1']
        xsex=sexout['col2']
        ysex=sexout['col3']
        dist=sqrt((row-ysex)**2+(col-xsex)**2)

        #   find object ID

        objIndex=where(dist == min(dist))
        objNumber=sexnumber[objIndex]
        objNumber=objNumber[0] # not sure why above line returns a list
        print 'object number = ',objNumber
        mask_image=self.prefix+'-'+str(self.agcnumber[i])+'-'+'galfit-mask.fits'
        os.rename('segmentation.fits',mask_image)

        #   use iraf imexam to replace object ID values with zero
        iraf.imreplace(mask_image,value=0,lower=objNumber-.5,upper=objNumber+.5)

        # get best x,y positions from sextractor
        #    - subtract 1 b/c sextractor catalog starts at one, but python is zero-indexed
        xcenter=xsex[objNumber-1]
        ycenter=ysex[objNumber-1]
        # define fit area in terms of the petrosian R90
        xminfit=xcenter-4*sdssPetroR90r_pixels
        xmaxfit=xcenter+4*sdssPetroR90r_pixels 
        yminfit=ycenter-4*sdssPetroR90r_pixels 
        ymaxfit=ycenter+4*sdssPetroR90r_pixels 

        print 'xcenter, ycenter = ',xcenter,ycenter
        print 'petrosian R90 in pixels = ',sdssPetroR90r_pixels
        print 'fitting region = [%5.2f, %5.2f, %5.2f, %5.2f]'%(xminfit,xmaxfit,yminfit,ymaxfit)

        # check for boundary issues
        #   row+4*PetroR90, col+4*PetroR90
        #   row-4*PetroR90, col-4*PetroR90

        iraf.imgets(image=sdss_corr_name_r,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=sdss_corr_name_r,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)

        if (xmaxfit > image_xmax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_xmax,xmaxfit)
            print self.prefix, self.agcnumber[i], ': setting xmaxfit to max x of image'
            xmaxfit=image_xmax

        if (ymaxfit > image_ymax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_ymax,ymaxfit)
            print self.prefix, self.agcnumber[i], ': setting ymaxfit to max y of image'
            ymaxfit=image_ymax

        if (xminfit < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f)'%(image_xmax,xminfit)
            print self.prefix, self.agcnumber[i], ': setting xminfit to 1'
            xminfit=1

        if (yminfit < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f)'%(image_xmax,yminfit)
            print self.prefix, self.agcnumber[i], ': setting yminfit to 1'
            yminfit=1

        # gather image parameters needed for galfit input
        #   approx sky value from fpC image header, keyword = SKY

        iraf.imgets(image=sdss_corr_name_r,param='SKY')#get RA of image
	sky=float(iraf.imgets.value)

        input_image=bias_sub_image
        sigma_image='none'
        psf_image=sky_sub_psf_image
        psf_oversampling=1
        # mask_image=mask_image
        convolution_size=100
        #   photometric zeropoint from fpC image header, keyword = FLUX20, then convert to zp
        iraf.imgets(image=sdss_corr_name_r,param='FLUX20')
	flux20=float(iraf.imgets.value)
        iraf.imgets(image=sdss_corr_name_r,param='EXPTIME')
	exptime=float(iraf.imgets.value)
        magzp=20.+2.5*log10(flux20/exptime)
        print 'flux20, exptime, magzp = ',flux20, exptime, magzp
        pscale=sdsspixelscale   # arcsec/pix
        xobj =xcenter
        yobj =ycenter
        mag_total =self.sdssr[i]
        Re =self.sdssExpRadr[i]/sdsspixelscale
        sersic_exp =1
        sersic_bulge=4
        axis_ratio =1./self.sdssExpABr[i]
        PA =self.sdssExpPhir[i]
        rad=self.sdssPetroR50r[i]/sdsspixelscale

        # run galfit 3 times
        galname=self.prefix+'-'+str(self.agcnumber[i])
        self.galflag[i,0],self.galflag[i,1],self.galflag[i,2],quitflag,self.galflag_too_faint[i]=rungalfit3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky)


        # check for other galaxies on this image
        nearby_gal = sqrt(1.*(run-self.sdssrun)**2+(rerun-self.sdssrerun)**2+(field-self.sdssfield)**2+(camcol-self.sdsscamcol)**2)
        print 'SDSS: number of galaxies to analyze in this image = ',len(nearby_gal[nearby_gal < 1.])
        return quitflag


    def getImages24(self,i,getimages, keepimages=1):
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
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')

        self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_unc.fits'

        ra=self.ra[i]
        dec=self.dec[i]
        s='echo %f %f > incoords'%(ra,dec)
        os.system(s)
        print 'deleting outcoords if it exists'
        os.system('rm outcoords')
        iraf.imcoords.wcsctran(image=self.mosaic24,input='incoords',output='outcoords',inwcs='world',outwcs='logical',verbose='no')
        coordout=atpy.Table('outcoords',type='ascii')
        col=coordout['col1']
        row=coordout['col2']
        # convert arrays to numbers
        col=col[0]
        row=row[0]


        ##############################################
        # GET CUTOUT OF GALAXY AND UNCERTAINTY IMAGE
        ##############################################
        if self.sdssPetroR90r[i] > (min_cutout_size/8.):
            sdssPetroR90r_pixels =self.sdssPetroR90r[i]/mipspixelscale
        else:
            sdssPetroR90r_pixels =(min_cutout_size/8.)/mipspixelscale

        # make a cutout of the galaxy (easier than trying to run sextractor on full mosaic
        # need x and y pixel values of the galaxy
        xmin=col-4*sdssPetroR90r_pixels
        xmax=col+4*sdssPetroR90r_pixels
        ymin=row-4*sdssPetroR90r_pixels
        ymax=row+4*sdssPetroR90r_pixels

        # get image dimensions of 24um mosaic

        iraf.imgets(image=self.mosaic24,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=self.mosaic24,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)
        
        # check that cutout region is not outside bounds of image

        if (xmax > image_xmax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_xmax,xmax)
            print self.prefix, self.agcnumber[i], ': setting xmax to max x of image'
            xmax=image_xmax

        if (ymax > image_ymax):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f)'%(image_ymax,ymax)
            print self.prefix, self.agcnumber[i], ': setting ymax to max y of image'
            ymax=image_ymax

        if (xmin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f)'%(image_xmax,xmin)
            print self.prefix, self.agcnumber[i], ': setting xmin to 1'
            xmin=1

        if (ymin < 1):
            print 'WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f)'%(image_xmax,ymin)
            print self.prefix, self.agcnumber[i], ': setting ymin to 1'
            ymin=1
        
        sex_image=self.prefix+'-'+str(self.agcnumber[i])+'-'+'galfit-cutout24.fits'
        unc_image=self.prefix+'-'+str(self.agcnumber[i])+'-'+'galfit-cutout-unc24.fits'

        try:
            s=self.mosaic24+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+sex_image)
            s=self.mosaic24unc+'[%i:%i,%i:%i] '%(int(round(xmin)),int(round(xmax)),int(round(ymin)),int(round(ymax)))
            iraf.imcopy(s,working_dir+unc_image)

        except:
            print 'Warning:  Problem creating cutout image, probably b/c it already exists'
            print '   going to delete existing cutout image and try imcopy again'
            iraf.imdel(sex_image)
            iraf.imdel(unc_image)
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
        mask_image=self.prefix+'-'+str(self.agcnumber[i])+'-'+'galfit-mask24.fits'
        iraf.imdel(mask_image)
        iraf.imcopy('segmentation.fits',mask_image)

        #   use iraf imexam to replace object ID values with zero
        iraf.imreplace(mask_image,value=0,lower=objNumber-.5,upper=objNumber+.5)

        # get best x,y positions from sextractor
        #    - subtract 1 b/c sextractor catalog starts at one, but python is zero-indexed
        xcenter=xsex[objNumber-1]
        ycenter=ysex[objNumber-1]


        ##############################################
        # GATHER GALFIT INPUT
        ##############################################
        input_image=sex_image
        sigma_image=unc_image
        psf_image='/Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits'
        #psf_image='/Users/rfinn/research/LocalClusters/Images/MKW11/24umWCS/PRF.fits'
        psf_oversampling=4
        # mask_image # already defined

        xobj =xcenter
        yobj =ycenter
        mag_total =self.sdssr[i]
        Re =self.sdssExpRadr[i]/mipspixelscale
        axis_ratio =1./self.sdssExpABr[i]
        PA =self.sdssExpPhir[i]
        rad=self.sdssPetroR50r[i]/mipspixelscale

        xminfit=1
        xmaxfit=xmax-xmin
        yminfit=1
        ymaxfit=ymax-ymin
        
        convolution_size=100
        if xmaxfit < 100:
            convolution_size=xmaxfit
        magzp=magzp
        pscale=mipspixelscale   # arcsec/pix
	sky=0

        ##############################################
        # RUN GALFIT 3 TIMES
        ##############################################

        galname=self.prefix+'-'+str(self.agcnumber[i])+'-24'
        # run galfit 3 times
        self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_too_faint24[i]=rungalfit3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky)

        return quitflag



def writegalfitimageparam(outfile,input_image,output_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale):
    outfile.write('# IMAGE PARAMETERS\n')
    outfile.write('A) '+input_image+'              # Input data image (FITS file)\n')
    outfile.write('B) '+output_image+'       # Name for the output image\n')
    outfile.write('C) %s                # Sigma image name (made from data if blank or "none") \n'%(sigma_image))
    outfile.write('D) '+psf_image+'     # Input PSF image and (optional) diffusion kernel\n')
    outfile.write('E) %i                   # PSF oversampling factor relative to data\n'%(psf_oversampling))
    outfile.write('F) '+mask_image+'           # Pixel mask (ASCII file or FITS file with non-0 values)\n')
    outfile.write('G)         # Parameter constraint file (ASCII)\n')
    outfile.write('H) '+str(int(round(xminfit)))+' '+str(int(round(xmaxfit)))+' '+str(int(round(yminfit)))+' '+str(int(round(ymaxfit)))+'     # Image region to fit (xmin xmax ymin ymax)\n')
    outfile.write('I) '+str(convolution_size)+' '+str(convolution_size)+'             # Size of convolution box (x y)\n')
    outfile.write('J) %5.2f              # Magnitude photometric zeropoint \n'%(magzp))
    outfile.write('K) %6.5f   %6.5f         # Plate scale (dx dy)  [arcsec/pix]\n'%(pscale,pscale))
    outfile.write('O) both                # Display type (regular, curses, both)\n')
    outfile.write('P) 0                   # Create output image only? (1=yes; 0=optimize) \n')
    outfile.write('S) 0                   # Modify/create objects interactively?\n')


def writegalfitsersic(outfile,objnumber,profile,xobj,yobj,mag,rad,sersic_exp,axis_ratio,pa):
    outfile.write(' \n')
    outfile.write('# Object number: %i \n'%(objnumber))
    outfile.write(' 0) %s             # Object type \n'%(profile))
    outfile.write(' 1) %8.1f  %8.1f 1 1  # position x, y        [pixel] \n'%(xobj,yobj))
    outfile.write(' 3) %5.2f      1       # total magnitude     \n'%(mag))
    outfile.write(' 4) %8.2f       1       #     R_e              [Pixels] \n'%(rad))
    outfile.write(' 5) %5.2f       1       # Sersic exponent (deVauc=4, expdisk=1)   \n'%(sersic_exp))
    outfile.write(' 9) %5.2f       1       # axis ratio (b/a)    \n'%(axis_ratio))
    outfile.write('10) %5.2f       1       # position angle (PA)  [Degrees: Up=0, Left=90] \n'%(pa))
    outfile.write(" Z) 0                  # Output option (0 = residual, 1 = Don't subtract)  \n")

def writegalfitsky(outfile,objnumber,sky):    
    outfile.write(' \n')
    outfile.write('# Object number: %i \n'%(objnumber))
    outfile.write(' 0) sky             # Object type \n')
    outfile.write(' 1) %8.1f   1  # sky background at center of fitting region [ADUs] \n'%(sky))
    outfile.write(' 2) 0      0       # total magnitude     \n')
    outfile.write(' 3) 0      0       #     R_e              [Pixels] \n')
    outfile.write(" Z) 0                  # Output option (0 = residual, 1 = Don't subtract)  \n")

def parsegalfit1comp(galfit_outimage):
    header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_SKY']
    fit_parameters=[]
    for hkey in header_keywords:
        iraf.imgets(image=galfit_outimage,param=hkey)
        t=iraf.imgets.value.split('+/-')
        #print hkey,t
        try:
            values=(float(t[0]),float(t[1]))# fit and error
        except ValueError:
            # look for * in the string, which indicates numerical problem
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
    return fit_parameters

def parsegalfit2comp(galfit_outimage):
    header_keywords=['1_XC','1_YC','1_MAG','1_RE','1_N','1_AR','1_PA','2_XC','2_YC','2_MAG','2_RE','2_N','2_AR','2_PA','3_SKY']
    fit_parameters=[]
    for hkey in header_keywords:
        iraf.imgets(image=galfit_outimage,param=hkey)
        t=iraf.imgets.value.split('+/-')
        try:
            values=(float(t[0]),float(t[1]))# fit and error
        except ValueError:
            # look for * in the string, which indicates numerical problem
            t[0]=t[0].replace('*','')
            t[1]=t[1].replace('*','')
            values=(float(t[0]),float(t[1]))# fit and error
        fit_parameters.append(values)
    return fit_parameters

def parsegalfit2compold(galfit_outimage):
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


def rungalfit3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky):
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
    initial_rad=rad
    profile = 'sersic'
    sersic_exp =2
    # open ds9 display
    d=ds9.ds9()
    galflag=zeros(3,'i')
    quitflag=0
    too_faint_flag=0
    for j in range(3):

        output_image=galname+'-'+str(j+1)+'Comp-galfit-out.fits'
        # create galfit input file
        galfile='galfit.input.'+str(j+1)+'Comp'
        galfit_input=open(galfile,'w')
        
        writegalfitimageparam(galfit_input,input_image,output_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale)
        
        if j == 0:
            # gather object parameters from sextractor output
            
            # loop through galfit 3 times...
            #    first time fits on sersic profile
            
            objnumber=1
            writegalfitsersic(galfit_input,objnumber,profile,xobj,yobj,mag_total,rad,sersic_exp,axis_ratio,PA)
            objnumber=2
            writegalfitsky(galfit_input,objnumber,sky)    
            


        elif j == 1:
            # read in output from round 1
            # get values from header of image[2]
            galfit_outimage=galname+'-'+'1Comp-galfit-out.fits[2]'
            
            (x_fit,y_fit,mag_fit,Re_fit,Nsersic_fit,axis_ratio_fit,pa_fit,sky_fit)=parsegalfit1comp(galfit_outimage)
            
            
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
            writegalfitsersic(galfit_input,objnumber,profile,x_fit[0],y_fit[0],mag,rad,nsers1,axis_ratio_fit[0],pa_fit[0])
            objnumber=2
            rad=0.5*rad1
            writegalfitsersic(galfit_input,objnumber,profile,x_fit[0],y_fit[0],mag,rad,2,axis_ratio_fit[0],pa_fit[0])
            objnumber=3
            writegalfitsky(galfit_input,objnumber,sky_fit[0])    

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
            x_fit1,y_fit1,mag_fit1,Re_fit1,Nsersic_fit1,axis_ratio_fit1,pa_fit1,x_fit2,y_fit2,mag_fit2,Re_fit2,Nsersic_fit2,axis_ratio_fit2,pa_fit2,sky_fit=parsegalfit2comp(galfit_outimage)

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
            writegalfitsersic(galfit_input,objnumber,profile,x_fit1[0],y_fit1[0],mag,rad1,nsers1,axis_ratio_fit1[0],pa_fit1[0])
            objnumber=2
            writegalfitsersic(galfit_input,objnumber,profile,x_fit2[0],y_fit2[0],mag,rad2,nsers2,axis_ratio_fit2[0],pa_fit2[0])
            objnumber=3
            pa=PA
            axis_ratio=0.5*(axis_ratio_fit1[0]+axis_ratio_fit2[0])
            #xobj=0.5*(x_fit1[0]+x_fit2[0])
            #yobj=0.5*(y_fit1[0]+y_fit2[0])
            writegalfitsersic(galfit_input,objnumber,profile,xobj,yobj,mag,rad3,nsers3,axis_ratio,pa)
            objnumber=4
            writegalfitsky(galfit_input,objnumber,sky)    

        galfit_input.close()
        
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
            flag=raw_input('press any key to continue, q to quit; s to stop running galfit on this galaxy')
            if flag.find('q') > -1:
                quitflag=1
            elif string.find('s') > -1:
                flag=raw_input('is galaxy too faint? (y or n)')
                if flag.find('y') > -1:
                    too_faint_flag=1
                break
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
                d.set('scale log')
                d.set('zoom to fit')
                print k
                if k == 2:
                    d.set('regions command {text 30 10 #text="Image" font="times 18 bold" color="red"}')
                if k == 3:
                    d.set('regions command {text 30 10 #text="Model" font="times 18 bold" color="red"}')
                if k == 4:
                    d.set('regions command {text 30 10 #text="Residual" font="times 18 bold" color="red"}')
            if j == 0:
                galfit_outimage=galname+'-'+'1Comp-galfit-out.fits[2]'
                (x_fit,y_fit,mag_fit,Re_fit,Nsersic_fit,axis_ratio_fit,pa_fit,sky_fit)=parsegalfit1comp(galfit_outimage)
                #d.set('frame 7')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit[0],Re_fit[0],Nsersic_fit[0],axis_ratio_fit[0],pa_fit[0])
                print s
                #d.set(s)
            if j == 1:
                galfit_outimage=galname+'-'+'2Comp-galfit-out.fits[2]'
                x_fit1,y_fit1,mag_fit1,Re_fit1,Nsersic_fit1,axis_ratio_fit1,pa_fit1,x_fit2,y_fit2,mag_fit2,Re_fit2,Nsersic_fit2,axis_ratio_fit2,pa_fit2,sky_fit=parsegalfit2comp(galfit_outimage)
                #d.set('frame 7')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit1[0],Re_fit1[0],Nsersic_fit1[0],axis_ratio_fit1[0],pa_fit1[0])
                print s
                #d.set(s)
                #d.set('frame 8')
                s='regions command {text 10 5 #text="%4.1f, %5.1f, %3.1f, %5.2f, %3.0f" font="times 12" color="red"}'%(mag_fit2[0],Re_fit2[0],Nsersic_fit2[0],axis_ratio_fit2[0],pa_fit2[0])
                #d.set(s)
            raw_input('hit any key when ready to view subcomponent images \n')
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

            for k in range(2,endframe):
                if k == 5:
                    continue
                s='frame '+str(k)
                d.set(s)
                d.set('scale log')
                d.set('zoom to fit')
            string=raw_input('hit any key when ready to continue (q to quit; s to stop running galfit for this galaxy) \n')
            d.set('saveimage png image.png')
            img_name=image_id+str(j+1)+'Comp.png'
            os.rename('image.png',img_name)
            if string.find('q') > -1:
                quitflag=1
                break
            elif string.find('s') > -1:
                print 'gal and quitflags = ',galflag[0],galflag[1],galflag[2],quitflag
                flag=raw_input('is galaxy too faint? (y or n)')
                if flag.find('y') > -1:
                    too_faint_flag=1
                break
        else:
            print "not diplaying images b/c galfit didn't finish"
            print "not going to try to fit anything else w/this galaxy (if there is anymore...)"
            return galflag[0],galflag[1],galflag[2],quitflag
    return galflag[0],galflag[1],galflag[2],quitflag,too_faint_flag

#infile=open(galfit_log,'r')
#lines=readlines(infile)
#infile.close()
#t=lines[7]
#xfit=float(t[14:22])
#yfit=float(t[23:32])
#mag_tot=float(t[35:41])
#Re=float(t[42:51])
#sersic_exp=float(t[55:59])
#PA
mkw11=cluster('MKW11')
