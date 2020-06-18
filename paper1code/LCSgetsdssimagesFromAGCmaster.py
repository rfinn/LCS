#!/usr/bin/env python
"""
    GOAL
      get sdss cutouts for all sdss galaxies in the LCS fields
      these will be fed into galfit 

    PROCEDURE
      get fpC image - contains galaxy image
      get psField image - used to reconstruct psf
      get row,col of galaxy on fpC image
      get psf at the location of galaxy using read_PSF
      make a cutout of galaxy (+/- 4xPetroR90)
      run galfit
        get most of input parameters from fpC image header
      save results
    
"""

from pylab import *
import os, pyfits,ds9
from LCSReadmasterBaseNSA import *
from scipy.interpolate import interp1d
import urllib, atpy
from pyraf import iraf
iraf.stsdas()
iraf.analysis()
iraf.isophote()
iraf.tables()
iraf.ttools()

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
        self.galflag_stop=zeros(len(self.ra),'i') # if fitting was stopped, either b/c too faint or results were not reasonable
        self.galflag_stop24=zeros(len(self.ra),'i') # 
        self.working_dir_sdss=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/SDSS/'
        self.working_dir_mips=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        self.galfit_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/'
        self.diskflag=zeros([len(self.ra),6],'i') # flag for 1comp-1 2comp-1 2comp-2 3comp-1 3comp-2 3comp-3
        self.diskflag24=zeros([len(self.ra),6],'i') # flag for 1comp-1 2comp-1 2comp-2 3comp-1 3comp-2 3comp-3
        self.galfit_sdssR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_sdssR90=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_mipsR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_mipsR90=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
    def rungalfit(self,getimages=1,sdssflag=1,mipsflag=1,continue_flag=0):
        # wrapper function to run galfit analysis on the sdss and mips images
        #
        # get images = 1 to get SDSS image or create 24um cutout
        # get images = 0 - can use this if all the images already exist
        #
        # sdssflag = 1 to run galfit analysis on the sdss images
        #
        # mipsflag = 1 to run galfit analysis on the 24um images
    
        # wrapper for getImages
        # calls getImages for a series of galaxies
        # for now, just use images that are on 24 micron image
        start_index=0
        index=arange(start_index,len(self.ra))
        on24index=index[self.On24ImageFlag[index]]
        #i=self.agcdict[230351]
        #i=self.agcdict[230369]
        #i=self.agcdict[-99991]

        results_file=self.galfit_dir+self.prefix+'-galfitResults-spirals-sdss.dat'
        results_file24=self.galfit_dir+self.prefix+'-galfitResults-spirals-24.dat'
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
            try:
                os.remove(results_file)
            except OSError:
                print 'looks like this is the first time running sdss for ',self.prefix
            try:
                os.remove(results_file24)
            except OSError:
                print 'looks like this is the first time running mips for ',self.prefix
        print 'here is what I think the start_index is: ',start_index
        ngal=0
        for i in range(len(self.ra)):
            if (self.On24ImageFlag[i] & self.spiralFlag[i]):
                if ngal < start_index:
                    ngal += 1
                    continue
                if sdssflag:
                    print 'running galfit analysis for SDSS image of ',i,self.prefix,'-',self.agcnumber[i]
                    quitflag=self.getImages(i,getimages)
                    if quitflag:
                        self.writegalflagsSDSS()
                        return
                    if self.galflag_stop[i] < .1:
                        
                        print 'running display results for SDSS image of ',i,self.prefix,'-',self.agcnumber[i]
                        quitflag,self.galflag_stop[i]=self.displaygalfitresults(i,band=0)
                        if quitflag:
                            self.writegalflagsSDSS()
                            return
                        print 'running call_measuredisk for SDSS image of ',i,self.prefix,'-',self.agcnumber[i]
                        if self.galflag_stop[i] < .1:
                            self.call_measuredisk(i,wave_band=0)

                    # add line to output file
                    outfile=open(results_file,'a')
                    output_string=' %i %i %i %i %i %i %i %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f \n'%(i,self.On24ImageFlag[i],self.galflag[i][0],self.galflag[i][1],self.galflag[i][2],self.galflag_stop[i],self.galflag_too_faint[i],self.galfit_sdssR50[i][0],self.galfit_sdssR50[i][1],self.galfit_sdssR50[i][2],self.galfit_sdssR90[i][0],self.galfit_sdssR90[i][1],self.galfit_sdssR90[i][2])
                    outfile.write(output_string)
                    outfile.close()
                if mipsflag:
                    print 'running galfit analysis for MIPS image of ',self.prefix,'-',self.agcnumber[i]
                    quitflag=self.getImages24(i,getimages)
                    if quitflag:
                        self.writegalflags24()
                        return
                    if self.galflag_stop24[i] < .1:
                        quitflag,self.galflag_stop24[i]=self.displaygalfitresults(i,band=1)
                        if quitflag:
                            self.writegalflags24()
                            return
                        if self.galflag_stop24[i] < .1:
                            self.call_measuredisk(i,wave_band=1)

                    # add line to output file
                    outfile=open(results_file24,'a')
                    output_string=' %i %i %i %i %i %i %i %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f \n'%(i,self.On24ImageFlag[i],self.galflag24[i][0],self.galflag24[i][1],self.galflag24[i][2],self.galflag_stop24[i],self.galflag_too_faint24[i],self.galfit_mipsR50[i][0],self.galfit_mipsR50[i][1],self.galfit_mipsR50[i][2],self.galfit_mipsR90[i][0],self.galfit_mipsR90[i][1],self.galfit_mipsR90[i][2])
                    outfile.write(output_string)
                    outfile.close()
            
        self.writegalflagsSDSS()
        self.writegalflags24()

            
    def writegalflagsSDSS(self):
        outfile=self.prefix+'-galfitSDSSFlags.dat'
        out1=open(outfile,'w')
        for i in range(len(self.ra)):
            out1.write('%i %i %i %i %i %i\n'%(i,self.On24ImageFlag[i],self.galflag[i,0],self.galflag[i,1],self.galflag[i,2],self.galflag_too_faint[i]))
        out1.close()
    def writegalflags24(self):
        outfile=self.prefix+'-galfit24Flags.dat'
        out1=open(outfile,'w')
        for i in range(len(self.ra)):
            out1.write('%i %i %i %i %i %i\n'%(i,self.On24ImageFlag[i],self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],self.galflag_too_faint24[i]))
        out1.close()
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

    def parsegalfit(self):
        # still working on this - just a collection of potentially useful code at this point
        # read all output from galfit and save in three arrays
        if self.gaflag[i,j]:
            image_id=galname+'-'
            galfit_log=image_id+str(j+1)+'Comp-fit.log'
            #    - display results (like ds9 multiextension data cube -> use xpa)
            #
            infile=open(galfit_log,'r')
            galfit_results=infile.readlines()
            infile.close()


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

        return

    def displaygalfitresults(self,i,band=0):
        # one comp fit
        # band = 0 for sdss
        # band = 1 for mips

        galname=os.getcwd()+'/'+self.prefix+'-'+str(self.agcnumber[i])
        model_mag=[]
        if band == 0:
            print 'running for sdss galaxies!'
            #self.readgalflags()
            galflag=self.galflag
            working_dir=self.working_dir_sdss
            diskflag=self.diskflag
        if band == 1:
            print 'running for mips galaxies!'
            #self.readgalflags24()
            galflag = self.galflag24
            galname=galname+'-24'
            working_dir=self.working_dir_mips
            diskflag=self.diskflag24
        os.chdir(working_dir)
        
        quitflag=0
        galflag_stop=0
        d=ds9.ds9()
        d.set('cd '+os.getcwd())
        for j in range(3):
            component_flag=0
            model_mag=[]
            print i,j,galflag[i,j]
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
                    for m in range(15):
                        print galfit_results[m].rstrip()
                elif j == 1:
                    endframe=9
                    for m in range(15,32):
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
                string=raw_input('hit any key when ready to continue (q to quit; s to stop running galfit for this galaxy) \n')
                if string.find('q') > -1:
                    quitflag = 1
                    return quitflag,galflag_stop
                elif string.find('s') > -1:
                    galflag_stop=1
                    return quitflag,galflag_stop
                for k in range(j+1):
                    s='is component %i part of the disk? (1=yes, 0=no) \n'%(k+1)
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
                    if compflag == 1:
                        if j == 0:
                            ncomp=j+k
                            model_mag.append(float(galfit_results[7][35:41]))
                            
                        elif j == 1:
                            ncomp=j+k
                            if k == 0:
                                model_mag.append(float(galfit_results[15+7][35:41]))
                            elif k == 1:
                                model_mag.append(float(galfit_results[15+9][35:41]))
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

    def call_measuredisk(self,i,wave_band=0):
        # gather necessary input for measuredisk
        #galname=os.getcwd()+'/'+self.prefix+'-'+str(self.agcnumber[i])
        working_dir=os.getcwd()+'/'
        print working_dir
        galname=self.prefix+'-'+str(self.agcnumber[i])
        model_mag=[]
        if wave_band == 0:
            print 'running call_measuredisk for sdss galaxies!'
            #self.readgalflags()
            galflag=self.galflag
            working_dir=self.working_dir_sdss
            minr=2
        if wave_band == 1:
            print 'running call_measuredisk for mips galaxies!'
            #self.readgalflags24()
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

        xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky=parsegalfit1comp(os.getcwd()+'/'+output_image+'[2]')
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
        for j in range(3):
            disk_image=galname+'-diskimage-'+str(j+1)+'Comp.fits'
            if os.path.exists(working_dir+disk_image):#if galflag[i,j]:

                iraf.imgets(image=working_dir+disk_image,param='MAG_TOTAL')
                mag_tot=float(iraf.imgets.value)
                iraf.imgets(image=working_dir+disk_image,param='naxis1')
                maxr=float(iraf.imgets.value)

                self.measuredisk(i,disk_image,mag_tot,xc,yc,PA,iellip,initialr,minr,maxr,mag_zp,band=wave_band)

    def call_measureRadius(self,i,wave_band=0):
        # gather necessary input for measuredisk
        #galname=os.getcwd()+'/'+self.prefix+'-'+str(self.agcnumber[i])
        working_dir=os.getcwd()+'/'
        print working_dir
        galname=self.prefix+'-'+str(self.agcnumber[i])
        model_mag=[]
        if wave_band == 0:
            print 'running call_measuredisk for sdss galaxies!'
            #self.readgalflags()
            galflag=self.galflag
            working_dir=self.working_dir_sdss
            minr=2
        if wave_band == 1:
            print 'running call_measuredisk for mips galaxies!'
            #self.readgalflags24()
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

        xc,yc,mag,Re,Nsersic,axis_ratio,PA,sky=parsegalfit1comp(os.getcwd()+'/'+output_image+'[2]')
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
        for j in range(3):
            disk_image=galname+'-diskimage-'+str(j+1)+'Comp.fits'
            if os.path.exists(working_dir+disk_image):#if galflag[i,j]:

                iraf.imgets(image=working_dir+disk_image,param='MAG_TOTAL')
                mag_tot=float(iraf.imgets.value)
                iraf.imgets(image=working_dir+disk_image,param='naxis1')
                maxr=float(iraf.imgets.value)

                R50,R90=self.measureRadius(i,disk_image,wave_band,mag_zp)
                print 'R50, R90 = ',R50,R90
                if wave_band == 0:
                    self.galfit_sdssR50[i,j]=R50
                    self.galfit_sdssR90[i,j]=R90
                elif wave_band == 1:
                    self.galfit_mipsR50[i,j]=R50
                    self.galfit_mipsR90[i,j]=R90

    def measuredisk(self,i,image,mag_tot,xcenter,ycenter,ipa,iellip,initialr,minr,maxr,zp,band=0,nframe=1,myradius=15):
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

        #run ellipse
        t=mfile.split('.')
        efile=t[0]+'.tab'
        efile_ascii=t[0]+'.dat'
        imprefix=t[0]
        print 'Running ellipse to fit isophotes to galaxy:'
        try:
            os.remove(efile)
        except OSError:
            print 'everything is ok'
        print "First pass, letting PA and e vary"
        iraf.ellipse(input=image,output=efile,x0=xcenter,y0=ycenter,hcenter='no',sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no')
        print 'Displaying isophotes from first pass.  Hit q in DS9 window to quit'
        iraf.isoexam(table=efile)
        if os.path.exists(working_dir+'junk.txt'):
            os.remove(working_dir+'junk.txt')
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout=working_dir+'junk.txt')
        s="awk '{print $2, $7, $9, $11, $13}' < "+working_dir+"junk.txt > "+working_dir+"junk2.txt"
        os.system(s)
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
                    #11 - X0, 13 - Y0
                    newxcenter=float(t[3])
                    newycenter=float(t[4])
                    break
            s='rm '+efile
            os.system(s)
            iraf.ellipse(input=image,output=efile,x0=newxcenter,y0=newycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=newellip,hellip='yes')

            print 'Displaying isophotes from second pass using r = ',myradius
            print 'Hit q in the DS9 window to quit'
            iraf.isoexam(table=efile)
                
            flag=str(raw_input('Are you happy with the fit?  y=yes n=no x=quit '))
            flag=str(flag)
            print 'this is what I think you typed ',flag
            if flag.find('n') > -1:
                flag2=str(raw_input('What is the problem? r=set new radius x=quit '))
                flag2=str(flag2)
                if flag2.find('r') > -1:
                    myr=input('Enter new radius to use ')
                    myradius=float(myr)
                    s='rm '+efile
                    os.system(s)
                    repeatflag=1
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

                print 'repeatflag = ',repeatflag        
        # convert ellipse table from binary to ascii
        if os.path.exists(efile_ascii):
            os.remove(efile_ascii)
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout=efile_ascii)

    def measureRadius(self,i,image,band,zp):
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
        if band == 0:
            flux_tot = flux_tot*53.9075
        ellipse_output=atpy.Table(efile_ascii,type='ascii')

        # read in sma and enclosed flux from ellipse output
        sma=ellipse_output['col2']
        enc_flux=ellipse_output['col22']
        print zp, mag_total, flux_tot, max(enc_flux)
        figure()
        plot(sma,enc_flux)
        title(image)
        xlabel('semi-major axis (pixels)')
        ylabel('Enclosed Flux (ADU or something like that)')
        axhline(y=0.5*flux_tot,ls='--',color='k',label='_nolegend_')
        axhline(y=0.9*flux_tot,ls=':',color='k',label='_nolegend_')

        # fit enclosed flux vs semi-major axis
        profile_func=interp1d(enc_flux,sma)

        # measure R50 (where enclosed flux = 50% total, R90
        R50=profile_func(0.5*flux_tot)
        iraf.hedit(images=image,fields='LCS_R50',value=R50,add='yes',verify='no')
        try:
            R90=profile_func(0.9*flux_tot)
        except ValueError:
            print '\n ',image[0],': F90 is outside interpolation range!!! \n'
            R90=0
        iraf.hedit(images=image,fields='LCS_R90',value=R90,add='yes',verify='no')
        axvline(x=R50,ls='--',color='r',label='LCS R50')
        axvline(x=R90,ls=':',color='r',label='LCS R90')
        axvline(x=self.sdssPetroR50r[i]/sdsspixelscale,ls='--',color='k',label='SDSS R50')
        axvline(x=self.sdssPetroR90r[i]/sdsspixelscale,ls=':',color='k',label='SDSS R90')
        axvline(x=self.sdssIsoAr[i],ls=':',color='c',label='SDSS ISO')
        legend(loc='lower right')
        t=image.split('.')
        fname=working_dir+t[0]+'-radialPlot.png'
        savefig(fname)
        return R50,R90
    
    def getImages(self,i,getimages=0, keepimages=1):
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
            
        field=self.sdssfield[i]
        run=self.sdssrun[i]
        rerun=self.sdssrerun[i]
        camcol=self.sdsscamcol[i]
        row=self.sdssrowc[i]
        row=row[2]
        col=self.sdsscolc[i]
        col=col[2]
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
        # s='wget --no-http-keep-alive -S http://das.sdss.org/'+sdss_path_c+sdss_corr_name_r+'.gz'
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

        # subtract soft bias from image (1000 added to all DN)
        bias_sub_image="fpC-"+tmp_run+"-r"+str(camcol)+"-"+tmp_field+"BiasSub.fit"
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
        mask_image=self.prefix+'-'+str(self.agcnumber[i])+'-'+'galfit-mask.fits'
        os.rename(working_dir+'segmentation.fits',working_dir+mask_image)

        #   use iraf imexam to replace object ID values with zero
        iraf.imreplace(working_dir+mask_image,value=0,lower=objNumber-.5,upper=objNumber+.5)

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

        iraf.imgets(image=working_dir+sdss_drC_name_r,param='naxis1')#get RA of image
	image_xmax=float(iraf.imgets.value)
        iraf.imgets(image=working_dir+sdss_drC_name_r,param='naxis2')#get RA of image
	image_ymax=float(iraf.imgets.value)

        if (xmaxfit > image_xmax):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f) \n'%(image_xmax,xmaxfit)
            print self.prefix, self.agcnumber[i], ': setting xmaxfit to max x of image'
            xmaxfit=image_xmax

        if (ymaxfit > image_ymax):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xmaxfit = %5.2f) \n'%(image_ymax,ymaxfit)
            print self.prefix, self.agcnumber[i], ': setting ymaxfit to max y of image'
            ymaxfit=image_ymax

        if (xminfit < 1):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, xminfit = %5.2f) \n'%(image_xmax,xminfit)
            print self.prefix, self.agcnumber[i], ': setting xminfit to 1'
            xminfit=1

        if (yminfit < 1):
            print '\n WARNING: fit region extends beyond image boundary (image size = %5.2f, yminfit = %5.2f) \n'%(image_xmax,yminfit)
            print self.prefix, self.agcnumber[i], ': setting yminfit to 1'
            yminfit=1

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
        mag_total =self.sdssr[i]
        Re =self.sdssExpRadr[i]/sdsspixelscale
        sersic_exp =1
        sersic_bulge=4
        axis_ratio =1./self.sdssExpABr[i]
        PA =self.sdssExpPhir[i]
        rad=self.sdssPetroR50r[i]/sdsspixelscale

        # run galfit 3 times
        galname=working_dir+self.prefix+'-'+str(self.agcnumber[i])
        self.galflag[i,0],self.galflag[i,1],self.galflag[i,2],quitflag,self.galflag_stop[i],self.galflag_too_faint[i]=rungalfit3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky)


        # check for other galaxies on this image
        nearby_gal = sqrt(1.*(run-self.sdssrun)**2+(rerun-self.sdssrerun)**2+(field-self.sdssfield)**2+(camcol-self.sdsscamcol)**2)
        print 'SDSS: number of galaxies to analyze in this image = ',len(nearby_gal[nearby_gal < 1.])
        return quitflag

    def justGetSDSSImages(self):
        for i in range(len(self.ra)):
            if self.On24ImageFlag[i]:
                self.justGetImages(i)

    def justGetImages(self,i):
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
            
        field=self.sdssfield[i]
        run=self.sdssrun[i]
        rerun=self.sdssrerun[i]
        camcol=self.sdsscamcol[i]
        row=self.sdssrowc[i]
        row=row[2]
        col=self.sdsscolc[i]
        col=col[2]
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
        # s='wget --no-http-keep-alive -S http://das.sdss.org/'+sdss_path_c+sdss_corr_name_r+'.gz'
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
        d.set('cd '+working_dir)
        # clean up any previously created galfit files
        os.system('rm fit.log')
        os.system('rm galfit.??')

        self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_unc.fits'

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

        galname=working_dir+self.prefix+'-'+str(self.agcnumber[i])+'-24'
        # run galfit 3 times
        
        self.galflag24[i,0],self.galflag24[i,1],self.galflag24[i,2],quitflag,self.galflag_stop24[i],self.galflag_too_faint24[i]=rungalfit3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky)

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
    working_dir=os.getcwd()+'/'
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
    working_dir=os.getcwd()+'/'
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


def rungalfit3times(galname,input_image,sigma_image,psf_image,psf_oversampling,mask_image,xminfit,xmaxfit,yminfit,ymaxfit,convolution_size,magzp,pscale,xobj,yobj,mag_total,rad,axis_ratio,PA,sky,ntimes=3,interactive=1):
    
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


    initial_rad=rad
    profile = 'sersic'
    sersic_exp =2
    # open ds9 display
    d=ds9.ds9()
    galflag=zeros(3,'i')
    quitflag=0
    stop_gal_flag=0
    too_faint_flag=0
    working_dir=os.getcwd()+'/'
    d.set('cd '+working_dir)
    if ntimes == 3:
        loops=arange(3)
    elif ntimes == 2:
        loops=arange(2)
    elif ntimes == 1:
        loops=arange(1)
    elif ntimes= -3:
        loops=array([2],'i')
    elif ntimes= -2:
        loops=array([1],'i')
    for j in loops:

        output_image=galname+'-'+str(j+1)+'Comp-galfit-out.fits'
        # create galfit input file
        galfile=working_dir+'galfit.input.'+str(j+1)+'Comp'
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
            writegalfitsersic(galfit_input,objnumber,profile,xobj,yobj,mag,rad2,nsers2,axis_ratio_fit2[0],pa_fit2[0])
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

            for k in range(2,endframe):
                if k == 5:
                    continue
                if k == 6:
                    continue
                s='frame '+str(k)
                d.set(s)
                d.set('scale log')
                d.set('zoom to fit')
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

mkw11=cluster('MKW11')
