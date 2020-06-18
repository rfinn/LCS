#!/usr/bin/env python
"""
    
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
iraf.digiphot()
#iraf.apphot()
iraf.daophot()

min_cutout_size=100. # in arcsec
mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

    #getimages=1
class cluster(baseClusterNSA):
    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)



    def measuresnr24(self):
        working_dir=homedir+'research/LocalClusters/MIPS_SNR/'
        s='mkdir -p '+working_dir
        os.system(s)
        os.chdir(working_dir)
        self.mosaic24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        #self.mosaic24unc=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_std.fits'
        self.mosaic24unc=homedir+'research/LocalClusters/MIPS/rawdata/'+self.prefix+'/FullMosaic/mosaic_noise.fits'
        working_dir=os.getcwd()
        ra=self.ra[self.On24ImageFlag]
        dec=self.dec[self.On24ImageFlag]
        outfile=open(working_dir+'/'+self.prefix+'_incoords','w')
        for i in range(len(ra)):
            outfile.write('%12.8f %12.8f \n'%(ra[i],dec[i]))
        outfile.close()
        output_coords=working_dir+'/'+self.prefix+'_outcoords'
        if os.path.exists(output_coords):
            os.remove(output_coords)
        input_coords=working_dir+'/'+self.prefix+'_incoords'
        print input_coords,os.getcwd()
        iraf.imcoords.wcsctran(image=self.mosaic24,input=input_coords,output=output_coords,inwcs='world',outwcs='logical',verbose='no')
        # run qphot
        #iraf.apphot.qphot(image=self.mosaic24,cbox=2,coords=output_coords,apertures='2.0, 3.0, 4.0, 5.0, 6.0' ,interactive='no',annulus=10, dannulus=5)
        skyfile=working_dir+'/'+self.prefix+"_sky"
        sky = open(skyfile,'w')
        for i in range(len(ra)):
            sky.write("0.0 \n")
        sky.close()
        aps = open("apertures",'w')
        #aps.write("1,1.5,2,2.6,3,3.5,4,4.5,5,5.5")
        aps.write("2.6")
        aps.close()

        #runiraf()
        datfile=self.prefix+"_phot.dat"
        if os.path.exists(datfile):
            os.remove(datfile)
        iraf.digiphot.daophot.phot(image=self.mosaic24,coords=output_coords,output=datfile,calgorithm='none',skyfile=skyfile,salgori="file",aperture="apertures",interactive="no",verify='no',verbose='no')

        input=open(datfile,'r')
        aperture=zeros(len(ra),'f')
        counts=zeros(len(ra),'f')
        area=zeros(len(ra),'f')
        j=0
        i=0
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
                aperture[i]=float(t[0])
                try:
                    counts[i]=float(t[1])
                except ValueError:
                    counts[i]=0
                try:
                    area[i]=float(t[2])
                except ValueError:
                    area[i]=0
                i += 1
        input.close()


        datfile=self.prefix+"_phot_unc.dat"
        if os.path.exists(datfile):
            os.remove(datfile)
        iraf.digiphot.daophot.phot(image=self.mosaic24unc,coords=output_coords,output=datfile,calgorithm='none',skyfile=skyfile,salgori="file",aperture="apertures",interactive="no",verify='no',verbose='no')
        ##iraf.digiphot.apphot.phot(image,coords="noisecoords.dat",output="noise.dat",calgorithm='none',skyfile="sky",salgori="file",aperture="apertures",interactive="no",verbose='yes')


        input=open(datfile,'r')
        uncaperture=zeros(len(ra),'f')
        unccounts=zeros(len(ra),'f')
        uncarea=zeros(len(ra),'f')
        dataflag=ones(len(ra),'i')
        j=0
        i=0
        for line in input:
            if line.find('#') > -1: #skip lines with '#' in them
                continue
            if line.find('mosaic') > -1: #skip lines with '#' in them
                j=0
                continue
            j=j+1
            if (j > 3):
                print j, line
                t = line.split()
                uncaperture[i]=float(t[0])
                try:
                    unccounts[i]=float(t[1])
                except ValueError:
                    dataflag[i]=0
                #uncarea[i]=float(t[2])
                i += 1
        input.close()

        for i in range(len(counts)):
            print i, counts[i],unccounts[i],abs(counts[i]/unccounts[i])
        self.snr24=zeros(len(self.ra),'f')
        self.snr24[self.On24ImageFlag]=counts/unccounts
        self.f24NSA=zeros(len(self.ra),'f')
        self.f24NSAerr=zeros(len(self.ra),'f')
        self.f24NSA[self.On24ImageFlag]=counts
        self.f24NSAerr[self.On24ImageFlag]=unccounts
        #coordout=atpy.Table(output_coords,type='ascii')
        #col=coordout['col1']
        #row=coordout['col2']
        # convert arrays to numbers
        #col=col[0]
        #row=row[0]
        #figure()
        outfile=open(homedir+'research/LocalClusters/NSAmastertables/SNR24/'+self.prefix+'_snr24NSA.dat','w')
        for i in range(len(self.f24NSA)):
            outfile.write('%7.4e %7.4e %5.2e \n'%(self.f24NSA[i],self.f24NSAerr[i],self.snr24[i]))
        outfile.close()
                

mkw11=cluster('MKW11')
mkw8=cluster('MKW8')
awm4=cluster('AWM4')
a2052=cluster('A2052')
a2063=cluster('A2063')
ngc=cluster('NGC6107')
coma=cluster('Coma')
herc=cluster('Hercules')
a1367=cluster('A1367')

def copy_images_from_coma():
    mylocalclusters=[mkw8,awm4,ngc,herc,a1367]
    mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc,coma,herc,a1367]
    for cl in mylocalclusters:
        s='scp -r coma:research/LocalClusters/Images/'+cl.prefix+'/24umWCS ~/research/LocalClusters/Images/'+cl.prefix+'/.'
        # just get apex tables
        s='scp -r coma:research/LocalClusters/Images/'+cl.prefix+'/24umWCS/'+cl.prefix+'-WCS-mosaic_extract.tbl ~/research/LocalClusters/Images/'+cl.prefix+'/24umWCS/.'
        os.system(s)

def measure_snr():
    mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc,coma,herc,a1367]
    #mylocalclusters=[mkw8,awm4,a2052,a2063,ngc,herc,a1367]
    for cl in mylocalclusters:
        cl.measuresnr24()
    

measure_snr()
#mkw11.measuresnr24()
