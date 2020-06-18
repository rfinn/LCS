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

def findnearest(x1,y1,x2,y2,delta):#use where command
	matchflag=1
	nmatch=0
	d=sqrt((x1-x2)**2 + (y1-y2)**2)#x2 and y2 are arrays
	index=arange(len(d))
	t=index[d<delta]
	matches=t
	if len(matches) > 0:
		nmatch=len(matches)
		if nmatch > 1:
			imatch=index[(d == min(d[t]))]
		else:
			imatch=matches[0]			
	else:
		imatch = 0
		matchflag = 0

	return imatch, matchflag,nmatch


class cluster:
    def __init__(self,clustername):
        self.prefix=clustername
        self.image24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_minus_median_extract.fits'
        self.noise24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_unc.fits'
        self.rotatedimage24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'

        self.imagepath24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'
        self.sdssimagepath=homedir+'research/LocalClusters/Images/'+self.prefix+'/SDSS/'
        self.sex_image=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        self.unc_image=homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_unc.fits'

        infile=homedir+'research/LocalClusters/NSAmastertables/NSAwithAGC/'+clustername+'_NSAmastertable_topcat.fits'

        self.n=atpy.Table(infile)


    def runsextractor24(self):
        os.chdir(self.imagepath24)
        #cp sextractor files to this director
        os.system('cp '+homedir+'research/LocalClusters/sextractor/default.param .')
        os.system('cp '+homedir+'research/LocalClusters/sextractor/default.nnw .')
        s='sex '+self.sex_image+' -c '+homedir+'research/LocalClusters/sextractor/default.sex.24um.galfit -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+self.unc_image
        os.system(s)

	#save test.cat
        s='cp test.cat '
        testcat24=self.prefix+'-test.cat'
        os.rename('test.cat',testcat24)
    def readsexcat(self):
        self.sex=atpy.Table(self.imagepath24+self.prefix+'-test.cat',type='ascii')
        print 'writing new table'
        newtab=atpy.Table()
        newtab.add_column('x',self.sex['col2'])
        newtab.add_column('y',self.sex['col3'])
        outfile=self.prefix+'_sex_xy.txt'
        if os.path.exists(outfile):
            os.remove(outfile)
        newtab.write(outfile,type='ascii')
    def displayresults(self):
        iraf.display(self.sex_image,1,fill='yes',contrast=0.01)
        iraf.display(self.sex_image,2,fill='yes',contrast=0.01)
        iraf.display('check.fits',3,fill='yes',contrast=0.01)
        iraf.display('background.fits',4,contrast=0.01)
        iraf.tvmark(2,self.prefix+'_sex_xy.txt',color=204,radii=2,mark='circle')
    def match2NSA(self):

        delta=10./3600.
        self.imatch=zeros(len(self.n.RA),'i')
        self.nmatch=zeros(len(self.n.RA),'i')
        self.matchflag=zeros(len(self.n.RA),'bool')

        for i in range(len(self.n.RA)):
            if (self.prefix.find('MKW11')>-1) & (i == 308):
                self.imatch[i], self.matchflag[i],self.nmatch[i]= findnearest(self.n.RA[i],self.n.DEC[i],self.sex['col8'],self.sex['col9'],30./3600.)
                print 'found a match for ',self.prefix,' ',self.n.NSAID[i]
            else:
                self.imatch[i], self.matchflag[i],self.nmatch[i]= findnearest(self.n.RA[i],self.n.DEC[i],self.sex['col8'],self.sex['col9'],delta)
        
        stable=atpy.Table()
        stable.add_column('NSA_RA',self.n.RA)
        stable.add_column('NSA_DEC',self.n.DEC)
        stable.add_column('NSAID',self.n.NSAID)
        stable.add_column('MATCHFLAG24',self.matchflag)
        stable.add_column('NMATCH24',self.nmatch)
        stable.add_column('MATCH_INDEX24',self.imatch)
        t=self.sex['col2']
        stable.add_column('X_IMAGE',t[self.imatch]*self.matchflag)
        t=self.sex['col3']
        stable.add_column('Y_IMAGE',t[self.imatch]*self.matchflag)
        t=self.sex['col8']
        stable.add_column('RA',t[self.imatch]*self.matchflag)
        t=self.sex['col9']
        stable.add_column('DEC',t[self.imatch]*self.matchflag)
        t=self.sex['col10']
        stable.add_column('FLUX_ISO',t[self.imatch]*self.matchflag)
        t=self.sex['col11']
        stable.add_column('FLUXERR_ISO',t[self.imatch]*self.matchflag)
        t=self.sex['col12']
        stable.add_column('MAG_ISO',t[self.imatch]*self.matchflag)
        t=self.sex['col13']
        stable.add_column('MAGERR_ISO',t[self.imatch]*self.matchflag)

        t=self.sex['col14']
        stable.add_column('FLUX_ISOCOR',t[self.imatch]*self.matchflag)
        t=self.sex['col15']
        stable.add_column('FLUXERR_ISOCOR',t[self.imatch]*self.matchflag)
        t=self.sex['col16']
        stable.add_column('MAG_ISOCOR',t[self.imatch]*self.matchflag)
        t=self.sex['col17']
        stable.add_column('MAGERR_ISOCOR',t[self.imatch]*self.matchflag)


        t=self.sex['col18']
        stable.add_column('FLUX_AUTO',t[self.imatch]*self.matchflag)
        t=self.sex['col19']
        stable.add_column('FLUXERR_AUTO',t[self.imatch]*self.matchflag)
        t=self.sex['col20']
        stable.add_column('MAG_AUTO',t[self.imatch]*self.matchflag)
        t=self.sex['col21']
        stable.add_column('MAGERR_AUTO',t[self.imatch]*self.matchflag)

        t=self.sex['col22']
        stable.add_column('FLUX_BEST',t[self.imatch]*self.matchflag)
        t=self.sex['col23']
        stable.add_column('FLUXERR_BEST',t[self.imatch]*self.matchflag)
        t=self.sex['col24']
        stable.add_column('MAG_BEST',t[self.imatch]*self.matchflag)
        t=self.sex['col25']
        stable.add_column('MAGERR_BEST',t[self.imatch]*self.matchflag)

        t=self.sex['col26']
        stable.add_column('KRON_RADIUS',t[self.imatch]*self.matchflag)
        t=self.sex['col27']
        stable.add_column('PETRO_RADIUS',t[self.imatch]*self.matchflag)

        t=self.sex['col28']
        stable.add_column('FLUX_PETRO',t[self.imatch]*self.matchflag)
        t=self.sex['col29']
        stable.add_column('FLUXERR_PETRO',t[self.imatch]*self.matchflag)
        t=self.sex['col30']
        stable.add_column('MAG_PETRO',t[self.imatch]*self.matchflag)
        t=self.sex['col31']
        stable.add_column('MAGERR_PETRO',t[self.imatch]*self.matchflag)


        t=self.sex['col35']
        stable.add_column('FLUX_MAX',t[self.imatch]*self.matchflag)

        t=self.sex['col36']
        stable.add_column('MU_MAX',t[self.imatch]*self.matchflag)

        t=self.sex['col37']
        stable.add_column('ISOAREA_IMAGE',t[self.imatch]*self.matchflag)


        t=self.sex['col43']
        stable.add_column('THETA_IMAGE',t[self.imatch]*self.matchflag)
        t=self.sex['col44']
        stable.add_column('ERRTHETA_IMAGE',t[self.imatch]*self.matchflag)

        t=self.sex['col47']
        stable.add_column('THETA_J2000',t[self.imatch]*self.matchflag)
        t=self.sex['col48']
        stable.add_column('ERRTHETA_J2000',t[self.imatch]*self.matchflag)
        t=self.sex['col49']
        stable.add_column('ELONGATION',t[self.imatch]*self.matchflag)

        t=self.sex['col50']
        stable.add_column('ELLIPTICITY',t[self.imatch]*self.matchflag)
        t=self.sex['col51']
        stable.add_column('FWHM_IMAGE',t[self.imatch]*self.matchflag)
        t=self.sex['col52']
        stable.add_column('FWHM_DEG',t[self.imatch]*self.matchflag)
        t=self.sex['col53']
        stable.add_column('FLAGS',t[self.imatch]*self.matchflag)
        t=self.sex['col54']
        stable.add_column('CLASS_STAR',t[self.imatch]*self.matchflag)


        t=self.sex['col55']
        stable.add_column('FLUX_APER1',t[self.imatch]*self.matchflag)
        t=self.sex['col56']
        stable.add_column('FLUX_APER2',t[self.imatch]*self.matchflag)
        t=self.sex['col57']
        stable.add_column('FLUX_APER3',t[self.imatch]*self.matchflag)
        t=self.sex['col58']
        stable.add_column('FLUXERR_APER1',t[self.imatch]*self.matchflag)
        t=self.sex['col59']
        stable.add_column('FLUXERR_APER2',t[self.imatch]*self.matchflag)
        t=self.sex['col60']
        stable.add_column('FLUXERR_APER3',t[self.imatch]*self.matchflag)
        t=self.sex['col61']
        stable.add_column('MAG_APER1',t[self.imatch]*self.matchflag)
        t=self.sex['col62']
        stable.add_column('MAG_APER2',t[self.imatch]*self.matchflag)
        t=self.sex['col63']
        stable.add_column('MAG_APER3',t[self.imatch]*self.matchflag)
        t=self.sex['col64']
        stable.add_column('MAGERR_APER1',t[self.imatch]*self.matchflag)
        t=self.sex['col65']
        stable.add_column('MAGERR_APER2',t[self.imatch]*self.matchflag)
        t=self.sex['col66']
        stable.add_column('MAGERR_APER3',t[self.imatch]*self.matchflag)
        t=self.sex['col67']
        stable.add_column('FLUX_RADIUS1',t[self.imatch]*self.matchflag)
        t=self.sex['col68']
        stable.add_column('FLUX_RADIUS2',t[self.imatch]*self.matchflag)
        t=self.sex['col69']
        stable.add_column('FLUX_RADIUS3',t[self.imatch]*self.matchflag)


        

        outfile=homedir+'research/LocalClusters/NSAmastertables/Sex24Tables/'+self.prefix+'_sex24.fits'
        if os.path.exists(outfile):
            os.remove(outfile)

        stable.write(outfile)

for cname in clusternames:
    cl=cluster(cname)
    #cl.runsextractor24()
    cl.readsexcat()
    cl.match2NSA()


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
