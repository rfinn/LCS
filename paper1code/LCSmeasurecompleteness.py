#!/usr/bin/env python

'''
GOAL:
  to measure the completeness of the 24um scans used for the LC survey

USEAGE:
  from the os command line
  
  LCSmeasurecompleteness.py MKW11

  this will measure the completeness for the MKW11 field

  all possible clusternames are 
  ['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']

PROCEDURE:
  - read in existing sextractor catalog for image
  - create catalog of new galaxies (point sources) that are 
      - placed at random positions w/in the image
      - have a random magnitude that is drawn randomly from flux range [fmin,fmax]
      - this program assumes that a coverage image exists
      - sources are placed only in areas w/sufficient coverage (ncov=6)
      - adds 100 sources per scan

  - add galaxies to the mips scan using iraf.artdata.mkobject
  - run sextractor on simulated image
  - search sextractor output to see if the artificial galaxies are detected 
  - loops through this process 10x to create a sample of 1000 artificial
  - plot results.  output plots:
      - compare input and recovered fluxes
      - plot fraction of galaxies detected versus flux

NOTES:
- adapting this code to put artificial disk galaxies in 
- want to see how recovered size (FLUX_RADIUS) compares to input size as a function of magnitude and size.
- let Re range from 1 to 30 arcsec
- let mag range as with completeness estimate, maybe going a little brighter
- let B/A range from .2 - 1
- let PA range from 0 - 90

'''

from pylab import *
import os
import sys
import atpy
from pyraf import iraf
import pyfits

from LCScommon import *
import mystuff

iraf.artdata()



zp=18.53
nstar=100
fmin=0.01
fmingal=2
fmax=20
minmag=zp-2.5*log10(fmax)
maxmag=zp-2.5*log10(fmin)
minmag=15.
maxmag=22.
maxmaggal=maxmag-4
minmaggal=minmag-4

rmin=1/mipspixelscale
rmax=30/mipspixelscale
BAmin=0.2
BAmax=1

PAmin=0
PAmax=180

# conversion between observed flux and micro-Jy
convfactor=141.086 

class cluster:
    def __init__(self,clustername):
        self.prefix=clustername
        self.figuredir=homedir+'research/LocalClusters/Completeness/'
        self.imagedir = homedir+'research/LocalClusters/Images/'+self.prefix+'/24umWCS/'
        self.inputimage=self.imagedir+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
        self.uncimage=self.imagedir+self.prefix+'-WCS-mosaic_unc.fits'
        self.artimage=self.imagedir+'mosaic_completeness.fits'
        self.covimage=self.imagedir+self.prefix+'-WCS-mosaic_cov.fits'
        os.chdir(self.imagedir)

        # read in original sextractor catalog
        file=self.imagedir+self.prefix+'-test.cat'
        self.xgal0,self.ygal0,self.magbest0,self.magbesterr0,self.fluxbest0,self.fluxbesterr0,self.fluxradius0,self.BA0,self.PA0,self.PAerr0=self.readsexcat(file)

        #sexout=atpy.Table(file,type='ascii')
        #sexnumber=sexout['col1']
        #xgal0=sexout['col2']
        #ygal0=sexout['col3']
        #magbest0=sexout['col28']
        #magbesterr0=sexout['col29']
        #fluxbest0=sexout['col26']
        #fluxbesterr0=sexout['col27']

    def readsexcat(self, cat):
        infile=open(cat,'r')
        ngal=0
        for line in infile:
            if line.startswith('#'):
                continue
            ngal += 1
        infile.close()

        infile=open(cat,'r')
        xgal0=zeros(ngal,'f')
        ygal0=zeros(ngal,'f')
        magbest0=zeros(ngal,'f')
        magbesterr0=zeros(ngal,'f')
        fluxbest0=zeros(ngal,'f')
        fluxbesterr0=zeros(ngal,'f')
        fluxradius0=zeros(ngal,'f')
        BA0=zeros(ngal,'f')
        PA0=zeros(ngal,'f')
        PAerr0=zeros(ngal,'f')
        i=0
        for line in infile:
            if line.startswith('#'):
                continue
            t=line.split()
            xgal0[i]=float(t[1])
            ygal0[i]=float(t[2])
            fluxbest0[i]=float(t[21])
            fluxbesterr0[i]=float(t[22])
            magbest0[i]=float(t[23])
            magbesterr0[i]=float(t[24])
            fluxradius0[i]=float(t[67])
            BA0[i]=1-float(t[49])
            PA0[i]=float(t[42])
            PAerr0[i]=float(t[43])

            i += 1
        infile.close()
        return xgal0,ygal0,magbest0,magbesterr0,fluxbest0,fluxbesterr0,fluxradius0,BA0,PA0,PAerr0


        
    def createartcat(self,nstar=100,covimageflag=0,covimage='null',ncov=6,galcatflag=0):
        # add sources at random positions on image
        # even if a previous source is there, we need to know how crowding affects completeness
        # if there is a covereage map, like in the case of the rotated 24um image
        # need to take into account that some areas of the image have no exposure

        # ncov = min number of images

        iraf.imgets(image=self.inputimage,param='naxis1')#get RA of image
        xmax=float(iraf.imgets.value)
        iraf.imgets(image=self.inputimage,param='naxis2')#get RA of image
        ymax=float(iraf.imgets.value)

        xpos=[]
        ypos=[]
        mag=[]
        rad=[]
        BA=[]
        PA=[]

        if covimageflag:
            covimage=pyfits.open(covimage)
            cov_data=(covimage[0].data)
            covimage.close()

        i=0
        while i < nstar:
            xtemp=int(round(1+rand()*(xmax-2)))
            ytemp=int(round(1+rand()*(ymax-2)))
            if covimageflag:
                # check coverage map to make sure point is on science area
                # coverage map values > 6 are ok
                if cov_data[ytemp,xtemp] < ncov:
                    continue
            xpos.append(xtemp)
            ypos.append(ytemp)
            # create magnitudes that sample linearly w/flux
            #flux=fmin+rand()*(fmax-fmin) 
            #m=zp-2.5*log10(flux)
            if galcatflag:
                m=minmaggal+rand()*(maxmaggal-minmaggal)
            else:
                m=minmag+rand()*(maxmag-minmag)
            mag.append(m)
            r=rmin+rand()*(rmax-rmin)
            rad.append(r)
            b=BAmin+rand()*(BAmax-BAmin)
            BA.append(b)
            p=PAmin+rand()*(PAmax-PAmin)
            PA.append(p)
            i=i+1
        self.xpos=array(xpos,'f')
        self.ypos=array(ypos,'f')
        mag=array(mag,'f')
        rad=array(rad,'f')
        BA=array(BA,'f')
        PA=array(PA,'f')
        if galcatflag:
            outf='galaxies.coords.dat'
        else:
            outf='stars.coords.dat'
        outfile=open(self.imagedir+outf,'w')
        for i in range(len(self.xpos)):
            if galcatflag:
                # x, y, mag, model (expdisk/devauc) radius B/A pa 
                s='%8.2f %8.2f %8.2f expdisk %6.2f %5.2f %6.2f\n'%(self.xpos[i],self.ypos[i],mag[i], rad[i],BA[i], PA[i])
            else:
                s='%8.2f %8.2f %8.2f \n'%(self.xpos[i],self.ypos[i],mag[i])
            outfile.write(s)
        outfile.close()


    def addgalaxies(self,simimage,objectcat='stars.coords.dat'):

        if os.path.exists(simimage):
            os.remove(simimage)
        os.system('cp /Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits .')#convolve star w/SSC PRF
        iraf.mkobjects(input=self.inputimage,output=simimage,objects=objectcat,radius=15.875,star='mips24_prf_mosaic_2.45_4x.fits',magzero=zp,background=0.,gain=5.,rdnoise=0.,poisson='no')
        #iraf.display(self.inputimage,1,contrast=0.01,fill='yes')
        #iraf.display(simimage,2,contrast=0.01,fill='yes')
        #iraf.display(simimage,3,contrast=0.01,fill='yes')
        #iraf.tvmark(1,'stars.coords.dat',mark='circle',radii=20,color='204')
        #iraf.tvmark(2,'stars.coords.dat',mark='circle',radii=20,color='204')

    def runsextractor(self):
        sex_image=self.artimage
        #sex_image=self.inputimage
        catalogname=self.imagedir+self.prefix+'-artdata-test.cat'
        os.system('cp ~/research/LocalClusters/sextractor/default.param .')
        os.system('cp ~/research/LocalClusters/sextractor/default.conv .')
        os.system('cp ~/research/LocalClusters/sextractor/default.nnw .')
        #os.system('sex %s -c default.sex.sdss.galfit -CATALOG_NAME %s -CHECKIMAGE_NAME check-sim.fits, segmentation-sim.fits'%(sex_image,catalogname))
	s='sex '+sex_image+' -c '+homedir+'research/LocalClusters/sextractor/default.sex.24um.galfit -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+self.uncimage+' -CATALOG_NAME '+catalogname+' -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME check-sim.fits'
        print s
	os.system(s)


    def findgalaxies(self,objectcat='stars.coords.dat',galflag=0):
        # read in sextractor catalog w/artificial sources
        file=self.imagedir+self.prefix+'-artdata-test.cat'
        xgal1,ygal1,magbest1,magbesterr1,fluxbest1,fluxbesterr1,fluxradius1,BA1,PA1,PAerr1=self.readsexcat(file)
        #sexout=atpy.Table(self.imagedir+file,type='ascii')
        #sexnumber=sexout['col1']
        #xgal1=sexout['col2']
        #ygal1=sexout['col3']
        #magbest1=sexout['col28']
        #magbesterr1=sexout['col29']
        #fluxbest1=sexout['col26']
        #fluxbesterr1=sexout['col27']

        # read in positions of artificial sources
        artdat=atpy.Table(self.imagedir+objectcat,type='ascii')
        artx=artdat['col1']
        arty=artdat['col2']
        artmag=artdat['col3']
        if galflag:
            artrad=artdat['col5']
            artBA=artdat['col6']
            artPA=artdat['col7']
        # loop through list of artificial sources and see if artificial sources are detected
        newgalflag=ones(len(xgal1),'bool')
        delta=1.#max number of pixels for a match
        newgalflag=ones(len(xgal1),'bool')
        for i in range(len(xgal1)):
            (imatch, matchflag,nmatch)=findnearest(xgal1[i],ygal1[i],self.xgal0,self.ygal0,delta)
            if matchflag > 0.:
                #print fluxbest1[i],self.fluxbest0[imatch]
                dflux=abs(fluxbest1[i] - self.fluxbest0[imatch])/self.fluxbest0[imatch]
                #print i,fluxbest1[i],self.fluxbest0[imatch],dflux
                if dflux < .1:#position of real galaxy, flux difference less than 15% -> not a new galaxy
                    newgalflag[i] = 0

        print 'number of galaxies in original sextractor catalog = ',len(self.xgal0)
        print 'number of galaxies detected = ',len(xgal1)
        xmeas=xgal1[newgalflag]
        print 'number of galaxies that area new = ',len(xmeas)
        ymeas=ygal1[newgalflag]
        magbestmeas=magbest1[newgalflag]
        magbestmeaserr=magbesterr1[newgalflag]
        fluxbestmeas=fluxbest1[newgalflag]
        fluxbestmeaserr=fluxbesterr1[newgalflag]
        fluxradiusmeas=fluxradius1[newgalflag]
        BAmeas=BA1[newgalflag]
        PAmeas=PA1[newgalflag]

        delta=2.
        artmagmeas=zeros(len(artmag),'f')
        artmagmeaserr=zeros(len(artmag),'f')
        artfluxmeas=zeros(len(artmag),'f')
        artfluxmeaserr=zeros(len(artmag),'f')
        artfluxradiusmeas=zeros(len(artmag),'f')
        artBAmeas=zeros(len(artmag),'f')
        artPAmeas=zeros(len(artmag),'f')
        artmatchflag=zeros(len(artmag),'i')
        artmatchindex=zeros(len(artmag),'i')
        for i in range(len(artx)):
            (imatch, matchflag,nmatch)=findnearest(artx[i],arty[i],xmeas,ymeas,delta)
            #print i,matchflag,artx[i],arty[i],artmag[i]
            if matchflag:
                artmatchflag[i]=1
                artmatchindex[i]=imatch
                artmagmeas[i]=magbestmeas[imatch]
                artmagmeaserr[i]=magbestmeaserr[imatch]
                artfluxmeas[i]=fluxbestmeas[imatch]
                artfluxmeaserr[i]=fluxbestmeaserr[imatch]
                artfluxradiusmeas[i]=fluxradiusmeas[imatch]
                artBAmeas[i]=BAmeas[imatch]
                artPAmeas[i]=PAmeas[imatch]
        if galflag:
            return artx,arty,artmag,artrad,artBA,artPA,artmatchflag,artmagmeas,artmagmeaserr,artfluxmeas,artfluxmeaserr,artfluxradiusmeas,artBAmeas,artPAmeas
        else:
            return artx,arty,artmag,artmatchflag,artmagmeas,artmagmeaserr,artfluxmeas,artfluxmeaserr,artfluxradiusmeas,artBAmeas,artPAmeas
			    

    def plotresults(self):

        #make plots using all realizations 
        cla()
        clf()
        fsim=fsim*convfactor
        fs=compress((matchflagsim > 0.1) & (deblendsim < 1.5),fsim)
        #f1=compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas1)
        f2=compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas2)
        f3=compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas3)
        f4=compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas4)
        #f242=compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas24)

        r4=median(fs/f4)
        r3=median(fs/f3)
        r2=median(fs/f2)
        print "average ratios ap 4",average(fs/f4),r4,std((fs/f4)/average(fs/f2))
        print "average ratios ap 3",average(fs/f3),median(fs/f3),std((fs/f3)/average(fs/f3))
        print "average ratios ap 2",average(fs/f2),median(fs/f2),std((fs/f2)/average(fs/f2))

        s='f4 w/apcor = %3.2f(%4.2f)'%(r4,average(abs(fs-f4*r4)/fs))
        plot(fs,f4*r4,'b.',label=s)
        plot(fs,f4,'bo',label='f4')
        s='f3 w/apcor = %3.2f(%4.2f)'%(r3,average(abs(fs-f3*r3)/fs))
        plot(fs,f3*r3,'g.',label=s)
        plot(fs,f3,'go',label='f3')
        s='f2 w/apcor = %3.2f(%4.2f)'%(r2,average(abs(fs-f2*r2)/fs))
        plot(fs,f2*r2,'r.',label=s)
        plot(fs,f2,'ro',label='f2')
        legend(loc='best')
        x=arange(0.,max(fs),10.)
        y=x
        plot(x,y,'k-')
        xlabel('F(24) Input')
        ylabel('F(24) measured')
        #axis([0.,50.,0.,50.])
        s=self.figuredir+str(self.prefix)+'fluxcomp.eps'
        savefig(s)

    def plotcompleteness(self):
        figure(figsize=(10,8))
        cla()
        clf()
        nbins=20
        minflux=fmin
        maxflux=fmax
        df=1.#(fmax-fmin)/(1.*nbins)
        dmag=.25
        bins=arange(minflux,(maxflux+df),df)
        #bins=arange(maxmag,(minmag+dmag),dmag)
        (xbin,ybin,ybinerr)=mystuff.completeness(bins,self.fsim,self.matchflag)
        #(xbin,ybin,ybinerr)=mystuff.completeness(bins,self.magsim,self.matchflag)
        s=self.figuredir+str(self.prefix)+'-FracComplvsFlux.dat'
        outdat=open(s,'w')
        print "Completeness vs Input Flux"
        for i in range(len(xbin)):
	    #print i, xbin[i],ybin[i],ybinerr[i]
	    t='%8.2f %8.4f %8.4f\n'%(xbin[i],ybin[i],ybinerr[i])
	    outdat.write(t)
        outdat.close()
        cla()
        clf()
        plot(xbin,ybin,'ko')
        errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k')

        axhline(y=1.0,ls='-')
        axhline(y=.8,ls='--')
        axvline(x=80.0,ls=':',color='b')
        xlabel('$Input \ Flux (\mu Jy)$',fontsize=22)
        ylabel('$Completeness$',fontsize=22)
        title(self.prefix)
        axis([0.,max(xbin)+df,-.05,1.05])

        s=self.figuredir+str(self.prefix)+'-FracComplvsFlux.eps'
        savefig(s)

        figure(figsize=(10,8))
        cla()
        clf()
        x=self.fsim[self.matchflag]
        y=self.fmeas[self.matchflag]
        erry=self.fmeaserr[self.matchflag]
        subplot(2,1,1)

        plot(x,y,'k.')
        errorbar(x,y,yerr=erry,fmt=None,ecolor='k')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.5)
        plot(xl,xl,'k-')

        ylabel('$Measured \ Flux $',fontsize=22)
        title(self.prefix)
        subplot(2,1,2)
        diff=(y-x)/x*100
        plot(x,diff,'k.')
        axhline(y=0)
        xmin,xmax=xlim()
        axis([xmin,xmax,-50,50])
        xlabel('$Input \ Flux (\mu Jy)$',fontsize=22)
        ylabel('$\% Difference \ (obs-input)/input $',fontsize=22)

        s=self.figuredir+str(self.prefix)+'-SimvsMeasFlux.eps'
        savefig(s)
    
        #os.system('cp *.eps /Users/rfinn/clusters/spitzer/completeness/.')

    def plotgalsim(self):

        infile=self.figuredir+self.prefix+'-galaxy-sim.fits'
        galsim=atpy.Table(infile)
        # 'XSIM',self.xsim,unit='pixel',description='xcoord of simulated galaxies')
        #'YSIM',self.ysim,unit='pixel',description='xcoord of simulated galaxies')
        #'FLUX_SIM',self.fsim,unit='MJy/sr',description='')
        #'MAGSIM',self.magsim,unit='mag',description='mag of simulated galaxies')
        #'RADSIM',self.radsim,unit='pixel',description='Re of simulated galaxies')
        #'BASIM',self.BAsim,unit='',description='B/A of simulated galaxies')
        #'PASIM',self.PAsim,unit='deg',description='PA of simulated galaxies')
        #'MATCHFLAG',self.matchflag,unit='bool',description='recovered?')
        #'MAGMEAS',self.magmeas,unit='mag',description='')
        #'MAGMEAS_ERR',self.magmeaserr,unit='mag',description='')
        #'FLUXMEAS',self.fmeas,unit='MJy/sr',description='')
        #'FLUXMEAS_ERR',self.fmeaserr,unit='MJy/sr',description='')
        #'RADIUSMEAS',self.radmeas,unit='pixel',description='')
        #'BAMEAS',self.BAmeas,unit='',description='')
        #'PAMEAS',self.PAmeas,unit='deg',description='')


        self.fsim=galsim

        figure(figsize=(10,8))
        subplots_adjust(wspace=.35,hspace=.35)
        cla()
        clf()

        x=[galsim.FLUX_SIM,galsim.RADSIM,galsim.BASIM,galsim.PASIM]
        y=[galsim.FLUXMEAS,galsim.RADMEAS,galsim.BAMEAS,galsim.PAMEAS]
        ylab='$Completeness$'
        xlab=['$Sim \ Flux $','$Sim \ Radius $','$Sim \ B/A $','$Sim \ PA $']

        for i in range(4):
            subplot(2,2,i+1)

            nbins=20
            minflux=min(x[i])
            maxflux=max(x[i])
            df=1.#(fmax-fmin)/(1.*nbins)
            dmag=.25
            df=1.*(maxflux-minflux)/20.
            bins=arange(minflux,(maxflux+df),df)

            (xbin,ybin,ybinerr)=mystuff.completeness(bins,x[i],galsim.MATCHFLAG)

            plot(xbin,ybin,'ko')
            errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k')

            axhline(y=1.0,ls='-')
            axhline(y=.8,ls='--')
            
            axis([0.,max(xbin)+df,-.05,1.05])

            ylabel(ylab,fontsize=22)
            xlabel(xlab[i],fontsize=22)
            if i  == 0:
                title('$'+self.prefix+'$',fontsize=22)
                #subplot(5,5,2)
                #scatter(x[i],y[i],c=log10(galsim.FLUX_SIM),vmin=0,vmax=2.7,cmap='jet')
                #xmin,xmax=xlim()
                #xl=arange(xmin,xmax,.1)
                #plot(xl,xl,'k-')
                #axis([0,50,0,50])

            #    ax=gca()
            #    ax.set_xscale('log')
            #    ax.set_yscale('log')


        s=self.figuredir+str(self.prefix)+'-GalSimFracCompl.eps'
        savefig(s)

        figure(figsize=(10,8))

        subplots_adjust(wspace=.35,hspace=.35)
        cla()
        clf()
        x=[galsim.FLUX_SIM,galsim.RADSIM,galsim.BASIM,galsim.PASIM]
        y=[galsim.FLUXMEAS,galsim.RADMEAS,galsim.BAMEAS,galsim.PAMEAS]
        ylab=['$Measured \ Flux $','$Measured \ Radius $','$Measured \ B/A $','$Measured \ PA $']
        xlab=['$Sim \ Flux $','$Sim \ Radius $','$Sim \ B/A $','$Sim \ PA $']

        for i in range(4):
            subplot(2,2,i+1)
            scatter(x[i],y[i],c=log10(galsim.FLUX_SIM),vmin=0,vmax=3.,cmap='jet')
            xmin,xmax=xlim()
            xl=arange(xmin,xmax,.1)
            plot(xl,xl,'k-')
            ylabel(ylab[i],fontsize=22)
            xlabel(xlab[i],fontsize=22)
            if i  == 0:
                title('$'+self.prefix+'$',fontsize=22)
                #subplot(5,5,2)
                #scatter(x[i],y[i],c=log10(galsim.FLUX_SIM),vmin=0,vmax=2.7,cmap='jet')
                #xmin,xmax=xlim()
                #xl=arange(xmin,xmax,.1)
                #plot(xl,xl,'k-')
                #axis([0,50,0,50])

            #    ax=gca()
            #    ax.set_xscale('log')
            #    ax.set_yscale('log')

        s=self.figuredir+str(self.prefix)+'-GalSimvsMeas.eps'
        savefig(s)
    
        #os.system('cp *.eps /Users/rfinn/clusters/spitzer/completeness/.')

    def completeness_star(self):#measure completeness on final image where sextractor is doing detection
        os.chdir(self.imagedir)
        x=[]
        y=[]
        mag=[]
        matchflag=[]
        magmeas=[]
        magmeaserr=[]
        fluxmeas=[]
        fluxmeaserr=[]
        for i in range(10):
            self.createartcat(covimageflag=1,covimage=self.covimage,ncov=6)
            self.addgalaxies(self.artimage)
            self.runsextractor()
            artx,arty,artmag,artmatchflag,artmagmeas,artmagmeaserr,artfluxmeas,artfluxmeaserr,artfluxradiusmeas,artBAmeas,artPAmeas=self.findgalaxies()


            x=x+artx.tolist()
            y=y+arty.tolist()
            mag=mag+artmag.tolist()
            matchflag=matchflag+artmatchflag.tolist()
            magmeas=magmeas+artmagmeas.tolist()
            magmeaserr=magmeaserr+artmagmeaserr.tolist()
            fluxmeas=fluxmeas+artfluxmeas.tolist()
            fluxmeaserr=fluxmeaserr+artfluxmeaserr.tolist()
        self.x=x
        self.y=y
        self.mag=mag

        self.xsim=array(x,'f')
        self.ysim=array(y,'f')
        self.magsim=array(mag,'f')
        self.matchflag=array(matchflag,'bool')
        self.magmeas=array(magmeas,'f')
        self.magmeaserr=array(magmeaserr,'f')
        self.fmeas=array(fluxmeas,'f')
        self.fmeaserr=array(fluxmeaserr,'f')
        self.fsim=10.**(-1*(self.magsim-zp)/2.5)
        compdat=atpy.Table()
        compdat.add_column('XSIM',self.xsim,unit='pixel',description='xcoord of simulated galaxies')
        compdat.add_column('YSIM',self.ysim,unit='pixel',description='xcoord of simulated galaxies')
        compdat.add_column('FLUX_SIM',self.fsim,unit='MJy/sr',description='')
        compdat.add_column('MAGSIM',self.magsim,unit='mag',description='mag of simulated galaxies')
        compdat.add_column('MATCHFLAG',self.matchflag,unit='bool',description='recovered?')
        compdat.add_column('MAGMEAS',self.magmeas,unit='mag',description='')
        compdat.add_column('MAGMEAS_ERR',self.magmeaserr,unit='mag',description='')
        compdat.add_column('FLUXMEAS',self.fmeas,unit='MJy/sr',description='')
        compdat.add_column('FLUXMEAS_ERR',self.fmeaserr,unit='MJy/sr',description='')


        s=self.figuredir+self.prefix+'-completeness.fits'
        if os.path.exists(s):
            os.remove(s)
        compdat.write(self.figuredir+self.prefix+'-completeness.fits',type='fits')
        self.plotcompleteness()

    def completeness_gal(self):#measure completeness on final image where sextractor is doing detection
        os.chdir(self.imagedir)
        x=[]
        y=[]
        mag=[]
        rad=[]
        BA=[]
        PA=[]
        matchflag=[]
        magmeas=[]
        magmeaserr=[]
        fluxmeas=[]
        fluxmeaserr=[]
        fluxradmeas=[]
        BAmeas=[]
        PAmeas=[]
        for i in range(5):
            self.createartcat(covimageflag=1,nstar=100,covimage=self.covimage,ncov=6,galcatflag=1)
            self.addgalaxies(self.artimage,objectcat='galaxies.coords.dat')
            self.runsextractor()
            artx,arty,artmag,artrad,artBA,artPA,artmatchflag,artmagmeas,artmagmeaserr,artfluxmeas,artfluxmeaserr,artfluxradiusmeas,artBAmeas,artPAmeas=self.findgalaxies(objectcat='galaxies.coords.dat',galflag=1)
            x=x+artx.tolist()
            y=y+arty.tolist()
            mag=mag+artmag.tolist()
            rad=rad+artrad.tolist()
            BA=BA+artBA.tolist()
            PA=PA+artPA.tolist()
            matchflag=matchflag+artmatchflag.tolist()
            magmeas=magmeas+artmagmeas.tolist()
            magmeaserr=magmeaserr+artmagmeaserr.tolist()
            fluxmeas=fluxmeas+artfluxmeas.tolist()
            fluxmeaserr=fluxmeaserr+artfluxmeaserr.tolist()
            fluxradmeas=fluxradmeas+artfluxradiusmeas.tolist()
            BAmeas=BAmeas+artBAmeas.tolist()
            PAmeas=PAmeas+artPAmeas.tolist()
        self.x=x
        self.y=y
        self.mag=mag

        self.xsim=array(x,'f')
        self.ysim=array(y,'f')
        self.magsim=array(mag,'f')
        self.radsim=array(rad,'f')
        self.BAsim=array(BA,'f')
        self.PAsim=array(PA,'f')
        self.matchflag=array(matchflag,'bool')
        self.magmeas=array(magmeas,'f')
        self.magmeaserr=array(magmeaserr,'f')
        self.fmeas=array(fluxmeas,'f')
        self.fmeaserr=array(fluxmeaserr,'f')
        self.fsim=10.**(-1*(self.magsim-zp)/2.5)
        self.radmeas=array(fluxradmeas,'f')
        self.BAmeas=array(BAmeas,'f')
        self.PAmeas=array(PAmeas,'f')
        compdat=atpy.Table()
        compdat.add_column('XSIM',self.xsim,unit='pixel',description='xcoord of simulated galaxies')
        compdat.add_column('YSIM',self.ysim,unit='pixel',description='xcoord of simulated galaxies')
        compdat.add_column('FLUX_SIM',self.fsim,unit='MJy/sr',description='')
        compdat.add_column('MAGSIM',self.magsim,unit='mag',description='mag of simulated galaxies')
        compdat.add_column('RADSIM',self.radsim,unit='pixel',description='Re of simulated galaxies')
        compdat.add_column('BASIM',self.BAsim,unit='',description='B/A of simulated galaxies')
        compdat.add_column('PASIM',self.PAsim,unit='deg',description='PA of simulated galaxies')
        compdat.add_column('MATCHFLAG',self.matchflag,unit='bool',description='recovered?')
        compdat.add_column('MAGMEAS',self.magmeas,unit='mag',description='')
        compdat.add_column('MAGMEAS_ERR',self.magmeaserr,unit='mag',description='')
        compdat.add_column('FLUXMEAS',self.fmeas,unit='MJy/sr',description='')
        compdat.add_column('FLUXMEAS_ERR',self.fmeaserr,unit='MJy/sr',description='')
        compdat.add_column('RADMEAS',self.radmeas,unit='pixel',description='')
        compdat.add_column('BAMEAS',self.BAmeas,unit='',description='')
        compdat.add_column('PAMEAS',self.PAmeas,unit='deg',description='')

        s=self.figuredir+self.prefix+'-galaxy-sim.fits'
        if os.path.exists(s):
            os.remove(s)
        compdat.write(s,type='fits')
        self.plotgalsim()

clusternames=['MKW11','MKW8']

mkw11=cluster('MKW11')
#mkw11.completeness_gal()

print sys.argv, len(sys.argv)
if len(sys.argv) > 1:
    prefix=sys.argv[1]
    cl=cluster(prefix)
    #cl.completeness_star()
    cl.completeness_gal()

def runalllcs():
    mkw8=cluster('MKW8')
    mkw8.completeness_star()
    ngc=cluster('NGC6107')
    ngc.completeness_star()
    
