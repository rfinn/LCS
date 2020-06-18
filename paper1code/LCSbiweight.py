#!/usr/bin/env python

'''
    Written by Rose A. Finn, updated on January 5, 2015

    PURPOSE: 
      This program calculates the biweight center and scale for the LCS clusters
      using existing programs from the astropy.stats package.  Errors on the center
      and scale are calculating using bootstrap resampling (1000 samples is the default).

      
    CALLING SEQUENCE

       from within ipython

       % run  ~/svnRepository/pythonCode/LCSbiweight.py
       getbiweightall()

       to see the results plotted with velocity histogram for one cluster

       mkw11.plotvhist()

       to see a multipanel plot for all clusters, type

       plotall()

    INPUT PARAMETERS
      none
      
    OUTPUT PARAMETERS
      none
    
    EXAMPLES
      see calling sequence
    
    PROCEDURE

      The NSA catalog is cut to include only those galaxies with velocities within
      4000 km/s of the central velocity (from the literature) and within a projected
      radius of 1 degree.  The biweight location and scale are calculated, and the
      location is used as the median, and the biweight location and scale are recalculated.
      This is repeated until the scale changes by less than 1 km/s.  This typically requires
      2 iterations.

    REQUIRED PYTHON MODULES
      numpy
      pylab
      astropy
      scipy

    ADDITIONAL REQUIRED MODULES
      LCScommon
      LCSReadmasterBaseNSA

    NOTES
      Required Files

        homedir+'research/LocalClusters/NSAmastertables/NSAwithAGC/'+clustername+'_NSAmastertable_topcat.fits'

'''

import numpy as np
import pylab as pl
import scipy.stats
from astropy.stats import biweight_location, biweight_midvariance, sigma_clip, bootstrap
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from LCScommon import *
from LCSReadmasterBaseNSA import *
import mystuff as my
dz=4000./3.e5
dtheta=1.
scale_cut=3.

def centralbi(x, M=None):
    x=np.array(x,'f')
    if M is None:
        M=np.median(x)
    
    MAD=np.median(abs(x-M))
    ui=((x-M)/(6*MAD))
    top=np.sum((x-M)*((1-ui**2)**2))
    bottom=np.sum((1-ui**2)**2)
    
    
    cbi=M + (top/bottom)
    #print self.clustername
    # print(cbi)
    
    #finds the biweight scale
    n=len(x)
    usbi=((x-M)/(9*MAD))
    upper= sum(((x-M)**2)*((1-usbi**2)**4))
    lower=sum((1-usbi**2)*(1-5*usbi**2))
    sbi=sqrt(n)*((sqrt(upper))/(abs(lower)))
    return cbi,sbi
def getbiweight(z):
    biweightscale=biweight_midvariance(z)
    biweightlocation=biweight_location(z)

    flag=abs(z-biweightlocation)/biweightscale < scale_cut
    #flag=np.ones(len(z),'bool')
    repeatflag=1
    nloop=0
    #print biweightlocation, biweightscale
    oldbiweightscale=biweightscale
    while repeatflag:
        newdata=z[flag]
        biweightscale=biweight_midvariance(newdata, M=biweightlocation)
        biweightlocation=biweight_location(newdata, M=biweightlocation)
        oldflag = flag
        #flag=abs(z-biweightlocation)/biweightscale < scale_cut
        nloop += 1
        #print nloop, biweightlocation, biweightscale, len(newdata), sum(flag)
        #if sum(flag == oldflag) == len(flag): 
        #    repeatflag=0
        #    print 'nloop = ', nloop
        if abs(biweightscale - oldbiweightscale) < 1.: 
            repeatflag=0
            #print 'nloop = ', nloop
        if nloop > 5:
            repeatflag = 0
        oldbiweightscale=biweightscale
        #print nloop, biweightlocation, biweightscale
    #flag=abs(z-biweightlocation)/biweightscale < 4.
    #biweightscale=biweight_midvariance(z[flag],M=biweightlocation)
    return biweightlocation, biweightscale


class cluster():
    def __init__(self,clustername):
        self.prefix=clustername
        self.john_prefix=john_prefix[self.prefix]
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]
        self.biweightvel=clusterbiweightcenter[self.prefix]
        self.biweightscale=clusterbiweightscale[self.prefix]
        #infile=homedir+'research/LocalClusters/NSAmastertables/NSAwithAGC/'+clustername+'_NSAmastertable_topcat.fits'
        #self.n=atpy.Table(infile)
        infile=homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSA.fits'
        self.n=fits.getdata(infile)

        #infile=homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSAmastertable.fits'
        #self.nsa=atpy.Table(infile)
        self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. # in Mpc
        self.r200deg=self.r200*1000./my.DA(self.biweightvel/3.e5,h)/3600.
        dr=sqrt((self.cra-self.n.RA)**2+(self.cdec-self.n.DEC)**2)/self.r200deg
        dv=abs(self.n.ZDIST*3.e5-self.biweightvel)/self.biweightscale
        self.membflag=(dv < 3.) & (dr < 1.)
    def runbiweight(self):
        zflag=abs(self.n.Z - self.cz) < dz
        # calculate angular separation corresponding to a physical radius of 1.7 Mpc
        # 1.71 Mpc corresponds to a projected radius of 1 degree at Coma
        Mpc_per_degree = (cosmo.angular_diameter_distance(z=self.cz).value)*np.pi/180.
        dtheta = 1.71/Mpc_per_degree
        print self.prefix, dtheta
        thetaflag=sqrt((self.n.RA-self.cra)**2 + (self.n.DEC - self.cdec)**2) < dtheta
        self.bootflag=zflag & thetaflag #& (self.n.ISDSS > 0)
        v=3.e5*(self.n.Z)#/(1+self.cz)
        
        self.biweight=self.getbiweight(v[self.bootflag])#,clipiters=None)

        self.clipflag=abs(v-self.biweight[0])/self.biweight[3] < scale_cut
    def getbiweight(self,x,clipiters=None,nbootstrap=1000):
        #z=sigma_clip(x,sig=3,iters=clipiters,cenfunc=biweight_location,varfunc=biweight_midvariance)
        z=x # skip sigma clipping

        #biweightlocation=biweight_location(z)
        #biweightscale=biweight_midvariance(z)

        biweightlocation, biweightscale=getbiweight(z)


            
        # calculate bootstrap errors
    
        nboot=nbootstrap

        boot=bootstrap(z,bootnum=nboot)
        row,col=boot.shape
        bootlocation=np.zeros(row,'f')
        bootscale=np.zeros(row,'f')
        for i in range(row):
            #bootlocation[i]=biweight_location(boot[i,:])
            #bootscale[i]=biweight_midvariance(boot[i,:])
            bootlocation[i],bootscale[i]=getbiweight(boot[i,:])
    
        # get percentiles
        location_lower=np.percentile(bootlocation,q=16)
        location_upper=np.percentile(bootlocation,q=82)
        location_median=np.percentile(bootlocation,q=50)
        scale_lower=np.percentile(bootscale,q=16)
        scale_upper=np.percentile(bootscale,q=82)
        scale_median=np.percentile(bootscale,q=50)

        return biweightlocation,location_upper-biweightlocation,biweightlocation-location_lower,biweightscale,scale_upper-biweightscale,biweightscale-scale_lower,location_median,scale_median
    def plotvhist(self,plotsingle=1):
        if plotsingle:
            pl.figure()
            pl.clf()
        mybins=arange(4000,13000,100)
        #pl.hist(self.n.Z*3.e5,bins=mybins,color='k',histtype='step')
    
        #y=pl.hist(self.n.Z[self.bootflag]*3.e5,bins=mybins,color='b')
        y=pl.hist(self.n.ZDIST[self.bootflag]*3.e5,bins=mybins,color='0.8',histtype='stepfilled')

        g=scipy.stats.norm(loc=self.biweight[0],scale=self.biweight[3]).pdf(mybins)
        scale=sum(y[0])/sum(g)
        pl.plot(mybins,scale*g,'k-',lw=1.5)
        ax=pl.gca()
        
        pl.axvline(self.biweight[0],ls='-',color='k',lw=1.5,ymax=max(scale*g)/30.)
        pl.axis([4000,13000,0,30])
        pl.xticks(arange(4500,13000,4000),fontsize=10)
        pl.yticks(arange(0,40,10),fontsize=10)

        pl.text(.05,.8,'$'+self.prefix+'$',transform=ax.transAxes,horizontalalignment='left',fontsize=12)
        
        s='$ N = %i $'%(sum((self.bootflag & self.membflag)))

        pl.text(.92,.8,s,transform=ax.transAxes,horizontalalignment='right',fontsize=12)



        # read in table

# get initial estimate of cluster redshift

# get RA and Dec of cluster center

# create flag for galaxies with Delta_v < 1000 km/s and Delta_theta < 1 deg

# run sigma_clip on remaining sample using biweight estimates for location and scale

runall=1
print 'MKW11'
mkw11=cluster('MKW11')
if runall:
    print 'Coma'
    coma=cluster('Coma')
    mkw8=cluster('MKW8')
    awm4=cluster('AWM4')
    a2052=cluster('A2052')
    a2063=cluster('A2063')
    ngc=cluster('NGC6107')
    herc=cluster('Hercules')
    a1367=cluster('A1367')
    clustersbydistance=[a1367,mkw11,coma,mkw8,ngc,awm4,a2052,a2063,herc]
    clustersbylx=[mkw11,ngc,mkw8,awm4,herc,a1367,a2063,a2052,coma]
    clustersbydistance=[a1367,mkw11,coma,mkw8,ngc,awm4,a2063,a2052,herc]
#clustersbylx=[mkw11,coma]
def getbiweightall():
    for c in clustersbydistance:
        c.runbiweight()
        #print '%10s: center= %5i (%5i) +%3i-%3i scale= %4i (%4i) +%3i-%3i'%(c.prefix,c.biweight[0],c.biweight[6],c.biweight[1],c.biweight[2],c.biweight[3],c.biweight[7],c.biweight[4],c.biweight[5])
        print '%10s: center= %5i (%5i) +%3i-%3i scale= %4i (%4i) +%3i-%3i'%(c.prefix,c.biweight[0],c.biweightvel,c.biweight[1],c.biweight[2],c.biweight[3],c.biweightscale,c.biweight[4],c.biweight[5])

    
def plotall():
    # plot results
    pl.figure(figsize=(6.5,4))
    pl.subplots_adjust(bottom=.15,left=.1,hspace=.1,wspace=.1,top=.95,right=.95)
    pl.clf()
    i=1
    for c in clustersbydistance:
        subplot(3,3,i)
        c.plotvhist(plotsingle=0)
        multiplotaxes(i)
        i+=1
    ax=pl.gca()
    pl.text(-.6,-.32,'$v_r \ (km/s) $',fontsize=14,transform=ax.transAxes,horizontalalignment='center')
    pl.text(-2.38,1.65,'$N_{gal} $',fontsize=14,transform=ax.transAxes,horizontalalignment='center',rotation=90)
    pl.savefig('/Users/rfinn/research/LocalClusters/SamplePlots/LCSbiweight.eps')
