#!/usr/bin/env python

'''
 written by Rose A. Finn on 2/3/2013

 GOAL:
    - create a mastertable for each LCS cluster that uses the NASA-Sloan atlas as the parent sample 
 
 METHOD:
    - match NSA to 24-micron apex catalog
    - pull AGC data (NSA contains AGC number)
    - match with galaxy zoo info
    - append new columns using atpy
    - write out new table using atpy
 
 UPDATES:
 
 2015/07/13

 this program is only used to calculate local density and write out local density files
 
'''
 
 
 
 


from pylab import *
import os, atpy
from LCScommon import *

#import ReadAGCsav
from pyraf import iraf
import mystuff as my
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-v',"--velocitycut",help='velocity cut to use when calculating local density.  Galaxies within +/- vcut will be used.  Default value is 1500 km/s.',default=1500.,type=float)

args = parser.parse_args()


class cluster:
    def __init__(self,clustername):
        self.prefix=clustername
        self.image24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_minus_median_extract.fits'
        self.noise24=homedir+'research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_unc.fits'
        if (clustername.find('A1367') > -1):
            self.image24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+self.prefix+'ch1r1_mosaic_minus_median_extract.fits'
            self.noise24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+self.prefix+'ch1r1_mosaic_unc.fits'
        elif (clustername.find('Herc')>-1):#for Abell 1367 and Hercules cluster
            self.image24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+self.prefix+'ch1r1_mosaic_minus_median_extract.fits'
            self.noise24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+self.prefix+'ch1r1_mosaic_unc.fits'

        # read NSA table for each cluster
        infile=homedir+'research/NSA/'+self.prefix+'_NSA.Fits'
        self.ndat=atpy.Table(infile,type='fits')
        self.nsadir=homedir+'research/NSA/'

        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix] 
        self.biweightvel=clusterbiweightcenter[self.prefix]
        self.biweightscale=clusterbiweightscale[self.prefix]
        self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. # in Mpc
        self.r200deg=self.r200*1000./my.DA(self.cz,h)/3600.

        self.cdMpc=self.biweightvel/H0
        self.cdcm=self.cdMpc*3.e24
        self.csigma=self.biweightscale
        self.mcl=my.clusterMass(self.csigma,self.cz,h)
        self.AngDistance=my.DA(self.cz,h)

    def geton24imageflag(self):
        iraf.imgets(image=self.image24,param='CD1_1')#get x plate scale on rotated image
        #print iraf.imgets.value
        xplate=abs(float(iraf.imgets.value))#deg/pixel
        dpix=deltaCutout/3600./xplate/2. # requires a 50 arcsec buffer b/w center of galaxy and edge of image
        iraf.imgets(image=self.image24,param='naxis1')#get x value corresponding to RA 
        xpixmax=(int(iraf.imgets.value))#deg/pixel
        iraf.imgets(image=self.image24,param='naxis2')#get x value corresponding to RA 
        ypixmax=(int(iraf.imgets.value))#deg/pixel

        # write RA and Dec to ascii file
        t = atpy.Table()
        t.add_column('RA',self.ndat.RA)
        t.add_column('DEC',self.ndat.DEC)
        outfile=self.nsadir+self.prefix+'_RADEC.txt'
        out1=open(outfile,'w')
        for i in range(len(self.ndat.RA)):
            s='%12.8f %12.8f \n'%(self.ndat.RA[i],self.ndat.DEC[i])
            out1.write(s)
        out1.close()
        
        # transform RA and Dec to pixels using wcsctran
        print 'transforming 24um coords'
        outcoords=self.nsadir+str(self.prefix)+'xy24.txt'
        if os.path.exists(outcoords):
            os.remove(outcoords)
        iraf.imcoords.wcsctran(image=self.image24,input=outfile,output=outcoords,inwcs='world',outwcs='logical',verbose='no')

        # read in pixel coordinates
        self.mipscoords=atpy.Table(outcoords,type='ascii')
        x=self.mipscoords['col1']
        y=self.mipscoords['col2']
        self.On24ImageFlag= (x > dpix) & (x < (xpixmax-dpix)) & (y > dpix) & (y < (ypixmax-dpix))  

        print self.prefix,': # on 24um image = ',sum(self.On24ImageFlag)
        # check to see if pixels are off image and set flag accordingly

    def localdensity5(self):
        # get local density by summing mass w/in 300 kpc
        # NOTE:
        #  - this measure will be unreliable for galaxies near the edge of the 3 deg area
        #  - this will be fine for galaxies on 24um image
        sdssflag=self.ndat.ISDSS > -1
        DA=self.AngDistance
        x=self.ndat.RA
        y=self.ndat.DEC
        xref=self.ndat.RA[sdssflag]
        yref=self.ndat.DEC[sdssflag]
        n1=6
        sigma5=zeros(len(x),'d')
        sigma10=zeros(len(x),'d')
        for i in range(len(x)):
            deltav=abs(self.ndat.ZDIST[sdssflag]-self.ndat.ZDIST[i])*3.e5
            d=sqrt((x[i]-xref)**2+(y[i]-yref)**2)*3600./1000.*DA#d in Mpc
            vflag = deltav < args.velocitycut
            #print i,sum(vflag)
            d=d[vflag]#apply a velocity cut of +/- 1500 km/s
            d.sort()#sort in ascending order, zeroth element is distance from galaxy to itself
            sig=0
            if len(d) < n1:
                print 'not enough points to calculate local density'
                print 'only have ',len(d),' galaxies w/in 1500 km/s'
                print 'skipping to next entry'
                continue
            else:
                sigma5[i]=1./(4.*pi)*(1.*5)/(d[5])**2
            try:
                sigma10[i]=1./(4.*pi)*(1.*10)/(d[10])**2
            except IndexError:
                continue
        self.sigma_5=sigma5
        self.sigma_10=sigma10
        

    def localdensity(self):
        # get local density by summing mass w/in 300 kpc
        # NOTE:
        #  - this measure will be unreliable for galaxies near the edge of the 3 deg area
        #  - this will be fine for galaxies on 24um image
        sdssflag=self.ndat.ISDSS > -1
        DA=self.AngDistance
        x=self.ndat.RA
        y=self.ndat.DEC
        xref=self.ndat.RA[sdssflag]
        yref=self.ndat.DEC[sdssflag]
        n1=3
        n2=6
        sigma=zeros(len(x),'d')
        for i in range(len(x)):
            deltav=abs(self.ndat.ZDIST[sdssflag]-self.ndat.ZDIST[i])*3.e5
            d=sqrt((x[i]-xref)**2+(y[i]-yref)**2)*3600./1000.*DA#d in Mpc
            d=d[deltav<args.velocitycut]#apply a velocity cut of +/- 1500 km/s
            d.sort()#sort in ascending order, zeroth element is distance from galaxy to itself
            sig=0
            '''
            if len(d) < n2:
                print 'not enough points to calculate local density'
                print 'only have ',len(d),' galaxies w/in ',args.velocitycut
                if len(d) < n1:
                    print 'ut oh!'
                for j in range(n1,len(d)):
                    sig += (1.*j)/(d[j])**2

            else:
                for j in range(n1,n2+1):
                    sig += (1.*j)/(d[j])**2
            '''
            
            sigma[i]=1./(4.*pi)*d[1]
        self.sigma_nn=sigma
        
        return
    def localdensitybymass(self):#find local density, using nearest neighbors n1 through n2
        DA=self.AngDistance #kpc/arcsec
        angDist_300kpc=300./DA/3600. # angular dist corresponding to 300 kpc, in degrees
        x=self.ndat.RA
        y=self.ndat.DEC
        # should probably use SDSS as the reference sample, but for now, using NSA
        sdssflag=self.ndat.ISDSS > -1
        xref=self.ndat.RA[sdssflag]
        yref=self.ndat.DEC[sdssflag]
        self.rhomass=zeros(len(x),'d')
        for i in range(len(x)):
            deltav=abs(self.ndat.ZDIST[sdssflag]-self.ndat.ZDIST[i])*3.e5
            d=sqrt((x[i]-xref)**2+(y[i]-yref)**2)*3600./1000.*DA#d in Mpc
            d=d[deltav<args.velocitycut]#apply a velocity cut of +/- 1500 km/s
            neighbor_flag=d < .3

            self.rhomass[i]=sum(self.ndat.MASS[neighbor_flag])
            #print i, self.rhomass[i],sum(neighbor_flag)
    def matchnsa2sdssphot(self):
        return
    def writeoutput(self):
        # append flag to NSA table 
        # create new table
        ldat=atpy.Table()
        #ldat.add_column('On24ImageFlag',self.On24ImageFlag)

        # will add more columns here as time permits
        ldat.add_column('NSAID',self.ndat.NSAID,unit='',description='NSA ID')
        ldat.add_column('RA',self.ndat.RA,unit='deg',description='RA')
        ldat.add_column('DEC',self.ndat.DEC,unit='deg',description='DEC')
        ldat.add_column('Z',self.ndat.Z,unit='',description='redshift')
        ldat.add_column('ZDIST',self.ndat.ZDIST,unit='',description='NSA ZDIST')
        ldat.add_column('RHOMASS',self.rhomass,unit='Msun',description='Mass of galaxies w/in 300 kpc')
        ldat.add_column('SIGMA_NN',self.sigma_nn,unit='Ngal/Mpc^2', description='3rd to 6th nearest neighbor')
        ldat.add_column('SIGMA_5',self.sigma_5,unit='Ngal/Mpc^2', description='to 5th nearest neighbor')
        ldat.add_column('SIGMA_10',self.sigma_10,unit='Ngal/Mpc^2', description='to 10th nearest neighbor')

        # write out table
        outfile=homedir+'research/LocalClusters/NSAmastertables/LocalDensityTables/'+self.prefix+'_localdensity.fits'
        if os.path.exists(outfile):
            os.remove(outfile)
        ldat.write(outfile)


if __name__ == '__main__':
    mkw11=cluster('MKW11')
    mkw8=cluster('MKW8')
    awm4=cluster('AWM4')
    a2052=cluster('A2052')
    a2063=cluster('A2063')
    ngc=cluster('NGC6107')
    coma=cluster ('Coma')
    herc=cluster('Hercules')
    a1367=cluster('A1367')

    mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc,coma,herc,a1367]
    for cl in mylocalclusters:
        print  '\n ',cl.prefix,'\n'
        #cl.geton24imageflag()
        #cl.getsdssspeccat()
        #cl.getsdssphotcat()
        cl.localdensity()
        cl.localdensity5()
        cl.localdensitybymass()
        cl.writeoutput()

