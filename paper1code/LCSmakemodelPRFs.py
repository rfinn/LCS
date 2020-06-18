#!/usr/bin/env python

import SAcutout
from LCScommon import *
print clusternames
#clusternames=['Coma']

for prefix in clusternames:

    xstar=[]
    ystar=[]
    ra=[]
    dec=[]
    infile='/Users/rfinn/research/LocalClusters/Images/'+prefix+'/24umWCS/'+prefix+'-WCS-mosaic_extract.tbl'
    mosaic='/Users/rfinn/research/LocalClusters/Images/'+prefix+'/24umWCS/'+prefix+'-WCS-mosaic_minus_median_extract.fits'

    infile=homedir+'research/LocalClusters/PRF/'+prefix+'-starlist.tbl'
    in1=open(infile,'r')
    for line in in1:
        if line.startswith('\\'):
            continue
        elif line.startswith('|'):
            continue
        else:
            t=line.split()
            # select lines with 
            # (status == 1) & (SNR > 10) & ( N_PS < 2)
            #    17                 18          2
            # x = 9, y = 11
            xstar.append(float(t[8]))
            ystar.append(float(t[10]))
            ra.append(float(t[3]))
            dec.append(float(t[5]))
    in1.close()

    # make cutouts of stars
    dx=15
    dy=15
    for i in range(len(xstar)):
        print xstar[i],ystar[i],mosaic
        counter='%02i'%(i)
        outimage='/Users/rfinn/research/LocalClusters/PRF/'+prefix+'/star-'+counter+'.fits'
        #def cutout(filename, xc, yc, xw=25, yw=25, units='pixels', outfile=None,clobber=True, useMontage=False, coordsys='celestial', verbose=False):
        SAcutout.cutout(mosaic,outimage,xstar[i]+1,ystar[i]+1,xw=dx,yw=dy,units='pixels')
    
