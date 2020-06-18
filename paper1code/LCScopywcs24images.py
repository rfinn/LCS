#!/usr/bin/env python
'''
useage

LCScopywcs24imges.py MKW8

'''


import os
import sys
prefix=sys.argv[1]
destination='/home/rfinn/research/LocalClusters/Images/'+prefix+'/24umWCS/'
try:
    os.mkdir(destination)
except OSError:
    print "error when trying to create ",destination
    print "directory probably already exists, so going to ignore error :)"

imagepath='/home/rfinn/research/LocalClusters/MIPS/rawdata/'+prefix+'/FullMosaic/'
combinepath='/home/rfinn/research/LocalClusters/MIPS/rawdata/'+prefix+'/FullMosaic/Combine-mosaic/'

files=['mosaic_minus_median_extract.fits','mosaic_cov.fits','mosaic_unc.fits','mosaic_std.fits']
s='cp '+imagepath+files[0]+' '+destination+prefix+'-WCS-'+files[0]
print s
os.system(s)

for i in range(1,len(files)):
    s='cp '+combinepath+files[i]+' '+destination+prefix+'-WCS-'+files[i]
    os.system(s)
s='cp '+imagepath+'mosaic_extract.tbl '+destination+'/'+prefix+'-WCS-mosaic_extract.tbl'
os.system(s)
