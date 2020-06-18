#!/usr/bin/env python
'''
useage

LCScopyapexfinalcat.py

'''


import os
import sys


clusters=['A1367','A2052','A2063','AWM4','Coma','Hercules','MKW11','MKW8','NGC6107']
destination='/home/rfinn/research/LocalClusters/ApexFinalCatalogs/'
#os.mkdir(destination)

files=['mosaic_minus_median_extract.fits','mosaic_cov.fits','mosaic_unc.fits','mosaic_std.fits']
files=['mosaic_extract.tbl','mosaic_extract_raw.tbl']
#s='cp '+imagepath+files[0]+' '+destination+prefix+files[0]
#print s
#os.system(s)

for cl in clusters:

    imagepath='/home/rfinn/research/LocalClusters/MIPS/rawdata/'+cl+'/FullMosaic/'

    for i in range(len(files)):
        s='cp '+imagepath+files[i]+' '+destination+cl+files[i]
        os.system(s)
