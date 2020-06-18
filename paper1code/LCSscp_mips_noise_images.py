#!/usr/bin/env python

import os
#from LCScommon import *

clusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
clusternames=['MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
#clusternames=['MKW11','Coma']

for cl in clusternames:
    outdir='/Users/rfinn/research/LocalClusters/MIPS/rawdata/'+cl+'/FullMosaic/'
    s='mkdir -p '+outdir
    os.system(s)
    s="scp coma:research/LocalClusters/MIPS/rawdata/"+cl+'/FullMosaic/mosaic_noise.fits '+outdir+'mosaic_noise.fits'
    os.system(s)
