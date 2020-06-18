#!/usr/bin/env python
from pylab import *
import ds9
infile='mosaic_extract.tbl'
outfile='mosaic_extract_stars.tbl'
input=open(infile,'r')
output=open(outfile,'w')
i=0
for line in input:
    if line.find('\\') > -1:
        output.write(line)
        continue
    if line.find('|') > -1:
        output.write(line)
        continue
    f=line.split()#check to make sure source is far from edge
    if (float(f[1]) < 15.) :
        continue
    if (float(f[2]) < 15.) :
        continue
    if (float(f[1]) > 128.) :
        continue
    if (float(f[2]) > 128.) :
        continue
    if (i > 3):
        print "Got 3 sources in mosaic_extract.tbl\n"
        break
    output.write(line)
    i=i+1
input.close()
output.close()
#    os.system('prf_estimate.pl -n prf_estimate_8_11.nl')
#    os.system('cp prf_estimate/PRF.fits cal/PRF_estimate.fits')
