#!/usr/bin/env python

import glob,os
from pylab import *

os.chdir('/home/rfinn/research/LocalClusters/cutouts/Coma/temp/')
oldlist=glob.glob('*rot.fits')
#oldlist=oldlist.sort()
os.chdir('/home/rfinn/research/LocalClusters/cutouts/Coma/')
newlist=glob.glob('*rot.fits')
#newlist=newlist.sort()
diff=set(newlist) - set(oldlist)

outfile=open('/home/rfinn/research/LocalClusters/NewComaImages.txt','w')
for file in diff:
    s=file+'\n'
    outfile.write(s)
    s='cp '+file+' new24rot/.'
    os.system(s)
