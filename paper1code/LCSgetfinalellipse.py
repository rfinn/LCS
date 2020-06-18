#!/usr/bin/env python
"""
grab contents of directories of post-ellipse-processed images so we can track what happened to each galaxy.

did r-band make it to final finished directory?

did 24um make it to final finished directory?

if not, what happened?

NerabyObjects
OffCenter
PartialImages
PeculiarGalaxies

Finished

Finished also has reject subdirectory

"""

from pylab import *
import glob
import os
from LCScommon import *

originaldir='/home/rfinn/research/LocalClusters/cutouts/' #where original cutouts are

cutoutpath='/home/alissa/LocalClusters/cutouts/' # where processed cutouts are

subdirectories=['NearbyObjects','OffCenter','PartialImages','PeculiarGalaxies','Finished','Finished/reject']

for cl in clusternames: #loop over clusters

    #get list of original cutouts
    #make dictionary that associates agc name with list index
    #loop over subdirectories
    #ellipseflag24 for finished 24
    #ellipseid24 to track why galaxy was rejected 
    #ellipseflag for finished rband
    #ellipseid to track why galaxy was rejected 
    #add columns to mastertable?  or should I do that in LCSmkcutouts?
