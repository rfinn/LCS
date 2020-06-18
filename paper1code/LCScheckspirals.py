#!/usr/bin/env python

import sys
import ds9
from LCScommon import *

#################

import argparse
parser=argparse.ArgumentParser()
#parser.add_argument("agcnumber",help='agcnumber')
parser.add_argument("-a",'--all',help="run for all clusters",action="store_true")
parser.add_argument("-c",'--cluster',help="cluster name (run one cluster only)",action="store")
#parser.add_argument("-s",'--startindex', help="starting index (to continue from previous)",action="store")
#parser.add_argument("-d",'--display', help="display result in ds9",action="store_true")
args = parser.parse_args()

#################


if args.cluster:
    c=args.cluster
    for id in spiral_100_nozoo[c]:
        image=homedir+'research/LocalClusters/GalfitAnalysis/'+c+'/NSA/'+c+'-'+str(id)+'-parent-r.fits'
        try:
            d.set('frame delete all')
        except:
            d=ds9.ds9()
            d.set('frame delete all')
        d.set('file new '+image)
        d.set('zscale')
        print 'now viewing ',c+'-'+str(id)
        t = raw_input('hit any key to view next image (x to quit)\n')
        if t.find('x') > -1:
            sys.exit()

if args.all:
    for c in clusternames:
        for id in spiral_100_nozoo[c]:
            image=homedir+'research/LocalClusters/GalfitAnalysis/'+c+'/NSA/'+c+'-'+str(id)+'-parent-r.fits'
            try:
                d.set('frame delete all')
            except:
                d=ds9.ds9()
                d.set('frame delete all')
            d.set('file new '+image)
            d.set('zscale')
