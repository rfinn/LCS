#!/usr/bin/env python

from LCScommon import *
import numpy as np
import os
import urllib
import webbrowser
import ds9
import sys
from LCSReadmasterBaseNSA import *

### SET UP REFERENCE IMAGES ###
# open image of Hubble tuning fork
webbrowser.open('http://upload.wikimedia.org/wikipedia/commons/thumb/2/21/HubbleTuningFork.jpg/728px-HubbleTuningFork.jpg')

def query_classification():
    print 'classification options: \n'
    print 'S0=0 Sa=1 Sb=2 Sc=3 Sd=4 distorted merger=5 huh?=6 \n'
    t=raw_input('Enter your best guess or q to quit\n')
    if t.find('q') > -1:
        return -1
    return int(t)

class cluster(baseClusterNSA):
    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)
        mypath=os.getcwd()
        if mypath.find('Users') > -1:
            print "Running on Rose's mac pro"
            infile='/Users/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'
        elif mypath.find('home') > -1:
            print "Running on coma"
            infile=homedir+'research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'

    def classify(self):
        flag=self.spiralflag & self.On24ImageFlag
        self.galaxyclass=np.zeros(len(self.ra),'i')
        for i in range(len(self.ra)):
            if flag[i]:
                self.getNSApage(i)
                self.displaycutout(i)
                self.print_ttype(i)
                self.galaxyclass[i]=query_classification()
# loop through galaxies
        self.writeclassifications()
# open NSA webpage for galaxy
    def getNSApage(self,i):
        webaddress='http://www.nsatlas.org/getAtlas.html?search=nsaid&nsaID='+str(self.n.NSAID[i])+'&submit_form=Submit'
        webbrowser.open(webaddress)
    def displaycutout(self,i):
        d=ds9.ds9()
        d.set('frame delete all')
        nsa_parent=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+str(self.n.IAUNAME[i])+'-parent-'+str(self.n.PID[i])+'.fits'

        s='file new '+nsa_parent
        try:
            d.set(s)
        except:
            print 'problem opening parent image'
            print nas_parent

    def print_ttype(self,i):
        if self.agc.AGCMATCHFLAG[i]:
            print 'Burstein type = ',self.agc.BSTEINTYPE[i]
    def writeclassifications(self):
        output=homedir+'research/LocalClusters/NSAmastertables/MorphClassifications/'+self.prefix+'_visual_classifications.fits'
        gtab=atpy.Table()
        gtab.add_column('NSAID',self.n.NSAID)
        gtab.add_column('SPIRAL_SUBCLASS',self.galaxyclass)
        if os.path.exists(output):
            os.remove(output)

        gtab.write(output)

# open r-band image in ds9

# print T-type if available

# query user for galaxy type
cname=sys.argv[1]
cl=cluster(cname)
cl.classify()
