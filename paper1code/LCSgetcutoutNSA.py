#!/usr/bin/env python

from LCScommon import *
import pylab as pl
import numpy as np
import os
import urllib
import ds9
from LCSReadmasterBaseNSA import *

### SET UP REFERENCE IMAGES ###
# open image of Hubble tuning fork
urllib.urlopen('http://upload.wikimedia.org/wikipedia/commons/thumb/2/21/HubbleTuningFork.jpg/728px-HubbleTuningFork.jpg')

def query_classification():
    print 'classification options: \n'
    print 'Sa=1 Sb=2 Sc=3 Sd=4 distorted merger=5 huh?=6 \n'
    t=raw_input('Enter your best guess \n')
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
                self.getNSApage(self,i)
                self.displaycutout(self,i)
                self.print_ttype(self,i)
                self.galaxyclass[i]=query_classification()
# loop through galaxies

# open NSA webpage for galaxy
    def getNSApage(self,i):
        webaddress='http://www.nsatlas.org/getAtlas.html?search=nsaid&nsaID='+str(self.n.NSAID[i])+'&submit_form=Submit'
        urllib.urlopen(webaddress)
    def displaycutout(self,i):
        d=ds9.ds9()
        d.file()
    def print_ttype(self,i):
        
# open r-band image in ds9

# print T-type if available

# query user for galaxy type
