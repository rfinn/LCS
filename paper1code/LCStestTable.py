#!/usr/bin/env -python
import os
import atpy
from LCSReadmasterBase import *

''' 

'''
mypath=os.getcwd()

if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

    
class cluster(baseCluster):
    
    def __init__(self,clustername):
        baseCluster.__init__(self,clustername)

    def redoTable(self):
        t=column_stack((self.sdssu,self.sdssg,self.sdssr,self.sdssi,self.sdssz))
	c1=pyfits.Column(name='SDSS_MAGS',format='5E',unit='',array=t)
        mastertb=pyfits.new_table([c1])#,c2,c3,c4,c5])
	s=homedir+'/research/LocalClusters/MasterTables/'+self.prefix+'NICEtable.fits'
	mastertb.writeto(s,clobber='yes')
