#!/usr/bin/env python
import os
from LCScommon import *
mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

class cluster:
    def __init__(self,clustername):
        self.prefix=clustername
        self.mkregfile()

    def mkregfile(self):
        infile1=homedir+'research/LocalClusters/NSAmastertables/RADecFiles/'+self.prefix+'_RADEC.csv'
        infile=open(infile1,'r')
        ra=[]
        dec=[]
        for line in infile:
            if line.find('ra') > -1:
                continue
            t=line.split(',')
            ra.append(t[0])
            dec.append(t[1])
        infile.close()
	s=homedir+'research/LocalClusters/RegionsFiles/'+self.prefix+'.NSA.reg'
	out1=open(s,'w')
	out1.write("global color=red font=\"helvetica 10 normal\" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\nfk5\n")

        for r,d in zip(ra,dec):
                string1 = "circle(%12.8f, %12.8f, 15\") \n"%(float(r),float(d))
                out1.write(string1)
        out1.close()


for clname in clusternames:
    cl=cluster(clname)
