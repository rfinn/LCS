#!/usr/bin/env python
'''
usage - run this program to display sdss cutouts with SDSS exponential fit overlaid

ds9 must be running first!

'''
from pylab import *
import os
from LCScommon import *
from LCSReadmasterBase import baseCluster
import mystuff as my

class cluster(baseCluster):
    #__init__(self,clustername):
    #    mypath=os.getcwd()
    #    print 'running for cluster ',clustername
        
    def showCutouts(self):
        flag = self.sdssflag & self.On24ImageFlag
        agcnumber=self.agcnumber[flag]
        exprad=self.sdssExpRadr[flag]
        expAB=self.sdssExpABr[flag]
        expPhi=self.sdssExpPhir[flag]
        ra=self.sdssra[flag]
        dec=self.sdssdec[flag]
        raw_input('Make sure that ds9 is open!  \n Hit return when ready.')
        for i in range(len(agcnumber)):
            outim=self.cutoutpath+self.prefix+'-'+str(agcnumber[i])+'-cutout-sdss.fits'
            os.system('xpaset -p ds9 frame 1')
            s='cat '+outim+' | xpaset ds9 fits '
            os.system(s)
            outfile1=self.cutoutpath+'test.reg'
            outf1=open(outfile1,'w')
            outf1.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
            outf1.write('fk5 \n')
            print 'ellipse(%12.6f, %12.6f, %12.8f, %12.8f, %5.1f) \n'%(ra[i],dec[i],exprad[i]/3600.,exprad[i]/expAB[i]/3600.,expPhi[i])
            #s='ellipse(%12.6, %12.6, %12.6, %12.6, %5.1f)'%(ra[i],dec[i],exprad[i]/3600.,exprad[i]/expAB[i]/3600.,expPhi[i])
            s='ellipse(%12.6f, %12.6f, %12.8f, %12.8f, %5.1f) \n'%(ra[i],dec[i],exprad[i]/3600.,exprad[i]/expAB[i]/3600.,expPhi[i])
            print s
            outf1.write(s)
            outf1.close()
            s='xpaset -p ds9 regions load '+outfile1
            print s
            os.system(s)

            #print ra[i],dec[i],exprad[i],exprad[i]/expAB[i],(expPhi[i]+90)
            s='echo "image; ellipse %5.1f %5.1f %5.1f\" %5.1f\" %5.1f" |xpaset -p ds9 regions -coord fk5 '%(ra[i],dec[i],exprad[i],exprad[i]/expAB[i],(expPhi[i]))
            os.system(s)
            #s='echo "circle 50 50 20" | xpaset -p ds9 regions -coord image '
            #os.system(s)

            st=raw_input('Hit q to quit, any other key to continue \n')
            if my.beginsWith('q',st):
                break

cl=cluster(clusternames[1])
cl.showCutouts()
