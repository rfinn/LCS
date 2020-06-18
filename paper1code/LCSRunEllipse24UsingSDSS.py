#!/usr/bin/env python
'''

LCSRunEllipse24UsingSDSS.py MKW11
where second word is the cluster prefix

options for cluster prefix names are
MKW11, MKW8, AWM4, A2063, A2052, NGC6107, Coma, A1367, Hercules
'''
#read mastertable - get ellipse flag

#for galaxies w/ellipse flag = 1
#grab sdss ellipse output table

#run ellipse on 2um image using PA and ellip from SDSS

#can't just run using sdss table b/c pixel scale is different
import glob
from pyraf import iraf
import os,sys
from LCSReadmasterBase import *
from LCScommon import *

cname=sys.argv[1]
#read mastertable - get ellipse flag

#for galaxies w/ellipse flag = 1

#grab sdss ellipse output table

#run ellipse on 24um image using sdss table

#if this doesn't work, run ellipse on 2um image using PA and ellip from SDSS

#set output path based on user
t=os.getlogin()#returns username
if t.find('rfinn') > -1:
    outpathroot='/home/rfinn/'
else:
    outpathroot='/home/share/'

class cluster(baseCluster):

    def runEllipse24UsingSDSS(self):
        agcnumber=self.agcnumber[self.ellipseflag]
        #mcutoutpath='/home/rfinn/research/LocalClusters/MaskedCutouts/'+self.prefix+'/'
        mcutoutpath='/home/rfinn/research/LocalClusters/MaskedCutouts/'+self.prefix+'/'
        os.chdir(mcutoutpath)
        s=outpathroot+'research/LocalClusters/Ellipse24UsingSDSS/'+self.prefix
        try:
            os.mkdir(s)
        except OSError:
            pass
        for i in range(len(agcnumber)):
            print agcnumber[i]
        
            sdssEllipseTable= '/home/rfinn/research/LocalClusters/EllipseTables/'+self.prefix+'/'+self.prefix+'-'+str(agcnumber[i])+'-cutout-sdss.tab'
            #mipsImage=mcutoutpath+'m'+self.prefix+'-'+str(agcnumber[i])+'-cutout-24-rot.fits'
            mipsImage='m'+self.prefix+'-'+str(agcnumber[i])+'-cutout-24-rot.fits'
            #outim=cutoutpath+self.prefix+'-'+str(agcSpiral[i])+'-cutout-sdss.fits \n'
            os.system('rm junk.txt')
            if os.path.isfile(sdssEllipseTable):
                iraf.tprint(table=sdssEllipseTable,pwidth='INDEF',showhdr='no', Stdout='junk.txt')
                os.system("awk '{print $2, $7, $9, $11, $13}' < junk.txt > junk2.txt")
            else:
                print 'Warning: ',sdssEllipseTable,' does not exist'
                sdssEllipseTable= '/home/rfinn/research/LocalClusters/EllipseTables/'+self.prefix+'/'+self.prefix+'-'+str(agcnumber[i])+'-cutout-sdss.dat'
                s="awk '{print $2, $7, $9, $11, $13}' <"+sdssEllipseTable+" > junk2.txt"
                os.system(s)
            #run ellipse a second time, keeping PA and ellip fixed
            infile=open('junk2.txt','r')
            for line in infile:
                t=line.split()
                if float(t[0]) > myradius:
                    newellip=float(t[1])
                    if newellip < .05:#min value that ellipse can handle
                        newellip=.05
                    newPA=float(t[2])
                    if newPA < -90:
                        newPA=newPA+180
                    elif newPA > 90:
                        newPA = newPA-180
                    #11 - X0, 13 - Y0
                    newxcenter=float(t[3])
                    newycenter=float(t[4])


            t=mipsImage.split('.')
            efile=self.prefix+'-'+str(agcnumber[i])+'.24UsingSDSS.tab'
            copyefile=outpathroot+'research/LocalClusters/Ellipse24UsingSDSS/'+self.prefix+'/'+self.prefix+'-'+str(agcnumber[i])+'.24UsingSDSS.tab'


            #iraf.ellipse(input=mipsImage,output=efile,inellip=sdssEllipseTable)#trying to use sdss input
            if os.path.exists(efile):
                s=efile+' exists.  Rerun?  (y=yes, any key to skip)'
                rerunFlag=raw_input(s)
                if rerunFlag.find('y') > -1:
                    os.remove(efile)
                    iraf.ellipse(input=mipsImage,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',recenter='no',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=newellip,hellip='yes')

                else:
                    continue
            else:
                iraf.ellipse(input=mipsImage,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',recenter='no',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=newellip,hellip='yes')
            #outfile24=homedir+'research/LocalClusters/EllipseTables/'+self.prefix+'/'+self.prefix+'-'+str(agcSpiral[i])+'-cutout-24-rot.dat \n'

            print 'Displaying isophotes from ellipse fit.  Hit q in DS9 window to quit'
            iraf.isoexam(table=efile)
            s='cp '+efile+' '+copyefile
            print s
            os.system(s)


raw_input('Make sure ds9 is open.  Hit return when ready.')
iraf.stsdas()
iraf.analysis()
iraf.isophote()
iraf.tables()
iraf.ttools()

ipa=0
xcenter=21
ycenter=21
minr=2
initialr=6
maxr=20
iellip = .05
evalrad=15#radius to measure PA and ellip at
myradius=15

sdssxcenter=50.5
sdssycenter=50.5
sdssminr=2
sdssinitialr=8
sdssmaxr=49
evalrad=15#radius to measure PA and ellip at

##for ncl in range(1,len(clusternames)):
##    if ncl == 0:
##        mkw11=cluster(clusternames[ncl])
##        cl=mkw11
##    if ncl == 1:
##        mkw8=cluster(clusternames[ncl])
##        cl=mkw8
##    if ncl == 2:
##        awm4=cluster(clusternames[ncl])
##        cl=awm4
##    if ncl == 3:
##        a2063=cluster(clusternames[ncl])
##        cl=a2063
##    if ncl == 4:
##        a2052=cluster(clusternames[ncl])
##        cl=a2052
##    if ncl == 5:
##        ngc6107=cluster(clusternames[ncl])
##        cl=ngc6107
##    if ncl == 6:
##        coma=cluster(clusternames[ncl])
##        cl=coma
##    if ncl == 7:
##        a1367=cluster(clusternames[ncl])
##        cl=a1367
##    if ncl == 8:
##        hercules=cluster(clusternames[ncl])
##        cl=hercules
##    cl.runEllipse24UsingSDSS()

#reworked the script to run one cluster at a time so that students can run
cl=cluster(cname)
cl.runEllipse24UsingSDSS()


