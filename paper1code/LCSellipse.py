#!/usr/bin/env python

import glob
from pyraf import iraf
import os

def runellipse(files,xcenter,ycenter,minr,initialr,maxr,nframe=1):
    for i in range(len(files)):
    #for i in range(5):
        mfile=files[i]
        repeatflag=1
        while (repeatflag > 0.1):
            iraf.display(mfile,frame=nframe, fill='yes')
            outfile1='m'+mfile
            print mfile
            print 'Running imedit to mask out other sources in the field:'
            print 'Enter b to mask out a circular region'
            print 'Enter a to mark the corners of a rectangle'
            print 'Enter q when finished'
            try:
                os.remove(outfile1)
            except OSError:
                print 'everything is ok'
            print 'running imedit ',mfile, outfile1
            iraf.imedit(mfile,outfile1)
                
            t=mfile.split('.')
            efile=t[0]+'.tab'
            print 'Running ellipse to fit isophotes to galaxy:'
            print 'Enter h to continue fitting'
            try:
                os.remove(efile)
            except OSError:
                print 'everything is ok'
            iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr)
            print 'Displaying isophotes.  Hit q to quit'
            iraf.isoexam(table=efile)
#            try:
#                iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr)
#            except:
#                print efile," already exists so I am deleting it. Hit Cntrl-c if you don't want to delete"
#                s='rm '+efile
#                os.system(s)
#                iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr)
                
            flag=str(raw_input('Are you happy with the fit?  y=yes n=no x=quit '))
            flag=str(flag)
            print 'this is what I think you typed ',flag
            if flag.find('y') > -1:
                s='mv *'+t[0]+'* Finished/'
                os.system(s)
                repeatflag=0
                print 'i think repeatflag = 0', repeatflag
            if flag.find('n') > -1:
                s='rm *'+t[0]+'*.tab'
                os.system(s)
                s='rm m'+t[0]+'*.fits'
                os.system(s)
                repeatflag=1
                print 'i think repeatflag = 1', repeatflag
            if flag.find('x') > -1:
                repeatflag=0
                print 'i think repeatflag = 0', repeatflag
                return
            print 'repeatflag = ',repeatflag
            


            

os.system('mkdir Finished')
raw_input('Make sure ds9 is open.  Hit return when ready.')
iraf.stsdas()
iraf.analysis()
iraf.isophote()
mipsfiles=glob.glob('*cutout-24.fits')
sdssrfiles=glob.glob('*cutout-sdss.fits')

xcenter=23.0
ycenter=23.0
minr=2
initialr=6
maxr=20

sdssxcenter=50.5
sdssycenter=50.5
sdssminr=2
sdssinitialr=8
sdssmaxr=49

runellipse(mipsfiles,xcenter,ycenter,minr,initialr,maxr)
runellipse(sdssrfiles,sdssxcenter,sdssycenter,sdssminr,sdssinitialr,sdssmaxr,nframe=2)

