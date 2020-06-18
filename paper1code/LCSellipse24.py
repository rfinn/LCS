#!/usr/bin/env python

import glob
from pyraf import iraf
import os


def runimedit(mfile,outfile1,nframe):
    continueWithProgram=1
    continueWithObject=1
    repeatflag=1
    while (repeatflag > 0.1):

        iraf.display(mfile,frame=nframe, fill='yes')

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

        flag=str(raw_input('Are you happy with the editing?  n=no x=quit y (or any other key) = yes '))
        flag=str(flag)
        print 'this is what I think you typed ',flag
        if flag.find('n') > -1:
            flag2=str(raw_input('What is wrong?  r=redo masking, o=nearby object, p=partial image, x=quit '))
            if flag2.find('r') > -1:
                s='rm '+outfile1
                os.system(s)
                repeatflag=1
                print 'i think repeatflag = 1 ', repeatflag
            elif flag2.find('o') > -1:
                s='rm '+outfile1
                os.system(s)

                s='mv '+mfile+' NearbyObjects/'
                os.system(s)
                continueWithObject=0
                return continueWithProgram,continueWithObject
            elif flag2.find('p') > -1:
                s='rm '+outfile1
                os.system(s)

                s='mv '+mfile+' PartialImages/'
                os.system(s)
                continueWithObject=0
                return continueWithProgram,continueWithObject
            elif flag2.find('x') > -1:
                continueWithProgram=0
                repeatflag=0
                print 'i think repeatflag = 0', repeatflag
                return continueWithProgram,continueWithObject
            else: 
                repeatflag=0

        elif flag.find('x') > -1:
            print 'i think you want to exit'
            continueWithProgram=0
            repeatflag=0
            return continueWithProgram,continueWithObject
        else:
            repeatflag=0

    return continueWithProgram,continueWithObject


def runellipse(files,xcenter,ycenter,minr,ipa,initialr,maxr,iellip,nframe=1,myradius=15):
    initialradius=myradius
    for i in range(len(files)):
        myradius=initialradius
    
        mfile=files[i]

        #mask image
        outfile1='m'+mfile
        continueWithProgram,continueWithObject=runimedit(mfile,outfile1,nframe)
        if not continueWithProgram:
            print "quitting program"
            return
        if not continueWithObject:
            print "going on to next image"
            continue

        #run ellipse
        t=mfile.split('.')
        efile=t[0]+'.tab'
        imprefix=t[0]
        print mfile, imprefix
        print 'Running ellipse to fit isophotes to galaxy:'
        try:
            os.remove(efile)
        except OSError:
            print 'everything is ok'
        print "First pass, letting PA and e vary"
        iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no',interactive='no')
        print 'Displaying isophotes from first pass.  Hit q in DS9 window to quit'
        iraf.isoexam(table=efile)

        os.system('rm junk.txt')
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout='junk.txt')
        os.system("awk '{print $2, $7, $9, $11, $13}' < junk.txt > junk2.txt")
        #run ellipse a second time, keeping PA and ellip fixed
        #allow user to adjust the radius where PA and ellip are measured
        repeatflag=1
        myellipflag=0
        myPAflag=0
        interactiveflag=0
        while (repeatflag > 0.1):
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
                    break
            s='rm '+efile
            os.system(s)
            if interactiveflag:
                iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=newellip,hellip='yes',recenter='yes',xylearn='yes',interactive='yes')
            elif (not myellipflag) and (not myPAflag):
                iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=newellip,hellip='yes',recenter='yes',xylearn='yes',interactive='no')
            elif (not myellipflag) and myPAflag:
                iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=myPA,hpa='yes',ellip=newellip,hellip='yes',recenter='yes',xylearn='yes',interactive='no')
            elif (myellipflag) and (not myPAflag):
                iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=myellip,hellip='yes',recenter='yes',xylearn='yes',interactive='no')
            elif (myellipflag) and (myPAflag):
                iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=myPA,hpa='yes',ellip=myellip,hellip='yes',recenter='yes',xylearn='yes',interactive='no')

            print 'Displaying isophotes from second pass using r = ',myradius
            print 'Hit q in the DS9 window to quit'
            iraf.isoexam(table=efile)
                
            flag=str(raw_input('Are you happy with the fit?  y=yes n=no x=quit '))
            flag=str(flag)
            print 'this is what I think you typed ',flag
            if flag.find('n') > -1:
                flag2=str(raw_input('What is the problem?  c=off-center r=set new radius e=set ellipticity p=set PA i=run interactively x=quit '))
                flag2=str(flag2)
                if flag2.find('r') > -1:
                    myr=input('Enter new radius to use ')
                    myradius=float(myr)
                    s='rm '+efile
                    os.system(s)
                    repeatflag=1
                elif flag2.find('e') > -1:
                    myellip=float(input('Enter new ellipticity (min=0.05) '))
                    if myellip < .05:#min value that ellipse can handle
                        myellip=.05
                    s='rm '+efile
                    os.system(s)
                    repeatflag=1
                    myellipflag=1
                elif flag2.find('p') > -1:
                    myPA=float(input('Enter new PA (min=-90, max=90) '))

                    if myPA < -90:
                        myPA=myPA+180
                    elif myPA > 90:
                        myPA = myPA-180

                    s='rm '+efile
                    os.system(s)
                    repeatflag=1
                    myPAflag=1
                elif flag2.find('i') > -1:
                    interactiveflag=1
                    s='rm '+efile
                    os.system(s)
                    repeatflag=1

                elif flag2.find('x') > -1:
                    repeatflag=0
                    return
                elif flag2.find('c') > -1:

                    s='mv *'+imprefix+'* OffCenter/'
                    print s
                    os.system(s)
                    repeatflag=0
                    print "repeatflag = ",repeatflag
            elif flag.find('x') > -1:
                repeatflag=0
                print 'i think repeatflag = 0', repeatflag
                return
            else:
                s='mv *'+imprefix+'* Finished/'
                os.system(s)
                repeatflag=0
                print 'i think repeatflag = 0 ', repeatflag

            print 'repeatflag = ',repeatflag
            

os.system('mkdir Finished')
os.system('mkdir NearbyObjects')
os.system('mkdir PartialImages')
os.system('mkdir OffCenter')
raw_input('Make sure ds9 is open.  Hit return when ready.')
iraf.stsdas()
iraf.analysis()
iraf.isophote()
iraf.tables()
iraf.ttools()
mipsfiles=glob.glob('*cutout-24.fits')
mipsfiles.sort()

mipswcsfiles=glob.glob('*cutout-24-rot.fits')
mipswcsfiles.sort()

sdssrfiles=glob.glob('*cutout-sdss.fits')
sdssrfiles.sort()

ipa=0
xcenter=21.0
ycenter=21.0
minr=2
initialr=5
maxr=20
iellip = .05
evalrad24=5#radius to measure PA and ellip at

sdssxcenter=50.5
sdssycenter=50.5
sdssminr=2
sdssinitialr=8
sdssmaxr=49
evalrad=15#radius to measure PA and ellip at

runellipse(mipswcsfiles,xcenter,ycenter,minr,ipa,initialr,maxr,iellip,myradius=evalrad24)
#runellipse(sdssrfiles,sdssxcenter,sdssycenter,sdssminr,ipa,sdssinitialr,sdssmaxr,iellip,nframe=2,myradius=evalrad)

