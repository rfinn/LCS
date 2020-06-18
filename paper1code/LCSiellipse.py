#!/usr/bin/env python

'''

useage:  LCSiellipse.py fileprefix

run from OffCenter subdirectory

the program:
- grabs all files with CLUSTER*.fits
- makes sure that masked image mCLUSTER*.fits exists
  - will run imedit if mask is not found
- runs ellipse interactively
- saves result in ../Final subdirectory




'''

import glob
from pyraf import iraf
import os
from pylab import sqrt
import sys

def writeregfile(x,y,majoraxis,ellip,pa):
    outfile=open('ds9.reg','w')
    outfile.write('global color=green\n')
    outfile.write('physical \n')
    s='ellipse(%5.1f,%5.1f,%5.1f,%5.1f,%5.1f)'%(x,y,majoraxis,ellip*majoraxis,pa)
    outfile.write(s)
    outfile.close()
    

def findellipse(image,x,y):
    repeatflag=1
    while repeatflag:
        ellip=float(raw_input('enter ellip '))
        pa=float(raw_input('enter PA (between -90=+x and 90=-x) '))
        majoraxis=float(raw_input('enter Major Axis '))
        s='echo "image; ellipse %5.1f %5.1f %5.1f %5.1f %5.1f" |xpaset ds9 regions'%(x,y,majoraxis,(1-ellip)*majoraxis,(pa+90))
        os.system(s)
        flag=str(raw_input('are you happy with the ellipse?  y=yes n=no x=quit '))
        if flag.find('y') > -1:
            return ellip,pa
        elif flag.find('x') > -1:
            return 

    
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
    #print 'got here'
    #print files
    for i in range(len(files)):
        myradius=initialradius
    
        mfile=files[i]

        #mask image
        
        outfile1='m'+mfile
        if os.path.isfile(outfile1):
            print "found masked file ",outfile1
            print "skipping imedit and running ellipse interactively"
            continueWithProgram=1
            continueWithObject=1

        else:
            print "can't find masked file ",outfile1
            print "running imedit"
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
        try:
            os.remove(efile)
        except OSError:
            print 'everything is ok'
            #continue

        print "First pass through ellipse to find center"
        iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='no',sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no',interactive='no')
        #print 'Displaying isophotes from first pass.  Hit q in DS9 window to quit'
        #iraf.isoexam(table=efile)

        os.system('rm junk.txt')
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout='junk.txt')
        os.system("awk '{print $2, $7, $9, $11, $13}' < junk.txt > junk2.txt")
        #run ellipse a second time, keeping PA and ellip fixed
        #allow user to adjust the radius where PA and ellip are measured
        infile=open('junk2.txt','r')
        for line in infile:
            t=line.split()
            if float(t[0]) > myradius:
                newxcenter=float(t[3])
                newycenter=float(t[4])
                break
        s='rm '+efile
        os.system(s)

        #draw ellipse with ds9
        iraf.display(outfile1,1)
        (myellip,mypa)=findellipse(outfile1,newxcenter,newycenter)
        flag2=str(raw_input('Do you want to skip this one?  y=skip, any other key to continue '))
        if flag2.find('y') > -1:
            s='mv *'+imprefix+'* ../PeculiarGalaxies/'
            print s
            os.system(s)
            continue
        #run ellipse interactively
        #allow user to repeat until satisfied with script
        repeatflag=1
        while repeatflag:
            s='rm '+efile
            os.system(s)
            iraf.ellipse(input=outfile1,output=efile,x0=newxcenter,y0=newycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa0=mypa,hpa='yes',ellip0=myellip,hellip='yes',interactive='no')

            print 'Displaying isophotes from second pass using r = ',myradius
            print 'Hit q in the DS9 window to quit'
            iraf.isoexam(table=efile)
                
            flag=str(raw_input('Are you happy with the fit?  y=yes n=no x=quit '))
            flag=str(flag)
            print 'this is what I think you typed ',flag
            if flag.find('n') > -1:
                s='rm '+efile
                os.system(s)
                repeatflag=1
            elif flag.find('x') > -1:
                repeatflag=0
                print 'i think repeatflag = 0', repeatflag
                return
            else:
                s='mv *'+imprefix+'* ../Finished/'
                os.system(s)
                repeatflag=0
                print 'i think repeatflag = 0 ', repeatflag

            print 'repeatflag = ',repeatflag
            
def runellipseold(files,xcenter,ycenter,minr,ipa,initialr,maxr,iellip,nframe=1,myradius=15):
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
        iraf.ellipse(input=outfile1,output=efile,x0=xcenter,y0=ycenter,hcenter='no',sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no')
        print 'Displaying isophotes from first pass.  Hit q in DS9 window to quit'
        iraf.isoexam(table=efile)

        os.system('rm junk.txt')
        iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout='junk.txt')
        os.system("awk '{print $2, $7, $9, $11, $13}' < junk.txt > junk2.txt")
        #run ellipse a second time, keeping PA and ellip fixed
        #allow user to adjust the radius where PA and ellip are measured
        repeatflag=1
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
            iraf.ellipse(input=outfile1,output=efile,x0=newxcenter,y0=newycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr,pa=newPA,hpa='yes',ellip=newellip,hellip='yes')

            print 'Displaying isophotes from second pass using r = ',myradius
            print 'Hit q in the DS9 window to quit'
            iraf.isoexam(table=efile)
                
            flag=str(raw_input('Are you happy with the fit?  y=yes n=no x=quit '))
            flag=str(flag)
            print 'this is what I think you typed ',flag
            if flag.find('n') > -1:
                flag2=str(raw_input('What is the problem?  c=off-center r=set new radius x=quit '))
                flag2=str(flag2)
                if flag2.find('r') > -1:
                    myr=input('Enter new radius to use ')
                    myradius=float(myr)
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
            

raw_input('Make sure ds9 is open.  Hit return when ready.')
iraf.stsdas()
iraf.analysis()
iraf.isophote()
iraf.tables()
iraf.ttools()

#t=os.getcwd()
#s=t.split('cutouts')
#t=s[1].split('/')
#prefix=t[1]
#s=prefix+'*cutout-24.fits'
#mipsfiles=glob.glob(s)
#s=prefix+'*cutout-sdss.fits'
#print s
fileprefix=sys.argv[1]
print fileprefix
sdssrfiles=glob.glob(fileprefix)
print sdssrfiles


ipa=0

xcenter=23.0
ycenter=23.0
minr=2
initialr=6
maxr=20
iellip = .05
evalrad=15#radius to measure PA and ellip at

sdssxcenter=50.5
sdssycenter=50.5
sdssminr=2
sdssinitialr=8
sdssmaxr=49
evalrad=15#radius to measure PA and ellip at
#print 'sdssrfiles',sdssrfiles

os.system('mkdir ../PeculiarGalaxies')
#runellipse(mipsfiles,xcenter,ycenter,minr,ipa,initialr,maxr,iellip)
runellipse(sdssrfiles,sdssxcenter,sdssycenter,sdssminr,ipa,sdssinitialr,sdssmaxr,iellip,nframe=2,myradius=evalrad)

