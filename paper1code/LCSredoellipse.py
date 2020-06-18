#!/usr/bin/env python

import glob
from pyraf import iraf
import os

def rerunellipse(files,xcenter,ycenter,minr,initialr,maxr,nframe=1):
    repeatflag=1
    while (repeatflag > 0.1):
        mfile=files[i]
        t=mfile.split('.')
        k=t[0].split('m')
        efile=k[1]+'.tab'
        s='rm '+efile
        os.system(s)

        s="ellipse input=%s output=%s x0=%.1f y0=%.1f hcenter+ sma0=%.1f minsma=%.1f maxsma=%.1f inter+ \n"%(mfile,efile,xcenter,ycenter,initialr,minr,maxr)
#        s="ellipse(input=%s, output=%s, x0=%.1f, y0=%.1f, hcenter+, sma0=%.1f, minsma=%.1f, maxsma=%.1f, interactive=+) \n"%(mfile,efile,xcenter,ycenter,initialr,minr,maxr)
	

	iraf.ellipse(input=mfile,output=efile,x0=xcenter,y0=ycenter,hcenter='yes',sma0=initialr,minsma=minr,maxsma=maxr, interactive='yes')
	print 'Displaying isophotes.  Hit q to quit'
	iraf.isoexam(table=efile)
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
            

            


sdssredo=[250401,251496,251805,251807,252015,252376,252380,252437,252440,252453,252593,252757,252813,252828,253969,253983,258099,258128,268239,714770,714828,714841,716314,716342,736730,736782]

mipsredo=[-99999,250382,250401,250418,250436,250485,251805,252006,252376,252428,252429,252430,252431,252435,252437,252755,252758,252761,252763,252766,252769,252815,252818,253957,253960,253971,253993,253994,258099,258122,714770,714779,714791,714826,714841,716330,716355,736719,736730,736737,736749,736793,736864,736890,736917,736923,736941]

clustername='A2052'


Sdssrfiles=[]
for i in range(len(sdssredo)):
    name='m'+clustername+'-'+str(sdssredo[i])+'-cutout-sdss.fits'
    sdssrfiles.append(name)


mipsfiles=[]
for i in range(len(mipsredo)):
    name='m'+clustername+'-'+str(mipsredo[i])+'-cutout-24.fits'
    mipsfiles.append(name)

print sdssrfiles
print mipsfiles
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

rerunellipse(mipsfiles,xcenter,ycenter,minr,initialr,maxr)
rerunellipse(sdssrfiles,sdssxcenter,sdssycenter,sdssminr,sdssinitialr,sdssmaxr,nframe=1)

