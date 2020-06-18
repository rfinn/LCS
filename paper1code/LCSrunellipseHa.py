#!/usr/bin/env python
'''
DESCRIPTION:

Run iraf.stsdas.analysis.isophote.ellipse on R-band image.  Take the parameters
from the R-band fit and use this as input for running ellipse on H-alpha image.

In both cases, the 

PROCEDURE:

- pass in r_image and Ha_image
- assumes mask is named r_image+mask 



USEAGE:

uat_runellipse.py r_image.fits ha_image.fits

'''

import sys
import os
from pylab import *
from pyraf import iraf
from astropy.io import fits
from astropy.wcs import WCS
import ds9


mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/share/'



# get spirals catalog
infile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all.fits'
hdulist=fits.open(infile)
s=hdulist[1].data
hdulist.close()
agcdict=dict((a,b) for a,b in zip(s.AGCNUMBER,arange(len(s.AGCNUMBER))))
nsadict=dict((a,b) for a,b in zip(s.NSAID,arange(len(s.NSAID))))

def getNSA(agcnumber):
    agcnumber=int(agcnumber)
    pa=s.SERSIC_PHI[agcdict[agcnumber]]
    ba=s.SERSIC_BA[agcdict[agcnumber]]
    ra=s.RA[agcdict[agcnumber]]
    dec=s.DEC[agcdict[agcnumber]]
    return ra,dec,pa,1-ba

def call_measure_disk(input_image,mask_image=None,ipa=0,xcenter=None,ycenter=None,minr=2,initialr=20,maxr=100,iellip = .05,wave_band=0,keepfixed=0):
    print 'mask image = ',mask_image
    fdulist = fits.open(input_image)
    t=fdulist[0].data
    n2,n1=t.shape
    if xcenter == None:
        xc=n1/2.
        yc=n2/2.
    else:
        xc=xcenter
        yc=ycenter
    fdulist.close()

    if ipa < -90:
        ipa=ipa+180
    elif ipa > 90:
        ipa = ipa-180
        
    if initialr < 5:
        initialr = 5
    if initialr > 100:
        initialr = 50

    maxrad=max(n1,n2)/2.
    if initialr > maxrad:
        initialr = 0.25*maxrad
    if wave_band == 1:
        initialr=5
    if maxr > (n1/2.):
        maxr=n1-5
    # make up a value for now
    mag_zp=25.
    try:
        d.set('frame delete all')
    except:
        d=ds9.ds9()

    d.set('frame 1')
    d.set('single')
    # display r-band image if running Halpha image
    if wave_band == 1:
        d.set('frame 2')
        s='file '+input_image
        try:
            d.set(s)
        except:
            print 'trouble loading ',input_image

    print xc,yc,iellip,ipa,initialr,minr,maxr,mag_zp
    if keepfixed:
        newxcenter,newycenter,newellip,newPA=measure_disk(input_image,xc,yc,ipa,iellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=1,mask_image=mask_image)
        t=raw_input('enter x to quit, i to rerun in interactive mode, any other key to continue\n')
        if t.find('x') > -1:
            return 1
        if t.find('i') > -1:
            newxcenter,newycenter,newellip,newPA=measure_disk(i,disk_image,newxcenter,newycenter,newPA,newellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=1,mask_image=mask_image,einteractive=1)

    else:

        newxcenter,newycenter,newellip,newPA=measure_disk(input_image,xc,yc,ipa,iellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=0,mask_image=mask_image)
        t=raw_input('enter x to quit, i to rerun in interactive mode, any other key to continue\n')
        if t.find('x') > -1:
            return 1
        if t.find('i') > -1:
            newxcenter,newycenter,newellip,newPA=measure_disk(input_image,newxcenter,newycenter,newPA,newellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=0,einteractive=1,mask_image=mask_image)
        newxcenter,newycenter,newellip,newPA=measure_disk(input_image,newxcenter,newycenter,newPA,newellip,initialr,minr,maxr,mag_zp,band=wave_band,keepfixed=1,einteractive=0,mask_image=mask_image)

    return newxcenter,newycenter,newellip,newPA

def measure_disk(input_image,xcenter,ycenter,ipa,iellip,initialr,minr,maxr,zp,band=0,nframe=1,myradius=25,keepfixed=0,einteractive=0,mask_image=None):
    print 'inside measure disk'
    print 'mask image = ',mask_image
    myradius=initialr
    recentervalue='yes'
    interactivevalue='no'

    print '\n Running ellipse on ',input_image,'\n'
    mfile=input_image
    
    xcenter_cutout=xcenter
    ycenter_cutout=ycenter
    
    t=mfile.split('.')
    efile=t[0]+'.tab'
    efile_ascii=t[0]+'.dat'
    imprefix=t[0]
    print 'Running ellipse to fit isophotes to galaxy:',mfile
    if os.path.exists(efile):
        os.remove(efile)
    if keepfixed:
        print input_image,efile,xcenter_cutout,ycenter_cutout,initialr,minr,maxr,ipa
        iraf.ellipse(input=input_image,output=efile,dqf=mask_image,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='yes',recenter=recentervalue,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='yes',ellip=iellip,hellip='yes',interactive='yes',step=0.1,linear='no')
                
    else:
        print "First pass, letting PA and e vary"
        print input_image,efile,xcenter_cutout,ycenter_cutout,initialr,minr,maxr,ipa,
        if mask_image == None:
            iraf.ellipse(input=input_image,output=efile,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='no',recenter=0,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no',interactive=einteractive,step=0.1,linear='no')
        else:
            iraf.ellipse(input=input_image,output=efile,dqf=mask_image,x0=xcenter_cutout,y0=ycenter_cutout,hcenter='no',recenter=0,sma0=initialr,minsma=minr,maxsma=maxr,pa=ipa,hpa='no',ellip=iellip,hellip='no',interactive='yes',step=0.1,linear='no')
    print 'Displaying isophotes from first pass.  Hit q in DS9 window to quit'
    iraf.isoexam(table=efile)
    try:
        os.system('rm tp.00*')
    except:
        print 'no tp.00? files to remove'
    iraf.isoimap(input_image,table=efile)
    # get final ellipse parameters to return to the user
    if os.path.exists('junk.txt'):
        os.remove('junk.txt')
    iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout='junk.txt')
    s="awk '{print $2, $7, $9, $11, $13}' < "+"junk.txt > "+"junk2.txt"
    os.system(s)
    infile=open('junk2.txt','r')
    for line in infile:
        t=line.split()
        if float(t[0]) > myradius:

            newellip=float(t[1])
            if newellip < .05: # min value that ellipse can handle
                newellip=.05
            newPA=float(t[2])
            if newPA < -90:
                newPA=newPA+180
            elif newPA > 90:
                newPA = newPA-180
                
            newxcenter=float(t[3])
            newycenter=float(t[4])

    if os.path.exists(efile_ascii):
        os.remove(efile_ascii)
    iraf.tprint(table=efile,pwidth='INDEF',showhdr='no', Stdout=efile_ascii)
    return newxcenter,newycenter,newellip,newPA

iraf.stsdas()
iraf.analysis()
iraf.isophote()
iraf.tables()
iraf.ttools()

agcnumber=sys.argv[1]
haimage=agcnumber+'_Ha.fits'
rimage=agcnumber+'_R.fits'
mask=rimage.split('.fits')[0]+'mask.fits'
# get NSA input parameters for disk

ra,dec,pa,ellip=getNSA(agcnumber)

pa=90-pa
w= WCS(rimage)
px,py = w.wcs_world2pix(ra,dec,1)
# run ellipse on r image

maxradius=500
newxcenter,newycenter,newellip,newPA=call_measure_disk(rimage,wave_band=0,mask_image=mask,iellip=ellip,ipa=pa,xcenter=px,ycenter=py,keepfixed=1,maxr=maxradius)

# use ellipse parameters from r-band as input for Halpha

newxcenter,newycenter,newellip,newPA=call_measure_disk(haimage,wave_band=1,iellip=ellip,ipa=pa,xcenter=px,ycenter=py,minr=2,initialr=20,maxr=maxradius,keepfixed=1,mask_image=mask)

#newxcenter,newycenter,newellip,newPA=call_measure_disk(haimage,wave_band=1,ipa=newPA,xcenter=newxcenter,ycenter=newycenter,minr=2,initialr=20,maxr=100,iellip = newellip,keepfixed=1,mask_image=mask)

# convolve Halpha to 6" FWHM
mkw11mosaic_fwhm=4.3
pscale=.423
outsize=5.9/pscale # 6" divided by pixelscale
csize=sqrt((outsize/2.35)**2 - (mkw11mosaic_fwhm/2.35)**2)
# make convolved image

if os.path.isfile('c'+haimage):
    os.remove('c'+haimage)
iraf.imfilter.gauss(input=haimage,output='c'+haimage,sigma=5.9)
# run ellipse on convolved image
disk_image='c'+haimage
call_measure_disk(disk_image,wave_band=1,iellip=.05,ipa=pa,xcenter=px,ycenter=py,minr=2,initialr=20,maxr=maxradius,keepfixed=1,mask_image=mask)
os.rename(disk_image.split('.fits')[0]+'.dat',disk_image.split('.fits')[0]+'.e05.dat')


if ellip > .6:
    ellip = .6
newxcenter,newycenter,newellip,newPA=call_measure_disk('c'+haimage,wave_band=1,iellip=ellip,ipa=pa,xcenter=px,ycenter=py,minr=2,initialr=20,maxr=maxradius,keepfixed=1,mask_image=mask)

# run ellipse on convolved image
