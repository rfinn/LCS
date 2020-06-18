#!/usr/bin/env python
"""
adapted from wfediscsreduce24.py

re-written for LCS 24 um scans 

useage:

LCSreduce24.py  MKW11

prefix is the cluster name as listed in /Users/rfinn/research/WifiEdiscs/MipsScans/Rudnick/

This will run flatfield for each aor, then combine the file lists so that you can run mosaic and apex once for the entire mosaic.

"""

import sys, glob, os
#import numarray as n
import numpy as N
import pylab
#from math import *
from pyraf import iraf
#import pyfits
import random
import mystuff
#bcdpath=r13810688/ch1/bcd

SqDegS=0.

#iraf.images()
#iraf.images.imutil()
#iraf.artdata()
#import spitzergetnoise

def findnearest(x1,y1,x2,y2,delta):#x2 is array
	dmin = 100000000000000000000000000000000000.
	matchflag=1
	nmatch=0
	for i in range(len(x2)):
		d = N.sqrt((x1-x2[i])**2 + (y1-y2[i])**2)
		if d < delta:
			nmatch=nmatch+1
		if d < dmin:
			dmin = d
			imatch = i

	
	if dmin > delta:
		imatch = 0
	matchflag = 0
	return imatch, matchflag,nmatch




def runflatfield():
	os.chdir(bcdpath)
	os.system('SpitzerMakeFilelists.py')
	os.system('flatfield.pl -n flatfield_24_LC.nl')



def extractphot():
    combinepath=bcdpath+'pbcd/Combine/'

    if (SqDegS > 0.1):
	    combinepath=bcdpath
    os.chdir(combinepath)
    os.system('mkdir cal')
    os.system('mkdir cdf')
    os.system('cp /Users/rfinn/research/clusters/spitzer/MIPS24_PRF_HD_center.fits cal/.')
    os.system('cp /Applications/mopex/cal/mips24_prf_mosaic_2.45_4x.fits cal/.')
    #os.system('cp /Users/rfinn/clusters/spitzer/PRF_estimate_cl1216.fits prf_estimate_8_11.nl cal/MIPS24_PRF_HD_center.fits')
    os.system('cp /Users/rfinn/research/clusters/spitzer/*.nl cdf/.')
    #os.system('cp /Users/rfinn/clusters/spitzer/apex_1frame_MIPS24_step2.nl cdf/.')
    #os.system('cp /Users/rfinn/clusters/spitzer/prf_estimate_8_11.nl cdf/.')

    os.system('apex_1frame.pl -n apex_1frame_MIPS24_step1.nl')
    input=open('output_apex_step1/mosaic_extract.tbl','r')
    output=open('output_apex_step1/mosaic_extract_clean.tbl','w')
    i=0
    for line in input:
	if line.find('\\') > -1:
	    output.write(line)
	    continue
	if line.find('|') > -1:
	    output.write(line)
	    continue
	f=line.split()#check to make sure source is far from edge
	if (float(f[1]) < 15.) :
	    continue
	if (float(f[2]) < 15.) :
	    continue
	if (float(f[1]) > 128.) :
	    continue
	if (float(f[2]) > 128.) :
	    continue
	if (i > 3):
	    print "Got 3 sources in mosaic_extract.tbl\n"
	    break
	output.write(line)
	i=i+1
    input.close()
    output.close()
    os.system('prf_estimate.pl -n prf_estimate_8_11.nl')
    os.system('cp prf_estimate/PRF.fits cal/PRF_estimate.fits')
    os.system('apex_1frame.pl -n apex_1frame_MIPS24_step2.nl')
    #os.system('apex_qa.pl -n apex_qa_myredux.nl')#produce point source subtracted image to gauge how well detection went
    #image='apex_qa/Mosaic/residual_mosaic.fits'
    #iraf.images.tv.display(image,3,contrast=0.01)#CL1040


def writephotcat():

    combinepath=bcdpath+'pbcd/'
    os.chdir(combinepath)
    print combinepath

    #apertures are ("1,1.5,2,2.6,3,3.5,4,4.5,5,5.5") pixels
    #os.system('rm ApertureError')#adding this so that noise is remeasured after I updated getnoise program
    try:
	in2=open('ApertureError','r')
	nap=0
	for line in in2:
	    if line.find('#') >-1:
		continue
	    nap += 1
	aveaperr=N.zeros(nap,'f')
	in2.close()
	in2=open('ApertureError','r')
	i=0
	for line in in2:
	    if line.find('#') >-1:
		continue
	    t=line.split()
	    aveaperr[i]=float(t[0])
	    i += 1
	in2.close()
	print 'Found ApertureError'
	print aveaperr
    except:
	print "Couldn't find file ApertureError, so measuring errors"
	(aveap,aveaperr,avearea,aveareaerr,a,b)=spitzergetnoise.runit()
	out2=open('ApertureError','w')
	out2.write('#apertures are ("1,1.5,2,2.6,3,3.5,4,4.5,5,5.5") pixels \n')
	for k in range(len(aveaperr)):
	    aveaperr[k]=float(aveaperr[k])
	    s=str(aveaperr[k])+'\n'
	    out2.write(s)
	out2.close()
    input=open('mosaic_extract_raw.tbl','r')
    output=open('mosaic_extract_final.tbl','w')
    i=0
    for line in input:
	if line.find('Conversion') > -1:
	    t=line.split('=')
	    convfactor=float(t[1])#conversion from ADU to uJy
	    aperr=aveaperr*convfactor #convert noise in ADU to uJy using conv factor from apex
	    print "Conversion Factor = ",convfactor
	    #print "aveaperr = ",aveaperr
	    #print "aperr = ",aperr
	    continue
	if line.find('\\') > -1:
	    output.write(line)
	    continue
	if line.find('|srcid') > -1:
	    head2='| err_ap1| err_ap2| err_ap3| err_ap4| err_ap5| err_ap6| err_ap7| err_ap8| err_ap9| err_ap10| \n' 
	    line2=line[0:(len(line)-2)]+head2
	    output.write(line2)
	    continue
	if line.find('|') > -1:
	    output.write(line)
	    continue
	f=line.split()
	#snr=float(f[26])/float(aperr[2])
	snr=float(f[18])
	if snr > 0:
	    errors= "  %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e "%(aperr[0],aperr[1],aperr[2],aperr[3],aperr[4],aperr[5],aperr[6],aperr[7],aperr[8],aperr[9])
	    #print errors
	    line2=line[0:(len(line)-2)]+errors+'\n'
	    output.write(line2)
	i=i+1
    input.close()
    output.close()


    if SqDegS < 0.1:
	    string='cp mosaic_extract_final.tbl '+catalogpath+prefix+'mosaic_extract_final.tbl'
	    os.system(string)
	    string='cp mosaic_minus_median_extract.fits /Users/rfinn/research/WifiEdiscs/FinalImages/'+prefix+'_mosaic_minus_median.fits'
	    os.system(string)

def completeness():#measure completeness on final image
	
    #combinepath=bcdpath+'pbcd/Combine/output_apex_step2'
    if SqDegS > 0.1:
	    combinepath=bcdpath+'/output_apex_step2'
	    os.chdir(combinepath)
	    os.system('cp mosaic_extract_raw.tbl mosaic_extract_final.tbl')
    else:
	    combinepath=bcdpath+'pbcd/Combine/apex_1frame_step2'
	    os.chdir(combinepath)

    file='mosaic_extract_final.tbl'
    input=open(file,'r')
    xgal=[]#positions of previous detections with snr > 3
    ygal=[]
    fap4gal=[]
    for line in input:
	if line.find('#') > -1: #skip lines with '#' in them
	    continue
	if line.find('\\') > -1: #skip lines with '#' in them
	    continue
	if line.find('|') > -1: #skip lines with '#' in them
	    continue
	t=line.split()
	xgal.append(float(t[8]))
	ygal.append(float(t[10]))
	fap4gal.append(float(t[28]))
    input.close()
    xgal=N.array(xgal,'f')
    ygal=N.array(ygal,'f')


    fsimall=[]
    matchflagsimall=[]
    f2all=[]
    f3all=[]
    f4all=[]
    deblendsimall=[]
    snrsimall=[]

    myminmag=24.75
    mymaxmag=27.4
    myfmin=10.**((25.-mymaxmag)/2.5)#ZP=25
    myfmax=10.**((25.-myminmag)/2.5)#ZP=25


    #below is loop to create image w/artificial sources, extract, and compare

    for k in range(100):
	    createflag=1.#create image w/artificial sources?
	    detectflag=1.#detect sources in image?
	    if createflag > 0.1:
		    xnoise=[]
		    ynoise=[]
		    infile=open('noisecoords.dat','r')#read in list of coordinates corresponding to positions where no real source exists.  These are generated by spitzergetnoise.py.
		    for line in infile:
			    t=line.split()
			    xnoise.append(float(t[0]))
			    ynoise.append(float(t[1]))
	    infile.close()
	
	
	    nstar=10
    
	    xsim=N.zeros(nstar,'d')
	    ysim=N.zeros(nstar,'d')
	    msim=N.zeros(nstar,'d')
	    outfile=open('stars.coords.dat','w')
	    for i in range(nstar):
	    #j=int(round(1.*len(xnoise)*random.uniform(0,1)))

	    #xsim[i]=xnoise[j]
	    #ysim[i]=ynoise[j]
		    j=0
		    for j in range(10000):
			    xt=int(round(random.uniform(5.,125.)))
			    yt=int(round(random.uniform(5.,140.)))
			    d=pylab.sqrt((xt-xgal)**2+(yt-ygal)**2)#make sure sim galaxies are not near real galaxies
			    if min(d) > -1.:
				    d2=pylab.sqrt((xt-xsim)**2+(yt-ysim)**2)#make sure sim points are not on top of each other
				    if min(d2) > 5.:
					    print i,'got a good point after ',j,' tries',xt,yt
					    break
				    j=j+1
		    xsim[i]=xt
		    ysim[i]=yt
		    k=random.uniform(myfmin,myfmax)
		    msim[i]=25.-2.5*pylab.log10(k)
	    #print k,msim[i] 
		    s='%8.2f %8.2f %8.2f \n' % (xsim[i],ysim[i],msim[i])
		    outfile.write(s)
	    outfile.close()
	      
	
	#os.system('rm stars.coords.dat')
	#iraf.starlist('stars.coords.dat',nstars=100,spatial='uniform',xmax=130,ymax=145,luminosity='uniform',minmag=22.,maxmag=30.0,mzero=22.0,sseed='INDEF',power=0.6,alpha=0.74,lseed='INDEF')
	
    
	    os.system('rm mosaic-completeness.fits')
        #iraf.mkobjects(input='mosaic_minus_median_extract.fits',output='mosaic-completeness.fits',objects='stars.coords.dat',radius=1.13,magzero=25.,background=0.,gain=5.,rdnoise=0.,poisson='no')#don't convolve w/PRF
	    #os.system('cp ../cal/MIPS24_PRF_HD_center.fits .')#convolve star w/SSC PRF
	    os.system('cp ../cal/mips24_prf_mosaic_2.45_4x.fits .')#convolve star w/SSC PRF
	    iraf.mkobjects(input='mosaic_minus_median_extract.fits',output='mosaic-completeness.fits',objects='stars.coords.dat',radius=14,star='mips24_prf_mosaic_2.45_4x.fits',magzero=25.,background=0.,gain=5.,rdnoise=0.,poisson='no')
        #os.system('cp ../cal/PRF_estimate.fits .')#convolve gaussian w/measured PRF
        #iraf.mkobjects(input='mosaic_minus_median_extract.fits',output='mosaic-completeness.fits',objects='stars.coords.dat',radius=15,star='PRF_estimate.fits',magzero=25.,background=0.,gain=5.,rdnoise=0.,poisson='no')
	    os.system('ls *.fits')
	    os.system('pwd')
	    iraf.display('mosaic_minus_median_extract.fits',1,contrast=0.01)
	    iraf.display('mosaic-completeness.fits',2,contrast=0.01)
	    iraf.tvmark(1,'stars.coords.dat')
	    iraf.tvmark(2,'stars.coords.dat')
	    fsim=10.**((25.-msim)/2.5)#ZP=25

	    if createflag < .1:#read in positions and magnitudes of artdata sources
		    xsim=[]
		    ysim=[]
		    msim=[]
		    infile=open('stars.coords.dat','r')
		    for line in infile:
			    if line.find('#') > -1:
				    continue
			    t=line.split()
			    xsim.append(float(t[0]))
			    ysim.append(float(t[1]))
			    msim.append(float(t[2]))
		    infile.close()
		    xsim=N.array(xsim,'f')
		    ysim=N.array(ysim,'f')
		    msim=N.array(msim,'f')
		    
		    fsim=10.**((25.-msim)/2.5)#ZP=25

	    if detectflag > 0.1:#now run detection on mosaic-completeness.fits
		    if SqDegS > 0.1:
			    combinepath=bcdpath
		    else:
			    combinepath=bcdpath+'pbcd/Combine/'
		    os.chdir(combinepath)
		    print combinepath
		    #os.system('apex_1frame.pl -n apex_1frame_MIPS24_step2.nl -i output_apex_step2/mosaic-completeness.fits')
	
		    #combinepath=bcdpath+'pbcd/Combine/output_apex_step2'

	
		    if SqDegS > 0.1:
			    s='cp /Users/rfinn/clusters/spitzer/apex_1frame_step2all_400.nl '+bcdpath+'cdf/'
			    os.system(s)
			    os.system('apex_1frame.pl -n apex_1frame_step2all_400.nl -i output_apex_step2/mosaic-completeness.fits')
			    combinepath=bcdpath+'output_apex_step2/'
		    else:
			    os.system('apex_1frame.pl -n apex_1frame_step2all.nl -i apex_1frame_step2/mosaic-completeness.fits')
			    combinepath=bcdpath+'pbcd/Combine/apex_1frame_step2'
		    os.chdir(combinepath)
		    print combinepath
		    file='mosaic-completeness_extract_raw.tbl'
		    input=open(file,'r')
		    ngal=0
		    for line in input:
			    if line.find('Conversion') > -1:
				    t=line.split('=')
				    convfactor=float(t[1])#conversion from ADU to uJy
  	    #aperr=aveaperr*convfactor #convert noise in ADU to uJy using conv factor from apex
				    print "Conversion Factor = ",convfactor
	    #print "aveaperr = ",aveaperr
	    #print "aperr = ",aperr
				    continue
			    if line.find('#') > -1: #skip lines with '#' in them
				    continue
			    if line.find('\\') > -1: #skip lines with '#' in them
				    continue
			    if line.find('|') > -1: #skip lines with '#' in them
				    continue
			    ngal=ngal+1
		    input.close()
    
	

	    id24 = N.zeros(ngal,'f')
	    imagex24 = N.zeros(ngal,'f')
	    imagey24  = N.zeros(ngal,'f')
	    ra24 = N.zeros(ngal,'f')
	    dec24 = N.zeros(ngal,'f')
	    f24 = N.zeros(ngal,'d')#flux
	    errf24 = N.zeros(ngal,'d')
	    fap1 = N.zeros(ngal,'d')#flux in aperture 1 (1,1.5,2,2.6,3,3.5,4,4.5,5.,5.5) pixels
	    fap2 = N.zeros(ngal,'d')#flux
	    fap3 = N.zeros(ngal,'d')#flux
	    fap4 = N.zeros(ngal,'d')#flux in ap 4 - this is one w/ap cor of 1.67 (Calzetti et al 2007)
	    fap5 = N.zeros(ngal,'d')#flux
	    fap6 = N.zeros(ngal,'d')#flux
	    fap7 = N.zeros(ngal,'d')#flux
	    fap8 = N.zeros(ngal,'d')#flux
	    fap9 = N.zeros(ngal,'d')#flux
	    fap10 = N.zeros(ngal,'d')#flux
	    snr24 = N.zeros(ngal,'d')#SNR calculated by mopex
	    deblend = N.zeros(ngal,'f')#SNR calculated by mopex
	    

	    input=open(file,'r')
	    i=0
	    output=open('xy24raw.dat','w')
	    for line in input:
		    if line.find('#') > -1: #skip lines with '#' in them
			    continue
		    if line.find('\\') > -1: #skip lines with '#' in them
			    continue
		    if line.find('|') > -1: #skip lines with '#' in them
			    continue
	 
	
		    t=line.split()
	#print "length of t = ",len(t)
	#print (t[8]),(t[10]),(t[13]),(t[14]),(t[18]),(t[2]),(t[23]),(t[24]),(t[25]),(t[26]),(t[27]),(t[28]),(t[29]),(t[30]),(t[31]),(t[32])

		    (imagex24[i],imagey24[i],f24[i],errf24[i],snr24[i],deblend[i],fap1[i],fap2[i],fap3[i],fap4[i],fap5[i],fap6[i],fap7[i],fap8[i],fap9[i],fap10[i])=(float(t[8]),float(t[10]),float(t[13]),float(t[14]),float(t[18]),float(t[2]),float(t[25]),float(t[26]),float(t[27]),float(t[28]),float(t[29]),float(t[30]),float(t[31]),float(t[32]),float(t[33]),float(t[34]))
		    s='%6.2f %6.2f \n'%(imagex24[i],imagey24[i])
		    output.write(s)

		    i=i+1
	    input.close()#44 -> 43
	    output.close()
	    iraf.tvmark(1,'xy24raw.dat',color=204,radi=2)
	    iraf.tvmark(2,'xy24raw.dat',color=204,radi=2)
    
	    delta=1.#max number of pixels for a match

	    #get rid of objects that were detected in original image.  Otherwise, matching will think any object near a sim galaxy is the sim galaxy.  A faint galaxy placed on type of a pre-existing bright galaxy will be detected.

            newgalflag=N.ones(len(imagex24),'i')
	    for i in range(len(imagex24)):
		    (imatch, matchflag,nmatch)=findnearest(imagex24[i],imagey24[i],xgal,ygal,delta)
		    if matchflag > 0.:
			    dflux=abs(fap4gal[imatch] - fap4[i])/fap4[i]
			    if dflux < .1:#position of real galaxy, flux difference less than 10% -> not a new galaxy
				    newgalflag[i] = 0
	    #keep only galaxies that are new
	    imagex24 = N.compress(newgalflag,imagex24)
	    imagey24  = N.compress(newgalflag,imagey24)
	    fap1 = N.compress(newgalflag,fap1)
	    fap2 = N.compress(newgalflag,fap2)
	    fap3 = N.compress(newgalflag,fap3)
	    fap4 = N.compress(newgalflag,fap4)
	    fap5 = N.compress(newgalflag,fap5)
	    fap6 = N.compress(newgalflag,fap6)
	    fap7 = N.compress(newgalflag,fap7)
	    fap8 = N.compress(newgalflag,fap8)
	    fap9 = N.compress(newgalflag,fap9)
	    fap10 =N.compress(newgalflag,fap10)
	    snr24 =N.compress(newgalflag,snr24)
	    deblend = N.compress(newgalflag,deblend)

	    delta=2.#max number of pixels for a match
	    matchflagsim=N.zeros(len(xsim),'i')
	    fmeas1=N.zeros(len(xsim),'f')
	    fmeas2=N.zeros(len(xsim),'f')
	    fmeas3=N.zeros(len(xsim),'f')
	    fmeas4=N.zeros(len(xsim),'f')
	    fmeas5=N.zeros(len(xsim),'f')
	    fmeas6=N.zeros(len(xsim),'f')
	    fmeas7=N.zeros(len(xsim),'f')
	    fmeas8=N.zeros(len(xsim),'f')
	    fmeas9=N.zeros(len(xsim),'f')
	    fmeas10=N.zeros(len(xsim),'f')
	    fmeas24=N.zeros(len(xsim),'f')
	    deblendsim=N.zeros(len(xsim),'f')
	    snrsim=N.zeros(len(xsim),'f')
	    for i in range(len(xsim)):
		    (imatch, matchflag,nmatch)=findnearest(xsim[i],ysim[i],imagex24,imagey24,delta)
		    matchflagsim[i]=matchflag
		    if matchflag > .1:
			    fmeas1[i]=fap1[int(imatch)]
			    fmeas2[i]=fap2[int(imatch)]
			    fmeas3[i]=fap3[int(imatch)]
			    fmeas4[i]=fap4[int(imatch)]
			    fmeas5[i]=fap5[int(imatch)]
			    fmeas6[i]=fap6[int(imatch)]
			    fmeas7[i]=fap7[int(imatch)]
			    fmeas8[i]=fap8[int(imatch)]
			    fmeas9[i]=fap9[int(imatch)]
			    fmeas10[i]=fap10[int(imatch)]
			    fmeas24[i]=f24[int(imatch)]
			    deblendsim[i]=deblend[int(imatch)]
			    snrsim[i]=snr24[int(imatch)]
			    



	    fsimall=fsimall+list(fsim)
	    matchflagsimall=matchflagsimall+list(matchflagsim)
	    f2all=f2all+list(fmeas2)
	    f3all=f3all+list(fmeas3)
	    f4all=f4all+list(fmeas4)
	    deblendsimall=deblendsimall+list(deblendsim)
	    snrsimall=snrsimall+list(snrsim)


    fsim=N.array(fsimall,'f')
    matchflagsim=N.array(matchflagsimall,'f')
    fmeas2=N.array(f2all,'f')
    fmeas3=N.array(f3all,'f')
    fmeas4=N.array(f4all,'f')
    deblendsim=N.array(deblendsimall,'f')
    snrsim=N.array(snrsimall,'f')


    #make plots using all realizations 
    pylab.cla()
    pylab.clf()
    fsim=fsim*convfactor
    fs=pylab.compress((matchflagsim > 0.1) & (deblendsim < 1.5),fsim)
    #f1=pylab.compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas1)
    f2=pylab.compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas2)
    f3=pylab.compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas3)
    f4=pylab.compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas4)
    #f242=pylab.compress((matchflagsim > 0.1) & (deblendsim < 1.5),fmeas24)

    r4=pylab.median(fs/f4)
    r3=pylab.median(fs/f3)
    r2=pylab.median(fs/f2)
    print "average ratios ap 4",pylab.average(fs/f4),r4,pylab.std((fs/f4)/pylab.average(fs/f2))
    print "average ratios ap 3",pylab.average(fs/f3),pylab.median(fs/f3),pylab.std((fs/f3)/pylab.average(fs/f3))
    print "average ratios ap 2",pylab.average(fs/f2),pylab.median(fs/f2),pylab.std((fs/f2)/pylab.average(fs/f2))

    s='f4 w/apcor = %3.2f(%4.2f)'%(r4,pylab.average(abs(fs-f4*r4)/fs))
    pylab.plot(fs,f4*r4,'b.',label=s)
    pylab.plot(fs,f4,'bo',label='f4')
    s='f3 w/apcor = %3.2f(%4.2f)'%(r3,pylab.average(abs(fs-f3*r3)/fs))
    pylab.plot(fs,f3*r3,'g.',label=s)
    pylab.plot(fs,f3,'go',label='f3')
    s='f2 w/apcor = %3.2f(%4.2f)'%(r2,pylab.average(abs(fs-f2*r2)/fs))
    pylab.plot(fs,f2*r2,'r.',label=s)
    pylab.plot(fs,f2,'ro',label='f2')
    #pylab.plot(fs,f1,'co',label='f1')
    #pylab.plot(fs,f242,'k.',label='f24')
    pylab.legend(loc='best')
    x=N.arange(0.,max(fs),10.)
    y=x
    pylab.plot(x,y,'k-')
    #y=2.*x
    #pylab.plot(x,y,'k--')
    #y=3.*x
    #pylab.plot(x,y,'k--')
    #y=4.*x
    #pylab.plot(x,y,'k--')
    #y=5.*x
    #pylab.plot(x,y,'k--')
    pylab.xlabel('F(24) Input')
    pylab.ylabel('F(24) measured')
    #pylab.axis([0.,50.,0.,50.])
    s=str(prefix)+'fluxcomp.eps'
    pylab.savefig(s)

    pylab.cla()
    pylab.clf()


    nbins=20
    fmin=10.#min(fsim)
    fmax=max(fsim)
    df=5.#(fmax-fmin)/(1.*nbins)
    bins=N.arange(fmin,(fmax+df),df)



    (xbin,ybin,ybinerr)=mystuff.completeness(bins,fsim,matchflagsim)
    s=str(prefix)+'FracComplvsFlux.dat'
    outdat=open(s,'w')
    print "Completeness vs Input Flux"
    for i in range(len(xbin)):
	    print i, xbin[i],ybin[i],ybinerr[i]
	    t='%8.2f %8.4f %8.4f\n'%(xbin[i],ybin[i],ybinerr[i])
	    outdat.write(t)
    outdat.close()
    #for i in range(len(fsim)):
	#if snrsim[i] > 3.:
	#    print i, fsim[i],matchflagsim[i],deblendsim[i],abs(fsim[i]-fmeas4[i]*1.67)/fsim[i],snrsim[i]
    #(xbin,ybin2,ybin2err)=mystuff.scipyhist2(bins,fmeas4)
    #pylab.plot(xbin,ybin,'bo')
    #pylab.plot(xbin,ybin2,'ro')
    #s=str(prefix)+'NDetectvsFlux.eps'
    #pylab.savefig(s)

    pylab.cla()
    pylab.clf()
    pylab.plot(xbin,ybin,'ko')
    pylab.errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k')
    s=str(prefix)+'FracComplvsFlux.eps'
    pylab.axhline(y=1.0,ls='-')
    pylab.axhline(y=.8,ls='--')
    pylab.axvline(x=80.0,ls=':',color='b')
    pylab.xlabel('Input Flux (uJy)')
    pylab.ylabel('Completeness')
    pylab.axis([0.,max(xbin)+df,-.05,1.05])

    pylab.savefig(s)
    
    if SqDegS < 0.1:

	    os.system('cp *.eps /Users/rfinn/clusters/spitzer/completeness/.')
	    os.system('cp *vsFlux.dat /Users/rfinn/clusters/spitzer/completeness/.')



##################   Main Program Starts Here!   #########################

prefix=sys.argv[1]
mypath=os.getcwd()
if mypath.find('home') > -1:
	rootpath='/home/rfinn/research/LocalClusters/MIPS/rawdata/'+prefix
	catalogpath='/home/rfinn/research/LocalClusters/MIPS/catalogs/' #catalogpath is the directory where the final catalog will be copied to
if mypath.find('Users') > -1:
	rootpath='/Users/rfinn/research/LocalClusters/MIPS/rawdata/'+prefix
	catalogpath='/Users/rfinn/research/LocalClusters/MIPS/catalogs/' #catalogpath is the directory where the final catalog will be copied to

print 'got here 1'
os.chdir(rootpath)
dirs=glob.glob('r*')#get listing of aor #s
os.system('mkdir FullMosaic')
i=0
print 'got here 2'
for rdir in dirs:#run flatfield on each aor
	bcdpath=rootpath+'/'+rdir+'/ch1/bcd/'
	spitzersourcpath=bcdpath+'pbcd/'#where final catalog is kept & where spitzersource shoud be run
	os.chdir(bcdpath)
	runflatfield()
	
	s='cp DmaskList.txt '+rootpath+'/FullMosaic/'+str(i)+'DmaskList.txt'
	os.system(s)
	s='cp SigmaList.txt '+rootpath+'/FullMosaic/'+str(i)+'SigmaList.txt'
	os.system(s)
	path=bcdpath+'mopex_flat_scan/Correct/'
	os.chdir(path)
	print path
	s="sed -e 's@mopex_flat_scan/Correct@"+path+"@g'< correct_InputImageList.txt > mycorrect_InputImageList.txt"
	print s
	os.system(s)
	s='cp '+bcdpath+'mopex_flat_scan/Correct/mycorrect_InputImageList.txt '+rootpath+'/FullMosaic/'+str(i)+'correct_InputImageList.txt'
	os.system(s)

	print i,rdir
	i += 1

print 'got here 3'
#make mosaic from flattened images
p=rootpath+'/FullMosaic/'
os.chdir(p)
os.system('rm All*.txt')
print 'got here 4'
os.system('cat *DmaskList.txt > AllDmaskList.txt')
print 'got here 5'
os.system('cat *InputImageList.txt > AllInputImageList.txt')
print 'got here 6'
os.system('cat *SigmaList.txt > AllSigmaList.txt')
print 'got here 7'
if mypath.find('home') > -1:
	s="sed -e 's@MKW8@"+prefix+"@g'< /home/rfinn/mopex/mycdf/MKW8scan24.nl > /home/rfinn/mopex/mycdf/"+prefix+"scan24.nl"
	print s
	os.system(s)

if mypath.find('Users') > -1:
	s="sed -e 's@MKW8@"+prefix+"@g'< /Users/rfinn/mopex/mycdf/MKW8scan24.nl > /Users/rfinn/mopex/mycdf/"+prefix+"scan24.nl"
	print s
	os.system(s)

s='echo Now run mopex gui mosaic+apex using namelist '+prefix+'scan24.nl'
os.system(s)

	


#os.system('mosaic.pl -n mosaic_24_kiss.nl')
#create24nl()#creates 24um apex_1frame nl for each galaxy


# didn't use these for KISS - ran apex through gui
#extractphot()#ran this on Ken's images
#extractphotnoring()
#writephotcat()#had to run this on Ken's images
#completeness()#then running this to estimate completeness on Ken's images

#os.chdir(spitzersourcepath)
#os.system('spitzersource.py 1')



