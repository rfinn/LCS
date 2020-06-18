#!/usr/bin/env python

import sys
from pylab import *
import numpy as np

infile=sys.argv[1]
print infile
in1=open(infile,'r')
#data=np.loadtxt(infile,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))
alldat=in1.readlines()
in1.close()
ngal=len(alldat)
for i in range(len(alldat)):
    alldat[i]=alldat[i].rstrip()
    alldat[i]=alldat[i].replace('INDEF','-00')
data=np.loadtxt(alldat,unpack=True)

column,sma,intens,intErr,pixVar,rms,ellip,ellipErr,pa,paErr,x0,y0,x0Err,y0Err,grad,gradErr,gradRerr,rsma,mag,maglErr,maguErr,tfluxE,tfluxC,tmagE,tmagC,npixE,mpixC,A3,B3,A4,B4,A3Err,B3Err,A4Err,B4Err,ndata,nflag,niter,stop,abig,sarea=data

#trying to figure out how to estimate the noise on the enclosed flux
##figure()
##myvar1=intens*ndata
##myvar2=intens*sarea
##plot(intErr**2*ndata,myvar1,'bo')
##plot(intErr**2*ndata,myvar2,'rs')
##print 'hey!'
##xmin,xmax=xlim()
##xl=arange(xmin,xmax)
##plot(xl,xl,'k:')
##plot(intErr**2*ndata,pixVar**2/sarea,'g^')
##show()

#plot enclosed flux vs intensity*area
figure()
myenclflux=zeros(len(intens),'f')
for i in range(len(intens)):
    t=0
    for j in range(i+1):
        if j < 1:
            print 'got here!',i,j,intens[j],sma[j]
            t=intens[j]*pi*(sma[j])**2
        else:
            t=t+intens[j]*pi*((sma[j])**2-(sma[j-1])**2)
    myenclflux[i]=t*(1-ellip[i])
    #myenclflux[i]=sum(intens[0:i+1]*pi*(sma[0:i+1]**2))#this is about a factor of 9 too big
##    #myenclflux[i]=sum(intens[0:i+1]*ndata[0:i+1])

#myenclflux=myenclflux*(1-ellip)
#myenclflux=intens*pi*sma**2/(1-ellip)
#plot(tfluxE,intens*sarea,'bo')
#plot(intens*sarea,tfluxC,'ro')
plot(tfluxE,myenclflux,'g^')
plot(tfluxC,myenclflux,'r^')
xmin,xmax=xlim()
xl=arange(xmin,xmax)
plot(xl,xl,'k:')
show()
