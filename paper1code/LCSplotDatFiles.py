#!/usr/bin/env python
from pylab import *
import numpy
import glob

#generate list of files to be imported
#out of order-tfmList=glob.glob('NGC6107/Alissa/NGC6107-*-cutout-24.dat')

#generate list-create array and arange numbers sm to lg
sdssList=glob.glob('/Users/rfinn/LocalClusters/cutouts/NGC6107/NGC6107-*-cutout-sdss.dat')
agcname=[]
for fil in sdssList:
    t=fil.split('-')
    agcname.append(float(t[1]))
agcname=array(agcname,'f')
sortedindex=agcname.argsort()
for i in range(10): #used for running tests
    sortedi=sortedindex[i]
    datfilesdss=sdssList[sortedi]
    n=datfilesdss.split('sdss')
    datfile24=n[0]+'24.dat'
    #create and set 24 variables
    sma24=[]
    intens24=[]
    intens24err=[]
    infile=open(datfile24,'r')
    lines=infile.readlines()
    for line in lines:
        t=line.split()
        sma24.append(float(t[1]));
        intens24.append(float(t[2]));
        intens24err.append(float(t[3]));
    infile.close()
    sma24=array(sma24,'f')
    intens24=array(intens24,'f')
    intens24err=array(intens24err,'f')
    #create and set sdss variables
    sma=[]
    intens=[]
    intenserr=[]
    infile=open(datfilesdss,'r')
    lines=infile.readlines()
    for line in lines:
        t=line.split()
        sma.append(float(t[1]));
        intens.append(float(t[2]));
        intenserr.append(float(t[3]));
    infile.close()
    sma=array(sma,'f')
    intens=array(intens,'f')
    intenserr=array(intenserr,'f')
    #perform necessary conversions
    #SMA pixels to arcsec
    sma24 = sma24*2.450016
    sma = sma*1.15
    #INTENS MJy/sr to microJy
    intens24 = intens24*141
    intens = (intens-930.292308)*.0229
    intens24err = intens24err*141
#intenserr = intentserr*.0229
    name=agcname[agcname.argsort()][i]
    #create plots
    if i<16:
        figure(1)
        subplot(4,4,(i+1))
        plot(sma,intens,'bo')
        plot(sma24,intens24,'go')
        xlabel('SMA (arcsec)',size='small')
        ylabel('Intensity(microJy)',size='small')
        title(name,size='small')
        show()

savefig('testplot.eps')
