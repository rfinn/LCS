#!/usr/bin/env python
#Erin's code
from pylab import *
from pyraf import iraf
import pylab
import os
import numpy
import glob
import pyfits

# Import 24 micron data for one galaxy
# Initialize variables
data1 = numpy.genfromtxt("A2063-254017-cutout-24.dat", unpack=True)
row_num1 = data1[0,:]
sma_pix1 = data1[1,:]
intens1, intens_err1 = data1[2,:], data1[3,:]

# Import SDSS data for one galalxy
# Initialize variables
data2 = numpy.genfromtxt("A2063-254017-cutout-sdss.dat", unpack=True)
row_num2 = data2[0,:]
sma_pix2 = data2[1,:]
intens2, intens_err2 = data2[2,:], data2[3,:]

# Convert sma in pixels to arcseconds
# 24 micron conversion from header, SDSS from web
sma_as1 = sma_pix1 * 2.450016
sma_as2 = sma_pix2 * 1.15

# Convert intensity from MJy/sr to microJy
intens1 = intens1 * 141
intens2 = intens2 * 0.02229
intens_err1 = intens_err1 * 141e-6
intens_err2 = intens_err2 * 0.02229e-6

# Plot of Intensity vs. Semi-major Axis
figure(33)
clf()
semilogy(sma_as1, intens1, 'r.')
semilogy(sma_as2, intens2, 'b+')
xlabel('Semi-major Axis (Arcseconds)')
ylabel('Intensity (microJY)')
title('Intensity vs. Semi-major Axis')
savefig("A2063_intensity.eps")

# All galaxies
dat24 = glob.glob('*cutout-24.dat')
datSDSS = glob.glob('*cutout-sdss.dat')
im24 = glob.glob('A2063*cutout-24.fits')
imSDSS = glob.glob('A2063*cutout-sdss.fits')
mask24 = glob.glob('mA2063*cutout-24.fits')
maskSDSS = glob.glob('mA2063*cutout-sdss.fits')
for j in range(32):
    i = 0
    n = j+1
    s = str(n)
    name = "A2063_implot"+s+".eps"

    figure(j+1)
    clf()
    subplots_adjust(left=0.1, right=.95,bottom=.15,top=0.9,wspace=0.15,hspace=0.1)
    while i < 6:
        frame=i*5

        subplot(6,5,frame+1)
        fits=pyfits.open(imSDSS[(j*6)+i])
        im=fits[0].data.copy()
        #axis([1.,5.,1.,5.])
        fits.close()
        axis('equal')
        imshow(-1.*(im),interpolation='nearest',origin='upper',cmap='binary')#,vmin=myvmin,vmax=myvmax)                           
        ax=gca()
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        gal1 = str(imSDSS[(j*6)+i])
        galname1 = gal1[6:12]
        text(.9, .5, galname1, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)

        subplot(6,5,frame+2)
        fits=pyfits.open(maskSDSS[(j*6)+i])
        im=fits[0].data.copy()
        #axis([1.,5.,1.,5.])
        fits.close()
        axis('equal')
        imshow(-1.*(im),interpolation='nearest',origin='upper',cmap='binary')#,vmin=myvmin,vmax=myvmax)                           
        ax=gca()
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        gal2 = str(maskSDSS[(j*6)+i])
        galname2 = gal2[7:13]
        text(.9, .5, galname2, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)

        subplot(6,5,frame+3)
        fits=pyfits.open(im24[(j*6)+i])
        im=fits[0].data.copy()
        #axis([1.,5.,1.,5.])
        fits.close()
        axis('equal')
        imshow(-1.*(im),interpolation='nearest',origin='upper',cmap='bone')#,vmin=myvmin,vmax=myvmax)                           
        ax=gca()
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        gal3 = str(im24[(j*6)+i])
        galname3 = gal3[6:12]
        text(.9, .5, galname3, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)

        subplot(6,5,frame+4)
        fits=pyfits.open(mask24[(j*6)+i])
        im=fits[0].data.copy()
        #axis([1.,5.,1.,5.])
        fits.close()
        axis('equal')
        imshow(-1.*(im),interpolation='nearest',origin='upper',cmap='bone')#,vmin=myvmin,vmax=myvmax)                           
        ax=gca()
        ax.set_yticklabels(([]))
        ax.set_xticklabels(([]))
        gal4 = str(dat24[(j*6)+i])
        galname4 = gal4[6:12]
        text(.9, .5, galname4, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)

        gal = str(dat24[(j*6)+i])
        galname = gal[6:12]
        if dat24[(j*6) +i] == 'A2063-254016-cutout-24.dat':
            i = i+1
            continue
        elif dat24[(j*6)+i] == 'A2063-714934-cutout-24.dat':
            i = i+1
            continue
        elif dat24[(j*6)+i] == 'A2063-714967-cutout-24.dat':
            i = i+1
            continue
        elif ((j*6)+i) > len(dat24):
            print "done"
            break
        else:
            data24 = numpy.genfromtxt(dat24[(j*6)+i], unpack=True)
            dataSDSS = numpy.genfromtxt(datSDSS[(j*6)+i], unpack=True)
            row_num24, row_numS = data24[0,:], dataSDSS[0,:]
            sma24, smaS = data24[1,:] * 2.450016, dataSDSS[1,:] * 1.15
            intens24, intensS =data24[2,:] * 141, (dataSDSS[2,:]-919) * 0.02229
            intens_err24, intens_errS = data24[3,:] * 141, (dataSDSS[3,:]) * 0.02229
            subplot(6,5,frame+5)
            plot(sma24, intens24, 'r.')
            errorbar(sma24, intens24, yerr=intens_err24,fmt=None,ecolor='r')
            plot(smaS, intensS, 'b+')
            errorbar(smaS, intensS, yerr=intens_errS,fmt=None,ecolor='b')
            axhline(y=1,color='r',ls=':')
            axhline(y=.2,color='b',ls=':')
            ax=gca()
            ax.set_yscale('log')
            frame=(i+1)*1.
            axis([0.,60.,0.01,1200.])
            ax=gca()
            if (frame <  30):
                ax.set_xticklabels(([]))
                ax.set_yticklabels(([]))
           # else:
                #xticks(arange(0, 90, 20), fontsize=10)
            #test=(i/4.-floor(i/4.))
            #if (test > .2):
            #    ax.set_yticklabels(([]))
            #else:
            #    yticks(fontsize=10)            
            i = i+1
        ax=gca()
        text(.6, .9, galname, horizontalalignment='left', verticalalignment='center',transform=ax.transAxes)
    
   # text(-100, 0.01, 'SMA (arcsec)', horizontalalignment='center', verticalalignment='center', weight='bold')
   # text(-295, 10e6, 'Intensity (microJy)', horizontalalignment='center', verticalalignment='center', weight='bold',rotation=90)
   # text(-90, 10e13, 'Intensity vs. Semi-major Axis', horizontalalignment='center', verticalalignment='center', weight='bold')
    #savefig(name)
       



