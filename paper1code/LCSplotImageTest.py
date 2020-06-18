#!/usr/bin/env python
from pylab import *
import pyfits
import glob

mipsfiles=glob.glob('*cutout*.fits')

for i in range(1):
    fits=pyfits.open(mipsfiles[i])
    name=mipsfiles[i]
    im=fits[0].data.copy()
#    im[where(im<0)]=0
#    im[where(im>(.08))]=.08
    fits.close()
    axis('equal')
    imshow(-1.*(im),cmap='gray')#,interpolation='nearest',origin='upper',cmap='gray')#,vmin=myvmin,vmax=myvmax)
savefig('test.eps')
#    ax=gca()
#    ax.set_xticklabels(([]))
#    ax=gca()
#    ax.set_yticklabels(([]))
