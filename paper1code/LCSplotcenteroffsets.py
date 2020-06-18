#!/usr/bin/env python
from pylab import *

infile=open('/home/rfinn/junk')
x=[]
y=[]
for line in infile:
    if line.find('1') < 0:
        continue
    t=line.split()
    x.append(float(t[1]))
    y.append(float(t[2]))


infile.close()
x=array(x,'f')
y=array(y,'f')
d=sqrt((x-51)**2+(y-51)**2)
figure()
hist(d,20,cumulative='True',normed='True',histtype='step')
axhline(.9,color='r',ls=':')
axhline(.95,color='r',ls='--')
xlabel('distance (pixels/arcsec)')
title('NGC6107:  SDSS Image Offset from Central Pixel')
