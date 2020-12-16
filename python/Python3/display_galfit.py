#!/usr/bin/env python

import os
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.visualization import simple_norm
from astropy import units as u
from astropy.wcs import WCS
from matplotlib import pyplot as plt

homedir = os.getenv("HOME")
from scipy.stats import scoreatpercentile
os.sys.path.append(homedir+'/github/virgowise/')
import rungalfit as rg #This code has all the defined functions that I can use

def display_galfit_model(image,percentile1=.5,percentile2=99.5,p1residual=5,p2residual=99,cmap='viridis',zoom=None):
    
    '''
    ARGS:
    percentile1 = min percentile for stretch of image and model
    percentile2 = max percentile for stretch of image and model
    p1residual = min percentile for stretch of residual
    p2residual = max percentile for stretch of residual
    cmap = colormap, default is viridis
    '''
    # model name
    filename = image
    pngname = image.split('.fits')[0]+'.png'
    image,h = fits.getdata(filename,1,header=True)
    model = fits.getdata(filename,2)
    residual = fits.getdata(filename,3)

    if zoom is not None:
       print("who's zoomin' who?")
       # display central region of image
       # figure out how to zoom

       # get image dimensions and center
       xmax,ymax = image.shape
       xcenter = int(xmax/2)
       ycenter = int(ymax/2)

       # calculate new size to display based on zoom factor
       new_xradius = int(xmax/2/(float(zoom)))
       new_yradius = int(ymax/2/(float(zoom)))

       # calculate pixels to keep based on zoom factor
       x1 = xcenter - new_xradius
       x2 = xcenter + new_xradius
       y1 = ycenter - new_yradius
       y2 = ycenter + new_yradius
       
       # check to make sure limits are not outsize image dimensions
       if (x1 < 1):
          x1 = 1
       if (y1 < 1):
          y1 = 1
       if (x2 > xmax):
          x2 = xmax
       if (y2 > ymax):
          y2 = ymax

       # cut images to new size
       image = image[x1:x2,y1:y2]
       model = model[x1:x2,y1:y2]
       residual = residual[x1:x2,y1:y2]         
       pass
    wcs = WCS(h)
    images = [image,model,residual]
    titles = ['image','model','residual']
    v1 = [scoreatpercentile(image,percentile1),
          scoreatpercentile(image,percentile1),
          scoreatpercentile(residual,p1residual)]
    v2 = [scoreatpercentile(image,percentile2),
          scoreatpercentile(image,percentile2),
          scoreatpercentile(residual,p2residual)]
    norms = [simple_norm(image,'asinh',max_percent=percentile2),
             simple_norm(image,'asinh',max_percent=percentile2),
             simple_norm(residual,'linear',max_percent=p2residual)]
             
    plt.figure(figsize=(14,6))
    plt.subplots_adjust(wspace=.0)
    for i,im in enumerate(images): 
       plt.subplot(1,3,i+1,projection=wcs)
       plt.imshow(im,origin='lower',cmap=cmap,vmin=v1[i],vmax=v2[i],norm=norms[i])
       plt.xlabel('RA')
       if i == 0:
          plt.ylabel('DEC')
       else:
          ax = plt.gca()
          ax.set_yticks([])
       plt.title(titles[i],fontsize=16)
    plt.savefig(pngname)

def print_galfit_model(image):
    t = rg.parse_galfit_1comp(image,printflag=True)
    #print(t)
