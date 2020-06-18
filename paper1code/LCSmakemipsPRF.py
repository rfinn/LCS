#!/usr/bin/env python


from LCScommon import *
from pyraf import iraf
import ds9
print clusternames
d=ds9.ds9()
reviewflag=1
dxstar=15
byhand=1
#iraf()
clusternames=['A2052']
for prefix in clusternames:

    infile='/Users/rfinn/research/LocalClusters/PRF/'+prefix+'-starlist.tbl'
    mosaic='/Users/rfinn/research/LocalClusters/Images/'+prefix+'/24umWCS/'+prefix+'-WCS-mosaic_minus_median_extract.fits'
    psf='/Users/rfinn/research/LocalClusters/PRF/'+prefix+'/'+prefix+'psf_star.fits'
    if byhand:
        d.set('frame delete all')
        s='file new '+mosaic
        d.set(s)
        print 'Find an appropriate star'
        print 'Use Region -> centroid to find the x,y coord of centroid'
        try:
            xstar=float(raw_input('enter the x coord of the star \n'))
        except:
            print "sorry, didn't get that"
            xstar=float(raw_input('enter the x coord of the star \n'))

        try:
            ystar=float(raw_input('enter the y coord of the star \n'))
        except:
            print "sorry, didn't get that"
            ystar=float(raw_input('enter the y coord of the star \n'))

        psf='/Users/rfinn/research/LocalClusters/PRF/'+prefix+'/'+prefix+'psf_star_byhand.fits'
    else:
        in1=open(infile,'r')
        ngal=0
        maxflux=0
        for line in in1:
            if line.startswith('\\'):
                continue
            elif line.startswith('|'):
                continue
            else:
                t=line.split()
                try:
                    k=float(t[18])
                except IndexError:
                    print line
                    print 'problem with a line in ',infile
                    continue
                if  (float(t[18]) > maxflux):
                    maxline=line
                    maxflux=float(t[18])

        print prefix, ngal 
        in1.close()
        t=maxline.split()
        xstar=float(t[8])
        ystar=float(t[10])

    if os.path.exists(psf):
        os.remove(psf)
    inimage=mosaic+'[%i:%i,%i:%i]'%(int(round(xstar-dxstar)),int(round(xstar+dxstar-1)),int(round(ystar-dxstar)),int(round(ystar+dxstar-1)))
    iraf.imcopy(inimage,psf)

    if reviewflag:
        
        d.set('frame delete all')
        s='file new '+psf

        try:
            d.set(s)
            d.set('zoom to fit')
        except:
            print "couldn't access: ",s
        t=raw_input('hit any key to continue  \n')
        
