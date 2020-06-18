#!/usr/bin/env python

from pylab import *
import os
#import ReadAGCsav
from LCScommon import *
import atpy
from matplotlib.backends.backend_pdf import PdfFile
delta=100.#width of cutouts in arcsec
ramin=170.
ramax=250.
decmax=38.
decmin=0.
zmin=0.01366#min z cut, z(coma)-3 sigma
zmax=0.04333#max z cut, z(A2052)+3 sigma
vmin=zmin*3.e5
vmax=zmax*3.e5

agc=atpy.Table(homedir+'idl/programs/idl_alfa/agcnorthLCS.fits')

def drawbox(data,style):#feed in center x,y,dx,dy,rotation E of N
    #xcoords of unrotated box, going around CCW
    xl=array([data[0]-0.5*data[2],data[0]+0.5*data[2],data[0]+0.5*data[2],data[0]-0.5*data[2],data[0]-0.5*data[2]],'d')
    yl=array([data[1]-0.5*data[3],data[1]-0.5*data[3],data[1]+0.5*data[3],data[1]+0.5*data[3],data[1]-0.5*data[3] ],'d')

    xl=array([-0.5*data[2],+0.5*data[2],+0.5*data[2],-0.5*data[2],-0.5*data[2]],'d')
    yl=array([-0.5*data[3],-0.5*data[3],+0.5*data[3],+0.5*data[3],-0.5*data[3] ],'d')

    ang=data[4]*pi/180.*-1.#convert rotation to radians
    #rotate coordinates
    xp=cos(ang)*xl-sin(ang)*yl
    yp=sin(ang)*xl+cos(ang)*yl

    #put back on absolute scale
    xp=data[0]+xp
    yp=data[1]+yp
    #draw rotated box
    plot(xp,yp,style)


def plotagcLCS():#agcra,agcdec):
    figure(figsize=(15,7.5))
    clf()


    #plot(agc.radeg[velflag],agc.decdeg[velflag],'k.')
    dx=2.
    dy=dx
    ramin
    x=arange(ramin,(ramax+dx),dx)
    y=arange(decmin,(decmax+dx),dy)

    z=zeros([len(y),len(x)],'f')
    ylength=len(y)
    xlength=len(x)
    #for i in range(len(agcra)):
#	    xbin=int(round((agcra[i]-x[0])/dx))
#	    ybin=int(round((agcdec[i]-y[0])/dy))
#	    if (xbin > -1) & (xbin < xlength):
#		if (ybin > -1) & (ybin < ylength):
#		    z[ybin,xbin] += 1		  
    ncon=15
    #contourf(x,y,z,ncon,alpha=.8)#,cmap=cm.Greys)
    hexbin(agc.RA,agc.DEC,gridsize=100,bins='log',cmap=cm.gray_r)
    cra=[]
    cdec=[]
    for i in range(len(clusterz)):
        s=' '+clusternames[i]
        text(clusterRA[clusternames[i]],clusterDec[clusternames[i]],s,fontsize=20,color='r')
        cra.append(clusterRA[clusternames[i]])
        cdec.append(clusterDec[clusternames[i]])
        drawbox(cluster24Box[clusternames[i]],'r-')

    plot(cra,cdec,'ro',markersize=4,color='r')


    # mark area covered by ALFLAFA a40 data release, spring region
    # 7h30m < RA < 16h30m
    # 04 < dec < 16
    # 24 < dec < 28
    alfara=array([7.5,7.5,16.5,16.5,7.5],'f')*15
    alfadec1=array([4,16,16,4,4],'f')
    alfadec2=array([24,28,28,24,24],'f')
    plot(alfara,alfadec1,'c-',lw=2)
    plot(alfara,alfadec2,'c-',lw=2)
    axis([173,248,0,38.])
    
    xlabel('$ RA \ (deg) $',fontsize=24)
    ylabel('$ Dec \ (deg) $',fontsize=24)
    savefig(homedir+'research/LocalClusters/SamplePlots/LCSagc.eps')

## run these two lines in ipython, then run this script    
# agc=ReadAGCsav.agc()#only need this line if running match2agc
# agc.cull()
##
# plotagcLCS(agc.radeg,agc.decdeg)
plotagcLCS()
