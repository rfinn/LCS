#!/usr/bin/env python

#################

import argparse
parser=argparse.ArgumentParser()
#parser.add_argument("agcnumber",help='agcnumber')
parser.add_argument("-a",'--all',help="run for all clusters",action="store_true")
parser.add_argument("-c",'--cluster',help="cluster name (run one cluster only)",action="store")
parser.add_argument("-s",'--startindex', help="starting index (to continue from previous)",action="store")
parser.add_argument("-z",'--nozoo', help="plot spirals w/no zoo classification",action="store_true")
#parser.add_argument("-d",'--display', help="display result in ds9",action="store_true")
args = parser.parse_args()

#################

from astropy.io import fits
from LCScommon import *
import numpy as np
from pylab import *
import sys
from astropy.wcs import WCS
from wcsaxes import WCSAxes
mosaicpixelscale=.423
from scipy.stats import scoreatpercentile
from matplotlib import patches
from scipy import interpolate
ratio_r2ha=18.43



mips_sb_cut_mag=uJy2ABmag(mips_sb_cut*MJy_muJy_sqarc_conv)
sdss_sb_cut_mag=uJy2ABmag(sdss_sb_cut*nmgy_muJy_sqarc_conv)
def yaxiswcs(ax1,pscale,haflag=0):#when plotting lir, but on log scale
    x1, x2=ax1.get_xlim()
    ax2=twinx(ax1)
    y1, y2=ax1.get_ylim()
    t=ax1.get_xticklabels()
    #print t

    #print y1,y2
    if haflag:
        middle=(y1-y2)/2.
        ax2.set_ylim((y2-middle)*pscale,(y1-middle)*pscale)
    else:
        middle=(y2-y1)/2.
        ax2.set_ylim((y1-middle)*pscale,(y2-middle)*pscale)
    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position('left')
    #ax2.tick_bottoms()
    ax2.set_xticklabels(([]))
    
    
    ax3=twiny(ax1)
    if haflag:
        middle=(x1-x2)/2.
        ax3.set_xlim((x2-middle)*pscale,(x1-middle)*pscale)
    else:
        middle=abs(x2-x1)/2.
        ax3.set_xlim((x1-middle)*pscale,(x2-middle)*pscale)

    #ax3.set_xticklabels(fontsize=10)
    #ax2.yaxis.tick_left()
    ax3.xaxis.tick_bottom()
    ax3.xaxis.set_label_position('bottom')

    ax3.set_yticklabels(([]))


    ax1.set_xticklabels(([]))
    ax1.set_yticklabels(([]))
    ax1.tick_params(which='both',bottom='off',top='off',left='off',right='off')
    #print 'y limit in SFR =',getsfrfromlir(y1),getsfrfromlir(y2)

    

def plotintens(data_file,pls='-',pcolor='k',pmarker=None,pixelscale=2.45,ynorm=1,label=None):
    data1 = np.genfromtxt(data_file)
    sma_pix = data1[:,1]*pixelscale
    intens, intens_err = data1[:,2]*ynorm, data1[:,3]*ynorm
    mag=23.9-2.5*np.log10(intens)

    if label == None:
        plot(sma_pix,mag,ls=pls,color=pcolor,marker=pmarker)
    else:
        plot(sma_pix,mag,ls=pls,color=pcolor,label=label)
    yerru=-2.5*np.log10(intens+intens_err)+2.5*np.log10(intens)
    yerrl=-2.5*np.log10(intens) + 2.5*np.log10(intens+intens_err)
    yerrall=array(zip(yerrl,yerru))
    errorbar(sma_pix,mag,yerr=yerrall.T,fmt=None,ecolor=pcolor,label='_nolegend_')
    #axhline(y=0,ls='--',color='k')
    return sma_pix,intens

def plotcolor(mdata=None,rdata=None,pls='-',pcolor='k',mpscale=mipspixelscale,rpscale=sdsspixelscale,maxrad=40.,rynorm=23.14,mynorm=23.505):
    data1 = np.genfromtxt(mdata)
    msma = data1[:,1]*mpscale
    intens, intens_err = data1[:,2]*mynorm, data1[:,3]*mynorm
    flag=msma < maxrad
    tck=interpolate.splrep(msma[flag],intens[flag])

    data2 = np.genfromtxt(rdata)
    rsma = data2[:,1]*rpscale
    intens2, intens2_err = data2[:,2]*rynorm, data2[:,3]*rynorm

    flag=rsma < maxrad
    rtck=interpolate.splrep(rsma[flag],intens2[flag])

    dx=.1
    sma_new=arange(1,maxrad,dx)
    
    my=interpolate.splev(sma_new,tck,der=0)
    ry=interpolate.splev(sma_new,rtck,der=0)
    mmag=23.9-2.5*np.log10(my)
    rmag=23.9-2.5*np.log10(ry)
    #print sma_new
    #print mmag
    plot(sma_new,rmag-mmag,ls=pls,color=pcolor)
    #errorbar(sma_pix,intens,intens_err,fmt=None,ecolor=pcolor)
    return sma_new,rmag,mmag,my,ry

def plotfenclosed(data_file,pls='-',pcolor='k',pixelscale=2.45,maxrad=None):
    data1 = np.genfromtxt(data_file)
    sma_pix = data1[:,1]*pixelscale
    #tflux_e = data1[:,21]
    tflux_e = data1[:,22]
    if maxrad:
        t=(tflux_e[sma_pix < maxrad])

        try:
            norm=max(t)
        except ValueError:
            norm=max(tflux_e)
    else:
        norm=max(tflux_e)
    plot(sma_pix,tflux_e/norm*100.,ls=pls,color=pcolor)
    #errorbar(sma_pix,intens,intens_err,fmt=None,ecolor=pcolor)
    axhline(y=0,ls='--',color='k')


def plotisomag(data_file,pls='-',pcolor='k',pixelscale=2.45):
    data1 = np.genfromtxt(data_file)
    sma_pix = data1[:,1]*pixelscale
    mag = data1[:,18]
    magerr_l = data1[:,19]
    magerr_u = data1[:,19]
    magerr=array(zip(magerr_l,magerr_u),'f')
    
    plot(sma_pix,mag,ls=pls,color=pcolor)
    errorbar(sma_pix,mag,yerr=magerr.T,fmt=None,ecolor=pcolor)
    axhline(y=0,ls='--',color='k')


def plotimage(fits_image,vmin=0,vmax=4,haflag=0,dx=100,zoom=1,xc=None,yc=None):
    rfits=fits.open(fits_image)
    im=rfits[0].data.copy()
    if xc:
        xcenter = xc
        ycenter = yc
    else:
        yc,xc=im.shape
        xc=xc/2.
        yc=yc/2.
    x1=int(xc-dx)
    x2=int(xc+dx)
    y1=int(yc-dx)
    y2=int(yc+dx)
    #print x1,x2,y1,x2, dx
    rfits.close()
    ax=gca()
    axis('equal')
    try:
        v1,v2=scoreatpercentile(im[y1:y2,x1:x2],[5.,99.],limit=[0,15])
    except ValueError:
        try:
            v1,v2=scoreatpercentile(im[y1:y2,x1:x2],[5.,95.])#,limit=[0,15])
        except ValueError:
            v1=.01
            v2=2
    #print 'v1,v2 = ',v1,v2
    if fits_image.find('parent-r') > -1:
        v1=sdss_sb_cut*.5
    if fits_image.find('cutout24') > -1:
        v1=mips_sb_cut*.5
    if v2 < v1:
        v2=1
    if haflag:
        #imshow((im[int(.25*nx):int(.75*nx),int(.25*ny):int(.75*ny)]),interpolation='nearest',origin='upper',cmap='binary',vmin=vmin,vmax=vmax)
        if zoom:
            #print 'zooming'
            imfig=imshow((im[y1:y2,x1:x2].T),interpolation='nearest',origin='upper',cmap=cm.gray_r,vmin=vmin,vmax=vmax)
        else:
            imfig=imshow((im.T),interpolation='nearest',origin='upper',cmap=cm.gray_r,vmin=vmin,vmax=vmax)
        ax.invert_xaxis()
        
    else:
        if zoom:
            #print 'zooming'
            imfig=imshow((im[y1:y2,x1:x2]),interpolation='nearest',origin='lower',cmap=cm.gray_r,vmin=v1,vmax=v2)
        else:
            imfig=imshow((im),interpolation='nearest',origin='lower',cmap=cm.gray_r,vmin=v1,vmax=v2)
    #ax.set_yticklabels(([]))
    #ax.set_xticklabels(([]))
    return imfig
def plotimagewcs(fits_image,fig,location=[0,1,0,1],vmin=0,vmax=4,haflag=0):
    rfits=fits.open(fits_image)
    im=rfits[0].data.copy()
    wcs=WCS(rfits[0].header)
    rfits.close()
    
    ax=WCSAxes(fig,location,wcs=wcs)
    fig.add_axes(ax)
    axis('equal')
    if haflag:
        ny,nx=im.shape
        #imshow((im[int(.25*nx):int(.75*nx),int(.25*ny):int(.75*ny)]),interpolation='nearest',origin='upper',cmap='binary',vmin=vmin,vmax=vmax)
        ax.imshow((im),interpolation='nearest',cmap=cm.gray_r,vmin=vmin,vmax=vmax,origin='lower')
    else:
        ax.imshow((im),interpolation='nearest',cmap=cm.gray_r,vmin=vmin,vmax=vmax,origin='lower')
    #ax.set_yticklabels(([]))
    #ax.set_xticklabels(([]))

def putlabel(s):
    text(.08,.85,s,fontsize=16,transform=gca().transAxes,horizontalalignment='left')

def putsizelabel(s,color='k'):
    text(.95,.1,s,fontsize=12,transform=gca().transAxes,horizontalalignment='right',color=color)

def makeplots(i):
    # subplot dimensions
    nx=4
    ny=1

    figure(figsize=(12,3.5))
    subplots_adjust(left=.05,right=.95,wspace=.25,bottom=.2,top=.9)

    subplot(ny,nx,1)

    # plot r-band cutout image
    rfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.fits'
    plotimage(rfits,vmin=-.1,vmax=2.5)
    putlabel('$r-band $')
    if int(spirals.AGCNUMBER[i]) > 1.:
        slab='$'+str(spirals.NSAID[i])+' \ ('+str(int(spirals.AGCNUMBER[i]))+')$'
    else:
        slab='$'+str(spirals.NSAID[i])+'$'
    xlabel(slab,fontsize=20)
    subplot(ny,nx,2)
    # plot 24um cutout
    mipsfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.fits'
    plotimage(mipsfits,vmin=-.1,vmax=5)
    putlabel('$24 \mu m $')
    axis([.3,40,.1,10.])    
    subplot(ny,nx,3)
    # plot r and 24 profiles
    rdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.dat'
    crdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/c'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.dat'

    mipsdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.dat'
    mipse0=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.e05.dat'
    x,y=plotintens(rdat,pixelscale=sdsspixelscale,pcolor='k',label='r')
    #x,y=plotintens(crdat,pixelscale=sdsspixelscale,pcolor='.5',label='r:conv')
    mx,my=plotintens(mipsdat,pixelscale=mipspixelscale,pcolor='r',pmarker='o',pls=':',label='24')
    rscale=y[where(abs(x-6.) == min(abs(x-6.)))]/my[where(abs(mx-6.) == min(abs(mx-6.)))]
    #mx,my=plotintens(mipsdat,pixelscale=mipspixelscale,pcolor='m',ynorm=rscale,label='24:scaled')
    if os.path.isfile(mipse0):
        mx,my=plotintens(mipse0,pixelscale=mipspixelscale,pcolor='r',ynorm=1,label='24:e0')
    else:
        print 'no e=0 data file for ',spirals.NSAID[i]

    if spirals.CLUSTER[i] == 'MKW11':
        hadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/MKW11/'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        kprdat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/MKW11/'+str(int(spirals.AGCNUMBER[i]))+'_R.dat'
        chadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/MKW11/c'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        if os.path.isfile(hadat):
            x,y=plotintens(hadat,pixelscale=mosaicpixelscale,pcolor='b',label='Ha',ynorm=.02)

        if os.path.isfile(kprdat):
            print 'got kp R file'
            x,y=plotintens(kprdat,pixelscale=mosaicpixelscale,pcolor='y',label='R',ynorm=.02)
        if os.path.isfile(chadat):
            x,y=plotintens(chadat,pixelscale=mosaicpixelscale,pcolor='c',label='Ha:conv',ynorm=.02)
    #rscale=y[where(abs(x-spirals.SERSIC_TH50[i]) == min(abs(x-spirals.SERSIC_TH50[i])))]/my[where(abs(mx-spirals.SERSIC_TH50[i]) == min(abs(mx-spirals.SERSIC_TH50[i])))]
    gca().set_yscale('log')
    #gca().set_xscale('log')
    #axis([.3,100,.001,1000])
    #axis([.3,40,.001,100])
    axvline(x=spirals.cre1[i]*mipspixelscale,ls=':',color='r')
    axvline(x=spirals.SERSIC_TH50[i],ls=':',color='b')
    xlabel('$ sma \ (arcsec) $',fontsize=20)
    putlabel('$Intensity $')
    # 21 total flux enclosed by ellipse
    subplot(ny,nx,4)

    plotfenclosed(rdat,pixelscale=sdsspixelscale,pcolor='k')
    #plotfenclosed(crdat,pixelscale=sdsspixelscale,pcolor='c')
    plotfenclosed(mipsdat,pixelscale=mipspixelscale,pcolor='r')

    if spirals.CLUSTER[i] == 'MKW11':
        hadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/MKW11/'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        kprdat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/MKW11/'+str(int(spirals.AGCNUMBER[i]))+'_R.dat'
        chadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/MKW11/c'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        if os.path.isfile(hadat):
            plotfenclosed(hadat,pixelscale=mosaicpixelscale,pcolor='b')
        if os.path.isfile(kprdat):
            print 'got kp R file'
            x,y=plotintens(kprdat,pixelscale=mosaicpixelscale,pcolor='y',label='R',ynorm=.02)

        if os.path.isfile(chadat):
            plotfenclosed(chadat,pixelscale=mosaicpixelscale,pcolor='c')

    ax=gca()
    axis([.3,100,5.,120.])
    axis([.3,40,5.,120.])
    ax.set_yscale('log')
    #ax.set_xscale('log')

    #plotisomag(rdat,pixelscale=sdsspixelscale,pcolor='b')
    #plotisomag(mipsdat,pixelscale=mipspixelscale,pcolor='r')

    #axis([.3,100,14,26.])
    #ax=gca()
    #ax.invert_yaxis()
    #ax.set_xscale('log')

    axvline(x=spirals.cre1[i]*mipspixelscale,ls=':',color='r')
    axvline(x=spirals.SERSIC_TH50[i],ls=':',color='b')
    axvline(x=spirals.PETROTH90[i],ls=':',color='k')
    axvline(x=5.9,ls='--',color='k')
    axhline(y=50,ls=':',color='k')
    axhline(y=70,ls=':',color='k')
    xlabel('$ sma \ (arcsec) $',fontsize=20)
    putlabel('$Flux(<r) $')

    outfile=homedir+'research/LocalClusters/EllipseProfiles/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-ellipse-profiles.png'
    savefig(outfile)
    close()

def makeplotsha(i,dr=100,zoom=0,plotellipse=1,showsize=1,showe0=1,plot24=1,plothaconv=1):
    # subplot dimensions
    nx=3
    ny=2
    fig=figure(figsize=(9,6))
    subplots_adjust(left=.1,right=.95,wspace=.2,hspace=.3,bottom=.2,top=.9)



    # plot r-band cutout image

    #rfits=self.prefix+'-'+str(self.n.NSAID[i])+'-parent-r.fits'
    #rfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.fits'
    rfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-parent-r.fits'
    #plotimagewcs(rfits,fig,location=[.1,.6,.3,1],vmin=-.1,vmax=2.)
    w= WCS(rfits)
    px,py = w.wcs_world2pix(spirals.RA[i],spirals.DEC[i],1)
    #print 'rband x,y = ',px,py
    subplot(ny,nx,1)
    if plotellipse:
        ell=patches.Ellipse((dr/sdsspixelscale,dr/sdsspixelscale),width=2.*spirals.SERSIC_BA[i]*isorad.NSA[i]/sdsspixelscale,height=2.*isorad.NSA[i]/sdsspixelscale,angle=spirals.SERSIC_PHI[i],color='r',fill=False,linewidth=1,alpha=.7)
        gca().add_patch(ell)
    imfig=plotimage(rfits,vmin=-.1,vmax=2.,dx=dr/sdsspixelscale,zoom=zoom,xc=px,yc=py)
    
    putlabel('$r-band $')
    if showsize:
        s='$r = %4.1f$'%(isorad.NSA[i])
        putsizelabel(s)
    #x1,x2=xlim()
    #axis([int(.25*x1),int(.75*x2),int(.25*x1),int(.75*x2)])
    # plot 24um cutout

    ax=gca()
    yl='$\Delta Dec \ (arcsec) $'
    text(-.3,.5,yl,transform=ax.transAxes,verticalalignment='center',rotation=90,fontsize=15)
    yl='$\Delta RA \ (arcsec) $'
    text(.5,-.25,yl,transform=ax.transAxes,horizontalalignment='center',fontsize=15)

    text(.5,1.05,'$'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'$',transform=ax.transAxes,horizontalalignment='center',fontsize=20)

    yaxiswcs(ax,sdsspixelscale)


    subplot(ny,nx,2)



    mipsfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.fits'
    #plotimagewcs(mipsfits,fig,location=[.35,.6,.65,1],vmin=-.1,vmax=1.5)
    w= WCS(mipsfits)
    px,py = w.wcs_world2pix(spirals.RA[i],spirals.DEC[i],1)
    #print 'mips x,y = ',px,py

    if plotellipse:
        cir=Circle((dr/mipspixelscale,dr/mipspixelscale),radius=isorad.MIPSE0[i]/mipspixelscale,color='none',ec='r')
        gca().add_patch(cir)

        ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*spirals.SERSIC_BA[i]*isorad.MIPS[i]/mipspixelscale,height=2.*isorad.MIPS[i]/mipspixelscale,angle=spirals.SERSIC_PHI[i],color='r',fill=False,linewidth=1,alpha=.7)
        gca().add_patch(ell)

    imfig=plotimage(mipsfits,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=zoom,xc=px,yc=py)
    putlabel('$24 \mu m $')
    if showsize:
        s='$r = %4.1f, %4.1f$'%(isorad.MIPS[i],isorad.MIPSE0[i])
        putsizelabel(s)

    #title('$'+str(spirals.NSAID[i])+' \ ('+str(int(spirals.AGCNUMBER[i]))+')$',fontsize=20)
    #imfig.set_extent([-1.*dr,-1.*dr,dr,dr])

    ax=gca()

    yl='$\Delta Dec \ (arcsec) $'
    #text(-.3,.5,yl,transform=ax.transAxes,verticalalignment='center',rotation=90,fontsize=15)
    yl='$\Delta RA \ (arcsec) $'
    text(.5,-.25,yl,transform=ax.transAxes,horizontalalignment='center',fontsize=15)



    yaxiswcs(ax,mipspixelscale)

    subplot(ny,nx,3)
    # plot Halphaa cutout


    hafits=homedir+'research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/'+str(int(spirals.AGCNUMBER[i]))+'_Ha.fits'
    w= WCS(hafits)
    px,py = w.wcs_world2pix(spirals.RA[i],spirals.DEC[i],1)
    print 'Ha x,y = ',px,py

    if plotellipse:
        ell=patches.Ellipse((dr/mosaicpixelscale,dr/mosaicpixelscale),width=2.*spirals.SERSIC_BA[i]*isorad.HA[i]/mosaicpixelscale,height=2.*isorad.HA[i]/mosaicpixelscale,angle=spirals.SERSIC_PHI[i],color='r',fill=False,linewidth=1,alpha=.7)
        gca().add_patch(ell)

    #plotimagewcs(hafits,fig,location=[.7,.6,1,1],vmin=-5,vmax=50,haflag=1)
    imfig=plotimage(hafits,vmin=-5,vmax=50,haflag=1,dx=dr/mosaicpixelscale,zoom=zoom,xc=px,yc=py)
    putlabel(r'$H \alpha $')
    s='$r = %4.1f$'%(isorad.HA[i])
    #putsizelabel(s)

    ax=gca()
    yaxiswcs(ax,mosaicpixelscale,haflag=1)
    
    yl='$\Delta Dec \ (arcsec) $'
    #text(-.3,.5,yl,transform=ax.transAxes,verticalalignment='center',rotation=90,fontsize=15)
    yl='$\Delta RA \ (arcsec) $'
    text(.5,-.25,yl,transform=ax.transAxes,horizontalalignment='center',fontsize=15)

    #text(.5,1.05,slab,transform=ax.transAxes,horizontalalignment='center',fontsize=20)



    subplot(ny,nx,5)
    # plot r and 24 profiles
    rdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.dat'
    crdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/c'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.dat'

    mipsdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.dat'
    mipse0=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.e05.dat'

    x,y=plotintens(rdat,pixelscale=sdsspixelscale,pcolor='k',label='$r$',ynorm=nmgy_muJy_sqarc_conv)
    #x,y=plotintens(crdat,pixelscale=sdsspixelscale,pcolor='.5',label='r:conv')
    if plot24:
        mx,my=plotintens(mipsdat,pixelscale=mipspixelscale,pcolor='r',pmarker='o',pls='-',label='$24$',ynorm=MJy_muJy_sqarc_conv)

    #rscale=y[where(abs(x-6.) == min(abs(x-6.)))]/my[where(abs(mx-6.) == min(abs(mx-6.)))]
    #mx,my=plotintens(mipsdat,pixelscale=mipspixelscale,pcolor='m',ynorm=rscale,label='24:scaled')
    if os.path.isfile(mipse0) & showe0:
        mx,my=plotintens(mipse0,pixelscale=mipspixelscale,pcolor='r',label='24:e0',ynorm=MJy_muJy_sqarc_conv)
    else:
        print 'no e=0 data file for ',spirals.NSAID[i]

    if (spirals.CLUSTER[i] == 'MKW11') or (spirals.CLUSTER[i] == 'MKW8'):
        hadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        kprdat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/'+str(int(spirals.AGCNUMBER[i]))+'_R.dat'
        chadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/c'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        if os.path.isfile(hadat):
            x,y=plotintens(hadat,pixelscale=mosaicpixelscale,pcolor='b',label='Ha',ynorm=.02*nmgy_muJy_sqarc_conv)
        #if os.path.isfile(kprdat):
        #    x,y=plotintens(kprdat,pixelscale=mosaicpixelscale,pcolor='g',label='kpR',ynorm=.0015*nmgy_muJy_sqarc_conv)
        
        if os.path.isfile(chadat) and plothaconv:
            x,y=plotintens(chadat,pixelscale=mosaicpixelscale,pcolor='c',label='Ha:conv',ynorm=.0015*ratio_r2ha*nmgy_muJy_sqarc_conv)

    labelintensity(dr,i)

    labelintensity(35,i)

    xlabel('$ sma \ (arcsec) $',fontsize=20)
    legend(loc='center left',prop={'size':10},bbox_to_anchor=(1,0.5))


    # 21 total flux enclosed by ellipse
    #subplot(ny,nx,6)
    #if isorad.NSA[i] > 0.:
    #    rmax=isorad.NSA[i]
    #else:
    #    rmax=40.
    
    #plotfenclosed(rdat,pixelscale=sdsspixelscale,pcolor='k',maxrad=rmax)
    ##plotfenclosed(crdat,pixelscale=sdsspixelscale,pcolor='c')
    #plotfenclosed(mipsdat,pixelscale=mipspixelscale,pcolor='r',maxrad=rmax)

    #if (spirals.CLUSTER[i] == 'MKW11') or (spirals.CLUSTER[i] ==  'MKW8'):
    #    hadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
    #    chadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/c'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
    #    if os.path.isfile(hadat):
    #        plotfenclosed(hadat,pixelscale=mosaicpixelscale,pcolor='b',maxrad=rmax)
    #    if os.path.isfile(chadat):
    #        plotfenclosed(chadat,pixelscale=mosaicpixelscale,pcolor='c',maxrad=rmax) 

    #ax=gca()
    #axis([.3,100,5.,120.])
    #axis([.3,40,5.,120.])
    #ax.set_yscale('log')

    #axvline(x=spirals.cre1[i]*mipspixelscale,ls=':',color='r')
    #axvline(x=spirals.SERSIC_TH50[i],ls=':',color='b')
    #axvline(x=spirals.PETROTH90[i],ls=':',color='k')
    #axvline(x=5.9,ls='--',color='k')
    #axhline(y=50,ls=':',color='k')
    #axhline(y=70,ls=':',color='k')
    #xlabel('$ sma \ (arcsec) $',fontsize=20)
    #putlabel('$Flux(<r) $')

    outfile=homedir+'research/LocalClusters/EllipseProfiles/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-ellipse-profiles-Ha.png'
    savefig(outfile)
    outfile=homedir+'research/LocalClusters/EllipseProfiles/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-ellipse-profiles-Ha.eps'
    savefig(outfile)
    close()


def makeplotsv2(i,dr=40,zoom=1,plotellipse=1,plote0=1):
    # subplot dimensions
    nx=4
    ny=1
    figure(figsize=(12,3.))
    #figure(figsize=(7,6.))
    #subplots_adjust(left=.075,right=.95,wspace=.3,hspace=.25,bottom=.2,top=.9)
    subplots_adjust(left=.07,right=.97,wspace=.38,hspace=.25,bottom=.2,top=.85)

    rfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-parent-r.fits'
    mipsfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.fits'
    hdu=fits.open(rfits)
    hdat=hdu[0].data.copy()
    n2,n1=hdat.shape
    hdu.close()

    hdu=fits.open(mipsfits)
    hdat=hdu[0].data.copy()
    n2mips,n1mips=hdat.shape
    hdu.close()

    #print 'n1, n2 = ',n1,n2
    mindim=min([n1*sdsspixelscale,n2*sdsspixelscale,n1mips*mipspixelscale,n2mips*mipspixelscale])/2.
    if dr > (mindim-2):
        dr=(mindim-2)
    if (dr < isorad.NSA[i]):
        if (isorad.NSA[i] < (mindim-5)):
            dr=isorad.NSA[i]+2
        else:
            dr=(mindim-5)
                                                    
    #plotimagewcs(rfits,fig,location=[.1,.6,.3,1],vmin=-.1,vmax=2.)
    w= WCS(rfits)
    px,py = w.wcs_world2pix(spirals.RA[i],spirals.DEC[i],1)
    #print 'rband x,y = ',px,py
    subplot(ny,nx,1)
    if plotellipse:
        ell=patches.Ellipse((dr/sdsspixelscale,dr/sdsspixelscale),width=2.*spirals.SERSIC_BA[i]*isorad.NSA[i]/sdsspixelscale,height=2.*isorad.NSA[i]/sdsspixelscale,angle=spirals.SERSIC_PHI[i],color='r',fill=False,linewidth=1,alpha=.7)
        gca().add_patch(ell)
    imfig=plotimage(rfits,vmin=-.1,vmax=2.,dx=dr/sdsspixelscale,zoom=zoom,xc=px,yc=py)
    putlabel('$r-band $')
    s='$r = %4.1f$'%(isorad.NSA[i])
    putsizelabel(s)
    if spirals.AGNKAUFF[i] > .1:
        ax=gca()
        text(.1,.1,'$AGN$',fontsize=12,transform=gca().transAxes,horizontalalignment='left')




    ax=gca()
    yl='$\Delta Dec \ (arcsec) $'
    text(-.3,.5,yl,transform=ax.transAxes,verticalalignment='center',rotation=90,fontsize=15)
    yl='$\Delta RA \ (arcsec) $'
    text(.5,-.25,yl,transform=ax.transAxes,horizontalalignment='center',fontsize=15)

    text(.5,1.05,'$'+str(spirals.CLUSTER[i])+'$',transform=ax.transAxes,horizontalalignment='center',fontsize=20)

    yaxiswcs(ax,sdsspixelscale)

    subplot(ny,nx,2)


    #plotimagewcs(mipsfits,fig,location=[.35,.6,.65,1],vmin=-.1,vmax=1.5)
    w= WCS(mipsfits)
    px,py = w.wcs_world2pix(spirals.RA[i],spirals.DEC[i],1)
    if plotellipse:
        if plote0:
            cir=Circle((dr/mipspixelscale,dr/mipspixelscale),radius=isorad.MIPSE0[i]/mipspixelscale,color='none',ec='r')
            gca().add_patch(cir)

        ellip=1-spirals.SERSIC_BA[i]
        if ellip > .6:
            ellip=.6
        ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*(1-ellip)*isorad.MIPS[i]/mipspixelscale,height=2.*isorad.MIPS[i]/mipspixelscale,angle=spirals.SERSIC_PHI[i],color='r',fill=False,linewidth=1,alpha=.7)
        gca().add_patch(ell)

    imfig=plotimage(mipsfits,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=zoom,xc=px,yc=py)
    putlabel('$24 \mu m $')
    s='$r = %4.1f, %4.1f$'%(isorad.MIPS[i],isorad.MIPSE0[i])
    if 1.2*isorad.MIPS[i] < isorad.NSA[i]:
        putsizelabel(s,color='r')
    else:
        putsizelabel(s)
    
    ax=gca()
    s='$p_{cs} = %3.2f $'%(spirals.p_cs[i])
    text(.1,.1,s,fontsize=12,transform=gca().transAxes,horizontalalignment='left')


    if int(spirals.AGCNUMBER[i]) > 1.:
        slab='$'+str(spirals.NSAID[i])+' \ ('+str(int(spirals.AGCNUMBER[i]))+')$'
    else:
        slab='$'+str(spirals.NSAID[i])+'$'


    ax=gca()

    yl='$\Delta Dec \ (arcsec) $'
    text(-.3,.5,yl,transform=ax.transAxes,verticalalignment='center',rotation=90,fontsize=15)
    yl='$\Delta RA \ (arcsec) $'
    text(.5,-.25,yl,transform=ax.transAxes,horizontalalignment='center',fontsize=15)

    text(.5,1.05,slab,transform=ax.transAxes,horizontalalignment='center',fontsize=20)

    yaxiswcs(ax,mipspixelscale)

    subplot(ny,nx,3)
    # plot r and 24 profiles
    rdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.dat'
    mipsdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.dat'
    if not(os.path.isfile(mipsdat)):
        print 'no mips ellipse file for ',spirals.NSAID[i]
        return
    mipse0=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.e05.dat'

    x,y=plotintens(rdat,pixelscale=sdsspixelscale,pcolor='k',label='$r$',ynorm=nmgy_muJy_sqarc_conv)
    #x,y=plotintens(crdat,pixelscale=sdsspixelscale,pcolor='.5',label='r:conv')
    mx,my=plotintens(mipsdat,pixelscale=mipspixelscale,pcolor='r',pmarker='o',pls='-',label='$24$',ynorm=MJy_muJy_sqarc_conv)
    #rscale=y[where(abs(x-6.) == min(abs(x-6.)))]/my[where(abs(mx-6.) == min(abs(mx-6.)))]
    #mx,my=plotintens(mipsdat,pixelscale=mipspixelscale,pcolor='m',ynorm=rscale,label='24:scaled')
    if os.path.isfile(mipse0):
        if plote0:
            mx,my=plotintens(mipse0,pixelscale=mipspixelscale,pcolor='r',pls=':',label='$24:e0$',ynorm=MJy_muJy_sqarc_conv)
    else:
        print 'no e=0 data file for ',spirals.NSAID[i]

    #rscale=y[where(abs(x-spirals.SERSIC_TH50[i]) == min(abs(x-spirals.SERSIC_TH50[i])))]/my[where(abs(mx-spirals.SERSIC_TH50[i]) == min(abs(mx-spirals.SERSIC_TH50[i])))]
    #gca().set_yscale('log')
    #gca().set_xscale('log')
    #axis([.3,100,.001,100])
    #axis([.3,40,.01,500])
    #axis([0,dr,-1.6,3.1])
    labelintensity(dr,i)
    
    # 21 total flux enclosed by ellipse
    subplot(ny,nx,4)


    t=plotcolor(mdata=mipsdat,rdata=rdat,pls='-',pcolor='k',mpscale=mipspixelscale,rpscale=sdsspixelscale,maxrad=40.,rynorm=nmgy_muJy_sqarc_conv,mynorm=MJy_muJy_sqarc_conv)
    plotcolor(mdata=mipse0,rdata=rdat,pls=':',pcolor='k',mpscale=mipspixelscale,rpscale=sdsspixelscale,maxrad=40.,rynorm=nmgy_muJy_sqarc_conv,mynorm=MJy_muJy_sqarc_conv)


    axvline(x=isorad.MIPS[i],ls=':',color='r')
    axvline(x=isorad.NSA[i],ls=':',color='k')


    ax=gca()
    #axis([.3,100,5.,120.])
    axis([0,dr,-2,5])
    #ax.set_yscale('log')
    #ax.set_xscale('log')

    #plotisomag(rdat,pixelscale=sdsspixelscale,pcolor='b')
    #plotisomag(mipsdat,pixelscale=mipspixelscale,pcolor='r')

    #axis([.3,100,14,26.])
    #ax=gca()
    #ax.invert_yaxis()
    #ax.set_xscale('log')

    #axvline(x=spirals.cre1[i]*mipspixelscale,ls=':',color='r')
    #axvline(x=spirals.SERSIC_TH50[i],ls=':',color='b')
    #axvline(x=spirals.PETROTH90[i],ls=':',color='k')
    #axvline(x=5.9,ls='--',color='k')
    #axhline(y=50,ls=':',color='k')
    #axhline(y=70,ls=':',color='k')
    xlabel('$ sma \ (arcsec) $',fontsize=15)
    #ylabel('$\mu \ (mag_{AB}/arcsec^2) $',fontsize=15)
    ylabel('$\mu_r - \mu_{24} $',fontsize=15)
    putlabel('$Color$')

    outfile=homedir+'research/LocalClusters/EllipseProfiles/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-ellipse-profiles.png'
    savefig(outfile)
    outfile=homedir+'research/LocalClusters/EllipseProfiles/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-ellipse-profiles.eps'
    savefig(outfile)
    close()
    return t

def labelintensity(dr,i):
    axis([0,dr,16.15,26.1])
    ax=gca()
    ax.invert_yaxis()
    axvline(x=isorad.MIPS[i],ls='-',color='r')
    y1,y2=ylim()
    yfill=arange(y1,y2,.1)
    fill_betweenx(yfill,isorad.MIPS[i]-.2*isorad.MIPS[i],isorad.MIPS[i]+.2*isorad.MIPS[i],color='r',alpha=0.2)
    axvline(x=isorad.MIPSE0[i],ls=':',color='r')
    axvline(x=isorad.NSA[i],ls='-',color='k')
    axhline(y=mips_sb_cut_mag,ls='--',color='r')
    axhline(y=sdss_sb_cut_mag,ls='--',color='k')
    xlabel('$ sma \ (arcsec) $',fontsize=15)
    #ylabel('$ log_{10}(I \ (\mu Jy/arcsec^2)) $',fontsize=15)
    #yl='$ log_{10}(I \ (\mu Jy/arcsec^2)) $'
    yl='$ \mu_{AB} \ (mag/arcsec^2) $'
    text(-.28,.5,yl,transform=ax.transAxes,verticalalignment='center',rotation=90,fontsize=15)
    legend(loc='upper right',prop={'size':10})#,bbox_to_anchor=(1,0.5))
    putlabel('$Intensity$' )


def runall(cluster):
    keepflag = (spirals.CLUSTER == cluster) # & (spirals.SERSIC_TH50 > mipspixelscale)
    index=arange(len(spirals.RA))
    keepindex=index[keepflag]
    for i in keepindex:
        print spirals.NSAID[i]
        makeplotsv2(i,dr=40,zoom=1)

def runone(nsaid,zoom=1,dr=40,plotellipse=1):
    print spirals.CLUSTER[nsadict[nsaid]],nsaid
    t=makeplotsv2(nsadict[nsaid],zoom=zoom,dr=dr,plotellipse=plotellipse)
    #return t

# set up parameters
infile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all.fits'
# read in LCSspirals catalog
hdr=fits.open(infile)
spirals=hdr[1].data
hdr.close()
infile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_isorad.fits'
# read in LCSspirals catalog
hdr=fits.open(infile)
isorad=hdr[1].data
hdr.close()
nsadict=dict((a,b) for a,b in zip(spirals.NSAID,arange(len(spirals.NSAID))))
agcdict=dict((a,b) for a,b in zip(spirals.AGCNUMBER,arange(len(spirals.NSAID))))

infile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_isorad.fits'
# read in LCSspirals catalog
hdr=fits.open(infile)
isorad=hdr[1].data
hdr.close()

# check these at some point...
problems=[146015,103957,104125,142656,142676,142819,162780]

if args.all: 
        index=arange(len(spirals.RA))
        if args.startindex:
            startindex=int(args.startindex)
        else:
            startindex=0
        
        for i in range(startindex,len(spirals.RA)):
            # keep clean sample only
            if spirals.p_cs[i] > 0.6:
                print i,spirals.NSAID[i]
                if spirals.NSAID[i] in problems:
                    print 'skipping ',spirals.CLUSTER[i], spirals.NSAID[i]
                else:
                    makeplotsv2(i,dr=40,zoom=1)

if args.nozoo: 
    for c in clusternames:
        for id in spiral_100_nozoo[c]:
            i=nsadict[id]
            if spirals.NSAID[i] in problems:
                print 'skipping ',spirals.CLUSTER[i], spirals.NSAID[i]
            else:
                makeplotsv2(i,dr=40,zoom=1)


if args.cluster:
    runall(args.cluster)
#makeplots(nsadict[78100])
#runall('AWM4')
#runall('A2063')

