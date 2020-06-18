#!/usr/bin/env python

'''

GOAL:
- make a figure showing color, r, and 24 cutout for truncated galaxies
- repeat for non-truncated galaxies

USAGE:

In [12]: %run ~/svnRepository/pythonCode/LCSmaketileplot.py

In [13]: s.plotcolorsort()


# to plot all galaxies ordered by NSAID
s.plotcolor_all(plotellipse=0, showall=False,  orderbynsaid=True, blueonly=True)

# to plot all galaxies ordered by Re(24)/Re(r)
s.plotcolor_all(plotellipse=0, showall=False, blueonly=True)


to generate randomly selected galaxies with sizes in a particular range:

paperplots()


Written by Rose A. Finn, 4/4/14

Updated 30-Aug-2015



'''

import urllib
from LCSanalyzeblue import *
#import aplpy
import glob
from PIL import Image
from pylab import *
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib import patches
from pyraf import iraf
# get NSA cutout

tilefontsize=12

def plotimage(fits_image,vmin=0,vmax=4,haflag=0,dx=100,zoom=1,xc=None,yc=None,hdu=None,linear=False, noise=False):
    rfits=fits.open(fits_image)
    if hdu:
        im=rfits[hdu].data
    else:
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
        v1=sdss_sb_cut*.9
        v2=v1*500.
    if fits_image.find('cutout24') > -1:
        v1=mips_sb_cut*.1
        v2=mips_sb_cut*125.
    if fits_image.find('mips') > -1:
        v1=mips_sb_cut*-1
        v2=mips_sb_cut*125.
    if fits_image.find('residual') > -1:
        v1=mips_sb_cut*-1
        v2=mips_sb_cut*124.
    if noise:
        v1=-2.*mips_sb_cut
        v2=mips_sb_cut*20
    if zoom:
        #print 'zooming'
        if linear:
            imfig=imshow((im[y1:y2,x1:x2]),interpolation='nearest',origin='lower',cmap=cm.gray_r,vmin=v1,vmax=v2)
        else:
            imfig=imshow((im[y1:y2,x1:x2]),interpolation='nearest',origin='lower',cmap=cm.gray_r,vmin=v1,vmax=v2,norm=LogNorm())
    else:
        imfig=imshow((im).T,interpolation='nearest',origin='upper',cmap=cm.gray_r,vmin=v1,vmax=v2)
    #ax.set_yticklabels(([]))
    #ax.set_xticklabels(([]))
    return imfig

class myspirals(spirals):

    def getNSAcutouts(self):
        for i in range(len(self.s.RA)):
            filename=homedir+'research/LocalClusters/NSAcolorcutouts/'+self.s.IAUNAME[i]+'.cutout.jpg'
            if os.path.exists(filename):
                continue
            else:
                url='http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_0/'+self.s.SUBDIR[i]+'/'+self.s.IAUNAME[i]+'.cutout.jpg'
                urllib.urlretrieve(url,filename=homedir+'research/LocalClusters/NSAcolorcutouts/'+self.s.IAUNAME[i]+'.cutout.jpg')

    def sortbytruncation(self):
        for i in range(len(self.s.RA)):
            filename=homedir+'research/LocalClusters/NSAcolorcutouts/'+self.s.IAUNAME[i]+'.cutout.jpg'
            if self.galfitflag[i]:


                if self.truncflag[i]:
                    os.system('cp '+filename+' '+homedir+'/research/LocalClusters/NSAcolorcutouts/Truncated/.')
                else:
                    os.system('cp '+filename+' '+homedir+'/research/LocalClusters/NSAcolorcutouts/Normal/.')
            else:
                    os.system('cp '+filename+' '+homedir+'/research/LocalClusters/NSAcolorcutouts/NonGalfit/.')



    def plotallcolor(self):
        figure(figsize=(8,11))
        subplots_adjust(hspace=0.005,wspace=0.01)
        files=glob.glob(homedir+'research/LocalClusters/NSAcolorcutouts/Truncated/*.jpg')
        for i in range(len(files)):
            subplot(11,8,i+1)
            im=Image.open(files[i])
            arr=asarray(im)
            imshow(arr)
            xticks([])
            yticks([])
        savefig(homedir+'research/LocalClusters/SamplePlots/tileplot_truncated.png')
        

        figure(figsize=(10,12))
        subplots_adjust(hspace=0.005,wspace=0.01)
        files=glob.glob(homedir+'research/LocalClusters/NSAcolorcutouts/Normal/*.jpg')
        for i in range(len(files)):
            subplot(12,10,i+1)
            im=Image.open(files[i])
            arr=asarray(im)
            imshow(arr)
            xticks([])
            yticks([])
        savefig(homedir+'research/LocalClusters/SamplePlots/tileplot_normal.png')

    def plotcolorsort(self):
        index=argsort(self.sizeratio)
        n=0
        q=1
        figure(figsize=(6,10))
        subplots_adjust(hspace=0.01,wspace=0.01)
        ny=9
        nx=5

        flag=self.sampleflag & self.sbflag
        print 'number of galaxies to plot = ',sum(flag)
        minsize=self.sizeratio[index[flag]][0]
        for i in index:
            if (flag[i]):

                n += 1
                im=homedir+'research/LocalClusters/NSAcolorcutouts/'+self.s.IAUNAME[i]+'.cutout.jpg'
                subplot(ny,nx,n)
                im=Image.open(im)
                arr=asarray(im)
                imshow(arr)
                xticks([])
                yticks([])
                if n > (nx*ny-1):
                    #subplot(nx,ny,4)
                    ax=gca()

                    s='$ %4.2f < R_e(24)/R_e(r) < %4.2f $'%(minsize,self.sizeratio[i])
                    text(-2,9.3,s,transform=ax.transAxes,verticalalignment='center',horizontalalignment='center',fontsize=20)
                    savefig(homedir+'research/LocalClusters/SamplePlots/tileplot_q'+str(q)+'.png')
                    savefig(homedir+'research/LocalClusters/SamplePlots/tileplot_q'+str(q)+'.eps')
                    print 'Quartile ',q,' ends at size ratio = ',self.sizeratio[i]
                    q += 1
                    n=0
                    minsize=self.sizeratio[i]
                    
                    figure(figsize=(6,10))
                    subplots_adjust(hspace=0.01,wspace=0.01)
                lastsize=self.sizeratio[i]

        s='$ %4.2f < R_e(24)/R_e(r) < %4.2f $'%(minsize,lastsize)
        text(.5,9.3,s,transform=ax.transAxes,verticalalignment='center',horizontalalignment='center',fontsize=20)

        savefig(homedir+'research/LocalClusters/SamplePlots/tileplot_q'+str(q)+'.png')
        savefig(homedir+'research/LocalClusters/SamplePlots/tileplot_q'+str(q)+'.eps')

                
                

    def plotprofilesconv(self,i):

        colorimage=homedir+'research/LocalClusters/NSAcolorcutouts/'+self.s.IAUNAME[i]+'.cutout.jpg'

        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.s.CLUSTER[i]+'/NSA/'
        working_dir24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.s.CLUSTER[i]+'/24um/'


        subcomp_image=working_dir+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-1Comp-galfit-out.fits'
        subcomp_image24_conv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-conv-1Comp-galfit-out.fits'


        galname=self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])
        nrow=2
        ncol=3
        xticksize=10
        yticksize=10
        
        x1,x2,x3=(.1,.41,.72)
        dx=.24
        y1,y2,y3=(.05,.36,.67)
        dy=.26
        vminsdss=-1
        vmaxsdss=1
        vminsdssapl=-100
        vmaxsdssapl=400
        
        vmin24=-40
        vmax24=10.
        subplots_adjust(left=0.05, right=.95,bottom=.1,top=0.9,wspace=0.3,hspace=0.2)
        
        fig=figure(figsize=(11,10))
        
        clf()
        

        fig=gcf()
        
        try:
            subf=aplpy.FITSFigure(subcomp_image,hdu=1,figure=fig,subplot=[x1,y3,dx,dy])
        except IOError:
            return
        subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)
        #subf.show_colorscale(cmap='gist_heat')
        axratio=self.galfit.axisratio1[i]
        phi=self.galfit.pa1[i]
        subf.tick_labels.set_font(size='x-small')
        
        s='$'+galname+'$'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)
        s='AGN Flag = %i'%(self.agn1[i])
        subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')

        s='$M_r = %5.1f,\ log(M_*)=%5.2f$'%(self.sdssMr[i],log10(self.stellarmass[i]))
        subf.add_label(.05,.9,s,size='large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')

        s='r-band'
        subf.add_label(.05,.1,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        
        
        
        subf=aplpy.FITSFigure(subcomp_image,hdu=2,figure=fig,subplot=[x2,y3,dx,dy])
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)
        #subf.show_ellipses(self.ra[i],self.dec[i],self.galfit.re1[i]/3600.,self.galfit.re1[i]*axratio/3600.,angle=phi,edgecolor='r')
        #subf.show_ellipses(self.ra[i],self.dec[i],self.galfit.R90[i]/3600.,self.galfit.R90[i]*axratio/3600.,angle=phi,edgecolor='r')
        subf.tick_labels.set_font(size='x-small')

        s='$ Model $'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)


        # mag, Re, n, B/A, phi
        s='$ Sersic \ param: \ m_r= %5.1f,Re=%5.1f$'%(self.galfit.mag1[i],self.galfit.re1[i]*sdsspixelscale)
        subf.add_label(.95,.9,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        s='$n=%5.1f,B/A=%5.2f, PA=%5.1f $'%(self.galfit.nsersic1[i],self.galfit.axisratio1[i],self.galfit.pa1[i])
        subf.add_label(.95,.82,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        s='$ NSA \ param: \ m_r= %5.1f,Re=%5.1f$'%(22.5-2.5*log10(self.n.SERSICFLUX[i,4]),self.n.SERSIC_TH50[i])
        subf.add_label(.95,.18,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        s='$n=%5.1f,B/A=%5.2f, PA=%5.1f $'%(self.n.SERSIC_N[i],self.n.SERSIC_BA[i],self.n.SERSIC_PHI[i])
        subf.add_label(.95,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image,hdu=3,figure=fig,subplot=[x3,y3,dx,dy])
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)

        s='$ Residual $'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)

        s='Spiral Flag = '+str(self.spiralflag[i])+' (%5.2f)'%(self.zoo.p_cs[i])
        subf.add_label(.5,.9,s,relative=True,color='white')

        s='Ellip Flag = '+str(self.ellipticalflag[i])+' (%5.2f)'%(self.zoo.p_el[i])
        subf.add_label(.5,.82,s,relative=True,color='white')


        # second row - with convolution
        fig=gcf()
        try:
            subf=aplpy.FITSFigure(subcomp_image24,hdu=1,figure=fig,subplot=[x1,y2,dx,dy])
        except IOError:
            return
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        s='$ dv/\sigma = %5.2f,\ dr/R_{200} = %5.2f$'%(self.dv[i],self.drR200[i])
        #subf.add_label(.5,1.05,s,fontsize=12,relative=True)
        
        subf.tick_labels.set_font(size='x-small')
        s='SNR(24)=%5.1f, SFR=%5.2f'%(self.snr24[i],self.ce.SFR[i])
        subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')
        if self.HIflag[i]:
            s='$ log_{10}(M_{HI}) = %5.2f$'%(log10(self.HImass[i]))
        else:
            s='$No \ HI$'

        subf.add_label(.05,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        
        s='24um'
        subf.add_label(.05,.1,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        

        subf=aplpy.FITSFigure(subcomp_image24,hdu=2,figure=fig,subplot=[x2,y2,dx,dy])
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        subf.tick_labels.set_font(size='x-small')
        if self.galfit24.numerical_error_flag24[i]:
            s='$ Sersic \ param**: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.mag1[i],self.galfit24.re1[i]*mipspixelscale)
        else:
            s='$ Sersic \ param: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.mag1[i],self.galfit24.re1[i]*mipspixelscale)
        subf.add_label(.95,.9,s,fontsize=12,relative=True,color='white',horizontalalignment='right',family='serif',weight='medium')

        s='$ n=%5.1f,B/A=%4.2f,PA=%5.1f $'%(self.galfit24.nsersic1[i],self.galfit24.axisratio1[i],self.galfit24.pa1[i])
        subf.add_label(.95,.82,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        #s='$ SE \ param: \ m= %5.1f,Re=%5.1f$'%((self.sex24.MAG_BEST[i]),self.sex24.FLUX_RADIUS1[i]*mipspixelscale)
        #subf.add_label(.95,.18,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        #s='$B/A=%5.2f, PA=%5.1f $'%(1-self.sex24.ELLIPTICITY[i],self.sex24.THETA_IMAGE[i])
        #subf.add_label(.95,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        # ratio of Re(24)/Re(r)
        s='$ R_e(24)/R_e(r)=%5.1f $'%(self.galfit24.re1[i]*mipspixelscale/self.n.SERSIC_TH50[i])
        subf.add_label(.95,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image24,hdu=3,figure=fig,subplot=[x3,y2,dx,dy])
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        subf.tick_labels.set_font(size='x-small')
        s='$\chi^2nu= %6.3f$'%(self.galfit24.chi2nu[i])
        subf.add_label(.15,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')



        # third row - with convolution
        fig=gcf()
        try:
            subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=1,figure=fig,subplot=[x1,y1,dx,dy])
        except IOError:
            return
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)

        #subf.add_label(.5,1.05,s,fontsize=12,relative=True)
        
        subf.tick_labels.set_font(size='x-small')
        s='SNR(24)=%5.1f, SFR=%5.2f'%(self.snr24[i],self.ce.SFR[i])
        #subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')

        s='$ dv/\sigma = %5.2f,\ dr/R_{200} = %5.2f$'%(self.dv[i],self.drR200[i])
        subf.add_label(.05,.9,s,size='large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        
        s='24um+conv'
        subf.add_label(.05,.1,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')

        subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=2,figure=fig,subplot=[x2,y1,dx,dy])
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        subf.tick_labels.set_font(size='x-small')
        if self.galfit24.fcnumerical_error_flag24[i]:
            s='$ Sersic \ param**: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.cmag1[i],self.galfit24.cre1[i]*mipspixelscale)
        else:
            s='$ Sersic \ param: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.cmag1[i],self.galfit24.cre1[i]*mipspixelscale)
        subf.add_label(.95,.9,s,fontsize=12,relative=True,color='white',horizontalalignment='right',family='serif',weight='medium')

        s='$ n=%5.1f,B/A=%4.2f,PA=%5.1f $'%(self.galfit24.fcnsersic1[i],self.galfit24.fcaxisratio1[i],self.galfit24.fcpa1[i])
        subf.add_label(.95,.82,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        # ratio of Re(24)/Re(r)
        s='$ R_e(24)/R_e(r)=%5.1f $'%(self.galfit24.cre1[i]*mipspixelscale/self.n.SERSIC_TH50[i])
        subf.add_label(.95,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        #s='$ SE \ param: \ m= %5.1f,Re=%5.1f$'%((self.sex24.MAG_BEST[i]),self.sex24.FLUX_RADIUS1[i]*mipspixelscale)
        #subf.add_label(.88,.18,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        #s='$B/A=%5.2f, PA=%5.1f $'%(1-self.sex24.ELLIPTICITY[i],self.sex24.THETA_IMAGE[i])
        #subf.add_label(.88,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=3,figure=fig,subplot=[x3,y1,dx,dy])
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        subf.tick_labels.set_font(size='x-small')

        s='$ \chi^2nu= %6.3f$'%(self.galfit24.cchi2nu[i])
        subf.add_label(.15,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')

        figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+str(galname)+'-galfitResults-conv.png'
        savefig(figname)
    def plotcolor_subset(self,size1=0,size2=3,ngal=5,showall=True,pointsource=True,plotellipse=False):
        '''
        need to send 5 indices at a time to plotcolor_r_24

        this function does a randomly selected set

        updating on 1/6/16 to avoid plotting duplicate images

        using np.random.shuffle

        '''
        nrow=ngal
        if pointsource:
            indices = where((self.sizeratio > size1) & (self.sizeratio < size2)  & self.sampleflag& ~self.agnflag & self.gim2dflag)
        else:
            indices = where((self.sizeratio > size1) & (self.sizeratio < size2) & self.sampleflag & ~self.agnflag & self.gim2dflag & self.pointsource)
        np.random.shuffle(indices[0])
        plotindices=indices[0][0:nrow]
        if showall:
            self.plotcolor_r_24_showall(plotindices, nrow=nrow,plotnoconv=True,plotellipse=plotellipse)
        else:
            self.plotcolor_r_24_showall(plotindices, nrow=nrow,plotellipse=plotellipse)

    def plotcolor_all(self,ngal=5,showall=False,orderbynsaid=False,plotellipse=0,orderbyssfr=False,blueonly=False):
        '''
        need to send 5 indices at a time to plotcolor_r_24

        this function does a randomly selected set

        ngal specifies number of galaxies per figure (number of rows)
        
        '''
        nrow=ngal
        # sort according to SIZE_RATIO
        if blueonly:
            flag=self.bluesampleflag
        else:
            flag=self.sampleflag
        indices=argsort(self.sizeratio)
        sorted_sampleflag=flag[argsort(self.sizeratio)]
        if orderbynsaid:
            indices=argsort(self.s.NSAID)
            sorted_sampleflag=flag[argsort(self.s.NSAID)]
        elif orderbyssfr:
            indices=argsort(self.s.SFR)
            sorted_sampleflag=flag[argsort(self.ssfr)]
        #print indices
        sample_indices=indices[sorted_sampleflag]
        # number of figures = sample / ngal
        nfigure=int(floor(len(sample_indices)*1./ngal))
        print 'making ',nfigure,' figures'
        # may need partial page
        if (((1.*len(sample_indices))%(1.*ngal)) > 0.1):
                nfigure += 1
        # loop over all indices, sending ngal galaxies each time to plotcolor_r_24
        # then save figure
        
        for k in range(nfigure):
            if showall:
                self.plotcolor_r_24_showall(sample_indices[k*5:k*5+ngal], nrow=nrow,plotellipse=plotellipse,plotnoconv=True)
            else:
                self.plotcolor_r_24_showall(sample_indices[k*5:k*5+ngal], nrow=nrow,plotellipse=plotellipse)
            minsize=self.sizeratio[k*5]
            maxsize=self.sizeratio[k*5+ngal-1]
            #figname='LCS_mosaic_%02d_%02d.png'%(int(minsize*10),int(maxsize*10))
            figname='LCS_mosaic_blue_%02d.png'%(k)
            if orderbynsaid:
                figname='LCS_mosaic_blue_NSAID_%02d.png'%(k)
            savefig(homedir+'/research/LocalClusters/SamplePlots/'+figname)
            close('all')
    def plotcolor_r_24(self,plotindices, dr=40,zoom=1,nrow=5,plotellipse=True):
        '''
        plot color, r-band, and 24um

        make a collage of 10 randomly chosen spirals with size < .45

        make another collage of 10 randomly chosen spirals with size > .7

        make sure that all images are displayed on the same grayscale
        
        '''
        # get indices of size < 0.45
        #indices = where((self.sizeratio > size1) & (self.sizeratio < size2) & self.sampleflag)
        #print indices
        # select 10 at random
        ncol=6
        #close('all')
        figure(figsize=(2*ncol,2*nrow))
        subplots_adjust(hspace=.01,wspace=.01)
        clf()
                


        for k in range(len(plotindices)):
            i=plotindices[k]
            #print k,i

            colorimage=homedir+'research/LocalClusters/NSAcolorcutouts/'+self.s.IAUNAME[i]+'.cutout.jpg'

            working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.s.CLUSTER[i]+'/NSA/'
            working_dir24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.s.CLUSTER[i]+'/24um/'


            subcomp_image=working_dir+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-1Comp-galfit-out.fits'
            subcomp_image24_conv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'
            subcomp_image24_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-1Comp-galfit-out.fits'
            subcomp_image24_conv_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'
            mips_model=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-model.fits'
            mips_galaxy=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-galaxy.fits'
            mips_residual=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-residual.fits'
            # no convolution model
            mips_model_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-model.fits'
            mips_galaxy_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-galaxy.fits'
            mips_residual_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-residual.fits'
            # fixed BA and PA model
            mips_model_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-mips-model.fits'
            mips_galaxy_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-mips-galaxy.fits'
            mips_residual_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-mips-residual.fits'

            # copy galfit output file into individual images so it places nice with imshow()
            # model + conv
            if not(os.path.exists(mips_model)):
                iraf.imcopy(subcomp_image24_conv+'[2]',mips_model)
            if not(os.path.exists(mips_residual)):
                iraf.imcopy(subcomp_image24_conv+'[3]',mips_residual)
            if not(os.path.exists(mips_galaxy)):
                iraf.imcopy(subcomp_image24_conv+'[1]',mips_galaxy)

            # model + no conv
            if not(os.path.exists(mips_model_noconv)):
                iraf.imcopy(subcomp_image24_noconv+'[2]',mips_model_noconv)
            if not(os.path.exists(mips_residual_noconv)):
                iraf.imcopy(subcomp_image24_noconv+'[3]',mips_residual_noconv)

            # model w/conv and fixed BA + PA                            
            if not(os.path.exists(mips_model_fixedBA)):
                iraf.imcopy(subcomp_image24_conv_fixedBA+'[2]',mips_model_fixedBA)
            if not(os.path.exists(mips_residual_fixedBA)):
                iraf.imcopy(subcomp_image24_conv_fixedBA+'[3]',mips_residual_fixedBA)

            #print subcomp_image24_conv
            rfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(self.s.CLUSTER[i])+'/NSA/'+str(self.s.CLUSTER[i])+'-'+str(self.s.NSAID[i])+'-parent-r.fits'
            mipsfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(self.s.CLUSTER[i])+'/24um/'+str(self.s.CLUSTER[i])+'-'+str(self.s.NSAID[i])+'-galfit-cutout24.fits'

            hdu=fits.open(rfits)
            hdat=hdu[0].data.copy()
            n2,n1=hdat.shape
            hdu.close()

            hdu=fits.open(mipsfits)
            hdat=hdu[0].data.copy()
            n2mips,n1mips=hdat.shape
            hdu.close()





            fs=tilefontsize
            subplot(nrow,ncol,ncol*k+1)
            if k == 0:
                title('$NSA$',fontsize=fs)
            im=Image.open(colorimage)
            arr=asarray(im)
            imsize=arr.shape[0]*sdsspixelscale
            dr=imsize/2.
            imshow(arr,origin='upper')
            xticks([])
            yticks([])
            #im.close()
            galname=self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])
            ax=gca()
            text(.01,.87,'$'+str(self.s.CLUSTER[i])+'$',transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='w')
            if self.s.fcnumerical_error_flag24[i] > 0.1:
                s='$ *%3.2f $'%(self.sizeratio[i])
            else:
                s='$ %3.2f $'%(self.sizeratio[i])

            text(.98,.87,s,transform=ax.transAxes,horizontalalignment='right',fontsize=fs,color='w')

            if self.s.fcnumerical_error_flag24[i] > 0.1:
                s='$ *%3.2f $'%(self.size_ratio_corr[i])
            else:
                s='$ %3.2f $'%(self.size_ratio_corr[i])

            text(.98,.02,s,transform=ax.transAxes,horizontalalignment='right',fontsize=fs,color='w')

            text(.01,.02,'$'+str(self.s.NSAID[i])+'$',transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='w')

            
            mindim=min([n1*sdsspixelscale,n2*sdsspixelscale,n1mips*mipspixelscale,n2mips*mipspixelscale])/2.
            if dr > (mindim-2):
                dr=(mindim-2)

            subplot(nrow,ncol,ncol*k+2)
            if k == 0:
                title('$r-band$',fontsize=fs)

    
                
            #hdu=fits.open(subcomp_image)
            #image_data=hdu[1].data
            #imshow(image_data,cmap='gray')
            #colorbar()
            #xticks([])
            #yticks([])
            #hdu.close()
            w= WCS(rfits)
            px,py = w.wcs_world2pix(self.s.RA[i],self.s.DEC[i],1)

            imfig=plotimage(rfits,vmin=-.1,vmax=2.,dx=dr/sdsspixelscale,zoom=1,xc=px,yc=py)
            xticks([])
            yticks([])

            ax=gca()
            if plotellipse:
                ell=patches.Ellipse((dr/sdsspixelscale,dr/sdsspixelscale),width=2.*self.s.SERSIC_BA[i]*self.s.SERSIC_TH50[i]/sdsspixelscale,height=2.*self.s.SERSIC_TH50[i]/sdsspixelscale,angle=self.s.SERSIC_PHI[i],color='r',fill=False,linewidth=1,alpha=.7)
                gca().add_patch(ell)


            
            #text(.5,.9,'$'+str(self.s.CLUSTER[i])+'$',transform=ax.transAxes,horizontalalignment='center',fontsize=16)



            subplot(nrow,ncol,ncol*k+3)
            if k == 0:
                title('$24\mu m \ contrast1$',fontsize=fs)

            w= WCS(mipsfits)
            px,py = w.wcs_world2pix(self.s.RA[i],self.s.DEC[i],1)

            imfig=plotimage(mipsfits,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True)
            ax=gca()
            xticks([])
            yticks([])
            if plotellipse:
                print dr/mipspixelscale, px, py
                #ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.caxisratio1[i]*self.s.cre1[i]/mipspixelscale,height=2.*self.s.cre1[i]/mipspixelscale,angle=self.s.cpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
                #ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.caxisratio1[i]*self.s.cre1[i],height=2.*self.s.cre1[i],angle=self.s.cpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
                ell=patches.Ellipse((self.s.xc1[i],self.s.yc1[i]),width=2.*self.s.caxisratio1[i]*self.s.cre1[i],height=2.*self.s.cre1[i],angle=self.s.cpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
                ax.add_patch(ell)
                #if self.s.cnumerical_error_flag24[i]:
                #    ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.axisratio1[i]*self.s.re1[i]/mipspixelscale,height=2.*self.s.re1[i]/mipspixelscale,angle=self.s.pa1[i],color='b',fill=False,linewidth=1,alpha=.7)
                #    ax.add_patch(ell)

            
            w= WCS(mips_galaxy)
            px,py = w.wcs_world2pix(self.s.RA[i],self.s.DEC[i],1)
            print px,py

            subplot(nrow,ncol,ncol*k+4)
            if k == 0:
                title('$24\mu m$',fontsize=fs)


            imfig=plotimage(mipsfits,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True,noise=True)
            xticks([])
            yticks([])

            ax=gca()
            if plotellipse:
                print dr/mipspixelscale, px, py
                #ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.caxisratio1[i]*self.s.cre1[i]/mipspixelscale,height=2.*self.s.cre1[i]/mipspixelscale,angle=self.s.cpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
                ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.faxisratio1[i]*self.s.cre1[i],height=2.*self.s.fre1[i],angle=self.s.fpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
                ax.add_patch(ell)
                if self.s.fnumerical_error_flag24[i]:
                    ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.faxisratio1[i]*self.s.fre1[i]/mipspixelscale,height=2.*self.s.fre1[i]/mipspixelscale,angle=self.s.pa1[i],color='b',fill=False,linewidth=1,alpha=.7)
                    ax.add_patch(ell)

            
            subplot(nrow,ncol,ncol*k+5)
            if k == 0:
                title('$model$',fontsize=fs)

            imfig=plotimage(mips_model,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True,noise=True)
            #model=fits.open(subcomp_image24_conv)
            #d=model[2].data
            #imshow(d)
            xticks([])
            yticks([])

            subplot(nrow,ncol,ncol*k+6)
            if k == 0:
                title('$residual$',fontsize=fs)

            imfig=plotimage(mips_residual,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True, noise=True)
            xticks([])
            yticks([])

                        



    def plotcolor_r_24_showall(self,plotindices, dr=40,zoom=1,nrow=5,plotellipse=True,plotnoconv=False):
        '''
        plot color, r-band, and 24um

        make a collage of 5 randomly chosen spirals with size < .45

        make another collage of 5 randomly chosen spirals with size > .7

        make sure that all images are displayed on the same grayscale

        show fits with no conv, conv, conv+fixedBA
        
        '''
        # get indices of size < 0.45
        #indices = where((self.sizeratio > size1) & (self.sizeratio < size2) & self.sampleflag)
        #print indices
        # select 10 at random
        if plotnoconv:
            ncol=8
        else:
            ncol=6
        #close('all')
        figure(figsize=(1.8*ncol,1.8*nrow))
        subplots_adjust(hspace=.01,wspace=.01)
        clf()
                


        for k in range(len(plotindices)):
            i=plotindices[k]
            #print k,i
            colorimage=homedir+'research/LocalClusters/NSAcolorcutouts/'+self.s.IAUNAME[i]+'.cutout.jpg'

            working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.s.CLUSTER[i]+'/NSA/'
            working_dir24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.s.CLUSTER[i]+'/24um/'


            subcomp_image=working_dir+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-1Comp-galfit-out.fits'
            subcomp_image24_conv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'
            subcomp_image24_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-1Comp-galfit-out.fits'
            subcomp_image24_conv_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-1Comp-galfit-out.fits'
            mips_model=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-model.fits'
            mips_galaxy=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-galaxy.fits'
            mips_residual=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-residual.fits'
            # no convolution model
            mips_model_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-model.fits'
            mips_galaxy_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-galaxy.fits'
            mips_residual_noconv=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-noconv-mips-residual.fits'
            # fixed BA and PA model
            mips_model_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-mips-model.fits'
            mips_galaxy_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-mips-galaxy.fits'
            mips_residual_fixedBA=working_dir24+self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])+'-24-fixedBA-mips-residual.fits'

            # copy galfit output file into individual images so it places nice with imshow()
            # model + conv
            if not(os.path.exists(mips_model)):
                iraf.imcopy(subcomp_image24_conv+'[2]',mips_model)
            if not(os.path.exists(mips_residual)):
                iraf.imcopy(subcomp_image24_conv+'[3]',mips_residual)
            if not(os.path.exists(mips_galaxy)):
                iraf.imcopy(subcomp_image24_conv+'[1]',mips_galaxy)

            # model + no conv
            if not(os.path.exists(mips_model_noconv)):
                iraf.imcopy(subcomp_image24_noconv+'[2]',mips_model_noconv)
            if not(os.path.exists(mips_residual_noconv)):
                iraf.imcopy(subcomp_image24_noconv+'[3]',mips_residual_noconv)

            # model w/conv and fixed BA + PA                            
            if not(os.path.exists(mips_model_fixedBA)):
                iraf.imcopy(subcomp_image24_conv_fixedBA+'[2]',mips_model_fixedBA)
            if not(os.path.exists(mips_residual_fixedBA)):
                iraf.imcopy(subcomp_image24_conv_fixedBA+'[3]',mips_residual_fixedBA)

            #print subcomp_image24_conv
            rfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(self.s.CLUSTER[i])+'/NSA/'+str(self.s.CLUSTER[i])+'-'+str(self.s.NSAID[i])+'-parent-r.fits'
            mipsfits=homedir+'research/LocalClusters/GalfitAnalysis/'+str(self.s.CLUSTER[i])+'/24um/'+str(self.s.CLUSTER[i])+'-'+str(self.s.NSAID[i])+'-galfit-cutout24.fits'

            hdu=fits.open(rfits)
            hdat=hdu[0].data.copy()
            n2,n1=hdat.shape
            hdu.close()

            hdu=fits.open(mipsfits)
            hdat=hdu[0].data.copy()
            n2mips,n1mips=hdat.shape
            hdu.close()




            fs=tilefontsize
            subplot(nrow,ncol,ncol*k+1)
            if k == 0:
                title('$NSA$',fontsize=fs)
            im=Image.open(colorimage)
            arr=asarray(im)
            imsize=arr.shape[0]*sdsspixelscale
            dr=imsize/2.
            imshow(arr,origin='upper')
            xticks([])
            yticks([])
            #im.close()
            galname=self.s.CLUSTER[i]+'-'+str(self.s.NSAID[i])
            ax=gca()
            text(.01,.87,'$'+str(self.s.CLUSTER[i])+'$',transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='w')
            if self.s.fcnumerical_error_flag24[i] > 0.1:
                s='$ *%3.2f $'%(self.sizeratio[i])
            else:
                s='$ %3.2f $'%(self.sizeratio[i])
            s='$ %3.2f $'%(self.sizeratio[i])
            if self.upperlimit[i]:
                if self.pointsource[i]:
                    s='$ < %3.2f* $'%(self.sizeratio[i])
                else:
                    s='$ < %3.2f $'%(self.sizeratio[i])
            text(.98,.87,s,transform=ax.transAxes,horizontalalignment='right',fontsize=fs,color='w')

            if self.AGNKAUFF[i]:
                s='$ AGN $'
                text(.98,.02,s,transform=ax.transAxes,horizontalalignment='right',fontsize=fs,color='w')

            text(.01,.02,'$'+str(self.s.NSAID[i])+'$',transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='w')

            
            mindim=min([n1*sdsspixelscale,n2*sdsspixelscale,n1mips*mipspixelscale,n2mips*mipspixelscale])/2.
            if dr > (mindim-2):
                dr=(mindim-2)

            subplot(nrow,ncol,ncol*k+2)
            if k == 0:
                title('$r-band$',fontsize=fs)

    
                
            #hdu=fits.open(subcomp_image)
            #image_data=hdu[1].data
            #imshow(image_data,cmap='gray')
            #colorbar()
            #xticks([])
            #yticks([])
            #hdu.close()
            w= WCS(rfits)
            px,py = w.wcs_world2pix(self.s.RA[i],self.s.DEC[i],1)

            imfig=plotimage(rfits,vmin=-.1,vmax=2.,dx=dr/sdsspixelscale,zoom=1,xc=px,yc=py)
            xticks([])
            yticks([])

            ax=gca()
    
            if plotellipse:
                ell=patches.Ellipse((dr/sdsspixelscale,dr/sdsspixelscale),width=2.*self.s.SERSIC_BA[i]*self.s.SERSIC_TH50[i]/sdsspixelscale,height=2.*self.s.SERSIC_TH50[i]/sdsspixelscale,angle=self.s.SERSIC_PHI[i],color='r',fill=False,linewidth=1,alpha=.7)
                gca().add_patch(ell)

            self.add_Re_n(gca(),self.s.Rd[i]/self.DA[i],self.s.ng[i],0)
            ax=gca()
            if self.gim2dflag[i]:
                s='$B/T=%3.2f$'%(self.s.B_T_r[i])
            else:
                s='$B/T=?$'
            #print 'B/T = ',s
            text(.98,.87,s,transform=ax.transAxes,horizontalalignment='right',fontsize=fs,color='k')
            
            #text(.5,.9,'$'+str(self.s.CLUSTER[i])+'$',transform=ax.transAxes,horizontalalignment='center',fontsize=16)



            subplot(nrow,ncol,ncol*k+3)
            if k == 0:
                title('$24\mu m \ contrast1$',fontsize=fs)

            w= WCS(mipsfits)
            px,py = w.wcs_world2pix(self.s.RA[i],self.s.DEC[i],1)

            imfig=plotimage(mipsfits,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True)
            ax=gca()
            xticks([])
            yticks([])
            
            w= WCS(mips_galaxy)
            px,py = w.wcs_world2pix(self.s.RA[i],self.s.DEC[i],1)
            #print px,py
            if self.pointsource[i]:
                ax=gca()
                text(.02,.025,'$R_e < %3.2f$'%(mipspixelscale),horizontalalignment='left',transform=ax.transAxes,color='r')

            
            subplot(nrow,ncol,ncol*k+4)
            if k == 0:
                title('$24\mu m \ contrast2$',fontsize=fs)

            imfig=plotimage(mipsfits,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True,noise=True)
            xticks([])
            yticks([])

            if plotnoconv:
                subplot(nrow,ncol,ncol*k+7)
                if k == 0:
                    title('$model$',fontsize=fs)

                imfig=plotimage(mips_model_noconv,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True,noise=True)
                #model=fits.open(subcomp_image24_conv)
                #d=model[2].data
                #imshow(d)
                xticks([])
                yticks([])
                #self.add_Re_n(gca(),self.s.re1[i]*mipspixelscale,self.s.nsersic1[i],self.s.numerical_error_flag24[i])
                self.add_Re_n_werror(gca(),self.s.fre1[i]*mipspixelscale,self.s.fre1err[i]*mipspixelscale,self.s.fnsersic1[i],self.s.fnsersic1err[i],self.s['fnumerical_error_flag24'][i])

                if plotellipse:
                    ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.faxisratio1[i]*self.s.fre1[i]/mipspixelscale,height=2.*self.s.fre1[i]/mipspixelscale,angle=self.s.fpa1[i],color='r',fill=False,linewidth=1,alpha=.7)

                    #ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.caxisratio1[i]*self.s.cre1[i],height=2.*self.s.cre1[i],angle=self.s.cpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
                    gca().add_patch(ell)



                subplot(nrow,ncol,ncol*k+8)
                if k == 0:
                    title('$residual$',fontsize=fs)

                imfig=plotimage(mips_residual_noconv,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True, noise=True)
                xticks([])
                yticks([])
                self.add_chisq(gca(),self.s.fchi2nu[i])


            subplot(nrow,ncol,ncol*k+5)
            ax=gca()
            if k == 0:
                title('$24\mu m \ model$',fontsize=fs)# fixed BA

            imfig=plotimage(mips_model_fixedBA,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True,noise=True)
            xticks([])
            yticks([])

            self.add_Re_n_werror(ax,self.s.fcre1[i]*mipspixelscale,self.s.fcre1err[i]*mipspixelscale,self.s.fcnsersic1[i],self.s.fcnsersic1err[i],self.s['fcnumerical_error_flag24'][i])

            if plotellipse:
                ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.fcaxisratio1[i]*self.s.fcre1[i],height=2.*self.s.fcre1[i],angle=self.s.fcpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
                gca().add_patch(ell)

            subplot(nrow,ncol,ncol*k+6)
            if k == 0:
                title('$24\mu m \ residual$',fontsize=fs)

            imfig=plotimage(mips_residual_fixedBA,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True, noise=True)
            xticks([])
            yticks([])
            self.add_chisq(gca(),self.s.fcchi2nu[i])
            #s='$GZoo \ p_{cs} = %5.2f$'%(self.s.p_cs[i])
            #text(.98,.87,s,transform=gca().transAxes,horizontalalignment='right',color='k',fontsize=12)

            ## subplot(nrow,ncol,ncol*k+9)
            ## if k == 0:
            ##     title('$model w/conv$',fontsize=fs)

            ## imfig=plotimage(mips_model,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True,noise=True)
            ## xticks([])
            ## yticks([])
            ## self.add_Re_n(gca(),self.s.cre1[i]*mipspixelscale,self.s.cnsersic1[i],self.s.cnumerical_error_flag24[i])
            ## if plotellipse:
            ##     #ell=patches.Ellipse((dr/mipspixelscale,dr/mipspixelscale),width=2.*self.s.axisratio1[i]*self.s.re1[i]/mipspixelscale,height=2.*self.s.re1[i]/mipspixelscale,angle=self.s.pa1[i],color='r',fill=False,linewidth=1,alpha=.7)

            ##     ell=patches.Ellipse((self.s.cxc1[i],self.s.cyc1[i]),width=2.*self.s.caxisratio1[i]*self.s.cre1[i],height=2.*self.s.cre1[i],angle=self.s.cpa1[i],color='r',fill=False,linewidth=1,alpha=.7)
            ##     gca().add_patch(ell)

            
            ## subplot(nrow,ncol,ncol*k+10)
            ## if k == 0:
            ##     title('$residual$',fontsize=fs)

            ## imfig=plotimage(mips_residual,vmin=-.1,vmax=1.5,dx=dr/mipspixelscale,zoom=1,xc=px,yc=py,linear=True, noise=True)
            ## xticks([])
            ## yticks([])
            ## self.add_chisq(gca(),self.s.cchi2nu[i])

            
    def add_Re_n(self,ax,re,n,numflag):
        #print 'inside add_Re_n ',re,n
        fs=tilefontsize
        if numflag:
            text(.02,.85,'$ n=%2.1f *$'%(n),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')

            text(.02,.02,'$ R_d=%4.2f * $'%(re),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')
        else:
            text(.02,.85,'$ n=%2.1f $'%(n),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')

            text(.02,.02,'$ R_d=%4.2f $'%(re),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')
    def add_Re_n_werror(self,ax,re,re_error,n,n_error,numflag):
        #print 'inside add_Re_n ',re,n
        fs=tilefontsize
        if numflag:
            text(.02,.85,'$ n=%2.2f(%2.2f) *$'%(n,n_error),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')

            text(.02,.02,'$ Re=%4.2f(%4.2f) * $'%(re,re_error),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')
        else:
            text(.02,.85,'$ n=%2.2f(%2.2f) $'%(n,n_error),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')

            text(.02,.02,'$ Re=%4.2f(%4.2f) $'%(re,re_error),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')

    def add_chisq(self,ax,chi):
        fs=tilefontsize
        text(.02,.02,r'$ \chi ^2_{\nu}=%2.1f $'%(chi),transform=ax.transAxes,horizontalalignment='left',fontsize=fs,color='k')

    
    def extrastuff(self):
        xticksize=10
        yticksize=10
        
        x1,x2,x3=(.1,.41,.72)
        dx=.24
        '''
            y1,y2,y3=(.05,.36,.67)
            dy=.26
            vminsdss=-1
            vmaxsdss=1
            vminsdssapl=-100
            vmaxsdssapl=400
        
            vmin24=-40
            vmax24=10.
            subplots_adjust(left=0.05, right=.95,bottom=.1,top=0.9,wspace=0.3,hspace=0.2)
        
            fig=figure(figsize=(6.5,9))
        
            clf()
        

            fig=gcf()
        
            try:
                subf=aplpy.FITSFigure(subcomp_image,hdu=1,figure=fig,subplot=[x1,y3,dx,dy])
            except IOError:
                return
            subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)
            #subf.show_colorscale(cmap='gist_heat')
            axratio=self.galfit.axisratio1[i]
            phi=self.galfit.pa1[i]
            subf.tick_labels.set_font(size='x-small')
        
            s='$'+galname+'$'
            subf.add_label(.5,1.05,s,fontsize=14,relative=True)

        
        
        
            subf=aplpy.FITSFigure(subcomp_image,hdu=2,figure=fig,subplot=[x2,y3,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)
            subf.tick_labels.set_font(size='x-small')

            s='$ Model $'
            subf.add_label(.5,1.05,s,fontsize=14,relative=True)


    
            subf=aplpy.FITSFigure(subcomp_image,hdu=3,figure=fig,subplot=[x3,y3,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)



            subf=aplpy.FITSFigure(subcomp_image24,hdu=2,figure=fig,subplot=[x2,y2,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
            subf.tick_labels.set_font(size='x-small')

    
            subf=aplpy.FITSFigure(subcomp_image24,hdu=3,figure=fig,subplot=[x3,y2,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
            subf.tick_labels.set_font(size='x-small')

        
    
            # third row - with convolution
            fig=gcf()
            try:
                subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=1,figure=fig,subplot=[x1,y1,dx,dy])
            except IOError:
                return
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)

            subf.tick_labels.set_font(size='x-small')

            subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=2,figure=fig,subplot=[x2,y1,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
            subf.tick_labels.set_font(size='x-small')

            subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=3,figure=fig,subplot=[x3,y1,dx,dy])
            subf.tick_labels.hide()
            subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
            subf.tick_labels.set_font(size='x-small')

    
        figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+str(galname)+'-galfitResults-conv.png'
        savefig(figname)

            
        # plot 10 row x 3 column plot

        # repeat for size > .7 galaxies
        '''
        
        

# get r image

# get 24um image

infile=homedir+'research/LocalClusters/NSAmastertables/LCS_all_size.fits'
s=myspirals(infile)

def paperplots():
    s.plotcolor_subset(size1=0.0,size2=.3,showall=False)
    savefig('/Users/rfinn/research/LocalClusters/SamplePlots/LCS_mosaic_rand_00_03.eps')
    s.plotcolor_subset(size1=0.4,size2=.7,showall=False)
    savefig('/Users/rfinn/research/LocalClusters/SamplePlots/LCS_mosaic_rand_04_07.eps')
    s.plotcolor_subset(size1=0.9,size2=2,showall=False)
    savefig('/Users/rfinn/research/LocalClusters/SamplePlots/LCS_mosaic_rand_09_20.eps')   

    
