#!/usr/bin/env python
'''

useage

LCSPlotProfiles.py # runs for all clusters

'''



import pyfits
from LCScommon import *
from pylab import *
import os,sys
import mystuff as my
#from LCSReadmasterBaseWithProfileFits import *
from LCSReadmasterBaseNSA import *
from scipy.stats import scoreatpercentile
import numpy as np
import aplpy
from pyraf import iraf

def getv1v2(image,nhdr=1,pmin=.25,pmax=99.75):
    hdr=pyfits.open(image)
    im=hdr[nhdr].data.copy()
    try:
        v1,v2=scoreatpercentile(im,[pmin,pmax],limit=[-.2,20])
    except ValueError:
        try:
            v1,v2=scoreatpercentile(im,[pmin,pmax])#,limit=[0,15])
        except ValueError:
            v1=.01
            v2=2
    hdr.close()
    print image
    print 'v1,v2 = ',v1,v2
    return v1,v2

def transcoordspix2wcs(cimage,incoords):
    outcoords=incoords+'.radec'
    s='rm '+outcoords
    os.system(s)
    iraf.imcoords.wcsctran(image=cimage,input=incoords,output=outcoords,outwcs='world',inwcs='logical',verbose='no')
    in1=open(outcoords,'r')
    data=np.loadtxt(outcoords,unpack=True)#usecols=(0,1))
    xra,yra=data
    return xra,yra


class cluster(baseClusterNSA):

    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)
        self.readsnr24NSA()
        #self.selectFlag=self.ellipseflag
        self.readGalfitSersicResults()
        self.analyze_mips=self.On24ImageFlag & (self.snrse > snr24cut) & (self.n.SERSIC_TH50 > mipspixelscale)
        self.analyze_mips=self.On24ImageFlag & (self.snrse > snr24cut) & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag
    def readGalfitSersicResults(self):
        try:
            sersicparam_file=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_SDSS.fits'
            self.galfit=atpy.Table(sersicparam_file)
        except IOError:
            # this is a cheat for clusters that I haven't run sdss analysis on
            # probably will phase sdss analysis out anyway
            # b/c getting similar results from NSA image
            sersicparam_file=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_NSA.fits'
            self.galfit=atpy.Table(sersicparam_file)

        sersicparam24_file=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_24.fits'
        self.galfit24=atpy.Table(sersicparam24_file)
        sersicparamNSA_file=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_NSA.fits'
        self.galfitNSA=atpy.Table(sersicparamNSA_file)


    def readGalfitResults(self):
        self.galflag=zeros([len(self.ra),3],'i')
        self.galflag24=zeros([len(self.ra),3],'i')
        self.galflag_too_faint24=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.galflag_too_faint=zeros(len(self.ra),'i') # to track if 24um image is too faint for galfit fitting
        self.galflag_stop=zeros(len(self.ra),'i') # if fitting was stopped, either b/c too faint or results were not reasonable
        self.galflag_stop24=zeros(len(self.ra),'i') # 

        self.galfit_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/'
        self.galfit_sdssR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_sdssR90=zeros([len(self.ra),3],'f')  # R90 for 1-comp 2comp 3comp fits
        self.galfit_mipsR50=zeros([len(self.ra),3],'f')  # R50 for 1-comp 2comp 3comp fits
        self.galfit_mipsR90=zeros([len(self.ra),3],'f')  # R90 for 1-comp 2comp 3comp fits
        results_file=self.galfit_dir+self.prefix+'-galfitResults-sdss.dat'
        results_file24=self.galfit_dir+self.prefix+'-galfitResults-24.dat'

        infile=open(results_file24,'r')
        i=0
        for line in infile:
            t=line.split()
            i=int(t[0])
            index,On24ImageFlag,self.galflag24[i][0],self.galflag24[i][1],self.galflag24[i][2],self.galflag_stop24[i],self.galflag_too_faint24[i],self.galfit_mipsR50[i][0],self.galfit_mipsR50[i][1],self.galfit_mipsR50[i][2],self.galfit_mipsR90[i][0],self.galfit_mipsR90[i][1],self.galfit_mipsR90[i][2]=t
        infile.close()

        infile=open(results_file,'r')
        for line in infile:
            t=line.split()
            i=int(t[0])
            index,On24ImageFlag,self.galflag[i][0],self.galflag[i][1],self.galflag[i][2],self.galflag_stop[i],self.galflag_too_faint[i],self.galfit_sdssR50[i][0],self.galfit_sdssR50[i][1],self.galfit_sdssR50[i][2],self.galfit_sdssR90[i][0],self.galfit_sdssR90[i][1],self.galfit_sdssR90[i][2]=t

        infile.close()

        self.galfit_sdssR50=self.galfit_sdssR50*sdsspixelscale
        self.galfit_sdssR90=self.galfit_sdssR90*sdsspixelscale
        self.galfit_mipsR50=self.galfit_mipsR50*mipspixelscale
        self.galfit_mipsR90=self.galfit_mipsR90*mipspixelscale

        sersicparam_file=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_SDSS.fits'
        self.galfit=atpy.Table(sersicparam_file)
        sersicparam24_file=homedir+'research/LocalClusters/NSAmastertables/GalfitSersicResults/'+self.prefix+'_GalfitSersicParam_24.fits'
        self.galfit24=atpy.Table(sersicparam24_file)

    def run_plotprofiles(self,startindex=0):
        for i in range(startindex,len(self.ra)):
            if self.analyze_mips[i]: #On24ImageFlag[i] & (self.snr24[i] > snr24cut) & self.spiralflag[i]:
                self.plotprofilesconv(i)
                close('all')
    def plotprofiles(self,i):
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'
        working_dir24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        galname=self.prefix+'-'+str(self.n.NSAID[i])
        nrow=2
        ncol=3
        xticksize=10
        yticksize=10
        
        redshift=self.n.ZDIST#(self.supervopt[i]-self.clustervel)/self.clustersigma
        dr=self.drR200[i]
        mipssnr=self.snr24[i]
        agn=self.agn1[i]
        vminsdss=-1
        vmaxsdss=1
        vminsdssapl=-100
        vmaxsdssapl=400
        
        vmin24=-40
        vmax24=10.
        subplots_adjust(left=0.05, right=.95,bottom=.1,top=0.9,wspace=0.4,hspace=0.6)
        
        fig=figure(figsize=(15,9))
        
        clf()
        
        subcomp_image=working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits'
        subcomp_image24=working_dir24+self.prefix+'-'+str(self.n.NSAID[i])+'-24-1Comp-galfit-out.fits'

        fig=gcf()
        try:
            subf=aplpy.FITSFigure(subcomp_image,hdu=1,figure=fig,subplot=[.1,.55,.23,.35])
        except IOError:
            return
        subf.set_theme('publication')
        subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)
        #subf.show_colorscale(cmap='gist_heat')
        axratio=self.galfit.axisratio1[i]
        phi=self.galfit.pa1[i]
        subf.tick_labels.set_font(size='x-small')
        
        s='$'+galname+'$'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)
        #s='$ r-band$'
        #subf.add_label(.1,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        #title(s)
        s='AGN Flag = %i'%(self.agn1[i])
        subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')

        s='$M_r = %5.1f,\ log(M_*)=%5.2f$'%(self.sdssMr[i],log10(self.stellarmass[i]))
        subf.add_label(.15,.9,s,size='large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        
        
        
        subf=aplpy.FITSFigure(subcomp_image,hdu=2,figure=fig,subplot=[.38,.55,.23,.35])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)
        #subf.show_ellipses(self.ra[i],self.dec[i],self.galfit.re1[i]/3600.,self.galfit.re1[i]*axratio/3600.,angle=phi,edgecolor='r')
        #subf.show_ellipses(self.ra[i],self.dec[i],self.galfit.R90[i]/3600.,self.galfit.R90[i]*axratio/3600.,angle=phi,edgecolor='r')
        subf.tick_labels.set_font(size='x-small')

        s='$ Model $'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)


        # mag, Re, n, B/A, phi
        s='$ Sersic \ param: \ m_r= %5.1f,Re=%5.1f$'%(self.galfit.mag1[i],self.galfit.re1[i]*sdsspixelscale)
        subf.add_label(.88,.9,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        s='$n=%5.1f,B/A=%5.2f, PA=%5.1f $'%(self.galfit.nsersic1[i],self.galfit.axisratio1[i],self.galfit.pa1[i])
        subf.add_label(.88,.82,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        s='$ NSA \ param: \ m_r= %5.1f,Re=%5.1f$'%(22.5-2.5*log10(self.n.SERSICFLUX[i,4]),self.n.SERSIC_TH50[i])
        subf.add_label(.88,.18,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        s='$n=%5.1f,B/A=%5.2f, PA=%5.1f $'%(self.n.SERSIC_N[i],self.n.SERSIC_BA[i],self.n.SERSIC_PHI[i])
        subf.add_label(.88,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image,hdu=3,figure=fig,subplot=[.68,.55,.23,.35])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmaxsdss,vmax=-1*vminsdss)

        s='$ Residual $'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)

        s='$ Spiral \ Flag\ = \ '+str(self.spiralflag[i])+'\ (%5.2f)$'%(self.zoo.p_cs[i])
        subf.add_label(.5,.9,s,fontsize=12,relative=True,color='white')

        s='$ Ellip \ Flag\ = \ '+str(self.ellipticalflag[i])+'\ (%5.2f)$'%(self.zoo.p_el[i])
        subf.add_label(.5,.82,s,fontsize=12,relative=True,color='white')

        fig=gcf()
        try:
            subf=aplpy.FITSFigure(subcomp_image24,hdu=1,figure=fig,subplot=[.1,.1,.23,.35])
        except IOError:
            return
        subf.set_theme('publication')
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        s='$ dv/\sigma = %5.2f,\ dr/R_{200} = %5.2f$'%(self.dv[i],self.drR200[i])
        subf.add_label(.5,1.05,s,fontsize=12,relative=True)
        
        subf.tick_labels.set_font(size='x-small')
        s='SNR(24)=%5.1f, SFR=%5.2f'%(self.snr24[i],self.ce.SFR[i])
        subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')
        if self.HIflag[i]:
            s='$ log_{10}(M_{HI}) = %5.2f$'%(log10(self.HImass[i]))
        else:
            s='$No \ HI$'

        subf.add_label(.15,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        
        
        subf=aplpy.FITSFigure(subcomp_image24,hdu=2,figure=fig,subplot=[.38,.1,.23,.35])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        subf.tick_labels.set_font(size='x-small')
        if self.galfit24.numerical_error_flag24[i]:
            s='$ Sersic \ param**: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.mag1[i],self.galfit24.re1[i]*mipspixelscale)
        else:
            s='$ Sersic \ param: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.mag1[i],self.galfit24.re1[i]*mipspixelscale)
        subf.add_label(.9,.9,s,fontsize=12,relative=True,color='white',horizontalalignment='right',family='serif',weight='medium')

        s='$ n=%5.1f,B/A=%4.2f,PA=%5.1f $'%(self.galfit24.nsersic1[i],self.galfit24.axisratio1[i],self.galfit24.pa1[i])
        subf.add_label(.9,.82,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        #s='$ SE \ param: \ m= %5.1f,Re=%5.1f$'%((self.sex24.MAG_BEST[i]),self.sex24.FLUX_RADIUS1[i]*mipspixelscale)
        #subf.add_label(.88,.18,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        #s='$B/A=%5.2f, PA=%5.1f $'%(1-self.sex24.ELLIPTICITY[i],self.sex24.THETA_IMAGE[i])
        #subf.add_label(.88,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        # ratio of Re(24)/Re(r)
        s='$ R_e(24)/R_e(r)=%5.1f $'%(self.galfit24.re1[i]*mipspixelscale/self.n.SERSIC_TH50[i])
        subf.add_label(.95,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image24,hdu=3,figure=fig,subplot=[.68,.1,.23,.35])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale()#vmin=-1*vmax24,vmax=-1*vmin24)
        subf.tick_labels.set_font(size='x-small')

        figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+str(galname)+'-galfitResults.png'
        savefig(figname)

    def plotprofilesconv(self,i,showlabel=0):
        working_dir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'
        working_dir24=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        galname=self.prefix+'-'+str(self.n.NSAID[i])
        nrow=2
        ncol=3
        xticksize=10
        yticksize=10
        
        x1,x2,x3=(.1,.41,.72)
        dx=.24
        y1,y2,y3=(.05,.36,.67)
        dy=.26
        redshift=self.n.ZDIST#(self.supervopt[i]-self.clustervel)/self.clustersigma
        dr=self.drR200[i]
        mipssnr=self.snr24[i]
        agn=self.agn1[i]
        vminsdss=-1
        vmaxsdss=1
        vminsdssapl=-100
        vmaxsdssapl=400
        
        vmin24=-40
        vmax24=10.
        subplots_adjust(left=0.05, right=.95,bottom=.1,top=0.9,wspace=0.3,hspace=0.2)
        
        fig=figure(figsize=(11,10))
        
        clf()
        
        subcomp_image=working_dir+self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits'
        subcomp_image24=working_dir24+self.prefix+'-'+str(self.n.NSAID[i])+'-24-1Comp-galfit-out.fits'
        subcomp_image24=working_dir24+self.prefix+'-'+str(self.n.NSAID[i])+'-24-noconv-1Comp-galfit-out.fits'
        subcomp_image24_conv=working_dir24+self.prefix+'-'+str(self.n.NSAID[i])+'-24-conv-1Comp-galfit-out.fits'

        fig=gcf()
        try:
            subf=aplpy.FITSFigure(subcomp_image,hdu=1,figure=fig,subplot=[x1,y3,dx,dy])
        except IOError:
            return
        subf.set_theme('publication')
        v1,v2=getv1v2(subcomp_image,nhdr=3)
        subf.show_grayscale(vmin=v1,vmax=v2)#vmin=-1*vmaxsdss,vmax=-1*vminsdss)
        #subf.show_colorscale(cmap='gist_heat')
        axratio=self.galfit.axisratio1[i]
        phi=self.galfit.pa1[i]
        subf.tick_labels.set_font(size='x-small')
        
        s='$'+galname+'$'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)
        #s='$ r-band$'
        #subf.add_label(.1,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        #title(s)
        if showlabel:
            s='AGN Flag = %i'%(self.agn1[i])
            subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')

            s='$M_r = %5.1f,\ log(M_*)=%5.2f$'%(self.sdssMr[i],log10(self.stellarmass[i]))
            subf.add_label(.05,.9,s,size='large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')

        s='r-band'
        subf.add_label(.05,.1,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='black')
        
        
        
        subf=aplpy.FITSFigure(subcomp_image,hdu=2,figure=fig,subplot=[x2,y3,dx,dy])
        subf.set_theme('publication')
        subf.tick_labels.hide()

        subf.show_grayscale(vmin=v1,vmax=v2)
        #subf.show_ellipses(self.ra[i],self.dec[i],self.galfit.re1[i]/3600.,self.galfit.re1[i]*axratio/3600.,angle=phi,edgecolor='r')
        #subf.show_ellipses(self.ra[i],self.dec[i],self.galfit.R90[i]/3600.,self.galfit.R90[i]*axratio/3600.,angle=phi,edgecolor='r')
        subf.tick_labels.set_font(size='x-small')

        s='$ Model $'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)


        # mag, Re, n, B/A, phi
        if showlabel:
            s='$ Sersic \ param: \ m_r= %5.1f,Re=%5.1f$'%(self.galfit.mag1[i],self.galfit.re1[i]*sdsspixelscale)
            subf.add_label(.95,.9,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

            s='$n=%5.1f,B/A=%5.2f, PA=%5.1f $'%(self.galfit.nsersic1[i],self.galfit.axisratio1[i],self.galfit.pa1[i])
            subf.add_label(.95,.82,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

            s='$ NSA \ param: \ m_r= %5.1f,Re=%5.1f$'%(22.5-2.5*log10(self.n.SERSICFLUX[i,4]),self.n.SERSIC_TH50[i])
            subf.add_label(.95,.18,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        if showlabel:
            s='$n=%5.1f,B/A=%5.2f, PA=%5.1f $'%(self.n.SERSIC_N[i],self.n.SERSIC_BA[i],self.n.SERSIC_PHI[i])
            subf.add_label(.95,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image,hdu=3,figure=fig,subplot=[x3,y3,dx,dy])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale(vmin=v1,vmax=v2)#vmin=-1*vmaxsdss,vmax=-1*vminsdss)

        s='$ Residual $'
        subf.add_label(.5,1.05,s,fontsize=14,relative=True)

        if showlabel:
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
        subf.set_theme('publication')
        v1,v2=getv1v2(subcomp_image24,nhdr=3)
        subf.show_grayscale(vmin=v1,vmax=v2)
        s='$ dv/\sigma = %5.2f,\ dr/R_{200} = %5.2f$'%(self.dv[i],self.drR200[i])
        #subf.add_label(.5,1.05,s,fontsize=12,relative=True)
        
        subf.tick_labels.set_font(size='x-small')
        if showlabel:
            s='SNR(24)=%5.1f, SFR=%5.2f'%(self.snr24[i],self.ce.SFR[i])
            subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')
            if self.HIflag[i]:
                s='$ log_{10}(M_{HI}) = %5.2f$'%(log10(self.HImass[i]))
            else:
                s='$No \ HI$'

                subf.add_label(.05,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        
        s='24um'
        subf.add_label(.05,.1,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='black')
        

        subf=aplpy.FITSFigure(subcomp_image24,hdu=2,figure=fig,subplot=[x2,y2,dx,dy])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale(vmin=v1,vmax=v2)
        subf.tick_labels.set_font(size='x-small')
        if showlabel:
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
        subf.add_label(.95,.1,s,size='xx-large',relative=True,color='red',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image24,hdu=3,figure=fig,subplot=[x3,y2,dx,dy])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale(vmin=v1,vmax=v2)
        subf.tick_labels.set_font(size='x-small')
        s='$\chi^2nu= %6.3f$'%(self.galfit24.chi2nu[i])
        subf.add_label(.15,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')



        # third row - with convolution
        fig=gcf()
        try:
            subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=1,figure=fig,subplot=[x1,y1,dx,dy])
        except IOError:
            return
        subf.set_theme('publication')
        subf.show_grayscale(vmin=v1,vmax=v2)

        #subf.add_label(.5,1.05,s,fontsize=12,relative=True)
        
        subf.tick_labels.set_font(size='x-small')
        if showlabel:
            s='SNR(24)=%5.1f, SFR=%5.2f'%(self.snr24[i],self.ce.SFR[i])
            #subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='white')

            s='$ dv/\sigma = %5.2f,\ dr/R_{200} = %5.2f$'%(self.dv[i],self.drR200[i])
            subf.add_label(.05,.9,s,size='large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
        
        s='24um+conv'
        subf.add_label(.05,.1,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='black')

        subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=2,figure=fig,subplot=[x2,y1,dx,dy])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale(vmin=v1,vmax=v2)
        subf.tick_labels.set_font(size='x-small')
        if showlabel:
            if self.galfit24.cnumerical_error_flag24[i]:
                s='$ Sersic \ param**: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.cmag1[i],self.galfit24.cre1[i]*mipspixelscale)
            else:
                s='$ Sersic \ param: \ m=%5.1f,Re=%5.1f$'%(self.galfit24.cmag1[i],self.galfit24.cre1[i]*mipspixelscale)
            subf.add_label(.95,.9,s,fontsize=12,relative=True,color='white',horizontalalignment='right',family='serif',weight='medium')

            s='$ n=%5.1f,B/A=%4.2f,PA=%5.1f $'%(self.galfit24.cnsersic1[i],self.galfit24.caxisratio1[i],self.galfit24.cpa1[i])
            subf.add_label(.95,.82,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        # ratio of Re(24)/Re(r)
        s='$ R_e(24)/R_e(r)=%5.1f $'%(self.galfit24.cre1[i]*mipspixelscale/self.n.SERSIC_TH50[i])
        subf.add_label(.95,.1,s,size='xx-large',relative=True,color='red',horizontalalignment='right')

        #s='$ SE \ param: \ m= %5.1f,Re=%5.1f$'%((self.sex24.MAG_BEST[i]),self.sex24.FLUX_RADIUS1[i]*mipspixelscale)
        #subf.add_label(.88,.18,s,fontsize=12,relative=True,color='white',horizontalalignment='right')

        #s='$B/A=%5.2f, PA=%5.1f $'%(1-self.sex24.ELLIPTICITY[i],self.sex24.THETA_IMAGE[i])
        #subf.add_label(.88,.1,s,fontsize=12,relative=True,color='white',horizontalalignment='right')


        subf=aplpy.FITSFigure(subcomp_image24_conv,hdu=3,figure=fig,subplot=[x3,y1,dx,dy])
        subf.set_theme('publication')
        subf.tick_labels.hide()
        subf.show_grayscale(vmin=v1,vmax=v2)
        subf.tick_labels.set_font(size='x-small')

        s='$ \chi^2nu= %6.3f$'%(self.galfit24.cchi2nu[i])
        subf.add_label(.15,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')

        if showlabel:
            figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+str(galname)+'-galfitResults-conv.png'
        else:
            figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+str(galname)+'-galfitResults-conv-nolabels.eps'
        
        savefig(figname)


mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

#mkw11=cluster('MKW11')

#mkw11.run_plotprofiles(startindex=0)

#coma=cluster('Coma')
#coma.run_plotprofiles(startindex=1063)

#a2052=cluster('A2052')
#a2052.run_plotprofiles()

#a2063=cluster('A2063')
#a2063.run_plotprofiles()
#coma.plotprofiles()
#herc.plotprofiles()
#a1367.plotprofiles()

try:
    cl=cluster(sys.argv[1])
    cl.run_plotprofiles()
except:
    print 'appears like you are running in ipython'
    print 'try: \n cl=cluster("MKW11") \n cl.run_plotprofiles() \n or \n cl.plotprofiles(i)'
