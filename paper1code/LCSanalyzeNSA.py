#!/usr/bin/env python
import pyfits
from LCScommon import *
from pylab import *
import os
import mystuff as my
#import aplpy
import ds9
from astropy.io import ascii
import astrofuncs
#from LCSReadmasterBaseWithProfileFits import *
from LCSReadmasterBaseNSA import *
import matplotlib.cm as cm
#import chary_elbaz_24um as chary
#from scipy.stats.binom import cdf
Lsol=3.826e33#normalize by solar luminosity
bellconv=9.8e-11#converts Lir (in L_sun) to SFR/yr
bellconv=4.5e-44#Kenn 98 conversion fro erg/s to SFR/yr

truncation_fraction=0.5
usecomaflag=1
minstellarmass=1.e10
#these correpond to area w/more uniform covereage
MKW824um=array([220.16377,3.4883817,1.3137727,2.5,12.7456],'f')
MKW1124um=array([202.36305,11.746882,1.2454248,2.9,206.4],'f')
NGC24um=array([244.30994,34.933704,1.2865442,2.5,321.317],'f')

colors=['r','b','c','g','m','y','k','r','0.5']
shapes=['o','x','p','d','s','^','>','<','v']
clusterlist=['Coma','A1367','A2052','A2063']
fieldcolor='0.7'
#usefwhm24=1
fieldColor='0.7'
Remin=.1

snr24cut=4
minsize_kpc=1.3 # one mips pixel at distance of hercules


mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

figuredir=homedir+'Dropbox/Research/LocalClusters/SamplePlots/'
figuredir=homedir+'research/LocalClusters/SamplePlots/'


def xaxissfrlog(ax1):#when plotting log(lir)
	locs, labels = xticks()
	ax1=gca()
	ticks=ax1.xaxis.get_major_ticks()
	a=ticks[0]
	xfontsize=a.label1.get_fontsize()
	y1, y2=ax1.get_ylim()
	xmin,xmax,ymin,ymax=ax1.axis()
	ylab=y2+0.03*(y2-y1)
	for x in locs:
	    if x < xmin:
		continue
	    if x > xmax:
		continue
	    x2=getsfrfromlir(10.**x)
	    s='$%3.2e$'%(x2)
	    text(x,ylab,s,fontsize=xfontsize,horizontalalignment='center')#,transform=ax1.transAxes)

def getsfrfromlir(lir):
	sfr=lir*bellconv*Lsol
	return sfr


def percentdiff(a,b):
    pdiff=abs(a-b)/((a+b)/2.)*100.
    return pdiff
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


def plotcolormaglines():
    xl=arange(8.5,13,.5)
    yl=.17*(xl-10)+5.
    xl=arange(-22,-14,.5)
    yl=1.897-0.175*xl # from Wyder et al 2007
    plot(xl,yl,'k-')
    #plot(xl,yl-.5,'k--')
    #plot(xl,yl+.5,'k--')
    yb=2.39+0.075*(xl+20)-.808*tanh((xl+20.32)/1.81)
    yb=1.73-0.17*xl-2.
    plot(xl,yb,'b-')
    yb=1.73-0.17*xl-.5
    plot(xl,yb,'r-')


class cluster(baseClusterNSA):
    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)
        mypath=os.getcwd()
        if mypath.find('Users') > -1:
            print "Running on Rose's mac pro"
            infile='/Users/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'
        elif mypath.find('home') > -1:
            print "Running on coma"
            infile=homedir+'research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'

        self.mipssnrflag = self.mipssnr > 6.
        try:
            self.readsnr24NSA()
        except:
            print self.prefix,": couldn't read SNR24 file"
        #try:
        #    self.readGalfitSersicResults()
        #except:
        #    print self.prefix,": couln't read galfit sersic results"
        #try:
        #    self.readGalfitResults()
        #except:
        #    print self.prefix,": couldn't read galfit results"
        #self.size24=self.sex24.FWHM_DEG*3600.
        self.size24=self.sex24.FLUX_RADIUS1*mipspixelscale
        self.sizeratio=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50#(self.gim2d.Rhlr_2/self.gim2d.Scale_2)
        self.mingalaxymass=5.e9
        self.rmagcut=20.+5.*log10(self.cdMpc/(clusterbiweightcenter['Hercules']/H0))
        self.rmag=22.5-2.5*log10(self.n.SERSICFLUX[:,2])
        try:
            #self.galfitflag=self.On24ImageFlag & (self.snr24>snr24cut) & ~self.galfit24.numerical_error_flag24   & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag& ~self.galfit24.cnumerical_error_flag24 #& ~self.agnflag #& (self.ce.LIR > 5.1e8) & (self.stellarmass > minstellarmass)#& (self.rmag < self.rmagcut) #
            self.galfitflag=self.On24ImageFlag & (self.snrse>snr24cut) &  (self.n.SERSIC_TH50 > 2.*mipspixelscale) & self.spiralflag & (self.ce.LIR > 5.1e8) & (self.stellarmass > minstellarmass)# & ~self.agnflag & (self.rmag < self.rmagcut) #& ~self.galfit24.numerical_error_flag24
            
                                                                                                             
            self.galfitsample=self.On24ImageFlag & (self.snrse>snr24cut) & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag& ~self.agnflag  & (self.ce.LIR > 5.1e8) & (self.stellarmass > minstellarmass)

        except:
            print 'no galfit data yet?  get moving!'
        #self.sampleflag= self.On24ImageFlag & self.spiralflag & (self.ce.LIR > 5.1e8) & (self.nsamag[:,4] < (19.+5.*log10(self.cz/clusterz['Hercules']))) & (self.n.SERSIC_TH50*self.AngDistanceCl > 2.*1.8)  & (self.snrse>snr24cut)

        
        # exclude objects that had issues w/galfit fit, like nearby neighbor
        gal_ids=visual_cut[self.prefix]
        for id in gal_ids:
            try:
                if self.spiralflag[self.nsadict[id]]:
                    print 'Check out fit for this spiral ',self.prefix,' NSAID=',id
                self.galfitflag[self.nsadict[id]]=0
            except:
                print 'ERROR: problem resetting galfitflag with ',self.prefix,' NSAID=',id

        self.member=(self.dvflag) & (self.drR200 < 1.3)
        self.nearfield=(self.dvflag) & (self.drR200 > 1.3) & (self.drR200 < 2.)
        self.field=((self.dvflag) & (self.drR200 > 2.)) | ~self.dvflag
        #self.member=self.dvflag
        self.sample24flag=self.galfitflag & self.spiralflag# self.On24ImageFlag & (self.snr24>3) & ~self.agnflag & (self.n.SERSIC_TH50 > Remin) & self.spiralflag# & (log10(self.stellarmass) > 9.5) & (log10(self.stellarmass) < 12)
        self.blueclustersample=self.member & self.blueflag & self.sample24flag
        self.bluefieldsample=~self.member & self.blueflag & self.sample24flag
        self.greenclustersample=self.member & self.greenflag & self.sample24flag
        self.greenfieldsample=~self.member & self.greenflag & self.sample24flag
        self.redclustersample=self.member & self.redflag & self.sample24flag
        self.redfieldsample=~self.member & self.redflag & self.sample24flag
        self.varlookup={'stellarmass':log10(self.stellarmass),'Re':self.n.SERSIC_TH50,'R24':self.galfit24.re1*mipspixelscale,'NUV':self.n.ABSMAG[:,1],'r':self.n.ABSMAG[:,4],'m24':self.sex24.MAG_BEST,'redshift':self.n.ZDIST,'NUVr':(self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]),'NUV24':(self.n.ABSMAG[:,1]-self.sex24.MAG_BEST),'24mass':(self.sex24.MAG_BEST-log10(self.stellarmass)),'ratioR':self.sex24.FLUX_RADIUS1*mipspixelscale/self.n.SERSIC_TH50,'BT':self.gim2d.B_T_r}
        self.sb_obs=self.galfit24.cmag1 + 2.5*log10(pi*((self.galfit24.cre1*mipspixelscale)**2))#*self.s.caxisratio1)

        self.sampleflag= self.On24ImageFlag & self.spiralflag & (self.galfit24.cnumerical_error_flag24 < .1) & (~self.agnflag) 

        self.sb_obs=self.galfit24.cmag1 + 2.5*log10(pi*((self.galfit24.cre1*mipspixelscale)**2)*self.galfit24.caxisratio1)
        self.DA=zeros(len(self.n.SERSIC_TH50))
        for i in range(len(self.DA)):
            if self.membflag[i]:
                self.DA[i] = self.n.SERSIC_TH50[i]*astrofuncs.DA(self.biweightvel/3.e5,H0/100.)
            else:
                self.DA[i] = self.n.SERSIC_TH50[i]*astrofuncs.DA(self.n.ZDIST[i],H0/100.)

        self.sampleflag =  (self.galfit24.cmag1 > .1)  & (self.n.SERSIC_TH50*self.DA > minsize_kpc) &(self.spiralflag) & (self.galfit24.cnumerical_error_flag24 < .5) & (self.ce.LIR_ZDIST > 5.e8) & (self.sb_obs < 20.5) #& (self.jmass.MSTAR_50 > minmass) #& (self.s.SERSIC_BA > 0.2)#& (self.s.SERSIC_TH50 < 15.)                                                                                                     #self.sb_obs=self.s.cmag1 + 2.5*log10(pi*((self.s.cre1*mipspixelscale)**2))#*self.s.caxisratio1)
    def nozoo(self):
        try:
            d.set('frame delete all')
        except:
            d=ds9.ds9()

        names=self.n.NSAID[~self.zoo.match_flag & self.On24ImageFlag]
        index=[]
        for n in names:
            index.append(self.nsadict[n])
        for j in range(len(names)):
            i=index[j]
            print self.prefix, self.n.NSAID[i]
            s='file new '+homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'+self.prefix+'-'+str(self.n.NSAID[i])+'-parent-r.fits'
            d.set(s)
            d.set('zoom to fit')
            t=raw_input('hit any key to continue \n')
    def readsnr24NSA(self):
        infile=homedir+'research/LocalClusters/NSAmastertables/SNR24/'+self.prefix+'_snr24NSA.dat'
        snrdat=atpy.Table(infile,type='ascii')
        self.f24NSA=snrdat['col1']
        self.f24NSAerr=snrdat['col2']
        self.snr24=snrdat['col3']

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


    def reviewimages(self):
        d=ds9.ds9()
        imagedir=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/24um/'
        imagedirNSA=homedir+'research/LocalClusters/GalfitAnalysis/'+self.prefix+'/NSA/'

        for i in range(len(self.galfit24.re1)):
            if self.On24ImageFlag[i] & (self.snr24[i] > snr24cut):
                print 'galaxy number ',i
                print '-------------------------'
                print 'var     NSA      24'
                print '-------------------------'
                print 'n   = %5.1f, %5.1f+/-%3.2f'%(self.n.SERSIC_N[i],self.galfit24.nsersic1[i],self.galfit24.nsersic1err[i])
                print 'Re  = %5.1f, %5.1f+/-%3.2f'%(self.n.SERSIC_TH50[i]/mipspixelscale,self.galfit24.re1[i],self.galfit24.re1err[i])
                print 'mag = %5.1f, %5.1f+/-%3.2f'%(22.5-2.5*log10(self.n.NMGY[i,4]) + self.n.EXTINCTION[i,4],self.galfit24.mag1[i],self.galfit24.mag1err[i])
                print 'B/A = %5.1f, %5.1f+/-%3.2f'%(self.n.SERSIC_BA[i],self.galfit24.axisratio1[i],self.galfit24.axisratio1err[i])
                print 'phi = %5.1f, %5.1f+/-%3.2f'%(self.n.SERSIC_PHI[i],self.galfit24.pa1[i],self.galfit24.pa1err[i])
                print 'snr = %5.1f'%(self.snr24[i])
                print 'nerr = ',self.galfit24.numerical_error_flag24[i]

                
                d.set('frame delete all')
                
                s='file new '+imagedir+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout24.fits'
                d.set(s)
                s='file new '+imagedir+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-cutout-unc24.fits'
                d.set(s)
                s='file new '+imagedir+self.prefix+'-'+str(self.n.NSAID[i])+'-galfit-mask24.fits'
                d.set(s)
                s='file new '+imagedir+self.prefix+'-'+str(self.n.NSAID[i])+'-24-1Comp-galfit-out.fits[2]'
                d.set(s)
                s='file new '+imagedir+self.prefix+'-'+str(self.n.NSAID[i])+'-24-1Comp-galfit-out.fits[3]'
                d.set(s)
                try:
                    
                    s='file new '+imagedirNSA+self.prefix+'-'+str(self.n.NSAID[i])+'-1Comp-galfit-out.fits[1]'
                    d.set(s)
                except:
                    print "WARNING:  couldn't load ",imagedir+self.prefix+'-'+str(self.n.NSAID[i])+'-24-1Comp-subcomps.fits[3]'
                d.set('tile')
                for k in range(1,7):
                    s='frame '+str(k)
                    d.set(s)
                    d.set('scale log')
                    d.set('zoom to fit')
                string=raw_input('hit any key when ready to continue (q to quit) \n')
                if string.find('q') > -1:
                    quitflag=1
                    break

    def plotgimvsNSA(self,plotsingle=1):
        if plotsingle:
            figure()
        keepflag=self.gim2d.matchflag & self.galfitflag

        subplot(2,1,1)
        plot(self.n.SERSIC_TH50[keepflag],self.gim2d.Rhlr_2[keepflag]/self.gim2d.Scale_2[keepflag],'b.')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k--')
        plot(xl,.8*xl,'k--')
        xlabel('NSA SERSIC_TH50 (arcsec)')
        ylabel('GIM2D R50 (r-band) (arcsec)')
        gca().set_xscale('log')
        gca().set_yscale('log')
        subplot(2,1,2)
        plot(self.n.SERSIC_N[keepflag],self.gim2d.ng[keepflag],'b.')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k--')
        xlabel('NSA SERSIC_N')
        ylabel('GIM2D ng (r-band)')

    def plotGalfitR90(self):
        
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag & keep24flag)
        figure()
        x=self.galfitNSA.R90[keepflag]
        y=self.galfit24.R90[keepflag]
        plot(x,y,'ko',label='1 Comp Fit')
        legend(loc='upper right', numpoints=1)
        xlabel('SDSS R90 (arcsec)')
        ylabel('MIPS R90 (arcsec)')
        xl=arange(51)
        plot(xl,xl,'k--')
    def plotGalfitR90vsNSA(self):
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag & keep24flag)
        figure(figsize=(12,5))
        subplot(1,2,1)
        y=self.galfit24.R50[keepflag]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'bo',label='R50')
        y=self.galfit24.R90[keepflag]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        #plot(x,y,'ro',label='R90')
        y=self.galfit24.re1[keepflag]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'ko',label='Re')

        y=self.galfit24.R50[keepflag & self.spiralflag]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag & self.spiralflag]
        plot(x,y,'g*',mec='g',label='Zoo Spiral',markersize=10)
        y=self.galfit24.R50[keepflag & self.galfit24.numerical_error_flag24]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag & self.galfit24.numerical_error_flag24]
        plot(x,y,'rx',mec='r',label='Num Err',markersize=10)
        y=self.galfit24.re1[keepflag & self.galfit24.numerical_error_flag24]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag & self.galfit24.numerical_error_flag24]
        plot(x,y,'rx',mec='r',label='_nolegend_',markersize=10)

        ax=gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        xl=arange(100)
        plot(xl,xl,'k--')
        xlabel('NSA SERSIC_TH50 (arcsec)')
        ylabel('GALFIT MIPS Radius (arcsec)')
        legend(numpoints=1, loc='upper left')
        axis([1, 300,1,600])
        subplot(1,2,2)
        y=self.galfitNSA.R50[keepflag]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'bo',label='R50')
        y=self.galfitNSA.R90[(keepflag)]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        #plot(x,y,'ro',label='R90')
        y=self.galfitNSA.re1[keepflag]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'ko',label='Re')

        y=self.galfitNSA.R50[keepflag & self.spiralflag]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag & self.spiralflag]
        plot(x,y,'g*',mec='g',label='Zoo Spiral',markersize=10)

        y=self.galfitNSA.R50[keepflag & self.galfitNSA.numerical_error_flag]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag & self.galfitNSA.numerical_error_flag]
        plot(x,y,'rx',mec='r',label='Num Err',markersize=10)


        ax=gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        #legend(loc='upper left', numpoints=1)
        xlabel('NSA SERSIC_TH50 (arcsec)')
        ylabel('GALFIT SDSS Radius (arcsec)')
        xl=arange(100)
        plot(xl,xl,'k--')
        legend(numpoints=1, loc='upper left')
        axis([1, 300,1,600])

    def plotGalfitDASvsNSA(self): # compare results from using DAS image w/NSA images
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag)# & keep24flag)
        figure(figsize=(15,8))
        subplots_adjust(wspace=.25,hspace=.35)
        subplot(2,3,4)
        y=self.galfit.re1[keepflag]*sdsspixelscale
        x=self.galfitNSA.re1[keepflag]*sdsspixelscale
        plot(x,y,'ro',label='Re')
        nerror_flag=self.galfit.numerical_error_flag | self.galfitNSA.numerical_error_flag
        y=self.galfit.re1[keepflag & nerror_flag]
        x=self.galfitNSA.re1[keepflag & nerror_flag]
        plot(x,y,'b^',label='Num Err',markersize=8)

        #errorbar(x,y,yerr=self.galfit.re1err[keepflag]*sdsspixelscale,fmt=None)
        gca().set_yscale('log')
        gca().set_xscale('log')

        xlabel('Galfit Re from NSA Images (arcsec)')
        ylabel('Galfit Re from DAS Images (arcsec)')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')

        subplot(2,3,2)
        y=self.galfit.nsersic1[keepflag]
        x=self.galfitNSA.nsersic1[keepflag]
        plot(x,y,'ro',label='Sersic Index')

        y=self.galfit.nsersic1[keepflag & (self.zoo.p_cs > 0.7)]
        x=self.galfitNSA.nsersic1[keepflag & (self.zoo.p_cs > 0.7)]
        plot(x,y,'g*',label='Zoo Spiral',markersize=8)

        nerror_flag=self.galfit.numerical_error_flag | self.galfitNSA.numerical_error_flag
        y=self.galfit.nsersic1[keepflag & nerror_flag]
        x=self.galfitNSA.nsersic1[keepflag & nerror_flag]
        plot(x,y,'b^',label='Num Err',markersize=8)
#errorbar(x,y,yerr=self.galfit.nsersic1err[keepflag],fmt=None,color='r')

        legend(loc='upper left',numpoints=1)
        
        xlabel('NSA SERSIC_N')
        ylabel('Galfit N')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')
        axhline(y=1,ls='--',color='k')
        axhline(y=4,ls='--',color='k')

        subplot(2,3,3)
        y=self.galfit.pa1[keepflag]
        x=self.galfitNSA.pa1[keepflag]
        plot(x,y,'ro',label='PA')
        nerror_flag=self.galfit.numerical_error_flag | self.galfitNSA.numerical_error_flag
        y=self.galfit.pa1[keepflag & nerror_flag]
        x=self.galfitNSA.pa1[keepflag & nerror_flag]
        plot(x,y,'b^',label='Num Err',markersize=8)

        legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.pa1err[keepflag],fmt=None,color='r')

        xlabel('NSA SERSIC_PHI')
        ylabel('Galfit PA + 90')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')

        subplot(2,3,1)
        y=self.galfit.axisratio1[keepflag]
        x=self.galfitNSA.axisratio1[keepflag]
        plot(x,y,'ro',label='B/A')
        nerror_flag=self.galfit.numerical_error_flag | self.galfitNSA.numerical_error_flag
        y=self.galfit.axisratio1[keepflag & nerror_flag]
        x=self.galfitNSA.axisratio1[keepflag & nerror_flag]
        plot(x,y,'b^',label='Num Err',markersize=8)

        legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.axisratio1err[keepflag],fmt=None,color='r')
        xlabel('NSA SERSIC_BA')
        ylabel('GALFIT B/A')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')


        subplot(2,3,5)
        y=self.galfit.R50[keepflag]
        x=self.galfitNSA.R50[keepflag]
        plot(x,y,'ro',label='R50 Galfit')
        nerror_flag=self.galfit.numerical_error_flag | self.galfitNSA.numerical_error_flag
        y=self.galfit.R50[keepflag & nerror_flag]
        x=self.galfitNSA.R50[keepflag & nerror_flag]
        plot(x,y,'b^',label='Num Err',markersize=8)

        legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.axisratio1err[keepflag],fmt=None,color='r')
        xlabel('Galfit R50 from NSA Images')
        ylabel('Galfit R50 from DAS Images')
        ax=gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')

        subplot(2,3,6)
        y=self.galfit.R90[keepflag]
        x=self.galfitNSA.R90[keepflag]
        plot(x,y,'ro',label='R90 Galfit')
        nerror_flag=self.galfit.numerical_error_flag | self.galfitNSA.numerical_error_flag
        y=self.galfit.R90[keepflag & nerror_flag]
        x=self.galfitNSA.R90[keepflag & nerror_flag]
        plot(x,y,'b^',label='Num Err',markersize=8)

        legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.axisratio1err[keepflag],fmt=None,color='r')
        xlabel('Galfit R90 from NSA Images')
        ylabel('Galfit R90 from DAS Images')
        ax=gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')

#legend(numpoints=1, loc='upper left')


    def plotGalfitvsNSA(self):
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag)# & keep24flag)
        figure(figsize=(10,10))
        subplot(2,2,1)
        y=self.galfit.re1[keepflag]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        #plot(x,y,'ro',label='Re DAS image')
        y=self.galfitNSA.re1[keepflag]*sdsspixelscale
        plot(x,y,'ko',label='All')

        y=self.galfit.re1[keepflag & (self.ellipticalflag)]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=self.galfit.re1[keepflag & (self.spiralflag)]*sdsspixelscale
        x=self.n.SERSIC_TH50[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)

        legend(loc='upper left',numpoints=1)


        #errorbar(x,y,yerr=self.galfit.re1err[keepflag]*sdsspixelscale,fmt=None)
        gca().set_yscale('log')
        gca().set_xscale('log')
        ax=gca()
        text(1.1,1.1,self.prefix+' r-band Results',fontsize=18,horizontalalignment='center',transform=ax.transAxes)

        xlabel('NSA SERSIC_TH50 (arcsec)')
        ylabel('Galfit Re (arcsec)')
        axis([.2,250,2.e-3,3.e5])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')

        subplot(2,2,2)
        x=self.n.SERSIC_N[keepflag]
        y=self.galfitNSA.nsersic1[keepflag]
        plot(x,y,'ko',label='All')

        y=self.galfitNSA.nsersic1[keepflag & (self.ellipticalflag)]
        x=self.n.SERSIC_N[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=self.galfitNSA.nsersic1[keepflag & (self.spiralflag)]
        x=self.n.SERSIC_N[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)


        legend(loc='upper left',numpoints=1)
        
        xlabel('NSA SERSIC_N')
        ylabel('Galfit N')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')
        axhline(y=1,ls='--',color='k')
        axhline(y=4,ls='--',color='k')
        gca().set_yscale('log')
        subplot(2,2,3)
        y=self.galfitNSA.pa1[keepflag]
        x=self.n.SERSIC_PHI[keepflag]
        plot(x,y,'ko',label='All')

        y=self.galfitNSA.pa1[keepflag & (self.ellipticalflag)]
        x=self.n.SERSIC_PHI[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=self.galfitNSA.pa1[keepflag & (self.spiralflag)]
        x=self.n.SERSIC_PHI[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)

        legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.pa1err[keepflag],fmt=None,color='r')

        xlabel('NSA SERSIC_PHI')
        ylabel('Galfit PA')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')

        subplot(2,2,4)
        y=self.galfit.axisratio1[keepflag]
        x=self.n.SERSIC_BA[keepflag]
        plot(x,y,'ko',label='All')

        y=self.galfitNSA.axisratio1[keepflag & (self.ellipticalflag)]
        x=self.n.SERSIC_BA[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=self.galfitNSA.axisratio1[keepflag & (self.spiralflag)]
        x=self.n.SERSIC_BA[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)

        legend(loc='lower right',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.axisratio1err[keepflag],fmt=None,color='r')
        xlabel('NSA SERSIC_BA')
        ylabel('GALFIT B/A')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')

        #legend(numpoints=1, loc='upper left')

    def plotGalfit24vsNSA(self,convflag=1,plotsingle=1):
        #keep24flag=self.snr24se > 5
        #keepflag=(self.On24ImageFlag & keep24flag) & (self.n.SERSIC_TH50 > mipspixelscale )& ~self.galfit24.numerical_error_flag24
        keepflag=self.galfitflag
        print 'number of galaxies that meet analysis criteria = ',sum(keepflag)
        if plotsingle:
            figure(figsize=(8,6))
            subplots_adjust(wspace=.25,hspace=.35)
        subplot(2,2,1)
        if convflag:
            y=self.galfit24.cre1[keepflag]*mipspixelscale
        else:
            y=self.galfit24.re1[keepflag]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'ko',label='All')

        if convflag:
            y=self.galfit24.cre1[keepflag & (self.ellipticalflag)]*mipspixelscale
        else:
            y=self.galfit24.re1[keepflag & (self.ellipticalflag)]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        if convflag:
            y=self.galfit24.cre1[keepflag & (self.spiralflag)]*mipspixelscale
        else:
            y=self.galfit24.re1[keepflag & (self.spiralflag)]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)

        if plotsingle:
            legend(loc='upper left',numpoints=1)
        axis([0,20,0,20])

        #errorbar(x,y,yerr=self.galfit.re1err[keepflag]*sdsspixelscale,fmt=None)
        #gca().set_yscale('log')
        #gca().set_xscale('log')

        axhline(y=7.1,ls='--',color='k',label='res')
        ax=gca()
        text(90,7.5,r'$1.22 \lambda/D$',horizontalalignment='right')
        xlabel('NSA SERSIC_TH50 (arcsec)')
        ylabel('Galfit MIPS Re (arcsec)')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')
        ax=gca()
        if plotsingle:
            text(1.1,1.1,self.prefix+' 24um Results',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
        subplot(2,2,2)
        if convflag:
            y=self.galfit24.cnsersic1[keepflag]
        else:
            y=self.galfit24.nsersic1[keepflag]
        x=self.n.SERSIC_N[keepflag]
        plot(x,y,'ko',label='All')

        if convflag:
            y=self.galfit24.cnsersic1[keepflag & (self.ellipticalflag)]
        else:
            y=self.galfit24.nsersic1[keepflag & (self.ellipticalflag)]
        x=self.n.SERSIC_N[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        if convflag:
            y=self.galfit24.cnsersic1[keepflag & (self.spiralflag)]
        else:
            y=self.galfit24.nsersic1[keepflag & (self.spiralflag)]
        x=self.n.SERSIC_N[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)
#errorbar(x,y,yerr=self.galfit.nsersic1err[keepflag],fmt=None,color='r')

        if plotsingle:
            legend(loc='upper left',numpoints=1)
        xlabel('NSA SERSIC_N')
        ylabel('Galfit MIPS N')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')
        axhline(y=1,ls='--',color='k')
        axhline(y=4,ls='--',color='k')
        axis([-.05,6.2,-.05,6.2])
        subplot(2,2,3)
        if convflag:
            y=self.galfit24.cmag1[keepflag]
        else:
            y=self.galfit24.mag1[keepflag]
            #x=self.n.
        self.rmag=22.5-2.5*log10(self.n.SERSICFLUX[:,2])
        x=self.rmag[keepflag]
        #       if convflag:
        #   y=self.galfit24.cpa1[keepflag]
        #else:
        #    y=self.galfit24.pa1[keepflag]
        #x=self.n.SERSIC_PHI[keepflag]
        plot(x,y,'ko',label='All')

        #if convflag:
        #    y=self.galfit24.cpa1[keepflag & (self.ellipticalflag)]
        #else:
        #    y=self.galfit24.pa1[keepflag & (self.ellipticalflag)]
        #x=self.n.SERSIC_PHI[keepflag & (self.ellipticalflag)]
        #plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        #if convflag:
        #    y=self.galfit24.cpa1[keepflag & (self.spiralflag)]
        #else:
        #    y=self.galfit24.pa1[keepflag & (self.spiralflag)]
        #x=self.n.SERSIC_PHI[keepflag & (self.spiralflag)]
        #plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)

        axis([13.,20.,10.,17.])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl-3,'k:',label='r - 3')

        if plotsingle:
            legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.pa1err[keepflag],fmt=None,color='r')

        #xlabel('NSA SERSIC_PHI')
        #ylabel('Galfit MIPS PA')

        xlabel('NSA r mag')
        ylabel('Galfit MIPS mag')

        subplot(2,2,4)
        if convflag:
            yall=self.galfit24.caxisratio1
        else:
            yall=self.galfit24.axisratio1
        y=yall[keepflag]
        x=self.n.SERSIC_BA[keepflag]
        plot(x,y,'ko',label='All')

        y=yall[keepflag & (self.ellipticalflag)]
        x=self.n.SERSIC_BA[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=yall[keepflag & (self.spiralflag)]
        x=self.n.SERSIC_BA[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)
        if plotsingle:
            legend(loc='lower right',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.axisratio1err[keepflag],fmt=None,color='r')
        axis([0,1,0,1])
        xlabel('NSA SERSIC_BA')
        ylabel('GALFIT MIPS B/A')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')

        #legend(numpoints=1, loc='upper left')

    def plotsex24vsNSA(self):
        keep24flag=self.snr24 > 3
        keepflag=(self.On24ImageFlag & keep24flag)
        figure(figsize=(10,10))
        subplots_adjust(wspace=.25,hspace=.35)
        subplot(2,2,1)
        y=self.sex24.FLUX_RADIUS1[keepflag]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'ko',label='Re')

        y=self.sex24.FLUX_RADIUS1[keepflag & (self.ellipticalflag)]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=self.sex24.FLUX_RADIUS1[keepflag & (self.spiralflag)]*mipspixelscale
        x=self.n.SERSIC_TH50[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)


        legend(loc='upper left',numpoints=1)
        axis([.5,300,2,50])

        #errorbar(x,y,yerr=self.galfit.re1err[keepflag]*sdsspixelscale,fmt=None)
        gca().set_yscale('log')
        gca().set_xscale('log')
        axhline(y=7.1,ls='--',color='k',label='res')
        text(90,7.5,r'$1.22 \lambda/D$',horizontalalignment='center')
        xlabel('NSA SERSIC_TH50 (arcsec)')
        ylabel('SE MIPS Re (arcsec)')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')
        ax=gca()
        text(1.1,1.1,self.prefix+' 24um Results',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
        subplot(2,2,2)
        x=self.n.ABSMAG[:,4]
        y=self.sex24.MAG_BEST
        plot(x[keepflag],y[keepflag],'ko')
        subplot(2,2,3)
        y=self.sex24.THETA_IMAGE[keepflag]
        x=self.n.SERSIC_PHI[keepflag]
        plot(x,y,'ko',label='All')

        y=self.sex24.THETA_IMAGE[keepflag & (self.ellipticalflag)]
        x=self.n.SERSIC_PHI[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=self.sex24.THETA_IMAGE[keepflag & (self.spiralflag)]
        x=self.n.SERSIC_PHI[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)

        legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.pa1err[keepflag],fmt=None,color='r')

        xlabel('NSA SERSIC_PHI')
        ylabel('SE MIPS PA')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')

        subplot(2,2,4)
        yall=1-self.sex24.ELLIPTICITY
        y=yall[keepflag]

        x=self.n.SERSIC_BA[keepflag]
        plot(x,y,'ko',label='All')

        y=yall[keepflag & (self.ellipticalflag)]
        x=self.n.SERSIC_BA[keepflag & (self.ellipticalflag)]
        plot(x,y,'rs',mec='k',label='Zoo E',markersize=5)

        y=yall[keepflag & (self.spiralflag)]
        x=self.n.SERSIC_BA[keepflag & (self.spiralflag)]
        plot(x,y,'b*',mec='b',label='Zoo Sp',markersize=8)

        legend(loc='lower right',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.axisratio1err[keepflag],fmt=None,color='r')
        xlabel('NSA SERSIC_BA')
        ylabel('GALFIT MIPS B/A')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')

        #legend(numpoints=1, loc='upper left')

    def plotGalfitSDSSvs24(self):
        keep24flag=self.snr24 > 3
        galkeep=(self.galfit.re1 > .1) & (self.galfit24.re1 > .1)
        keepflag=(self.On24ImageFlag & keep24flag)# & galkeep)
        figure(figsize=(12,12))

        subplot(2,2,1)
        x=self.galfitNSA.re1[keepflag]#*sdsspixelscale
        y=self.galfit24.re1[keepflag]#*mipspixelscale
        plot(x,y,'ro',label='Re')
        legend(loc='upper left',numpoints=1)
        axhline(y=7.1,ls='--',color='k')
        errorbar(x,y,yerr=self.galfit24.re1err[keepflag]*mipspixelscale,xerr=self.galfit.re1err[keepflag]*sdsspixelscale,fmt=None)
        gca().set_yscale('log')
        gca().set_xscale('log')

        xlabel('Galfit SDSS Re (arcsec)')
        ylabel('Galfit MIPS Re (arcsec)')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')
        title(self.prefix)
        
        subplot(2,2,2)
        x=self.galfitNSA.nsersic1[keepflag]
        y=self.galfit24.nsersic1[keepflag]

        plot(x,y,'ro',label='Sersic Index')
        #errorbar(x,y,yerr=self.galfit.nsersic1err[keepflag],fmt=None,color='r')

        legend(loc='upper left',numpoints=1)
        
        xlabel('Galfit SDSS Sersic N')
        ylabel('Galfit MIPS Sersic N')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')
        axhline(y=1,ls='--',color='k')
        axhline(y=4,ls='--',color='k')

        subplot(2,2,3)
        #x=arccos(-1.*sin(self.galfit.pa1[keepflag]/180.*pi))*180/pi
        x=self.galfitNSA.pa1[keepflag]
        y=self.galfit24.pa1[keepflag]
        for i in range(len(y)):
            if y[i] < 0:
                y[i]=180+y[i]
            
        plot(x,y,'ro',label='PA')
        legend(loc='upper left',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.pa1err[keepflag],fmt=None,color='r')

        xlabel('Galfit SDSS PA')
        ylabel('Galfit MIPS PA')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')

        subplot(2,2,4)
        x=self.galfitNSA.axisratio1[keepflag]
        y=self.galfit24.axisratio1[keepflag]
        plot(x,y,'ro',label='B/A')
        legend(loc='lower right',numpoints=1)
        #errorbar(x,y,yerr=self.galfit.axisratio1err[keepflag],fmt=None,color='r')
        xlabel('Galfit SDSS B/A')
        ylabel('GALFIT MIPS B/A')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.1)
        plot(xl,xl,'k:')

        #legend(numpoints=1, loc='upper left')

    def plotratioRevsdrR200(self):
        keep24flag=self.snr24 > 1
        keep24flag=self.sex24.MATCHFLAG24
        keepflag=(self.On24ImageFlag & keep24flag & self.dvflag)
        figure()
        y=self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]
        x=self.drR200[keepflag]
        plot(x,y,'ko',markersize=6,)
        xbin,ybin,ybinerr=my.binitbinsmedian(0.,3.5,4,x,y)
        plot(xbin,ybin,'k-')
        plot(xbin,ybin+ybinerr,'k:')
        plot(xbin,ybin-ybinerr,'k:')

        #legend(loc='upper left', numpoints=1)
        xlabel('$\Delta R/R_{200} $',fontsize=16)
        ylabel('$ R_e(24\mu m) / R_e (NSA) $',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratiovsdr.eps'
        savefig(fname)
        print '\n ratio Re vs dr/R200 \n'
        rho,p=spearman(x,y)


    def plotratioRedvdr(self):
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag & keep24flag)# & self.dvflag)
        figure()
        #y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]

        x=self.drR200[keepflag]
        y=(self.n.ZDIST[keepflag]*3.e5-self.biweightvel)/self.biweightscale

        #z=log10(self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag])
        z=(self.galfit24.re1[keepflag]*mipspixelscale-self.n.SERSIC_TH50[keepflag])/self.n.SERSIC_TH50[keepflag]
        hist(z,bins=arange(-10,10))
        figure()
        sp=scatter(x,y,c=z,vmin=-2,vmax=5,cmap='jet')
        cb=colorbar(sp)

        #legend(loc='upper left', numpoints=1)
        xlabel('$\Delta R/R_{200} $',fontsize=16)
        ylabel('$ \Delta v/\sigma$',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratiodvdr.eps'
        savefig(fname)
        #print '\n ratio Re vs dr/R200 \n'
        #rho,p=spearman(x,y)

    def plotratio24nsadist3d(self):
        keep24flag=self.snr24 > 1
        keep24flag=self.sex24.MATCHFLAG24
        keepflag=self.galfitflag#(self.On24ImageFlag & keep24flag & ~self.ellipticalflag)# & self.dvflag)


        #z=log10(self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag])
        if convflag:
            y=(self.galfit24.cre1[keepflag]*mipspixelscale)
        else:
            y=(self.galfit24.re1[keepflag]*mipspixelscale)

        x=self.n.SERSIC_TH50[keepflag]
        #x=self.galfit.re1[keepflag]*sdsspixelscale
        z=self.dist3d[keepflag]
        #hist(z,bins=arange(-10,10))
        figure()
        sp=scatter(x,y,c=z,cmap='jet',vmin=0,vmax=10)
        cb=colorbar(sp)
        axis([0,25.,0,25.])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')
        #legend(loc='upper left', numpoints=1)
        xlabel('$R_e(NSA) (arcsec) $',fontsize=16)
        ylabel('$R_e(24) (arcsec)$',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratio24nsa.eps'
        savefig(fname)
        #print '\n ratio Re vs dr/R200 \n'
        #rho,p=spearman(x,y)

    def plotratio24nsaphi(self):
        keep24flag=self.snr24 > 1.5
        keep24flag=self.sex24.MATCHFLAG24
        keepflag=(self.On24ImageFlag & keep24flag & ~self.ellipticalflag)


        #z=log10(self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag])
        y=(self.galfit24.re1[keepflag]*mipspixelscale)
        x=self.n.SERSIC_TH50[keepflag]
        #x=self.galfit.re1[keepflag]*sdsspixelscale
        z=self.cluster_phi[keepflag]
        #hist(z,bins=arange(-10,10))
        figure()
        sp=scatter(x,y,c=z,cmap='jet',vmin=0,vmax=90)
        cb=colorbar(sp)
        #axis([0,25.,0,25.])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        plot(xl,xl,'k:')
        #legend(loc='upper left', numpoints=1)
        xlabel('$R_e(NSA) (arcsec) $',fontsize=16)
        ylabel('$R_e(24) (arcsec)$',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratio24nsa.eps'
        savefig(fname)
        #print '\n ratio Re vs dr/R200 \n'
        #rho,p=spearman(x,y)

    def plotratio24nsalir(self):
        keep24flag=self.snr24 > 1.5
        keep24flag=self.sex24.MATCHFLAG24
        keepflag=(self.On24ImageFlag & keep24flag & ~self.ellipticalflag)# & self.dvflag)


        #z=log10(self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag])
        y=(self.galfit24.re1[keepflag & self.member]*mipspixelscale)
        erry=(self.galfit24.re1err[keepflag & self.member]*mipspixelscale)
        x=self.n.SERSIC_TH50[keepflag & self.member]
        #x=self.galfit.re1[keepflag & self.member]*sdsspixelscale
        z=log10(self.n.LIR[keepflag & self.member])
        #hist(z,bins=arange(-10,10))
        figure()
        errorbar(x,y,yerr=erry,fmt=None,ecolor='k')
        sp=scatter(x,y,c=z,cmap='jet',vmin=8,vmax=10,s=60)
        cb=colorbar(sp)

        y=(self.galfit24.re1[keepflag & ~self.member]*mipspixelscale)
        erry=(self.galfit24.re1err[keepflag & ~self.member]*mipspixelscale)
        x=self.n.SERSIC_TH50[keepflag & ~self.member]
        x=self.galfit.re1[keepflag & ~self.member]*sdsspixelscale
        z=log10(self.n.LIR[keepflag & ~self.member])
        #hist(z,bins=arange(-10,10))
        errorbar(x,y,yerr=erry,fmt=None,ecolor='k')
        sp=scatter(x,y,c=z,cmap='jet',vmin=8,vmax=10,marker='^',s=50)

        axis([0,50.,0,30.])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax)
        ymin,ymax=ylim()
        yl=arange(ymin,ymax)
        plot(yl,yl,'k:')
        #legend(loc='upper left', numpoints=1)
        xlabel('$R_e(NSA) (arcsec) $',fontsize=16)
        ylabel('$R_e(24) (arcsec)$',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratio24nsa.eps'
        savefig(fname)
        #print '\n ratio Re vs dr/R200 \n'
        #rho,p=spearman(x,y)

    def plotdvdrlir(self):
        keep24flag=(self.snr24 > 1) & self.on24Imageflag & self.spiralflag
        #keepflag=(self.On24ImageFlag & keep24flag & ~self.ellipticalflag)# & self.dvflag)
                                                  #figure()
        #y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]

        x=self.drR200[keepflag]
        y=(self.n.ZDIST[keepflag]*3.e5-self.biweightvel)/self.biweightscale

        #z=log10(self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag])
        z=log10(self.ce.LIR[keepflag])
        #z=self.sex24.FLUX_RADIUS1[keepflag]/self.n.SERSIC_TH50[keepflag]
        #hist(z,bins=arange(-10,10))
        figure()
        sp=scatter(x,y,c=z,vmin=8,vmax=10,cmap='jet')
        #cb=colorbar(sp)

        #legend(loc='upper left', numpoints=1)
        xlabel('$\Delta R/R_{200} $',fontsize=16)
        ylabel('$ \Delta v/\sigma$',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratiodvdr.eps'
        savefig(fname)
        #print '\n ratio Re vs dr/R200 \n'
        #rho,p=spearman(x,y)

    def plotdvdrsize(self):
        keepflag=(self.snr24 > 3) & self.On24ImageFlag & self.spiralflag
        #keepflag=(self.On24ImageFlag & keep24flag & ~self.ellipticalflag)# & self.dvflag)
                                                  #figure()
        #y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]

        x=self.drR200[keepflag]
        y=(self.n.ZDIST[keepflag]*3.e5-self.biweightvel)/self.biweightscale

        #z=log10(self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag])
        z=log10(self.ce.LIR[keepflag])
        z=self.sex24.FLUX_RADIUS1[keepflag]/self.n.SERSIC_TH50[keepflag]
        #hist(z,bins=arange(-10,10))
        figure()
        sp=scatter(x,y,s=40.*z,color='r',alpha=0.5)#vmin=8,vmax=10,cmap='jet')

        keepflag=~self.sex24.MATCHFLAG24 & self.On24ImageFlag & self.spiralflag
        x=self.drR200[keepflag]
        y=(self.n.ZDIST[keepflag]*3.e5-self.biweightvel)/self.biweightscale
        plot(x,y,'k.')

        #legend(loc='upper left', numpoints=1)
        xlabel('$\Delta R/R_{200} $',fontsize=22)
        ylabel('$ \Delta v/\sigma$',fontsize=22)
        title('$ '+self.prefix+' $',fontsize=22)
        axhline(y=0,color='k',ls='-')
        axhline(y=3,color='k',ls='--')
        axhline(y=-3,color='k',ls='--')
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_dvdrsize.eps'
        savefig(fname)
        #print '\n ratio Re vs dr/R200 \n'
        #rho,p=spearman(x,y)

    def plotratioR50vsdrR200(self):
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag & keep24flag)
        figure()
        y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]
        x=self.drR200[keepflag]
        plot(x,y,'k.',label='SNR24 > 1')

        keep24flag=self.snr24 > 3
        keepflag=(self.On24ImageFlag & keep24flag)
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]
        x=self.drR200[keepflag]
        plot(x,y,'bo',label='SNR24 > 3')

        y=self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag]
        x=self.drR200[keepflag]
        plot(x,y,'ro',label='Re(24)/Re(NSA)')

        legend(loc='upper left', numpoints=1)
        xlabel('$\Delta R/R_{200} $')
        ylabel('R50_mips/R50_SDSS (arcsec)')
    def plotratioR90vsdrR200(self):
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag & keep24flag)
        figure()
        y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit_mipsR90[keepflag,0]/self.galfit_sdssR90[keepflag,0]
        x=self.drR200[keepflag]
        plot(x,y,'k.',label='SNR24 > 1')

        keep24flag=self.snr24 > 3
        keepflag=(self.On24ImageFlag & keep24flag)
        y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit_mipsR90[keepflag,0]/self.galfit_sdssR90[keepflag,0]
        x=self.drR200[keepflag]
        plot(x,y,'bo',label='SNR24 > 3')

        legend(loc='upper left', numpoints=1)
        xlabel('$\Delta R/R_{200} $')
        ylabel('R90_mips/R90_SDSS (arcsec)')

    def plotratioR50vsclusterphi(self):
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag & keep24flag)
        figure()
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]
        x=self.cluster_phi[keepflag]
        plot(x,y,'k.',label='SNR24 > 1')
        xbin,ybin,ybinerr=my.binitbinsmedian(0.,90,4,x,y)
        plot(xbin,ybin,'k-')
        plot(xbin,ybin+ybinerr,'k:')
        plot(xbin,ybin-ybinerr,'k:')

        keep24flag=self.snr24 > 3
        keepflag=(self.On24ImageFlag & keep24flag)
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]
        x=self.cluster_phi[keepflag]
        plot(x,y,'bo',label='SNR24 > 3')
        xbin,ybin,ybinerr=my.binitbinsmedian(0.,90,4,x,y)
        plot(xbin,ybin,'k-')
        plot(xbin,ybin+ybinerr,'k:')
        plot(xbin,ybin-ybinerr,'k:')

        legend(loc='upper left', numpoints=1)
        xlabel('PHI (arcsec)')
        ylabel('R50_mips/R50_SDSS (arcsec)')
    def plotratioRevslocaldensity(self):
        #keep24flag=self.sex24.MATCHFLAG24
        #keepflag=(self.On24ImageFlag & keep24flag & self.dvflag) & ~self.ellipticalflag & ~self.agnflag
        keepflag=self.sample24flag & self.dvflag
        figure(figsize=(14,6))
        subplot(1,2,1)
        #y=self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag]
        y=self.size24[keepflag]/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit_mipsR90[keepflag,0]/self.galfit_sdssR90[keepflag,0]
        x=log10(self.ld.SIGMA_NN[keepflag])
        plot(x,y,'ko')
        xbin,ybin,ybinerr=my.binitbinsmedian(0.2,2.7,4,x,y)
        plot(xbin,ybin,'k-')
        plot(xbin,ybin+ybinerr,'k:')
        plot(xbin,ybin-ybinerr,'k:')
        xlabel('$log_{10}(\Sigma_{NN})$',fontsize=16)

        ylabel('$ R_e(24\mu m)/R_e (NSA) $',fontsize=16)
        print '\n ratio Re vs Sigma_NN \n'
        rho,p=spearman(x,y)


        subplot(1,2,2)
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit_mipsR90[keepflag,0]/self.galfit_sdssR90[keepflag,0]
        x=log10(self.ld.RHOMASS[keepflag])
        plot(x,y,'ko',label='SNR24 > 3')
        xbin,ybin,ybinerr=my.binitbinsmedian(9.3,12.,4,x,y)
        plot(xbin,ybin,'k-')
        plot(xbin,ybin+ybinerr,'k:')
        plot(xbin,ybin-ybinerr,'k:')

        #gca().set_xscale('log')
        #legend(loc='upper left', numpoints=1)
        xlabel(r'$log_{10}( \Sigma_{mass} (r < 300kpc))$',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratiovslocaldensity.eps'
        savefig(fname)
        print '\n ratio Re vs Sigma_mass \n'
        rho,p=spearman(x,y)

    def plotratioR50vsclusterphi(self):
        keep24flag=self.snr24 > 1
        keepflag=(self.On24ImageFlag & keep24flag)
        figure()
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]
        x=self.cluster_phi[keepflag]
        plot(x,y,'k.',label='SNR24 > 1')

        keep24flag=self.snr24 > 3
        keepflag=(self.On24ImageFlag & keep24flag)
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        y=self.galfit_mipsR50[keepflag,0]/self.galfit_sdssR50[keepflag,0]
        x=self.cluster_phi[keepflag]
        plot(x,y,'bo',label='SNR24 > 3')

        legend(loc='upper left', numpoints=1)
        xlabel('PHI (arcsec)')
        ylabel('R50_mips/R50_SDSS (arcsec)')


    def plotratioRevsclusterphi(self):
        keep24flag=self.snr24 > 3
        keepflag=self.galfitflag & self.member
        figure(figsize=(14,6))
        subplot(1,2,1)
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        y=self.galfit24.re1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        x=self.cluster_phi[keepflag]
        plot(x,y,'ko',markersize=6)
        xbin,ybin,ybinerr=my.binitbinsmedian(0.,90,4,x,y)
        plot(xbin,ybin,'k-')
        plot(xbin,ybin+ybinerr,'k:')
        plot(xbin,ybin-ybinerr,'k:')

        xlabel('$ \phi (deg)$',fontsize=16)
        ylabel('$ R_e(24\mu m)/R_e(NSA) $',fontsize=16)

        print '\n ratio Re vs cluster_phi \n'
        rho,p=spearman(x,y)

        subplot(1,2,2)
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit_mipsR90[keepflag,0]/self.n.SERSIC_TH50[keepflag]
        y=self.galfit24.R90[keepflag]/self.n.SERSIC_TH50[keepflag]
        x=self.cluster_phi[keepflag]
        plot(x,y,'ko',markersize=6)
        xbin,ybin,ybinerr=my.binitbinsmedian(0.,90,4,x,y)
        plot(xbin,ybin,'k-')
        plot(xbin,ybin+ybinerr,'k:')
        plot(xbin,ybin-ybinerr,'k:')

        xlabel('$ \phi (deg)$',fontsize=16)
        ylabel('$ R_{90}(24\mu m)/R_e(NSA) $',fontsize=16)
        fname=homedir+'research/LocalClusters/SamplePlots/'+self.prefix+'_ratiovsclusterphi.eps'
        savefig(fname)
        print '\n ratio R90(24)/Re(NSA) vs cluster_phi \n'
        rho,p=spearman(x,y)



    def plotsizeHIdef(self,colors,shapes,plotsingle=1,convflag=0):
        keepflag=self.galfitflag & (self.n.HIflag) #& self.dvflag
        if plotsingle:
            figure()
        if convflag:
            y=self.galfit24.cre1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
            yer=self.galfit24.cre1err[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        else:
            y=self.galfit24.re1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
            yer=self.galfit24.re1err[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]

        #y=self.sex24.FLUX_RADIUS1[keepflag]/self.n.SERSIC_TH50[keepflag]
        x=self.HIDef[keepflag]
        z=self.dist3d[keepflag]*20
        z=30.*ones(len(self.dist3d[keepflag]))
        if len(x) > 0:
            plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=8)
            #scatter(x,y,s=z,alpha=0.5,marker=shapes,color=colors)
            errorbar(x,y,yerr=yer,fmt=None,ecolor=colors)
        

        if plotsingle:
            xlabel('$HI \ Def$',fontsize=20)
            ylabel('$R_e(24)/R_e(r) $',fontsize=20)
            title(self.prefix)

        # plot HI upper limits for spirals

        #keepflag=(self.On24ImageFlag & (self.sex24.MATCHFLAG24) & (self.snr24>3) &  ~(self.n.HIflag) & self.dvflag & (self.sex24.FLUX_RADIUS1 > 0) & ~self.agnflag & self.spiralflag)
        #y1=self.sex24.FLUX_RADIUS1[keepflag]/self.n.SERSIC_TH50[keepflag]
        #x1=self.HIDef_ulim[keepflag]
        #errorbar(x1,y1,xerr=0.04,fmt=None,xuplims=True,ecolor='k',label='_nolegend_')

        return x,y

    def plotsizeclusterphi(self,colors,shapes,plotsingle=1,convflag=0):
        keepflag=self.galfitflag & (self.member) #& self.dvflag
        if plotsingle:
            figure()
        if convflag:
            y=self.galfit24.cre1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
            yer=self.galfit24.cre1err[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        else:
            y=self.galfit24.re1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
            yer=self.galfit24.re1err[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        x=self.cluster_phi[keepflag]
        if len(x) > 0:
            plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=8)
            errorbar(x,y,yerr=yer,fmt=None,ecolor=colors)
        

        if plotsingle:
            xlabel('$\Phi$',fontsize=20)
            ylabel('$R_e(24)/R_e(r) $',fontsize=20)
            title(self.prefix)


        return x,y

    def plotsizesSFR(self,colors,shapes,plotsingle=1,convflag=1):
        keepflag=self.galfitflag & self.dvflag

        #keepflag=(self.On24ImageFlag & (self.sex24.MATCHFLAG24) &  self.dvflag)
        if plotsingle:
            figure()
        y=self.galfit24.cre1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        #x=self.ce.SFR[keepflag]#/self.stellarmass[keepflag]*1.e9
        #x=self.sex24.MAG_BEST[keepflag]
        #y=self.sex24.FLUX_RADIUS1[keepflag]*mipspixelscale#/self.n.SERSIC_TH50[keepflag]
        #x=self.galfit24.re1[keepflag]*mipspixelscale#/self.n.SERSIC_TH50[keepflag]
        #y=self.galfit24.re1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        #y=self.n.SERSIC_TH50[keepflag]
        x=self.ce.SFR_ZCLUST[keepflag]/self.stellarmass[keepflag]*1.e9
        plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=10)

        #keepflag=(self.On24ImageFlag & (self.sex24.MATCHFLAG24) & (self.snr24>3)  & ~self.dvflag & (self.sex24.FLUX_RADIUS1 > 0) & ~self.agnflag)
        #y1=self.sex24.FLUX_RADIUS1[keepflag]/self.n.SERSIC_TH50[keepflag]
        #x1=self.ce.SFR[keepflag]/self.stellarmass[keepflag]*1.e9
        #plot(x1,y1,'bo',label='_nolegend_')
        return x,y

    def plotratioRevsHaEW(self,plotsingle=1,convflag=1):
        #keepflag=(self.On24ImageFlag & (self.snr24>3)  & (self.n.HAEW > 4.) & self.dvflag & (self.sex24.FLUX_RADIUS1 > 0) & ~self.agnflag)
        #keepflag=(self.On24ImageFlag & (self.sex24.MATCHFLAG24) & (self.n.HIflag) & self.dvflag)
        if plotsingle:
            figure()
        #y=self.galfit24.re1[keepflag]/self.n.SERSIC_TH50[keepflag]
        keepflag=self.galfitflag & self.dvflag & (self.n.ISDSS > -1)
        if convflag:
            y=self.galfit24.cre1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        else:
            y=self.galfit24.re1[keepflag]*mipspixelscale/self.n.SERSIC_TH50[keepflag]
        x=self.n.HAEW[keepflag]
        plot(x,y,'ko',label='1 Comp Fit')

        #y1=self.sex24.FLUX_RADIUS2[keepflag]/self.n.SERSIC_TH50[keepflag]
        #y1=self.sex24.FLUX_RADIUS2[keepflag]/self.sex24.FLUX_RADIUS1[keepflag]
        #x1=self.n.HAEW[keepflag]
        #plot(x1,y1,'bo',label='1 Comp Fit')

        #xbin,ybin,ybinerr=my.binitbinsmedian(-.3,1.4,4,x,y)
        #plot(xbin,ybin,'k-')
        #plot(xbin,ybin+ybinerr,'k:')
        #plot(xbin,ybin-ybinerr,'k:')
        #print '\n ratio Re vs HIdef \n'
        #rho,p=spearman(x,y)
        #print '\n ratio Re vs HIdef (ignoring lowest HIdef point) \n'
        #rho,p=spearman(x[x> -.1],y[x > -.1])
        return x,y
    def plotRefieldcluster(self):
        keepflag=(self.On24ImageFlag & (self.snr24>3) & self.dvflag)
        figure()
        y=self.galfit24.re1[keepflag]
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'ko',label='1 Comp Fit')
        keepflag=(self.On24ImageFlag & (self.snr24>3) & ~self.dvflag)
        y=self.galfit24.re1[keepflag]
        x=self.n.SERSIC_TH50[keepflag]
        plot(x,y,'bo',label='1 Comp Fit')
        xlabel('NSA SERSIC_TH50')
        ylabel('Re(24um)')


        print '\n ratio Re vs HIdef \n'
        rho,p=spearman(x,y)
        print '\n ratio Re vs HIdef (ignoring lowest HIdef point) \n'
        rho,p=spearman(x[x> -.1],y[x > -.1])

    def plotRevsEnv(self):
        
        keepflag=(self.On24ImageFlag & (self.snr24>3) & self.dvflag)
        figure(figsize=(12,10))
        environment=[self.ld.SIGMA_NN,self.ld.RHOMASS,self.drR200]
        env_xlabels=['SIGMA_NN','SIGMA_MASS','dr/R200']
        for i in range(len(environment)):
            subplot(2,2,i+1)
            y=self.n.SERSIC_TH50[self.dvflag]
            x=environment[i][self.dvflag]
            rho,p=spearman(x,y)
            s='all (%5.2f, %5.2f)'%(rho,p)
            plot(x,y,'ko',label=s)
            print '\n ratio Re vs environment(%i) all \n'%(i)
            y=self.n.SERSIC_TH50[keepflag]
            x=environment[i][keepflag]
            print '\n ratio Re vs environment(%i) (24um sources)\n'%(i)
            rho,p=spearman(x,y)
            s='24 (%5.2f, %5.2f)'%(rho,p)
            plot(x,y,'ro',label=s)
            xlabel(env_xlabels[i])
            ylabel('NSA SERSIC_TH50')
            legend(numpoints=1,loc='upper right')

    def plotGalfitRvsHIDef(self):
        keepflag=(self.On24ImageFlag & self.spiralflag)
        figure()
        y1=self.galfit_sdssR50[keepflag,0]
        y2=self.galfit_mipsR50[keepflag,0]
        y=y1-y2
        x=self.HIDef[keepflag]
        #plot(x,y,'ko',label='1 Comp Fit')
        y1=self.galfit_sdssR50[keepflag,1]
        y2=self.galfit_mipsR50[keepflag,1]
        y=y1-y2
        plot(x,y,'go',label='2 Comp Fit')
        y1=self.galfit_sdssR50[keepflag,2]
        y2=self.galfit_mipsR50[keepflag,2]
        y=y1-y2
        #plot(x,y,'ro',label='3 Comp Fit')
        
        y1=self.galfit_sdssR90[keepflag,1]
        y2=self.galfit_mipsR90[keepflag,1]
        y=y1-y2
        plot(x,y,'go',label='2 Comp R90')


        x=self.HIDef[keepflag & self.member]
        y1=self.galfit_sdssR90[keepflag & self.member,1]
        y2=self.galfit_mipsR90[keepflag & self.member,1]
        y=y1-y2
        plot(x,y,'g^',label='2 Comp R90',markersize=12)

        legend(loc='lower right', numpoints=1)
        xlabel('HIDef')
        ylabel('SDSS R50-MIPS R50')
        axhline(y=0,ls='--',color='k')
    def plotGalfitR50(self):
        keepflag=(self.On24ImageFlag & self.spiralflag)
        figure()
        x=self.galfit_sdssR50[keepflag,0]
        y=self.galfit_mipsR50[keepflag,0]
        #plot(x,y,'ko',label='1 Comp Fit')
        x=self.galfit_sdssR50[keepflag,1]
        y=self.galfit_mipsR50[keepflag,1]
        plot(x,y,'go',label='2 Comp Fit')
        x=self.galfit_sdssR50[(keepflag & self.member),1]
        y=self.galfit_mipsR50[(keepflag & self.member),1]
        plot(x,y,'g^',label='2 Comp Fit Memb',markersize=12)

        x=self.galfit_sdssR50[keepflag,2]
        y=self.galfit_mipsR50[keepflag,2]
        #plot(x,y,'ro',label='3 Comp Fit')
        legend(loc='upper right', numpoints=1)
        xlabel('SDSS R50')
        ylabel('MIPS R50')
        xl=arange(20)
        plot(xl,xl,'k--')

    def plotfwhm24(self):
        y=self.sex24.FWHM_DEG*3600.
        x=self.sex24.MAG_BEST
        flag=self.sex24.MATCHFLAG24
        figure()
        plot(x[flag],y[flag],'ko',label='All')
        flag=self.sex24.MATCHFLAG24 & self.ellipticalflag
        plot(x[flag],y[flag],'ro',mec='r',label='Zoo E')
        flag=self.sex24.MATCHFLAG24 & self.spiralflag
        plot(x[flag],y[flag],'bo',mec='b',label='Zoo Sp')
        xlabel('MAG_BEST')
        ylabel('FWHM (arcsec)')
        legend(numpoints=1)
        axhline(y=6,ls='--')

    def plotfwhm24isoarea(self):
        figure(figsize=(10,10))
        for i in range(4):
            subplot(2,2,i+1)
            x=self.sex24.FWHM_DEG*3600.
            if i == 0:
                y=self.sex24.ISOAREA_IMAGE
                ylab='ISO_AREA (pix**2)'
            elif i == 1:
                y=self.sex24.KRON_RADIUS
                ylab='KRON_RADIUS'
            elif i == 2:
                y=self.sex24.PETRO_RADIUS
                ylab='PETRO_RADIUS'
            elif i == 3:
                y=self.sex24.FLUX_RADIUS1
                ylab='FLUX_RADIUS1'
            flag=self.sex24.MATCHFLAG24
            plot(x[flag],y[flag],'ko',label='All')
            flag=self.sex24.MATCHFLAG24 & self.ellipticalflag
            plot(x[flag],y[flag],'ro',mec='r',label='Zoo E')
            flag=self.sex24.MATCHFLAG24 & self.spiralflag
            plot(x[flag],y[flag],'bo',mec='b',label='Zoo Sp')
            ylabel(ylab)
            xlabel('FWHM (arcsec)')
            #legend(numpoints=1)
            #axhline(y=6,ls='--')

    def plotfwhm24Re(self):
        x=self.sex24.FWHM_DEG*3600.
        y=self.galfit24.re1*mipspixelscale
        flag=self.sex24.MATCHFLAG24
        figure()
        plot(x[flag],y[flag],'ko',label='All')
        flag=self.sex24.MATCHFLAG24 & self.ellipticalflag
        plot(x[flag],y[flag],'ro',mec='r',label='Zoo E')
        flag=self.sex24.MATCHFLAG24 & self.spiralflag
        plot(x[flag],y[flag],'bo',mec='b',label='Zoo Sp')

        xlabel('FWHM (arcsec)')
        ylabel('GALFIT Re (arcsec)')
        axis([0,25,0,25])
        legend(numpoints=1)
        xl=arange(26)
        plot(xl,xl,'k--')
        axhline(y=3.5,ls=':')
        axvline(x=6,ls=':')

    def plotsex24Re(self):
        figure()
        y=(self.sex24.FLUX_RADIUS1)*mipspixelscale
        x=self.galfit24.re1*mipspixelscale
        flag=self.sex24.MATCHFLAG24 & (self.galfit24.re1 > .1)
        print 'FLUX_RADIUS1 vs Re'
        spearman(x[flag],y[flag])

        plot(x[flag],y[flag],'ko',label=self.prefix)

        print 'FLUX_RADIUS2 vs Re'
        y=(self.sex24.FLUX_RADIUS2)*mipspixelscale
        spearman(x[flag],y[flag])
        #plot(x[flag],y[flag],'bo',label='FLUX_RADIUS2')

        y=self.sex24.FWHM_DEG*3600.
        print 'FWHM vs Re'
        spearman(x[flag],y[flag])
        #plot(x[flag],y[flag],'ro',mec='r',label='FWHM')

        y=sqrt(self.sex24.ISOAREA_IMAGE)*mipspixelscale
        print 'sqrt(ISOAREA) vs Re'
        spearman(x[flag],y[flag])

        #plot(x[flag],y[flag],'go',label='sqrt(ISOAREA)')

        ylabel('SE FLUX_RADIUS (arcsec)')
        xlabel('GALFIT Re (arcsec)')
        axis([0,18,0,15])
        legend(numpoints=1)
        xl=arange(15)
        plot(xl,xl,'k--')
        #axhline(y=3.5,ls=':')
        #axvline(x=6,ls=':')

    def compareRad24(self):
        figure()
        plot(self.r30F24,self.r30EncFluxF24,'bo')
        plot(self.r90F24,self.r90EncFluxF24,'ro')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        plot(xl,xl,'k-')
        plot(xl,1.3*xl,'k:')
        plot(xl,.7*xl,'k:')
        plot(self.r30F24[self.mipssnrflag],self.r30EncFluxF24[self.mipssnrflag],'b^',markersize=12)
        plot(self.r90F24[self.mipssnrflag],self.r90EncFluxF24[self.mipssnrflag],'r^',markersize=12)

    def compareRadSDSS(self):
        figure()
        plot(self.r30SDSS,self.r30EncFluxSDSS,'bo')
        plot(self.r90SDSS,self.r90EncFluxSDSS,'ro')
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        plot(xl,xl,'k-')
        plot(xl,1.3*xl,'k:')
        plot(xl,.7*xl,'k:')

    def compareR90(self):
        #figure()
        flag=self.ellipseflag & self.spiralflag & (self.r50SDSS > 6.) & self.member
        plot(self.r50SDSS[flag],self.r50F24[flag],'bo')
        plot(self.r90SDSS[flag],self.r50F24[flag],'ro')
        flag=self.ellipseflag & self.spiralflag & (self.r50SDSS > 6.) & ~self.member
        plot(self.r50SDSS[flag],self.r50F24[flag],'bo',markeredgecolor='b',markerfacecolor='None')
        plot(self.r90SDSS[flag],self.r50F24[flag],'ro',markeredgecolor='r',markerfacecolor='None')
        axis([0,40,0,40])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        plot(xl,xl,'k-')
        plot(xl,1.3*xl,'k:')
        plot(xl,.7*xl,'k:')
        axis([0,40,0,40])

        title(self.clustername)

    def comparediffR90localdens(self):
        #figure()
        flag=self.ellipseflag & self.spiralflag & (self.r50SDSS > 6.) & self.member
        plot(self.localdens[flag],self.r50SDSS[flag]-self.r50F24[flag],'bo')
        plot(self.localdens[flag],self.r90SDSS[flag]-self.r50F24[flag],'ro')
        flag=self.ellipseflag & self.spiralflag & (self.r50SDSS > 6.) & ~self.member
        plot(self.localdens[flag],self.r50SDSS[flag]-self.r50F24[flag],'bo',markeredgecolor='b',markerfacecolor='None')
        plot(self.localdens[flag],self.r90SDSS[flag]-self.r50F24[flag],'ro',markeredgecolor='r',markerfacecolor='None')
        axis([1,1000,-25,30])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,1)
        #plot(xl,xl,'k-')
        #plot(xl,1.3*xl,'k:')
        #plot(xl,.7*xl,'k:')
        #axis([0,40,0,40])
        axhline(0,color='k')
        ax=gca()
        ax.set_xscale('log')
        axis([1,1000,-25,30])
        title(self.clustername)

    def plotagn(self):
        figure()
        clf()
        plot((self.logn2halpha),(self.logo3hbeta),'k.', label='_nolabel_')

        plot(self.logn2halpha[self.agn2],self.logo3hbeta[self.agn2],'co',markersize=12, label='_nolabel_')
        plot(self.logn2halpha[self.agn1],self.logo3hbeta[self.agn1],'go',markersize=8, label='_nolabel_')
        plot(self.logn2halpha[self.agn3],self.logo3hbeta[self.agn3],'ro',markersize=4, label='_nolabel_')

        #draw AGN diagnostic lines
        x=arange(-3,.4,.01)
        y=(.61/(x-.47)+1.19)
        #Kewley
        plot(x,y,'c',label='Kewley & Dopita 2002')
        x=arange(-3,.0,.01)
        y =(.61/(x-.05)+1.3)#Kauffman 2003?
        plot(x,y,'g',label='Kauffmann et al. 2003')
        y = ((-30.787+(1.1358*x)+((.27297)*(x)**2))*tanh(5.7409*x))-31.093 #Stasinska 2006	    
        plot(x,y,'r',label='Stasinska et al. 2006')
        axis([-2,.75,-1.5,1.5])
        xlabel(r'$\log_{10}(NII/H\alpha)$',fontsize=20)
        ylabel(r'$\log_{10}(OIII/H\beta)$',fontsize=20)
        legend(loc='upper left')
        savefig(figuredir+self.prefix+'_agn.eps')

    def plotpositions(self):

        #figure()
        #clf()

    #draw footprint of mips data, if applicable
        
        #if self.clustername.find('MKW8') > -1:
        #    drawbox(MKW824um,'r-')
        #if self.clustername.find('MKW11') > -1:
        #    drawbox(MKW1124um,'r-')
        #if self.clustername.find('NGC6107') > -1:
        #    drawbox(NGC24um,'r-')


        #scatter(ra[flag],dec[flag],s=(20-agcmag10[flag]/10)*20+20,color='.8')
        #plot(ra[flag],dec[flag],'k.')

        plot(self.ra[self.sdssflag],self.dec[self.sdssflag],'k.', alpha=0.5,markersize=4,label='SDSS')
        plot(self.ra[self.HIflag],self.dec[self.HIflag],'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='HI')
        plot(self.ra[self.spiralflag],self.dec[self.spiralflag],'g^',markerfacecolor='None',markeredgecolor='g',markersize=12,label='Spiral')

        plot(self.ra[self.apexflag],self.dec[self.apexflag],'ro', markerfacecolor='r',markeredgecolor='b',markersize=4,label='24um')
            #plot(ra[flag],dec[flag],'k.')
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x


        #legend(loc='upper right',numpoints=1)


        title(self.clustername,fontsize=14)
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        axis('equal')
        drawbox(cluster24Box[self.clustername],'g-')
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax+1,2,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(round(ymin),ymax+1,2,'i'),fontsize=10)
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.eps'

        #savefig(s)

    def plotpositionson24(self,racenter=0,deccenter=0,usefwhm24=0,plotsingle=1):
        #racenter=self.cra
        #deccenter=self.cdec
        if plotsingle:
            figure(figsize=(8,8))
        cir=Circle((self.clusterra-racenter,self.clusterdec-deccenter),radius=1.3*self.r200deg,color='None',ec='k')
        gca().add_patch(cir)


        baseflag = self.dvflag & self.On24ImageFlag
        flag=baseflag
        ra=self.ra-racenter
        dec=self.dec-deccenter

        hexbin(ra,dec,cmap=cm.Greys,gridsize=40,vmin=0,vmax=10)#,extent=(racenter-1.2,racenter+1.2,deccenter-1.2,deccenter+1.2))
        #plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'k.', color='0.5',markersize=4,label='$ NSA $')
        #flag = self.HIflag & baseflag
        #plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='$ HI $')
        #flag=self.galfitflag & baseflag
        #plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'co',markerfacecolor='c',markeredgecolor='k',markersize=3,label='$ Spiral $')
        #flag=baseflag & ~self.agnflag & ~self.ellipticalflag & self.snr24flag
        #plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'ro', markerfacecolor='r',markeredgecolor='r',markersize=1,label='_nolabel_')


        ra=self.ra-racenter
        dec=self.dec-deccenter
	dx=0.15
	dy=dx
        ramin,ramax=xlim()
        decmin,decmax=ylim()
	x=arange(ramin,(ramax+dx),dx)
	y=arange(decmin,(decmax+dx),dy)



	#z=zeros([len(y),len(x)],'f')
	#ylength=len(y)
	#xlength=len(x)
	#for i in range(len(ra)):
        #    #print i, len(ra)
	#    xbin=int(round((ra[i]-x[0])/dx))
	#    ybin=int(round((dec[i]-y[0])/dy))
	#    if (xbin > -1) & (xbin < xlength):
	#	if (ybin > -1) & (ybin < ylength):
	#	    z[ybin,xbin] += 1		  
	#ncon=8
        #cmin=1.
        #cmax=24.
        #t=arange(log10(cmin),log10(cmax),(log10(cmax)-log10(cmin))/(1.*ncon))
        #
        #contours=10.**t
        #contours=arange(1.,25.,4)
	#contour(x,y,z,contours,linewidth=1)
        #contour(x,y,z,ncon,linewidth=1)
        #print z
	#contourf(x,y,z,contours,alpha=.2,cmap=cm.Greys)



        if usefwhm24:
            flag=(self.sex24.MATCHFLAG24) & (~self.ellipticalflag) & (~self.agnflag) & baseflag
            
            scalefactor=self.size24/self.n.SERSIC_TH50*20
        else:

            
            scalefactor=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50*40
            
        #scatter(self.ra[flag]-racenter,self.dec[flag]-deccenter,s=50*self.SFR24[flag],color='r',alpha=0.5,label='$ 24\mu m $')
        flag=self.galfitflag & self.dvflag
        scatter(self.ra[flag]-racenter,self.dec[flag]-deccenter,s=scalefactor[flag],color='b',edgecolor='k',alpha=0.8,label='$ Spiral $')
        
        flag2=self.On24ImageFlag & ~self.agnflag & self.spiralflag & (self.n.SERSIC_TH50 > self.mingalaxysize) & ~flag  & (self.stellarmass > minstellarmass)#& (self.ce.LIR > 5.1e8) & (self.rmag < self.rmagcut) & ~flag & self.dvflag
        plot(self.ra[flag2]-racenter,self.dec[flag2]-deccenter,'r.')
        #legend(loc='upper left',numpoints=1,scatterpoints=1)

        ax=gca()
        fsize=14
        if plotsingle:
            fsize=18
        text(.05,.9,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #axis('equal')
        #axis([-1.7,1.7,-1.7,1.7])

        #xticks(arange(-1,2,1))
        #yticks(arange(-1,2,1))
        #xmin,xmax=xlim()
        #print xmin,xmax
        #x1=round(xmin)
        #x2=round(xmax)
        #xt=arange(x1,x2+1)
        #xticks(xt)
        #ymin,ymax=ylim()
        #yticks(arange(round(ymin),ymax,'i'),fontsize=10)

        t=cluster24Box[self.clustername]
        drawbox([t[0]-racenter,t[1]-deccenter,t[2],t[3],t[4]],'g-')
        ax=gca()
        ax.invert_xaxis()
        if plotsingle:
            xlabel('$ \Delta RA \ (deg) $',fontsize=22)
            ylabel('$ \Delta DEC \ (deg) $',fontsize=22)
            legend(numpoints=1,scatterpoints=1)
        #if self.prefix.find('Hercules') > -1:
        #    legend(numpoints=1,scatterpoints=1)
            

    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.eps'

        #savefig(s)

    def plotpositionson24WithRatio(self):
        #plot(self.ra[self.sdssflag & self.On24ImageFlag],self.dec[self.sdssflag& self.On24ImageFlag],'k.', alpha=0.5,markersize=2,label='SDSS')
        plotflag=self.spiralflag & self.ellipseflag & ~self.agn1 #self.mipssnrflag & (~self.agn2) #& velFlag
        #plotflag=self.spiralflag & self.ellipseflag & ~self.agn1 #self.mipssnrflag & (~self.agn2) #& velFlag
        #print 'Number of spirals to plot = ',sum(plotflag)
        plot(self.ra[plotflag],self.dec[plotflag],'wo',markerfacecolor='0.2',markeredgecolor='w',markersize=4,label='Spiral')
        #axis('equal')
        drawbox(cluster24Box[self.clustername],'g-')
        a,b=xlim()
        xticks(arange(round(a),b,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(round(ymin),ymax,1,'i'),fontsize=10)
        ra=self.ra[self.On24ImageFlag]
        dec=self.dec[self.On24ImageFlag]
	dx=0.15
	dy=dx
        ramin,ramax=xlim()
        decmin,decmax=ylim()
	x=arange(ramin,(ramax+dx),dx)
	y=arange(decmin,(decmax+dx),dy)

	z=zeros([len(y),len(x)],'f')
	ylength=len(y)
	xlength=len(x)
	for i in range(len(ra)):
	    xbin=int(round((ra[i]-x[0])/dx))
	    ybin=int(round((dec[i]-y[0])/dy))
	    if (xbin > -1) & (xbin < xlength):
		if (ybin > -1) & (ybin < ylength):
		    z[ybin,xbin] += 1		  
	ncon=8
        cmin=1.
        cmax=24.
        t=arange(log10(cmin),log10(cmax),(log10(cmax)-log10(cmin))/(1.*ncon))
        
        contours=10.**t
        contours=arange(1.,25.,2)
	#contour(x,y,z,contours,linewidth=1)
        #contour(x,y,z,ncon,linewidth=1)
        #print z
	contourf(x,y,z,contours,alpha=.8)#cmap=cm.Greys)

        scale=4.e5*(self.sSFR90-self.sSFR50)
        print scale[plotflag]
        scatter(self.ra[plotflag],self.dec[plotflag],s=(scale[plotflag]),c='r',alpha=0.7)
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x

        title(self.clustername,fontsize=12)
        #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #axis('equal')
        #yticks(arange(round(ymin),ymax+1,1,'i'),fontsize=10)
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.eps'

        #savefig(s)

        

    def plotrelativepositionson24(self):
        cir=Circle((self.clusterra,self.clusterdec),radius=self.r200deg,color='0.85',ec='k',alpha=0.3)
        gca().add_patch(cir)

        plot(self.ra[self.sdssflag & self.On24ImageFlag]-self.clusterra,self.dec[self.sdssflag& self.On24ImageFlag]-self.clusterdec,'k.', alpha=0.5,markersize=4,label='SDSS')
        plot(self.ra[self.HIflag & self.On24ImageFlag]-self.clusterra,self.dec[self.HIflag & self.On24ImageFlag]-self.clusterdec,'bo', markerfacecolor='None',markeredgecolor='b',markersize=6,label='HI')

        plot(self.ra[self.apexflag]-self.clusterra,self.dec[self.apexflag]-self.clusterdec,'ro', markerfacecolor='r',markeredgecolor='b',markersize=4,label='24um')
        plot(array([0]),array([0]),'kx',markersize=15,lw=8,label='_nolegend_')#mark cluster center with a red x

        plot(self.ra[self.agnflag & self.On24ImageFlag],self.dec[self.agnflag & self.On24ImageFlag],'c*',label='AGN',markersize=4)
        plot(array([self.clusterra]),array([self.clusterdec]),'kx',markersize=15,lw=8)#mark cluster center with a red x

        #if self.prefix.find('Herc') > -1:
        #    legend(loc='upper left',numpoints=1,scatterpoints=1,prop=10)

        drawbox(cluster24Box[self.clustername],'g-')




        title(self.clustername,fontsize=12)
        axis('equal')


        axis([-1.5,1.5,-2.,2.])
        xmin,xmax=xlim()
        xticks(arange(-1,2,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        yticks(arange(-2,3,1,'i'),fontsize=10)

    def plotcolormag(self,usefwhm24=0,plotsingle=1,convflag=1):
        if plotsingle:
            figure()
        #color=self.sdssumag-self.sdssrmag
        color=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]
        mag=log10(self.stellarmass)
        mag=self.n.ABSMAG[:,4]
        baseflag=self.On24ImageFlag & self.member
        #flag= self.dvflag
        plot(mag[baseflag],color[baseflag],'k.',color='0.5',label='All',markersize=3,alpha=0.5)
        flag= baseflag & self.spiralflag
        scatter(mag[flag],color[flag],color='b',s=10,label='Zoo Sp',alpha=0.5)

        if usefwhm24:
            ratio=self.size24/self.n.SERSIC_TH50
            flag=self.sex24.MATCHFLAG24 & baseflag & ~self.agnflag & ~self.ellipticalflag
        else:
            #flag=self.snr24flag & baseflag & ~self.agnflag & ~self.ellipticalflag
            flag = self.galfitflag & self.member
            if convflag:
                ratio=40.*self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            else:
                ratio=20.*self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50

        scatter(mag[flag],color[flag],color='r',s=(ratio[flag]),alpha=0.5,label='R24/Re')
        #hexbin(mag[flag],color[flag],cmap=cm.jet,gridsize=20)
        flag= baseflag & self.HIflag
        if usefwhm24:
            flag=self.sex24.MATCHFLAG24 & baseflag & ~self.agnflag
        else:
            flag=self.snr24flag & baseflag & ~self.agnflag
        ymin=-0.1
        ymax=1.2
        if self.prefix.find('11') > -1:
            leg=legend(numpoints=1,loc='lower right',scatterpoints=1,markerscale=0.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
            for t in leg.get_texts():
                t.set_fontsize('small')
        #axis([8,13,-0.1,7])
        axis([-24,-14,-0.1,8])
        dy=.2
        #yticks(arange(ymin,ymax+dy,dy))
        ax=gca()
        s='$'+self.prefix+'$'
        text(0.06,.85,s,horizontalalignment='left',transform=ax.transAxes,fontsize=14)
        if plotsingle:
            xlabel('$log_{10}(M_* \ (M_\odot)) $',fontsize=16)
            ylabel('$M_{NUV} \ -  \ M_r $',fontsize=16)
        plotcolormaglines()




    def plotcolormagfield(self,usefwhm24=0,plotsingle=1):
        if plotsingle:
            figure()
        #color=self.sdssumag-self.sdssrmag
        color=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]
        mag=log10(self.stellarmass)
        mag=self.n.ABSMAG[:,4]
        baseflag=self.On24ImageFlag & ~self.member
        #flag= self.dvflag
        try:
            plot(mag[baseflag],color[baseflag],'k.',color='0.5',label='All',markersize=3)
        except:
            print 'oops'
        flag= baseflag & self.spiralflag
        try:
            scatter(mag[flag],color[flag],color='b',s=10,label='Zoo Sp',alpha=0.5)
        except:
            print 'apparently no field spirals in ',self.prefix

        if usefwhm24:
            ratio=self.size24/self.n.SERSIC_TH50
            flag=self.sex24.MATCHFLAG24 & baseflag & ~self.agnflag & ~self.ellipticalflag
        else:
            #flag=self.snr24flag & baseflag & ~self.agnflag & ~self.ellipticalflag
            ratio=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
            flag=self.galfitflag & ~self.member
        try:
            scatter(mag[flag],color[flag],color='r',s=(ratio[flag])*20,alpha=0.5,label='R24/Re')
        except:
            print 'oops'
        #hexbin(mag[flag],color[flag],cmap=cm.jet,gridsize=20)
        ymin=-0.1
        ymax=1.2
        if self.prefix.find('11') > -1:
            leg=legend(numpoints=1,loc='lower right',scatterpoints=1,markerscale=0.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
            for t in leg.get_texts():
                t.set_fontsize('small')
        #axis([8,13,-0.1,7])
        axis([-24,-14,-0.1,8])
        dy=.2
        #yticks(arange(ymin,ymax+dy,dy))
        ax=gca()
        s='$'+self.prefix+'$'
        text(0.06,.85,s,horizontalalignment='left',transform=ax.transAxes,fontsize=14)
        if plotsingle:
            xlabel('$log_{10}(M_* \ (M_\odot)) $',fontsize=16)
            ylabel('$M_{NUV} \ -  \ M_r $',fontsize=16)

        plotcolormaglines()

    def plotcolormass(self,usefwhm24=0,plotsingle=1,convflag=1):
        if plotsingle:
            figure()
        #color=self.sdssumag-self.sdssrmag
        color=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]
        mag=log10(self.stellarmass)
        baseflag=self.On24ImageFlag & self.member
        plot(mag[baseflag],color[baseflag],'k.',color='0.5',label='All',markersize=3,alpha=0.5)

        flag= baseflag & self.spiralflag & ~self.agnflag
        scatter(mag[flag],color[flag],color='b',s=10,label='Zoo Sp')

        if usefwhm24:
            ratio=self.size24/self.n.SERSIC_TH50
            flag=self.sex24.MATCHFLAG24 & baseflag & ~self.agnflag & ~self.ellipticalflag
        else:
            #flag=self.snr24flag & baseflag & ~self.agnflag & ~self.ellipticalflag
            flag = self.galfitflag & self.member
            if convflag:
                ratio=40.*self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            else:
                ratio=20.*self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50

        scatter(mag[flag],color[flag],color='r',s=(ratio[flag]),alpha=0.5,label='R24/Re')
        #hexbin(mag[flag],color[flag],cmap=cm.jet,gridsize=20)
        if self.prefix.find('11') > -1:
            leg=legend(numpoints=1,loc='lower right',scatterpoints=1,markerscale=0.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
            for t in leg.get_texts():
                t.set_fontsize('small')
        axis([8,13,-0.1,7])
        #axis([-24,-14,-0.1,8])
        dy=.2
        #yticks(arange(ymin,ymax+dy,dy))
        ax=gca()
        s='$'+self.prefix+'$'
        text(0.06,.85,s,horizontalalignment='left',transform=ax.transAxes,fontsize=14)
        axvline(x=10,color='k',ls=':')
        if plotsingle:
            xlabel('$log_{10}(M_* \ (M_\odot)) $',fontsize=16)
            ylabel('$M_{NUV} \ -  \ M_r $',fontsize=16)
        #plotcolormaglines()



    def plotsizemass(self,shapes,colors,plotsingle=1,convflag=0):
        if plotsingle:
            figure()

        if convflag:
            y=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.cre1err*mipspixelscale/self.n.SERSIC_TH50
        else:
            y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50
        #size=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
        #size=self.sex24.FLUX_RADIUS2/self.n.SERSIC_TH50
        mass=log10(self.stellarmass)
        flag=self.galfitflag & self.member
        plot(mass[flag],size[flag],'ko',marker=shapes,color=colors,label=self.prefix,markersize=8)
        xc=mass[flag]
        yc=y[flag]
        flag=self.bluefieldsample | self.greenfieldsample | self.redfieldsample
        flag=self.galfitflag & ~self.member
        plot(mass[flag],size[flag],'wo',marker=shapes,color=colors,mfc='None',label='_nolegend_',markersize=8)
        xf=mass[flag]
        yf=size[flag]
        return xc,yc,xf,yf

    def plotsizelocalmassdensity(self,shapes,colors,plotsingle=1,convflag=0):
        if plotsingle:
            figure()
        colors='0.5'
        if convflag:
            y=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.cre1err*mipspixelscale/self.n.SERSIC_TH50
        else:
            y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50
        #size=self.sex24.FLUX_RADIUS2/self.n.SERSIC_TH50
        x=log10(self.ld.RHOMASS)
        #x=log10(self.ld.SIGMA_NN)
        flag=self.sampleflag & self.dvflag
        plot(x[flag],y[flag],'ko',marker=shapes,color=colors,label=self.prefix,markersize=8)


        errorbar(x[flag],y[flag],yerr=yer[flag],fmt=None,ecolor=colors)

        return x[flag],y[flag]

    def plotsizelocaldensity(self,shapes,colors,plotsingle=1,ld=0,convflag=0,sbflag=0):
        if plotsingle:
            figure()
        #y=self.size24/self.n.SERSIC_TH50
        y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
        if convflag:
            y=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
        #size=self.sex24.FLUX_RADIUS2/self.n.SERSIC_TH50
        #x=log10(self.ld.RHOMASS)
        if ld == 0:
            x=log10(self.ld.SIGMA_NN)
        elif ld == 1:
            x=log10(self.ld.SIGMA_5)

        elif ld == 2:
            x=log10(self.ld.SIGMA_10)
        #flag=self.sample24flag & self.dvflag
        flag=self.sampleflag & self.dvflag
        plot(x[flag],y[flag],'ko',marker=shapes,color=colors,mec=colors,label=self.prefix,markersize=8)
        if convflag:
            yer=self.galfit24.cre1err*mipspixelscale/self.n.SERSIC_TH50
        else:
            yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50
        errorbar(x[flag],y[flag],yerr=yer[flag],fmt=None,ecolor=colors)
        if plotsingle:
            title(self.prefix)
            axis([0,2.5,-.1,2.5])
            r,p=spearman(x[flag],y[flag])            
        return x[flag],y[flag]

    def plotsizeradius(self,shapes,colors,plotsingle=1,convflag=0):
        if plotsingle:
            figure()
        colors='0.5'

        if convflag:
            #y=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            #yer=self.galfit24.cre1err*mipspixelscale/self.n.SERSIC_TH50
            y=self.galfit24.cre1*mipspixelscale/(self.gim2d.Rhlr_1/self.gim2d.Scale_1)
            yer=self.galfit24.cre1err*mipspixelscale/(self.gim2d.Rhlr_1/self.gim2d.Scale_1)
        else:
            y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50

        #y=self.size24/self.n.SERSIC_TH50
        #y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50

        #size=self.sex24.FLUX_RADIUS2/self.n.SERSIC_TH50
        #x=log10(self.ld.RHOMASS)
        x=self.drR200
        #flag=self.sample24flag & self.dvflag
        flag=self.sampleflag & self.dvflag & self.gim2d.matchflag
        plot(x[flag],y[flag],'ko',marker=shapes,color=colors,label=self.prefix,markersize=8)
        #yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50
        errorbar(x[flag],y[flag],yerr=yer[flag],fmt=None,ecolor=colors)
        if plotsingle:
            title(self.prefix)
            print self.prefix
            r,p=spearman(x[flag],y[flag])
        return x[flag],y[flag]

    def plotsizedist3d(self,shapes,colors,plotsingle=1,convflag=0):
        if plotsingle:
            figure()
        #y=self.size24/self.n.SERSIC_TH50

        if convflag:
            y=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.cre1err*mipspixelscale/self.n.SERSIC_TH50
        else:
            y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50

        #y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50

        #size=self.sex24.FLUX_RADIUS2/self.n.SERSIC_TH50
        #x=log10(self.ld.RHOMASS)
        x=self.dist3d
        #flag=self.sample24flag & self.dvflag
        flag=self.sampleflag & self.dvflag
        plot(x[flag],y[flag],'ko',marker=shapes,color=colors,label=self.prefix,markersize=8)
        #yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50
        errorbar(x[flag],y[flag],yerr=yer[flag],fmt=None,ecolor=colors)

        return x[flag],y[flag]
    

    def plotsizecolor(self,colors,shapes,plotsingle=1,convflag=0):
        if plotsingle:
            figure()
        colors='0.5'

        if convflag:
            y=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.cre1err*mipspixelscale/self.n.SERSIC_TH50
        else:
            y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50

        #y=self.size24/self.n.SERSIC_TH50
        #y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
        #y=self.n.SERSIC_TH50
        #x=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]

        #x=self.n.ABSMAG[:,1]-self.sex24.MAG_BEST
        #x=22.5-2.5*log10(self.n.NMGY[:,1])-self.sex24.MAG_BEST
        x=self.n.ABSMAG[:,1]-self.M24
        #x=self.sex24.MAG_BEST
        #yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50
        flag=self.sampleflag & self.member
        plot(x[flag],y[flag],'ko',marker=shapes,color=colors,label=self.prefix,markersize=8)
        errorbar(x[flag],y[flag],yerr=yer[flag],fmt=None,ecolor=colors)
        flag2=self.galfitflag & self.field

        if sum(flag2)>0:
            plot(x[flag2],y[flag2],'ko',marker=shapes,color=fieldcolor,mec=fieldcolor,label='_nolegend_',markersize=8)
            errorbar(x[flag2],y[flag2],yerr=yer[flag2],fmt=None,ecolor=fieldcolor)
        
        return x[flag],y[flag],x[flag2],y[flag2]


    def plotNUV24ssfr(self,colors,shapes,plotsingle=1):
        if plotsingle:
            figure()
        colors='0.5'

        #y=self.size24/self.n.SERSIC_TH50
        col=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
        #y=self.n.SERSIC_TH50
        #x=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]

        #x=self.n.ABSMAG[:,1]-self.sex24.MAG_BEST
        #x=22.5-2.5*log10(self.n.NMGY[:,1])-self.sex24.MAG_BEST
        y=self.n.ABSMAG[:,1]-self.M24#log10(self.stellarmass)
        x=self.M24-log10(self.stellarmass)
        #x=self.sex24.MAG_BEST
        #yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50
        flag=self.sampleflag & self.member
        sp=scatter(x[flag],y[flag],marker=shapes,c=col[flag],vmin=.6,vmax=1.1,label=self.prefix,s=40)
        if self.prefix.find('MKW11') > -1:
            colorbar(sp)
        #errorbar(x[flag],y[flag],yerr=yer[flag],fmt=None,ecolor=colors)
        flag2=self.galfitflag & self.field

        if sum(flag2)>0:
            scatter(x[flag],y[flag],marker=shapes,c=col[flag],vmin=.2,vmax=3,label='_nolegend_',s=40)
        return x[flag],y[flag],x[flag2],y[flag2]


    def plotsizemasscolor(self,i,plotsingle=1):
        if plotsingle:
            figure()
        size=self.size24/self.n.SERSIC_TH50
        mass=log10(self.stellarmass)
        if i ==0:
            cflag=self.blueclustersample
            fflag=self.bluefieldsample
            pcolor='blue'
        elif i ==1:
            cflag=self.greenclustersample
            fflag=self.greenfieldsample
            pcolor='green'
        elif i == 2:
            cflag=self.redclustersample
            fflag=self.redfieldsample
            pcolor='red'                
        plot(mass[cflag],size[cflag],'ko',mfc=pcolor,mec=pcolor,label='_nolegend_',markersize=4)
        xc=mass[cflag]
        yc=size[cflag]
        plot(mass[fflag],size[fflag],'ks',mec=pcolor,mfc='None',label='_nolegend_',markersize=4)
        xf=mass[fflag]
        yf=size[fflag]
        return xc,yc,xf,yf

    def plotR24vsRe(self,shapes,colors,colorflag,fieldcolorflag,plotsingle=1):
        if plotsingle:
            figure()
        y=self.size24
        #y=self.galfit24.re1
        x=self.n.SERSIC_TH50
        flag=(colorflag) & (x < 100.)
        plot(x[flag],y[flag],'ko',marker=shapes,color=colors,label='_nolegend_',markersize=5)
        xc=x[flag]
        yc=y[flag]
        #flag=baseflag & ~self.member
        #flag=(self.bluefieldsample | self.greenfieldsample) & (x < 100.)
        flag=(fieldcolorflag) & (x < 100.)
        plot(x[flag],y[flag],'ws',marker=shapes,mec=colors,color=colors,mfc='None',label='_nolegend_',markersize=5)
        xf=x[flag]
        yf=y[flag]
        return xc,yc,xf,yf



    def plotRe24vsReold(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale
        jyerr=(y-y2)/2.

        x=self.n.SERSIC_TH50
        z=25.*ones(len(self.snr24))
        #z=self.snr24
        keepflag=self.sampleflag #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        errs=np.array(zip(jyerr[keepflag],zeros(sum(keepflag))),'f')
        errs2=np.array(zip(zeros(sum(keepflag)),jyerr[keepflag]),'f')
        #errorbar(x[keepflag],y[keepflag],errs.T,fmt=None,ecolor='0.5',capsize=0)
        #errorbar(x[keepflag],y2[keepflag],errs2.T,fmt=None,ecolor='0.5',capsize=0)

        sp=scatter(x[keepflag],y2[keepflag],marker='o',s=2.*z[flag],c='k',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)
        errorbar(x[keepflag],y2[keepflag],y2err[keepflag],fmt=None,ecolor='k')
        #        flag=keepflag & self.field 
        #if sum(flag > 0):


            #sp=scatter(x[flag],y[flag],marker='^',s=z[flag],c='w',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)

        flag=keepflag & self.nearfield
        #if sum(flag > 0):
        #    sp=scatter(x[flag],y2[flag],marker='o',s=2.*z[flag],c='k',edgecolors='k')
            #errorbar(x[flag],y2[flag],yerr[flag],fmt=None,ecolor='k')
            #sp=scatter(x[flag],y[flag],marker='^',s=z[flag],c='w',edgecolors='k')
            #flag=keepflag & self.member
            #if sum(flag > 0):
            #sp=scatter(x[flag],y2[flag],marker='o',s=2.*z[flag],c='k',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)
            #errorbar(x[flag],y2[flag],yerr[flag],fmt=None,ecolor='k')
            #sp=scatter(x[flag],y[flag],marker='^',s=z[flag],c='w',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)

        #flag=self.sampleflag & ~self.field & ~self.member
        #if sum(flag > 0):
        #    sp=scatter(x[flag],y[flag],marker='o',s=1.5*z[flag],c='None',edgecolors='b')#,c=concentration[keepflag])#,cmap=jet)

        #flag=keepflag & self.greenflag & ~self.member
        #if sum(flag > 0):
        #    sp=scatter(x[flag],y[flag],marker='o',s=z[flag],c='None',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)
        #flag=keepflag & self.greenflag & self.member
        #if sum(flag) > 0:
        #    sp=scatter(x[flag],y[flag],marker='o',s=z[flag],c='k',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)
        #flag=keepflag & self.redflag & ~self.member
        #if sum(flag) > 0:
        #    sp=scatter(x[flag],y[flag],marker='o',s=z[flag],c='None',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)
        #flag=keepflag & self.redflag & self.member
        #if sum(flag) > 0:
        #    sp=scatter(x[flag],y[flag],marker='o',s=z[flag],c='k',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)

        #sp=scatter(x[keepflag],y[keepflag],marker='o',s=self.dist3d[keepflag]*10,c=concentration[keepflag])#,cmap=jet)
        #cb=colorbar(sp)
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        xl=arange(1,100)
        plot(xl,xl,'k-')

        plot(xl,truncation_fraction*xl,color='r',ls='--')
        #plot(xl,.7*xl,color='b',ls='--')
        #axvline(x=2.*2.45,color='k',ls=':')
        #axvline(x=self.mingalaxysize,color='k',ls=':')
        #axhline(y=2.*2.45,color='k',ls=':')
        gca().set_yscale('log')
        gca().set_xscale('log')
        axis([1,120,.3,100])
        if plotsingle:
            xlabel('$ R_e \ NSA \ (arcsec)$',fontsize=22)
            ylabel('$ R_e \ 24 \mu m \ (arcsec)$',fontsize=22)

    def plotRe24vsRe(self,plotsingle=1,sbcutobs=20.5,colorflag=0):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale
        jyerr=(y-y2)/2.

        x=self.n.SERSIC_TH50
        z=30.*ones(len(self.snr24))
        #z=self.snr24
        keepflag=self.sampleflag & self.member & (self.sb_obs < sbcutobs)#& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        #sp=scatter(self.n.SERSIC_TH50[self.galfitflag],self.galfit24.cre1[self.galfitflag]*mipspixelscale,marker='o',s=z[keepflag],c='k',edgecolors='k')#,c=concentration[keepflag])#,cmap=jet)
        if colorflag:
            scatter(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,s=30,c=self.sb_obs[keepflag],vmin=sbmin,vmax=sbmax)
        else:
            plot(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,'ko')

        errorbar(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,self.galfit24.cre1err[keepflag]*mipspixelscale,fmt=None,ecolor='k')
        keepflag=self.sampleflag & ~self.member & (self.sb_obs < sbcutobs)
        if colorflag:
            scatter(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,s=30,c=self.sb_obs[keepflag],vmin=sbmin,vmax=sbmax)
        else:
            plot(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,'ko',color='0.5')
        errorbar(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,self.galfit24.cre1err[keepflag]*mipspixelscale,fmt=None,ecolor='0.5')

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        xl=arange(1,100)
        plot(xl,xl,'k-')

        plot(xl,truncation_fraction*xl,color='0.5',ls='--')
        gca().set_yscale('log')
        gca().set_xscale('log')
        axis([1,120,.3,100])
        if plotsingle:
            xlabel('$ R_e \ NSA \ (arcsec)$',fontsize=22)
            ylabel('$ R_e \ 24 \mu m \ (arcsec)$',fontsize=22)


    def plotRe24vsmag(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale
        jyerr=(y-y2)/2.

        x=self.galfit24.mag1
        z=30.*ones(len(self.snr24))
        #z=self.snr24
        keepflag=self.sampleflag #& self.member 
        sp=scatter(x[keepflag],y[keepflag],marker='o',s=z[keepflag],c=self.sb_obs[keepflag],edgecolors='k',vmin=13,vmax=25)#,c=concentration[keepflag])#,cmap=jet)
        #errorbar(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,self.galfit24.cre1err[keepflag]*mipspixelscale,fmt=None,ecolor='k')
        keepflag=self.sampleflag & ~self.member 
        #plot(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,'ko',color='0.5')
        #errorbar(self.n.SERSIC_TH50[keepflag],self.galfit24.cre1[keepflag]*mipspixelscale,self.galfit24.cre1err[keepflag]*mipspixelscale,fmt=None,ecolor='0.5')

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        xl=arange(11,19)
        yl=(1/pi*10**((22.-xl)/5))
        plot(xl,yl,'k-')

        #plot(xl,truncation_fraction*xl,color='0.5',ls='--')
        gca().set_yscale('log')
        #gca().set_xscale('log')
        axis([10,20,1,150])
        if plotsingle:
            xlabel('$ mag \ 24 $',fontsize=22)
            ylabel('$ R_e \ 24 \mu m \ (arcsec)$',fontsize=22)
            cb=colorbar(sp)
            

    def plotRe24vsReoneplot(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale
        jyerr=(y-y2)/2.

        x=self.n.SERSIC_TH50
        z=25.*ones(len(self.snr24))
        #z=self.snr24
        keepflag=self.sampleflag #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        errs=np.array(zip(jyerr[keepflag],zeros(sum(keepflag))),'f')
        errs2=np.array(zip(zeros(sum(keepflag)),jyerr[keepflag]),'f')
        #errorbar(x[keepflag],y[keepflag],errs.T,fmt=None,ecolor='0.5',capsize=0)
        #errorbar(x[keepflag],y2[keepflag],errs2.T,fmt=None,ecolor='0.5',capsize=0)

        flag=keepflag & self.field 
        if sum(flag > 0):
            sp=scatter(x[flag],y2[flag],marker='o',s=z[flag],c='b',edgecolors='b')#,c=concentration[keepflag])#,cmap=jet)
            errorbar(x[flag],y2[flag],yerr[flag],fmt=None,ecolor='b')
            #sp=scatter(x[flag],y[flag],marker='^',s=z[flag],c='w',edgecolors='b')#,c=concentration[keepflag])#,cmap=jet)
        xb=x[flag]
        yb=y2[flag]
        flag=keepflag & self.nearfield
        if sum(flag > 0):
            sp=scatter(x[flag],y2[flag],marker='o',s=z[flag],c='g',edgecolors='g')
            errorbar(x[flag],y2[flag],yerr[flag],fmt=None,ecolor='g')
            #sp=scatter(x[flag],y[flag],marker='^',s=z[flag],c='w',edgecolors='g')
        xg=x[flag]
        yg=y2[flag]

        flag=keepflag & self.member
        if sum(flag > 0):
            sp=scatter(x[flag],y2[flag],marker='o',s=z[flag],c='r',edgecolors='r')#,c=concentration[keepflag])#,cmap=jet)
            errorbar(x[flag],y2[flag],yerr[flag],fmt=None,ecolor='r')
            #sp=scatter(x[flag],y[flag],marker='^',s=z[flag],c='w',edgecolors='r')#,c=concentration[keepflag])#,cmap=jet)
        xr=x[flag]
        yr=y2[flag]

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        xl=arange(1,100)
        plot(xl,xl,'k-')

        plot(xl,truncation_fraction*xl,color='0.5',ls='--')
        plot(xl,.8*xl,color='0.5',ls='--')
        #axvline(x=2.*2.45,color='k',ls=':')
        axvline(x=self.mingalaxysize,color='k',ls=':')
        #axhline(y=2.*2.45,color='k',ls=':')
        gca().set_yscale('log')
        gca().set_xscale('log')
        axis([1,120,.3,100])
        if plotsingle:
            xlabel('$ R_e \ NSA \ (arcsec)$',fontsize=22)
            ylabel('$ R_e \ 24 \mu m \ (arcsec)$',fontsize=22)

        return xb,yb,xg,yg,xr,yr
    def plotRe24vsRegim2d(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale
        jyerr=(y-y2)/2.

        x=self.gim2d.Rhlr_2/self.gim2d.Scale_2
        keepflag=self.sampleflag & self.gim2d.matchflag #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        errs=np.array(zip(jyerr[keepflag],zeros(sum(keepflag))),'f')
        errs2=np.array(zip(zeros(sum(keepflag)),jyerr[keepflag]),'f')
        errorbar(x[keepflag],y[keepflag],errs.T,fmt=None,ecolor='0.5',capsize=0)
        errorbar(x[keepflag],y2[keepflag],errs2.T,fmt=None,ecolor='0.5',capsize=0)
        plot(x[keepflag],y[keepflag],'k^',mfc=None)
        plot(x[keepflag],y2[keepflag],'ro')
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        xl=arange(1,100)
        plot(xl,xl,'k-')

        plot(xl,truncation_fraction*xl,color='0.5',ls='--')
        plot(xl,.8*xl,color='0.5',ls='--')
        #axvline(x=2.*2.45,color='k',ls=':')
        axvline(x=self.mingalaxysize,color='k',ls=':')
        #axhline(y=2.*2.45,color='k',ls=':')
        gca().set_yscale('log')
        gca().set_xscale('log')
        axis([1,120,.3,100])
        if plotsingle:
            xlabel('$ R_e \ GIM2D \ (arcsec)$',fontsize=22)
            ylabel('$ R_e \ 24 \mu m \ (arcsec)$',fontsize=22)

    def plotsizevsBT(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.sizeratio
        x=self.gim2d.B_T_r
        keepflag=self.sampleflag & self.gim2d.matchflag & ~self.member #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        plot(x[keepflag],y[keepflag],'k^',mfc=None)
        keepflag=self.sampleflag & self.gim2d.matchflag & self.member #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        plot(x[keepflag],y[keepflag],'ro',mfc=None)
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        if plotsingle:
            xlabel('$ GIM2D \ B/T $',fontsize=22)
            ylabel('$ R_e(24)/R_e(NSA)$',fontsize=22)



    def plotsmoothnessvsBT(self,plotsingle=1):
        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        flag=self.sampleflag & self.gim2d.matchflag
        y=self.gim2d.S2g_1   
        z=self.sizeratio
        x=self.gim2d.B_T_r
        keepflag=self.sampleflag & self.gim2d.matchflag 
        #sp=scatter(x[flag],y[flag],marker='o',s=50,c=z[flag],vmin=0,vmax=1.,edgecolor='None')#,c=concentration[keepflag])#,cmap=jet)
        #if plotsingle:
        #    cb=colorbar(sp)
        flag=keepflag & self.member
        plot(x[flag],y[flag],'r^',mfc=None)
        flag=keepflag & ~self.member
        plot(x[flag],y[flag],'bo',mfc=None)
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        if plotsingle:
            xlabel('$ GIM2D \ B/T $',fontsize=22)
            ylabel('$ GIM2D Smoothness$',fontsize=22)

    def plotRe24vsReSNR(self,plotsingle=1):
        # with SNR for colorbar

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale
        jyerr=(y-y2)/2.

        x=self.n.SERSIC_TH50
        z=(self.snr24)
        #z=self.snr24
        keepflag=self.sampleflag #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        errs=np.array(zip(jyerr[keepflag],zeros(sum(keepflag))),'f')
        errs2=np.array(zip(zeros(sum(keepflag)),jyerr[keepflag]),'f')
        errorbar(x[keepflag],y[keepflag],errs.T,fmt=None,ecolor='0.5',capsize=0)
        errorbar(x[keepflag],y2[keepflag],errs2.T,fmt=None,ecolor='0.5',capsize=0)
        flag=keepflag
        sp=scatter(x[flag],y2[flag],marker='o',s=50,c=z[flag],edgecolor='None')#,c=concentration[keepflag])#,cmap=jet)
        errorbar(x[flag],y2[flag],yerr[flag],fmt=None)
        sp=scatter(x[flag],y[flag],marker='^',s=50,c=z[flag],edgecolor='None')#,c=concentration[keepflag])#,cmap=jet)
        cb=colorbar(sp)

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        xl=arange(1,100)
        plot(xl,xl,'k-')

        plot(xl,truncation_fraction*xl,color='0.5',ls='--')
        plot(xl,.8*xl,color='0.5',ls='--')
        #axvline(x=2.*2.45,color='k',ls=':')
        axvline(x=self.mingalaxysize,color='k',ls=':')
        #axhline(y=2.*2.45,color='k',ls=':')
        gca().set_yscale('log')
        gca().set_xscale('log')
        axis([1,120,.3,100])
        if plotsingle:
            xlabel('$ R_e \ NSA \ (arcsec)$',fontsize=22)
            ylabel('$ R_e \ 24 \mu m \ (arcsec)$',fontsize=22)

    def plotdiffRevsSNR(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale

        x=self.n.SERSIC_TH50
        yplot = (y2-x)/x
        yplot_scaled = (y2*mipspixelscale-x)/x
        yplot2 = (y-x)/x
        xplot=self.snr24
        z=25.*ones(len(self.snr24))
        keepflag=self.sampleflag #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        plot(xplot[keepflag],yplot[keepflag],'ko')
        plot(xplot[keepflag],yplot_scaled[keepflag],'b*')
        errorbar(xplot[keepflag],yplot[keepflag],y2err[keepflag]/xplot[keepflag],fmt=None,ecolor='0.5')
        plot(xplot[keepflag],yplot2[keepflag],'wo')
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        #gca().set_yscale('log')
        gca().set_xscale('log')
        axhline(y=0)
        axis([2,210,-1.4,1.4])
        if plotsingle:
            xlabel('$ SNR 24 $',fontsize=22)
            ylabel('$ |R_e \ 24 \mu m - R_e|/R_e $',fontsize=22)

    def plotdiffRevsRe(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.galfit24.re1*mipspixelscale
        yerr=self.galfit24.re1err*mipspixelscale
        y2=self.galfit24.cre1*mipspixelscale
        y2err=self.galfit24.cre1err*mipspixelscale

        x=self.n.SERSIC_TH50
        yplot = (y2-x)/x
        yplot_scaled = (y2*mipspixelscale-x)/x
        yplot2 = (y-x)/x
        xplot=self.snr24
        z=25.*ones(len(self.snr24))
        keepflag=self.sampleflag #& ~self.galfit24.cnumerical_error_flag24 #& self.dvflag
        plot(x[keepflag],yplot[keepflag],'ko')
        #plot(x[keepflag],yplot_scaled[keepflag],'b*')
        errorbar(x[keepflag],yplot[keepflag],y2err[keepflag]/xplot[keepflag],fmt=None,ecolor='0.5')
        #plot(x[keepflag],yplot2[keepflag],'wo')
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        #gca().set_yscale('log')
        gca().set_xscale('log')
        axhline(y=0)
        axis([1,90,-1.4,1.4])
        if plotsingle:
            xlabel('$ R_e $',fontsize=22)
            ylabel('$ |R_e \ 24 \mu m - R_e|/R_e $',fontsize=22)

    def plotRevsn(self,plotsingle=1,convflag=1):

        keepflag=self.sampleflag #& self.dvflag
        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)

        if convflag:

            x2=self.galfit24.cnsersic1[keepflag]
            x2err=self.galfit24.cnsersic1err[keepflag]
            y2=self.galfit24.cre1[keepflag]*mipspixelscale
            y2err=self.galfit24.cre1err[keepflag]*mipspixelscale

        else:
            x2=self.galfit24.nsersic1[keepflag]
            x2err=self.galfit24.nsersic1err[keepflag]
            y2=self.galfit24.re1[keepflag]*mipspixelscale
            y2err=self.galfit24.re1err[keepflag]*mipspixelscale

        ny=self.n.SERSIC_TH50[keepflag]
        nx=self.n.SERSIC_N[keepflag]

        errorbar(x2,y2,yerr=y2err,xerr=x2err,fmt='o',color='r')
        #errorbar(x,y,yerr=yerr,xerr=xerr,fmt='o',color='r')
        plot(nx,ny,'bo')
        for i in range(len(x2)):
            #xl=array([nx[i],x2[i]])
            #yl=array([ny[i],
            arrow(nx[i],ny[i],x2[i]-nx[i],y2[i]-ny[i],lw=.5)#,head_width=.2,head_length=.4,ec='k',fc='k')
        #errorbar(nx,ny,nyerr,fmt='o')
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        axis([0.,9.,.3,30.])
        if plotsingle:
            xlabel('$ Sersic \ index $',fontsize=22)
            ylabel('$ R_e  (arcsec)$',fontsize=22)
        ax.set_yscale('log')
        #ax.set_xscale('log')

    def plotRestellarmass(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        x=self.stellarmass

        y=self.n.SERSIC_TH50

        flag= self.field & self.On24ImageFlag & self.spiralflag
        if sum(flag > 0):
            sp=plot(x[flag],y[flag],'bo')
        flag=self.On24ImageFlag & self.member& self.spiralflag
        if sum(flag > 0):
            sp=plot(x[flag],y[flag],'ro')

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        gca().set_yscale('log')
        gca().set_xscale('log')
        axhline(self.mingalaxysize,color='k',ls='--')
        axvline(self.mingalaxymass,color='k',ls='--')

        axis([1.e8,1.e13,.1,100])
        if plotsingle:
            xlabel('$ M_*/M_\odot $',fontsize=22)
            ylabel('$ R_e(r) \ (arcsec)$',fontsize=22)
    def plotRermag(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        x=self.rmag

        y=self.n.SERSIC_TH50

        flag= self.field & self.On24ImageFlag & self.spiralflag
        if sum(flag > 0):
            sp=plot(x[flag],y[flag],'bo')
        flag=self.On24ImageFlag & self.member& self.spiralflag
        if sum(flag > 0):
            sp=plot(x[flag],y[flag],'ro')

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        gca().set_yscale('log')
        #gca().set_xscale('log')
        axhline(self.mingalaxysize,color='k',ls='--')
        axvline(self.rmagcut,color='k',ls='--')

        axis([13,23,.1,100])
        if plotsingle:
            xlabel('$ M_*/M_\odot $',fontsize=22)
            ylabel('$ R_e(r) \ (arcsec)$',fontsize=22)

    def plotlirRe(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        y=self.ce.LIR

        x=self.n.SERSIC_TH50
        baseflag=self.On24ImageFlag & self.sex24.MATCHFLAG24 & (self.snr24 > snr24cut) & self.spiralflag
        flag=baseflag & self.field
        if sum(flag > 0):
            plot(x[flag],y[flag],'bo')
        flag=baseflag & self.member
        if sum(flag > 0):
            plot(x[flag],y[flag],'ro')
        baseflag=self.On24ImageFlag & self.sex24.MATCHFLAG24 & (self.snr24 < snr24cut) & self.spiralflag
        plot(x[baseflag],y[baseflag],'k.')
        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        gca().set_yscale('log')
        gca().set_xscale('log')
        axvline(self.mingalaxysize,color='k',ls='--')
        axhline(self.Lir80,color='k',ls='--')
        #axis([3,60,3,90])
        axis([.1,100,8.e7,2.e11])
        if plotsingle:
            ylabel('$ L_{IR} $',fontsize=22)
            xlabel('$ R_e(r) \ (arcsec)$',fontsize=22)


    def plotlirstellarmass(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        x=self.stellarmass

        y=self.ce.LIR
        baseflag=self.On24ImageFlag & self.sex24.MATCHFLAG24 & (self.snr24 > snr24cut) & self.spiralflag
        flag=baseflag & self.field 
        if sum(flag > 0):
            plot(x[flag],y[flag],'bo')
        flag=baseflag & self.member
        if sum(flag > 0):
            plot(x[flag],y[flag],'ro')
        baseflag=self.On24ImageFlag & self.sex24.MATCHFLAG24 & (self.snr24 < snr24cut) & self.spiralflag
        plot(x[baseflag],y[baseflag],'k.')

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        axvline(self.mingalaxymass,color='k',ls='--')
        axhline(self.Lir80,color='k',ls='--')

        gca().set_yscale('log')
        gca().set_xscale('log')
        axis([1.e8,1.e13,8.e7,2.e11])

        if plotsingle:
            xlabel('$ M_*/M_\odot $',fontsize=22)
            ylabel('$ L_{IR}$',fontsize=22)


    def plotlirrmag(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
            title(self.prefix)
        x=self.rmag

        y=self.ce.LIR
        baseflag=self.On24ImageFlag & self.sex24.MATCHFLAG24 & (self.snr24 > snr24cut) & self.spiralflag
        flag=baseflag & self.field 
        if sum(flag > 0):
            plot(x[flag],y[flag],'bo')
        flag=baseflag & self.member
        if sum(flag > 0):
            plot(x[flag],y[flag],'ro')
        baseflag=self.On24ImageFlag & self.sex24.MATCHFLAG24 & (self.snr24 < snr24cut) & self.spiralflag
        plot(x[baseflag],y[baseflag],'k.')

        ax=gca()
        fsize=14
        text(.05,.85,'$'+self.clustername+'$',fontsize=fsize,transform=ax.transAxes,horizontalalignment='left')

        axvline(self.rmagcut,color='k',ls='--')
        axhline(self.Lir80,color='k',ls='--')

        gca().set_yscale('log')
        #gca().set_xscale('log')
        axis([13.,23.,8.e7,2.e11])

        if plotsingle:
            xlabel('$ M_*/M_\odot $',fontsize=22)
            ylabel('$ L_{IR}$',fontsize=22)

            

    def plotc24vsRe(self,plotsingle=1):

        if plotsingle:
            figure(figsize=(8,6))
        keep24flag=self.snr24 > 2
        keepflag=(self.On24ImageFlag & keep24flag & ~self.galfit24.numerical_error_flag24) #& ~self.ellipticalflag)
        y=self.galfit24.re1*mipspixelscale
        x=self.n.SERSIC_TH50

        #plot(x[keepflag],y[keepflag],'k.')
        flag=keepflag & self.member
        #plot(x[flag],y[flag],'r.')
        scatter(x[keepflag],y[keepflag],marker='o',s=self.dist3d[keepflag]*10,alpha=.2)
        xl=arange(1,500)
        plot(xl,xl,'k:')
        axvline(x=5)
        axhline(y=5)
        #gca().set_yscale('log')
        #gca().set_xscale('log')
        axis([0,20,0,20])
        xlabel('$ R_e \ NSA \ (arcsec)$',fontsize=22)
        ylabel('$ R_e \ 24 \mu m \ (arcsec)$',fontsize=22)



    def plotdvdr(self,usefwhm24=0,plotsingle=1):
        if plotsingle:
            figure()
        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)/self.r200deg
        dv=((self.n.ZDIST*3.e5)-self.biweightvel)/self.biweightscale

        #membflag=(dr/self.r200deg < 1) & (abs(dv) < 3.*self.biweightscale)

        plot(dr[self.On24ImageFlag],dv[self.On24ImageFlag],'k.', color='0.5',markersize=3,label='All')

        if usefwhm24:
            flag=self.sex24.MATCHFLAG24 & self.On24ImageFlag & ~self.agnflag & ~self.ellipticalflag
            ratio=self.size24/self.n.SERSIC_TH50
        else:
            ratio=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50*20
        
        #print scale[plotflag]
        #print scale[plotflag]
        #plot(dr[plotflag],dv[plotflag],'bo',markersize=6)
        scatter(dr[self.galfitflag],dv[self.galfitflag],s=ratio[self.galfitflag],color='r',alpha=0.5,label='24um')

        #plot(dr[flag],dv[flag],'r.')

        flag=self.On24ImageFlag & self.spiralflag
        plot(dr[flag],dv[flag],'b.',mfc='b',mec='b',markersize=3,label='Zoo Sp')
        
        #ymin=3500
        #ymax=14000
        #ymin=.014
        #ymax=.045
        #axis([0,2,ymin,ymax])
        #xmin,xmax=xlim()
        #xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)
        #yticks(arange(4000,ymax,4000,'i'),fontsize=10)
        ax=gca()
        text(.1,.85,'$'+self.clustername+'$',fontsize=14,transform=ax.transAxes,horizontalalignment='left')
        if plotsingle:
            xlabel('$R_{proj} \ (deg) $',fontsize=16)
            ylabel('$NSA \ zdist*c \ (km/s) $',fontsize=16)
        #axhline(self.biweightvel/3.e5,ls='-',color='k')
        #axhline((self.biweightvel+3*self.biweightscale)/3.e5,ls='--',color='k')
        #axhline((self.biweightvel-3*self.biweightscale)/3.e5,ls='--',color='k')
        #axvline(self.r200deg,ls=':',color='k')
        axhline(0,ls='-',color='k')
        axhline(3,ls='--',color='k')
        axhline(-3,ls='--',color='k')
        axvline(1,ls=':',color='k')
        if self.prefix.find('Herc') > -1:
            leg=legend(numpoints=1,loc='lower right',scatterpoints=1,markerscale=0.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
            for t in leg.get_texts():
                t.set_fontsize('small')

        #legend(loc='lower right',numpoints=1)


    def plotredshiftdr(self,usefwhm24=0,plotsingle=1):
        if plotsingle:
            figure()
        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        dv=(self.n.ZDIST)#-self.biweightvel)#/self.biweightscale

        flag=self.On24ImageFlag  & ~self.member & ~self.spiralflag
        #plot(dr[flag],dv[flag],'ko', mfc='None',markersize=5,label='Field')
        flag=self.On24ImageFlag   & ~self.spiralflag
        #hexbin(dr[flag],dv[flag],cmap=cm.Greys,gridsize=15,vmin=1,vmax=4)
        plot(dr[flag],dv[flag],'ko', color='k',mec='k',markersize=1,label='Non-Sp')

        flag=self.On24ImageFlag  & self.spiralflag
        #hexbin(dr[flag],dv[flag],cmap=cm.Greys,gridsize=15,vmin=1,vmax=4)
        #plot(dr[flag],dv[flag],'k.', color='b',markersize=5,label='Member',alpha=0.3)


        if usefwhm24:
            flag=self.sex24.MATCHFLAG24 & self.On24ImageFlag & ~self.agnflag & ~self.ellipticalflag
            ratio=self.size24/self.n.SERSIC_TH50
        else:
            ratio=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50*40

        flag=self.On24ImageFlag & self.spiralflag & self.member
        flag=self.sampleflag 
        #plot(dr[flag],dv[flag],'bo',mfc='b',mec='b',markersize=4,label='Zoo Sp')

        #flag=self.On24ImageFlag & self.spiralflag & self.field
        flag=self.On24ImageFlag & self.spiralflag #& self.member
        plot(dr[flag],dv[flag],'ro',mfc='r',mec='r',markersize=2,label='Sp')
        flag=self.sampleflag
        plot(dr[flag],dv[flag],'bo',mfc='b',mec='b',markersize=2,label='Sp+24um')
        
        #scatter(dr[self.sampleflag],dv[self.sampleflag],s=30,color=ratio,cmap=cm.jet_r,label='Re(24)/Re')

        #plot(dr[flag],dv[flag],'r.')




        
        ymin=3500
        ymax=14000
        ymin=.01
        ymax=.05
        axis([0,1.8,ymin,ymax])
        xmin,xmax=xlim()
        xt=arange(0,2,.5)
        xticks(xt,fontsize=10)
        yticks(arange(ymin+.005,ymax,.01),fontsize=10)
        #xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)
        #yticks(arange(4000,ymax,4000,'i'),fontsize=10)
        ax=gca()
        text(.1,.85,'$'+self.clustername+'$',fontsize=14,transform=ax.transAxes,horizontalalignment='left')
        if plotsingle:
            xlabel('$R_{proj} \ (deg) $',fontsize=12)
            ylabel('$NSA \ zdist*c \ (km/s) $',fontsize=12)
        axhline(self.biweightvel/3.e5,ls='-',color='k')
        axhline((self.biweightvel+3*self.biweightscale)/3.e5,ls='--',color='k')
        axhline((self.biweightvel-3*self.biweightscale)/3.e5,ls='--',color='k')
        axvline(1.*self.r200deg,ls=':',color='k')
        if self.prefix.find('Herc') > -1:
            leg=legend(numpoints=1,loc='lower right',scatterpoints=1,markerscale=1,borderpad=.0,labelspacing=.0,handletextpad=.0,borderaxespad=0)
            for t in leg.get_texts():
                t.set_fontsize('small')

        #legend(loc='lower right',numpoints=1)

    def plotredshiftdrcolorbar(self,usefwhm24=0,plotsingle=1):
        if plotsingle:
            figure()
        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        dv=(self.n.ZDIST)#-self.biweightvel)#/self.biweightscale

        flag=self.On24ImageFlag   & ~self.spiralflag
        plot(dr[flag],dv[flag],'ko',mfc='None',markersize=3,label='All')
        #flag=self.On24ImageFlag  & self.member & ~self.spiralflag
        #hexbin(dr[flag],dv[flag],cmap=cm.Greys,gridsize=15,vmin=1,vmax=4)
        #plot(dr[flag],dv[flag],'k.', color='0.5',markersize=4,label='Member')



        if usefwhm24:
            flag=self.sex24.MATCHFLAG24 & self.On24ImageFlag & ~self.agnflag & ~self.ellipticalflag
            ratio=self.size24/self.n.SERSIC_TH50
        else:
            ratio=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50*20

        #flag=self.On24ImageFlag & self.spiralflag & self.member
        #plot(dr[flag],dv[flag],'bo',mfc='b',mec='b',markersize=4,label='Zoo Sp')
        gcolor=self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]
        sp=scatter(dr[self.galfitflag],dv[self.galfitflag],s=30,c=gcolor[self.galfitflag],label='Spiral',cmap='jet',vmin=1,vmax=6)
        #cb=colorbar(sp)
        #flag=self.On24ImageFlag & self.spiralflag & self.field
        #plot(dr[flag],dv[flag],'bo',mfc='None',mec='b',markersize=5,label='_nolegend_')
        
        #scatter(dr[self.galfitflag],dv[self.galfitflag],s=ratio[self.galfitflag],color='r',alpha=0.5,label='24um')

        #plot(dr[flag],dv[flag],'r.')




        
        ymin=3500
        ymax=14000
        ymin=.014
        ymax=.045
        axis([0,1.8,ymin,ymax])
        xmin,xmax=xlim()
        xt=arange(0,2,.5)
        xticks(xt)
        #xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)
        #yticks(arange(4000,ymax,4000,'i'),fontsize=10)
        ax=gca()
        text(.1,.85,'$'+self.clustername+'$',fontsize=14,transform=ax.transAxes,horizontalalignment='left')
        if plotsingle:
            xlabel('$R_{proj} \ (deg) $',fontsize=16)
            ylabel('$NSA \ zdist*c \ (km/s) $',fontsize=16)
        axhline(self.biweightvel/3.e5,ls='-',color='k')
        axhline((self.biweightvel+3*self.biweightscale)/3.e5,ls='--',color='k')
        axhline((self.biweightvel-3*self.biweightscale)/3.e5,ls='--',color='k')
        axvline(1.*self.r200deg,ls=':',color='k')
        if self.prefix.find('Herc') > -1:
            leg=legend(numpoints=1,loc='lower right',scatterpoints=1,markerscale=2,borderpad=.2,labelspacing=.2,handletextpad=.2)
            for t in leg.get_texts():
                t.set_fontsize('small')

        #legend(loc='lower right',numpoints=1)

    def plotveldr(self):

        dr=sqrt((self.ra-self.clusterra)**2+(self.dec-self.clusterdec)**2)
        dv=(self.supervopt-self.biweightvel)

        membflag=(dr/self.r200deg < 1) & (abs(dv) < 3.*self.biweightscale)

        plot(dr,self.supervopt,'k.',markersize=3)
        plot(dr[membflag],self.supervopt[membflag],'bo',markersize=6,mfc='None',mec='b')
        plot(dr[self.On24ImageFlag],self.supervopt[self.On24ImageFlag],'ro',markersize=10,mfc='None',mec='r')
        ymin=3500
        ymax=14000
        axis([0,3,ymin,ymax])
        xmin,xmax=xlim()
        xticks(arange(round(xmin),xmax+1,1,'i'),fontsize=10)

        yticks(arange(4000,ymax,4000,'i'),fontsize=10)
        title(self.clustername,fontsize=12)
        axhline(self.biweightvel,ls='-',color='r')
        axhline(self.biweightvel+3*self.biweightscale,ls='--',color='r')
        axhline(self.biweightvel-3*self.biweightscale,ls='--',color='r')


    def plotvelhist(self):
        figure()
        bins=30
        x1=self.allvelocity
        (yhist,xhist,patches)=hist(x1,bins)
        xhist=xhist[0:len(xhist)-1]+0.5*(xhist[1]-xhist[0])
        mymean= average(x1)
        mystd=std(x1)
        print mystd
        norm=max(yhist)
        xmin=3000
        xmax=15000
        xplot=arange(xmin,xmax,50)
        y1=norm*exp(-((xplot -self.clustervel)**2)/(2*self.clustersigma**2))
        plot(xplot,y1,'r-')
        xlabel('Recession Velocity ')
        xlim(xmin,xmax)

        axvline(self.clustervel,ymin=0,ymax=60,color='r')


    def plotSFR24hist(self,plotsingle=1):
        #figure(1)
        if plotsingle:
            figure(figsize=(10,8))

        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.dvflag

	y=hist(log10(self.ce.SFR[flag]),bins=8,histtype='step',color='k')
	ax=gca()
	#ax.set_yscale('log')
        axis([-2.5,1.5,0,20])
        xmin,xmax=xlim()
        xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)


    def plotSFR24hist(self,plotsingle=1):
        #figure(1)
        if plotsingle:
            figure(figsize=(10,8))

        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.dvflag

	y=hist(log10(self.ce.SFR[flag]),bins=8,histtype='step',color='k')
	ax=gca()
	#ax.set_yscale('log')
        axis([-2.5,1.5,0,20])
        xmin,xmax=xlim()
        xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotLIRhist(self,plotsingle=1,htype='stepfilled',myhatch='\\',plotclname=1):
        if plotsingle:
            figure(figsize=(6,4))
            subplots_adjust(left=.15, bottom=.15, right=.95, top=.85)
        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.dvflag

        mybins=arange(7,12,.5)
        y=hist(log10(self.ce.LIR[flag]),bins=mybins,histtype=htype,hatch=myhatch,label=self.prefix,lw=2,color='0.7')
        ax=gca()
        xmin=7.25
        xmax=11.5
        if plotsingle:
            axis([xmin,xmax,0,13.9])
        else:
            axis([xmin,xmax,1,45])
        #xmin,xmax=xlim()
        axvline(x=log10(self.Lir80),color='k',ls='--')
        fsize=10
        if plotsingle:
            fsize=14
        xticks(arange(int(xmin),xmax+.5,2,'i'),fontsize=fsize)
        yticks(arange(0,45,10,'i'),fontsize=fsize)
        #ymin,ymax=ylim()
        if plotsingle:
            
            xmin=10.**(xmin)*Lsol*4.5e-44
            xmax=10.**(xmin)*Lsol*4.5e-44
            ax=gca()
            xaxissfrlog(ax)
        if plotsingle:
            fsize=20
        else:
            fsize=12
        if plotclname:
            text(.9,.8,'$ '+self.clustername+' $',fontsize=fsize,transform=ax.transAxes,horizontalalignment='right')
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)
        if plotsingle:
            xlabel('$ log_{10} (L_{IR}/L_\odot) $',fontsize=18)
            ylabel('$ N_{gal} $',fontsize=18)
            text(.5,1.1,'$ SFR \ (M_\odot/yr) $',fontsize=fsize,transform=ax.transAxes,horizontalalignment='center')
    def plotHImasshist(self):
        #figure(1)
        flag= self.HIflag & self.dvflag
        mybins=arange(8,11,.2)
	y=hist(log10(self.HImass[flag]),bins=mybins,histtype='step',color='k',label=self.prefix)
	ax=gca()
	#ax.set_yscale('log')

        xmin,xmax=xlim()
        xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        axis([8,11,1,55])
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotHIDefhist(self):
        #figure(1)
        flag= self.HIflag & self.sdssflag
        #mybins=arange(8,11,.2)
	y=hist((self.HIDef[flag]),bins=10,histtype='step',color='k')

        flag= self.HIflag &  self.sdssflag
        axvline(x=0,ls='--',color='r')
        #mybins=arange(8,11,.2)
	#y=hist((self.HIDef[flag]),bins=20,histtype='step',color='b')
        ax=gca()
	#ax.set_yscale('log')
        #axis([8,11,0,55])
        xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        ymin,ymax=ylim()
        axis([-2,2,0,ymax])
        title(self.clustername)
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotDiamHImass(self):
        flag= self.HIflag 
        # log10 HI mass on y axis 
	y=log10(self.HImass)
        # some measure of D25 on x axis
        x=self.logD25kpc
        # plot only those galaxies w/HI detection
        flag= self.HIflag & self.sdssflag
        plot(x[flag],y[flag],'.')
        # plot toribio et al relation
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        ax=gca()
        xmin,xmax=xlim()
        xline=arange(xmin,xmax+.1,.1)
        yline=8.72+1.25*(xline-log10(2.))
        plot(xline,yline,'r-')
        xlabel('$\mathrm{log_{10}(Diameter \ (D_{25}/kpc))}$',fontsize=16)
        ylabel('$\mathrm{log_{10}(M_{HI}/M_\odot)}$',fontsize=16)
        title(self.clustername)

    def plotMrHImass(self):
        flag= self.HIflag 
        # log10 HI mass on y axis 
	y=log10(self.HImass)
        # some measure of D25 on x axis
        x=self.sdssMr 
        # plot only those galaxies w/HI detection
        flag= self.HIflag & self.sdssflag
        plot(x[flag],y[flag],'.')
        # plot toribio et al relation
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        ax=gca()
        xmin,xmax=xlim()
        xline=arange(xmin,xmax+.1,.1)
        yline=6.44-0.18*xline
        plot(xline,yline,'r-')
        xlabel('$\mathrm{M_r}$',fontsize=16)
        ylabel('$\mathrm{log_{10}(M_{HI}/M_\odot)}$',fontsize=16)
        title(self.clustername)

    def plotgrHImass(self):
        # log10 HI mass on y axis 
	y=log10(self.HImass)
        # sdss deredened g-r color
        x=self.sdssgmag-self.sdssrmag
        # plot only those galaxies w/HI detection
        flag= self.HIflag & self.sdssflag
        plot(x[flag],y[flag],'.')
        # plot toribio et al relation
        # log(M_HI/Msun) = 8.72 + 1.25 log(D_25,r/kpc)
        ax=gca()
        xmin,xmax=xlim()
        xline=arange(xmin,xmax+.1,.1)
        yline=8.84+1.81*xline
        plot(xline,yline,'r-')
        xlabel('$\mathrm{g-r \ magnitude}$',fontsize=16)
        ylabel('$\mathrm{log_{10}(M_{HI}/M_\odot)}$',fontsize=16)
        title(self.clustername)
    def plotratioHImass(self,colors,shapes):
        #figure(1)
        #plotflag=self.apexflag & self.HIflag & ~self.agn2 & self.spiralflag
        #plotflag=self.apexflag  & ~self.agn2 & self.spiralflag
        x=self.HImass[self.plotratio90Flag]/self.stellarmass[self.plotratio90Flag]
        y=self.ratio90[self.plotratio90Flag]
        plot(x,y,'ko',color=colors,marker=shapes)
	#ax=gca()
	#ax.set_yscale('log')
        #axis([-2.5,1.5,0,22])
        #xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        #ymin,ymax=ylim()
        #title(self.clustername)
        return x.tolist(),y.tolist()
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotSFR24stellmass(self,colors,shapes):
        #figure(1)
        #plotflag=self.apexflag & self.HIflag & ~self.agn2 & self.spiralflag
        #plotflag=self.apexflag  & ~self.agn2 & self.spiralflag

        #flag=self.ellipseflag & self.spiralflag & self.member & ~self.agnflag
        #flag=self.ellipseflag & self.member & ~self.agnflag
        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.member & self.spiralflag
        #flag=self.sampleflag & self.member
        print self.prefix,' numb of IR members = ',sum(flag)
        x=self.stellarmass[flag]
        y=self.ce.SFR[flag]
        #yerr=self.SFR24err[flag]
        #plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=8)
        if self.prefix.find('MKW11')> -1:
            clabel='Clusters Sp'

        else:
            clabel='_nolegend_'

        plot(x,y,'ko',color=colors,marker=shapes,label=clabel,markersize=8)
        #errorbar(x,y,yerr,fmt=None,color=colors,label='_nolegend_')

        #flag=self.ellipseflag & self.spiralflag & ~self.member & ~self.agn1
        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.field & self.spiralflag
        #flag=self.sampleflag & ~self.member
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        x1=self.stellarmass[flag]
        y1=self.ce.SFR_ZDIST[flag]
        #yerr=self.SFR24err[flag]
        if self.prefix.find('MKW11')> -1:
            clabel='Field Sp'

        else:
            clabel='_nolegend_'
            #colors='0.3'
        #plot(x1,y1,'k.',markeredgecolor=colors,marker=shapes,markerfacecolor=colors,label=clabel,markersize=10,lw=3,alpha=0.2)
        plot(x1,y1,'k.',markeredgecolor=fieldColor,marker=shapes,markerfacecolor=fieldColor,label=clabel,markersize=10,lw=3,alpha=0.5)

	#ax=gca()
	#ax.set_yscale('log')
        #axis([-2.5,1.5,0,22])
        #xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        #ymin,ymax=ylim()
        #title(self.clustername)
        return x.tolist(),y.tolist(),x1.tolist(),y1.tolist()
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

    def plotsizestellmass(self,colors,shapes,convflag=0,gim2d=1):
        flag=self.sampleflag &  self.member
        print self.prefix,' numb of IR members = ',sum(flag)
        x=self.stellarmass
        #y=self.galfit24.re1[flag]*mipspixelscale/self.n.SERSIC_TH50[flag]
        if convflag:
            y=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.cre1err*mipspixelscale/self.n.SERSIC_TH50
        else:
            y=self.galfit24.re1*mipspixelscale/self.n.SERSIC_TH50
            yer=self.galfit24.re1err*mipspixelscale/self.n.SERSIC_TH50

        if gim2d:
            if convflag:
                y=self.galfit24.cre1*mipspixelscale/(self.gim2d.Rhlr_2/self.gim2d.Scale_2)
                yer=self.galfit24.cre1err*mipspixelscale/(self.gim2d.Rhlr_2/self.gim2d.Scale_2)
            else:
                y=self.galfit24.re1*mipspixelscale/(self.gim2d.Rhlr_2/self.gim2d.Scale_2)
                yer=self.galfit24.re1err*mipspixelscale/(self.gim2d.Rhlr_2/self.gim2d.Scale_2)

        if self.prefix.find('MKW11')> -1:
            clabel='Clusters'

        else:
            clabel='_nolegend_'

        clabel=self.prefix
        plot(x[flag],y[flag],'ko',color=colors,marker=shapes,label=clabel,markersize=8)
        #errorbar(x,y,yerr,fmt=None,color=colors,label='_nolegend_')

        #flag=self.ellipseflag & self.spiralflag & ~self.member & ~self.agn1
        flag1=self.galfitflag &  self.field
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        x1=self.stellarmass[flag1]
        y1=y[flag1]#self.galfit24.re1[flag]*mipspixelscale/self.n.SERSIC_TH50[flag]
        #yerr=self.SFR24err[flag]
        if self.prefix.find('A2063')> -1:
            clabel='Field'

        else:
            clabel='_nolegend_'
            #colors='0.3'
        #plot(x1,y1,'k.',markeredgecolor=colors,marker=shapes,markerfacecolor=colors,label=clabel,markersize=10,lw=3,alpha=0.2)
        plot(x1,y1,'k.',markeredgecolor=fieldColor,marker=shapes,markerfacecolor=fieldColor,label=clabel,markersize=10,lw=3,alpha=0.5)

	#ax=gca()
	#ax.set_yscale('log')
        #axis([-2.5,1.5,0,22])
        #xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        #ymin,ymax=ylim()
        #title(self.clustername)
        return x[flag].tolist(),y[flag].tolist(),x1.tolist(),y1.tolist()
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)


    def plotsizestellmassdens(self,colors,shapes):
        flag=self.sampleflag &  self.member
        x=self.stellarmass[flag]/(pi*self.n.SERSIC_TH50[flag])
        y=self.galfit24.re1[flag]*mipspixelscale/self.n.SERSIC_TH50[flag]
        if self.prefix.find('MKW11')> -1:
            clabel='Clusters'

        else:
            clabel='_nolegend_'

        clabel=self.prefix
        plot(x,y,'ko',color=colors,marker=shapes,label=clabel,markersize=8)

        flag=self.sampleflag &  ~self.member
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        x1=self.stellarmass[flag]/(pi*self.n.SERSIC_TH50[flag])
        y1=self.galfit24.re1[flag]*mipspixelscale/self.n.SERSIC_TH50[flag]
        if self.prefix.find('A2063')> -1:
            clabel='Field'

        else:
            clabel='_nolegend_'
            #colors='0.3'
        plot(x1,y1,'k.',markeredgecolor=fieldColor,marker=shapes,markerfacecolor=fieldColor,label=clabel,markersize=10,lw=3,alpha=0.5)

        return x.tolist(),y.tolist(),x1.tolist(),y1.tolist()

    def plotSFR24HIperArea(self,colors,shapes):
        #figure(1)
        #plotflag=self.apexflag & self.HIflag & ~self.agn2 & self.spiralflag
        #plotflag=self.apexflag  & ~self.agn2 & self.spiralflag

        #flag=self.ellipseflag & self.spiralflag & self.member & ~self.agnflag
        #flag=self.ellipseflag & self.member & ~self.agnflag
        flag=self.apexflag & self.dvflag & ~self.agnflag
        x=self.HImass[flag]/(self.sdssPetroR90r[flag]**2)
        y=self.SFR24[flag]
        yerr=self.SFR24err[flag]
        plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=8)
        errorbar(x,y,yerr,fmt=None,color=colors,label='_nolegend_')

        #flag=self.ellipseflag & self.spiralflag & ~self.member & ~self.agn1
        flag=self.apexflag  & ~self.member & ~self.agnflag
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        x1=self.HImass[flag]/(self.sdssPetroR90r[flag]**2)
        y1=self.SFR24[flag]
        yerr=self.SFR24err[flag]
        if self.prefix.find('MKW11')> -1:
            clabel=self.prefix+' Field'

        else:
            clabel='_nolegend_'
            #colors='0.3'
        #plot(x1,y1,'k.',markeredgecolor=colors,marker=shapes,markerfacecolor=colors,label=clabel,markersize=10,lw=3,alpha=0.2)
        plot(x1,y1,'k.',markeredgecolor=fieldColor,marker=shapes,markerfacecolor=fieldColor,label=clabel,markersize=10,lw=3,alpha=0.2)

        return x.tolist(),y.tolist(),x1.tolist(),y1.tolist()

    def checkmorph(self):#print out files that our class and burstein type disagree or no burstein type
        flag=(self.morph==3)
        self.summerSpirals=self.agcnumber[flag]
        bflag=((self.bsteintype>=120)&(self.bsteintype<183))|((self.bsteintype>=300)&(self.bsteintype<400)) 
        self.bflag=bflag
        funnyflag= ((~(flag) & bflag) | (flag & ~bflag)) & self.morphflag & (self.bsteintype > 0)
        self.funnySpirals=self.agcnumber[funnyflag]
        self.ourtype=self.morph[funnyflag]
        self.theirtype=self.bsteintype[funnyflag]
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.funnySpirals.dat'
        outfile=open(s,'w')
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.funnySpirals'
        outfile2=open(s,'w')
        for i in range(len(self.funnySpirals)):
            if  self.prefix in 'Coma Hercules A1367':
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss.fits \n'
            else:
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss-g.fits \n'
            outfile.write(name)
            name=str(self.funnySpirals[i])+' '+str(self.ourtype[i])+' '+str(self.theirtype[i])+' \n'
            outfile2.write(name)
        outfile.close()
        outfile2.close()


        nobflag= ((self.bsteintype == 0) & (self.On24ImageFlag))
        self.funnySpirals=self.agcnumber[nobflag]
        self.ourtype=self.morph[nobflag]
        self.theirtype=self.bsteintype[nobflag]
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noBsteinSpirals.dat'
        outfile=open(s,'w')
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noBsteinSpirals'
        outfile2=open(s,'w')
        for i in range(len(self.funnySpirals)):
            if  self.prefix in 'Coma Hercules A1367':
                
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss.fits \n'
            else:
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss-g.fits \n'
            outfile.write(name)
            name=str(self.funnySpirals[i])+' '+str(self.ourtype[i])+' '+str(self.theirtype[i])+' \n'
            outfile2.write(name)
        outfile.close()
        outfile2.close()

        missmorphflag= ((self.morphflag == 0) & (self.On24ImageFlag))
        self.funnySpirals=self.agcnumber[missmorphflag]
        self.ourtype=self.morph[missmorphflag]
        self.theirtype=self.bsteintype[missmorphflag]
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noMorph.dat'
        outfile=open(s,'w')
        s=homedir+'research/LocalClusters/MorphologyF2011/'+self.prefix+'.noMorph'
        outfile2=open(s,'w')
        for i in range(len(self.funnySpirals)):
            if  self.prefix in 'Coma Hercules A1367':
                
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss.fits \n'
            else:
                name=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'+self.prefix+'-'+str(self.funnySpirals[i])+'-cutout-sdss-g.fits \n'
            outfile.write(name)
            name=str(self.funnySpirals[i])+' '+str(self.ourtype[i])+' '+str(self.theirtype[i])+' \n'
            outfile2.write(name)
        outfile.close()
        outfile2.close()

    def plotsSFRLocaldens(self,colors,shapes): 
        plotflag=self.ellipseflag & self.spiralflag
        plotflag=self.sex24.MATCHFLAG24 & ~self.ellipticalflag & ~self.agnflag & self.dvflag
        sSFR=self.size24[plotflag]/self.n.SERSIC_TH50[plotflag]
        plot(self.localdens[plotflag],sSFR,'ko',marker=shapes,color=colors,label=self.prefix,markersize=10)
        x=self.localdens[plotflag].tolist()
        y=sSFR.tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        return x,y

    def plotsSFRHIDef(self,colors,shapes): 
        plotflag=self.On24ImageFlag & ~self.agnflag & self.HIflag & self.sex24.MATCHFLAG24 #& (self.snr24 > 3)
        #print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        sSFR=self.ce.SFR[plotflag]/self.stellarmass[plotflag]*1.e9
        x=self.HIDef[plotflag]
        plot(x,sSFR,'ko',marker=shapes,color=colors,label=self.prefix,markersize=10)
        #print x
        #print sSFR
        x=x.tolist()
        y=sSFR.tolist()
        return x,y

    def plotsSFRratioLocaldens(self,colors,shapes): 
        plotflag=self.ellipseflag & self.spiralflag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        
        ry=self.sSFR90-self.sSFR50
        scatter(self.localdens[plotflag],ry[plotflag],marker=shapes,color=colors)
        x=self.localdens[plotflag].tolist()
        y=ry[plotflag].tolist()
        #print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralflag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.localdens[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        #print 'min local dens = ',min(self.localdens[plotflag])
        #x=x+self.localdens[plotflag].tolist()
        #y=y+self.sSFR[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y

    def plotsSFRLocaldens2(self,colors,shapes): 
        plotflag=self.ellipseflag & self.spiralflag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        
        scatter(self.localdens[plotflag],self.sSFR[plotflag],marker=shapes,color=colors)
        x=self.localdens[plotflag].tolist()
        y=self.sSFR[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralflag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.localdens[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        print 'min local dens = ',min(self.localdens[plotflag])
        x=x+self.localdens[plotflag].tolist()
        y=y+self.sSFR[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y

    def plotRatiosDr(self,colors,shapes): 
        plotflag=self.plotratio90Flag
        plotflag=self.ellipseflag & self.spiraflag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        ry=self.sSFR90-self.sSFR50
        scatter(self.dr[plotflag],y[plotflag],marker=shapes,color=colors)
        x=self.dr[plotflag].tolist()
        y=ry[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralflag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.dr[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        #x=x+self.dr[plotflag].tolist()
        #y=y+self.ratio90[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y
    def plotRatiosDr200(self,colors,shapes): 
        #plotflag=self.plotratio90Flag
        plotflag=self.ellipseflag & self.spiralflag
        print self.clustername, 'numb of successfully fit spirals = ',sum(plotflag)
        ry=self.sSFR90-self.sSFR50
        scatter(self.drR200[plotflag],ry[plotflag],marker=shapes,color=colors,label=self.prefix)
        x=self.drR200[plotflag].tolist()
        y=ry[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        #plotflag=(self.spiralflag & self.On24ImageFlag & ~self.apexflag)
        #scatter(self.drR200[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        #x=x+self.drR200[plotflag].tolist()
        #y=y+self.ratio90[plotflag].tolist()

##        ax=gca()
##        ax.set_yscale('log')
##        ax.set_xscale('log')
##        axis([0.1,1000,0.1,8])
        #xlabel('$Local \ Density$')
        #ylabel('$R_{24}/R_{r}$')
        return x,y
    def plotRatiosStellarMass(self,colors,shapes): 
        plotflag=self.plotratio90Flag
        scatter(self.stellarmass[plotflag],self.ratio90[plotflag],marker=shapes,color=colors)
        x=self.stellarmass[plotflag].tolist()
        y=self.ratio90[plotflag].tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        plotflag=(self.spiralflag & self.On24ImageFlag & ~self.apexflag)
        scatter(self.stellarmass[plotflag],zeros(len(self.localdens[plotflag])), marker='o',color=colors,s=6,alpha=0.2)
        print 'min local dens = ',min(self.localdens[plotflag])
        x=x+self.stellarmass[plotflag].tolist()
        y=y+self.ratio90[plotflag].tolist()
        return x,y
    def plotsSFR(self):
        #figure()
        title(self.clustername)
        #sSFR90=(self.mipsF90-self.mipsF50)/(self.SDSSF90-self.SDSSF50)
        #sSFR50=sSFR50[self.ellipseflag]
        #sSFR90=sSFR90[self.ellipseflag]

        flag=self.ellipseflag
        plot(self.sSFR50[flag],self.sSFR90[flag],'k.',label='All')
        errorbar(self.sSFR50[flag],self.sSFR90[flag],xerr=self.sSFR50err[flag],yerr=self.sSFR90err[flag],fmt=None,label='no_label')
        m,b=polyfit(self.sSFR50[flag],self.sSFR90[flag],1)
        print m,b
        #scatter(sSFR50[flag],sSFR90[flag],s=1.*(self.localdens[flag]),c='c',alpha=0.7,label='$\Sigma$')
        flag=self.ellipseflag & self.spiralflag
        plot(self.sSFR50[flag],self.sSFR90[flag],'bo',markeredgecolor='b',markerfacecolor='None',label='Field Spirals',markersize=8)
        flag=self.ellipseflag & self.spiralflag& self.member
        plot(self.sSFR50[flag],self.sSFR90[flag],'b^',markersize=10,label='Cluster Spirals')
        flag=self.ellipseflag & self.HIflag
        if sum(flag) > 1:
            scatter(self.sSFR50[flag],self.sSFR90[flag],s=1.*(self.flux100[flag]),c='r',alpha=0.7,label='HI')
        xlabel('$F_{24}/F_r(r<R_{50})$',fontsize=16)
        ylabel('$F_{24}/F_r(r<R_{90})$',fontsize=16)
        legend(loc='upper left',numpoints=1,scatterpoints=1,markerscale=1)
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.001)
        plot(xl,xl,'k-')
        plot(xl,m*xl,'g')

    def sSFRnew(self):
        figure()
        title(self.clustername)
        #sSFR90=(self.mipsF90-self.mipsF50)/(self.SDSSF90-self.SDSSF50)
        #sSFR50=sSFR50[self.ellipseflag]
        #sSFR90=sSFR90[self.ellipseflag]

        flag=self.ellipseflag
        plot(self.sSFR50[flag],self.sSFR5090[flag],'k.',label='All')
        errorbar(self.sSFR50[flag],self.sSFR5090[flag],xerr=self.sSFR50err[flag],yerr=self.sSFR90err[flag],fmt=None,label='no_label')
        m,b=polyfit(self.sSFR50[flag],self.sSFR5090[flag],1)
        print m,b
        #scatter(sSFR50[flag],sSFR90[flag],s=1.*(self.localdens[flag]),c='c',alpha=0.7,label='$\Sigma$')
        flag=self.ellipseflag & self.spiralflag
        plot(self.sSFR50[flag],self.sSFR5090[flag],'bo',markeredgecolor='b',markerfacecolor='None',label='Field Spirals',markersize=8)
        flag=self.ellipseflag & self.spiralflag& self.member
        plot(self.sSFR50[flag],self.sSFR5090[flag],'b^',markersize=10,label='Cluster Spirals')
        flag=self.ellipseflag & self.HIflag
        if sum(flag) > 1:
            scatter(self.sSFR50[flag],self.sSFR5090[flag],s=1.*(self.flux100[flag]),c='r',alpha=0.7,label='HI')
        xlabel('$F_{24}/F_r(r<R_{50})$',fontsize=16)
        ylabel('$F_{24}/F_r(R_{50} <r<R_{90})$',fontsize=16)
        legend(loc='upper left',numpoints=1,scatterpoints=1,markerscale=1)
        axis([-.01,.02,-.01,.02])
        xmin,xmax=xlim()
        xl=arange(xmin,xmax,.001)
        plot(xl,xl,'k-')
        plot(xl,m*xl,'g')
        axis([-.01,.02,-.01,.02])
        ax=gca()
        ax.set_yscale('log')
        ax.set_xscale('log')

    def integratedHI(self):
        flag=self.member & self.HIflag
        self.sumHIflux=sum(self.flux100[flag]/100.)
        print 'sum of HI flux = ',self.sumHIflux
        #scaled to z=0.4
        #ratio=self.dL

    def optiroffset(self):#plot X-ray contours w/opt-ir distances overlaid
        self.optirdist=sqrt((self.sdssra-self.mipsra)**2+(self.sdssdec-self.mipsdec)**2)
        maxdist=6./3600 #6 arcsec
        flag=self.apexflag & (self.optirdist < maxdist) #& self.On24ImageFlag 
        #figure()
        rimage='/home/share/research/LocalClusters/ColorImages/'+self.prefix+'/'+self.prefix+'-r.fits'
        ximage='/home/share/research/LocalClusters/ROSATimages/'+self.prefix+'.RASS.3.75x3.75.fits'
        #mylevels=[2.6,3.6,4.,5,
        gc=aplpy.FITSFigure(rimage)
        level=xraycontourlevels[self.prefix]
        gc.show_contour(ximage,layer='marker_set_1',smooth=4,colors='black',levels=level)
        gc.show_markers(self.sdssra[flag],self.sdssdec[flag],facecolor='red',marker='o',edgecolor='None',s=self.optirdist[flag]*3600*10)
        return gc
    def dalevschary(self):
        figure(figsize=[10,10])
        flag=self.apexflag & self.sdssflag
        redshift=self.cz*ones(len(flag))  # convert recession velocity to redshift
        ceLir,ceSFR=chary.chary_elbaz_24um(redshift[flag],self.mipsflux[flag])
        subplot(2,1,1)
        plot(self.mipsflux[flag],self.Lir[flag],'bo',self.mipsflux[flag],ceLir,'ro')
        ylabel('IR Luminosity')
        title('Comparison of Lir from Dale & Helou and Chary & Elbaz')
        axvline(x=671,ls='--')
        legend(['Dale & Helou','Chary & Elbaz','lower limit of C&E templates'],numpoints=1,loc='upper left')
        gca().set_yscale('log')
        gca().set_xscale('log')

        subplot(2,1,2)
        a=self.Lir[flag]
        b=ceLir
        percentdiff=(b-a)/(.5*(a+b))*100
        plot(self.mipsflux[flag],percentdiff,'go')
        axvline(x=671,ls='--')
        axhline(y=0,ls='-',color='k')
        gca().set_xscale('log')
        xlabel('MIPS flux (micro-Jy)')
        ylabel('% Difference')
        # compare SFRs
        figure()
        subplot(2,1,1)
        plot(self.mipsflux[flag],self.SFR24[flag],'bo',self.mipsflux[flag],ceSFR,'ro')
        ylabel('IR SFR')
        title('Comparison of IR SFR from Dale & Helou and Chary & Elbaz')
        axvline(x=671,ls='--')
        legend(['Dale & Helou','Chary & Elbaz','lower limit of C&E templates'],numpoints=1,loc='upper left')
        gca().set_yscale('log')
        gca().set_xscale('log')
        xlabel('MIPS-24 Flux (micro-Jy)')

        subplot(2,1,2)
        a=self.SFR24[flag]
        b=ceSFR
        percentdiff=(b-a)/(.5*(a+b))*100
        plot(self.stellarmass[flag],a,'bo',self.stellarmass[flag],b,'ro')
        # plot slope of one line from Elbaz et al 2011
        xl=arange(9,11.7,.1)
        xl=10.**xl
        ye=(xl/4.e9)
        textdx=2.e7
        plot(xl,ye,color='c',ls='-')
        text(xl[0]-textdx,ye[0],'$\mathrm{Elbaz \ z=0}$',color='c',fontsize=16,horizontalalignment='right')

        gca().set_xscale('log')
        gca().set_yscale('log')
        xlabel('Stellar Mass')
        ylabel('IR SFR')
    def spiralcoords(self):
        # output a list of ra and dec for spiral on the 24um image
        # sending this list to shoko so she can send me Halpha images
        ra=self.ra[self.galfitflag]
        dec=self.dec[self.galfitflag]
        filename=homedir+'research/LocalClusters/spiralcoords/'+self.prefix+'-spiralcoords.dat'
        ascii.write([ra,dec],filename,names=['RA (J2000)','DEC (J2000)'])
#mkw11=baseClusterNSA('MKW11')
#a2063=baseClusterNSA('A2063')
#a2052=baseClusterNSA('A2052')
#coma=baseClusterNSA('Coma')
#mkw8=baseClusterNSA('MKW8')
#awm4=baseClusterNSA('AWM4')
#ngc=baseClusterNSA('NGC6107')
#mycl=[mkw11,mkw8,awm4,ngc,a2052,a2063,coma]
#for cl in mycl:
#    cl.plotmipswise()



#def sSFRlocaldens(self):
def plotL24histall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        cl.plotL24hist()
    ax=gca()
    text(-.75,-.35,'$log_{10}(L_{24}/L_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotL24histAll.eps')
   
def plotSFR24histall():
    figure(figsize=[10,8])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotSFR24hist(plotsingle=0)
    ax=gca()
    text(-.75,-.35,'$log_{10}(SFR_{24} \  (M_\odot/yr))$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotSFR24histAll.eps')


def plotLIRhistall():
    figure(figsize=[6.5,4])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        cl.plotLIRhist(plotsingle=0)
        multiplotaxes(i)
        
    multiplotlabelsv2('$log_{10}(L_{IR}/L_\odot)$','$N_{gal}$')
    savefig(figuredir+'LIRhistAll.eps')


def plotLIRhistall_onepanel():
    figure(figsize=[10,8])
    clf()
    #subplots_adjust(wspace=.02,hspace=.02)
    i=0
    for cl in mylocalclusters:
        i +=1
        #subplot(3,3,i)
        cl.plotLIRhist(plotsingle=0,htype='step',myhatch='',plotclname=0)
        #multiplotaxes(i)

    legend(loc='upper right')
    xlabel('$log_{10}(L_{IR}/L_\odot)$',fontsize=22)
    ylabel('$N_{gal}$',fontsize=22)
    savefig(figuredir+'LIRhistAll_onepanel.eps')


def plotHImasshistall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        cl.plotHImasshist()
    ax=gca()
    text(-.75,-.35,'$HI \ Mass$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'HImasshistAll.eps')

def plotHIDefhistall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        try:
            cl.plotHIDefhist()
        except:
            title(cl.prefix)
            continue
    ax=gca()
    text(-.75,-.35,'$HI \ Deficiency$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$N_{gal}$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'HIDefhistAll.eps')

def plotpositionsall():
    figure(figsize=(15,10))
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotpositions()
    ax=gca()
    text(-.75,-.35,'RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotPositionsAll.eps')


def plotpositionson24all():
    figure(figsize=[8,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    noylabel=[2,3,5,6,8,9]
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        print cl.prefix, cl.clusterra,cl.clusterdec
        cl.plotpositionson24(cl.clusterra,cl.clusterdec,plotsingle=0)
        axis([-1.5,1.5,-1.8,1.8])
        xticks(arange(-1,2))
        yticks(arange(-1,2))
        ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))

        #cl.plotrelativepositionson24()
    ax=gca()
    text(-.5,-.25,'$\Delta RA \ (deg)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.4,1.5,'$\Delta Dec \ (deg) $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'PlotPositionsOn24All.eps')
    savefig(figuredir+'PlotPositionsOn24All.png')

def plotrelativepositionson24all():
    figure(figsize=[9,9])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotrelativepositionson24()
    leg=legend(numpoints=1)#,fontsize=12)
    for t in leg.get_texts():
        t.set_fontsize('small')
    ax=gca()
    text(-.75,-.35,'$\Delta$RA (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    text(-2.8,1.9,'$\Delta$Dec (deg)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotRelativePositionsOn24All.eps')

def plotRe24vsReall(sbcutobs=20.5,colorflag=0):
    figure(figsize=[10,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    noylabel=[2,3,5,6,8,9]
    i = 0
    for cl in mylocalclusters:
    #for cl in clustersbylx:
    #for cl in clustersbydistance:
        i += 1
        subplot(3,3,i)
        print cl.prefix, cl.clusterra,cl.clusterdec
        cl.plotRe24vsRe(plotsingle=0,sbcutobs=sbcutobs,colorflag=colorflag)
        #try:
        #    cl.plotRe24vsRe(plotsingle=0)
        #except:
        #    print 'looks like ',cl.prefix,' is not ready yet'
        #axis([-1.5,1.5,-1.8,1.8])
        #xticks(arange(-1,2))
        #yticks(arange(-1,2))
        ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))

        #cl.plotrelativepositionson24()
    ax=gca()
    text(-.5,-.25,'$R_e \ NSA \ (arcsec)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.4,1.5,'$R_e \ 24\mu m \ (arcsec) $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'Re24vsReall.eps')
    savefig(figuredir+'Re24vsReall.png')

def plotRe24vsRealloneplot():
    figure(figsize=[10,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    noylabel=[2,3,5,6,8,9]
    i = 0
    xba=[]
    yba=[]
    xga=[]
    yga=[]
    xra=[]
    yra=[]
    for cl in mylocalclusters:
    #for cl in clustersbylx:
    #for cl in clustersbydistance:
        i += 1
        #subplot(3,3,i)
        print cl.prefix, cl.clusterra,cl.clusterdec
        try:
            xb,yb,xg,yg,xr,yr=cl.plotRe24vsReoneplot(plotsingle=0)
        except:
            print 'looks like ',cl.prefix,' is not ready yet'
        xba=xba+xb.tolist()
        yba=yba+yb.tolist()
        xga=xga+xg.tolist()
        yga=yga+yg.tolist()
        xra=xra+xr.tolist()
        yra=yra+yr.tolist()
    print xba,yba
    xbin,ybin,ybinerr=my.binitave(array(xba),array(yba),5)
    plot(xbin,ybin,'bo',markersize=14)
    errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='b',label='_nolegend_')
    xbin,ybin,ybinerr=my.binitave(xga,yga,5)
    plot(xbin,ybin,'go',markersize=14)
    errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='g',label='_nolegend_')
    xbin,ybin,ybinerr=my.binitave(xra,yra,5)
    plot(xbin,ybin,'ro',markersize=14)
    errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='r',label='_nolegend_')
    ax=gca()
    text(.5,-.1,'$R_e \ NSA \ (arcsec)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-.1,.5,'$R_e \ 24\mu m \ (arcsec) $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'Re24vsRealloneplot.eps')
    savefig(figuredir+'Re24vsRealloneplot.png')

def plotdiffRevsSNRall():
    figure(figsize=[13,10])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    noylabel=[2,3,5,6,8,9]
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        print cl.prefix, cl.clusterra,cl.clusterdec
        cl.plotdiffRevsSNR(plotsingle=0)
        ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))

    ax=gca()
    text(-.5,-.25,'$SNR 24$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.4,1.5,'$(R_e(24)-R_e)/R_e $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'diffRevsSNRall.eps')
    savefig(figuredir+'diffRevsSNRall.png')

def plotdiffRevsReall():
    figure(figsize=[13,10])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    noylabel=[2,3,5,6,8,9]
    i = 0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        print cl.prefix, cl.clusterra,cl.clusterdec
        cl.plotdiffRevsRe(plotsingle=0)
        ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))
    ax=gca()
    text(-.5,-.25,'$R_e (r-band)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.4,1.5,'$(R_e(24)-R_e)/R_e $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'diffRevsReall.eps')
    savefig(figuredir+'diffRevsReall.png')

def plotselectioncuts():
    xlab=['$ M_*/M_\odot $','$ M_*/M_\odot $','$R_e \ NSA \ (arcsec)$','$r$','$r$']
    ylab=['$R_e \ NSA \ (arcsec)$','$ L_{IR} $','$ L_{IR} $','$R_e \ NSA \ (arcsec)$','$ L_{IR} $']
    figname=['Restellarmass.eps','lirstellarmass.eps','lirRe.eps','Rermag.eps','lirrmag.eps']
    for k in range(len(figname)):
        figure(figsize=[10,8])
        clf()
        subplots_adjust(wspace=.02,hspace=.02)
        noylabel=[2,3,5,6,8,9]
        i = 0
        for cl in clustersbydistance:
            i += 1
            subplot(3,3,i)
            print cl.prefix, cl.clusterra,cl.clusterdec
            if k == 0:
                cl.plotRestellarmass(plotsingle=0)
            elif k == 1:
                cl.plotlirstellarmass(plotsingle=0)
            elif k == 2:
                cl.plotlirRe(plotsingle=0)

            elif k == 3:
                cl.plotRermag(plotsingle=0)
            elif k == 4:
                cl.plotlirrmag(plotsingle=0)
            ax=gca()
            if i < 7:
                ax.set_xticklabels(([]))
            if i in noylabel:
                ax.set_yticklabels(([]))

        ax=gca()
        text(-.5,-.25,xlab[k],fontsize=22,horizontalalignment='center',transform=ax.transAxes)
        text(-2.4,1.5,ylab[k],fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

        savefig(figuredir+figname[k])


def plotveldrall():
    figure(figsize=[9,6])
    clf()
    subplots_adjust(wspace=.35,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotveldr()
    ax=gca()
    text(-.75,-.35,'dr (deg)',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-3.1,1.9,'$V_r$ (km/s)',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotVeldrAll.eps')


def plotdvdron24all():
    figure(figsize=[12,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        print cl.prefix
        #cl.plotdvdr(plotsingle=0)
        cl.plotredshiftdr(plotsingle=0)
        multiplotaxes(i)

    multiplotlabels('$\Delta r \ (deg) $','$Redshift $')
    savefig(figuredir+'PlotdvdrAllOn24.eps')
    savefig(figuredir+'PlotdvdrAllOn24.png')

def plotredshiftdron24all():
    figure(figsize=[6.5,4])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    #for cl in clustersbydistance:
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        print cl.prefix
        #cl.plotredshiftdrcolorbar(plotsingle=0)
        cl.plotredshiftdr(plotsingle=0)
        multiplotaxes(i)

    multiplotlabelsv2('$\Delta r \ (deg) $','$z$ ')
    savefig(figuredir+'plotredshiftdrallon24.eps')
    savefig(figuredir+'plotredshiftdrallon24.png')
    savefig(figuredir+'plotredshiftdrallon24.pdf')

def plotdvdron24all():
    figure(figsize=[12,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        print cl.prefix
        cl.plotdvdr(plotsingle=0)
        #cl.plotredshiftdr(plotsingle=0)
        multiplotaxes(i)

    multiplotlabels('$\Delta r \ (deg) $','$Redshift $')
    savefig(figuredir+'PlotdvdrAllOn24.eps')

def checkmorphall():
    for cl in mylocalclusters:
        cl.checkmorph()

def getSpirals():
    figure()
    clf()
    i=0
    for cl in mylocalclusters:
        i += 1
        cl.getFilesForProfileFitting()

def plotallsSFRratioLocaldens():
    figure(figsize=[15,10])
    clf()
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRratioLocaldens(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    xlabel('$Local \  Density $',fontsize=18)
    ylabel('$sSFR_{90} - sSFR_{50} $',fontsize=18)
    f=figuredir+'sSFRLocaldens.eps'
    savefig(f)

def plotsSFRHIDefall():
    figure(figsize=[10,8])
    clf()
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRHIDef(colors[i-1],shapes[i-1])
        xall=xall+x#.tolist()
        yall=yall+y#.tolist()
    axvline(x=0,ls='--',color='k')
    #axhline(y=0.2,ls='--',color='b')
    xall=array(xall,'d')
    yall=array(yall,'d')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    #print xall
    #print yall
    #print xbin,ybin
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='_nolegend_')
    errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.15,'$ rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.1,'$p = %4.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)

    #ax.set_yscale('log')
    axis([-.5,2,-.01,.25])
    xlabel('$HI \ Deficiency $',fontsize=18)
    ylabel('$sSFR \ (M_\odot \ yr^{-1} / 10^9 M_\odot )$',fontsize=18)
    legend(numpoints=1)
    f=figuredir+'sSFRHIDef.eps'
    savefig(f)

def plotallRatiosLocaldens():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRLocaldens(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlabel('$Local \  Density $',fontsize=18)
    ylabel('$SFR/M_* (M_\odot/yr/M_\odot)$',fontsize=18)
    f=figuredir+'sSFR2Localdens.eps'
    savefig(f)

def plotallsSFRLocaldens():
    figure(figsize=[15,10])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotsSFRLocaldens(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    legend(numpoints=1,loc='upper right')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    #axis([.9,1000,-.05e-10,.85e-9])
    xlabel('$Local \  Density $',fontsize=20)
    ylabel('$SFR/M_* \ (M_\odot/yr/M_\odot)$',fontsize=20)
    f=figuredir+'sSFRLocaldens.eps'
    savefig(f)

def plotallRatiosDr():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotRatiosDr(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    ax=gca()
    ax.set_xscale('log')
    axis([.9,1700,-.5,8])
    xlabel('$Local \  Density $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosLocaldens.eps'
    savefig(f)
def plotallRatiosDr():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotRatiosDr(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,9)
    plot(xbin,ybin,'k:')
    print yall
    ax=gca()
    axis([-.1,2.,-.5,7])
    xlabel('$\Delta r \ (deg)$',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figurdir+'RatiosDr.eps'
    savefig(f)

def plotallRatiosDr200():
    figure(figsize=[15,10])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    colors=['r','b','c','g','m','y','k','0.5','r']
    shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotRatiosDr200(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    legend(loc='upper right',scatterpoints=1)
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    ax=gca()
    xlabel('$\Delta r / R_{200} $',fontsize=18)
    ylabel('$sSFR_{90} - sSFR_{50}$',fontsize=18)
    f=figuredir+'RatiosDr200.eps'
    savefig(f)


def plotallRatiosSize():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=awm4
        if i == 3:
            cl=mkw8
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = herc
        if i == 8:
            cl = a1367
        if i == 9:
            cl = coma
        #subplot(3,3,i)
        x,y=cl.plotRatiosStellarMass(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,7)
    #plot(xbin,ybin,'k--')
    ax=gca()
    ax.set_xscale('log')
    axis([1.e9,2.e12,-.5,8])
    xlabel('$Stellar \ Mass (M_\odot) $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosStellarMass.eps'
    savefig(f)

def plotallRatiosStellarMass():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=awm4
        if i == 3:
            cl=mkw8
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = herc
        if i == 8:
            cl = a1367
        if i == 9:
            cl = coma
        #subplot(3,3,i)
        x,y=cl.plotRatiosStellarMass(colors[i-1],shapes[i-1])
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,7)
    plot(xbin,ybin,'k--')
    ax=gca()
    ax.set_xscale('log')
    axis([1.e9,2.e12,-.5,8])
    xlabel('$Stellar \ Mass (M_\odot) $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosStellarMass.eps'
    savefig(f)

def plotallRatiosHIStellarMass():
    #figure(figsize=[15,10])
    figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,y=cl.plotratioHImass(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
    xbin,ybin=my.binit(xall,yall,7)
    plot(xbin,ybin,'k-')
    xbin,ybin,ybinerr=my.biniterr(xall,yall,7)
    plot(xbin,ybin,'k--')
    ax=gca()
    xlabel('$HI \ Mass / Stellar \ Mass  $',fontsize=18)
    ylabel('$R_{90}(24\mu m)/R_{90}(r-band)$',fontsize=18)
    f=figuredir+'RatiosHIStellarMass.eps'
    savefig(f)

def plotSFR24StellarMassall():
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['k','k','k','k','k','k','k','k','k']
    #shapes=['o','^','s','*','p','d','v','h','<']

    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        #if cl.prefix.find('Coma') > -1:
        #    continue
        x,y,x1,y1=cl.plotSFR24stellmass(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
        x1all=x1all+x1#field galaxies
        y1all=y1all+y1

    xall=array(xall,'d')
    yall=array(yall,'d')
    x1all=array(x1all,'d')
    x1all=array(x1all,'d')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',color='r',markersize=14,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k')
    #print xbin,ybin,ybinerr
    #xbin,ybin,ybinerr=my.biniterr(xall,yall,5)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    #print xbin,ybin,ybinerr
    plot(xbin,ybin,'k-',marker='o',color='b',mec='k',markersize=14,label='Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k',lw=3,label='_nolegend_')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    xmin=2.e8
    xmax=3.e12
    axis([xmin,xmax,.001,60])
    # plot SF Main Sequence from Elbaz et al 2011
    xl=arange(9,12,.1)
    xl=10.**xl
    yl=.25e-9*xl
    plot(xl,yl,'c-',lw=3,label='Elbaz+ 2011')
    yl=.025e-9*xl
    plot(xl,yl,'k--',lw=3,label='_nolabel_')
    yl=.0025e-9*xl
    plot(xl,yl,'k--',lw=3,label='_nolabel_')
    legend(loc='upper left',numpoints=1)#labels=clusternames)
    #axis([8.e8,2e12,.002,70])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    xlabel('$Stellar \ Mass \ (M_\odot)$',fontsize=22)
    ylabel('$SFR_{IR} \ (M_\odot/yr)$',fontsize=22)
    f=figuredir+'SFR24StellarMass.eps'
    savefig(f)

    f=figuredir+'SFR24StellarMass.png'
    savefig(f)

    figure(figsize=(10,8))

    sSFRcl=yall/xall*1.e9
    sSFRf=y1all/x1all*1.e9

    mybins=arange(-4,0,.15)
    yc,xc,t=hist(log10(sSFRcl),bins=mybins,histtype='step',color='r',label='Cluster',hatch='\\',normed=True)
    yf,xf,t=hist(log10(sSFRf),bins=mybins,histtype='step',color='b',label='Field',hatch='/',normed=True)
    xlabel('$log_{10}(sSFR_{IR} \ (Gyr^{-1} ))$',fontsize=22)#'(M_\odot/yr/(10^9 M_\odot))$',fontsize=22)
    ylabel('$ Normalized \ Counts $',fontsize=22)
    legend()
    f=figuredir+'sSFRhist.eps'
    savefig(f)
    f=figuredir+'sSFRhist.png'
    savefig(f)

def plotsizestellarmassall(convflag=0,gim2dflag=1):
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['k','k','k','k','k','k','k','k','k']
    #shapes=['o','^','s','*','p','d','v','h','<']

    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    xg=[]
    yg=[]
    xc=[]
    yc=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        #if cl.prefix.find('Coma') > -1:
        #    continue
        x,y,x1,y1=cl.plotsizestellmass(colors[i-1],shapes[i-1],convflag=convflag,gim2d=gim2dflag)
        #print x,y
        xall=xall+x
        yall=yall+y
        x1all=x1all+x1#field galaxies
        y1all=y1all+y1

        if i < 5:
            xg=xg+x
            yg=yg+y
        else:
            xc=xc+x
            yc=yc+y

    xall=array(xall,'d')
    yall=array(yall,'d')
    xg=array(xg,'d')
    yg=array(yg,'d')
    xc=array(xc,'d')
    yc=array(yc,'d')
    x1all=array(x1all,'d')
    x1all=array(x1all,'d')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',color='r',markersize=14,label='LCS')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k')

    xbin,ybin,ybinerr=my.binit(xc,yc,4)
    #plot(xbin,ybin,'k-',marker='o',color='r',markersize=14,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k')

    xbin,ybin,ybinerr=my.binit(xg,yg,4)
    #plot(xbin,ybin,'k-',marker='o',color='g',markersize=14,label='Groups')
    #errorbar(xbin,ybin,ybinerr,fmt=None,color='g',ecolor='g')


   #print xbin,ybin,ybinerr
    #xbin,ybin,ybinerr=my.biniterr(xall,yall,5)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    #print xbin,ybin,ybinerr
    plot(xbin,ybin,'k-',marker='o',color='b',mec='k',markersize=14,label='Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k',lw=3,label='_nolegend_')
    axhline(y=1,color='k',ls='--')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    xmin=2.e8
    xmax=3.e12
    #axis([xmin,xmax,.01,5])
    # plot SF Main Sequence from Elbaz et al 2011
    leg=legend(loc='upper left',numpoints=1)#labels=clusternames)
    for t in leg.get_texts():
        t.set_fontsize('small')

    axis([2.e9,2e12,.1,5])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    print "Clusters: Re(24)/Re(r) vs stellar mass"
    spearman(xall,yall)
    print "Field: Re(24)/Re(r) vs stellar mass"
    spearman(x1all,y1all)
    print 'Cluster vs Field, Re(24)/Re(r), KS'
    D,p=ks(yall,y1all)
    xlabel('$Stellar \ Mass \ (M_\odot)$',fontsize=22)
    ylabel('$R_e(24)/R_e(r)$',fontsize=22)
    f=figuredir+'sizestellarmass.eps'
    savefig(f)

    f=figuredir+'sizestellarmass.png'
    savefig(f)

    figure(figsize=(10,8))

    sizecl=yall#/xall*1.e9
    sizef=y1all#/x1all*1.e9

    mybins=arange(-.6,.6,.1)
    yc,xc,t=hist(log10(sizecl),bins=mybins,histtype='step',color='r',label='Cluster',hatch='\\',normed=True)
    yf,xf,t=hist(log10(sizef),bins=mybins,histtype='step',color='b',label='Field',hatch='/',normed=True)
    xlabel('$log_{10}(R_e(24)/R_e(r))$',fontsize=22)#'(M_\odot/yr/(10^9 M_\odot))$',fontsize=22)
    ylabel('$ Normalized \ Counts $',fontsize=22)
    legend()
    f=figuredir+'sizehist.eps'
    savefig(f)
    f=figuredir+'sizehist.png'
    savefig(f)

def plotsizestellarmassnocoma():
    figure(figsize=[10,8])
    #figure()
    clf()

    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    i=0
    clusters=[mkw11,ngc,mkw8,awm4,herc,a1367,a2063,a2052]
    for cl in clusters:
        i += 1
        #subplot(3,3,i)
        #if cl.prefix.find('Coma') > -1:
        #    continue
        x,y,x1,y1=cl.plotsizestellmass(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
        x1all=x1all+x1#field galaxies
        y1all=y1all+y1

    xall=array(xall,'d')
    yall=array(yall,'d')
    x1all=array(x1all,'d')
    x1all=array(x1all,'d')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',color='r',markersize=14,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k')


   #print xbin,ybin,ybinerr
    #xbin,ybin,ybinerr=my.biniterr(xall,yall,5)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    #print xbin,ybin,ybinerr
    plot(xbin,ybin,'k-',marker='o',color='b',mec='k',markersize=14,label='Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k',lw=3,label='_nolegend_')
    axhline(y=1,color='k',ls='--')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    xmin=2.e8
    xmax=3.e12
    #axis([xmin,xmax,.01,5])
    # plot SF Main Sequence from Elbaz et al 2011
    legend(loc='upper left',numpoints=1)#labels=clusternames)
    axis([2.e9,2e12,.1,5])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    print "Clusters (No Coma): Re(24)/Re(r) vs stellar mass"
    spearman(xall,yall)
    print "Field  (No Coma): Re(24)/Re(r) vs stellar mass"
    spearman(x1all,y1all)
    print 'Cluster vs Field  (No Coma), Re(24)/Re(r), KS'
    D,p=ks(yall,y1all)
    xlabel('$Stellar \ Mass \ (M_\odot)$',fontsize=22)
    ylabel('$R_e(24)/R_e(r)$',fontsize=22)
    f=figuredir+'sizestellarmassnocoma.eps'
    savefig(f)

    f=figuredir+'sizestellarmassnocoma.png'
    savefig(f)

    figure(figsize=(10,8))

    sizecl=yall#/xall*1.e9
    sizef=y1all#/x1all*1.e9

    mybins=arange(-.6,.6,.1)
    yc,xc,t=hist(log10(sizecl),bins=mybins,histtype='step',color='r',label='Cluster',hatch='\\',normed=True)
    yf,xf,t=hist(log10(sizef),bins=mybins,histtype='step',color='b',label='Field',hatch='/',normed=True)
    xlabel('$log_{10}(R_e(24)/R_e(r))$',fontsize=22)#'(M_\odot/yr/(10^9 M_\odot))$',fontsize=22)
    ylabel('$ Normalized \ Counts $',fontsize=22)
    legend()
    f=figuredir+'sizehistnocoma.eps'
    savefig(f)
    f=figuredir+'sizehistnocoma.png'
    savefig(f)


def plotcheckf24all():
    figure()
    clf()
    for cl in mylocalclusters:
        cl.plotcheckvsf24(plotsingle=0)

def plotsizecolorall(convflag=0):
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['k','k','k','k','k','k','k','k','k']
    #shapes=['o','^','s','*','p','d','v','h','<']

    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    xa=[]
    ya=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        #if cl.prefix.find('Coma') > -1:
        #    continue
        x,y,x1,y1=cl.plotsizecolor(colors[i-1],shapes[i-1],plotsingle=0,convflag=convflag)
        #print x,y
        if len(x)>0:
            xall=xall+x.tolist()
            yall=yall+y.tolist()
        if len(x1) > 0:
            x1all=x1all+x1.tolist()#field galaxies
            y1all=y1all+y1.tolist()
        xa=xa+x.tolist()+x1.tolist()
        ya=ya+y.tolist()+y1.tolist()

    xall=array(xall,'d')
    yall=array(yall,'d')
    x1all=array(x1all,'d')
    x1all=array(x1all,'d')
    xall=array(xall,'d')
    yall=array(yall,'d')

    xbin,ybin,ybinerr=my.binit(xa,ya,4)
    plot(xbin,ybin,'k-',marker='o',color='k',markersize=14,label='All')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k')
    #xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    #plot(xbin,ybin,'k-',marker='o',color='b',mec='k',markersize=14,label='Field')
    #errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k',lw=3,label='_nolegend_')

    #print "Clusters: Re(24)/Re(r) vs NUV-24"
    #spearman(xall,yall)
    #print "Field: Re(24)/Re(r) vs NUV-24"
    #spearman(x1all,y1all)
    #print 'Cluster vs Field, Re(24)/Re(r), KS'
    #D,p=ks(yall,y1all)
    #print 'Cluster vs Field, NUV-24, KS'
    #D,p=ks(xall,x1all)

    legend(numpoints=1,loc='lower left')
    print 'Size vs NUV-24 color'
    r,p=spearman(xa,ya)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)

    ax=gca()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #xmin=2.e8
    #xmax=3.e12
    axhline(y=1,color='k',ls='--')
    ax=gca()
    ax.set_yscale('log')
    #axis([1,9,.2,5])
    # plot SF Main Sequence from Elbaz et al 2011
    #legend(loc='upper left',numpoints=1)#labels=clusternames)
    #axis([8.e8,2e12,.002,70])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    xlabel('$NUV - m_{24}$',fontsize=22)
    ylabel('$R_e(24)/R_e(r)$',fontsize=22)
    f=figuredir+'sizecolor.eps'
    savefig(f)

    f=figuredir+'sizecolor.png'
    savefig(f)

def plotNUV24ssfrall():
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['k','k','k','k','k','k','k','k','k']
    #shapes=['o','^','s','*','p','d','v','h','<']

    xall=[]
    yall=[]
    x1all=[]

    y1all=[]
    xa=[]
    ya=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        #if cl.prefix.find('Coma') > -1:
        #    continue
        x,y,x1,y1=cl.plotNUV24ssfr(colors[i-1],shapes[i-1],plotsingle=0)
        #print x,y
        if len(x)>0:
            xall=xall+x.tolist()
            yall=yall+y.tolist()
        if len(x1) > 0:
            x1all=x1all+x1.tolist()#field galaxies
            y1all=y1all+y1.tolist()
        xa=xa+x.tolist()+x1.tolist()
        ya=ya+y.tolist()+y1.tolist()

    xall=array(xall,'d')
    yall=array(yall,'d')
    x1all=array(x1all,'d')
    x1all=array(x1all,'d')
    xall=array(xall,'d')
    yall=array(yall,'d')

    xbin,ybin,ybinerr=my.binit(xa,ya,4)
    plot(xbin,ybin,'k-',marker='o',color='k',markersize=14,label='All')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',color='r',mec='k',markersize=14,label='Cluster')
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    plot(xbin,ybin,'k-',marker='o',color='b',mec='k',markersize=14,label='Field')
    #errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k',lw=3,label='_nolegend_')

    #print "Clusters: Re(24)/Re(r) vs NUV-24"
    #spearman(xall,yall)
    #print "Field: Re(24)/Re(r) vs NUV-24"
    #spearman(x1all,y1all)
    #print 'Cluster vs Field, Re(24)/Re(r), KS'
    #D,p=ks(yall,y1all)
    #print 'Cluster vs Field, NUV-24, KS'
    #D,p=ks(xall,x1all)

    legend(numpoints=1,loc='upper left',scatterpoints=1)
    print 'Size vs NUV-24 color'
    r,p=spearman(xa,ya)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)

    ax=gca()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #xmin=2.e8
    #xmax=3.e12
    #axhline(y=1,color='k',ls='--')
    ax=gca()
    #ax.set_yscale('log')
    #axis([1,9,.2,5])
    # plot SF Main Sequence from Elbaz et al 2011
    #legend(loc='upper left',numpoints=1)#labels=clusternames)
    #axis([8.e8,2e12,.002,70])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    xlabel('$M_{24} - log_{10}(M_*)$',fontsize=22)
    ylabel('$NUV - M_{24}$',fontsize=22)
    f=figuredir+'NUV24ssfr.eps'
    savefig(f)

    f=figuredir+'NUV24ssfr.png'
    savefig(f)


def plotsizestellarmassdensall():
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['k','k','k','k','k','k','k','k','k']
    #shapes=['o','^','s','*','p','d','v','h','<']

    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        #if cl.prefix.find('Coma') > -1:
        #    continue
        x,y,x1,y1=cl.plotsizestellmassdens(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
        x1all=x1all+x1#field galaxies
        y1all=y1all+y1

    xa=xall+x1all
    ya=yall+y1all


    xall=array(xall,'d')
    yall=array(yall,'d')
    x1all=array(x1all,'d')
    x1all=array(x1all,'d')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',color='r',markersize=14,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k')
    #print xbin,ybin,ybinerr
    #xbin,ybin,ybinerr=my.biniterr(xall,yall,5)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    #print xbin,ybin,ybinerr
    plot(xbin,ybin,'k-',marker='o',color='b',mec='k',markersize=14,label='Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k',ecolor='k',lw=3,label='_nolegend_')
    ax=gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    xmin=2.e8
    xmax=3.e12
    axhline(y=1,color='k',ls='--')
    #axis([xmin,xmax,.01,5])
    # plot SF Main Sequence from Elbaz et al 2011
    legend(loc='upper left',numpoints=1)#labels=clusternames)
    #axis([8.e8,2e12,.002,70])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    xlabel('$Stellar \ Mass/(\pi R_e^2) (M_\odot)$',fontsize=22)
    ylabel('$R_e(24)/R_e(r)$',fontsize=22)
    f=figuredir+'sizestellarmassdens.eps'
    savefig(f)



    f=figuredir+'sizestellarmassdens.png'
    savefig(f)

    #figure(figsize=(10,8))

    print 'Cluster:  Size vs Stellar Mass density'
    r,p=spearman(xall,yall)

    print 'Field:  Size vs Stellar Mass density'
    r,p=spearman(x1all,y1all)


    print 'Cluster & Field:  Size vs Stellar Mass density'
    r,p=spearman(xa,ya)

    #sizecl=yall#/xall*1.e9
    #sizef=y1all#/x1all*1.e9

    #mybins=arange(-.6,.6,.1)
    #yc,xc,t=hist(log10(sizecl),bins=mybins,histtype='step',color='r',label='Cluster',hatch='\\',normed=True)
    #yf,xf,t=hist(log10(sizef),bins=mybins,histtype='step',color='b',label='Field',hatch='/',normed=True)
    #xlabel('$log_{10}(R_e(24)/R_e(r))$',fontsize=22)#'(M_\odot/yr/(10^9 M_\odot))$',fontsize=22)
    #ylabel('$ Normalized \ Counts $',fontsize=22)
    #legend()
    #f=figuredir+'sizestellardensityhist.eps'
    #savefig(f)
    #f=figuredir+'sizestellardensityhist.png'
    #savefig(f)

def plotSFR24HIperAreaall():
    figure(figsize=[10,8])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    colors=['k','k','k','k','k','k','k','k','k']
    shapes=['o','^','s','*','p','d','v','h','<']
    xall=[]
    yall=[]
    x1all=[]
    y1all=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y,x1,y1=cl.plotSFR24HIperArea(colors[i-1],shapes[i-1])
        #print x,y
        xall=xall+x
        yall=yall+y
        x1all=x1all+x1#field galaxies
        y1all=y1all+y1
    legend(loc='upper right',numpoints=1)#labels=clusternames)
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k')
    print xbin,ybin,ybinerr
    #xbin,ybin,ybinerr=my.biniterr(xall,yall,5)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    print xbin,ybin,ybinerr
    plot(xbin,ybin,'k--')
    errorbar(xbin,ybin,ybinerr,fmt=None,color='k', lw=3)
    ax=gca()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #axis([8.e8,2e12,.002,70])
    #    text(-.75,-.35,'Local Density',fontsize=18,horizontalalignment='center',transform=ax.transAxes)
#    subplot(3,3,4)
#    text(-2.8,1.9,'$R_{24}/R_r$',fontsize=18,verticalalignment='center',rotation=90,transform=ax.transAxes)
    xlabel('$HI \ Mass \ / Area (M_\odot/arcsec^2)$',fontsize=20)
    ylabel('$SFR \ (M_\odot/yr)$',fontsize=20)
    f=figuredir+'SFR24HIperArea.eps'
    savefig(f)


def plotsSFRall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.plotsSFR()
    ax=gca()
    text(-.75,-.35,'$F_{24}/F_r(r<R_{50})$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,'$F_{24}/F_r(r<R_{90})$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)
    savefig(figuredir+'PlotsSFRAll.eps')


def plotsfspiralvsmassall():
    figure(figsize=[10,5])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    i=0
    n1field=0
    d1field=0
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        #self.galfitflag=self.On24ImageFlag & (self.snr24>3) & ~self.agnflag & ~self.galfit24.numerical_error_flag24  & self.spiralflag & (self.n.SERSIC_TH50 > 5.) & & (self.stellarmass > 1.e10)
        a=sum(cl.On24ImageFlag & cl.spiralflag   & cl.member & (cl.stellarmass > minstellarmass)& (cl.ce.LIR > 5.1e8)  & ~cl.agnflag)
        b=sum(cl.On24ImageFlag & cl.spiralflag & cl.member &(cl.stellarmass > minstellarmass)  & ~cl.agnflag)
        n1field=n1field+sum(cl.On24ImageFlag & cl.spiralflag   & cl.field & (cl.stellarmass > minstellarmass)& (cl.ce.LIR > 5.1e8)  & ~cl.agnflag) 
        d1field=d1field+sum(cl.On24ImageFlag & cl.spiralflag & cl.field &(cl.stellarmass > minstellarmass)  & ~cl.agnflag)
        r,errdown,errup=my.ratioerror(a,b)
        subplot(1,2,1)
        plot(cl.biweightscale,r,color=colors[i],marker=shapes[i],markersize=10,label=cl.prefix)
        x1.append(cl.biweightscale)
        y1.append(r)
        erry=zeros((2,1))
        erry[1]=errup
        erry[0]=errdown
        errorbar(cl.biweightscale,r,yerr=erry,fmt=None,ecolor=colors[i],label='_nolegend_')
        subplot(1,2,2)
        plot(cl.cLx,r,color=colors[i],marker=shapes[i],markersize=10,label=cl.prefix)
        errorbar(cl.cLx,r,yerr=erry,fmt=None,ecolor=colors[i],label='_nolegend_')

        x2.append(cl.cLx)
        y2.append(r)


        i += 1

    subplot(1,2,1)
    r,errdown,errup=my.ratioerror(n1field,d1field)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')
    ylabel('$ f(SF \ Spirals) $',fontsize=18)
    subplot(1,2,2)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')
    legend(loc='lower left',prop={'size':8},numpoints=1)
    ax=gca()
    ax.set_xscale('log')


    for i in range(1,3):
        subplot(1,2,i)
        ylim(0,1.05)
    xlabel1x2plot()    
    print 'SF Frac vs sigma'
    r,p=spearman(x1,y1)
    print 'SF Frac vs Lx'
    r,p=spearman(x2,y2)
    savefig(figuredir+'sfspiralvsmass.eps')

def plotfracspiralvsmassall():
    figure(figsize=[10,5])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    i=0
    n1field=0
    d1field=0
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        a=sum(cl.On24ImageFlag & cl.spiralflag  & cl.member)
        b=sum(cl.On24ImageFlag  & cl.member)
        n1field=n1field+sum(cl.On24ImageFlag & cl.spiralflag  & cl.field)
        d1field=d1field+sum(cl.On24ImageFlag  & cl.field)
        r,errdown,errup=my.ratioerror(a,b)
        subplot(1,2,1)
        plot(cl.biweightscale,r,color=colors[i],marker=shapes[i],markersize=10)
        x1.append(cl.biweightscale)
        y1.append(r)
        erry=zeros((2,1))
        erry[1]=errup
        erry[0]=errdown
        errorbar(cl.biweightscale,r,yerr=erry,fmt=None,ecolor=colors[i])
        subplot(1,2,2)
        plot(cl.cLx,r,color=colors[i],marker=shapes[i],markersize=10)
        errorbar(cl.cLx,r,yerr=erry,fmt=None,ecolor=colors[i])

        x2.append(cl.cLx)
        y2.append(r)


        i += 1

    subplot(1,2,1)
    r,errdown,errup=my.ratioerror(n1field,d1field)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')
    ylabel('$ f(Spirals) $',fontsize=18)
    subplot(1,2,2)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')

    ax=gca()
    ax.set_xscale('log')


    for i in range(1,3):
        subplot(1,2,i)
        ylim(0,1)
    xlabel1x2plot()
    print 'Frac Spirals vs sigma'
    r,p=spearman(x1,y1)
    print 'Frac Spirals vs Lx'
    r,p=spearman(x2,y2)
    savefig(figuredir+'fracspiralvsmass.eps')


def plotsigmaLx2panel():
    figure(figsize=[12,5])

    clf()
    subplots_adjust(left=.1,bottom=.2,right=.95,top=.95,wspace=.3)
    i=0
    for cl in mylocalclusters:
        subplot(1,2,1)
        plot(cl.cLx,cl.biweightscale,'ko',color=colors[i],marker=shapes[i],markersize=10,label=cl.prefix)
        subplot(1,2,2)
        plot(cl.cLx,log10(sum(cl.stellarmass[cl.member])),'ko',color=colors[i],marker=shapes[i],markersize=10,label=cl.prefix)
        i += 1
    subplot(1,2,1)
    gca().set_xscale('log')
    xlabel('$L_X \ (10^{43} erg/s)$',fontsize=22)
    ylabel('$\sigma \ (km/s) $',fontsize=22)

    leg=legend(numpoints=1,loc='upper left',scatterpoints=1,markerscale=.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
    for t in leg.get_texts():
        t.set_fontsize('small')

    subplot(1,2,2)
    gca().set_xscale('log')
    xlabel('$L_X \ (10^{43} erg/s) $',fontsize=22)
    ylabel('$log_{10}(\Sigma M_*/M_\odot) $',fontsize=22)
    savefig(figuredir+'sigmalxall.eps')
def plotsigmaLx():
    figure(figsize=[10,8])

    clf()
    subplots_adjust(left=.1,bottom=.2,right=.95,top=.95,wspace=.3)
    i=0
    for cl in mylocalclusters:
        plot(cl.cLx,cl.biweightscale,'ko',color=colors[i],marker=shapes[i],markersize=10,label=cl.prefix)

        i += 1
    gca().set_xscale('log')
    xlabel('$L_X \ (10^{43} erg/s)$',fontsize=22)
    ylabel('$\sigma \ (km/s) $',fontsize=22)

    leg=legend(numpoints=1,loc='upper left',scatterpoints=1,markerscale=.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
    for t in leg.get_texts():
        t.set_fontsize('small')

    savefig(figuredir+'sigmalxall.eps')

def plottrunspiralvsmassall(convflag=0):
    figure(figsize=[10,5])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    i=0
    n1field=0
    d1field=0
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        if convflag:
            a=sum(cl.On24ImageFlag & cl.galfitflag & (cl.galfit24.cre1*mipspixelscale < truncation_fraction*cl.n.SERSIC_TH50) & cl.member)
        else:
            a=sum(cl.On24ImageFlag & cl.galfitflag & (cl.galfit24.re1*mipspixelscale < truncation_fraction*cl.n.SERSIC_TH50) & cl.member)
        #a=sum(cl.On24ImageFlag & cl.galfitflag & (cl.galfit24.re1*mipspixelscale < .8*cl.n.SERSIC_TH50) & cl.membflag)
        b=sum(cl.On24ImageFlag & cl.galfitflag & cl.member)
        if convflag:
            n1field=n1field+sum(cl.On24ImageFlag & cl.galfitflag & (cl.galfit24.cre1*mipspixelscale < truncation_fraction*cl.n.SERSIC_TH50) & cl.field)
        else:
            n1field=n1field+sum(cl.On24ImageFlag & cl.galfitflag & (cl.galfit24.re1*mipspixelscale < truncation_fraction*cl.n.SERSIC_TH50) & cl.field)
        d1field=d1field+sum(cl.On24ImageFlag & cl.galfitflag & cl.field)
        r,errdown,errup=my.ratioerror(a,b)
        print cl.prefix, cl.cLx, r, errdown, errup, a, b
        subplot(1,2,1)
        plot(cl.biweightscale,r,color=colors[i],marker=shapes[i],markersize=10,label=cl.prefix)
        x1.append(cl.biweightscale)
        y1.append(r)
        erry=zeros((2,1))
        erry[1]=errup
        erry[0]=errdown
        errorbar(cl.biweightscale,r,yerr=erry,fmt=None,ecolor=colors[i],label='_nolegend_')
        subplot(1,2,2)
        plot(cl.cLx,r,color=colors[i],marker=shapes[i],markersize=10)
        newx=clusterXray[cl.prefix]
        plot(newx[0],r,color=colors[i],marker=shapes[i],mec=colors[i],markersize=10,mfc='None')
        errorbar(cl.cLx,r,yerr=erry,fmt=None,ecolor=colors[i])

        x2.append(cl.cLx)
        y2.append(r)


        i += 1

    subplot(1,2,1)
    r,errdown,errup=my.ratioerror(n1field,d1field)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')
    ylabel('$ f(Truncated \ Spirals) $',fontsize=18)
    legend(numpoints=1,loc='lower left',prop={'size':8})
    subplot(1,2,2)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')

    ax=gca()
    ax.set_xscale('log')


    for i in range(1,3):
        subplot(1,2,i)
        ylim(0,1.05)
    xlabel1x2plot()    
    print 'Trun Frac vs sigma'
    r,p=spearman(x1,y1)
    print 'Trun Frac vs Lx'
    r,p=spearman(x2,y2)
    savefig(figuredir+'trunspiralvsmass.eps')

def xlabel1x2plot():
    subplots_adjust(left=.1,bottom=.15,right=.95,top=.95,wspace=.2)
    subplot(1,2,1)
    xlabel('$ \sigma \ (km/s) $',fontsize=18)
    subplot(1,2,2)
    xlabel('$ L_X \ (10^{43} erg/s) $',fontsize=18)

def plotspiralvsmassall():
    figure(figsize=[10,8])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    x3=[]
    y3=[]
    x4=[]
    y4=[]
    i=0
    n1field=0
    n2field=0
    d1field=0
    d2field=0
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        a=sum(cl.On24ImageFlag & cl.spiralflag & (cl.snr24>3) & cl.member)
        b=sum(cl.On24ImageFlag & cl.spiralflag & cl.member)
        n1field=n1field+sum(cl.On24ImageFlag & cl.spiralflag & (cl.snr24>3) & cl.field)
        d1field=d1field+sum(cl.On24ImageFlag & cl.spiralflag & cl.field)
        r,errdown,errup=my.ratioerror(a,b)
        subplot(2,2,1)
        plot(cl.biweightscale,r,color=colors[i],marker=shapes[i],markersize=10)
        x1.append(cl.biweightscale)
        y1.append(r)
        erry=zeros((2,1))
        erry[1]=errup
        erry[0]=errdown
        errorbar(cl.biweightscale,r,yerr=erry,fmt=None,ecolor=colors[i])
        subplot(2,2,2)
        plot(cl.cLx,r,color=colors[i],marker=shapes[i],markersize=10)
        errorbar(cl.cLx,r,yerr=erry,fmt=None,ecolor=colors[i])

        x2.append(cl.cLx)
        y2.append(r)

        a=sum(cl.On24ImageFlag & cl.galfitflag & (cl.galfit24.re1*mipspixelscale < truncation_fraction*cl.n.SERSIC_TH50) & cl.member)
        b=sum(cl.On24ImageFlag & cl.galfitflag &  cl.member)

        n2field=n2field+sum(cl.On24ImageFlag & cl.galfitflag & (cl.galfit24.re1*mipspixelscale < truncation_fraction*cl.n.SERSIC_TH50) & cl.field)
        d2field=d2field+sum(cl.On24ImageFlag & cl.galfitflag &  cl.field)
        r,errdown,errup=my.ratioerror(a,b)
        print cl.prefix,cl.cLx, r, errdown, errup



        erry=zeros((2,1))
        erry[1]=errup
        erry[0]=errdown


        subplot(2,2,3)
        plot(cl.biweightscale,r,color=colors[i],marker=shapes[i],markersize=10)
        errorbar(cl.biweightscale,r,yerr=erry,fmt=None,ecolor=colors[i])
        x3.append(cl.biweightscale)
        y3.append(r)

        subplot(2,2,4)
        plot(cl.cLx,r,color=colors[i],marker=shapes[i],markersize=10)
        errorbar(cl.cLx,r,yerr=erry,fmt=None,ecolor=colors[i])
        x4.append(cl.cLx)
        y4.append(r)

        i += 1

    subplot(2,2,1)
    r,errdown,errup=my.ratioerror(n1field,d1field)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')
    ylabel('$ f(SF \ Spirals) $',fontsize=18)
    subplot(2,2,2)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')

    ax=gca()
    ax.set_xscale('log')

    subplot(2,2,3)
    r,errdown,errup=my.ratioerror(n2field,d2field)

    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')
    ylabel('$ f(Truncated \ Spirals) $',fontsize=18)
    xlabel('$ \sigma \ (km/s) $',fontsize=18)
    subplot(2,2,4)
    axhline(y=r,color='k',ls='-')
    axhline(r-errdown,color='k',ls=':')
    axhline(r+errup,color='k',ls=':')
    ax=gca()
    ax.set_xscale('log')
    xlabel('$ L_X \ (10^{43} erg/s) $',fontsize=18)


    for i in range(1,5):
        subplot(2,2,i)
        ylim(0,1)
    
    print 'SF Frac vs sigma'
    r,p=spearman(x1,y1)
    print 'SF Frac vs Lx'
    r,p=spearman(x2,y2)
    print 'Trun Frac vs sigma'
    r,p=spearman(x3,y3)
    print 'Trun Frac vs Lx'
    r,p=spearman(x4,y4)

    print 'Trun Frac vs Lx (w/out MKW11)'
    r,p=spearman(x4[1:],y4[1:])

    #text(.9,.15,'$ rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    #text(.9,.1,'$p = %4.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)

    #xbin,ybin,ybinerr=my.binitbins(-.5,2,5,xall,yall)
    #xbin,ybin,ybinerr=my.binit(xall,yall,4)
    #plot(xbin,ybin,'k-',marker='o',markersize=14,label='median')
    #errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #legend(loc='upper right',numpoints=1)
    #axis([-1,1.5,0,2.5])
    savefig(figuredir+'spiralpropvsmass.eps')



def plotsizeHIdefall(convflag=0):
    figure(figsize=[10,8])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    xall=[]
    yall=[]
    massall=[]
    i=0
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        x,y,m=cl.plotsizeHIdef(colors[i],shapes[i],plotsingle=0,convflag=convflag)
        xall=xall+x.tolist()
        yall=yall+y.tolist()
        massall=massall+m.tolist()
        i += 1
    ax=gca()
    xlabel('$ HI \ Deficiency $',fontsize=22)
    ylabel('$R_e(24)/R_e(r)$',fontsize=22)
    xall=array(xall,'f')
    yall=array(yall,'f')
    r,p=spearman(xall,yall)
    
    ax=gca()
    text(.9,.15,'$ rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.1,'$p = %4.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    print 'spearman for log(M*) < 10.41'
    t=spearman(xall[s<10.41],yall[s<10.41])
    #xbin,ybin,ybinerr=my.binitbins(-.5,2,5,xall,yall)
    xbin,ybin,ybinerr=my.binit(xall,yall,4)
    plot(xbin,ybin,'k-',marker='o',markersize=14,label='median')
    errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    leg=legend(loc='upper right',numpoints=1)
    for t in leg.get_texts():
        t.set_fontsize('small')

    axvline(x=0,color='k',ls='--')
    axhline(y=1,color='k',ls='--')
    gca().set_yscale('log')
    axis([-.7,1.5,.1,3])
    savefig(figuredir+'SizeHIdefAll.eps')

def plotsizeclusterphiall(convflag=0):
    figure(figsize=[10,8])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        x,y=cl.plotsizeclusterphi(colors[i],shapes[i],plotsingle=0,convflag=convflag)
        xall=xall+x.tolist()
        yall=yall+y.tolist()
        i += 1
    ax=gca()
    xlabel('$\Phi $',fontsize=22)
    ylabel('$R_e(24)/R_e(r)$',fontsize=22)
    xall=array(xall,'f')
    yall=array(yall,'f')
    r,p=spearman(xall,yall)

    ax=gca()
    text(.9,.15,'$ rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.1,'$p = %4.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)

    #xbin,ybin,ybinerr=my.binitbins(-.5,2,5,xall,yall)
    xbin,ybin,ybinerr=my.binit(xall,yall,4)
    plot(xbin,ybin,'k-',marker='o',markersize=14,label='median')
    errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    legend(loc='upper right',numpoints=1)
    #axis([-1,1.5,0,2.5])
    savefig(figuredir+'sizeclusterphiall.eps')
    savefig(figuredir+'sizeclusterphiall.png')
    savefig(figuredir+'sizeclusterphiall.pdf')

def plotsizesSFRall():
    figure(figsize=[10,8])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    xall=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        x,y=cl.plotsizesSFR(colors[i],shapes[i],plotsingle=0)
        xall=xall+x.tolist()
        yall=yall+y.tolist()
        i += 1
    ax=gca()
    xlabel('$sSFR \ (M_\odot \ yr^{-1} / 10^9 M_\odot )$',fontsize=22)
    ylabel('$R_{24}/R_r$',fontsize=22)
    xall=array(xall,'f')
    yall=array(yall,'f')
    #print xall
    #print yall
    xflag = (isnan(xall))
    xflag = ~ xflag
    yflag = (isnan(yall))
    yflag = ~yflag
    flag = xflag | yflag
    r,p=spearman(xall[flag],yall[flag])

    ax=gca()
    text(.9,.15,'$ rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.1,'$p = %4.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    xbin,ybin,ybinerr=my.binitave(xall,yall,7)
    xbin,ybin,ybinerr=my.binitave(xall,yall,7)
    plot(xbin,ybin,'k-',marker='o',markersize=14,label='_nolegend_')
    errorbar(xbin,ybin,yerr=ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    legend(loc='upper left',numpoints=1)
    savefig(figuredir+'SizesSFRfAll.eps')

def plotsizeHaEW():
    figure(figsize=[10,8])
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    xall=[]
    yall=[]
    for cl in mylocalclusters:
        #si += 1
        #ssubplot(3,3,i)
        x,y=cl.plotratioRevsHaEW(plotsingle=0)
        xall=xall+x.tolist()
        yall=yall+y.tolist()
    ax=gca()
    xlabel('$ Ha \ EW $',fontsize=22)
    ylabel('$R_{24}/R_r$',fontsize=22)
    xall=array(xall,'f')
    yall=array(yall,'f')
    spearman(xall,yall)
    print 'AFTER REMOVING HIGHEST POINT'
    ymax=max(yall)
    x2=xall[yall < ymax]
    y2=yall[yall < ymax]
    spearman(x2,y2)
    xbin,ybin,ybinerr=my.binitbins(0,150,5,xall,yall)
    #plot(xbin,ybin,'k-',marker='o',markersize=12,label='Clusters')

    savefig(figuredir+'SizeHaEWAll.eps')


def plotsizemassall():
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    figure(figsize=[10,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    xallg=[]
    yallg=[]
    xallc=[]
    yallc=[]
    x1all=[]
    y1all=[]
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y,x1,y1=cl.plotsizemass(shapes[i-1],colors[i-1],plotsingle=0)
        #multiplotaxes(i)

        #multiplotlabels('$log_{10}(M_*/M_\odot) $','$ R_{24}/R_r$ ')
        if i < 5:
            xallg=xallg+x.tolist()
            yallg=yallg+y.tolist()
        else:
            xallc=xallc+x.tolist()
            yallc=yallc+y.tolist()

        #if cl.prefix.find('Coma') > -1:
        #    print 'not including coma in size-mass plot'
        #else:
        #    xall=xall+x.tolist()
        #    yall=yall+y.tolist()

        xall=xall+x.tolist()
        yall=yall+y.tolist()

        x1all=x1all+x1.tolist()#field galaxies
        y1all=y1all+y1.tolist()
    xall=array(xall,'f')
    yall=array(yall,'f')
    xallg=array(xallg,'f')
    yallg=array(yallg,'f')
    xallc=array(xallc,'f')
    yallc=array(yallc,'f')
    x1all=array(x1all,'f')
    y1all=array(y1all,'f')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xall,yall)
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='Clusters')
    #errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xallg,yallg)
    #plot(xbin,ybin,'c-',marker='o',markersize=12,label='Groups')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,x1all,y1all)
    xbin,ybin,ybinerr=my.binit(x1all,y1all,5)
    plot(xbin,ybin,'k-',color='0.5',marker='s',mfc='0.5',markersize=13,label='Field')
    
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    xlabel('$log_{10} (M_*/M_\odot) $',fontsize=22)
    ylabel('$ R_{24}/R_r $',fontsize=22)
    legend(loc='upper right',numpoints=1)
    print 'cluster: size vs stellar mass'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.2,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.2,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)


    savefig(figuredir+'SizeMassAll.eps')

    print 'field: size vs stellar mass'
    spearman(x1all,y1all)

    print 'comparing field and cluster ratios'
    ks(yall,y1all)

    print 'comparing field and cluster masses'
    ks(xall,x1all)

    print 'comparing group and cluster ratios'
    ks(yallg,yallc)

    print 'comparing group and cluster masses'
    ks(xallg,xallc)


    print 'comparing group and field ratios'
    ks(yallg,y1all)

    print 'comparing group and field masses'
    ks(xallg,x1all)

def plotsizeradiusall(plotsingle=1,convflag=0):

    if plotsingle:
        figure(figsize=[10,8])
        clf()
        subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    xc=[]
    yc=[]
    xg=[]
    yg=[]

    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotsizeradius(shapes[i-1],colors[i-1],plotsingle=0,convflag=convflag)
        #multiplotaxes(i)

        xall=xall+x.tolist()
        yall=yall+y.tolist()
        if cl.prefix in clusterlist:
            xc=xc+x.tolist()
            yc=yc+y.tolist()
        else:
            xg=xg+x.tolist()
            yg=yg+y.tolist()

    xall=array(xall,'f')
    yall=array(yall,'f')
    xg=array(xg,'f')
    yg=array(yg,'f')
    xc=array(xc,'f')
    yc=array(yc,'f')
    xbin,ybin,ybinerr=my.binit(xall,yall,4)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='Median')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binit(xg,yg,4)
    plot(xbin,ybin,'b-',marker='o',markersize=12,label='_nolegend_')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b',label='_nolegend_')

    xbin,ybin,ybinerr=my.binit(xc,yc,4)
    plot(xbin,ybin,'r-',marker='o',markersize=12,label='_nolegend_')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='r',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,x1all,y1all)
    ax=gca()
    ax.set_yscale('log')
    axis([0,4.1,.2,5])
    axhline(y=1,ls='--',color='k')
    xlabel('$\Delta r/R_{200} $',fontsize=18)
    print 'cluster size vs projected radius'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    if plotsingle:
        ylabel('$ R_{e}(24)/R_e(r) $',fontsize=18)
        leg=legend(loc='lower right',numpoints=1)
        for t in leg.get_texts():
            t.set_fontsize('small')
        savefig(figuredir+'sizeradiusall.eps')
        savefig(figuredir+'sizeradiusall.png')

def plotsizedist3dall(convflag=1):
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    figure(figsize=[8,4])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotsizedist3d(shapes[i-1],colors[i-1],plotsingle=0,convflag=convflag)
        #multiplotaxes(i)

        xall=xall+x.tolist()
        yall=yall+y.tolist()

    xall=array(xall,'f')
    yall=array(yall,'f')
    xbin,ybin,ybinerr=my.binit(xall,yall,6)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='Median')
    #errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xallg,yallg)
    #plot(xbin,ybin,'c-',marker='o',markersize=12,label='Groups')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,x1all,y1all)
    ax=gca()
    ax.set_yscale('log')
    axhline(y=1,ls='--',color='k')
    xlabel('$(\Delta v/\sigma)^2 + (\Delta r/R_{200})^2 $',fontsize=18)
    ylabel('$ R_{e}(24)/R_e(r) $',fontsize=18)
    legend(loc='lower right',numpoints=1)

    print 'size vs dist3d'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    savefig(figuredir+'sizedist3dall.eps')
    savefig(figuredir+'sizedist3dall.png')


def plotsizelocaldensityall(convflag=0,sbflag=0):
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    figure(figsize=[10,5])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotsizelocaldensity(shapes[i-1],colors[i-1],plotsingle=0,convflag=convflag,sbflag=sbflag)
        #multiplotaxes(i)

        xall=xall+x.tolist()
        yall=yall+y.tolist()

    xall=array(xall,'f')
    yall=array(yall,'f')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='Median')
    #errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xallg,yallg)
    #plot(xbin,ybin,'c-',marker='o',markersize=12,label='Groups')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,x1all,y1all)
    axhline(.7,color='b',ls='--')
    axhline(.5,color='r',ls=':')
    xlabel('$log_{10}( \Sigma _{NN}) $',fontsize=22)
    ylabel('$ R_e(24)/R_e(r) $',fontsize=22)
    leg=legend(loc='upper left',numpoints=1)
    for t in leg.get_texts():
        t.set_fontsize('small')

    ax=gca()
    ax.set_yscale('log')

    axis([-.5,3,.2,5])
    print 'size vs local density'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    savefig(figuredir+'sizelocaldensityall.eps')
    savefig(figuredir+'sizelocaldensityall.png')

def plotsizelocaldensity(convflag=0,gim2dflag=0):
    figure(figsize=(15,7))
    subplots_adjust(left=.1,bottom=.15,right=.95,top=.95,wspace=.1)
    subplot(1,2,1)
    plotsizelocaldensity5all(plotsingle=0,convflag=convflag,gim2d=gim2dflag)
    ax=gca()
    text(.05,.1,'$(a)$',horizontalalignment='left',transform=ax.transAxes,fontsize=16)
    axhline(.7,color='b',ls='--')
    axhline(.5,color='r',ls=':')

    subplot(1,2,2)
    plotsizelocaldensity10all(plotsingle=0,convflag=convflag,gim2d=gim2dflag)
    ax=gca()
    text(.05,.1,'$(b)$',horizontalalignment='left',transform=ax.transAxes,fontsize=16)
    axhline(.7,color='b',ls='--')
    axhline(.5,color='r',ls=':')

    gca().set_yticklabels(([]))
    savefig(figuredir+'sizelocaldensity.eps')

def plotsizedensitycombo2(convflag=0):
    figure(figsize=(15,7))
    subplots_adjust(left=.1,bottom=.15,right=.95,top=.95,wspace=.1)
    subplot(1,2,1)
    plotsizelocalmassdensityall(plotsingle=0,convflag=convflag)
    ax=gca()
    text(.05,.1,'$(a)$',horizontalalignment='left',transform=ax.transAxes,fontsize=16)
    axhline(.7,color='b',ls='--')
    axhline(.5,color='r',ls=':')

    subplot(1,2,2)
    plotsizeradiusall(plotsingle=0,convflag=convflag)
    ax=gca()
    text(.05,.1,'$(b)$',horizontalalignment='left',transform=ax.transAxes,fontsize=16)
    axhline(.7,color='b',ls='--')
    axhline(.5,color='r',ls=':')

    gca().set_yticklabels(([]))
    savefig(figuredir+'sizedensitycombo2.eps')
def plotsizelocaldensity5all(plotsingle=1,convflag=0,gim2d=0):
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    if plotsingle:
        figure(figsize=[8,4])
        clf()
        subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotsizelocaldensity(shapes[i-1],colors[i-1],plotsingle=0,ld=1,convflag=convflag)
        #multiplotaxes(i)

        xall=xall+x.tolist()
        yall=yall+y.tolist()

    xall=array(xall,'f')
    yall=array(yall,'f')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='_nolegend_')
    #errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xallg,yallg)
    #plot(xbin,ybin,'c-',marker='o',markersize=12,label='Groups')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,x1all,y1all)
    axhline(1,color='k',ls='--')
    xlabel('$log_{10}( \Sigma _{5} \ (Gal/Mpc^2)) $',fontsize=22)
    ylabel('$ R_e(24)/R_e(r) $',fontsize=22)
    if plotsingle:
        leg=legend(loc='upper left',numpoints=1)
        for t in leg.get_texts():
            t.set_fontsize('small')
    else:
        leg=legend(loc='upper left',numpoints=1,markerscale=0.8,borderpad=.15, borderaxespad=0,handletextpad=.1,labelspacing=.2)#,fontsize='x-small')
        for t in leg.get_texts():
            t.set_fontsize('small')

    ax=gca()
    ax.set_yscale('log')

    axis([-1.3,2.2,.2,5])
    print 'size vs local density (sigma_5)'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=12)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=12)
    if plotsingle:
        savefig(figuredir+'sizelocaldensity5all.eps')
        savefig(figuredir+'sizelocaldensity5all.png')

def plotsizelocaldensity10all(plotsingle=1,convflag=0,gim2d=0):
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    if plotsingle:
        figure(figsize=[8,4])
        
        clf()
        subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotsizelocaldensity(shapes[i-1],colors[i-1],plotsingle=0,ld=2,convflag=convflag)
        #multiplotaxes(i)

        xall=xall+x.tolist()
        yall=yall+y.tolist()

    xall=array(xall,'f')
    yall=array(yall,'f')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='Median')
    #errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xallg,yallg)
    #plot(xbin,ybin,'c-',marker='o',markersize=12,label='Groups')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,x1all,y1all)
    axhline(1,color='k',ls='--')
    xlabel('$log_{10}( \Sigma _{10} \ (Gal/Mpc^2)) $',fontsize=22)

    

    ax=gca()
    ax.set_yscale('log')

    axis([-.9,2.2,.2,5])
    print 'size vs local density (sigma_10)'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=12)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=12)
    if plotsingle:
        ylabel('$ R_e(24)/R_e(r) $',fontsize=22)
        leg=legend(loc='upper left',numpoints=1)
        for t in leg.get_texts():
            t.set_fontsize('small')

        savefig(figuredir+'sizelocaldensity10all.eps')
        savefig(figuredir+'sizelocaldensity10all.png')

def plotsizelocalmassdensityall(plotsingle=1,convflag=0):
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    if plotsingle:
        figure(figsize=[8,4])
        clf()
        subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    xc=[]
    yc=[]
    xg=[]
    yg=[]
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y=cl.plotsizelocalmassdensity(shapes[i-1],colors[i-1],plotsingle=0,convflag=convflag)
        #multiplotaxes(i)

        xall=xall+x.tolist()
        yall=yall+y.tolist()
        if cl.prefix in clusterlist:
            xc=xc+x.tolist()
            yc=yc+y.tolist()
        else:
            xg=xg+x.tolist()
            yg=yg+y.tolist()

    xall=array(xall,'f')
    yall=array(yall,'f')
    xg=array(xg,'f')
    yg=array(yg,'f')
    xc=array(xc,'f')
    yc=array(yc,'f')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='All')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binit(xg,yg,5)
    plot(xbin,ybin,'b-',marker='o',markersize=12,label='Groups')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b',label='_nolegend_')

    xbin,ybin,ybinerr=my.binit(xc,yc,5)
    plot(xbin,ybin,'r-',marker='o',markersize=12,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='r',label='_nolegend_')


    xlabel('$log_{10}( \Sigma _{mass}/M_\odot) $',fontsize=22)
    ylabel('$ R_e(24)/R_e(r) $',fontsize=22)
    leg=legend(numpoints=1,loc='upper left',scatterpoints=1,markerscale=0.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
    for t in leg.get_texts():
        t.set_fontsize('small')

    ax=gca()
    ax.set_yscale('log')

    axis([8.3,12.3,.2,5])
    axhline(y=1,color='k',ls='--')
    print 'cluster size vs local mass density'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    if plotsingle:
        savefig(figuredir+'SizeLocalmassdensityAll.eps')


def plotsomethingall(plotsingle=1,convflag=0):
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    if plotsingle:
        figure(figsize=[8,4])
        clf()
        subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xall=[]
    yall=[]
    xc=[]
    yc=[]
    xg=[]
    yg=[]
    for cl in mylocalclusters:
        i += 1
        #subplot(3,3,i)
        x,y,xf,yf=cl.plotsizecolor(shapes[i-1],colors[i-1],plotsingle=0)
        #multiplotaxes(i)

        xall=xall+x.tolist()
        yall=yall+y.tolist()
        if cl.prefix in clusterlist:
            xc=xc+x.tolist()
            yc=yc+y.tolist()
        else:
            xg=xg+x.tolist()
            yg=yg+y.tolist()

    xall=array(xall,'f')
    yall=array(yall,'f')
    xg=array(xg,'f')
    yg=array(yg,'f')
    xc=array(xc,'f')
    yc=array(yc,'f')
    xbin,ybin,ybinerr=my.binit(xall,yall,5)
    plot(xbin,ybin,'k-',marker='o',markersize=12,label='All')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binit(xg,yg,5)
    plot(xbin,ybin,'b-',marker='o',markersize=12,label='Groups')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b',label='_nolegend_')

    xbin,ybin,ybinerr=my.binit(xc,yc,5)
    plot(xbin,ybin,'r-',marker='o',markersize=12,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='r',label='_nolegend_')


    xlabel('$log_{10}( \Sigma _{mass}/M_\odot) $',fontsize=22)
    ylabel('$ R_e(24)/R_e(r) $',fontsize=22)
    leg=legend(numpoints=1,loc='upper left',scatterpoints=1,markerscale=0.6,borderpad=.2,labelspacing=.2,handletextpad=.2)
    for t in leg.get_texts():
        t.set_fontsize('small')

    ax=gca()
    ax.set_yscale('log')

    axis([8.3,12.3,.2,5])
    axhline(y=1,color='k',ls='--')
    print 'cluster size vs local mass density'
    r,p=spearman(xall,yall)
    ax=gca()
    text(.9,.9,'$rho = %4.2f$'%(r),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    text(.9,.85,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=14)
    if plotsingle:
        savefig(figuredir+'SizeLocalmassdensityAll.eps')

def plotR24vsReall2():
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']

    figure(figsize=[10,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xallb=[]
    yallb=[]
    xallg=[]
    yallg=[]
    xallr=[]
    yallr=[]
    x1allb=[]
    y1allb=[]
    x1allg=[]
    y1allg=[]
    x1allr=[]
    y1allr=[]
    for cl in mylocalclusters:
        i += 1
        subplot(3,1,1)
        #x,y,x1,y1=cl.plotR24vsRe(shapes[i-1],colors[i-1],cl.blueclustersample,cl.bluefieldsample,plotsingle=0)
        xb,yb,x1b,y1b=cl.plotR24vsRe('o','b',cl.blueclustersample,cl.bluefieldsample,plotsingle=0)
        subplot(3,1,2)
        xg,yg,x1g,y1g=cl.plotR24vsRe('o','g',cl.greenclustersample,cl.greenfieldsample,plotsingle=0)
        subplot(3,1,3)
        xr,yr,x1r,y1r=cl.plotR24vsRe('o','r',cl.redclustersample,cl.redfieldsample,plotsingle=0)
        #multiplotaxes(i)

        #multiplotlabels('$log_{10}(M_*/M_\odot) $','$ R_{24}/R_r$ ')

        xallb=xallb+xb.tolist()
        yallb=yallb+yb.tolist()
        xallg=xallg+xg.tolist()
        yallg=yallg+yg.tolist()
        xallr=xallr+xr.tolist()
        yallr=yallr+yr.tolist()

        x1allb=x1allb+x1b.tolist()
        y1allb=y1allb+y1b.tolist()
        x1allg=x1allg+x1g.tolist()
        y1allg=y1allg+y1g.tolist()
        x1allr=x1allr+x1r.tolist()
        y1allr=y1allr+y1r.tolist()

    xallb=array(xallb,'f')
    yallb=array(yallb,'f')
    xallg=array(xallg,'f')
    yallg=array(yallg,'f')
    xallr=array(xallr,'f')
    yallr=array(yallr,'f')

    x1allb=array(x1allb,'f')
    y1allb=array(y1allb,'f')
    x1allg=array(x1allg,'f')
    y1allg=array(y1allg,'f')
    x1allr=array(x1allr,'f')
    y1allr=array(y1allr,'f')
    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xall,yall)
    subplot(3,1,1)
    xbin,ybin,ybinerr=my.binitave(xallb,yallb,6)
    plot(xbin,ybin,'b-',marker='o',markersize=12,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitave(x1allb,y1allb,6)
    xbf=xbin
    ybf=ybin
    plot(xbin,ybin,'b-',mec='b',marker='s',mfc='0.9',markersize=13,label='Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b',label='_nolegend_')

    legend(numpoints=1)#,loc='upper right')
    subplot(3,1,2)
    xbin,ybin,ybinerr=my.binitave(xallg,yallg,6)
    plot(xbin,ybin,'g-',marker='o',markersize=12,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='g',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitave(x1allg,y1allg,6)
    plot(xbin,ybin,'g-',mec='g',marker='s',mfc='0.9',markersize=13,label='Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b',label='_nolegend_')

    plot(xbf,ybf,'b-',mec='b',label='_nolegend_')
    subplot(3,1,3)
    xbin,ybin,ybinerr=my.binitave(xallr,yallr,6)
    plot(xbin,ybin,'r-',marker='o',markersize=12,label='Clusters')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='r',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitave(x1allr,y1allr,6)
    plot(xbin,ybin,'r-',mec='r',marker='s',mfc='0.9',markersize=13,label='Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='b',label='_nolegend_')
    plot(xbf,ybf,'b-',mec='b',label='_nolegend_')

    #xbin,ybin,ybinerr=my.binitbins(9.5,11.5,4,xallg,yallg)
    #plot(xbin,ybin,'c-',marker='o',markersize=12,label='Groups')
    #errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    for i in range(1,4):
        subplot(3,1,i)
        ax=gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        axis([.7,60,2.5,15])
        if i < 3:
            ax.set_xticklabels(([]))
        if i == 2:
            ylabel('$ R_{24} \ (arcsec) $',fontsize=22)
    xlabel('$R_e \ NSA \ (arcsec) $',fontsize=22)

    savefig(figuredir+'R24vsRe.eps')
    #print 'cluster size vs mass'
    #spearman(xall,yall)
    #print 'field size vs mass'
    #spearman(x1all,y1all)

    #flag=xall < 25
    #print 'comparing field and cluster 24um size'
    #ks(yall[flag],y1all)

    #print 'comparing field and cluster Re NSA'
    #ks(xall[flag],x1all)

    #print 'comparing group and cluster ratios'
    #ks(yallg,yallc)
    #print 'comparing group and cluster masses'
    #ks(xallg,xallc)
    #print 'comparing group and field ratios'
    #ks(yallg,y1all)
    #print 'comparing group and field masses'
    #ks(xallg,x1all)

def plotsizemasscolorall():

    figure(figsize=[10,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    xallb=[]
    yallb=[]
    xallg=[]
    yallg=[]
    xallr=[]
    yallr=[]
    x1allb=[]
    y1allb=[]
    x1allr=[]
    y1allr=[]
    x1allg=[]
    y1allg=[]
    for cl in mylocalclusters:
        x,y,x1,y1=cl.plotsizemasscolor(0,plotsingle=0)
        xallb=xallb+x.tolist()
        yallb=yallb+y.tolist()
        x1allb=x1allb+x1.tolist()#field galaxies
        y1allb=y1allb+y1.tolist()

        x,y,x1,y1=cl.plotsizemasscolor(1,plotsingle=0)
        xallg=xallg+x.tolist()
        yallg=yallg+y.tolist()
        x1allg=x1allg+x1.tolist()#field galaxies
        y1allg=y1allg+y1.tolist()

        x,y,x1,y1=cl.plotsizemasscolor(2,plotsingle=0)
        xallr=xallr+x.tolist()
        yallr=yallr+y.tolist()
        x1allr=x1allr+x1.tolist()#field galaxies
        y1allr=y1allr+y1.tolist()

    xallb=array(xallb,'f')
    yallb=array(yallb,'f')
    xallg=array(xallg,'f')
    yallg=array(yallg,'f')
    xallr=array(xallr,'f')
    yallr=array(yallr,'f')
    x1allb=array(x1allb,'f')
    y1allb=array(y1allb,'f')
    x1allg=array(x1allg,'f')
    y1allg=array(y1allg,'f')
    x1allr=array(x1allr,'f')
    y1allr=array(y1allr,'f')
    
    xbin,ybin,ybinerr=my.binitbins(9.5,12.,5,xallb,yallb)
    plot(xbin,ybin,'b-',marker='o',markersize=8,label='Blue Cluster Gal')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitbins(9.5,12.,5,xallg,yallg)
    plot(xbin,ybin,'g-',marker='o',markersize=8,label='Green Cluster Gal')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitbins(9.5,12.,5,xallr,yallr)
    plot(xbin,ybin,'r-',marker='o',markersize=8,label='Red Cluster Gal')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitbins(9.5,12.,5,x1allb,y1allb)
    #plot(xbin,ybin,'k-',color='b',marker='s',mfc='None',markersize=6,label='Blue Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitbins(9.5,12.,5,x1allg,y1allg)
    #plot(xbin,ybin,'k-',color='g',marker='s',mfc='None',markersize=6,label='Green Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xbin,ybin,ybinerr=my.binitbins(9.5,12.,5,x1allr,y1allr)
    #plot(xbin,ybin,'k-',color='r',marker='s',mfc='None',markersize=6,label='Red Field')
    errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',label='_nolegend_')

    xlabel('$log_{10} (M_*/M_\odot) $',fontsize=22)
    ylabel('$ R_{24}/R_r $',fontsize=22)
    legend(loc='upper right',numpoints=1)
    ax=gca()
    ax.set_yscale('log')
    savefig(figuredir+'SizeMassColorAll.eps')

    print 'comparing blue field and cluster ratios'
    ks(yallb,y1allb)

    print 'comparing blue and green cluster sizes'
    ks(yallb,yallg)
    print 'comparing blue and red cluster sizes'
    ks(yallb,yallr)
    print 'comparing green and red cluster sizes'
    ks(yallg,yallr)



def plotcompareR90all():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.compareR90()
    ax=gca()
    text(-.75,-.35,'$r-band \ radius$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,r'$24 \ radius$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(figuredir+'PlotsCompareR90All.eps')

def plotcomparediffR90localdensall():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    i=0
    for cl in mylocalclusters:
        i += 1
        subplot(3,3,i)
        cl.comparediffR90localdens()
    ax=gca()
    text(-.75,-.35,'$r-band \ radius$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    subplot(3,3,4)
    text(-2.8,1.9,r'$24 \ radius$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)


    savefig(figuredir+'PlotsCompareR90All.eps')

def plotabellclusters():
    figure(figsize=(8,10))
    a2052.plotpositionson24(usefwhm24=0,plotsingle=0)
    legend(numpoints=1,scatterpoints=1)
    a2063.plotpositionson24(plotsingle=0,usefwhm24=0)
    axis([228,232,5,11])
    

def plotoptirdistXray():
    figure(figsize=[10,10])
    clf()
    subplots_adjust(wspace=.25,hspace=.35)
    for cl in mylocalclusters:
        #subplot(3,3,i)
        cl.optiroffset()
        figfile=figuredir+cl.prefix+'OptIRDistXrayContour.eps'
        savefig(figfile)
    #ax=gca()
    #text(-.75,-.35,'$r-band \ radius$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    #subplot(3,3,4)
    #text(-2.8,1.9,r'$24 \ radius$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes)

def plotlcscolormag():
    figure(figsize=(10,8))
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    noylabel=[2,3,5,6,8,9]
    for cl in mylocalclusters:
        i=i+1
        subplot(3,3,i)
        cl.plotcolormag(plotsingle=0)

	ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))

    ax=gca()

    #text(-.75,-.35,'$M_r$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    #text(-2.8,1.9,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
    #text(-.5,-.25,'$log_{10}(M_* /M_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-.5,-.25,'$ M_r $',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.2,1.5,'$ NUV - r $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'LCScolormag.eps')
    savefig(figuredir+'LCScolormag.png')

def plotlcscolormass(convflag=0):
    figure(figsize=(10,8))
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    noylabel=[2,3,5,6,8,9]
    for cl in mylocalclusters:
        i=i+1
        subplot(3,3,i)
        cl.plotcolormass(plotsingle=0,convflag=convflag)

	ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))

    ax=gca()

    #text(-.75,-.35,'$M_r$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    #text(-2.8,1.9,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
    #text(-.5,-.25,'$log_{10}(M_* /M_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-.5,-.25,'$ log_{10}(M_*) $',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.3,1.5,'$ NUV - r $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

    savefig(figuredir+'LCScolormass.eps')
    savefig(figuredir+'LCScolormass.png')

def plotlcsfieldcolormag():
    figure(figsize=(12,8))
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    noylabel=[2,3,5,6,8,9]
    for cl in mylocalclusters:
        i=i+1
        subplot(3,3,i)
        cl.plotcolormagfield(plotsingle=0)

	ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))

    ax=gca()

    #text(-.75,-.35,'$M_r$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    #text(-2.8,1.9,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
    #text(-.5,-.25,'$log_{10}(M_* /M_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-.5,-.25,'$ M_r $',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.2,1.5,'$ NUV - r $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
    savefig(figuredir+'LCSfieldcolormag.eps')

def plotlcscolormagbylir():
    figure(figsize=(10,8))
    subplots_adjust(wspace=.01,hspace=.01)
    i=0
    noylabel=[2,3,5,6,8,9]
    for cl in mylocalclusters:
        i=i+1
        subplot(3,3,i)
        cl.plotcolormag(plotsingle=0)

	ax=gca()
        if i < 7:
            ax.set_xticklabels(([]))
        if i in noylabel:
            ax.set_yticklabels(([]))

    ax=gca()

    #text(-.75,-.35,'$M_r$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    #text(-2.8,1.9,'$u-r$',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
    text(-.5,-.25,'$log_{10}(M_* /M_\odot)$',fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.2,1.5,'$ NUV - r $',fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
    savefig(figuredir+'LCScolormag.eps')

def plotHImassRelations():
    figure(figsize=(12,10))
    subplots_adjust(wspace=.25,hspace=.35)
    clf()
    subplot(2,2,1)
    for cl in mylocalclusters:
        flag=cl.HIflag
        plot(cl.sdssvopt[flag]/1000,cl.HImass[flag],marker='.',ls='None',label=cl.prefix)
    xlabel('$\mathrm{Recession \ Velocity / (1000 \ km/s)}$',fontsize=16)
    ylabel('$\mathrm{HI \ mass \ (M_\odot)}$',fontsize=16)
    title('$\mathrm{HI mass vs redshift}$')
    ax=gca()
    ax.set_yscale('log')
    #legend(numpoints=1)
    subplot(2,2,2)
    for cl in mylocalclusters:
        cl.plotDiamHImass()
    title('$\mathrm{HI \ mass \ vs \ Diameter}$')
    #compare different mass estimates from Toribio
    subplot(2,2,3)
    for cl in mylocalclusters:
        cl.plotMrHImass()
    title('$\mathrm{HI mass vs Abs r Mag')
    subplot(2,2,4)
    for cl in mylocalclusters:
        cl.plotgrHImass()
    title('HI mass vs g-r color')

def plotcumulativez(plotsingle=1):
    varstring='redshift'
    xmin=zmin
    xmax=zmax+.005
    sxlabel='$ZDIST $'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'CumulativeRedshift.eps')

def plotcumulativemass(plotsingle=1):
    varstring='stellarmass'
    xmin=9.5
    xmax=12.5
    sxlabel='$log_{10}(M_*/M_\odot) $'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'CumulativeMass.eps')

def plotcumulativeRe(plotsingle=1):
    varstring='Re'
    xmin=1.
    xmax=25.
    sxlabel='$R_e \ (arcsec)$'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'CumulativeRe.eps')

def plotcumulativeR24(plotsingle=1):
    varstring='R24'
    xmin=0.
    xmax=15.
    sxlabel='$FLUX\_ RADIUS \ 24um \ (arcsec)$'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax)
    if plotsingle:
        savefig(figuredir+'CumulativeR24.eps')

def plotcumulativeratioR(plotsingle=1):
    varstring='ratioR'
    xmin=0.
    xmax=7.
    sxlabel='$ SERSIC\_ TH50/FLUX\_ RADIUS_{24}$'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax)
    savefig(figuredir+'CumulativeR24.eps')

def plotcumulativeNUVr(plotsingle=1):
    varstring='NUVr'
    xmin=0.5
    xmax=6.5
    sxlabel='$NUV - r $'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'CumulativeNUVr.eps')

def plotcumulativeNUV24(plotsingle=1):
    varstring='NUV24'
    xmin=-36.
    xmax=-25.
    sxlabel='$NUV - m_{24}$'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'CumulativeNUV24.eps')

def plotcumulative24mass(plotsingle=1):
    xmin=-2.
    xmax=12.
    varstring='24mass'
    sxlabel='$ M_{24} - log_{10}(M_*/M_\odot) $'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'Cumulative24mass.eps')

def plotcumulativeBT(plotsingle=1):
    xmin=-0.15
    xmax=1.05
    varstring='BT'
    sxlabel='$ GIM2D B/T $'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'CumulativeBT.eps')

def plotcumulativesmoothness(plotsingle=1):
    xmin=-0.15
    xmax=1.05
    varstring='BT'
    sxlabel='$ GIM2D B/T $'
    sylabel='$Cumulative \ Distribution $'
    plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=plotsingle)
    if plotsingle:
        savefig(figuredir+'CumulativeBT.eps')

def plotcumulative(varstring,sxlabel,sylabel,xmin,xmax,pltsingle=1):
    if pltsingle:
        figure(figsize=(6,4.))
        subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    i=0
    xall=[]
    yall=[]
    xallg=[]
    yallg=[]
    xallc=[]
    yallc=[]
    x1all=[]
    y1all=[]
    xnearc=[] # compare Coma and Abell 1367
    ynearc=[]
    xfarc=[] # with A2063 and A2052
    yfarc=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        plotvar=cl.varlookup[varstring]
        #cflag=cl.blueclustersample
        #fflag=cl.bluefieldsample
        cflag=cl.galfitflag & cl.member
        fflag=cl.galfitflag & cl.field
        if varstring.find('BT') > -1:
            cflag=cflag & cl.gim2d.matchflag 
            fflag=fflag & cl.gim2d.matchflag 
        x=plotvar[cflag]
        x1=plotvar[fflag]
        if i < 5:
            xallg=xallg+x.tolist()
        else:
            xallc=xallc+x.tolist()
        if cl.prefix.find('Coma') > -1:
            print 'not including coma in cluster sample for cumulative ',varstring,' plot'
            xnearc=xnearc+x.tolist()
        xall=xall+x.tolist() # all clusters
        #elif cl.prefix.find('A2063')> -1:
        #    print 'not including A2063 in cluster sample for cumulative ',varstring,' plot'
        #    xfarc=xfarc+x.tolist()

        if cl.prefix.find('A2052')> -1:
            xfarc=xfarc+x.tolist()
        if cl.prefix.find('A1367')> -1:
            xnearc=xnearc+x.tolist()


        x1all=x1all+x1.tolist()#field galaxies
    xall=array(xall,'f')
    xallg=array(xallg,'f')
    xallc=array(xallc,'f')
    x1all=array(x1all,'f')
    xnearc=array(xnearc,'f')
    xfarc=array(xfarc,'f')
    hist(xall,bins=len(xall),range=(xmin,xmax),cumulative=True,normed=True,histtype='step',color='red',label='Cluster')
    hist(x1all,bins=len(x1all),range=(xmin,xmax),cumulative=True,normed=True,histtype='step',color='blue',label='Field')
    xlabel(sxlabel,fontsize=18)
    ylabel(sylabel,fontsize=18)
    legend(loc='lower right')
    axis([xmin,xmax,-.01,1.05])

    print 'comparing field and cluster'
    D,p=ks(xall,x1all)
    ax=gca()
    text(.05,.9,'$D = %4.2f$'%(D),horizontalalignment='left',transform=ax.transAxes,fontsize=14)
    text(.05,.8,'$p = %5.4f$'%(p),horizontalalignment='left',transform=ax.transAxes,fontsize=14)
    print 'comparing near and far clusters'
    D,p=ks(xnearc,xfarc)

def plotcumulativesubplots():

    figure(figsize=(10,8))
    subplots_adjust(wspace=.3,hspace=.3, left=.1, bottom=.1, right=.95, top=.95)
    subplot(2,2,1)
    plotcumulativemass(plotsingle=0)
    text(.05,.1,'$(a)$',horizontalalignment='left',transform=gca().transAxes,fontsize=18)
    subplot(2,2,2)
    plotcumulativeRe(plotsingle=0)
    text(.05,.1,'$(b)$',horizontalalignment='left',transform=gca().transAxes,fontsize=18)
    subplot(2,2,3)
    plotcumulativez(plotsingle=0)
    text(.05,.1,'$(c)$',horizontalalignment='left',transform=gca().transAxes,fontsize=18)
    subplot(2,2,4)
    #plotcumulativeNUVr(plotsingle=0)
    plotcumulativeBT(plotsingle=0)
    text(.05,.1,'$(d)$',horizontalalignment='left',transform=gca().transAxes,fontsize=18)
    savefig(figuredir+'CumulativeSubplots.eps')
def compareSEandgalfit():
    figure()
    clusters=[coma,mkw11]
    c=['bo','ko']
    for i in range(len(clusters)):
        cl=clusters[i]

        y=(cl.sex24.FLUX_RADIUS1)*mipspixelscale
        x=cl.galfit24.re1*mipspixelscale
        flag=cl.sex24.MATCHFLAG24 & (cl.galfit24.re1 > .1) & (cl.snr24 > 3)
        #print 'FLUX_RADIUS1 vs Re'
        #spearman(x[flag],y[flag])
        if i == 0:
            plot(x[flag],y[flag],'bo',label='Coma')#,c[i],label=cl.prefix)
        if i == 1:
            plot(x[flag],y[flag],'ko',label='MKW11')#,c[i],label=cl.prefix)
    ylabel('$SE \ FLUX\_RADIUS \ (50\%) \ (arcsec) $',fontsize=18)
    xlabel('$ GALFIT \ R_e \ (arcsec) $',fontsize=18)
    axis([0,16,0,15])
    legend(numpoints=1,loc='upper left')
    xl=arange(15)
    plot(xl,xl,'k--')

    fig=figuredir+'SEGalfitcomparison.eps'
    savefig(fig)



mkw11=cluster('MKW11')
coma=cluster('Coma')
mkw8=cluster('MKW8')
awm4=cluster('AWM4')
a2052=cluster('A2052')
a2063=cluster('A2063')
ngc=cluster('NGC6107')
herc=cluster('Hercules')
a1367=cluster('A1367')
##print 'usecomaflag = ', usecomaflag
if usecomaflag:
    clustersbymass=[mkw11,awm4,mkw8,ngc,a2052,a2063,herc,a1367,coma]
    #clustersbydistance=[a1367,mkw11,coma,mkw8,ngc,awm4,a2052,a2063,herc]
    clustersbydistance=[a1367,mkw11,coma,mkw8,ngc,awm4,a2063,a2052,herc]
    clustersbylx=[mkw11,ngc,mkw8,awm4,herc,a1367,a2063,a2052,coma]
else:
    clustersbymass=[mkw11,awm4,mkw8,ngc,a2052,a2063,herc,a1367]
    clustersbydistance=[a1367,mkw11,mkw8,ngc,awm4,a2063,a2052,herc]
    clustersbylx=[mkw11,ngc,mkw8,awm4,herc,a1367,a2063,a2052]
#mylocalclusters=clustersbylx
mylocalclusters=clustersbydistance

##mylocalclusters=clustersbymass
#clusterswoutcoma=[mkw11,awm4,mkw8,ngc,a2052,a2063,herc,a1367]





#mylocalclusters=[mkw11,a2052,a2063]
#mylocalclusters=[mkw11,mkw8,awm4,ngc,a2063,a2052,coma]
#plotpositionsall()
#plotpositionson24all()
#plotveldrall()
#getSpirals()
#mkw11.fitprofiles()

#plotallRatiosLocaldens()

#print 'Summary statistics:'
#for cl in clustersbylx:
#    print cl.prefix, 'member spirals = ',sum((cl.galfitflag & cl.member)), ', field Sp = ',sum((cl.galfitflag & cl.field))

def plotpaperfigures(cflag=1,plotsample=0):
    if plotsample:
        plotsigmaLxall()
        plotredshiftdron24all()
        plotpositionson24all()
        plotlcscolormag()
        plotlcscolormass()
    #plotlcsfieldcolormag()
    #plotLIRhistall()
    #plotSFR24StellarMassall()
    plotcumulativesubplots()
    plotRe24vsReall()
    #plotsSFRHIDefall()
    #plotsizesSFRall()
    plotsizestellarmassall(convflag=cflag)
    plotsizecolorall(convflag=cflag)

    plotsizelocaldensity(convflag=cflag)
    plotsizedensitycombo2(convflag=cflag)
    #plotsizelocaldensity5all()
    #plotsizelocaldensity10all()
    #plotsizelocalmassdensityall()
    #plotsizeradiusall()
    #plotsizedist3dall()

    plottrunspiralvsmassall(convflag=cflag)
    plotsfspiralvsmassall()    
    plotsizeHIdefall(convflag=cflag)
def plotallcumulative():
    plotcumulativez()
    plotcumulativemass()
    plotcumulativeRe()
    plotcumulativeR24()
    plotcumulativeNUVr()
    plotcumulativeNUV24()
    plotcumulative24mass()
    plotcumulativeBT()
    #plotsizemassall()
#plotpaperfigures()


def plotspiralparameters(): # plot range of parameters to use as input for galfit simulation
    for cl in mylocalclusters:
        figure()
        title(cl.prefix)
        plotflag=cl.spiralflag 
        subplot(2,2,1)
        hist(cl.n.SERSIC_TH50)
        subplot(2,2,2)
        hist(cl.n.SERSIC_N)
        subplot(2,2,3)
        hist(cl.n.SERSIC_BA)
        subplot(2,2,4)

