#!/usr/bin/env python
import pyfits
from LCScommon import *
from pylab import *
import os
import mystuff as my
import aplpy
import ds9
from astropy.io import ascii

#from LCSReadmasterBaseWithProfileFits import *
from LCSReadmasterBaseNSA import *
import matplotlib.cm as cm
#import chary_elbaz_24um as chary
#from scipy.stats.binom import cdf
Lsol=3.826e33#normalize by solar luminosity
bellconv=9.8e-11#converts Lir (in L_sun) to SFR/yr
bellconv=4.5e-44#Kenn 98 conversion fro erg/s to SFR/yr

truncation_fraction=0.6
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
        try:
            self.readGalfitSersicResults()
        except:
            print self.prefix,": couln't read galfit sersic results"
        try:
            self.readGalfitResults()
        except:
            print self.prefix,": couldn't read galfit results"
        #self.size24=self.sex24.FWHM_DEG*3600.
        self.size24=self.sex24.FLUX_RADIUS1*mipspixelscale
        self.sizeratio=self.galfit24.cre1*mipspixelscale/(self.gim2d.Rhlr_2/self.gim2d.Scale_2)
        self.mingalaxymass=5.e9
        self.rmagcut=20.+5.*log10(self.cdMpc/(clusterbiweightcenter['Hercules']/H0))
        self.rmag=22.5-2.5*log10(self.n.SERSICFLUX[:,2])
        #self.galfitflag=self.On24ImageFlag & (self.snr24>10) & ~self.galfit24.numerical_error_flag24  & ~self.galfit24.cnumerical_error_flag24 & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag& ~self.agnflag #& (self.ce.LIR > 5.1e8) & (self.rmag < self.rmagcut) #(self.stellarmass > self.mingalaxymass)
        try:
            #self.galfitflag=self.On24ImageFlag & (self.snr24>snr24cut) & ~self.galfit24.numerical_error_flag24   & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag& ~self.galfit24.cnumerical_error_flag24 #& ~self.agnflag #& (self.ce.LIR > 5.1e8) & (self.stellarmass > minstellarmass)#& (self.rmag < self.rmagcut) #
            self.galfitflag=self.On24ImageFlag & (self.snr24>snr24cut) &  (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag & ~self.agnflag #& ~self.galfit24.numerical_error_flag24   #& (self.ce.LIR > 5.1e8) & (self.stellarmass > minstellarmass)#& (self.rmag < self.rmagcut) #
            self.galfitsample=self.On24ImageFlag & (self.snr24>snr24cut) & ~self.galfit24.numerical_error_flag24   & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag& ~self.agnflag & ~self.galfit24.cnumerical_error_flag24 & (self.ce.LIR > 5.13e8) & (self.stellarmass > minstellarmass)#& (self.rmag < self.rmagcut) #

        except:
            print 'no galfit data yet?  get moving!'
        
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
        self.varlookup={'stellarmass':log10(self.stellarmass),'Re':self.n.SERSIC_TH50,'R24':self.galfit24.re1*mipspixelscale,'NUV':self.n.ABSMAG[:,1],'r':self.n.ABSMAG[:,4],'m24':self.sex24.MAG_BEST,'redshift':self.n.ZDIST,'NUVr':(self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]),'NUV24':(self.n.ABSMAG[:,1]-self.sex24.MAG_BEST),'24mass':(self.sex24.MAG_BEST-log10(self.stellarmass)),'ratioR':self.sex24.FLUX_RADIUS1*mipspixelscale/self.n.SERSIC_TH50}

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
        flag=self.galfitflag & self.member
        sp=scatter(x[flag],y[flag],marker=shapes,c=col[flag],vmin=.6,vmax=1.1,label=self.prefix,s=40)
        if self.prefix.find('MKW11') > -1:
            colorbar(sp)
        #errorbar(x[flag],y[flag],yerr=yer[flag],fmt=None,ecolor=colors)
        flag2=self.galfitflag & self.field

        if sum(flag2)>0:
            scatter(x[flag],y[flag],marker=shapes,c=col[flag],vmin=.2,vmax=3,label='_nolegend_',s=40)
        return x[flag],y[flag],x[flag2],y[flag2]



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
	#ax.set_yscale('log')
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
        xticks(arange(int(xmin),xmax+.5,1,'i'),fontsize=fsize)
        yticks(fontsize=fsize)
        #ymin,ymax=ylim()
        if plotsingle:
            
            xmin=10.**(xmin)*Lsol*4.5e-44
            xmax=10.**(xmin)*Lsol*4.5e-44
            ax=gca()
            xaxissfrlog(ax)
        if plotsingle:
            fsize=20
        else:
            fsize=14
        if plotclname:
            text(.9,.8,'$ '+self.clustername+' $',fontsize=fsize,transform=ax.transAxes,horizontalalignment='right')
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)
        if plotsingle:
            xlabel('$ log_{10} (L_{IR}/L_\odot) $',fontsize=18)
            ylabel('$ N_{gal} $',fontsize=18)
            text(.5,1.1,'$ SFR \ (M_\odot/yr) $',fontsize=fsize,transform=ax.transAxes,horizontalalignment='center')
    def plotSFR24stellmass(self,colors,shapes):
        #figure(1)
        #plotflag=self.apexflag & self.HIflag & ~self.agn2 & self.spiralflag
        #plotflag=self.apexflag  & ~self.agn2 & self.spiralflag

        #flag=self.ellipseflag & self.spiralflag & self.member & ~self.agnflag
        #flag=self.ellipseflag & self.member & ~self.agnflag
        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.member & self.spiralflag & (self.ce.LIR > self.Lir80)
        #flag=self.galfitflag & self.member
        print self.prefix,' numb of IR members = ',sum(flag)
        x=self.stellarmass[flag]
        y=self.ce.SFR[flag]
        #yerr=self.SFR24err[flag]
        #plot(x,y,'ko',color=colors,marker=shapes,label=self.prefix,markersize=8)
        if self.prefix.find('MKW11')> -1:
            clabel='Cluster Sp'

        else:
            clabel='_nolegend_'

        plot(x,y,'ko',color=colors,marker=shapes,label=clabel,markersize=8)
        #errorbar(x,y,yerr,fmt=None,color=colors,label='_nolegend_')

        #flag=self.ellipseflag & self.spiralflag & ~self.member & ~self.agn1
        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.field & self.spiralflag & (self.ce.LIR > self.Lir80)
        #flag=self.galfitflag & ~self.member
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

        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.member & self.ellipticalflag & (self.ce.LIR > self.Lir80)
        #flag=self.galfitflag & ~self.member
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        xe=self.stellarmass[flag]
        ye=self.ce.SFR_ZDIST[flag]
        plot(xe,ye,'k.',label='_nolegend_')
        flag=self.sex24.MATCHFLAG24 & ~self.agnflag & self.field & self.ellipticalflag & (self.ce.LIR > self.Lir80)
        #flag=self.galfitflag & ~self.member
        print self.prefix,' numb of IR NON-members = ',sum(flag)
        xe=self.stellarmass[flag]
        ye=self.ce.SFR_ZDIST[flag]
        plot(xe,ye,'k.',color='0.5',label='_nolegend_')

	#ax=gca()
	#ax.set_yscale('log')
        #axis([-2.5,1.5,0,22])
        #xmin,xmax=xlim()
        #xticks(arange(int(xmin),xmax,1,'i'),fontsize=10)
        #ymin,ymax=ylim()
        #title(self.clustername)
        return x.tolist(),y.tolist(),x1.tolist(),y1.tolist()
        #yticks(arange(round(ymin),ymax+1,5,'i'),fontsize=10)

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

    def plotsSFRLocaldens(self,colors,shapes): 
        plotflag=self.ellipseflag & self.spiralflag
        plotflag=self.sex24.MATCHFLAG24 & ~self.ellipticalflag & ~self.agnflag & self.dvflag
        sSFR=self.size24[plotflag]/self.n.SERSIC_TH50[plotflag]
        plot(self.localdens[plotflag],sSFR,'ko',marker=shapes,color=colors,label=self.prefix,markersize=10)
        x=self.localdens[plotflag].tolist()
        y=sSFR.tolist()
        print 'min local dens = ',min(self.localdens[plotflag])
        return x,y

    def spiralsLocaldens(self):#,colors,shapes): 
        keepflag=self.On24ImageFlag &self.spiralflag& ~self.agnflag  & (self.stellarmass > minstellarmass)#
        sfflag= self.On24ImageFlag &self.spiralflag& ~self.agnflag  & (self.stellarmass > minstellarmass) & (self.ce.LIR > 5.1e8)

        y=array(sfflag[keepflag],'f')
        x=self.ld.SIGMA_5[keepflag]
        x2=self.ld.SIGMA_10[keepflag]
        x3=self.ld.SIGMA_NN[keepflag]
        x4=self.ld.RHOMASS[keepflag]
        print x,y
        print len(x),len(y),len(x2),len(x3),len(x4)
        return x.tolist(),x2.tolist(),x3.tolist(),x4.tolist(),y.tolist()

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


    def plotcheckvsf24(self,plotsingle=1):
        if plotsingle:
            figure()
        subplot(2,2,1)
        x=self.sex24.MAG_BEST
        y=self.galfit24.re1*mipspixelscale
        yer=self.galfit24.re1err*mipspixelscale
        flag=self.galfitflag & self.dvflag
        plot(x[flag],y[flag],'ko')
        errorbar(x[flag],y[flag],yer[flag],fmt=None)
        subplot(2,2,2)
        x=self.sex24.MAG_BEST
        y=self.n.SERSIC_TH50
        flag=self.galfitflag & self.dvflag
        plot(x[flag],y[flag],'ko')

        subplot(2,2,3)
        x=self.sex24.MAG_BEST
        y=22.5-2.5*log10(self.n.NMGY[:,4])
        flag=self.galfitflag & self.dvflag
        plot(x[flag],y[flag],'ko')


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
    figure(figsize=[10,8])
    clf()
    subplots_adjust(wspace=.02,hspace=.02)
    i=0
    for cl in mylocalclusters:
        i +=1
        subplot(3,3,i)
        cl.plotLIRhist(plotsingle=0)
        multiplotaxes(i)
        
    multiplotlabels('$log_{10}(L_{IR}/L_\odot)$','$N_{gal}$')
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

def plotfracsfspirallocaldens():
    figure(figsize=[15,10])
    #figure()
    clf()
    #subplots_adjust(wspace=.25,hspace=.35)
    #colors=['r','b','c','g','m','y','k','0.5','r']
    #shapes=['o','^','s','>','p','d','v','o','s']
    xall=[]
    x2all=[]
    x3all=[]
    x4all=[]
    yall=[]
    i=0
    for cl in mylocalclusters:
        i += 1
        x,x2,x3,x4,y=cl.spiralsLocaldens()
        xall=xall+x
        x2all=x2all+x2
        x3all=x3all+x3
        x4all=x4all+x4
        yall=yall+y
    legend(numpoints=1,loc='upper right')
    xlist=[xall,x2all,x3all,x4all]
    xlabels=['$\Sigma_5$','$\Sigma_{10}$','$\Sigma_{NN}$','$\Sigma_{M_*}$']
    ylabels=['$Fraction \ SF \ Spiral $','','$Fraction \ SF \ Spiral $','']
    xmin=[.1,.2,.5,4.e8]
    xmax=[100,100,400,2.e12]
    for i in range(len(xlist)):
        subplot(2,2,i+1)
        print len(xlist[i]),len(yall)
        plot(xlist[i],yall,'k.')
        #flag=yall > .1
        #t=xlist[i]
        #hist(t[flag],cumulative=True)#,color='b')
        #hist(t[i][~flag],cumulative=True)#,color='r')
        xbin,ybin,ybinerr=my.binitave(xlist[i],yall,10)
        plot(xbin,ybin,'ko',mfc='None',markersize=15)
        errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k')
        print xlabels[i]
        r,p=spearman(xlist[i],yall)
        spearman(xbin,ybin)
        ax=gca()
        text(.1,.15,'$ rho = %4.2f$'%(r),horizontalalignment='left',transform=ax.transAxes,fontsize=14)
        text(.1,.1,'$p = %4.4f$'%(p),horizontalalignment='left',transform=ax.transAxes,fontsize=14)

        ax=gca()
        ax.set_xscale('log')
        xlabel(xlabels[i],fontsize=20)
        ylabel(ylabels[i],fontsize=20)
        axis([xmin[i],xmax[i],-.05,1.05])

    #axis([.9,1000,-.05e-10,.85e-9])
    f=figuredir+'fracsfspiralLocaldens.eps'
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


def plotcheckf24all():
    figure()
    clf()
    for cl in mylocalclusters:
        cl.plotcheckvsf24(plotsingle=0)


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
    



mkw11=cluster('MKW11')
coma=cluster('Coma')
mkw8=cluster('MKW8')
awm4=cluster('AWM4')
a2052=cluster('A2052')
a2063=cluster('A2063')
ngc=cluster('NGC6107')
herc=cluster('Hercules')
a1367=cluster('A1367')
#print 'usecomaflag = ', usecomaflag
if usecomaflag:
    clustersbymass=[mkw11,awm4,mkw8,ngc,a2052,a2063,herc,a1367,coma]
    clustersbydistance=[a1367,mkw11,coma,mkw8,ngc,awm4,a2052,a2063,herc]
    clustersbylx=[mkw11,ngc,mkw8,awm4,herc,a1367,a2063,a2052,coma]
else:
    clustersbymass=[mkw11,awm4,mkw8,ngc,a2052,a2063,herc,a1367]
    clustersbydistance=[a1367,mkw11,mkw8,ngc,awm4,a2052,a2063,herc]
    clustersbylx=[mkw11,ngc,mkw8,awm4,herc,a1367,a2063,a2052]
mylocalclusters=clustersbylx
#mylocalclusters=[mkw11,coma,mkw8]


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

        
