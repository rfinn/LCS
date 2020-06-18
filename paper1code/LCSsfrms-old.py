#!/usr/bin/env python

'''


'''

from LCScommon import *
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import os
import mystuff as my
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.cosmology import WMAP9 as cosmo

import astrofuncs
#import chary_elbaz_24um as chary
import anderson
from scipy.stats import scoreatpercentile

import argparse# here is min mass = 9.75

parser = argparse.ArgumentParser(description ='Plots for SFR-M* paper')
parser.add_argument('--minmass', dest = 'minmass', default = 9., help = 'minimum stellar mass for sample.  default is log10(M*) > 7.9')
parser.add_argument('--diskonly', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')

args = parser.parse_args()

USE_DISK_ONLY = np.bool(np.float(args.diskonly))#True # set to use disk effective radius to normalize 24um size
mstarmin=float(args.minmass)
mstarmax=10.8
minmass=mstarmin #log of M*
ssfrmin=-12.
ssfrmax=-9

class spirals():
    def __init__(self,infile,usecoma=True,useherc=True,onlycoma=False,prefix='all'):
        self.prefix=prefix 
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print '\n prefix = \n',self.prefix
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%'

        hdulist=fits.open(homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all_fsps_v2.4_miles_chab_charlot_sfhgrid01.fits')
        self.jmass=hdulist[1].data
        hdulist.close()
        # use jmass.mstar_50 and jmass.mstar_err

        hdulist=fits.open(homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_isorad.fits')
        self.isorad=hdulist[1].data
        hdulist.close()

        hdulist=fits.open(homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_AGC.fits')
        self.agc=hdulist[1].data
        hdulist.close()


        hdulist=fits.open(infile)
        self.s=hdulist[1].data


        #self.membflag = (self.s.DR_R200 < 1.) & (abs(self.dv) < 3.)
        if usecoma == False:
            self.s=self.s[(self.s.CLUSTER != 'Coma') | (abs(self.s.DELTA_V) > 3.)]
            try:
                self.jmass=self.jmass[self.s.CLUSTER != 'Coma']
                self.isorad=self.isorad[self.s.CLUSTER != 'Coma']
                self.agc=self.agc[self.s.CLUSTER != 'Coma']
            except:
                print 'WARNING: problem matching to moustakas MSTAR_50 - probably ok'


            #self.agnflag=self.agnflag[self.s.CLUSTER != 'Coma']
            self.comaflag=False
        if onlycoma == True:
            self.s=self.s[self.s.CLUSTER == 'Coma']
            #self.jmass=self.jmass[self.s.CLUSTER == 'Coma']
            #self.isorad=self.isorad[self.s.CLUSTER == 'Coma']
            #self.agc=self.agc[self.s.CLUSTER == 'Coma']


        if useherc == False:
            self.s=self.s[self.s.CLUSTER != 'Hercules']
            #self.jmass=self.jmass[self.s.CLUSTER != 'Hercules']
            #self.isorad=self.isorad[self.s.CLUSTER != 'Hercules']
            #self.agc=self.agc[self.s.CLUSTER != 'Hercules']
            #self.logstellarmassTaylor=self.logstellarmassTaylor[self.s.CLUSTER != 'Hercules']


        cols=self.s.columns
        cnames=cols.names
        hdulist.close()
        self.logstellarmassTaylor=1.15+0.70*(self.s.ABSMAG[:,3]-self.s.ABSMAG[:,5]) -0.4*(self.s.ABSMAG[:,5]+ 5.*log10(h))
        bad_imag=self.logstellarmassTaylor < 5.
        newi=(self.s.ABSMAG[:,4]+self.s.ABSMAG[:,6])/2.
        #print len(newi)
        self.logstellarmassTaylor[bad_imag]=1.15+0.70*(self.s.ABSMAG[:,3][bad_imag]-newi[bad_imag]) -0.4*(newi[bad_imag]+ 5.*log10(h))


        self.AGNKAUFF=self.s['AGNKAUFF']
        self.AGNKEWLEY=self.s['AGNKEWLEY']
        self.AGNSTASIN=self.s['AGNSTASIN']
        self.AGNKAUFF=self.s['AGNKAUFF'] & (self.s.HAEW > 0.)
        self.AGNKEWLEY=self.s['AGNKEWLEY']& (self.s.HAEW > 0.)
        self.AGNSTASIN=self.s['AGNSTASIN']& (self.s.HAEW > 0.)
        self.cnumerical_error_flag24=self.s['fnumerical_error_flag24']
        self.fcnumerical_error_flag24=self.s['fcnumerical_error_flag24']
        self.AGNKAUFF= ((log10(self.s.O3FLUX/self.s.HBFLUX) > (.61/(log10(self.s.N2FLUX/self.s.HAFLUX)-.05)+1.3)) | (log10(self.s.N2FLUX/self.s.HAFLUX) > 0.))
        #y=(.61/(x-.47)+1.19)
        self.AGNKEWLEY= ((log10(self.s.O3FLUX/self.s.HBFLUX) > (.61/(log10(self.s.N2FLUX/self.s.HAFLUX)-.47)+1.19)) | (log10(self.s.N2FLUX/self.s.HAFLUX) > 0.3))


        self.upperlimit=self.s['RE_UPPERLIMIT'] # converts this to proper boolean array
        self.pointsource=self.s['POINTSOURCE'] # converts this to proper boolean array
        self.gim2dflag=self.s['matchflag'] & (self.s.Rd == self.s.Rd) # get rid of nan's in Rd
        self.zooflag=self.s['match_flag']
        self.nerrorflag=self.s['fcnumerical_error_flag24']
        # convert flags to boolean arrays
        for col in cnames:
            if (col.find('flag') > -1) | (col.find('AGN') > -1):
                #print col
                self.s.field(col)[:]=np.array(self.s[col],'bool')



                                        
        self.nsadict=dict((a,b) for a,b in zip(self.s.NSAID,arange(len(self.s.NSAID))))
        self.logstellarmass =  self.s.MSTAR_50 # self.logstellarmassTaylor # or
        #self.logstellarmass =  self.logstellarmassTaylor # or
        #self.define_supersize()
        # calculating magnitudes from fluxes provided from NSA 
        # 
        # m = 22.5 - 2.5 log_10 (flux_nanomaggies)
        # from http://www.sdss3.org/dr8/algorithms/magnitudes.php#nmgy
        self.nsamag=22.5-2.5*log10(self.s.NMGY)
        self.badfits=zeros(len(self.s.RA),'bool')
        #badfits=array([166134, 166185, 103789, 104181],'i')'
        nearbystar=[142655, 143485, 99840, 80878] # bad NSA fit; 24um is ok
        #nearbygalaxy=[103927,143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        # checked after reworking galfit
        nearbygalaxy=[143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        #badNSA=[166185,142655,99644,103825,145998]
        #badNSA = [
        badfits= nearbygalaxy#+nearbystar+nearbygalaxy
        badfits=array(badfits,'i')
        for gal in badfits:
            self.badfits[where(self.s.NSAID == gal)]  = 1

        
        self.sdssspecflag=(self.s.ISDSS > -1)
        self.emissionflag=((self.s.HAFLUX != 0.) & (self.s.HAFLUX != -9999.) & (self.s.N2FLUX != 0.)) | self.sdssspecflag
        self.alfalfaflag=(self.s.IALFALFA > -1) 
        self.mipsflag=(self.s.LIR_ZDIST > 0.)
        self.mipsflag=(self.s.FLUX_RADIUS1 > 0.)
        self.wiseflag = (self.s.W1FLG_3 < 2) & (self.s.W2FLG_3 < 2) & (self.s.W3FLG_3 < 2) & (self.s.W4FLG_3 < 2)
        # this allows for source confusion and the presence of some bad pixels within the aperture. 

        self.wiseagn=(self.s.W1MAG_3 - self.s.W2MAG_3) > 0.8
        self.agnflag = self.AGNKAUFF | self.wiseagn
        #self.agnkauff=self.s.AGNKAUFF > .1
        #self.agnkewley=self.s.AGNKEWLEY > .1
        #self.agnstasin=self.s.AGNSTASIN > .1
        self.dv = (self.s.ZDIST - self.s.CLUSTER_REDSHIFT)*3.e5/self.s.CLUSTER_SIGMA
        self.dvflag = abs(self.dv) < 3.

        #self.agnflag = self.agnkauff
        #self.galfitflag=(self.s.galfitflag > .1) #| (self.s.fcmag1 < .1)
        #self.galfitflag[(self.s.fcmag1 < .1)]=zeros(sum(self.s.fcmag1<.1))
        #self.agnflag = self.s.agnflag > .1
        #self.zooflag = self.s.match_flag > .1
        # self.gim2dflag = self.s.matchflag > .1
        self.membflag = (self.s.DR_R200 < 1.) & (abs(self.dv) < 3.)
        #self.membflag = abs(self.dv) < (-1.25*self.s.DR_R200 + 1.5)
        # selection of infalling galaxies from Oman+13
        self.membflag = abs(self.dv) < (-4./3.*self.s.DR_R200 + 2)
        # sharper cut determined by eye
        #self.membflag = abs(self.dv) < (-3./1.2*self.s.DR_R200 + 3)
        #self.nearexteriorflag = (self.s.DR_R200 > 1.) & (abs(self.dv) < 3.)
        self.nearexteriorflag = ~self.membflag & (abs(self.dv) < 3.)
        self.exteriorflag =  (abs(self.dv) > 3.)
        self.groupflag = ((self.s.CLUSTER == 'MKW11') | (self.s.CLUSTER == 'MKW8') | (self.s.CLUSTER == 'AWM4') | (self.s.CLUSTER == 'NGC6107'))
        self.clusterflag = ((self.s.CLUSTER == 'A1367') | (self.s.CLUSTER == 'Coma') | (self.s.CLUSTER == 'Hercules') | (self.s.CLUSTER == 'A2052') | (self.s.CLUSTER == 'A2063'))
        #environmental zones
        self.zone1=(self.s.DR_R200 < .5) & (abs(self.dv) < 3.)
        self.zone2=(self.s.DR_R200 > .5) & (self.s.DR_R200 < 1) & (abs(self.dv) < 3.)
        self.zone3=(self.s.DR_R200 > 1)  & (abs(self.dv) < 3.)
        self.zone4= (abs(self.dv) > 3.)
        self.HIflag = self.s.HIMASS > 0.
        
        self.sumagnflag=self.s.AGNKAUFF + self.s.AGNKEWLEY  + self.s.AGNSTASIN
        self.da=zeros(len(self.s.ZDIST),'f')
        q=.2
        baflag=self.s.SERSIC_BA < q
        self.incl=arccos(sqrt((self.s.SERSIC_BA**2-q**2)/(1.-q**2)))*(~baflag)+baflag*pi/2. # in radians
        # correct for inclination
        #self.isorad.NSA=self.isorad.NSA*cos(self.incl)
        #self.isorad.MIPS=self.isorad.MIPS*cos(self.incl)

        self.mag24=23.9-2.5*log10(self.s.FLUX24)
        self.NUV24=(self.nsamag[:,2])-self.mag24
        self.mag24se=18.526-2.5*log10(self.s.SE_FLUX_AUTO)
        
        #self.gi_corr=(self.nsamag[:,3]-self.nsamag[:,5])-(.17*(1-cos(self.incl))*((self.jmass.MSTAR_50)-8.19))
        self.gi_corr=(self.nsamag[:,3]-self.nsamag[:,5])-(.17*(1-cos(self.incl))*((self.logstellarmass)-8.19))
        for i in range(len(self.s.ZDIST)):
            self.da[i] = cosmo.angular_diameter_distance(self.s.ZDIST[i]).value 



        self.da=cosmo.angular_diameter_distance(self.s.ZDIST).value*Mpcrad_kpcarcsec  # kpc/arcsec
        for c in clusternames:
            if (c == 'Coma') & (usecoma == False):
                continue
            else:
                for id in spiral_100_nozoo[c]:
                    try:
                        self.spiralflag[self.nsadict[id]]=1
                    except:
                        print 'did not find ',id



        self.dist3d=sqrt((self.dv-3.)**2 + (self.s.DR_R200)**2)

        self.sb_obs=zeros(len(self.s.RA))
        flag= (~self.s['fcnumerical_error_flag24'])
        self.sb_obs[flag]=self.s.fcmag1[flag] + 2.5*log10(pi*((self.s.fcre1[flag]*mipspixelscale)**2)*self.s.fcaxisratio1[flag])
        self.DA=zeros(len(self.s.SERSIC_TH50))
        for i in range(len(self.DA)):
            if self.membflag[i]:
                self.DA[i] = cosmo.angular_diameter_distance(self.s.CLUSTER_REDSHIFT[i]).value*Mpcrad_kpcarcsec
            else:
                self.DA[i] = cosmo.angular_diameter_distance(self.s.ZDIST[i]).value*Mpcrad_kpcarcsec
        self.sizeflag=(self.s.SERSIC_TH50*self.DA > minsize_kpc) #& (self.s.SERSIC_TH50 < 20.)


        self.SIZE_RATIO_DISK = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_DISK[self.gim2dflag] = self.s.fcre1[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.s.Rd[self.gim2dflag]
        self.SIZE_RATIO_DISK_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_DISK_ERR[self.gim2dflag] = self.s.fcre1err[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.s.Rd[self.gim2dflag]
        self.SIZE_RATIO_gim2d = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_gim2d[self.gim2dflag] = self.s.fcre1[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.s.Rhlr_2[self.gim2dflag]
        self.SIZE_RATIO_gim2d_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_gim2d_ERR[self.gim2dflag] = self.s.fcre1err[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.s.Rhlr_2[self.gim2dflag]
        self.SIZE_RATIO_NSA = self.s.fcre1*mipspixelscale/self.s.SERSIC_TH50
        self.SIZE_RATIO_NSA_ERR=self.s.fcre1err*mipspixelscale/self.s.SERSIC_TH50

        if USE_DISK_ONLY:
            self.sizeratio = self.SIZE_RATIO_DISK
            self.sizeratioERR=self.SIZE_RATIO_DISK_ERR
        else:
            #self.sizeratio = self.SIZE_RATIO_gim2d
            #self.sizeratioERR=self.SIZE_RATIO_gim2d_ERR
            self.sizeratio = self.s.fcre1*mipspixelscale/self.s.SERSIC_TH50
            self.sizeratioERR=self.s.fcre1err*mipspixelscale/self.s.SERSIC_TH50
        self.massflag=self.logstellarmass > minmass
        self.Re24_kpc = self.s.fcre1*mipspixelscale*self.DA
        self.lirflag=(self.s.LIR_ZDIST > 5.2e8)
        self.galfitflag = (self.s.fcmag1 > .1)  & ~self.nerrorflag & (self.sb_obs < 20.) & (self.s.fcre1/self.s.fcre1err > .5)#20.)
        #override the galfit flag for the following galaxies
        self.galfit_override = [70588,70696,43791,69673,146875,82170, 82182, 82188, 82198, 99058, 99660, 99675, 146636, 146638, 146659, 113092, 113095, 72623,72631,72659, 72749, 72778, 79779, 146121, 146130, 166167, 79417, 79591, 79608, 79706, 80769, 80873, 146003, 166044,166083, 89101, 89108,103613,162792,162838, 89063]
        for id in self.galfit_override:
            try:
                self.galfitflag[self.nsadict[int(id)]] = True
            except KeyError:

                if self.prefix == 'no_coma':
                    print 'ids not relevant for nc'
                else:
                    sys.exit()
        #self.galfitflag = self.galfitflag 
        self.galfitflag[self.nsadict[79378]] = False
        self.galfitflag = self.galfitflag & ~self.badfits
        self.sbflag=self.sb_obs < 20.
        self.sfsampleflag = self.sizeflag & self.massflag & self.lirflag & ~self.badfits

        self.ur=self.s.ABSMAG[:,2]-self.s.ABSMAG[:,4]
        self.redflag=(self.ur > 2.3)
        self.greenflag=(self.ur > 1.8) & (self.ur < 2.3)
        self.blueflag=(self.ur<1.8)
        self.NUVr=self.s.ABSMAG[:,1] - self.s.ABSMAG[:,4]
        self.blueflag2=self.NUVr < 4
        self.greenflag = (self.NUVr < 5) & ~self.blueflag2

        # add galaxies with blue u-r colors but no galex data
        self.blue_nogalex = (self.s.NMGY[:,1] == 0.) & (self.blueflag)
        self.blueflag2[self.blue_nogalex] = np.ones(sum(self.blue_nogalex))

        self.basesampleflag = self.galfitflag  & self.sizeflag & self.massflag & self.lirflag & ~self.badfits  & self.gim2dflag
        self.sampleflag = self.galfitflag    & self.lirflag   & self.sizeflag & ~self.agnflag & self.sbflag#& self.massflag#& self.gim2dflag#& self.blueflag2
        self.allbutgalfitflag =  ~self.galfitflag & self.lirflag    & self.sizeflag & ~self.agnflag  #& self.massflag#& self.gim2dflag#& self.blueflag2
        if USE_DISK_ONLY:
            self.sampleflag = self.sampleflag & self.gim2dflag
        self.greensampleflag = self.galfitflag  & self.sizeflag & self.massflag & self.lirflag & ~self.badfits & self.greenflag & self.gim2dflag
        self.bluesampleflag = self.sampleflag & self.blueflag2
        self.unknownagn= self.sizeflag & self.massflag & self.lirflag & ~self.emissionflag & ~self.wiseflag
        self.virialflag = self.dv < (1.5-1.25*self.s.DR_R200)
        self.limitedsample=self.sampleflag & (self.logstellarmass > 9.5) & (self.logstellarmass < 10.2) & self.gim2dflag & (self.s.B_T_r < 0.2) & self.dvflag
        self.c90=self.s.FLUX_RADIUS2/self.s.fcre1
        self.size_ratio_corr=self.sizeratio*(self.s.faxisratio1/self.s.SERSIC_BA)
        self.truncflag=(self.sizeratio < 0.7) & self.sampleflag  & ~self.agnflag
        self.dL = self.s.ZDIST*3.e5/H0
        self.distmod=25.+5.*log10(self.dL)
        #best_distance=self.membflag * self.cdMpc + ~self.membflag*(self.n.ZDIST*3.e5/H0)
        self.LIR_BEST = self.s.LIR_ZCLUST * self.membflag + ~self.membflag*(self.s.LIR_ZDIST)
        self.SFR_BEST = self.s.SFR_ZCLUST * np.array(self.dvflag,'i') + np.array(~self.dvflag,'i')*(self.s.SFR_ZDIST)

        self.ssfr=self.SFR_BEST/(10.**(self.logstellarmass))
        self.ssfrerr=self.SFR_BEST/(10.**(self.logstellarmass))*(self.s.FLUX24ERR/self.s.FLUX24)
        self.ssfrms=np.log10(self.ssfr*1.e9/.08)
        self.sigma_ir=np.zeros(len(self.LIR_BEST),'d')
        self.sigma_irerr=np.zeros(len(self.LIR_BEST),'d')
        self.sigma_irerr[self.galfitflag]= np.sqrt(((self.LIR_BEST[self.galfitflag]*self.s.FLUX24ERR[self.galfitflag]/self.s.FLUX24ERR[self.galfitflag])/2/(np.pi*(self.s.fcre1[self.galfitflag]*self.DA[self.galfitflag])**2))**2+(2.*self.LIR_BEST[self.galfitflag]/2/(np.pi*(self.s.fcre1[self.galfitflag]*self.DA[self.galfitflag])**3)*self.s.fcre1err[self.galfitflag])**2)
        self.sigma_ir[self.galfitflag]= self.LIR_BEST[self.galfitflag]/2/(np.pi*(self.s.fcre1[self.galfitflag]*self.DA[self.galfitflag])**2)
        self.starburst = (self.ssfr*1.e9 > .16)
        self.compact_starburst = (self.ssfr*1.e9 > .16) & (self.sigma_ir > 5.e9) 


        self.agcdict=dict((a,b) for a,b in zip(self.s.AGCNUMBER,arange(len(self.s.AGCNUMBER))))
        self.nsadict=dict((a,b) for a,b in zip(self.s.NSAID,arange(len(self.s.NSAID))))
        self.massdensity=self.logstellarmass-log10(2*pi*(self.s.SERSIC_TH50*self.DA)**2)


    def plotSFRStellarmassSizeBlue(self,clustername=None,blueflag=True,spiralflag=False,plotbadfits=True,nocolor=False):
        minsize=.4
        maxsize=1.5
        figure(figsize=(10,8))
        baseflag = np.ones(len(self.sampleflag),'bool')
        if blueflag:
            baseflag = baseflag & self.blueflag2
        if spiralflag:
            baseflag = baseflag & self.spiralflag
        subplots_adjust(left=.12,bottom=.15,wspace=.02,hspace=.02)
        x_flags=[baseflag &  ~self.sampleflag & ~self.agnflag & self.membflag,
                 baseflag &  ~self.sampleflag & self.agnflag & self.membflag,
                 baseflag &  ~self.sampleflag & ~self.agnflag & ~self.membflag ,
                baseflag &  ~self.sampleflag & self.agnflag & ~self.membflag]
                 
        point_flags=[baseflag & self.sampleflag & ~self.agnflag & self.membflag,
                     baseflag & self.sampleflag & self.agnflag & self.membflag,
                     baseflag & self.sampleflag & ~self.agnflag & ~self.membflag,
                     baseflag & self.sampleflag & self.agnflag & ~self.membflag]

        bothax=[]
        y=self.SFR_BEST*1.58 # convert from salpeter to chabrier IMF according to Salim+07
        for i in range(4):
            plt.subplot(2,2,i+1)
            if clustername != None:
                x_flags[i] = x_flags[i] & (self.s.CLUSTER == clustername)
                point_flags[i] = point_flags[i] & (self.s.CLUSTER == clustername)
            if (i == 0) | (i==2):
                if plotbadfits:
                    plt.plot(self.logstellarmass[x_flags[i]],y[x_flags[i]],'kx',markersize=8,label='No Fit')
                if nocolor:
                    sp=plt.scatter(self.logstellarmass[point_flags[i]],y[point_flags[i]],c='k',vmin=0.1,vmax=1,cmap='jet_r',s=60,label='GALFIT')
                else:
                    sp=plt.scatter(self.logstellarmass[point_flags[i]],y[point_flags[i]],c=self.sizeratio[point_flags[i]],vmin=minsize,vmax=maxsize,cmap='jet_r',s=60,label='GALFIT')
            if (i == 1) | (i == 3):
                xbin,ybin,ybinerr=my.binitbins(9.4,11.,(11.-9.4)/.2,self.logstellarmass[point_flags[i-1]],y[point_flags[i-1] ])
                xbin,sbin,sbinerr=my.binitbins(9.4,11.,(11-9.4)/.2,self.logstellarmass[point_flags[i-1]],self.sizeratio[point_flags[i-1]])
                #xbin,ybin,ybinerr=my.binit(self.logstellarmass[point_flags[i]],self.SFR_BEST[point_flags[i] ],5)
                #xbin,sbin,sbinerr=my.binit(self.logstellarmass[point_flags[i]],self.sizeratio[point_flags[i]],5)
                errorbar(xbin,ybin,yerr=ybinerr,fmt=None,color='k',markersize=16,ecolor='k')
                if nocolor:
                    plt.scatter(xbin,ybin,c='k',s=300,cmap='jet_r',vmin=minsize,vmax=maxsize,marker='s')
                else:
                    plt.scatter(xbin,ybin,c=sbin,s=300,cmap='jet_r',vmin=minsize,vmax=maxsize,marker='s')
            plt.axis([9.1,11.75,7.e-2,32])
            self.plotelbaz()
            gca().set_yscale('log')
            a=plt.gca()
            bothax.append(a)
            plt.axvline(x=minmass,c='k',ls='--')
            plt.axhline(y=.086,c='k',ls='--')
            #if i > 2:
            #    xlabel('$log_{10}(M_* (M_\odot)) $',fontsize=22)
            if i == 0:
                a.set_xticklabels(([]))
                plt.text(0.1,0.9,'$Core$',transform=a.transAxes,horizontalalignment='left',fontsize=20)
                plt.title('$ SF \ Galaxies $',fontsize=22)
            if i == 1:
                a.set_xticklabels(([]))
                a.set_yticklabels(([]))
                plt.legend(loc='upper left',numpoints=1,scatterpoints=1)
                plt.title('$Median $',fontsize=22)
            if i == 2:
                plt.text(0.1,0.9,'$Exterior$',transform=a.transAxes,horizontalalignment='left',fontsize=20)
                plt.text(-0.2,1.,'$SFR \ (M_\odot/yr)$',transform=a.transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=24)

            if i == 3:
                a.set_yticklabels(([]))
                text(-0.02,-.2,'$log_{10}(M_*/M_\odot)$',transform=a.transAxes,horizontalalignment='center',fontsize=24)
            i += 1
        if not(nocolor):
            c=plt.colorbar(ax=bothax,fraction=.05,ticks=arange(minsize,maxsize,.1),format='%.1f')
            c.ax.text(2.2,.5,'$R_e(24)/R_e(r)$',rotation=-90,verticalalignment='center',fontsize=20)

        if nocolor:
            plt.savefig(homedir+'research/LocalClusters/SamplePlots/SFRStellarmassSizeBlue-bw.png')
            plt.savefig(homedir+'research/LocalClusters/SamplePlots/SFRStellarmassSizeBlue-bw.eps')
        else:
            plt.savefig(homedir+'research/LocalClusters/SamplePlots/SFRStellarmassSizeBlue.png')
            plt.savefig(homedir+'research/LocalClusters/SamplePlots/SFRStellarmassSizeBlue.eps')
    def plotelbaz(self):
        xe=np.arange(8.5,11.5,.1)
        xe=10.**xe
        ye=(.08e-9)*xe
        plt.plot(log10(xe),(ye),'k-',lw=1,label='$Elbaz+2011$')
        plt.plot(log10(xe),(2*ye),'k:',lw=1,label='$2 \ SFR_{MS}$')

if __name__ == '__main__':

    infile=homedir+'research/LocalClusters/NSAmastertables/LCS_all_size.fits'
    s=spirals(infile,prefix='all')
    nc=spirals(infile,usecoma=False,prefix='no_coma')

    def plotSFRStellarmass(self):
        plt.figure()
