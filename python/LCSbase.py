#!/usr/bin/env python

'''

GOAL:
- read catalog 


USAGE:
- in ipython


    
SIZE RATIO
- I set up self.size_ratio to be the size used in all plots.

REQUIRED MODULES:
- LCScommon

**************************
written by Rose A. Finn
May 2018
**************************

'''

from astropy.io import fits
import os
import sys
from astropy.cosmology import WMAP9 as cosmo
import numpy as np

# set environment variables to point to github repository
# call it GITHUB_PATH
# LCS_TABLE_PATH

# python magic to get environment variable
homedir = os.environ['HOME']

# assumes github folder exists off home directory
# and that LCS is a repository in github
lcspath = homedir+'/github/LCS/'
if not(os.path.exists(lcspath)):
    print 'could not find directory: ',lcspath
    sys.exit()

Mpcrad_kpcarcsec = 2. * np.pi/360./3600.*1000.
minmass=9.

class galaxies:
    def __init__(self, lcspath):

        self.jmass=fits.getdata(lcspath+'tables/LCS_Spirals_all_fsps_v2.4_miles_chab_charlot_sfhgrid01.fits')
        # use jmass.mstar_50 and jmass.mstar_err

        self.agc=fits.getdata(lcspath+'tables/LCS_Spirals_AGC.fits')

        self.s=fits.getdata(lcspath+'tables/LCS_all_size.fits')

        self.gim2d=fits.getdata(lcspath+'tables/LCS_all.gim2d.tab1.fits')
        
        # dictionary to look up galaxies by NSAID
        self.nsadict=dict((a,b) for a,b in zip(self.s.NSAID,np.arange(len(self.s.NSAID))))
        
    def get_agn(self):
        self.AGNKAUFF=self.s['AGNKAUFF'] & (self.s.HAEW > 0.)
        self.AGNKEWLEY=self.s['AGNKEWLEY']& (self.s.HAEW > 0.)
        self.AGNSTASIN=self.s['AGNSTASIN']& (self.s.HAEW > 0.)
        # add calculations for selecting the sample
        self.wiseagn=(self.s.W1MAG_3 - self.s.W2MAG_3) > 0.8
        self.agnflag = self.AGNKAUFF | self.wiseagn

    def get_galfit_flag(self):
        self.nerrorflag=self.s['fcnumerical_error_flag24']
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

    def get_size_flag(self):
        self.DA=zeros(len(self.s.SERSIC_TH50))
        self.DA[self.membflag] = cosmo.angular_diameter_distance(self.s.CLUSTER_REDSHIFT[self.membflag].value*Mpcrad_kpcarcsec)
        for i in range(len(self.DA)):
            if self.membflag[i]:
                self.DA[i] = cosmo.angular_diameter_distance(self.s.CLUSTER_REDSHIFT[i]).value*Mpcrad_kpcarcsec
            else:
                self.DA[i] = cosmo.angular_diameter_distance(self.s.ZDIST[i]).value*Mpcrad_kpcarcsec
        self.sizeflag=(self.s.SERSIC_TH50*self.DA > minsize_kpc) #& (self.s.SERSIC_TH50 < 20.)

    def select_sample(self):
        self.logstellarmass =  self.s.MSTAR_50 # self.logstellarmassTaylor # or
        self.massflag=self.s.MSTAR_50 > minmass
        self.Re24_kpc = self.s.fcre1*mipspixelscale*self.DA
        self.lirflag=(self.s.LIR_ZDIST > 5.2e8)

 
        self.sbflag=self.sb_obs < 20.

        self.sb_obs=zeros(len(self.s.RA))
        flag= (~self.s['fcnumerical_error_flag24'])
        self.sb_obs[flag]=self.s.fcmag1[flag] + 2.5*log10(np.pi*((self.s.fcre1[flag]*mipspixelscale)**2)*self.s.fcaxisratio1[flag])

        
        self.gim2dflag=self.s['matchflag'] & (self.gim2d.Rd == self.gim2d.Rd) # get rid of nan's in Rd

        #self.agnkauff=self.s.AGNKAUFF > .1
        #self.agnkewley=self.s.AGNKEWLEY > .1
        #self.agnstasin=self.s.AGNSTASIN > .1
        self.dv = (self.s.ZDIST - self.s.CLUSTER_REDSHIFT)*3.e5/self.s.CLUSTER_SIGMA
        self.dvflag = abs(self.dv) < 3.

        self.sampleflag = self.galfitflag    & self.lirflag   & self.sizeflag & ~self.agnflag & self.sbflag#& self.massflag#& self.gim2dflag#& self.blueflag2

    def calculate_sizeratio(self):
        self.SIZE_RATIO_DISK = np.zeros(len(self.gim2dflag))
        print len(self.s.fcre1),len(self.gim2dflag)
        a =  self.s.fcre1[self.gim2dflag]*mipspixelscale
        b = self.DA[self.gim2dflag]
        c = self.gim2d.Rd[self.gim2dflag]
        self.SIZE_RATIO_DISK[self.gim2dflag] =a*b/c
        
        self.SIZE_RATIO_DISK_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_DISK_ERR[self.gim2dflag] = self.s.fcre1err[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.gim2d.Rd[self.gim2dflag]
        self.SIZE_RATIO_gim2d = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_gim2d[self.gim2dflag] = self.s.fcre1[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.gim2d.Rhlr[self.gim2dflag]
        self.SIZE_RATIO_gim2d_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_gim2d_ERR[self.gim2dflag] = self.s.fcre1err[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.gim2d.Rhlr[self.gim2dflag]
        self.SIZE_RATIO_NSA = self.s.fcre1*mipspixelscale/self.s.SERSIC_TH50
        self.SIZE_RATIO_NSA_ERR=self.s.fcre1err*mipspixelscale/self.s.SERSIC_TH50

if __name__ == '__main__':


    g = galaxies(lcspath)
