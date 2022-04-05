#!/usr/bin/env python

'''
GOAL:
* get full LCS catalogs, not just area covered by MIPS scans
2

PROCEDURE
* read in parent file (GSWLC+Simard+A100)
* for each cluster (defined in LCScommon),
  * define flag for delta r < 4 R200, delta v < 4 sigma

* gather all flags
* cut parent catalog based on flag
* save parent catalog as LCS table



'''


import os
import numpy as np
from astropy.table import Table, Column
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.patches import Circle

import LCScommon as lcs

# store home directory as variable
homedir = os.getenv("HOME")

# define path to parent catalog
parent_catalog = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-tab3-vizier-A100-HIdef-28Aug2021.fits'

# read in parent catalog
pcat = Table.read(parent_catalog)

class cluster:
    def __init__(self,cname):
        self.ra = lcs.clusterRA[cname]
        self.dec = lcs.clusterDec[cname]
        self.vr = lcs.clusterbiweightcenter[cname]
        self.sigma = lcs.clusterbiweightscale[cname]    
        self.z = lcs.clusterz[cname]
        # from Finn+2008
        # R200 in Mpc
        self.r200_Mpc=2.02*(self.sigma)/1000./np.sqrt(lcs.OmegaL+lcs.OmegaM*(1.+self.z)**3)*lcs.H0/70. # in Mpc
        # get R200 in degrees
        self.r200_deg=self.r200_Mpc*1000./(cosmo.angular_diameter_distance(self.vr/3.e5).value*lcs.Mpcrad_kpcarcsec)/3600.

    def calc_keepflag(self,gcat):
        ''' measure projected distance of galaxies from cluster '''
        self.g_dr = np.sqrt((gcat['RA_1']-self.ra)**2+(gcat['DEC_1']-self.dec)**2)        
        g_vr = gcat['Z_1']*3.e5
        # save normalized quantities
        self.g_dv_sigma = np.abs(g_vr - self.vr)/self.sigma
        self.g_dr_r200 = self.g_dr/self.r200_deg
        self.keepflag = (self.g_dv_sigma < 3.) & (self.g_dr_r200 < 3.)


if __name__ == '__main__':

    dv_sigma = np.zeros(len(pcat),'f')
    dr_r200 = np.zeros(len(pcat),'f')
    keepflag = np.zeros(len(pcat),'bool')
    mycluster = np.ones(len(pcat),'S12')
    ntot=0
    for cname in lcs.clusternames:

        c = cluster(cname)
        c.calc_keepflag(pcat)
        keepflag[c.keepflag] = np.ones(np.sum(c.keepflag),'bool')
        # add column of velocity offset/sigma        
        # add column of DR/R200        
        dv_sigma[c.keepflag] = c.g_dv_sigma[c.keepflag]
        dr_r200[c.keepflag] = c.g_dr_r200[c.keepflag]
        for i in np.arange(len(pcat))[c.keepflag]:
            mycluster[i] = cname
        print('number of matches in {} = {}'.format(cname,np.sum(c.keepflag)))
        #break
        ntot+= np.sum(c.keepflag)
    # write output table, containing only galaxies within vicinity of LCS
    print()
    print('total sample size = {}'.format(ntot))
    print('total sample size = {}'.format(np.sum(keepflag)))
    outtab = Table(pcat[keepflag])
    c1 = Column(dv_sigma[keepflag],name='DV_SIGMA')
    c2 = Column(dr_r200[keepflag],name='DR_R200')
    c3 = Column(mycluster[keepflag],name='CLUSTER')

    outtab.add_columns([c1,c2,c3])
    outtab.write('LCS-GSWLC-NODR10AGN-Simard-tab1-tab3-A100.fits',format='fits',overwrite=True)
