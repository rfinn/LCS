#!/usr/bin/env python


'''
GOAL
----
* write out table for publication
* write out latex version of table

INPUT
-----
* LCS_all_size_KE_SFR.fits

OUTPUT
------
* table with size, HIdef, B/T for Luca Cortese

* table with size, HIdef, SFR, Mstar, GSWLC SFR, GSWLC Mstar, B/T, sample, membflag



'''

import os
import numpy as np
from astropy.table import Table

homedir = os.getenv("HOME")
intable = homedir+'/research/LCS/tables/LCS_all_size_KE_SFR.fits'
intab = Table.read(intable)

### For Cortese
tab1 = intab['sizeratio','sizeratio_err','HIDef','HIflag','B_T_r','membflag','sampleflag']
tab1 = tab1[tab1['sampleflag'] & tab1['HIflag'] & (tab1['HIDef'] != np.inf)]
tab1.write(homedir+'/research/LCS/tables/HIDef.fits',format='fits',overwrite=True)


### For publication


### latex version for publication
