#!/usr/bin/env python

import glob
import os

infiles=glob.glob('*_R.fits')


for file in infiles:#infiles:
    agcnumber=file.split('_R.fits')[0]
    print agcnumber
    os.system('uat_mask.py '+str(agcnumber))
    os.system('LCSrunellipseHa.py '+str(agcnumber))

