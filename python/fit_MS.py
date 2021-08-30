#!/usr/bin/env python


'''
GOAL:
* fit MS for full and BT<0.4 samples using
  * curve_fit
  * bootstrap 
  * sigma clipping

* determine errors in fitted parameters using 


'''
import sys
import os
from astropy.table import Table
from astropy.stats import bootstrap
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile
import numpy as np
homedir = os.getenv("HOME")
tabledir = homedir+'/research/LCS/tables/'

def linear_func(x,m,b):
    return m*x+b

def bootstrap_curvefit(self,x,y,N=100):
    indices = np.arange(len(ftab))

    # get resampled indices
    boot_indices = bootstrap(indices,N)
    bslope = np.zeros(N,'d')
    binter = np.zeros(N,'d')        
    for i,myindices in enumerate(boot_indices):
        myindices = np.array(myindices,'i')
        popt,pcov = curve_fit(linear_func,x[myindices],y[myindices])
        bslope[i] = popt[0]
        binter[i] = popt[1]
    bslope_lower = scoreatpercentile(bslope,16)
    bslope_upper = scoreatpercentile(bslope,84)        
    binter_lower = scoreatpercentile(binter,16)
    binter_upper = scoreatpercentile(binter,84)
    bslope_med = np.median(bslope)
    binter_med = np.median(binter)
        
    print('median slope = {:.2f}+{:.2f}-{:.2f}'.format(bslope_med,\
                                                       bslope_med - bslope_lower,\
                                                       bslope_upper - bslope_med))
    print('median inter = {:.2f}+{:.2f}-{:.2f}'.format(binter_med,\
                                                       binter_med - binter_lower,\
                                                       binter_upper - binter_med))
    return bslope_med,bslope_med-bslope_lower,bslope_upper - bslope_med,\
        binter_med,binter_med-binter_lower,binter_upper - binter_med,\
        

class fitMS():
    def __init__(self):
        
        pass

    def read_tables(self):
        # read

        self.field = Table.read(tabledir+'/LCS-sfr-mstar-field.fits')        
        self.fieldBT = Table.read(tabledir+'/LCS-sfr-mstar-field-BTcut.fits')

    def fit_MS_curvefit(self,ftab):
        # fit MS for full sample
        popt,pcov = curve_fit(linear_func,ftab['logmstar'],ftab['logsfr'])
        perr = np.sqrt(np.diag(pcov))
        print('Best-fit slope = {:.2f}+/-{:.2f}'.format(popt[0],perr[0]))
        print('Best-fit inter = {:.2f}+/-{:.2f}'.format(popt[1],perr[1]))        
        # fit MS for
    def fit_MS_bootstrap(self,ftab):
        t = bootstrap_curvefit(ftab['logmstar'],ftab['logsfr'])


                                                          
