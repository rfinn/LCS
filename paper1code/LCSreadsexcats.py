#!/usr/bin/env python
from pylab import *
import pyfits

def readcat(infile):
    hdulist=pyfits.open(infile)
    tbdata=hdulist[1].data

    
