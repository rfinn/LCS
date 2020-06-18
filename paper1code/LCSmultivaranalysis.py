#!/usr/bin/env python
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import pandas

infile=homedir+'research/LocalClusters/NSAmastertables/LCS_all_size.fits'
s=fits.getdata(infile)

