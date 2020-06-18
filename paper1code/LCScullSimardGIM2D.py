#!/usr/bin/env python
'''
written 11/09/2013 by Rose A. Finn

goal is to take tables 1-3 from Simard+2011

keep only galaxies with RA= [170, 250] and Dec=[0, 38], and z < .06

then I can use these sub-catalogs to match more efficiently to NSA catalogs

NOTE:  this did not work b/c the tables are not row-matched.  I ended up matching the rows
using topcat.  I matched table1_LCS.fits with table3.fits and created table1and3.fits

so I guess this program was partially useful b/c I was able to use table1_LCS.fits as the
reference table for the matching, and that saves quite a bit of time over using
table1.fits

Table 1 = n=4 buldge + disk

Table 3 = pure sersic model

'''

from LCScommon import *

import atpy

catdir=homedir+'research/SimardSDSS2011/'
#input_tables=['table1.fits','table2.fits','table3.fits']
#output_tables=['table1_LCS.fits','table2_LCS.fits','table3_LCS.fits']

input_tables=['table1.fits','table3.fits']
output_tables=['table1_LCS.fits','table3_LCS.fits']

# read in table 1
dat=atpy.Table(catdir+input_tables[0])
# create flag
keepflag = (dat._RA> lcsramin) & (dat._RA < lcsramax) & (dat._DE > lcsdecmin) & (dat._DE < lcsdecmax) & (dat.z < .06)
# write out table1_LCS.fits
subdat=dat.where(keepflag)
outfile=catdir+output_tables[0]
if os.path.exists(outfile):
    os.remove(outfile)
subdat.write(outfile)

# read in table 2
dat=atpy.Table(catdir+input_tables[1])
# write out table2_LCS.fits
subdat=dat.where(keepflag)
outfile=catdir+output_tables[1]
if os.path.exists(outfile):
    os.remove(outfile)
subdat.write(outfile)


## read in table 3
#dat=atpy.Table(catdir+input_tables[2])
## write out table3_LCS.fits
#subdat=dat.where(keepflag)
#outfile=catdir+output_tables[2]
#if os.path.exists(outfile):
#    os.remove(outfile)
#subdat.write(outfile)
