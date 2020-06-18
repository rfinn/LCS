#!/usr/bin/env python


# read in LCS
lcspath = '/Users/rfinn/research/NSAmastertables/'
class galaxies:
    def __init__(self, infile):

        self.jmass=fits.getdata(lcspath+'LCS_Spirals_all_fsps_v2.4_miles_chab_charlot_sfhgrid01.fits')
        # use jmass.mstar_50 and jmass.mstar_err

        self.agc=fits.getdata(homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_AGC.fits')

        self.s=fits.getdata(infile)

        self.gim2d=fits.getdata(homedir+'research/LocalClusters/NSAmastertables/LCS_all.gim2d.tab1.fits')



if __name__ == '__main__':

    infile=homedir+'research/LocalClusters/NSAmastertables/LCS_all_size.fits'
    g = galaxies(infile)
