#!/usr/bin/env python

'''
    Written by Rose Finn, updated on July 2, 2013

    PURPOSE: 
      This program calculates the total IR luminosity and SFR for the LCS sample
      using the Chary & Elbaz templates.  The program calculates three sets of 
      (LIR,SFR) using distances derived from the galaxy redshift, ZDIST and 
      the cluster redshift.  You should use the values for the cluster redshift
      for cluster members.

      The program also uses the completeness estimates that I made 

      /Users/rfinn/research/LocalClusters/Completeness

      to assign a weight to each galaxy.  
      
    CALLING SEQUENCE

      from the terminal command line, type:
        LCSgetcharySFRNSA.py

    INPUT PARAMETERS
      none
      
    OUTPUT PARAMETERS
      Creates a table
      
      outfile= homedir+'research/LocalClusters/NSAmastertables/CharyElbazTables/'+clustername+'_ce_lir.fits'

      that contains a line-matched table of the three sets of (LIR,SFR) and the weight

    EXAMPLES

    PROCEDURE

      Completeness and weight are assigned to each galaxy by interpolating 
      the completeness vs flux data.

    REQUIRED PYTHON MODULES
      numpy
      pylab
      os
      atpy
      scipy

    ADDITIONAL REQUIRED MODULES
      LCScommon
      mystuff
      chary_elbaz_24um

    NOTES
      Required Files

        homedir+'research/LocalClusters/Completeness/clustername-FracComplvsFlux.dat'
        homedir+'research/LocalClusters/NSAmastertables/Sex24Tables/'+clustername+'_sex24.fits'
        homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSAmastertable.fits'

'''


from pylab import *
import os
import numpy as np
from scipy.interpolate import interp1d
#import pyfits
import atpy

from LCScommon import *
import mystuff as my
#from LCSReadmasterBaseWithProfileFits import *
import chary_elbaz_24um as chary

class cluster:
    def __init__(self,clustername):
        self.cz=clusterz[clustername]
        infile=homedir+'research/LocalClusters/NSAmastertables/Sex24Tables/'+clustername+'_sex24.fits'
        self.sex24=atpy.Table(infile)
        infile=homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSAmastertable.fits'
        self.n=atpy.Table(infile)
        self.ceLIR=zeros(len(self.sex24.MATCHFLAG24),'d')
        self.ceSFR=zeros(len(self.sex24.MATCHFLAG24),'d')
        self.ceLIR_ZDIST=zeros(len(self.sex24.MATCHFLAG24),'d')
        self.ceSFR_ZDIST=zeros(len(self.sex24.MATCHFLAG24),'d')
        self.ceLIR_ZCLUST=zeros(len(self.sex24.MATCHFLAG24),'d')
        self.ceSFR_ZCLUST=zeros(len(self.sex24.MATCHFLAG24),'d')
        redshift=self.cz*ones(len(self.sex24.MATCHFLAG24))  # convert recession velocity to redshift




        # CALCULATE LIR AND SFR USING REDSHIFT
        #print self.sex24.FLUX_AUTO[self.sex24.MATCHFLAG24]
        # when I changed to using FLUX_AUTO, some entries have nan values
        # even though FLUX_BEST exists for these objects
        # going to replace array values with FLUX_BEST
        self.nanflag= (self.sex24.FLUX_AUTO != self.sex24.FLUX_AUTO)
        replaceindex=where(self.sex24.FLUX_AUTO != self.sex24.FLUX_AUTO)
        self.flux24 =  self.sex24.FLUX_AUTO
        for i in replaceindex:
            self.flux24[i]=self.sex24.FLUX_BEST[i]
        self.ceLIR[self.sex24.MATCHFLAG24],self.ceSFR[self.sex24.MATCHFLAG24]=chary.chary_elbaz_24um(self.n.Z[self.sex24.MATCHFLAG24],self.flux24[self.sex24.MATCHFLAG24]*mipsconv_MJysr_to_uJy)

        # CALCULATE LIR AND SFR ASSUMING THE GALAXY IS AT ZDIST
        self.ceLIR_ZDIST[self.sex24.MATCHFLAG24],self.ceSFR_ZDIST[self.sex24.MATCHFLAG24]=chary.chary_elbaz_24um(self.n.ZDIST[self.sex24.MATCHFLAG24],self.flux24[self.sex24.MATCHFLAG24]*mipsconv_MJysr_to_uJy)
        # CALCULATE LIR AND SFR ASSUMING THE GALAXY IS AT THE CLUSTER REDSHIFT
        self.ceLIR_ZCLUST[self.sex24.MATCHFLAG24],self.ceSFR_ZCLUST[self.sex24.MATCHFLAG24]=chary.chary_elbaz_24um(redshift[self.sex24.MATCHFLAG24],self.flux24[self.sex24.MATCHFLAG24]*mipsconv_MJysr_to_uJy)

        # READ IN COMPLETENESS VS FLUX DATA

        infile=homedir+'research/LocalClusters/Completeness/'+clustername+'-FracComplvsFlux.dat'
        cdat=np.loadtxt(infile)
        pflux=cdat[:,0] # flux in MJy/sr
        pcomp=cdat[:,1]
        # pre-pend (0,0) to list and remove trailing zeros
        flux=np.zeros(len(pflux),'d')
        comp=np.zeros(len(pcomp),'d')
        flux[1:]=pflux[0:-1]
        comp[1:]=pcomp[0:-1]
        #for i in range(len(flux)): print i, flux[i], comp[i]
        # older approach that didn't add a zero to the beginning of the dataset
        #comp[0:-1] # last row in file is all zeros, so getting rid of that
        #flux=flux[0:-1] # last row in file is all zeros, so getting rid of that

        # INTERPOLATE DATA
        interpfunc=interp1d(flux,comp,bounds_error=False,fill_value=1.0) # save for later use w/in loops
        # IF FLUX IS BEYOND MAX OF INTERPOLATION DATA, SET COMPLETENESS TO 1

        # GET INTERPOLATED COMPLETENESS FOR EACH FLUX
        self.completeness=zeros(len(self.sex24.MATCHFLAG24),'d')       
        self.weight=zeros(len(self.sex24.MATCHFLAG24),'d')
        self.completeness[self.sex24.MATCHFLAG24]=interpfunc(self.sex24.FLUX_AUTO[self.sex24.MATCHFLAG24])

        # I had originally set completeness for low flux values to zero
        # but this does not make sense.
        # It's better to make a min flux cut downstream
        #flag=self.sex24.FLUX_BEST < .51
        #self.completeness[flag]=zeros(len(flag),'d')
        flag=self.sex24.FLUX_BEST > 7.
        self.completeness[flag]=ones(sum(flag),'d')

        self.weight[self.sex24.MATCHFLAG24]=1./self.completeness[self.sex24.MATCHFLAG24]

        ###  WRITE OUT TABLE CONTAINING LIR AND SFRS  ###
        output=atpy.Table()
        output.add_column('NSAID',self.sex24.NSAID)
        output.add_column('MATCHFLAG24',self.sex24.MATCHFLAG24)
        output.add_column('SE_FLUX_AUTO',self.sex24.FLUX_AUTO,unit='MJy/sr')
        output.add_column('FLUX24',self.sex24.FLUX_AUTO*mipsconv_MJysr_to_uJy,unit='micro-Jy')
        output.add_column('FLUX24ERR',self.sex24.FLUXERR_AUTO*mipsconv_MJysr_to_uJy,unit='micro-Jy')
        output.add_column('LIR',(self.ceLIR),unit='Lsun')
        output.add_column('SFR',self.ceSFR,unit='Msun/yr')
        output.add_column('LIR_ZDIST',(self.ceLIR_ZDIST),unit='Lsun')
        output.add_column('SFR_ZDIST',self.ceSFR_ZDIST,unit='Msun/yr')

        output.add_column('LIR_ZCLUST',(self.ceLIR_ZCLUST),unit='Lsun')
        output.add_column('SFR_ZCLUST',self.ceSFR_ZCLUST,unit='Msun/yr')
        #output.add_column('MIPS_COMPLETENESS',self.completeness,unit='')
        output.add_column('MIPS_WEIGHT',self.completeness,unit='')


        outfile= homedir+'research/LocalClusters/NSAmastertables/CharyElbazTables/'+clustername+'_ce_lir.fits'
        if os.path.exists(outfile):
            os.remove(outfile)
        output.write(outfile)
clusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
clusternames=[ 'Hercules','A2063','NGC6107','Coma']
#cl=cluster('Coma')
for cname in clusternames:
    cl=cluster(cname)

