#!/usr/bin/env python

'''
GOAL:
* create a set of galaxies with a range of stellar mass and sSFR relative to MS
* evolve galaxies from some epoch tmax to the present
* evolve SFRs according to MS evolution given in Whitaker+2012
* evolve stellar mass based on SFR evolution and mass loss as describe in Poggianti+2013

INPUT:
* tmax
* number of time steps
* number of mass bins
* number of sSFR bins


OUTPUT: 
* table containing
  |----------|-------|-----|--------|-------|
  |lookbackt | Mstar | SFR |Mstar_0 | SFR_0 |
  |----------|-------|-----|--------|-------|
* this allows someone to link a galaxy with a particular stellar mass and SFR to its progenitor at t = lookbackt




'''

#######################################
### IMPORTS
#######################################

import numpy as np
from astropy.table import Table
from astropy.cosmology import z_at_value
import astropy.units as u
from astropy.cosmology import WMAP9

cosmo = WMAP9

#######################################
### FUNCTIONS
#######################################

def get_fraction_mass_retained(t):
    ''' 
    use Poggianti+2013 relation to determine fraction of mass retained  

    INPUT:
    * time in yr since the birth of the stellar pop

    RETURNS:
    * fraction of the mass that is retained after mass loss

    '''

    return 1.749 - 0.124*np.log10(t)
    pass


def SFR_MS(logMstar,z):
    ''' 
    get whitaker+2012 logSFR of main sequence galaxy of mass logMstar at redshift z   

    INPUT:
    * logMstar : log10 of stellar mass (in Msun)
    * z : redshift to calculate SFR for

    RETURNS:
    * logSFR : log10 of SFR in Msun/yr

    '''

    alpha = 0.70-0.13*z
    beta = 0.38 + 1.14*z - 0.19*z**2
    logSFR = alpha*(logMstar - 10.5) + beta
    return logSFR

def get_z_from_lookbackt(t):
    ''' 
    get redshift for a given lookback time, using astropy.cosmology 

    INPUT:
    * lookback time in Gyr
    '''

    try:
        z = np.zeros(len(t),'f')
        for i in range(len(t)):
            if t[i] == 0:
                z[i] = 0

            else:
                z[i] = z_at_value(cosmo.lookback_time,t[i]*u.Gyr)
            #print(t[i],z[i])
    except TypeError:
        z = z_at_value(cosmo.lookback_time,t*u.Gyr)
    
    return z

def evolve_galaxy(logmstar_tmax,delta_SFR,tmax,ntimestep,tmin=0):
    '''  
    GOAL: Evolve one galaxy from tmax to t=tmin 

    INPUT:
    * logmstar : log10 of stellar mass in units of Msun
    * delta_SFR : log10 of ratio of SFR relative to SFR of main sequence for logmstar
    * tmax : time at which to start evolution in Gyr
    * tstep : time step in Gyr
    * tmin : end point of evolution; default is 0 or present day


    RETURNS:
    * logmstar_tmin : log10 of stellar 

    '''
    # convert
    timearray = np.linspace(tmax,tmin,ntimestep)
    redshiftarray = get_z_from_lookbackt(timearray)
    mstar = np.zeros(len(timearray),'f')
    sfr = np.zeros(len(timearray),'f')
    # convert tmax to redshift
    zmax = get_z_from_lookbackt(tmax)
    logsfr_tmax = SFR_MS(logmstar_tmax,zmax) + delta_SFR
    #print('hey')
    #print(SFR_MS(logmstar_tmax,zmax),logmstar_tmax,zmax, delta_SFR)
    
    # get absolute SFR, give
    #print(zmax,logsfr_tmax)
    sfr[0] = 10.**logsfr_tmax
    mstar[0] = 10.**logmstar_tmax
    delta_mass = np.zeros(len(timearray))

    for i in range(1,len(timearray)):
        # calc new sfr

        sfr[i] = sfr[i-1]*10.**(SFR_MS(np.log10(mstar[i-1]),redshiftarray[i])-SFR_MS(np.log10(mstar[i-1]),redshiftarray[i-1]))
        dmass = (sfr[i]+sfr[i-1])*0.5*(timearray[i]-timearray[i-1])*1.e9 # 1e9 to convert from Gyr to yr
        mstar[i] = mstar[i-1]+dmass

        # track mass increments in order to calculate mass loss
        delta_mass[i]= dmass
        age_of_mass = timearray - timearray[i]

        # should really cut according to age of first mass loss in Poggianti+2013
        flag = age_of_mass > 0
        frac_retained = get_fraction_mass_retained(age_of_mass[flag]*1.e9)
        #print('hey hey')
        #print(sum(flag),len(frac_retained),frac_retained)
        mass_loss = np.sum(delta_mass[flag]*(1-frac_retained))
        mstar[i] -= mass_loss

    logmstar= np.log10(mstar)
    logsfr = np.log10(sfr)
    return timearray,redshiftarray,logmstar,logsfr,mstar,sfr



def create_grid(minmass=8,maxmass=12,massstep=0.2,minsfr=-2,maxsfr=2,sfrstep=0.2):
    ''' create a grid of models that spans stellar mass and SFR range '''

    logmstar = np.linspace(minmass,maxmass,(maxmass-minmass)/massstep)
    # get SFR relative to main sequence
    logsfr = np.linspace(minsfr,maxsfr,(maxsfr-minsfr)/sfrstep)

    mstar_mesh, sfr_mesh = np.meshgrid(logmstar,logsfr,indexing='ij')
    
    return mstar_mesh, sfr_mesh

def forward_model(minmass=8,maxmass=12,massstep=0.2,minsfr=-2,maxsfr=2,sfrstep=0.2,tmax=3,ntimestep=30):
    ''' this is the function that calls the others '''
    # create grid of mstar and sfr values

    mstar_mesh, sfr_mesh = create_grid(minmass=minmass,maxmass=maxmass,massstep=massstep,\
                                       minsfr=minsfr,maxsfr=maxsfr,sfrstep=sfrstep)
    print(len(mstar_mesh),mstar_mesh)
    print(mstar_mesh[0])
    print(mstar_mesh[:,0])
    print(sfr_mesh)
    # create arrays to store z=0 values of mass and sfr
    size_output_arrays = mstar_mesh.shape[0]*mstar_mesh.shape[1]*ntimestep
    logmstar_all = np.zeros(size_output_arrays,'f')
    logsfr_all = np.zeros(size_output_arrays,'f')    
    logmstar0_all = np.zeros(size_output_arrays,'f')
    logsfr0_all = np.zeros(size_output_arrays,'f')    
    lookbackt_all = np.zeros(size_output_arrays,'f')
    redshift_all = np.zeros(size_output_arrays,'f')
    timestep_all = np.zeros(size_output_arrays,'f')
    massnumber_all = np.zeros(size_output_arrays,'f')
    sfrnumber_all = np.zeros(size_output_arrays,'f')        
    # for each galaxy, evolve it to z=0

    for i in range(len(mstar_mesh[:,0])):
        for j in range(len(sfr_mesh[0])):
            # forward model galaxy
            lookbackt,redshift,logmstar,logsfr = evolve_galaxy(mstar_mesh[i,j],sfr_mesh[i,j],tmax,ntimestep)
            #print(lookbackt,logmstar,logsfr)
            logmstar0 = np.ones(len(logmstar))*logmstar[-1]
            logsfr0 = np.ones(len(logmstar))*logsfr[-1]            
            # save logmstar_0 and logsfr_0
            startindex = (i*len(sfr_mesh)+j)*ntimestep
            stopindex = startindex+ntimestep
            #print(i,j,startindex,stopindex)
            logmstar_all[startindex:stopindex] = logmstar
            logsfr_all[startindex:stopindex] = logsfr
            logmstar0_all[startindex:stopindex] = logmstar0
            logsfr0_all[startindex:stopindex] = logsfr0
            lookbackt_all[startindex:stopindex] = lookbackt
            redshift_all[startindex:stopindex] = redshift
            timestep_all[startindex:stopindex] = np.arange(ntimestep)            
            massnumber_all[startindex:stopindex] = np.ones(ntimestep)*mstar_mesh[i,j]
            sfrnumber_all[startindex:stopindex] = np.ones(ntimestep)*sfr_mesh[i,j]         

    # def write output
    newtable = Table([lookbackt_all,redshift_all,massnumber_all,sfrnumber_all,timestep_all,logmstar_all,logsfr_all,logmstar0_all,logsfr0_all],\
                     names=['lookbackt','redshift','massstep','sfrstep','timestep','logMstar','logSFR','logMstar0','logSFR0'])
    
    outfile = 'forward_model_sfms.fits'
    newtable.write(outfile,overwrite=True,format='fits')
#########################################################
####  MAIN PROGRAM
#########################################################

if __name__ == '__main__':
    ###########################
    ##### SET UP ARGPARSE
    ###########################
    import argparse
    parser = argparse.ArgumentParser(description ='Program to plot radial density profiles of filaments')
    parser.add_argument('--tmax', dest = 'tmax', default=3,help = 'Lookback time (Gyr) to start modeling at.  Default is 3 Gyr.')
    parser.add_argument('--minmass', dest = 'minmass', default = 9, help = 'Min logMstar for grid of models.  Default is 8')
    parser.add_argument('--maxmass', dest = 'maxmass', default = 11, help = 'Max logMstar for grid of models.  Default is 12')
    parser.add_argument('--massstep', dest = 'massstep', default = .5, help = 'Step size for gridding models in stellar mass.  default is 0.1')        
    parser.add_argument('--minsfr', dest = 'minsfr', default = -2, help = 'Min SFR relative to main sequence log(SFR/SFR_MS) for grid of models.  Default is -2')
    parser.add_argument('--maxsfr', dest = 'maxsfr', default = 2, help = 'Max SFR relative to main sequence log(SFR/SFR_MS) for grid of models.  Default is 2')
    parser.add_argument('--sfrstep', dest = 'sfrstep', default = 1, help = 'Step size for gridding models in sfr.  default is 0.1')        
    parser.add_argument('--ntimestep', dest = 'ntimestep', default = 10, help = 'Number of time steps to use when evolving models.  The default is 30, which corresponds to 100 Myr = 0.1 Gyr. ')    

    args = parser.parse_args()
    
    forward_model(minmass=args.minmass,maxmass=args.maxmass,massstep=args.massstep,\
                  minsfr=args.minsfr,maxsfr=args.maxsfr,sfrstep=args.sfrstep,
                  tmax=args.tmax,ntimestep=args.ntimestep)
