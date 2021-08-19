#!/usr/bin/env python

'''
OVERVIEW:
This program was developed to link the SFR,Mstar values of the z=0 field galaxies with their expected values 
in the past.  We are using the field galaxies as the starting point for simulating what happens to SFR when a
galaxy falls into a cluster.  However, the field galaxies have a higher sfr and lower Mstar value at the time 
of infall, and we want to take this into account.  This program creates a grid of galaxies at a user-selected
value of tmax.  The SFR is evolved using the Whitaker+XX evolution of the SF MS, and the mass loss is modeled
uing a relationship from Poggianti+XX.  The SFR and Mstar values at tmax are linked to their lower-redshift 
counterparts.  Additional programs can be used to link the z=0 field galaxies with their progenitors.

GOAL:
* create a set of galaxies with a range of stellar mass and sSFR relative to MS
* evolve galaxies from some epoch tmax to the present
* evolve SFRs according to MS evolution given in Whitaker+2012
* evolve stellar mass based on SFR evolution and mass loss as described in Poggianti+2013

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
    use Poggianti+2013 relation to determine fraction of mass retained.  This applies to 
    stellar populations with ages greater than 1.9E6 yrs.  For younger pops, the fraction 
    retained is just one.

    INPUT:
    * time in yr since the birth of the stellar pop

    RETURNS:
    * fraction of the mass that is retained after mass loss when age of population is t_yr

    '''
    #try:
    #print('in get_fraction_mass_retained, t = ',t)

    frac = np.ones(len(t))
    flag = t > 1.9e6
    frac[flag] = 1.749 - 0.124*np.log10(t[flag])

    #except:
    #    if t < 1.9e6:
    #        frac = 1
    #    else:
    #        frac = 1.749 - 0.124*np.log10(t) 
    return frac



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
    mstar = np.zeros(len(timearray),'d')
    sfr = np.zeros(len(timearray),'d')
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
    mass_loss = np.zeros(len(timearray))    

    for i in range(1,len(timearray)):

        # we are assuming that the mass doesn't change enough to affect our value of delta_sfr_ms
        # so we are using mstar[i-1] to get the previous and current SFR_MS
        delta_sfr_ms_ratio = 10.**(SFR_MS(np.log10(mstar[i-1]),redshiftarray[i]))/10.**(SFR_MS(np.log10(mstar[i-1]),redshiftarray[i-1]))
        # calc new sfr        
        sfr[i] = sfr[i-1]*delta_sfr_ms_ratio
        # calculate the mass increment and track it so we can calculate mass loss
        # 1e9 to convert from Gyr to yr because time array is in Gyr
        delta_mass[i] = (sfr[i]+sfr[i-1])*0.5*(timearray[i-1]-timearray[i])*1.e9 
        # calculate new stellar mass
        mstar[i] = mstar[i-1]+delta_mass[i]

        # now setting up mass loss, which we need to calculate for each mass increment
        # calculate age of each mass increment
        # this is the original array of times minus the current time of step i
        age_of_mass = timearray - timearray[i]

        
        # cut according to age of first mass loss in Poggianti+2013
        # this will eliminate any points that are downstream of the current step i
        # 
        # this also limits any time step that occurred less than 1.9 Myr ago b/c
        # no mass loss will have occurred yet.
        flag = (age_of_mass > 1.9e6/1e9) 
        # convert age from Gyr to yr
        # use Poggianti+2013 relation to get the fraction of mass retained in each delta_mass
        frac_retained = get_fraction_mass_retained(age_of_mass[flag]*1.e9)

        # calculate the sum of each mass increment times the amount of mass lost from that increment
        mass_loss[i] = np.sum(delta_mass[flag]*(1-frac_retained))

        # mass loss in this increment is mass_loss[i] - mass_loss[i-1]
        # because the formula gives the cumulative mass loss up until time of step i
        new_mass_loss = mass_loss[i] - mass_loss[i-1]

        # subtract the additional mass that was lost in this time step
        mstar[i] -= new_mass_loss

    logmstar= np.log10(mstar)
    logsfr = np.log10(sfr)
    return timearray,redshiftarray,logmstar,logsfr,mstar,sfr,mass_loss



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
    #print(len(mstar_mesh),mstar_mesh)
    #print(mstar_mesh[0])
    #print(mstar_mesh[:,0])
    #print(sfr_mesh)
    # create arrays to store z=0 values of mass and sfr
    size_output_arrays = int(mstar_mesh.shape[0]*mstar_mesh.shape[1]*ntimestep)
    #size_output_arrays = 20*40*30
    logmstar_all = np.zeros(size_output_arrays,'d')
    logsfr_all = np.zeros(size_output_arrays,'d')    
    logmstar0_all = np.zeros(size_output_arrays,'d')
    logsfr0_all = np.zeros(size_output_arrays,'d')    
    lookbackt_all = np.zeros(size_output_arrays,'d')
    redshift_all = np.zeros(size_output_arrays,'d')
    timestep_all = np.zeros(size_output_arrays,'d')
    massnumber_all = np.zeros(size_output_arrays,'d')
    sfrnumber_all = np.zeros(size_output_arrays,'d')
    mass_loss_all = np.zeros(size_output_arrays,'d')
    galnumber_all = np.zeros(size_output_arrays,'d')                
    # for each galaxy, evolve it to z=0
    galnumber = 0
    for i in range(len(mstar_mesh[:,0])):
        for j in range(len(sfr_mesh[0])):
            # forward model galaxy

            lookbackt,redshift,logmstar,logsfr,mstar,sfr,mass_loss = evolve_galaxy(mstar_mesh[i,j],sfr_mesh[i,j],tmax,ntimestep)
            #print(mstar_mesh[i,j],sfr_mesh[i,j])
            logmstar0 = np.ones(len(logmstar))*logmstar[-1]
            logsfr0 = np.ones(len(logmstar))*logsfr[-1]            
            # save logmstar_0 and logsfr_0
            startindex = (i*len(sfr_mesh[0])+j)*ntimestep
            stopindex = startindex+ntimestep
            #print(i,j,startindex,stopindex)
            logmstar_all[startindex:stopindex] = logmstar
            logsfr_all[startindex:stopindex] = logsfr
            logmstar0_all[startindex:stopindex] = logmstar0
            logsfr0_all[startindex:stopindex] = logsfr0
            lookbackt_all[startindex:stopindex] = lookbackt
            redshift_all[startindex:stopindex] = redshift
            mass_loss_all[startindex:stopindex] = mass_loss           
            timestep_all[startindex:stopindex] = np.arange(ntimestep)            
            massnumber_all[startindex:stopindex] = np.ones(ntimestep)*mstar_mesh[i,j]
            sfrnumber_all[startindex:stopindex] = np.ones(ntimestep)*sfr_mesh[i,j]
            galnumber_all[startindex:stopindex] = np.ones(ntimestep)*galnumber
            galnumber += 1


    # def write output
    newtable = Table([galnumber_all,lookbackt_all,redshift_all,massnumber_all,sfrnumber_all,timestep_all,logmstar_all,logsfr_all,logmstar0_all,logsfr0_all, mass_loss_all],\
                     names=['galnumber','lookbackt','redshift','massstep','sfrstep','timestep','logMstar','logSFR','logMstar0','logSFR0','mass_loss'])
    
    outfile = 'forward_model_sfms_tmax_{}.fits'.format(args.tmax)
    newtable.write(outfile,overwrite=True,format='fits')
    return mstar_mesh, sfr_mesh, newtable
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
    parser.add_argument('--massstep', dest = 'massstep', default = .1, help = 'Step size for gridding models in stellar mass.  default is 0.1')        
    parser.add_argument('--minsfr', dest = 'minsfr', default = -2, help = 'Min SFR relative to main sequence log(SFR/SFR_MS) for grid of models.  Default is -2')
    parser.add_argument('--maxsfr', dest = 'maxsfr', default = 2, help = 'Max SFR relative to main sequence log(SFR/SFR_MS) for grid of models.  Default is 2')
    parser.add_argument('--sfrstep', dest = 'sfrstep', default = .1, help = 'Step size for gridding models in sfr.  default is 0.1')        
    parser.add_argument('--ntimestep', dest = 'ntimestep', default = 50, help = 'Number of time steps to use when evolving models.  The default is 50, which corresponds to tmax/50 Gyr.  If tmax is 5 Gyr, then the timestep is 100 Myr = 0.1 Gyr. ')    

    args = parser.parse_args()
    
    t = forward_model(minmass=float(args.minmass),maxmass=float(args.maxmass),\
                      massstep=float(args.massstep),\
                      minsfr=float(args.minsfr),maxsfr=float(args.maxsfr),\
                      sfrstep=float(args.sfrstep),tmax=float(args.tmax),\
                      ntimestep=int(args.ntimestep))
