#!/usr/bin/env python

from astropy.io import fits,ascii
from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import argparse
from scipy.stats import ks_2samp
# the incomplete gamma function, for integrating the sersic profile
from scipy.optimize import curve_fit

###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Program to run analysis for LCS paper 2')
parser.add_argument('--nres', dest = 'nres', default = 20., help = 'number of resolution elements on profile.  default is 20.')
parser.add_argument('--ngal', dest = 'ngal', default = 100, help = 'number of simulated galaxies.  The default is 100.')
# min truncation radius, in terms of Re
parser.add_argument('--minrtrunc', dest = 'minrtrunc', default = 1.5, help = 'Minimum truncation radius, in terms of Re.  Default is 1.5')
# max truncation radius, in terms of Re
parser.add_argument('--maxrtrunc', dest = 'maxrtrunc', default = 4, help = 'Maximum truncation radius, in terms of Re.  Default is 4.')
# number of simulated galaxies


args = parser.parse_args()

###########################
##### DEFINITIONS
###########################

mipspixelscale=2.45

def sersic(x,Ie,n,Re):
    bn = 1.999*n - 0.327
    return Ie*np.exp(-1*bn*((x/Re)**(1./n)-1))
    

class simgal():
    def __init__(self,n,Re,Ie,Rtrunc):
        self.Re = Re
        self.n = n
        self.Rtrunc = Rtrunc
        self.Ie = Ie # normalize profile
        self.bn = 1.999*self.n - 0.327
        #print('new simgal: ',n,Re,Ie,Rtrunc)
    def get_sersic_profile(self):
        ## CREATE 1D SERSIC PROFILE BASED ON GALAXY PROFILE
        ## sampled in terms of mips pixels, since Re is in pixels
        self.radius = np.arange(.2,int(round(self.Rtrunc*self.Re)))
        self.radius = np.linspace(1,int(round(self.Rtrunc*self.Re)),int(args.nres))
        # from Greg's class notes, slide 14

        self.intensity = self.Ie*np.exp(-1.*self.bn*((self.radius/self.Re)**(1./self.n)-1))

        noise = np.sqrt(self.Ie)*np.random.normal(size=len(self.intensity))
        self.intensity += noise
        #self.logintensity = np.log(self.Ie)-

    
    def fit_sersic_profile(self):
        ## REFIT WITH A 1D SERSIC PROFILE
        popt,pcov = curve_fit(sersic,self.radius,self.intensity)
        self.Ie_sim = popt[0]
        self.n_sim = popt[1]
        self.Re_sim = popt[2]
        self.fit_intensity = sersic(self.radius,self.Ie_sim,self.n_sim,self.Re_sim)
        
    def plot_profiles(self):

        plt.figure()
        plt.plot(self.radius,self.intensity,label='input')
        plt.plot(self.radius,self.fit_intensity,label='best fit')        
        plt.axvline(x=self.Re,label='Re')
        plt.axvline(x=self.Re*self.Rtrunc,label='Rtrunc')
        plt.legend()
        #print(self.radius)
        #print(self.intensity)

    def get_results(self):
        return self.n_sim,self.Re_sim, self.Ie_sim

## SIMSAMPLE IS A COLLECTION OF SIMGAL'S
class simsample():
    def __init__(self,infile,nrandom=100):
        ## read file
        sizes = Table.read(infile)
        # keep only galaxies in paper 1 sample
        self.tab = sizes[sizes['sampleflag']]
        # saved data will be (n,Re,Ie)sim Rtrunc, and (n,Re,Ie)fit
        self.output = np.zeros((nrandom,7),'f')
        self.nrandom = nrandom
    def runall(self):
        self.keep_external()
        self.get_galaxy()
        self.call_simgal()
        self.write_output()
    def keep_external(self):
        self.tab = self.tab[~self.tab['membflag']]
        
    def get_galaxy(self):
        ## select random external galaxy
        self.Re_sim=1000

        self.index = np.random.randint(0,len(self.tab)-1,1)[0]
        #print('random index = ',index)
        # effective radius of 24um, in pixels
        self.Re_sim = self.tab['fcre1'][self.index]*mipspixelscale
        self.n_sim = self.tab['fcnsersic1'][self.index]            
        #self.Re_sim = np.random.choice(self.tab['fcre1'], self.nrandom)*mipspixelscale

        #self.n_sim = np.random.choice(self.tab['fcnsersic1'], self.nrandom)
            # truncation radius, in terms of Re
        #self.Rtrunc_sim = np.random.uniform(.5,2, size=self.nrandom)
        #self.Ie_sim = 10.*np.ones(self.nrandom,'f')
        
        

        self.Rtrunc_sim = np.random.uniform(float(args.minrtrunc),float(args.maxrtrunc), size=1)[0]
        self.Ie_sim = np.random.uniform(5,15, size=1)[0]
    def save_input(self,i):
        self.output[i,0] = self.n_sim
        self.output[i,1] = self.Re_sim
        self.output[i,2] = self.Ie_sim
        self.output[i,3] = self.Rtrunc_sim        
    def call_simgal(self):
        i = 0
        while i < self.nrandom:
            ## create simgal
            self.get_galaxy()
            if (self.Re_sim < mipspixelscale):
                self.get_galaxy()
            g = simgal(self.n_sim,self.Re_sim,self.Ie_sim,self.Rtrunc_sim)
            g.get_sersic_profile()
            try:
                g.fit_sersic_profile()
                self.output[i,4],self.output[i,5],self.output[i,6] = g.get_results()
                g.plot_profiles()
                self.save_input(i)
                i += 1
            except ValueError:
                print('\nWARNING: ValueError\n')                                
                print('trouble with that fit')
                continue
            except RuntimeError:
                print('\nWARNING: RuntimeError\n')                
                print('trouble with that fit')
                continue
            ## truncate, refit, get results
    def write_output(self):
        outtab = Table(self.output,names=['n','Re','Ie','Rtrunc','n_fit','Re_fit','Ie_fit'])
        outtab.write('trunc-sersic-1d-results.fits',format='fits',overwrite=True)
                       
        ## write output file

if __name__ == '__main__':

    infile = '/home/rfinn/research/LCS/tables/LCS_all_size_KE_SFR.fits'
    s = simsample(infile,nrandom=int(args.ngal))
    s.runall()
