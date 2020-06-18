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
parser.add_argument('--plotflag', dest = 'plotflag', default = False,action='store_true', help = 'plot profiles and fits')
# number of simulated galaxies


args = parser.parse_args()

###########################
##### DEFINITIONS
###########################

mipspixelscale=2.45

def sersic(x,Ie,n,Re):
    bn = 1.999*n - 0.327
    return Ie*np.exp(-1*bn*((x/Re)**(1./n)-1))
    
def truncfunc(x,a,b,c):
    return a-b*np.exp(c*x)
    

class simgal():
    def __init__(self,n,Re,Ie,Rtrunc):
        self.Re = Re
        self.n = n
        self.Rtrunc = Rtrunc
        self.Ie = Ie # normalize profile
        self.bn = 1.999*self.n - 0.327
        #print('new simgal: ',n,Re,Ie,Rtrunc)
    def get_sersic_profile(self,rmax=6,rmin=.2):
        ## rmax is in multiples of Re
        ## rmin is in pixels (sorry)
        ## CREATE 1D SERSIC PROFILE BASED ON GALAXY PROFILE
        ## sampled in terms of mips pixels, since Re is in pixels

        # create radius vector that ranges from rmin to a multiple of Re
        
        self.radius = np.arange(.2,int(round(self.Rtrunc*self.Re)))
        self.radius = np.linspace(rmin,int(round(rmax*self.Re)),int(args.nres))
        # from Greg's class notes, slide 14


        ## set the intensity to zero outside the truncation radius
        self.intensity = np.zeros(len(self.radius),'f')
        truncFlag = self.radius < self.Rtrunc*self.Re        
        self.intensity[truncFlag] = self.Ie*np.exp(-1.*self.bn*((self.radius[truncFlag]/self.Re)**(1./self.n)-1))

        noise = np.sqrt(self.Ie)*np.random.normal(size=len(self.intensity))
        #self.intensity += noise
        #self.logintensity = np.log(self.Ie)-

    
    def fit_sersic_profile(self):
        ## REFIT WITH A 1D SERSIC PROFILE
        popt,pcov = curve_fit(sersic,self.radius,self.intensity,p0=[2,2,2])
        self.Ie_sim = popt[0]
        self.n_sim = popt[1]
        self.Re_sim = popt[2]
        self.fit_intensity = sersic(self.radius,self.Ie_sim,self.n_sim,self.Re_sim)
        
    def plot_profiles(self,plotfit=True):

        plt.figure()
        plt.plot(self.radius,self.intensity,label='input')
        plt.axvline(x=self.Re,label='Re',color='b')
        plt.axvline(x=self.Re*self.Rtrunc,label='Rtrunc',color='k',ls='--')
        if plotfit:
            plt.plot(self.radius,self.fit_intensity,label='best fit')
            plt.axvline(x=self.Re_sim,label='Re_sim',color='r',ls='--')
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
        self.summary_plot_3panel()
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

        # using random numbers for now - not worrying about field values
        # until we understand
        self.Re_sim = np.random.uniform(1,20, size=1)[0]
        self.Re_sim = 5
        self.n_sim = np.random.uniform(0.5,4, size=1)[0]        
        #self.Re_sim = np.random.choice(self.tab['fcre1'], self.nrandom)*mipspixelscale

        #self.n_sim = np.random.choice(self.tab['fcnsersic1'], self.nrandom)
            # truncation radius, in terms of Re
        #self.Rtrunc_sim = np.random.uniform(.5,2, size=self.nrandom)
        #self.Ie_sim = 10.*np.ones(self.nrandom,'f')
        
        
        # sets the truncation radius for each galaxy
        self.Rtrunc_sim = np.random.uniform(float(args.minrtrunc),float(args.maxrtrunc), size=1)[0]
        self.Ie_sim = np.random.uniform(5,15, size=1)[0]
        self.Ie_sim = 1.
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
                if args.plotflag:
                    g.plot_profiles()
                self.save_input(i)
                i += 1
            except ValueError:
                print('\nWARNING: ValueError\n')                                
                print('trouble with that fit')
                print(self.n_sim,self.Re_sim,self.Ie_sim,self.Rtrunc_sim)
                if args.plotflag:
                    g.plot_profiles(plotfit=False)
                continue
            except RuntimeError:
                print('\nWARNING: RuntimeError\n')                
                print('trouble with that fit')
                print(self.n_sim,self.Re_sim,self.Ie_sim,self.Rtrunc_sim)
                if args.plotflag:
                    g.plot_profiles(plotfit=False)                
                continue
            ## truncate, refit, get results
    def write_output(self):
        outtab = Table(self.output,names=['n','Re','Ie','Rtrunc','n_fit','Re_fit','Ie_fit'])
        outtab.write('trunc-sersic-1d-results.fits',format='fits',overwrite=True)
        self.outtab = outtab
                       
        ## write output file
    def summary_plot(self):
        d = self.outtab
        plt.figure(figsize=(12,7))
        ptsize=10
        plt.subplots_adjust(wspace=.5,hspace=.4)
        allax=[]
        plt.subplot(2,3,1)
        plt.scatter(d['Rtrunc'],d['n_fit']/d['n'],label='n',c=d['Rtrunc'],s=ptsize)
        #self.add_1to1()
        plt.xlabel('Rtrunc')
        plt.ylabel('n fit/ n input')
        allax.append(plt.gca())
        x1,x2 = plt.xlim()
        #ylim(x1,2*x2)
        plt.subplot(2,3,2)
        plt.scatter(d['Rtrunc'],d['Re_fit']/d['Re'],label='Re',c=d['Rtrunc'],s=ptsize)
        #plt.hist(d['Re_fit']/d['Re'],bins=20)
        #add_1to1()
        plt.xlabel('Rtrunc')
        plt.ylabel('Re fit/Re input')
        x1,x2 = plt.xlim()
        #ylim(x1,2*x2)
        allax.append(plt.gca())
        plt.subplot(2,3,3)
        plt.scatter(d['Rtrunc'],d['Ie_fit']/d['Ie'],label='Ie',c=d['Rtrunc'],s=ptsize)
        #self.add_1to1()
        plt.xlabel('Rtrunc')
        plt.ylabel('Ie fit/Ie input')
        x1,x2 = plt.xlim()
        #ylim(x1,2*x2)
        allax.append(plt.gca())
        plt.subplot(2,3,4)
        plt.hist(d['Rtrunc'])
        plt.xlabel('Rtrunc/Re')
        #plt.xlim(0,5)
        allax.append(plt.gca())
        #plt.colorbar(ax=allax,label='Rtrunc')
        plt.subplot(2,3,5)
        plt.scatter(d['n'],d['Re_fit']/d['Re'],label='Re_fit/Re',c=d['Rtrunc'],s=ptsize)
        plt.xlabel('Sersic index')
        plt.ylabel('Re_fit/Re')
        #plt.ylim(0,10)
        allax.append(plt.gca())
        
        plt.subplot(2,3,6)
        plt.scatter(d['Rtrunc'],d['Re_fit']/d['Re'],label='Re_fit/Re',c=d['n'],s=ptsize)
        plt.xlabel('Truncation radius (Re)')
        plt.ylabel('Re_fit/Re')
        #plt.ylim(0,10)
        #allax.append(plt.gca())
        allax.append(plt.gca())
        # sketch a function to fit size ratio as a function
        # of truncation radius.  go to try
        # y = 1 - exp(-x)
        x1,x2 = plt.xlim()
        xl = np.linspace(x1,x2,100)
        popt,pcov = curve_fit(truncfunc,d['Rtrunc'],d['Re_fit']/d['Re'],p0=[1,1,.833])        
        print('fitting R/Re vs truncation radius with y = a-b*exp(-c*x/d): ',popt)
        yl = truncfunc(xl,popt[0],popt[1],popt[2])
        plt.plot(xl,yl)
        k=2      
        yl = np.arctan((xl*k))/(np.pi/2)
        #plt.plot(xl,yl)
        plt.colorbar(ax=allax,label='Rtrunc')
        plt.savefig('sersic-1d-sim.png')

    def summary_plot_3panel(self):
        d = self.outtab
        x = d['Rtrunc']        
        plt.figure(figsize=(12,3))
        ptsize=10
        plt.subplots_adjust(wspace=.5,hspace=.4)
        allax=[]
        plt.subplot(1,3,1)
        y = d['n_fit']/d['n']
        plt.scatter(x,y,label='n_fit/n',c=d['n'],s=ptsize)
        self.fit_curve(x,y,p0=[1,1,-1])
        plt.xlabel('Rtrunc')
        plt.ylabel('n fit/ n input')
        allax.append(plt.gca())
        
        plt.subplot(1,3,2)
        y = d['Re_fit']/d['Re']
        plt.scatter(x,y,label='Re_fit/Re',c=d['n'],s=ptsize)
        plt.xlabel('Rtrunc (Re)')
        plt.ylabel('Re_fit/Re')
        self.fit_curve(x,y,p0=[1,1,-.8])
        allax.append(plt.gca())
                               
        plt.subplot(1,3,3)
        y = d['Ie_fit']/d['Ie']
        plt.scatter(x,y,label='Ie',c=d['n'],s=ptsize)
        self.fit_curve(x,y,p0=[0,1,-.8])                            
        plt.xlabel('Rtrunc')
        plt.ylabel('Ie fit/Ie input')
        x1,x2 = plt.xlim()
        allax.append(plt.gca())
        
        plt.colorbar(ax=allax,label='Sersic n')
        plt.savefig('sersic-1d-sim-3panel.png')
    def fit_curve(self,x,y,p0=None):
        x1,x2 = plt.xlim()
        xl = np.linspace(x1,x2,100)
        if p0 is None:
            popt,pcov = curve_fit(truncfunc,x,y)
        else:
            popt,pcov = curve_fit(truncfunc,x,y,p0=p0)        
        print('fitting vs truncation radius with y = a-b*exp(c*x): ',popt)
        yl = truncfunc(xl,popt[0],popt[1],popt[2])
        plt.plot(xl,yl)
        return popt,pcov
        
    def add_1to1(self):
        x1,x2 = plt.xlim()
        xl= np.linspace(x1,x2,10)
        plt.plot(xl,xl,'k--',label='1:1')
    
if __name__ == '__main__':

    infile = '/home/rfinn/research/LCS/tables/LCS_all_size_KE_SFR.fits'
    s = simsample(infile,nrandom=int(args.ngal))
    s.runall()
