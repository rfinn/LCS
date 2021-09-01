#!/usr/bin/env python

'''
GOAL:
* assess the models for various combinations of tmax and tau using a maximum liklihood estimator

APPROACH:
* we have run many models to create a simulated core sample from the z=0 field sample
* we have used the KS test to rule out bad models.  
* we are looking for a complementary statistical approach that would select the "best-fit" models
* in our case, we will use the SFR distribution of the sim-core galaxies as the model, and compare with the observed core sample
* we will bin the sim-core data, and bin the core data.  compare model vs data in each bin

NOTES:
* using this as a reference: https://indico.cern.ch/category/6015/attachments/192/631/Statistics_Fitting_II.pdf


'''

from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
import glob
import multiprocessing as mp
import os

###############################################
##### DEFINITIONS
###############################################
homedir = os.getenv("HOME")
sfr_dir = homedir+'/research/LCS/sfr_modeling/'


###############################################
##### FUNCTIONS
###############################################

def read_simcore(filename):
    ''' read in output from LCSsimulate_infall_sfrs_mp.py  '''

    pass


def get_chisq(model,data,nbin=15,plothist=False):
    '''  

    INPUT:
    * model : e.g. sfrs of simcore galaxies
    * data : e.g. sfrs of core galaxies
    * nbin : number of bins to use for computing chisq

    OUTPUT: 
    * chisq

    '''

    # set up bins using range of model
    mbins = np.linspace(min(model),max(model),nbin)

    # bin the model and data
    nmodel,xbin,binnumb = binned_statistic(model,model,statistic='count',bins=mbins)
    ndata,xbin,binnumb = binned_statistic(data,data,statistic='count',bins=mbins)        
    # calculate chisq_i
    # this assumes poisson noise for model, where variance = # of points in bin
    
    # scaled error to match the size of the data sample
    # by err_scaled = err*(len(data)/len(model))
    uncertainty = nmodel*(len(data)/len(model))**2

    # calc diff b/w data and model, scaling model to get expected # of points for
    # a sample with the same size as data
    chisq_i = (ndata-nmodel*len(data)/len(model))**2/uncertainty
    
    # sum difference
    chisq = np.sum(chisq_i)

    if plothist:
        dnin = mybin[1]-mybin[0]
        plt.figure(figsize=(8,6))
        plt.errorbar(xbin[:-1]+0.5*dbin,nmodel,yerr=np.sqrt(nmodel),label='model')
        plt.errorbar(xbin[:-1]+0.5*dbin,ndata,yerr=np.sqrt(ndata),label='data')
        plt.ylabel('Counts')
        

    return chisq


def get_chisq_models(model_file,core_sfr):
    '''compare model to core sfrs for each unique value of tau   '''
    mtable = Table.read(model_file)

    # get list of unique tau values
    tau_values = list(set(mtable['tau']))
    tau_values.sort()
    
    all_flags = [mtable['tau'] == t for t in tau_values]
    all_chisq=[]
    for i,f in enumerate(all_flags):
        # compare
        all_chisq.append(get_chisq(mtable['logsfr'][f],core_sfr))
    return tau_values, all_chisq
        
def plot_chisq_tau(chisq,tau,labels):
    '''  plot chisq vs tau for each value of tmax'''
    plt.figure(figsize=(8,6))
    for i in range(len(labels)):
        lab = 'tmax={:.0f}Gyr'.format(labels[i])
        plt.plot(tau[i],chisq[i],label=lab,marker='o')
    plt.legend()
    plt.xlabel(r'$ \tau \ (Gyr)$')
    plt.ylabel('$\chi^2 $')    


if __name__ == '__main__':
    print('running main program')

    # read in core

    infile1 = homedir+'/research/LCS/tables/lcs-sfr-sim-BTcut.fits'
    
    lcs = Table.read(infile1)
    core_logsfr = (lcs['logSFR'])
    #core_dsfr = lcs['logSFR'] - get_MS(lcs['logMstar'])
    core_logmstar = (lcs['logMstar'])

    # get list of simcore files
    filelist = glob.glob(sfr_dir+'simcore*nmassmatch1_ndrawmass60.fits')

    filelist.sort()


    chisq_pool = mp.Pool(mp.cpu_count())
    myresults = [chisq_pool.apply_async(get_chisq_models,args=(f,core_logsfr)) for f in filelist]
    chisq_pool.close()
    chisq_pool.join()
    chisq_results = [r.get() for r in myresults]    

    #myresults = [chisq_pool.apply(get_chisq_models,args=(f,core_logsfr)) for f in filelist]
    #chisq_pool.close()
    #chisq_pool.join()
    #chisq_results = myresults
    
    tmax = []
    tau = []
    chisq = []
    # loop through files and calculate the chisq for each value of tau

    
    for i,f in enumerate(filelist):
        print(f)      
        # get tmax from filename
        t1 = f.split('tmax')
        t2 = t1[1].split('_')
        tmax.append(float(t2[0]))
        # calculate chisq
        #tau_values,all_chisq = get_chisq_models(f,core_logsfr)

        #tau.append(tau_values)
        #chisq.append(all_chisq)

        # run the rest in multiprocessing mode

    
        # calculate chisq
        #tau_values,all_chisq = get_chisq_models(f,core_logsfr)

        tau.append(chisq_results[i][0])
        chisq.append(chisq_results[i][1])


    # plot results
    plot_chisq_tau(chisq,tau,labels=tmax)
