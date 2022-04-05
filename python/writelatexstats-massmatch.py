#!/usr/bin/env python

'''
GOAL:
* write out the statistics table for the main sequence paper

'''
import sys
import os
from astropy.table import Table
from scipy.stats import ks_2samp, anderson_ksamp, binned_statistic

homedir = os.getenv("HOME")
tabledir = homedir+'/research/LCS/tables/'

sys.path.append(homedir+'/github/LCS/python/')
from lcs_paper2 import mass_match

NMASSMATCH = 30 # number of field to draw for each cluster galaxy
class writetable():
    def __init__(self):
        
        pass

    def read_tables(self):
        # read

        self.core = Table.read(tabledir+'/LCS-sfr-mstar-core.fits')
        self.infall = Table.read(tabledir+'/LCS-sfr-mstar-infall.fits')
        self.field = Table.read(tabledir+'/LCS-sfr-mstar-field.fits')        
        self.coreBT = Table.read(tabledir+'/LCS-sfr-mstar-core-BTcut.fits')
        self.infallBT = Table.read(tabledir+'/LCS-sfr-mstar-infall-BTcut.fits')
        self.fieldBT = Table.read(tabledir+'/LCS-sfr-mstar-field-BTcut.fits')
        
    def run_ks(self,massmatch=False,BTcut=False):
        all_results = []
        all_results_AD = []
        if BTcut:
            core = self.coreBT
            infall = self.infallBT
            field = self.fieldBT
        else:
            core = self.core
            infall = self.infall
            field = self.field

        # match new tables to old variables...

        mc = core['logmstar']
        sfrc = core['logsfr']
        dsfrc = core['dlogsfr']
        BTc = core['BT']
        
        mi = infall['logmstar']
        sfri = infall['logsfr']
        dsfri = infall['dlogsfr']
        BTi = infall['BT']
        
        mf = field['logmstar']
        sfrf = field['logsfr']
        dsfrf = field['dlogsfr']
        BTf = field['BT']
        
        if massmatch:
            seed = 23654
            keep_indices = mass_match(mc,mf,seed,nmatch=NMASSMATCH)            
            mf_matchc = mf[keep_indices]
            sfrf_matchc = sfrf[keep_indices]
            dsfrf_matchc = dsfrf[keep_indices]
            BTf_matchc = BTf[keep_indices]            
            ### match to infall
            keep_indices = mass_match(mi,mf,seed,nmatch=NMASSMATCH)                        
            mf_matchi = mf[keep_indices]
            sfrf_matchi = sfrf[keep_indices]
            dsfrf_matchi = dsfrf[keep_indices]
            BTf_matchi = BTf[keep_indices]                        

        ##############################################        
        # LCS core vs field: SFR, Stellar mass, dSFR
        ##############################################
        if massmatch:
            mf = mf_matchc
            sfrf = sfrf_matchc
            dsfrf = dsfrf_matchc
            BTf = BTf_matchc            
        print('########################################')
        
        print('LCS Core vs Field: SFR')
        t = ks_2samp(sfrc,sfrf)
        t_sfr_fc = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))        
        t = anderson_ksamp([sfrc,sfrf])
        t_sfr_fc_AD = [t[0],t[2]] 
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))               
        print()

        print('LCS Core vs Field: dSFR')
        t = ks_2samp(dsfrc,dsfrf)
        t_dsfr_fc = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        t = anderson_ksamp([dsfrc,dsfrf])
        t_dsfr_fc_AD = [t[0],t[2]]
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))                       
        print()
        
        print('LCS Core vs Field: Mstar')
        t = ks_2samp(mc,mf)
        t_mstar_fc = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        t=anderson_ksamp([mc,mf])
        t_mstar_fc_AD = [t[0],t[2]]
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))          
        print()
        
        print('LCS Core vs Field: BT')
        t = ks_2samp(BTc,BTf)
        t_BT_fc = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        t = anderson_ksamp([BTc,BTf])
        t_BT_fc_AD = [t[0],t[2]]
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))
        
        all_results.append(t_sfr_fc)
        all_results.append(t_dsfr_fc)
        all_results.append(t_mstar_fc)
        all_results.append(t_BT_fc)

        all_results_AD.append(t_sfr_fc_AD)
        all_results_AD.append(t_dsfr_fc_AD)
        all_results_AD.append(t_mstar_fc_AD)
        all_results_AD.append(t_BT_fc_AD)        
        
        ##############################################
        # LCS infall vs field: SFR, Stellar mass, dSFR
        ##############################################        
        if massmatch:
            mf = mf_matchi
            sfrf = sfrf_matchi
            dsfrf = dsfrf_matchi
            BTf = BTf_matchi            
        print('########################################')            
        
        print('LCS Infall vs Field: SFR')
        t = ks_2samp(sfri,sfrf)
        t_sfr_fi = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        t = anderson_ksamp([sfri,sfrf])
        t_sfr_fi_AD = [t[0],t[2]]
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))                       
        print()
        
        print('LCS Infall vs Field: dSFR')
        t = ks_2samp(dsfri,dsfrf)
        t_dsfr_fi = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        t  = anderson_ksamp([dsfri,dsfrf])
        t_dsfr_fi_AD = [t[0],t[2]]        
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))                       
        print()

        print('LCS Infall vs Field: Mstar')
        t = ks_2samp(mi,mf)
        t_mstar_fi = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        t = anderson_ksamp([mi,mf])
        t_mstar_fi_AD = [t[0],t[2]]
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))                       
        print()

        print('LCS Infall vs Field: BT')
        t = ks_2samp(BTi,BTf)
        t_BT_fi = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        t = anderson_ksamp([BTi,BTf])
        t_BT_fi_AD = [t[0],t[2]]
        print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))                       
        print()

        all_results.append(t_sfr_fi)
        all_results.append(t_dsfr_fi)
        all_results.append(t_mstar_fi)
        all_results.append(t_BT_fi)

        all_results_AD.append(t_sfr_fi_AD)
        all_results_AD.append(t_dsfr_fi_AD)
        all_results_AD.append(t_mstar_fi_AD)
        all_results_AD.append(t_BT_fi_AD)
        
        ##############################################        
        # LCS core vs infall: SFR, Stellar mass, dSFR
        ##############################################
        #if not massmatch:
        if massmatch:
            print('########################################')
        
            print('LCS Core vs Infall: SFR')
            t = ks_2samp(sfrc,sfri)
            t_sfr_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))        
            t = anderson_ksamp([sfrc,sfri])
            t_sfr_ci_AD = [t[0],t[2]]                            
            print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))
            
            print('LCS Core vs Infall: dSFR')
            t = ks_2samp(dsfrc,dsfri)
            t_dsfr_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
            t = anderson_ksamp([dsfrc,dsfri])
            t_dsfr_ci_AD = [t[0],t[2]]                                        
            print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))
            
            print('LCS Core vs Infall: Mstar')
            t = ks_2samp(mc,mi)
            t_mstar_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
            t = anderson_ksamp([mc,mi])
            t_mstar_ci_AD = [t[0],t[2]]
            print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))
            
            print('LCS Core vs Infall: BT')
            t = ks_2samp(BTc,BTi)
            t_BT_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
            t = anderson_ksamp([BTc,BTi])
            t_BT_ci_AD = [t[0],t[2]]                                        
            print('\tAD D = {:.2e}, pvalue = {:.2e}'.format(t[0],t[2]))
            
            all_results.append(t_sfr_ci)
            all_results.append(t_dsfr_ci)
            all_results.append(t_mstar_ci)
            all_results.append(t_BT_ci)

            all_results_AD.append(t_sfr_ci_AD)
            all_results_AD.append(t_dsfr_ci_AD)
            all_results_AD.append(t_mstar_ci_AD)
            all_results_AD.append(t_BT_ci_AD)            

        # changing to use the Anderson Darling test instead of KS
        return all_results_AD

    def get_stats(self,massmatch=True):

        # no mass match
        self.allstats = self.run_ks(massmatch=massmatch)
        # with mass match
        #self.allstats_mm = self.run_ks(massmatch=True)
        # no mass match
        self.allstatsBT = self.run_ks(BTcut=True,massmatch=massmatch)
        # with mass match
        #self.allstatsBT_mm = self.run_ks(massmatch=True,BTcut=True)

    def open_output(self):
        self.outfile = open('table1-massmatch.tex','w')
    def write_header(self):
        # latex header lines
        # write the header
        self.outfile.write('\\begin{table*}%[h] \n')
        self.outfile.write('\\centering \n')
        self.outfile.write('\\begin{tabular}{|c|c|c|c|c|} \n')
        self.outfile.write('\\hline\n')
        self.outfile.write('Samples &Mass Matched & Variable  &\multicolumn{1}{c|}{All $B/T$} &  \multicolumn{1}{c|}{$B/T < 0.3$}\\\\ \n')
        self.outfile.write('& & & {A.D. p value}& {A.D. p value} \\\\ \n')
        #self.outfile.write('& &  no mass match & mass match & no mass match & mass match\\\\ \n')
        self.outfile.write('\hline \hline \n')
        
        pass
    def close_output(self):
        self.outfile.close()
    def write_data(self):
        ##########################
        #### CORE VS FIELD
        ##########################

        # use AD statistics
        
        a = self.allstats[0][1]
        #b = self.allstats_mm[0][1]
        c = self.allstatsBT[0][1]
        #d = self.allstatsBT_mm[0][1]
        self.outfile.write('Core-Field    & Yes & $\log SFR$         &  {:.2e} &  {:.2e}  \\\\ \n'.format(a,c))

        a = self.allstats[1][1]
        #b = self.allstats_mm[1][1]
        c = self.allstatsBT[1][1]
        #d = self.allstatsBT_mm[1][1]        
        self.outfile.write(' && $\Delta \log $SFR  &  {:.2e} &  {:.2e}  \\\\ \n'.format(a,c))


        a = self.allstats[3][1]
        #b = self.allstats_mm[3][1]
        c = self.allstatsBT[3][1]
        #d = self.allstatsBT_mm[3][1]        
        self.outfile.write(' && $B/T$    & {:.2e} &  {:.2e}   \\\\ \n'.format(a,c))
        a = self.allstats[2][1]
        #b = self.allstats_mm[2][1]
        c = self.allstatsBT[2][1]
        #d = self.allstatsBT_mm[2][1]        
        self.outfile.write(' && $\log M_\star/M_\odot$    &  {:.2e} & {:.2e}   \\\\ \n'.format(a,c))
        
        i=3
        i +=1
        ##########################
        #### INFALL VS FIELD
        ##########################        
        a = self.allstats[i][1]
        #b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        #d = self.allstatsBT_mm[i][1]        
        self.outfile.write('\\hline \n')
        self.outfile.write('Infall-Field &Yes & $\log$ SFR & {:.2e} &  {:.2e}\\\\ \n'.format(a,c))

        i += 1
        
        a = self.allstats[i][1]
        #b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        #d = self.allstatsBT_mm[i][1]        
        self.outfile.write('&& $\Delta \log$SFR  &  {:.2e}   &  {:.2e} \\\\ \n'.format(a,c))

        i += 2
        
        a = self.allstats[i][1]
        #b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        #d = self.allstatsBT_mm[i][1]        
        self.outfile.write('&&$B/T$   &   {:.2e} &  {:.2e}  \\\\ \n'.format(a,c))
        i -= 1        
        
        a = self.allstats[i][1]
        #b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        #d = self.allstatsBT_mm[i][1]        
        self.outfile.write('&&$\log M_\star/M_\odot$   &  {:.2e} &  {:.2e}  \\\\ \n'.format(a,c))
        
        self.outfile.write('\\hline \n')
        ##########################
        #### CORE VS INFALL
        ##########################        

        i += 2
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]        
        self.outfile.write('Core-Infall &No    & $\\log$ SFR      &  {:.2e}    &  {:.2e}   \\\\ \n'.format(a,b))

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]                
        self.outfile.write('&& $\Delta \log$SFR&  {:.2e} &  {:.2e}  \\\\ \n'.format(a,b))


        i += 2
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]        
        self.outfile.write('&& $B/T$ & {:.2e}  & {:.2e}   \\\\ \n'.format(a,b))


        i -= 1
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]        
        self.outfile.write('&& $\log M_\star/M_\odot$ & {:.2e}  & {:.2e}  \\\\ \n'.format(a,b))
        
        pass

    def write_footer(self):
        self.outfile.write('\\hline \n')
        self.outfile.write('\\end{tabular} \n')
        self.outfile.write('\\caption{Summary statistics for SFR, $\Delta$SFR,  $B/T$, and stellar mass comparisons.  The field samples are mass-matched to both the core and infall samples, yet we include the mass comparisons for completeness.  Bold text indicates when the two populations differ at the 99.7\% or $\ge 3\sigma$ confidence interval (A.D. p-value$<$3.0e-03). NOTE: {scipy.stats.anderson\\_ksamp} floors the p value at 0.1\\% and caps the p value at 25\%, so it will not return p values below 1E-3 or above 0.25.} \n')

        self.outfile.write('\\label{tab:stats} \n')
        self.outfile.write('\\end{table*} \n')

        # write out footer

        # close file

        pass

