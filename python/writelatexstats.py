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
            keep_indices = mass_match(mc,mf)            
            mf_matchc = mf[keep_indices]
            sfrf_matchc = sfrf[keep_indices]
            dsfrf_matchc = dsfrf[keep_indices]
            BTf_matchc = BTf[keep_indices]            
            ### match to infall
            keep_indices = mass_match(mi,mf)            
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
        #print(anderson_ksamp([sfrc,sfrf]))
        print()

        print('LCS Core vs Field: dSFR')
        t = ks_2samp(dsfrc,dsfrf)
        t_dsfr_fc = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        #print(anderson_ksamp([dsfrc,dsfrf]))
        print()
        print('LCS Core vs Field: Mstar')
        t = ks_2samp(mc,mf)
        t_mstar_fc = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        #print(anderson_ksamp([mc,mf]))
        print()
        print('LCS Core vs Field: BT')
        t = ks_2samp(BTc,BTf)
        t_BT_fc = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        #print(anderson_ksamp([mc,mf]))

        all_results.append(t_sfr_fc)
        all_results.append(t_dsfr_fc)
        all_results.append(t_mstar_fc)
        all_results.append(t_BT_fc)        
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
        #print(anderson_ksamp([sfri,sfrf]))
        print()
        
        print('LCS Infall vs Field: dSFR')
        t = ks_2samp(dsfri,dsfrf)
        t_dsfr_fi = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        #print(anderson_ksamp([dsfri,dsfrf]))
        print()

        print('LCS Infall vs Field: Mstar')
        t = ks_2samp(mi,mf)
        t_mstar_fi = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        #print(anderson_ksamp([mi,mf]))
        print()

        print('LCS Infall vs Field: BT')
        t = ks_2samp(BTi,BTf)
        t_BT_fi = t
        print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
        #print(anderson_ksamp([mi,mf]))
        print()

        all_results.append(t_sfr_fi)
        all_results.append(t_dsfr_fi)
        all_results.append(t_mstar_fi)
        all_results.append(t_BT_fi)
        
        ##############################################        
        # LCS core vs infall: SFR, Stellar mass, dSFR
        ##############################################
        if not massmatch:
            print('########################################')
        
            print('LCS Core vs Infall: SFR')
            t = ks_2samp(sfrc,sfri)
            t_sfr_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))        

            print('LCS Core vs Infall: dSFR')
            t = ks_2samp(dsfrc,dsfri)
            t_dsfr_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
            
            print('LCS Core vs Infall: Mstar')
            t = ks_2samp(mc,mi)
            t_mstar_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))

            print('LCS Core vs Infall: BT')
            t = ks_2samp(BTc,BTi)
            t_BT_ci = t
            print('\tD = {:.2e}, pvalue = {:.2e}'.format(t[0],t[1]))
            
            all_results.append(t_sfr_ci)
            all_results.append(t_dsfr_ci)
            all_results.append(t_mstar_ci)
            all_results.append(t_BT_ci)            
        return all_results

    def get_stats(self):

        # no mass match
        self.allstats = self.run_ks()
        # with mass match
        self.allstats_mm = self.run_ks(massmatch=True)
        # no mass match
        self.allstatsBT = self.run_ks(BTcut=True)
        # with mass match
        self.allstatsBT_mm = self.run_ks(massmatch=True,BTcut=True)

    def open_output(self):
        self.outfile = open('table1.tex','w')
    def write_header(self):
        # latex header lines
        # write the header
        self.outfile.write('\\begin{table*}%[h] \n')
        self.outfile.write('\\centering \n')
        self.outfile.write('\\begin{tabular}{|c|c|c|c|c|c|} \n')
        self.outfile.write('\\hline\n')
        self.outfile.write('Samples &Variable  &\multicolumn{2}{c|}{All $B/T$} &  \multicolumn{2}{c|}{$B/T < 0.4$}\\\\ \n')
        self.outfile.write('& & \multicolumn{2}{c|}{KS p value}& \multicolumn{2}{c|}{KS p value} \\\\ \n')
        self.outfile.write('& &  no mass match & mass match & no mass match & mass match\\\\ \n')
        self.outfile.write('\hline \hline \n')
        
        pass
    def close_output(self):
        self.outfile.close()
    def write_data(self):
        ##########################
        #### CORE VS FIELD
        ##########################        
        a = self.allstats[0][1]
        b = self.allstats_mm[0][1]
        c = self.allstatsBT[0][1]
        d = self.allstatsBT_mm[0][1]
        self.outfile.write('LCS Core-Field    & $\log SFR$         & \\bf {:.2e} & \\bf {:.2e} & \\bf {:.2e}& \\bf {:.2e} \\\\ \n'.format(a,b,c,d))

        a = self.allstats[1][1]
        b = self.allstats_mm[1][1]
        c = self.allstatsBT[1][1]
        d = self.allstatsBT_mm[1][1]        
        self.outfile.write(' & $\Delta \log $SFR  & \\bf {:.2e} & \\bf {:.2e} & \\bf {:.2e} & \\bf {:.2e} \\\\ \n'.format(a,b,c,d,))

        a = self.allstats[2][1]
        b = self.allstats_mm[2][1]
        c = self.allstatsBT[2][1]
        d = self.allstatsBT_mm[2][1]        
        self.outfile.write(' & $\log M_\star/M_\odot$    & {:.2e} & {:.2e} & {:.2e} & {:.2e}  \\\\ \n'.format(a,b,c,d))

        a = self.allstats[3][1]
        b = self.allstats_mm[3][1]
        c = self.allstatsBT[3][1]
        d = self.allstatsBT_mm[3][1]        
        self.outfile.write(' & $B/T$    &\\bf {:.2e} &\\bf {:.2e} &\\bf {:.2e} &\\bf {:.2e}  \\\\ \n'.format(a,b,c,d))
        i=3
        i +=1
        ##########################
        #### INFALL VS FIELD
        ##########################        
        a = self.allstats[i][1]
        b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        d = self.allstatsBT_mm[i][1]        
        self.outfile.write('\\hline \n')
        self.outfile.write('LCS Infall-Field  & $\log$ SFR &\\bf {:.2e} &\\bf {:.2e} & {:.2e} &  {:.2e}\\\\ \n'.format(a,b,c,d))

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        d = self.allstatsBT_mm[i][1]        
        self.outfile.write('& $\Delta \log$SFR  & \\bf {:.2e}   & \\bf {:.2e} & {:.2e} &    {:.2e}\\\\ \n'.format(a,b,c,d))

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        d = self.allstatsBT_mm[i][1]        
        self.outfile.write('&$\log M_\star/M_\odot$   &  {:.2e} &  {:.2e} & {:.2e}  & {:.2e} \\\\ \n'.format(a,b,c,d))

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstats_mm[i][1]
        c = self.allstatsBT[i][1]
        d = self.allstatsBT_mm[i][1]        
        self.outfile.write('&$B/T$   &  {:.2e} &  {:.2e} & {:.2e}  & {:.2e} \\\\ \n'.format(a,b,c,d))
        
        
        self.outfile.write('\\hline \n')
        ##########################
        #### CORE VS INFALL
        ##########################        

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]        
        self.outfile.write('LCS Core-Infall    & $\\log$ SFR      & {:.2e}  & \\nodata  & {:.2e} &\\nodata  \\\\ \n'.format(a,b))

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]                
        self.outfile.write('& $\Delta \log$SFR& {:.2e} & \\nodata & {:.2e}  &\\nodata  \\\\ \n'.format(a,b))

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]        
        self.outfile.write('& $\log M_\star/M_\odot$ & {:.2e}  & \\nodata & {:.2e}  &\\nodata \\\\ \n'.format(a,b))

        i += 1
        
        a = self.allstats[i][1]
        b = self.allstatsBT[i][1]        
        self.outfile.write('& $B/T$ & {:.2e}  & \\nodata & {:.2e}  &\\nodata \\\\ \n'.format(a,b))
        pass

    def write_footer(self):
        self.outfile.write('\\hline \n')
        self.outfile.write('\\end{tabular} \n')
        self.outfile.write('\\caption{Summary statistics for SFR, $\Delta$SFR, and stellar mass comparisons.  Bold text indicates when the two populations differ at the 99.7\% or $\ge 3\sigma$ confidence interval (KS p-value$<$3.0e-03). } \n')

        self.outfile.write('\\label{tab:stats} \n')
        self.outfile.write('\\end{table*} \n')

        # write out footer

        # close file

        pass

