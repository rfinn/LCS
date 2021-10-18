
from astropy.table import Table

# read in table that was matched to LCSsize

intab = Table.read('LCS-GSWLC-NODR10AGN-Simard-tab1-tab3-A100-LCSsizes.fits')
# rename columns for backwards compatibility
#intab.rename_column('__B_T_r','__B_T_r2')
#intab.rename_column('(B/T)r','__B_T_r')
#intab.rename_column('(B/T)g','__B_T_g')
intab.rename_column('DR_R200_1','DR_R200')
#intab.rename_column('Rd_1','Rd')
#intab.rename_column('RA_1','RA')
#intab.rename_column('DEC_1','DEC')    
intab.write('LCS-GSWLC-NODR10AGN-Simard-tab1-tab3-A100-LCSsizes-fixednames.fits',format='fits',overwrite=True)
