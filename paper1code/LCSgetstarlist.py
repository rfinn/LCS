#!/usr/bin/env python


from LCScommon import *
import ds9
print clusternames
d=ds9.ds9()
reviewflag=1
for prefix in clusternames:

    infile='/Users/rfinn/research/LocalClusters/Images/'+prefix+'/24umWCS/'+prefix+'-WCS-mosaic_extract.tbl'

    outfile=homedir+'research/LocalClusters/PRF/'+prefix+'-starlist.tbl'
    outfile2=homedir+'research/LocalClusters/PRF/'+prefix+'-starlist.reg'
    out2=open(outfile2,'w')
    out2.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
    out2.write('fk5 \n')
    in1=open(infile,'r')
    out1=open(outfile,'w')
    ngal=0
    for line in in1:
        if line.startswith('\\'):
            out1.write(line)
        elif line.startswith('|'):
            out1.write(line)
        else:
            t=line.split()
            # select lines with 
            # (status == 1) & (SNR > 10) & ( N_PS < 2)
            #    17                 18          2
            #if (float(t[17]) == 1) & (float(t[18]) > 100.) & (float(t[2])<2):
            if  (float(t[18]) > 200.) & (float(t[18]) < 600.):# & (float(t[2])<2):
                out1.write(line)
            #    ra = 3, dec=5
                out2.write('circle(%12.8f,%12.8f,15") # color=red \n'%(float(t[3]),float(t[5])))
                ngal += 1
    print prefix, ngal 
    in1.close()
    out1.close()
    out2.close()
    if reviewflag:
        
        d.set('frame delete all')
        s='file new '+homedir+'research/LocalClusters/Images/'+prefix+'/24umWCS/'+prefix+'-WCS-mosaic_minus_median_extract.fits'

        try:
            d.set(s)
            d.set('zoom to fit')
            s='regions load '+outfile2
            d.set(s)
            print 'If you want to remove sources, in a separate window, type: \n'
            print 'emacs -nw ',outfile
        except:
            print "couldn't access: ",s
        t=raw_input('hit any key after finishing w/emacs \n')
        
