#!/usr/bin/env python
'''
goal is to write latex figure commands for all stars in each LCS field that were analyzed by galfit

the output is going in galfitsim.tex

'''
import os
import glob
from LCScommon import *

for cl in clusternames:
    print cl
    working_dir=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'
    os.chdir(working_dir)
    figname=homedir+'research/LocalClusters/GalfitAnalysis/CutoutPlots/'+cl+'*-galfitResults.eps'
    files=glob.glob(figname)
    narray=len(files)
    print 'number of files = ',narray
    ncount=0
    outfile='/Users/rfinn/Dropbox/Research/MyPapers/LCSpaper1/'+cl+'galfitstars.tex'
    out1=open(outfile,'w')
    out1.write('\\begin{figure*}[h]\n')
    out1.write('\includegraphics[width=\\textwidth]{/Users/rfinn/research/LocalClusters/SamplePlots/'+cl+'galfitstars.eps}\n')
    out1.write('\caption{GALFIT vs SE parameters for stars in field of '+cl+'.}\n')
    out1.write('\label{'+cl+'galfitstars} \n')
    out1.write('\end{figure*} \n')

    out1.write('\\begin{figure*}[h] \n')
    for file in files:
        
        out1.write('\includegraphics[width=\\textwidth]{'+file+'} \n')

        if ncount == 2:
            out1.write('\end{figure*} \n')
            #out1.write('\clearpage \n')
            out1.write('\\begin{figure*}[h] \n')
            ncount = -1
        ncount += 1
    out1.write('\end{figure*} \n')
        # take modulo of nfiles % 3.  If not = 0, write out an \end{figure} statement
    out1.close()
