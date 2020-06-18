#!/usr/bin/env python

# make subsets of AGC that contains galaxies w/in 3 deg of LCS centers


import ReadAGCsav
from pylab import *
from LCScommon import *


agc=ReadAGCsav.agc()

for cl in clusternames:

    racenter=clusterRA[cl]
    deccenter=clusterDec[cl]

    dagc=sqrt((racenter-agc.radeg)**2+(deccenter-agc.decdeg)**2)
    keepflag=(dagc<3)#agc galaxis w/in 3 degrees from cluster center

    agcnumber=agc.agcnumber[keepflag]
    which=agc.which[keepflag]
    radeg=agc.radeg[keepflag]
    decdeg=agc.decdeg[keepflag]
    a100=agc.a100[keepflag]
    b100=agc.b100[keepflag]
    mag10=agc.mag10[keepflag]
    inccode=agc.inccode[keepflag]
    posang=agc.posang[keepflag]
    description=agc.description[keepflag]
    bsteintype=agc.bsteintype[keepflag]
    vopt=agc.vopt[keepflag]
    verr=agc.verr[keepflag]
    extrc3=agc.extrc3[keepflag]
    extdirbe=agc.extdirbe[keepflag]
    vsource=agc.vsource[keepflag]
    ngcic=agc.ngcic[keepflag]
    flux100=agc.flux100[keepflag]
    rms100=agc.rms100[keepflag]
    v21=agc.v21[keepflag]
    width=agc.width[keepflag]
    widtherr=agc.widtherr[keepflag]
    telcode=agc.telcode[keepflag]
    detcode=agc.detcode[keepflag]
    hisource=agc.hisource[keepflag]
    statuscode=agc.statuscode[keepflag]
    snratio=agc.snratio[keepflag]
    ibandqual=agc.ibandqual[keepflag]
    ibandsrc=agc.ibandsrc[keepflag]
    irasflag=agc.irasflag[keepflag]
    icluster=agc.icluster[keepflag]
    hidata=agc.hidata[keepflag]
    iposition=agc.iposition[keepflag]
    ipalomar=agc.ipalomar[keepflag]
    rc3flag=agc.rc3flag[keepflag]
    agcdict=dict((a,b) for a,b in zip(agc.agcnumber,arange(len(agc.agcnumber))))

    outfile=homedir+'research/LocalClusters/AGCfiles/'+cl+'_AGC.fits'
    if os.path.exists(outfile):
        os.remove(outfile)
    # write out table
    self.ndatsub.write(outfile)

    
