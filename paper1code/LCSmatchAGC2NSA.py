#!/usr/bin/env python

#  Written by Rose A. Finn, Feb 11, 2013
# 


import atpy, os
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column

from pylab import *
import numpy as np
from LCScommon import *
mypath=os.getcwd()
from LCSReadmasterBaseNSA import *
def findnearest(x1,y1,x2,y2,delta):#use where command
	matchflag=1
	nmatch=0
	d=sqrt((x1-x2)**2 + (y1-y2)**2)#x2 and y2 are arrays
	index=arange(len(d))
	t=index[d<delta]
	matches=t
	if len(matches) > 0:
		nmatch=len(matches)
		if nmatch > 1:
			imatch=index[(d == min(d[t]))]
		else:
			imatch=matches[0]			
	else:
		imatch = 0
		matchflag = 0

	return imatch, matchflag,nmatch

#class cluster(baseClusterNSA):
class cluster():
    def __init__(self,clustername):
        #baseClusterNSA.__init__(self,clustername)
        infile=homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSAmastertable.fits'
        self.nsa=atpy.Table(infile)
        self.prefix=clustername
        #self.readsdsscsv()
    #Get current path so program can tell if this is being run on Becky or Rose's computer
    def readsdsscsv(self):
        infile=homedir+'research/LocalClusters/NSAmastertables/SDSSTables/'+self.prefix+'_SDSS_dr7.csv'
        scat=atpy.Table(infile,type='ascii',data_start=1)
        self.sdss_run=scat.col3
        self.sdss_rerun=scat.col4
        self.sdss_camcol=scat.col5
        self.sdss_field=scat.col6
        self.sdss_rowc=scat.col17
        self.sdss_objid=scat.col16
        self.sdss_colc=scat.col15
        self.sdss_r = scat.col10
        self.sdss_ra=scat.col1
        self.sdss_dec=scat.col2


	
    def match2agcv0(self,delta):
        ''' for matching with old catalog '''
        self.zoo_objid=[]
        self.zoo_nvote=zeros(len(self.ra),'i')
        amatch=zeros(len(self.ra),'i')
        amatch_flag=zeros(len(self.ra),'i')
        anmatch=zeros(len(self.ra),'i')
        for i in range(len(self.ra)):
            imatch,matchflag,nmatch=findnearest(self.ra[i],self.dec[i],agc.adat.RA,agc.adat.DEC,delta)
            #matchflag=0
            #try:
            #    imatch=zoo.zoodict[self.sdss_objid[i]]
            #    matchflag=1
            #except:
            #    print 'no match using dictionary', i,matchflag

            if matchflag:
		    #print i,self.n.ISDSS[i],' found match to AGC catalog'
		    if nmatch > 1:
			    print 'Warning - multiple matches! nmatch = ',nmatch
			    amatch[i]=imatch[0]
		    else:
			    amatch[i]=imatch
		    amatch_flag[i]=matchflag
		    anmatch[i]=nmatch

        # write out results as a fits table that is line-matched to cluster NSA table
        lcstab=atpy.Table()
        lcstab.add_column('NSARA',self.ra,unit='deg')
        lcstab.add_column('NSADEC',self.dec,unit='deg')
        lcstab.add_column('NSAID',self.n.NSAID)
        lcstab.add_column('AGCMATCHFLAG',array(amatch_flag,'bool'))
        t=agc.adat.AGCNUMBER[amatch]*amatch_flag
        lcstab.add_column('AGCNUMBER',t)
        t=agc.adat.WHICH[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('WHICH',t,)
        t=agc.adat.RA[amatch]*amatch_flag
        lcstab.add_column('RA',t,unit='deg')
        t=agc.adat.DEC[amatch]*amatch_flag
        lcstab.add_column('DEC',t,unit='deg')
        t=agc.adat.A100[amatch]*amatch_flag
        lcstab.add_column('A100',t,unit='arcsec')
        t=agc.adat.B100[amatch]*amatch_flag
        lcstab.add_column('B100',t,unit='arcsec')
        t=agc.adat.MAG10[amatch]*amatch_flag
        lcstab.add_column('MAG10',t,unit='mag')
        t=agc.adat.INCCODE[amatch]*amatch_flag
        lcstab.add_column('INCCODE',t,unit='')
        t=agc.adat.POSANG[amatch]*amatch_flag
        lcstab.add_column('POSANG',t,unit='')
        t=agc.adat.DESCRIPTION[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('DESCRIPTION',t,unit='')
        t=agc.adat.BSTEINTYPE[amatch]*amatch_flag
        lcstab.add_column('BSTEINTYPE',t,unit='')
        t=agc.adat.VOPT[amatch]*amatch_flag
        lcstab.add_column('VOPT',t,unit='km/s')
        t=agc.adat.VERR[amatch]*amatch_flag
        lcstab.add_column('VERR',t,unit='km/s')
        t=agc.adat.EXTRC3[amatch]*amatch_flag
        lcstab.add_column('EXTRC3',t,unit='')
        t=agc.adat.EXTDIRBE[amatch]*amatch_flag
        lcstab.add_column('EXTDIRBE',t,unit='')
        t=agc.adat.VSOURCE[amatch]*amatch_flag
        lcstab.add_column('VSOURCE',t,unit='')

        t=agc.adat.NGCIC[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('NGCIC',t,unit='')
        t=agc.adat.FLUX100[amatch]*amatch_flag
        lcstab.add_column('FLUX100',t,unit='')
        t=agc.adat.RMS100[amatch]*amatch_flag
        lcstab.add_column('RMS100',t,unit='')
        t=agc.adat.V21[amatch]*amatch_flag
        lcstab.add_column('V21',t,unit='')
        t=agc.adat.WIDTH[amatch]*amatch_flag
        lcstab.add_column('WIDTH',t,unit='')
        t=agc.adat.WIDTHERR[amatch]*amatch_flag
        lcstab.add_column('WIDTHERR',t,unit='')

        t=agc.adat.TELCODE[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('TELCODE',t,unit='')
        t=agc.adat.DETCODE[amatch]*amatch_flag
        lcstab.add_column('DETCODE',t,unit='')
        t=agc.adat.HISOURCE[amatch]*amatch_flag
        lcstab.add_column('HISOURCE',t,unit='')
        t=agc.adat.STATUSCODE[amatch]*amatch_flag
        lcstab.add_column('STATUSCODE',t,unit='')
        t=agc.adat.SNRATIO[amatch]*amatch_flag
        lcstab.add_column('SNRATIO',t,unit='')
        t=agc.adat.IBANDQUAL[amatch]*amatch_flag
        lcstab.add_column('IBANDQUAL',t,unit='')
        t=agc.adat.IBANDSRC[amatch]*amatch_flag
        lcstab.add_column('IBANDSRC',t,unit='')
        t=agc.adat.IRASFLAG[amatch]*amatch_flag
        lcstab.add_column('IRASFLAG',t,unit='')
        t=agc.adat.ICLUSTER[amatch]*amatch_flag
        lcstab.add_column('ICLUSTER',t,unit='')
        t=agc.adat.HIDATA[amatch]*amatch_flag
        lcstab.add_column('HIDATA',t,unit='')
        t=agc.adat.IPOSITION[amatch]*amatch_flag
        lcstab.add_column('IPOSITION',t,unit='')
        t=agc.adat.IPALOMAR[amatch]*amatch_flag
        lcstab.add_column('IPALOMAR' ,t,unit='')
        t=agc.adat.RC3FLAG[amatch]*amatch_flag
        lcstab.add_column('RC3FLAG'  ,t,unit='')


        outfile=homedir+'research/LocalClusters/NSAmastertables/AGCTables/'+self.prefix+'_AGC.fits'
        print outfile
        if os.path.exists(outfile):
            os.remove(outfile)
        lcstab.write(outfile,type='fits')
    def match2agc(self,delta):
        ''' for matching with Becky's csv file, which has lowercase names for the columns '''
        amatch=zeros(len(self.nsa.RA),'i')
        amatch_flag=zeros(len(self.nsa.RA),'i')
        anmatch=zeros(len(self.nsa.RA),'i')
        for i in range(len(self.nsa.RA)):
            imatch,matchflag,nmatch=findnearest(self.nsa.RA[i],self.nsa.DEC[i],agc.adat.radeg,agc.adat.decdeg,delta)
            #matchflag=0
            #try:
            #    imatch=zoo.zoodict[self.sdss_objid[i]]
            #    matchflag=1
            #except:
            #    print 'no match using dictionary', i,matchflag

            if matchflag:
		    #print i,self.n.ISDSS[i],' found match to AGC catalog'
		    if nmatch > 1:
			    print 'Warning - multiple matches! nmatch = ',nmatch
			    amatch[i]=imatch[0]
		    else:
			    amatch[i]=imatch
		    amatch_flag[i]=matchflag
		    anmatch[i]=nmatch

        # write out results as a fits table that is line-matched to cluster NSA table
        lcstab=atpy.Table()
        lcstab.add_column('NSARA',self.nsa.RA,unit='deg')
        lcstab.add_column('NSADEC',self.nsa.DEC,unit='deg')
        lcstab.add_column('NSAID',self.nsa.NSAID)
        lcstab.add_column('AGCMATCHFLAG',array(amatch_flag,'bool'))
        t=agc.adat.AGCnr[amatch]*amatch_flag
        lcstab.add_column('AGCNUMBER',t)
        t=agc.adat.which[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('WHICH',t,)
        t=agc.adat.radeg[amatch]*amatch_flag
        lcstab.add_column('RA',t,unit='deg')
        t=agc.adat.decdeg[amatch]*amatch_flag
        lcstab.add_column('DEC',t,unit='deg')
        t=agc.adat.a[amatch]*amatch_flag
        lcstab.add_column('A100',t,unit='arcsec')
        t=agc.adat.b[amatch]*amatch_flag
        lcstab.add_column('B100',t,unit='arcsec')
        t=agc.adat.zmag[amatch]*amatch_flag
        lcstab.add_column('MAG10',t,unit='mag')
        t=agc.adat.inccode[amatch]*amatch_flag
        lcstab.add_column('INCCODE',t,unit='')
        t=agc.adat.posang[amatch]*amatch_flag
        lcstab.add_column('POSANG',t,unit='')
        t=agc.adat.description[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('DESCRIPTION',t,unit='')
        t=agc.adat.bsteintype[amatch]*amatch_flag
        lcstab.add_column('BSTEINTYPE',t,unit='')
        t=agc.adat.vopt[amatch]*amatch_flag
        lcstab.add_column('VOPT',t,unit='km/s')
        t=agc.adat.verr[amatch]*amatch_flag
        lcstab.add_column('VERR',t,unit='km/s')
        t=agc.adat.extrc3[amatch]*amatch_flag
        lcstab.add_column('EXTRC3',t,unit='')
        t=agc.adat.extdirbe[amatch]*amatch_flag
        lcstab.add_column('EXTDIRBE',t,unit='')
        t=agc.adat.vsource[amatch]*amatch_flag
        lcstab.add_column('VSOURCE',t,unit='')

        t=agc.adat.ngcic[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('NGCIC',t,unit='')
        t=agc.adat.hiflux[amatch]*amatch_flag
        lcstab.add_column('FLUX100',t,unit='')
        t=agc.adat.rms[amatch]*amatch_flag
        lcstab.add_column('RMS100',t,unit='')
        t=agc.adat.v21[amatch]*amatch_flag
        lcstab.add_column('V21',t,unit='')
        t=agc.adat.width[amatch]*amatch_flag
        lcstab.add_column('WIDTH',t,unit='')
        t=agc.adat.widtherr[amatch]*amatch_flag
        lcstab.add_column('WIDTHERR',t,unit='')

        t=agc.adat.telcode[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        lcstab.add_column('TELCODE',t,unit='')
        t=agc.adat.detcode[amatch]*amatch_flag
        lcstab.add_column('DETCODE',t,unit='')
        t=agc.adat.hisource[amatch]*amatch_flag
        lcstab.add_column('HISOURCE',t,unit='')
        t=agc.adat.statuscode[amatch]*amatch_flag
        lcstab.add_column('STATUSCODE',t,unit='')
        t=agc.adat.snr[amatch]*amatch_flag
        lcstab.add_column('SNRATIO',t,unit='')
        t=agc.adat.ibandqual[amatch]*amatch_flag
        lcstab.add_column('IBANDQUAL',t,unit='')
        t=agc.adat.ibandsrc[amatch]*amatch_flag
        lcstab.add_column('IBANDSRC',t,unit='')
        t=agc.adat.irasflag[amatch]*amatch_flag
        lcstab.add_column('IRASFLAG',t,unit='')
        t=agc.adat.icluster[amatch]*amatch_flag
        lcstab.add_column('ICLUSTER',t,unit='')
        t=agc.adat.hidata[amatch]*amatch_flag
        lcstab.add_column('HIDATA',t,unit='')
        t=agc.adat.iposition[amatch]*amatch_flag
        lcstab.add_column('IPOSITION',t,unit='')
        t=agc.adat.ipalomar[amatch]*amatch_flag
        lcstab.add_column('IPALOMAR' ,t,unit='')
        t=agc.adat.rc3flag[amatch]*amatch_flag
        lcstab.add_column('RC3FLAG'  ,t,unit='')


        outfile=homedir+'research/LocalClusters/NSAmastertables/AGCTables/'+self.prefix+'_AGC.fits'
        print outfile
        if os.path.exists(outfile):
            os.remove(outfile)
        lcstab.write(outfile,type='fits')
    def match2agcshort(self,delta):
        ''' for matching with Becky's csv file, which has lowercase names for the columns '''
        amatch=zeros(len(self.nsa.RA),'i')
        amatch_flag=zeros(len(self.nsa.RA),'i')
        anmatch=zeros(len(self.nsa.RA),'i')
        for i in range(len(self.nsa.RA)):
            imatch,matchflag,nmatch=findnearest(self.nsa.RA[i],self.nsa.DEC[i],agc.adat.radeg,agc.adat.decdeg,delta)
            #matchflag=0
            #try:
            #    imatch=zoo.zoodict[self.sdss_objid[i]]
            #    matchflag=1
            #except:
            #    print 'no match using dictionary', i,matchflag

            if matchflag:
		    #print i,self.n.ISDSS[i],' found match to AGC catalog'
		    if nmatch > 1:
			    print 'Warning - multiple matches! nmatch = ',nmatch
			    amatch[i]=imatch[0]
		    else:
			    amatch[i]=imatch
		    amatch_flag[i]=matchflag
		    anmatch[i]=nmatch

        # write out results as a fits table that is line-matched to cluster NSA table
        lcstab=atpy.Table()
        lcstab.add_column('NSARA',self.nsa.RA,unit='deg')
        lcstab.add_column('NSADEC',self.nsa.DEC,unit='deg')
        lcstab.add_column('NSAID',self.nsa.NSAID)
        lcstab.add_column('AGCMATCHFLAG',array(amatch_flag,'bool'))
        t=agc.adat.AGCnr[amatch]*amatch_flag
        lcstab.add_column('AGCNUMBER',t)
        #t=agc.adat.which[amatch]
        #if sum(amatch_flag) < len(amatch):
        #    for i in range(len(t)):
        #        if ~amatch_flag[i]:
        #            t[i]=0
        #lcstab.add_column('WHICH',t,)
        t=agc.adat.radeg[amatch]*amatch_flag
        lcstab.add_column('RA',t,unit='deg')
        t=agc.adat.decdeg[amatch]*amatch_flag
        lcstab.add_column('DEC',t,unit='deg')
        t=agc.adat.a[amatch]*amatch_flag
        lcstab.add_column('A100',t,unit='arcsec')
        t=agc.adat.b[amatch]*amatch_flag
        lcstab.add_column('B100',t,unit='arcsec')
        t=agc.adat.zmag[amatch]*amatch_flag
        lcstab.add_column('MAG10',t,unit='mag')
        #t=agc.adat.inccode[amatch]*amatch_flag
        #lcstab.add_column('INCCODE',t,unit='')
        #t=agc.adat.posang[amatch]*amatch_flag
        #lcstab.add_column('POSANG',t,unit='')
        #t=agc.adat.description[amatch]
        #if sum(amatch_flag) < len(amatch):
        #    for i in range(len(t)):
        #        if ~amatch_flag[i]:
        #            t[i]=0
        #lcstab.add_column('DESCRIPTION',t,unit='')
        #t=agc.adat.bsteintype[amatch]*amatch_flag
        #lcstab.add_column('BSTEINTYPE',t,unit='')
        vopt=np.array(agc.adat.vopt,'f')
        t=vopt[amatch]*amatch_flag
        lcstab.add_column('VOPT',t,unit='km/s')
        #t=agc.adat.verr[amatch]*amatch_flag
        #lcstab.add_column('VERR',t,unit='km/s')
        #t=agc.adat.extrc3[amatch]*amatch_flag
        #lcstab.add_column('EXTRC3',t,unit='')
        #t=agc.adat.extdirbe[amatch]*amatch_flag
        #lcstab.add_column('EXTDIRBE',t,unit='')
        #t=agc.adat.vsource[amatch]*amatch_flag
        #lcstab.add_column('VSOURCE',t,unit='')

        #t=agc.adat.ngcic[amatch]
        #if sum(amatch_flag) < len(amatch):
        #    for i in range(len(t)):
        #        if ~amatch_flag[i]:
        #            t[i]=0
        #lcstab.add_column('NGCIC',t,unit='')
        t=agc.adat.hiflux[amatch]*amatch_flag
        lcstab.add_column('FLUX100',t,unit='')
        t=agc.adat.rms[amatch]*amatch_flag
        lcstab.add_column('RMS100',t,unit='')
        t=agc.adat.v21[amatch]*amatch_flag
        lcstab.add_column('V21',t,unit='')
        t=agc.adat.width[amatch]*amatch_flag
        lcstab.add_column('WIDTH',t,unit='')
        t=agc.adat.widtherr[amatch]*amatch_flag
        lcstab.add_column('WIDTHERR',t,unit='')

        #t=agc.adat.telcode[amatch]
        #if sum(amatch_flag) < len(amatch):
        #    for i in range(len(t)):
        #        if ~amatch_flag[i]:
        #            t[i]=0
        #lcstab.add_column('TELCODE',t,unit='')
        #t=agc.adat.detcode[amatch]*amatch_flag
        #lcstab.add_column('DETCODE',t,unit='')
        t=agc.adat.hisrc[amatch]*amatch_flag
        lcstab.add_column('HISOURCE',t,unit='')
        #t=agc.adat.statuscode[amatch]*amatch_flag
        #lcstab.add_column('STATUSCODE',t,unit='')
        t=agc.adat.snr[amatch]*amatch_flag
        lcstab.add_column('SNRATIO',t,unit='')
        #t=agc.adat.ibandqual[amatch]*amatch_flag
        #lcstab.add_column('IBANDQUAL',t,unit='')
        #t=agc.adat.ibandsrc[amatch]*amatch_flag
        #lcstab.add_column('IBANDSRC',t,unit='')
        #t=agc.adat.irasflag[amatch]*amatch_flag
        #lcstab.add_column('IRASFLAG',t,unit='')
        #t=agc.adat.icluster[amatch]*amatch_flag
        #lcstab.add_column('ICLUSTER',t,unit='')
        #t=agc.adat.hidata[amatch]*amatch_flag
        #lcstab.add_column('HIDATA',t,unit='')
        #t=agc.adat.iposition[amatch]*amatch_flag
        #lcstab.add_column('IPOSITION',t,unit='')
        #t=agc.adat.ipalomar[amatch]*amatch_flag
        #lcstab.add_column('IPALOMAR' ,t,unit='')
        #t=agc.adat.rc3flag[amatch]*amatch_flag
        #lcstab.add_column('RC3FLAG'  ,t,unit='')


        outfile=homedir+'research/LocalClusters/NSAmastertables/AGCTables/'+self.prefix+'_AGCshort.fits'
        print outfile
        if os.path.exists(outfile):
            os.remove(outfile)
        lcstab.write(outfile,type='fits')

    def match2agcv2(self,delta):
	# uses astropy.io.fits instead of atpy
        self.zoo_objid=[]
        self.zoo_nvote=zeros(len(self.nsa.RA),'i')
        amatch=zeros(len(self.nsa.RA),'i')
        amatch_flag=zeros(len(self.nsa.RA),'i')
        anmatch=zeros(len(self.nsa.RA),'i')
        for i in range(len(self.nsa.RA)):
            imatch,matchflag,nmatch=findnearest(self.nsa.RA[i],self.nsa.DEC[i],agc.adat.RA,agc.adat.DEC,delta)
            #matchflag=0
            #try:
            #    imatch=zoo.zoodict[self.sdss_objid[i]]
            #    matchflag=1
            #except:
            #    print 'no match using dictionary', i,matchflag

            if matchflag:
		    #print i,self.n.ISDSS[i],' found match to AGC catalog'
		    if nmatch > 1:
			    print 'Warning - multiple matches! nmatch = ',nmatch
			    amatch[i]=imatch[0]
		    else:
			    amatch[i]=imatch
		    amatch_flag[i]=matchflag
		    anmatch[i]=nmatch

        # write out results as a fits table that is line-matched to cluster NSA table
        lcstab=Table()
	#newcol=Column(data=np.array(newcol,'i'),name=col)

        col0 = fits.Column(name='NSAID',array=self.nsa.NSAID,format='J')
        col0 = fits.Column(name='NSARA',array=self.nsa.RA,unit='deg',format='E')
        col1 = fits.Column(name='NSADEC',array=self.nsa.DEC,unit='deg',format='E')

        col3 = fits.Column(name='AGCMATCHFLAG',array=array(amatch_flag,'bool'),format='L')
        t=agc.adat.AGCNUMBER[amatch]*amatch_flag
        col4 = fits.Column(name='AGCNUMBER',array=t,format='J')
        t=agc.adat.WHICH[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        col5 = fits.Column(name='WHICH',array=t,format='J')
        t=agc.adat.RA[amatch]*amatch_flag
        col6 = fits.Column(name='RA',array=t,unit='deg',format='E')
        t=agc.adat.DEC[amatch]*amatch_flag
        col7 = fits.Column(name='DEC',array=t,unit='deg',format='E')
        t=agc.adat.A100[amatch]*amatch_flag
        col8 = fits.Column(name='A100',array=t,unit='arcsec',format='J')
        t=agc.adat.B100[amatch]*amatch_flag
        col9 = fits.Column(name='B100',array=t,unit='arcsec',format='J')
        t=agc.adat.MAG10[amatch]*amatch_flag
        col10 = fits.Column(name='MAG10',array=t,unit='mag',format='J')
        t=agc.adat.INCCODE[amatch]*amatch_flag
        col11 = fits.Column(name='INCCODE',array=t,unit='',format='J')
        t=agc.adat.POSANG[amatch]*amatch_flag
        col12 = fits.Column(name='POSANG',array=t,unit='',format='J')
        t=agc.adat.DESCRIPTION[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        col13 = fits.Column(name='DESCRIPTION',array=t,unit='',format='A8')
        t=agc.adat.BSTEINTYPE[amatch]*amatch_flag
        col14 = fits.Column(name='BSTEINTYPE',array=t,unit='',format='J')
        t=agc.adat.VOPT[amatch]*amatch_flag
        col15 = fits.Column(name='VOPT',array=t,unit='km/s',format='J')
        t=agc.adat.VERR[amatch]*amatch_flag
        col16 = fits.Column(name='VERR',array=t,unit='km/s',format='J')
        t=agc.adat.EXTRC3[amatch]*amatch_flag
        col17 = fits.Column(name='EXTRC3',array=t,unit='',format='J')
        t=agc.adat.EXTDIRBE[amatch]*amatch_flag
        col18 = fits.Column(name='EXTDIRBE',array=t,unit='',format='J')
        t=agc.adat.VSOURCE[amatch]*amatch_flag
        col19 = fits.Column(name='VSOURCE',array=t,unit='',format='J')

        t=agc.adat.NGCIC[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        col20 = fits.Column(name='NGCIC',array=t,unit='',format='A10')
        t=agc.adat.FLUX100[amatch]*amatch_flag
        col21 = fits.Column(name='FLUX100',array=t,unit='',format='J')
        t=agc.adat.RMS100[amatch]*amatch_flag
        col22 = fits.Column(name='RMS100',array=t,unit='',format='J')
        t=agc.adat.V21[amatch]*amatch_flag
        col23 = fits.Column(name='V21',array=t,unit='',format='J')
        t=agc.adat.WIDTH[amatch]*amatch_flag
        col24 = fits.Column(name='WIDTH',array=t,unit='',format='J')
        t=agc.adat.WIDTHERR[amatch]*amatch_flag
        col25 = fits.Column(name='WIDTHERR',array=t,unit='',format='J')

        t=agc.adat.TELCODE[amatch]
        if sum(amatch_flag) < len(amatch):
            for i in range(len(t)):
                if ~amatch_flag[i]:
                    t[i]=0
        col26 = fits.Column(name='TELCODE',array=t,unit='',format='A8')
        t=agc.adat.DETCODE[amatch]*amatch_flag
        col27 = fits.Column(name='DETCODE',array=t,unit='',format='J')
        t=agc.adat.HISOURCE[amatch]*amatch_flag
        col28 = fits.Column(name='HISOURCE',array=t,unit='',format='I')
        t=agc.adat.STATUSCODE[amatch]*amatch_flag
        col29 = fits.Column(name='STATUSCODE',array=t,unit='',format='I')
        t=agc.adat.SNRATIO[amatch]*amatch_flag
        col30 = fits.Column(name='SNRATIO',array=t,unit='',format='I')
        t=agc.adat.IBANDQUAL[amatch]*amatch_flag
        col31 = fits.Column(name='IBANDQUAL',array=t,unit='',format='I')
        t=agc.adat.IBANDSRC[amatch]*amatch_flag
        col32 = fits.Column(name='IBANDSRC',array=t,unit='',format='I')
        t=agc.adat.IRASFLAG[amatch]*amatch_flag
        col33 = fits.Column(name='IRASFLAG',array=t,unit='',format='I')
        t=agc.adat.ICLUSTER[amatch]*amatch_flag
        col34 = fits.Column(name='ICLUSTER',array=t,unit='',format='I')
        t=agc.adat.HIDATA[amatch]*amatch_flag
        col35 = fits.Column(name='HIDATA',array=t,unit='',format='I')
        t=agc.adat.IPOSITION[amatch]*amatch_flag
        col36 = fits.Column(name='IPOSITION',array=t,unit='',format='I')
        t=agc.adat.IPALOMAR[amatch]*amatch_flag
        col37 = fits.Column(name='IPALOMAR',array=t,unit='',format='I')
        t=agc.adat.RC3FLAG[amatch]*amatch_flag
        col38 = fits.Column(name='RC3FLAG',array=t,unit='',format='I')

        cols=fits.ColDefs([col0,col1,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38])
        outfile=homedir+'research/LocalClusters/NSAmastertables/AGCTables/'+self.prefix+'_AGC.fits'
        if os.path.exists(outfile):
            os.remove(outfile)

        tbhdu=fits.new_table(cols)
        #hdu=tbhdu.header
        #thdulist=fits.HDUList([hdu,tbhdu])
        #thdulist.writeto(output24)
        tbhdu.writeto(outfile)


class agc:
    def __init__(self):
        # infile=homedir+'research/NSA/nsa_v0_1_2.fits'
        infile=homedir+'idl/programs/idl_alfa/agcnorthLCS.fits'
        infile=homedir+'idl/programs/idl_alfa/agctotal.fits'
        infile=homedir+'research/AGC/agcnorthminus1.fits'
        infile=homedir+'research/AGC/agcnorthLCSregion.fits'
        #infile=homedir+'research/AGC/agcnorthminus1.July2015.short.fits'
        self.adat=atpy.Table(infile,type='fits')
        try:
            self.agcdict=dict((a,b) for a,b in zip(self.adat.AGCnr,arange(len(self.adat.AGCnr))))
        except AttributeError:
            self.agcdict=dict((a,b) for a,b in zip(self.adat.AGCNUMBER,arange(len(self.adat.AGCNUMBER))))            



agc=agc()
# match radius = 3"/3600 -> deg
delta=3./3600. 
#mkw11=cluster('MKW11')
#mkw11.match2zoo(delta)
myclusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
#myclusternames=['MKW11']
for cname in myclusternames:
    cl=cluster(cname)
    print '\n',cl.prefix, '\n'
    #cl.match2agcshort(delta)
    cl.match2agcv2(delta) # uses pyfits to write table


