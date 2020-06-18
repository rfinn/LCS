#!/usr/bin/env python
'''
GOAL:
- match NSA with galaxy zoo data

USEAGE:
- Run from within ipython or on the command line
- the program will automatically run on all the clusters listed in myclusternames


NOTES:

07.07.15
--------
found issue with MKW11, like it didn't match correctly.
The galaxy NSA=169994 was not liksted as a spiral although it is
clearly a spiral.  Trying to figure out what is going on.

'''

#  Written by Rose A. Finn, Feb 11, 2013
# 
#  Updated 07.07.15

import atpy, os
from pylab import *
from LCScommon import *
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits

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

class cluster(baseClusterNSA):
    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)
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


    def match2zoo(self,delta):

        self.zoo_objid=[]
        self.zoo_RA=zeros(len(self.ra),'f')
        self.zoo_DEC=zeros(len(self.ra),'f')
        self.zoo_nvote=zeros(len(self.ra),'i')
        self.zoo_pel=zeros(len(self.ra),'f')
        self.zoo_pcw=zeros(len(self.ra),'f')
        self.zoo_pacw=zeros(len(self.ra),'f')
        self.zoo_pedge=zeros(len(self.ra),'f')
        self.zoo_pdk=zeros(len(self.ra),'f')
        self.zoo_pmg=zeros(len(self.ra),'f')
        self.zoo_pcs=zeros(len(self.ra),'f')
        self.zoo_p_el_debiased=zeros(len(self.ra),'f')
        self.zoo_p_cs_debiased=zeros(len(self.ra),'f')
        self.zoo_spiral=zeros(len(self.ra),'i')
        self.zoo_elliptical=zeros(len(self.ra),'i')
        self.zoo_uncertain=zeros(len(self.ra),'i')
        izoo_match=zeros(len(self.ra),'i')
        izoo_match_flag=zeros(len(self.ra),'i')
        izoophot_match=zeros(len(self.ra),'i')
        izoophot_match_flag=zeros(len(self.ra),'i')

        for i in range(len(self.ra)):
            imatch,matchflag,nmatch=findnearest(self.ra[i],self.dec[i],zoo.zRA,zoo.zDEC,delta)
            #matchflag=0
            #try:
            #    imatch=zoo.zoodict[self.sdss_objid[i]]
            #    matchflag=1
            #except:
            #    print 'no match using dictionary', i,matchflag

            if matchflag:
                print i,self.n.ISDSS[i],' found match to galaxy zoo spec sample dat'
                izoo_match[i]=imatch
                izoo_match_flag[i]=matchflag
                self.zoo_objid.append(zoo.zdat.OBJID[imatch])
                self.zoo_nvote[i]=zoo.zdat.NVOTE[imatch]
                self.zoo_pel[i]=zoo.zdat.P_EL[imatch]
                self.zoo_pcw[i]=zoo.zdat.P_CW[imatch]
                self.zoo_pacw[i]=zoo.zdat.P_ACW[imatch]
                self.zoo_pedge[i]=zoo.zdat.P_EDGE[imatch]
                self.zoo_pdk[i]=zoo.zdat.P_DK[imatch]
                self.zoo_pmg[i]=zoo.zdat.P_MG[imatch]
                self.zoo_pcs[i]=zoo.zdat.P_CS[imatch]
                self.zoo_p_el_debiased[i]=zoo.zdat.P_EL_DEBIASED[imatch]
                self.zoo_p_cs_debiased[i]=zoo.zdat.P_CS_DEBIASED[imatch]
                self.zoo_spiral[i]=zoo.zdat.SPIRAL[imatch]
                self.zoo_elliptical[i]=zoo.zdat.ELLIPTICAL[imatch]
                self.zoo_uncertain[i]=zoo.zdat.UNCERTAIN[imatch]
                self.zoo_RA[i]=zoo.zRA[imatch]
                self.zoo_DEC[i]=zoo.zDEC[imatch]
            else:
                #imatch,matchflag,nmatch=findnearest(self.ra[i],self.dec[i],zoo.zphotRA,zoo.zphotDEC,delta)
                imatch,matchflag,nmatch=findnearest(self.ra[i],self.dec[i],zoo.zphotRA,zoo.zphotDEC,delta)
                print 'no match to spec cat - got here',i,self.n.ISDSS[i]

                #try:
                #    imatch=zoo.zoophotdict[self.sdss_objid[i]]
                #    print 'found a match using phot dictionary'
                #    matchflag=1
                #except:
                #     print 'no match using phot dictionary', i

                if matchflag:
                    izoophot_match[i]=imatch
                    izoophot_match_flag[i]=matchflag
                    self.zoo_objid.append(zoo.zphotdat.OBJID[imatch])
                    self.zoo_nvote[i]=zoo.zphotdat.NVOTE[imatch]
                    self.zoo_pel[i]=zoo.zphotdat.P_EL[imatch]
                    self.zoo_pcw[i]=zoo.zphotdat.P_CW[imatch]
                    self.zoo_pacw[i]=zoo.zphotdat.P_ACW[imatch]
                    self.zoo_pedge[i]=zoo.zphotdat.P_EDGE[imatch]
                    self.zoo_pdk[i]=zoo.zphotdat.P_DK[imatch]
                    self.zoo_pmg[i]=zoo.zphotdat.P_MG[imatch]
                    self.zoo_pcs[i]=zoo.zphotdat.P_CS[imatch]
                    self.zoo_RA[i]=zoo.zRA[imatch]
                    self.zoo_DEC[i]=zoo.zDEC[imatch]

                else:
                    self.zoo_objid.append('null')

        # write out results as a fits table that is line-matched to cluster NSA table
        ztab=atpy.Table()
        zooflag=izoo_match_flag | izoophot_match_flag
        ztab.add_column('NSAID',self.n.NSAID)
        ztab.add_column('zooRA',self.zoo_RA)
        ztab.add_column('zooDEC',self.zoo_DEC)
        ztab.add_column('match_flag',zooflag,dtype='bool')
        ztab.add_column('spec_match_flag',izoo_match_flag,dtype='bool')
        ztab.add_column('match_index',izoo_match)
        ztab.add_column('phot_match_flag',izoophot_match_flag,dtype='bool')
        ztab.add_column('phot_match_index',izoophot_match)
        #ztab.add_column('OBJID',self.zoo_objid,dtype='|S16')
        ztab.add_column('nvote',self.zoo_nvote)
        ztab.add_column('p_el',self.zoo_pel)
        ztab.add_column('p_cw',self.zoo_pcw)
        ztab.add_column('p_acw',self.zoo_pacw)
        ztab.add_column('p_edge',self.zoo_pedge)
        ztab.add_column('p_dk',self.zoo_pdk)
        ztab.add_column('p_mg',self.zoo_pmg)
        ztab.add_column('p_cs',self.zoo_pcs)
        ztab.add_column('p_el_debiased',self.zoo_p_el_debiased)
        ztab.add_column('p_cs_debiased',self.zoo_p_cs_debiased)
        ztab.add_column('p_spiral',self.zoo_spiral)
        ztab.add_column('p_elliptical',self.zoo_elliptical)
        ztab.add_column('p_uncertain',self.zoo_uncertain)
        outfile=homedir+'research/LocalClusters/NSAmastertables/GalaxyZooTables/'+self.prefix+'_GalaxyZoo.fits'
        if os.path.exists(outfile):
            os.remove(outfile)
        ztab.write(outfile)

class Gzoo:
    def __init__(self):
        # infile=homedir+'research/NSA/nsa_v0_1_2.fits'
        infile=homedir+'research/GalaxyZoo/GalaxyZoo1_DR_table2_LCSregion.fits'
        self.zdat=atpy.Table(infile,type='fits')
        infile=homedir+'research/GalaxyZoo/GalaxyZoo1_DR_table3_LCSregion.fits'
        self.zphotdat=atpy.Table(infile,type='fits')
        self.zoodict=dict((a,b) for a,b in zip(self.zdat.OBJID,arange(len(self.zdat.OBJID))))
        self.zoophotdict=dict((a,b) for a,b in zip(self.zphotdat.OBJID,arange(len(self.zphotdat.OBJID))))
        self.convert_angles()
    def convert_angles(self):
        #
        # updating 07/07/15 to make use of astropy
        #
        #
        #RA = coord.Angle(self.zdat.RA,unit=u.hour)
        #DEC = coord.Angle(self.zdat.DEC,unit=u.degree)
        #self.zRA=RA.degree
        #self.zDEC=DEC.degree
        #RA = coord.Angle(self.zphotdat.RA,unit=u.hour)
        #DEC = coord.Angle(self.zphotdat.DEC,unit=u.degree)
        #self.zphotRA=RA.degree
        #self.zphotDEC=DEC.degree
        #
        self.zRA=zeros(len(self.zdat.RA),'d')
        self.zDEC=zeros(len(self.zdat.RA),'d')
        for i in range(len(self.zdat.RA)):
            r=self.zdat.RA[i].split(':')
            self.zRA[i]=(float(r[0])+float(r[1])/60.+float(r[2])/3600.)*15
            d=self.zdat.DEC[i].split(':')
            self.zDEC[i]=(abs(float(d[0]))+float(d[1])/60.+float(d[2])/3600.)
            if float(d[0]) < 0.:
                self.zDEC[i] = -1.*self.zDEC[i]

        self.zphotRA=zeros(len(self.zphotdat.RA),'d')
        self.zphotDEC=zeros(len(self.zphotdat.RA),'d')
        for i in range(len(self.zphotdat.RA)):
            r=self.zphotdat.RA[i].split(':')
            self.zphotRA[i]=(float(r[0])+float(r[1])/60.+float(r[2])/3600.)*15
            d=self.zphotdat.DEC[i].split(':')
            self.zphotDEC[i]=(abs(float(d[0]))+float(d[1])/60.+float(d[2])/3600.)
            if float(d[0]) < 0.:
                self.zphotDEC[i] = -1.*self.zphotDEC[i]
if __name__ == 'main':
    zoo=Gzoo()
    # match radius = 3"/3600 -> deg
    delta=3./3600. 
    mkw11=cluster('MKW11')
    #mkw11.match2zoo(delta)
    myclusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
    #myclusternames=['MKW11']
    for cname in myclusternames:
        cl=cluster(cname)
        print '\n',cl.prefix, '\n'
        cl.match2zoo(delta)
