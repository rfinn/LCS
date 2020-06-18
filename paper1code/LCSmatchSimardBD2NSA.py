#!/usr/bin/env python

#  Written by Rose A. Finn, Nov 9, 2013
# 
'''
goal is to match Luc Simard's B/D GIM2D catalog with my NSA catalogs.
Simard catalog is in research/SimardSDSS2011/table1and3.fits
this has 108 columns!!!
I would like to figure out a way to do this without have to write out every column
individually.  will sleep on it and look at this again in the morning.

G'night


------------------------------------------------------------
catalog description
------------------------------------------------------------
from http://vizier.cfa.harvard.edu/viz-bin/Cat?J/ApJS/196/11
------------------------------------------------------------

Byte-by-byte Description of file: table[12].dat

   Bytes Format Units       Label   Explanations

   1- 18  A18   ---         objID   SDSS "objID" object identifier (run, rerun,
                                    camcol, field, object)
  20- 29  F10.6 ---         z       ?=-99.99 SDSS redshift (spectroscopic if
                                    available; or photometric ("-1" in "Sp")
  31- 32  I2    ---         Sp      [-2/6] SDSS SpecClass value (G1)
  34- 40  F7.3  kpc/arcsec  Scale   ?=-99.99 Physical scale at redshift z
  42- 51  F10.3 Mpc+3       Vmax    ?=-99.99 Galaxy volume correction (Eq. 7)
  53- 58  F6.2  mag         gg2d    ?=-99.99 GIM2D B+D model g-band magnitude;
                                    (Eq. 1)
  60- 65  F6.2  mag         e_gg2d  ?=-99.99 Uncertainty in gg2d
  67- 72  F6.2  mag         rg2d    ?=-99.99 GIM2D B+D model r-band magnitude;
                                    (Eq. 1)
  74- 79  F6.2  mag         e_rg2d  ?=-99.99 Uncertainty in rg2d
  81- 86  F6.2  mag         gg2df   ?=-99.99 B+D model g-band fiber magnitude
  88- 93  F6.2  mag         rg2df   ?=-99.99 B+D model r-band fiber magnitude
  95-100  F6.2  mag         dCol    [-4.60,+3.83]?=-99.99 Delta fiber color (G2)
 102-107  F6.2  ---         (B/T)g  [0,1]?=-99.99 g-band bulge fraction
 109-114  F6.2  ---       e_(B/T)g  [0,1]?=-99.99 Uncertainty in (B/T)g
 116-121  F6.2  ---         (B/T)r  [0,1]?=-99.99 r-band bulge fraction
 123-128  F6.2  ---       e_(B/T)r  [0,1]?=-99.99 Uncertainty in (B/T)r
 130-135  F6.2  ---        (B/T)gf  ?=-99.99 g-band fiber bulge fraction
 137-142  F6.2  ---        (B/T)rf  ?=-99.99 r-band fiber bulge fraction
 144-150  F7.2  kpc         Rhlg    [0,1488]?=-99.99 g-band galaxy semi-major
                                    axis, half-light radius
 152-158  F7.2  kpc         Rhlr    [0,1488]?=-99.99 r-band galaxy semi-major
                                    axis, half-light radius
 160-166  F7.2  kpc         Rchl,g  [-336,308] g-band galaxy circular
                                    half-light radius
 168-174  F7.2  kpc         Rchl,r  [-336,287] r-band galaxy circular
                                    half-light radius
 176-181  F6.2  kpc         Re      ?=-99.99 Bulge semi-major effective radius
 183-188  F6.2  kpc         e_Re    ?=-99.99 Uncertainty in Re
 190-195  F6.2  ---         e       [0,1]?=-99.99 Bulge ellipticity (G3)
 197-202  F6.2  ---         e_e     [0,1]?=-99.99 Uncertainty in e
 204-210  F7.2  deg         phib    [-360,360] Bulge position angle (G4)
 212-217  F6.2  deg         e_phib  ?=-99.99 Uncertainty in phib
 219-224  F6.2  kpc         Rd      ?=-99.99 Exponential disk scale length
 226-231  F6.2  kpc         e_Rd    ?=-99.99 Uncertainty in Rd
 233-238  F6.2  deg         i       ?=-99.99 Disk inclination angle (1)
 240-245  F6.2  deg         e_i     ?=-99.99 Uncertainty in i
 247-253  F7.2  deg         phid    [-360,360] Disk position angle (G4)
 255-260  F6.2  deg         e_phid  ?=-99.99 Uncertainty in phid
 262-267  F6.2  arcsec      (dx)g   ?=-99.99 g-band B+D model center X offset
                                    (G5)
 269-274  F6.2  arcsec     e_(dx)g  ?=-99.99 Uncertainty in (dx)g
 276-281  F6.2  arcsec      (dy)g   ?=-99.99 g-band B+D model center Y offset
                                    (G5)
 283-288  F6.2  arcsec     e_(dy)g  ?=-99.99 Uncertainty in (dy)g
 290-295  F6.2  arcsec      (dx)r   ?=-99.99 r-band B+D model center X offset
                                    (G5)
 297-302  F6.2  arcsec     e_(dx)r  ?=-99.99 Uncertainty in (dx)r
 304-309  F6.2  arcsec      (dy)r   ?=-99.99 r-band B+D model center Y offset
                                    (G5)
 311-316  F6.2  arcsec     e_(dy)r  ?=-99.99 Uncertainty in (dy)r
 318-323  F6.2  ---         S2g     ?=-99.99 g-band image smoothness parameter
                                    (G6)
 325-330  F6.2  ---         S2r     ?=-99.99 r-band image smoothness parameter
                                    (G6)
 332-337  F6.2  mag         ggMag   ?=-99.99 Absolute, rest-frame g-band GIM2D
                                    galaxy magnitude (Eq. 3a)
 339-344  F6.2  mag        e_ggMag  ?=-99.99 Uncertainty in ggMag
 346-351  F6.2  mag         gbMag   ?=-99.99 Absolute, rest-frame g-band GIM2D
                                    bulge magnitude (Eq. 3c)
 353-358  F6.2  mag        e_gbMag  ?=-99.99 Uncertainty in gbMag
 360-365  F6.2  mag         gdMag   ?=-99.99 Absolute, rest-frame g-band GIM2D
                                    disk magnitude (Eq. 3e)
 367-372  F6.2  mag        e_gdMag  ?=-99.99 Uncertainty in gdMag
 374-379  F6.2  mag         rgMag   ?=-99.99 Absolute, rest-frame r-band GIM2D
                                    galaxy magnitude (Eq. 3b)
 381-386  F6.2  mag        e_rgMag  ?=-99.99 Uncertainty in rgMag
 388-393  F6.2  mag         rbMag   ?=-99.99 Absolute, rest-frame r-band GIM2D
                                    bulge magnitude (Eq. 3d)
 395-400  F6.2  mag        e_rbMag  ?=-99.99 Uncertainty in rbMag
 402-407  F6.2  mag         rdMag   ?=-99.99 Absolute, rest-frame r-band GIM2D
                                    disk magnitude (Eq. 3f)
 409-414  F6.2  mag        e_rdMag  ?=-99.99 Uncertainty in rdMag
 416-421  F6.2  ---         nb      [0,8]?=-99.99 Bulge Sersic index (4.00 for
                                    table 1)
 423-428  F6.2  ---         e_nb    [0,4]?=-99.99 Uncertainty in nb (0.00 for
                                    table 1)
 430-435  F6.2  ---         PpS     [0,1]?=-99.99 F-test probability (2)
 439-442  F4.2  ---         Pn4     [0,1]? F-test probability (only for
                                    table 2) (3)

Note (1): i=0 for face-on disk.

Note (2): That a B+D model is not required compared to a pure Sersic model.

Note (3): That a free nb B+D model is not required compared to a fixed
          nb=4 B+D model.


Byte-by-byte Description of file: table3.dat

   Bytes Format Units      Label    Explanations

   1- 18  A18   ---        objID    SDSS object identifier
  20- 29  F10.6 ---        z        ?=-99.99 SDSS redshift (spectroscopic if
                                    available or photometric ("-1" in "Sp")
  31- 32  I2    ---        Sp       [-2/6] SDSS SpecClass value (G1)
  36- 42  F7.3  kpc/arcsec Scale    ?=-99.99 Physical scale at redshift z
  44- 53  F10.3 Mpc+3      Vmax     ?=-99.99 Galaxy volume correction (Eq. 7)
  55- 60  F6.2  mag        gg2d     ?=-99.99 GIM2D pure Sersic model g-band
                                    magnitude
  62- 67  F6.2  mag        e_gg2d   ?=-99.99 Uncertainty in gg2d
  69- 74  F6.2  mag        rg2d     ?=-99.99 GIM2D pure Sersic model r-band
                                    magnitude
  76- 81  F6.2  mag        e_rg2d   ?=-99.99 Uncertainty in rg2d
  83- 88  F6.2  mag        gg2df    ?=-99.99 GIM2D pure Sersic model g-band
                                    fiber magnitude
  90- 95  F6.2  mag        rg2df    ?=-99.99 GIM2D pure Sersic model r-band
                                    fiber magnitude
  97-102  F6.2  mag        dCol     [-4.95,+5.73]?=-99.99 Delta fiber color (G2)
 104-109  F6.2  kpc        Rhlg     ?=-99.99 g-band galaxy semi-major axis,
                                    half-light radius
 111-116  F6.2  kpc        Rhlr     ?=-99.99 r-band galaxy semi-major axis,
                                    half-light radius
 118-124  F7.2  kpc        Rchl,g   [-336,284] g-band galaxy circular half-light
                                    radius
 126-132  F7.2  kpc        Rchl,r   [-336,284] r-band galaxy circular half-light
                                    radius
 134-139  F6.2  ---        e        ?=-99.99 Galaxy ellipticiy (G3)
 141-146  F6.2  ---        e_e      ?=-99.99 Uncertainty in e
 148-154  F7.2  deg        phi      [-360,360] Galaxy position angle (G4)
 156-161  F6.2  deg        e_phi    ?=-99.99 Uncertainty in phi
 163-168  F6.2  arcsec     (dx)g    [-42,32]?=-99.99 g-band pure Sersic model
                                    center X offset (G5)
 170-175  F6.2  arcsec     e_(dx)g  ?=-99.99 Uncertainty in (dx)g
 177-182  F6.2  arcsec     (dy)g    [-38,30]?=-99.99 g-band pure Sersic model
                                    center Y offset (G5)
 184-189  F6.2  arcsec     e_(dy)g  ?=-99.99 Uncertainty in (dy)g
 191-196  F6.2  arcsec     (dx)r    [-31,30]?=-99.99 r-band pure Sersic model
                                    center X offset (G5)
 198-203  F6.2  arcsec     e_(dx)r  ?=-99.99 Uncertainty in (dx)r
 205-210  F6.2  arcsec     (dy)r    [-38,30]?=-99.99 r-band pure Sersic model
                                    center Y offset (G5)
 212-217  F6.2  arcsec     e_(dy)r  ?=-99.99 Uncertainty in (dy)r
 219-224  F6.2  ---        S2g      [-89,98]?=-99.99 g-band image smoothness
                                    parameter (G6)
 226-231  F6.2  ---        S2r      [-75,99]?=-99.99 r-band image smoothness
                                    parameter (G6)
 233-238  F6.2  mag        ggMag    [-37,7]?=-99.99 Absolute, rest-frame g-band
                                    GIM2D galaxy magnitude (Eq. 3a)
 240-245  F6.2  mag        e_ggMag  ?=-99.99 Uncertainty in ggMag
 247-252  F6.2  mag        rgMag    [-36,6]?=-99.99 Absolute, rest-frame r-band
                                    GIM2D galaxy magnitude (Eq. 3b)
 254-259  F6.2  mag        e_rgMag  ?=-99.99 Uncertainty in rgMag
 261-266  F6.2  ---        ng       [0,8]?=-99.99 Galaxy Sersic index
 268-273  F6.2  ---        e_ng     ?=-99.99 Uncertainty in ng


Global notes:

Note (G1): Flags as follows:
    -1 = photometric;
    -2 = no redshift available
     0 = unknown: spectrum not classifiable (zConf<0.25)
     1 = star
     2 = galaxy
     3 = QSO
     4 = high-redshift quasar, z>2.3
     5 = Spectrum of blank sky.
     6 = STAR_LATE: Star dominated by molecular bands M or later.
Note (G2): Delta fiber color defined as: (g-r)gim2d,fiber-(g-r)SDSS,fiber.

Note (G3): e=1-b/a, e=0 for a circular bulge.

Note (G4): Measured clockwise from the +y axis of SDSS images.

Note (G5): From Column|Row position given by colc(g|r)|rowc(g|r) on SDSS
           corrected image.
Note (G6): As defined in Simard et al. (2009A&A...508.1141S).

'''


import atpy, os
from pylab import *
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

class cluster(baseClusterNSA):
    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)
        self.readsdsscsv()
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
	self.useraflag=0
	if len(self.sdss_ra) != len(self.ra):
		print 'WARNING: problem with sdss cat for ',self.prefix
		self.useraflag=1

    def match2zoo(self,delta):

	#self.zoo_objid=[]
        i_match=zeros(len(self.ra),'i')
        i_match_flag=zeros(len(self.ra),'i')
        z=zeros(len(self.ra),'i')
        for i in range(len(self.ra)):
		if self.useraflag:
			imatch,matchflag,nmatch=findnearest(self.ra[i],self.dec[i],bd.zdat._RA,bd.zdat._DE,delta)
			if matchflag:
				#print i,nmatch, self.n.ISDSS[i],' found match to simard B/D sample'
				#print self.sdss_objid[i],bd.zdat.objID_1[imatch]
				i_match[i]=imatch
				i_match_flag[i]=matchflag
			else:
				print i,'Oh no!  no match for galaxy ',i
				#print self.sdss_objid[i]

		else:
			try:
				imatch=bd.zdict[str(self.sdss_objid[i])]
				#print i,'found a match using dictionary'
				i_match[i]=imatch
				i_match_flag[i]=1
				#print i_match[i],i_match_flag[i]
				#if i_match_flag[i] == 0:
				#		print 'heyyyyyyyyyyyyyyy!!!!!!'
			except KeyError:
				print 'no match with dictionary'
        # write out results as a fits table that is line-matched to cluster NSA table
        #ncols=shape(bd.zdat)[1]
        #nrows=shape(bd.zdat)[0]
        #newarray=zeros((nrows,ncols))
        print self.prefix,': ',len(i_match)-sum(i_match_flag),'/',len(i_match),' galaxies not matched'
        #ztab=bd.zdat.rows(i_match)
        #ztabsubbd.where(imatch)
        otab=atpy.Table()
        otab.add_column('matchflag',i_match_flag,dtype='bool')
	

        otab.add_column('matchindex',i_match)
	
        zdatindex=i_match[i_match_flag]

        #print zdatindex
        for i in range(len(bd.zdat.names)):
            dtype=bd.zdat.columns[i]
            col=zeros(len(self.ra),dtype)

            #print zdatindex
            for j in range(len(i_match)):
                if i_match_flag[j]:
                    col[j]=bd.zdat[bd.zdat.names[i]][i_match[j]]
            #col[i_match_flag]=bd.zdat[bd.zdat.names[i]][zdatindex]
            # get rid of column names that start with _ or __ b/c they cause trouble down the road
            if bd.zdat.names[i].startswith('__'):			
                colname=bd.zdat.names[i][2:]

            elif bd.zdat.names[i].startswith('_'):
                colname=bd.zdat.names[i][1:]
            else:
			
                colname=bd.zdat.names[i]
            print colname
            otab.add_column(colname,col,unit=bd.zdat.units[bd.zdat.names[i]])
        outfile=homedir+'research/LocalClusters/NSAmastertables/SimardGIM2D/'+self.prefix+'_GIM2D.fits'
        if os.path.exists(outfile):
            os.remove(outfile)
        otab.write(outfile)

class simard:
    def __init__(self):
        # infile=homedir+'research/NSA/nsa_v0_1_2.fits'
        #infile=homedir+'research/SimardSDSS2011/vizier_votable-9.vot'
        #self.zdat=atpy.Table(infile,type='vo')
        infile=homedir+'research/SimardSDSS2011/table1and3.fits'
        self.zdat=atpy.Table(infile,type='fits')
        self.zRA=self.zdat._RA
        self.zDEC=self.zdat._DE
	self.zdict=dict((a,b) for a,b in zip(self.zdat.objID_1,arange(len(self.zRA))))
bd=simard()
# match radius = 3"/3600 -> deg
delta=2./3600. 
#mkw11=cluster('MKW11')
#mkw11.match2zoo(delta)
myclusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
#myclusternames=['MKW11']
for cname in myclusternames:
    cl=cluster(cname)
    print '\n',cl.prefix, '\n'
    cl.match2zoo(delta)
