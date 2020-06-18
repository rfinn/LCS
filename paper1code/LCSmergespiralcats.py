#!/usr/bin/env python
'''

GOAL:
   merge various catalogs to make one catalog for all spirals in LCS
   - this will enable a much quicker analysis of results than using LCSanalyzeNSA.py


USAGE:
   from w/in ipython

   %run ~/Dropbox/pythonCode/LCSmergespiralcats.py -r
   mergedata()
   s=spirals() # this will append new radius data onto tables (combine upper limits, etc)

   analysis is then done with LCSanalyzespirals.py

OUTPUT:
   outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all.fits'

'''
#try:
#   import pyfits
#except ImportError:
#   from astropy.io import fits
from LCScommon import *
from pylab import *
import os
import mystuff as my
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
import sys
#import aplpy
#import ds9
#from astropy.io import ascii

#from LCSReadmasterBaseWithProfileFits import *
from LCSReadmasterBaseNSA import *


import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-r',"--readtables",help='read in mastertables',action='store_true')
parser.add_argument('-s',"--spirals",help='create LCS_spirals_all.fits (otherwise it will make LCS_all.fits)',default=False,action='store_true')

args = parser.parse_args()

loadclusters=args.readtables


minstellarmass=1.e9
class cluster(baseClusterNSA):
    def __init__(self,clustername):
        baseClusterNSA.__init__(self,clustername)
        mypath=os.getcwd()
        if mypath.find('Users') > -1:
            print "Running on Rose's mac pro"
            infile='/Users/rfinn/research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'
        elif mypath.find('home') > -1:
            print "Running on coma"
            infile=homedir+'research/LocalClusters/MasterTables/'+clustername+'mastertable.WithProfileFits.fits'

        self.mipssnrflag = self.mipssnr > 6.
        try:
            self.readsnr24NSA()
        except:
            print self.prefix,": couldn't read SNR24 file"
        try:
            self.readGalfitSersicResults()
        except:
            print self.prefix,": couln't read galfit sersic results"
        try:
            self.readGalfitResults()
        except:
            print self.prefix,": couldn't read galfit results"
        #self.size24=self.sex24.FWHM_DEG*3600.
        self.size24=self.sex24.FLUX_RADIUS1*mipspixelscale
        self.sizeratio=self.galfit24.cre1*mipspixelscale/self.n.SERSIC_TH50#(self.gim2d.Rhlr_2/self.gim2d.Scale_2)
        self.mingalaxymass=5.e9
        self.rmagcut=20.+5.*log10(self.cdMpc/(clusterbiweightcenter['Hercules']/H0))
        self.rmag=22.5-2.5*log10(self.n.SERSICFLUX[:,2])


        self.galfitflag=self.On24ImageFlag & (self.snrse>snr24cut) &  (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag & (self.ce.LIR > 5.1e8) & (self.stellarmass > minstellarmass)# & ~self.agnflag & (self.rmag < self.rmagcut) #& ~self.galfit24.numerical_error_flag24   
                                                                                                             
        self.galfitsample=self.On24ImageFlag & (self.snrse>snr24cut) & (self.n.SERSIC_TH50 > mipspixelscale) & self.spiralflag& ~self.agnflag  & (self.ce.LIR > 5.1e8) & (self.stellarmass > minstellarmass)


        # exclude objects that had issues w/galfit fit, like nearby neighbor
        gal_ids=visual_cut[self.prefix]
        for id in gal_ids:
            try:
                if self.spiralflag[self.nsadict[id]]:
                    print 'Check out fit for this spiral ',self.prefix,' NSAID=',id
                self.galfitflag[self.nsadict[id]]=0
            except:
                print 'ERROR: problem resetting galfitflag with ',self.prefix,' NSAID=',id

        '''    
        self.member=(self.dvflag) & (self.drR200 < 1.3)

        self.nearfield=(self.dvflag) & (self.drR200 > 1.3) & (self.drR200 < 2.)
        self.field=((self.dvflag) & (self.drR200 > 2.)) | ~self.dvflag
        #self.member=self.dvflag
        self.sample24flag=self.galfitflag & self.spiralflag# self.On24ImageFlag & (self.snr24>3) & ~self.agnflag & (self.n.SERSIC_TH50 > Remin) & self.spiralflag# & (log10(self.stellarmass) > 9.5) & (log10(self.stellarmass) < 12)
        self.blueclustersample=self.member & self.blueflag & self.sample24flag
        self.bluefieldsample=~self.member & self.blueflag & self.sample24flag
        self.greenclustersample=self.member & self.greenflag & self.sample24flag
        self.greenfieldsample=~self.member & self.greenflag & self.sample24flag
        self.redclustersample=self.member & self.redflag & self.sample24flag
        self.redfieldsample=~self.member & self.redflag & self.sample24flag
        self.varlookup={'stellarmass':log10(self.stellarmass),'Re':self.n.SERSIC_TH50,'R24':self.galfit24.re1*mipspixelscale,'NUV':self.n.ABSMAG[:,1],'r':self.n.ABSMAG[:,4],'m24':self.sex24.MAG_BEST,'redshift':self.n.ZDIST,'NUVr':(self.n.ABSMAG[:,1]-self.n.ABSMAG[:,4]),'NUV24':(self.n.ABSMAG[:,1]-self.sex24.MAG_BEST),'24mass':(self.sex24.MAG_BEST-log10(self.stellarmass)),'ratioR':self.sex24.FLUX_RADIUS1*mipspixelscale/self.n.SERSIC_TH50,'BT':self.gim2d.B_T_r}
        '''
        self.makedqtable()

    def makedqtable(self): 
        # derived quantities
        #my_columns=['SIZE_RATIO','STELLARMASS','SNR_SE','RMAG','DELTA_DEC','DELTA_RA', 'DELTA_V','DR_R200','CLUSTER_PHI','HIflag','HIDef','HImass','NUVr_color','agnflag','galfitflag','CLUSTER_SIGMA','CLUSTER_REDSHIFT','CLUSTER_LX','CLUSTER']
        # removing SIZE_RATIO and replacing using upper limits, etc
        my_columns=['SIZERATIO','STELLARMASS','SNR_SE','RMAG','DELTA_DEC','DELTA_RA', 'DELTA_V','DR_R200','CLUSTER_PHI','HIflag','HIDef','HImass','NUVr_color','agnflag','galfitflag','CLUSTER_SIGMA','CLUSTER_REDSHIFT','CLUSTER_LX','APEXFLUX','APEXFLUXERR','APEX_SNR','CLUSTER']
        arrays=[self.sizeratio,self.stellarmass,self.snrse,self.rmag,self.delta_dec,self.delta_ra,self.dv,self.drR200,self.cluster_phi,self.HIflag,self.HIDef,self.HImass,self.NUVr_color,self.agnflag,self.galfitflag,self.clustersigma*ones(len(self.sizeratio),'f'),self.cz*ones(len(self.sizeratio),'f'),self.cLx*ones(len(self.sizeratio),'f'),self.mipsflux,self.mipsfluxerr,self.mipssnr]

        datatypes=['d','d','d','d','d','d','d','d','d','i','d','d','d','i','i','f','f','f','d','d','d']
        allcolumns=[]
        self.dq=Table()
        for i in range(len(my_columns) - 1):
            newcol=Column(data=np.array(arrays[i],datatypes[i]),name=my_columns[i])
            self.dq.add_column(newcol)
            if my_columns[i].find('CLUSTER') > -1:
                print newcol
        # add column containing cluster name
        clustername=[]
        for i in range(len(self.sizeratio)):
            clustername.append(self.prefix)
        newcol=Column(data=np.array(clustername,'S8'),name='CLUSTER')
        self.dq.add_column(newcol)
        self.allclustername=clustername


if loadclusters:
    mkw11=cluster('MKW11')
    ngc=cluster('NGC6107')
    coma=cluster('Coma')
    mkw8=cluster('MKW8')
    awm4=cluster('AWM4')
    a2052=cluster('A2052')
    a2063=cluster('A2063')
    herc=cluster('Hercules')
    a1367=cluster('A1367')
    clustersbylx=[mkw11,ngc,mkw8,awm4,herc,a1367,a2063,a2052,coma]
    #clustersbylx=[mkw11]
    mylocalclusters=clustersbylx

#clustersbymass=[mkw11,awm4,mkw8,ngc,a2052,a2063,herc,a1367,coma]
#clustersbydistance=[a1367,mkw11,coma,mkw8,ngc,awm4,a2052,a2063,herc]

def mergedata():
    nsa_columns=['NSAID','IAUNAME','SUBDIR','RA','DEC','ZDIST','SERSIC_TH50','SERSIC_N','SERSIC_BA','SERSIC_PHI','PETROTH50','PETROTH90','D4000','HAEW','VDISP','FA','HAFLUX','N2FLUX','HBFLUX','O3FLUX','AHDEW','AV','ISDSS','IALFALFA','NMGY','NMGY_IVAR','ABSMAG','SERSICFLUX','CLUMPY','ASYMMETRY','RUN','CAMCOL','FIELD','RERUN']
    nsa_format=['J','S','S','E','E','E','E','E','E','E','E','E','E','E','E','E','E']
    n_columns=['HIMASS','AGNKAUFF','AGNKEWLEY','AGNSTASIN','AGCNUMBER']
    n_format=['E']
    #galfit24_columns=['mag1','mag1err','nsersic1','nsersic1err','re1','re1err','axisratio1','axisratio1err','pa1','pa1err','xc1','yc1','numerical_error_flag24','chi2nu','cmag1','cmag1err','cnsersic1','cnsersic1err','cre1','cre1err','caxisratio1','caxisratio1err','cpa1','cpa1err','cxc1','cyc1','cnumerical_error_flag24','cchi2nu','fcmag1','fcmag1err','fcnsersic1','fcnsersic1err','fcre1','fcre1err','fcaxisratio1','fcaxisratio1err','fcpa1','fcpa1err','fcxc1','fcyc1','fcnumerical_error_flag24','fcchi2nu']
    galfit24_columns=['fmag1','fmag1err','fnsersic1','fnsersic1err','fre1','fre1err','faxisratio1','faxisratio1err','fpa1','fpa1err','fxc1','fyc1','fnumerical_error_flag24','fchi2nu','fcmag1','fcmag1err','fcnsersic1','fcnsersic1err','fcre1','fcre1err','fcaxisratio1','fcaxisratio1err','fcpa1','fcpa1err','fcxc1','fcyc1','fcnumerical_error_flag24','fcchi2nu']
    galfit24_format=['E','E','E','E']
    gim2d_columns=['matchflag','B_T_r','e__B_T_r','S2g_1','Re','e_Re','Rd','e_Rd','Rhlr_2','ng','e_ng']
    gim2d_format=['L','E','E','E','E','E','E','E']
    zoo_columns=['p_elliptical','p_spiral','p_el','p_cs','p_uncertain','p_mg','p_edge','p_dk','match_flag']
    zoo_format=['E','E','E','E','E','E','E','E','L']
    ce_columns=['LIR_ZDIST','SFR_ZDIST','FLUX24','FLUX24ERR','SE_FLUX_AUTO','LIR_ZCLUST','SFR_ZCLUST','MATCHFLAG24','MIPS_WEIGHT']

    zoo_format=['E','E']
    se_columns=['FLUX_BEST','FLUX_AUTO','PETRO_RADIUS','FLAGS','FLUX_RADIUS1','FLUX_RADIUS2']
    ld_columns=['SIGMA_NN','SIGMA_5','SIGMA_10','RHOMASS']
    wise_columns=['W1MAG_3','W1FLG_3','W2MAG_3','W2FLG_3','W3MAG_3','W3FLG_3','W4MAG_3','W4FLG_3']
    mstar_columns=['MSTAR_AVG','MSTAR_50','SFR_AVG','SFR100_AVG','SFRAGE','TAU']
    ld_format=['E','E','E','E']
    my_columns=['STELLARMASS','SNR_SE','RMAG','DELTA_DEC','DELTA_RA', 'DELTA_V','DR_R200','CLUSTER_PHI','HIflag','HIDef','NUVr_color','agnflag','galfitflag','CLUSTER_SIGMA','CLUSTER_REDSHIFT','CLUSTER_LX','CLUSTER','APEXFLUX','APEXFLUXERR','APEX_SNR']
    
    allcolumns=[nsa_columns,n_columns,galfit24_columns,gim2d_columns,zoo_columns,ce_columns,ld_columns,my_columns,wise_columns,se_columns,mstar_columns]
    integer_columns=['NSAID','ISDSS','IALFALFA','AGNKAUFF','AGNKEWLEY','AGNSTASIN','matchflag','match_flag','FLG','FLAGS']
    nsa_multicolumn=['NMGY','NMGY_IVAR','ABSMAG','SERSICFLUX','CLUMPY','ASYMMETRY']
    #print allcolumns
    #ldat=Table()
    ldat=[]
    for collist in allcolumns:
            
        for col in collist:
            print col,str(col)
            newcol=[]
            for cl in mylocalclusters:
                if 'NSAID' in collist:
                    tabdat=cl.nsa
                elif 'HIMASS' in collist:
                    tabdat=cl.n
                elif 'fcmag1' in collist:
                    tabdat=cl.galfit24
                elif 'B_T_r' in collist:
                    tabdat=cl.gim2d
                elif 'p_mg' in collist:
                    tabdat=cl.zoo
                elif 'LIR_ZDIST' in collist:
                    tabdat=cl.ce             
                elif 'SIGMA_NN' in collist:
                    tabdat=cl.ld
                elif 'STELLARMASS' in collist:
                    tabdat=cl.dq
                elif 'W1MAG_3' in collist:
                    tabdat=cl.wise
                elif 'FLUX_RADIUS1' in collist:
                    tabdat=cl.sex24
                elif 'MSTAR_50' in collist:
                    tabdat=cl.jmass
                if args.spirals:
                    newcol = newcol + tabdat[col][cl.On24ImageFlag & cl.spiralflag].tolist()
                else:
                    newcol = newcol + tabdat[col][cl.On24ImageFlag].tolist()
            if col.find('NSAID') > -1:
                newcol=fits.Column(array=np.array(newcol,'i'),name=col, format='J')
            elif (col.find('flag') > -1) | (col.find('AGN') > -1):
                newcol=fits.Column(array=np.array(newcol,dtype='bool'),name=col,format='L')
            elif col in integer_columns:#(col.find('flag') > -1) | (col.find('AGN') > -1) |  (col.find('ISDSS') > -1):
                newcol=fits.Column(array=np.array(newcol,dtype='i'),name=col,format='J')
                #newcol=Column(data=np.array(newcol),name=col,dtype='i')
                #print newcol
            elif col == 'CLUSTER':
                newcol=fits.Column(array=np.array(newcol,'S8'),name=col,format='8A')
            elif col == 'IAUNAME':
                newcol=fits.Column(array=np.array(newcol,'S19'),name=col,format='19A')
            elif col == 'SUBDIR':
                newcol=fits.Column(array=np.array(newcol,'S27'),name=col,format='27A')
            elif col in nsa_multicolumn:
                newcol=fits.Column(array=newcol,name=col, format='7D')
            else:
                newcol=fits.Column(array=np.array(newcol,'d'),name=col, format='E')
            #ldat.add_column(newcol)
            ldat=ldat+[newcol]

            #col0a = fits.Column(name=col,array=newcol,format=aformat)
            #print homedir   

    

    #hdu=fits.BinTableHDU.from_columns(ldat)
    print len(ldat)
    #for l in ldat:
    #        print l.size,len(l)
    #cols=fits.ColDefs(ldat)
    tbhdu=fits.BinTableHDU.from_columns(ldat)
    #thdulist=fits.HDUList([hdu,tbhdu
    if args.spirals:
        outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all.fits'
    else:
        outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_all.fits'
    tbhdu.writeto(outfile,clobber='yes')
    #if os.path.exists(outfile):
    #    os.remove(outfile)
    



    #allcols=fits.ColDefs([col0a,col0b,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36])

    #ldat.write(outfile)
    #tbhdu=fits.new_table(allcols)
    #tbhdu.writeto(outfile)


#    for col in my_columns:
#        newcol=[]
#        for cl in clustersbymass:
#            newcol = newcol + cl.col[cl.galfitflag].tolist()
#        newcol=np.array(newcol,'d')
#        ldat.add_column(col,newcol)


class spirals():
        '''
        read in spiral table and append info on combined radius measurements
        '''
        
        def __init__(self):
            if args.spirals:
                outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all.fits'
            else:
                outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_all.fits'
            hdq=fits.open(outfile)
            self.s=hdq[1].data
            orig_cols=self.s.columns
            self.nsadict=dict((a,b) for a,b in zip(self.s.NSAID,arange(len(self.s.NSAID))))
            # define new columns
            point_sources=[166113,166083,72768,72633,69673,166129,79608,113092,72625,79540,70657,79545,69593,113065,166682,78100,103773,166167,72461,72490,79531,79591,166064,166051,166083]
            bad_24_fits=[72722, 80769, 68339, 68305, 82198, 78100, 70634] # can't get 24um to converge, but not point sources; use unconvolved radius as upper limit (includes ring galaxy)
            #bad_fit=[82198] # don't know what
            use_free_param_model=[113107]
            # create a new re1_24 array that uses fixed BA, free model, or unconvolved model
            # point sources - set Re to 1 pixel and mark as upper limit
            # bad fits but not point sources - used model without convolution as upper limit
            # one case where freely fit BA/PA is warranted (113107) b/c NSA PA is aligned with bar whereas 24um emission is aligned with disk
            print len(self.nsadict),len(self.s.RA)
            self.re_upperlimit=zeros(len(self.s.RA),'bool')
            self.pointsource=zeros(len(self.s.RA),'bool')

            self.super_re1=ones(len(self.s.fcre1))*self.s.fcre1
            self.super_re1err=ones(len(self.s.fcre1))*self.s.fcre1err
            ## for id in point_sources:
            ##     try:
            ##         i=self.nsadict[int(id)]
            ##     except KeyError:
            ##         continue
            ##     print 'replacing radius measurement!',self.s.fcre1[i],self.super_re1[i],1            
            ##     self.super_re1[i]=1.
            ##     self.super_re1err[i]=1.
            ##     self.re_upperlimit[i]=1
            ##     self.pointsource[i]=1
            ##     print 'take 2: replacing radius measurement!',self.s.fcre1[i],self.super_re1[i],1            
            ## for id in bad_24_fits:
            ##     i=self.nsadict[id]
            ##     self.super_re1[i]=self.s.fre1[i]
            ##     self.super_re1err[i]=self.s.fre1err[i]
            ##     self.re_upperlimit[i]=1
            ## for id in use_free_param_model:
            ##     i=self.nsadict[id]
            ##     self.super_re1[i]=self.s.fre1[i]
            ##     self.super_re1err[i]=self.s.fre1err[i]
            self.SIZE_RATIO=self.super_re1*mipspixelscale/self.s.SERSIC_TH50
            self.SIZE_RATIOERR=self.super_re1err*mipspixelscale/self.s.SERSIC_TH50

            # add columns to table and rewrite
            # self.pointsource
            # self.re_upperlimit
            # self.super_re1
            # self.SIZE_RATIO
            c1=fits.Column(name='POINTSOURCE', format='L',array=self.pointsource)
            c2=fits.Column(name='RE_UPPERLIMIT', format='L',array=self.re_upperlimit)
            c3=fits.Column(name='SUPER_RE1', format='E',array=self.super_re1)
            c3a=fits.Column(name='SUPER_RE1ERR', format='E',array=self.super_re1err)
            c4=fits.Column(name='SIZE_RATIO', format='E',array=self.SIZE_RATIO)
            c4a=fits.Column(name='SIZE_RATIOERR', format='E',array=self.SIZE_RATIOERR)
            new_cols=fits.ColDefs([c1,c2,c3,c3a,c4,c4a])
            hdu=fits.BinTableHDU.from_columns(orig_cols+new_cols)
            if args.spirals:
                outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all_size.fits'
            else:
                outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_all_size.fits'
            hdu.writeto(outfile,clobber='yes')

def readtable():
    if args.spirals:
        outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all.fits'
    else:
        outfile=homedir+'research/LocalClusters/NSAmastertables/LCS_all.fits'
    sdat=Table.read(outfile)
    return sdat
