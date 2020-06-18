#!/usr/bin/env python

from LCScommon import *
import atpy
from pylab import *
class cluster:
    def __init__(self,clustername):
        nsafile=homedir+'research/LocalClusters/NSAmastertables/'+clustername+'_NSAmastertable.fits'
        self.ndat=atpy.Table(nsafile)
        agcfile=homedir+'research/LocalClusters/MasterTables/'+clustername+'mastertable.fits'
        self.adat=atpy.Table(agcfile)
        self.prefix=clustername
    def matchnsa2agc(self):
        aredshift=self.adat.SUPERVOPT/3.e5
        index=arange(len(self.adat.SUPERRA))
        self.mindistance=zeros(len(self.ndat.RA),'d')
        self.matchindex=zeros(len(self.ndat.RA),'i')
        outcoords=open(homedir+'research/LocalClusters/NSAmastertables/'+self.prefix+'_NSAnoAGCsdsscoords.dat','w')
        for i in range(len(self.ndat.RA)):
            distance=sqrt((1.*self.ndat.RUN[i]-self.adat.SDSSRUN)**2 +(1.*self.ndat.CAMCOL[i]-self.adat.SDSSCAMCOL[i])**2 + (1.*self.ndat.FIELD[i]-self.adat.SDSSFIELD[i])**2)
            #print 'matching by run/camcol/field = ',index[distance == min(distance)],min(distance)
            distance2=sqrt((self.ndat.RA[i]-self.adat.SUPERRA)**2 +(self.ndat.DEC[i]-self.adat.SUPERDEC)**2 +(self.ndat.Z[i]-aredshift)**2)
            #print 'matching by RA, Dec, redshift = ',index[distance2 == min(distance2)],min(distance2)*3600
            distance3=sqrt((self.ndat.RA[i]-self.adat.SUPERRA)**2 +(self.ndat.DEC[i]-self.adat.SUPERDEC)**2 )
            if min(distance3)*3600. > 4.:
                imatch=index[distance3 == min(distance3)]
                print self.ndat.IAUNAME[i],index[distance3 == min(distance3)],min(distance3)*3600,self.ndat.ISDSS[i],self.adat.SDSSflag[index[distance3 == min(distance3)]],self.adat.AGCNUMBER[index[distance3 == min(distance3)]],self.ndat.RA[i],self.ndat.DEC[i],self.adat.SUPERRA[imatch],self.adat.SUPERDEC[imatch],self.ndat.Z[i],self.adat.On24ImageFlag[imatch]

                outcoords.write('%s19-%i-%i %12.8f %12.8f\n'%(self.ndat.IAUNAME[i],self.ndat.On24ImageFlag[i],int(min(distance3)*3600.),self.ndat.RA[i],self.ndat.DEC[i]))
            #print self.ndat.RA[i],self.adat.SUPERRA[index[distance3 == min(distance3)]],self.ndat.DEC[i],self.adat.SUPERRA[index[distance3 == min(distance3)]]
            #print self.ndat.RA[i]-self.adat.SUPERRA[index[distance3 == min(distance3)]],self.ndat.DEC[i]-self.adat.SUPERDEC[index[distance3 == min(distance3)]]
            self.mindistance[i]=min(distance3)*3600.
            try:
                self.matchindex[i]=index[distance3 == min(distance3)]
            except ValueError:
                self.matchindex[i]=index[distance3 == min(distance3)][0]
        outcoords.close()
        #figure()
        #hist(self.mindistance)
        #xlabel('distance to nearest point')
        #figure()
        #hist(self.matchindex)
        #xlabel('match index')

    def matchagc2nsa(self):
        print 'MATCHING AGC TO NSA'
        aredshift=self.adat.SUPERVOPT/3.e5
        index=arange(len(self.ndat.RA))
        self.mindistance=zeros(len(self.adat.SUPERRA),'d')
        self.matchindex=zeros(len(self.adat.SUPERRA),'i')
        outcoords=open(homedir+'research/LocalClusters/NSAmastertables/'+self.prefix+'_AGCnoNSAsdsscoords.dat','w')
        for i in range(len(self.adat.SUPERRA)):
            distance2=sqrt((self.ndat.RA-self.adat.SUPERRA[i])**2 +(self.ndat.DEC-self.adat.SUPERDEC[i])**2 +(self.ndat.Z-aredshift[i])**2)
            #print 'matching by RA, Dec, redshift = ',index[distance2 == min(distance2)],min(distance2)*3600
            distance3=sqrt((self.ndat.RA-self.adat.SUPERRA[i])**2 +(self.ndat.DEC-self.adat.SUPERDEC[i])**2 )
            if min(distance3)*3600. > 4.:
                imatch=index[distance3 == min(distance3)]
                print min(distance3)*3600,self.ndat.ISDSS[imatch],self.adat.SDSSflag[i],self.adat.AGCNUMBER[i],self.adat.SUPERRA[i],self.adat.SUPERDEC[i],self.ndat.RA[imatch],self.ndat.DEC[imatch],self.ndat.Z[imatch],self.adat.On24ImageFlag[i],self.adat.VSOURCE[i]
                outcoords.write('%i-%i-%i %12.8f %12.8f \n'%(self.adat.AGCNUMBER[i],self.adat.On24ImageFlag[i],int(min(distance3)*3600.),self.adat.SUPERRA[i],self.adat.SUPERDEC[i]))
            #print self.ndat.RA[i],self.adat.SUPERRA[index[distance3 == min(distance3)]],self.ndat.DEC[i],self.adat.SUPERRA[index[distance3 == min(distance3)]]
            #print self.ndat.RA[i]-self.adat.SUPERRA[index[distance3 == min(distance3)]],self.ndat.DEC[i]-self.adat.SUPERDEC[index[distance3 == min(distance3)]]
            self.mindistance[i]=min(distance3)*3600.
            try:
                self.matchindex[i]=index[distance3 == min(distance3)]
            except ValueError:
                self.matchindex[i]=index[distance3 == min(distance3)][0]
        outcoords.close()
        #figure()
        #hist(self.mindistance)
        #xlabel('distance to nearest point')
        #figure()
        #hist(self.matchindex)
        #xlabel('match index')
    def printnosdssmatch(self):
        nosdssflag=~self.adat.SDSSflag & ~self.adat.SDSSphotflag
        print 'AGC WITH NO SDSS MATCH'
        print 'feed these coords into http://skyserver.sdss3.org/dr9/en/tools/chart/list.asp'
        outfile=open(homedir+'research/LocalClusters/RegionsFiles/'+self.prefix+'_NoSDSSMatch.reg','w')
        
        outfile.write('global color=cyan font="helvetica 10 normal" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source \n')
        outfile.write('fk5 \n')
        for i in range(len(nosdssflag)):
            if nosdssflag[i]:
                print self.adat.AGCNUMBER[i],self.adat.AGCRA[i],self.adat.AGCDEC[i]
                outfile.write('circle(%12.8f, %12.8f, 30") # color=red \n'%(self.adat.AGCRA[i],self.adat.AGCDEC[i]))
        outfile.close()
        
        noagcflag=self.adat.SDSSflag & ~self.adat.AGCflag
        print 'SDSS WITH NO AGC MATCH'
        print 'feed these coords into http://skyserver.sdss3.org/dr9/en/tools/chart/list.asp'
        outfile=open(homedir+'research/LocalClusters/RegionsFiles/'+self.prefix+'_SDSSWithNoAGCMatch.reg','w')
        
        outfile.write('global color=cyan font="helvetica 10 normal" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source \n')
        outfile.write('fk5 \n')
        for i in range(len(noagcflag)):
            if noagcflag[i]:
                print self.adat.AGCNUMBER[i],self.adat.AGCRA[i],self.adat.AGCDEC[i]
                outfile.write('circle(%12.8f, %12.8f, 30") # color=red \n'%(self.adat.AGCRA[i],self.adat.AGCDEC[i]))
        outfile.close()
mkw11=cluster('MKW11')

mkw8=cluster('MKW8')
awm4=cluster('AWM4')
a2052=cluster('A2052')
a2063=cluster('A2063')
ngc=cluster('NGC6107')
coma=cluster('Coma')
herc=cluster('Hercules')
a1367=cluster('A1367')

mylocalclusters=[mkw11,mkw8,awm4,a2052,a2063,ngc,coma,herc,a1367]

for cl in mylocalclusters:

    cl.matchnsa2agc()
    cl.matchagc2nsa()

for cl in mylocalclusters:
    print cl.prefix
    print 'number of galaxies on 24um image in agc = ',sum(cl.adat.On24ImageFlag)
    print 'number of galaxies on 24um image in NSA = ',sum(cl.ndat.On24ImageFlag)

