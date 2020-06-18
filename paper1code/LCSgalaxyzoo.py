#!/usr/bin/env python
from pylab import *
from pyraf import iraf
import pyfits
import sqlclsdss3
import glob
import os
import mystuff as my
import ReadAGCsav
from LCScommon import *
from matplotlib.backends.backend_pdf import PdfFile

delta=100.#width of cutouts in arcsec
ramin=170.
ramax=250.
decmax=38.
zmin=0.01366#min z cut, z(coma)-3 sigma
zmax=0.04333#max z cut, z(A2052)+3 sigma
vmin=zmin*3.e5
vmax=zmax*3.e5
		
class cluster:
    def __init__(self):
	self.prefix=names[ncl]
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]
	self.biweightvel=clustercbi[self.prefix]
	self.biweightscale=clustersbi[self.prefix]
	self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. #in Mpc
        self.r200deg=self.r200*1000./my.DA(self.cz,h)/3600. # in Degrees
        self.dr=3.#get galaxies w/in 3 degrees

    def getsdsszoocat(self):
        drsearch=3.*60.#search radius in arcmin for sdss query
	sdss3_url="http://skyserver.sdss3.org/dr8/en/tools/search/x_sql.asp"
	print "getting galaxy zoo morphologies"
        query="select n.distance, g.objID, z.dr7objid, z.nvote, z.p_el, z.p_cw,z.p_acw,z.p_edge,z.p_dk,z.p_mg,z.p_cs,z.p_el_debiased,z.p_cs_debiased,z.spiral,z.elliptical,z.uncertain from galaxy g, specobj s, zooSpec z, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and z.objid = g.objid and s.z < %5.4f and s.z > %5.4f" % (self.cra,self.cdec,drsearch,zmax,zmin)
        try:
            lines=sqlclsdss3.query(query).readlines()
        except IOError:
            print "IOError for cluster",self.prefix,i," trying spec query again"
            lines=sqlclsdss3.query(query).readlines()
        print self.prefix,": got number + 1 of galaxy zoo spec objects = ",len(lines)
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxyzoospec.dat'
        outfile=open(n,'w')
        j=0
        if (len(lines) > 1.):
            for line in lines[1:]:
                if j < 0:
                    print line
                    j=j+1
                outfile.write(line)
        outfile.close()

    def getsdsszooNoSpec(self):
        drsearch=3.*60.#search radius in arcmin for sdss query
        print "getting galaxyzooNoSpec cat for cluster",self.prefix
	query="select  n.distance, g.objID, z.specobjid, z.dr7objid, z.nvote, z.p_el, z.p_cw,z.p_acw,z.p_edge,z.p_dk,z.p_mg,z.p_cs from galaxy g, zooNoSpec z, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and z.objid = g.objid " % (self.cra,self.cdec,drsearch)
        try:
            lines=sqlclsdss3.query(query).readlines()
        except IOError:
            print "IOError for cluster",self.prefix," trying phot query again"
            lines=sqlclsdss3.query(query).readlines()

	print "got number+1 phot objects = ",len(lines)
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxyzooNoSpec.dat'
        outfile=open(n,'w')
        j=0
        if (len(lines) > 1.):
            for line in lines[1:]:
                if j < 0:
                    print line
                    j=j+1
                outfile.write(line)
        outfile.close()

	
names=['MKW11','MKW8','AWM4', 'A2063','A2052','NGC6107', 'Coma','A1367','Hercules']

for ncl in range(len(names)):
    if ncl == 0:
        mkw11=cluster()
        cl=mkw11
    if ncl == 1:
        mkw8=cluster()
        cl=mkw8
    if ncl == 2:
        awm4=cluster()
        cl=awm4
    if ncl == 3:
        a2063=cluster()
        cl=a2063
    if ncl == 4:
        a2052=cluster()
        cl=a2052
    if ncl == 5:
        ngc6107=cluster()
        cl=ngc6107
    if ncl == 6:
        coma=cluster()
        cl=coma
    if ncl == 7:
        a1367=cluster()
        cl=a1367
    if ncl == 8:
        hercules=cluster()
        cl=hercules
    #cl.getsdsszoocat()

    cl.getsdsszooNoSpec()
