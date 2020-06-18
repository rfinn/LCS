#!/usr/bin/env python
"""
usage
LCSgetsdsscats.py

this will get the spectroscopic and photometric catalogs from dr7 (can pick other versions - see sqlcl.py)

12/3/12
updated getsdssphotcat() so that it checks the returned catalog for a 'Server.ScriptTimeout' error.
If the error is found, it will rerun the query and repeat up to 10 times.  This is still apparently
not enough for some clusters, or some times.

"""


from pylab import *
#from pyraf import iraf
#import pyfits
import sqlcl
#import glob
import os
import time
import mystuff as my
#import ReadAGCsav
from LCScommon import *
#from matplotlib.backends.backend_pdf import PdfFile

mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'


class cluster:
    def __init__(self):
	self.prefix=clusternames[ncl]
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]
	self.biweightvel=clustercbi[self.prefix]
	self.biweightscale=clustersbi[self.prefix]
	self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. #in Mpc
        self.r200deg=self.r200*1000./my.DA(self.cz,h)/3600. # in Degrees


        self.dr=3.#get galaxies w/in 3 degrees
	self.cutoutpath=cutoutpath+self.prefix+'/'
        #if self.prefix.find('A2063')>-1:
        #    self.dr=5.
    
    def getsdsscat(self):
        print 'Getting SDSS spec cat for ',self.prefix
        drsearch=self.dr*60.#search radius in arcmin for sdss query
        #zmin=self.cz-.005
        #zmax=self.cz+.005
        #from this, we will make a field sample and a cluster sample
        #query="select n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr, s.plate, s.fiberID, s.tile, g.objID,  g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z, g.run, g.rerun, g.camcol, g.field,g.err_u,g.err_g,g.err_r,g.err_i,g.err_z, g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and (g.PrimTarget & 0x00000040) > 0 and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)
        #query="select n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr, s.plate, s.fiberID, s.tile, g.objID,  g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z, g.run, g.rerun, g.camcol, g.field,g.err_u,g.err_g,g.err_r,g.err_i,g.err_z, g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)
        # removing PrimTarget selection flag
        query="select n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr, s.plate, s.fiberID, s.tile, g.objID,  g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z, g.run, g.rerun, g.camcol, g.field,g.err_u,g.err_g,g.err_r,g.err_i,g.err_z, g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and l.LineId = 6565" % (self.cra,self.cdec,drsearch,zmax,zmin)
#        query="select n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr, s.plate, s.fiberID, s.tile, g.objID  from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and (g.PrimTarget & 0x00000040) > 0 and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)#g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z 
        #print query
        try:
            lines=sqlcl.query(query).readlines()
        except IOError:
            print "IOError for cluster",self.prefix," trying spec query again"
            lines=sqlcl.query(query).readlines()
        print self.prefix,": got number + 1 of spec objects = ",len(lines)
        n=homedir+'research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.dat'
        outfile=open(n,'w')
        j=0
        if (len(lines) > 1.):
            for line in lines[1:]:
                if j < 0:
                    print line
                    j=j+1
                outfile.write(line)
        outfile.close()


    def getsdssphotcat(self):
	print 'Getting SDSS phot cat for ',self.prefix
        drsearch=self.dr*60.#search radius in arcmin for sdss query
        #zmin=self.cz-.005
        #zmax=self.cz+.005
        #from this, we will make a field sample and a cluster sample

	flag=0
        nrun=1
        print 'getting to while loop'
        while flag == 0:
            print 'inside while loop'
            #Vg=0.3556-0.7614*((self.avegr)-0.6148)#(V-g) from Blanton et al 2003
            #query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.plate_ID, g.MJD,  from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and (g.g < %5.2f) and ((0.384*g.g + 0.716*g.r)< %5.2f)" % (self.ra[i],self.dec[i],drsearch,(mr+1.5),mr)
            #changed so that only galaxies w/out spectra are returned
            #query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.objid, g.specObjID, g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z,  g.run, g.rerun, g.camcol, g.field, g.err_u,g.err_g,g.err_r,g.err_i,g.err_z,g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and  (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
            query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.objid, g.specObjID, g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z,  g.run, g.rerun, g.camcol, g.field, g.err_u,g.err_g,g.err_r,g.err_i,g.err_z,g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g where g.r < 22 and g.ra > %12.8f and g.ra < %12.8f and g.dec > %12.8f and g.dec < %12.8f and (g.specObjID = 0)" % (self.cra-drsearch/2.,self.cra+drsearch/2.,self.cdec-drsearch/2.,self.cdec+drsearch/2.)
            #query="select g.ra, g.dec from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and  (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)#changed so that only galaxies w/out spectra are returned
            # the following timed out in 10 min
            #query="select count(*) from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and  (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)#changed so that only galaxies w/out spectra are returned 
            # the following timed out at 10 min
            #query="select count(*) from galaxy g where (g.ra between %12.8f and %12.8f) and (g.dec between %12.8f and %12.8f) and (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra-drsearch,self.cra+drsearch,self.cdec-drsearch,self.cdec+drsearch)
            # now trying this        
            #query="select count(*) from galaxy g where (g.ra between %12.8f and %12.8f) and (g.dec between %12.8f and %12.8f) and (g.specObjID = 0)" % (self.cra-drsearch,self.cra+drsearch,self.cdec-drsearch,self.cdec+drsearch)
            # sdss website says the following query completes in 18 sec.  let's see how it does...
            #query='SELECT p.ra, p.dec, p.ModelMag_i, p.extinction_i FROM TargetInfo t, PhotoTag p WHERE (t.primtarget & 0x00000006>0) and p.objid=t.targetobjid'
            # this does, in fact, complete very quickly!

            #query="select count(*) from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
        
            #        query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.objid, g.specObjID, g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z,  g.run, g.rerun, g.camcol, g.field, g.err_u,g.err_g,g.err_r,g.err_i,g.err_z,g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g where (g.ra between %12.8f and %12.8f) and (g.dec between %12.8f and %12.8f) and (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra-drsearch,self.cra+drsearch,self.cdec-drsearch,self.cdec+drsearch)#changed so that only galaxies w/out spectra are returned
            print query
            start_time=time.time()
            try:
                lines=sqlcl.query(query).readlines()
            except IOError:
                print "IOError for cluster",self.prefix," trying phot query again"
                lines=sqlcl.query(query).readlines()

            elapsed_time=time.time() - start_time
            print 'time to execute query = ',elapsed_time, ' sec, ',elapsed_time/60.,' min'
            print "got number+1 phot objects = ",len(lines)

            n=homedir+'research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.dat'
            outfile=open(n,'w')
            j=0
            flag=1
            if (len(lines) > 1.):
                for line in lines[1:]:
                    if j < 0:
                        print line
                        j=j+1
                    outfile.write(line)
                    if line.find('Server.ScriptTimeout') > -1:
                        flag=0
                    elif line.find('Timeout') > -1:
                        flag = 0
            outfile.close()
            nrun += 1
            if nrun > 15:
                return
            if flag == 0:
                print self.prefix
                print 'Running query again b/c of ScriptTimeout'
                print 'starting attempt = ',nrun

    def getsdssphotcatv2(self):  
        # going to split query into 3 separate calls.  Hopefully this will alleviate the timeout errors!
        # then can merge files with 'join'
  	print 'Getting SDSS phot cat for ',self.prefix
        drsearch=self.dr*60.#search radius in arcmin for sdss query

        for k in range(3):
            if k == 0:
                #query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.objid, g.specObjID, g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and  (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
                # removing PrimTarget constraint
                #query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.objid, g.specObjID, g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
                # adding r mag cut
                query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.objid, g.specObjID, g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.r < 22 and g.objID = n.objID and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
            if k == 1:
                #query="select g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and  (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
                query="select g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.r < 22  and g.objID = n.objID and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
            if k == 2:
                #query="select g.run, g.rerun, g.camcol, g.field, g.err_u,g.err_g,g.err_r,g.err_i,g.err_z,g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and  (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)

                query="select g.run, g.rerun, g.camcol, g.field, g.err_u,g.err_g,g.err_r,g.err_i,g.err_z,g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.r < 22 and g.objID = n.objID and (g.specObjID = 0)" % (self.cra,self.cdec,drsearch)
        # part 1 of query
            part=k+1
            print 'running part %i of query'%(part)
            flag=0
            nrun=1
            print 'getting to while loop'
            while flag == 0:
                print 'inside while loop'
                #changed so that only galaxies w/out spectra are returned
                print query
                start_time=time.time()
                try:
                    lines=sqlcl.query(query).readlines()
                except IOError:
                    print "IOError for cluster",self.prefix," trying phot query again"
                    lines=sqlcl.query(query).readlines()

                elapsed_time=time.time() - start_time
                print 'time to execute query = ',elapsed_time, ' sec, ',elapsed_time/60.,' min'
                print "got number+1 phot objects = ",len(lines)

                n=homedir+'research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.p'+str(part)+'.dat'
                outfile=open(n,'w')
                j=0
                flag=1
                if (len(lines) > 1.):
                    for line in lines[1:]:
                        if j < 0:
                            print line
                            j=j+1
                        outfile.write(line)
                        if line.find('Server.ScriptTimeout') > -1:
                            flag=0
                        elif line.find('Timeout') > -1:
                            flag = 0
                        elif line.find('ERROR') > -1:
                            flag=0
                        elif line.find('error') > -1:
                            flag=0
                outfile.close()
                nrun += 1
                if nrun > 15:
                    break
                if flag == 0:
                    print self.prefix
                    print 'Running query again b/c of ScriptTimeout'
                    print 'starting attempt = ',nrun
    def joinphotcats(self):
        n1=homedir+'research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.p1.dat'
        n2=homedir+'research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.p2.dat'
        n3=homedir+'research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.p3.dat'
        in1=open(n1,'r')
        t=in1.readlines()
        nlines=len(t)
        in1.close()

        in1=open(n1,'r')
        in2=open(n2,'r')
        in3=open(n3,'r')
        
        output=open(homedir+'research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.dat','w')
        for i in range(nlines):
            line1=in1.readline()
            line2=in2.readline()
            line3=in3.readline()
            outline=line1.rstrip()+', '+line2.rstrip()+', '+line3
            output.write(outline)
        output.close()
        in1.close()
        in2.close()
        in3.close()
#for ncl in range(len(clusternames)):
#for ncl in range(1):
#clusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
cl_list=['A2063', 'A2052']
# coma - done
# A1367 ddone
stilltodo=[3,4,5]
# still have A2052 and NGC6107 as of 12/17/12 
stilltodo=[4,5]
#  now getting some other weird error, so retrying A2063
stilltodo=[0]
for ncl in range(len(clusternames)):
#for ncl in range(6,7):
#for ncl in stilltodo:
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


    cl.getsdsscat()
    print 'Getting SDSS phot cats for ',cl.prefix

    #cl.getsdssphotcat()

    cl.getsdssphotcatv2()
    cl.joinphotcats()
