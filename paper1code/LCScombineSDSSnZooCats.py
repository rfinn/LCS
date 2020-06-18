#!/usr/bin/env python

#read in sdss spec cat
#read in sdss galaxy zoo cat
#match based on objid
#write out cat that includes all spec cat + zoo cat info

#read in sdss phot cat, keeping only objects w/out spec
#read in galaxyzoo No Spec cat
#match based on objid
#write out cat that includes phot + zoo cat info for all objects w/no dr7 spectra

from LCScommon import *
from pylab import *

class cluster:
    def __init__(self):
	self.prefix=names[ncl]
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]


    def getsdsscoords(self):
        ngal=0
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.dat'
        outfile=open(n,'r')
        for line in outfile:
            ngal += 1
        outfile.close()

	print 'got ',ngal,' galaxies in sdss spec catalog'
        self.sdssdr=zeros(ngal,'f')
        self.sdssRA=zeros(ngal,'f')
        self.sdssDec=zeros(ngal,'f')
        self.sdssu=zeros(ngal,'f')
        self.sdssg=zeros(ngal,'f')
        self.sdssr=zeros(ngal,'f')
        self.sdssi=zeros(ngal,'f')
        self.sdssz=zeros(ngal,'f')
        self.sdssspecz=zeros(ngal,'f')
        self.sdssHaEW=zeros(ngal,'f')
        self.sdssHaEWerr=zeros(ngal,'f')
	self.sdssplate=zeros(ngal,'f')
	self.sdssfiberID=zeros(ngal,'f')
	self.sdsstile=zeros(ngal,'f')
	self.sdssobjid=zeros(ngal,'L')

        outfile=open(n,'r')
        n2='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.WithZoo.dat'
        outfile2=open(n2,'w')
        i=0
        id=[]
        for line in outfile:
                t=line.split(',')
                #for j in range(len(t)):
                #    t[j]=float(t[j])
                #(self.sdssdr[i],self.sdssRA[i],self.sdssDec[i],self.sdssu[i],self.sdssg[i],self.sdssr[i],self.sdssi[i],self.sdssz[i],self.sdssspecz[i],self.sdssHaEW[i],self.sdssHaEWerr[i],self.sdssplate[i],self.sdssfiberID[i],self.sdsstile[i],self.sdssobjid[i])=t
                self.sdssobjid[i]=t[14]
                #print self.sdssobjid[i]
                try:
                    zooIndex=self.zooSpecDict[self.sdssobjid[i]]
                    outline=line.rstrip()+', 1,'+self.zooSpecLines[zooIndex]
                except KeyError:
                    #print 'No Match to Galaxy Zoo'
                    outline=line.rstrip()+', 0,'+'0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 \n'
                outfile2.write(outline)
                i += 1
        outfile.close()
        outfile2.close()
    def getsdssphotcoords(self):
        #n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and (g.PrimTarget & 0x00000040) > 0 and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)
        ngal=0
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.dat'
        outfile=open(n,'r')
        for line in outfile:
		t=line.split(',')
                #print 'length of t in photcoords = ',len(t)
		if float(t[8]) > 1: #keeps only objects w/out spectra
			continue
		
		ngal += 1
        outfile.close()

	print 'got ',ngal,' galaxies in sdss phot catalog'
        self.sdssphotdr=zeros(ngal,'f')
        self.sdssphotRA=zeros(ngal,'f')
        self.sdssphotDec=zeros(ngal,'f')
        self.sdssphotu=zeros(ngal,'f')
        self.sdssphotg=zeros(ngal,'f')
        self.sdssphotr=zeros(ngal,'f')
        self.sdssphoti=zeros(ngal,'f')
        self.sdssphotz=zeros(ngal,'f')
	self.sdssphotobjid=zeros(ngal,'L')

        outfile=open(n,'r')

        n2='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.WithZoo.dat'
        outfile2=open(n2,'w')

        i=0
        id=[]
        for line in outfile:
                t=line.split(',')
		if float(t[8]) > 1: #keeps only objects w/out spectra
			continue
                #for j in range(len(t)):
                #   t[j]=float(t[j])

                #self.sdssphotRA[i]=t[0]
		#self.sdssphotDec[i]=t[1]
		#self.sdssphotu[i]=t[2]
		#self.sdssphotg[i]=t[3]
		#self.sdssphotr[i]=t[4]
		#self.sdssphoti[i]=t[5]
		#self.sdssphotz[i]=t[6]
		self.sdssphotobjid[i]=t[7]

                try:
                    zooIndex=self.zooNoSpecDict[self.sdssphotobjid[i]]
                    outline=line.rstrip()+', 1,'+self.zooNoSpecLines[zooIndex]
                except KeyError:
                    outline=line.rstrip()+', 0,'+'0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 \n'

                outfile2.write(outline)

                    #print 'Failure in naming convention'
                i += 1
        outfile.close()
        outfile2.close()


    def readGalaxyZooSpecCat(self):
        ngal=0
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxyzoospec.dat'
        outfile=open(n,'r')
        for line in outfile:
            ngal += 1
        outfile.close()

        #select n.distance, g.objID, z.dr7objid, z.nvote, z.p_el, z.p_cw,z.p_acw,z.p_edge,z.p_dk,z.p_mg,z.p_cs,z.p_el_debiased,z.p_cs_debiased,z.spiral,z.elliptical,z.uncertain

	print 'got ',ngal,' galaxies in sdss galaxyZooSpec catalog'
        self.zooSpecObjID=zeros(ngal,'f')
        self.zooSpecDr7ObjID=zeros(ngal,'L')
        self.zooSpecNvote=zeros(ngal,'f')
        self.zooSpecPel=zeros(ngal,'f')
        self.zooSpecPcs=zeros(ngal,'f')
        self.zooSpecPacw=zeros(ngal,'f')
        self.zooSpecPedge=zeros(ngal,'f')
        self.zooSpecPdk=zeros(ngal,'f')
        self.zooSpecPmg=zeros(ngal,'f')
        self.zooSpecPcs=zeros(ngal,'f')
        self.zooSpecPelDebiased=zeros(ngal,'f')
	self.zooSpecPcsDebiased=zeros(ngal,'f')
	self.zooSpecSpiral=zeros(ngal,'f')
	self.zooSpecElliptical=zeros(ngal,'f')
	self.zooSpecUncertain=zeros(ngal,'i')

        self.zooSpecLines=[]
        outfile=open(n,'r')
        i=0
        for line in outfile:
                self.zooSpecLines.append(line)
                t=line.split(',')
                for j in range(len(t)):
                    if j == 2:
                        t[j]=int(t[j])
                    else:
                        t[j]=float(t[j])
                    

                (distance,self.zooSpecObjID[i],self.zooSpecDr7ObjID[i],self.zooSpecNvote[i],self.zooSpecPel[i],self.zooSpecPcs[i],self.zooSpecPacw[i],self.zooSpecPedge[i],self.zooSpecPdk[i],self.zooSpecPmg[i],self.zooSpecPcs[i],self.zooSpecPelDebiased[i],self.zooSpecPcsDebiased[i],self.zooSpecSpiral[i],self.zooSpecElliptical[i],self.zooSpecUncertain[i])=t
                #print t[2],self.zooSpecDr7ObjID[i]
                i += 1
        outfile.close()
        self.zooSpecDict=dict((a,b) for a,b in zip(self.zooSpecDr7ObjID,arange(len(self.zooSpecDr7ObjID))))

    def readGalaxyZooNoSpecCat(self):
        ngal=0
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxyzooNoSpec.dat'
        outfile=open(n,'r')
        for line in outfile:
            ngal += 1
        outfile.close()

        #n.distance, g.objID, z.specobjid, z.dr7objid, z.nvote, z.p_el, z.p_cw,z.p_acw,z.p_edge,z.p_dk,z.p_mg,z.p_cs

	print 'got ',ngal,' galaxies in sdss galaxyZooNoSpec catalog'
        self.zooNoSpecObjID=zeros(ngal,'f')
        self.zooNoSpecDr7ObjID=zeros(ngal,'L')
        self.zooNoSpecNvote=zeros(ngal,'f')
        self.zooNoSpecPel=zeros(ngal,'f')
        self.zooNoSpecPcs=zeros(ngal,'f')
        self.zooNoSpecPacw=zeros(ngal,'f')
        self.zooNoSpecPedge=zeros(ngal,'f')
        self.zooNoSpecPdk=zeros(ngal,'f')
        self.zooNoSpecPmg=zeros(ngal,'f')
        self.zooNoSpecPcs=zeros(ngal,'f')
        
        self.zooNoSpecLines=[]
        outfile=open(n,'r')
        i=0
        for line in outfile:
                self.zooNoSpecLines.append(line)
                t=line.split(',')
                for j in range(len(t)):
                    t[j]=float(t[j])

                (distance,gobjid,self.zooNoSpecObjID[i],self.zooNoSpecDr7ObjID[i],self.zooNoSpecNvote[i],self.zooNoSpecPel[i],self.zooNoSpecPcs[i],self.zooNoSpecPacw[i],self.zooNoSpecPedge[i],self.zooNoSpecPdk[i],self.zooNoSpecPmg[i],self.zooNoSpecPcs[i])=t
                i += 1
        outfile.close()

        self.zooNoSpecDict=dict((a,b) for a,b in zip(self.zooNoSpecDr7ObjID,arange(len(self.zooNoSpecDr7ObjID))))


names=['MKW11','MKW8','AWM4','A2063','A2052','NGC6107','Coma','A1367','Hercules',]

for ncl in range(len(names)):
#for ncl in range(3):
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
    print 'running for ',cl.prefix
    cl.readGalaxyZooSpecCat()
    cl.getsdsscoords()
    cl.readGalaxyZooNoSpecCat()
    cl.getsdssphotcoords()

