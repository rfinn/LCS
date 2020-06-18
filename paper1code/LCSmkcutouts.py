#!/usr/bin/env python
"""


"""

from pylab import *
from pyraf import iraf
import pyfits
import sqlcl
import glob
import os
import mystuff as my
import ReadAGCsav
from LCScommon import *
from matplotlib.backends.backend_pdf import PdfFile

#cutoutpath='/home/rfinn/research/LocalClusters/cutouts/'
cutoutpath='/home/rfinn/research/LocalClusters/cutouts/'

		
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

class mpacatalog:
    def __init__(self):
	    infile='/home/rfinn/research/MPAJHUcatalogs/gal_line_dr7_v5_2.fit'
	    hdulist=pyfits.open(infile)
	    tbdata=hdulist[1].data
	    hdulist.close()
	    self.plateid=tbdata.field('PLATEID')
	    self.fiberid=tbdata.field('FIBERID')
	    self.halpha=tbdata.field('H_ALPHA_FLUX')
	    self.hbeta=tbdata.field('H_BETA_FLUX')
	    self.hdelta=tbdata.field('H_DELTA_FLUX')
	    self.n2=tbdata.field('NII_6584_FLUX')
	    self.o3=tbdata.field('OIII_5007_FLUX')
	    self.halphaEW=tbdata.field('H_ALPHA_EQW')
	    self.hbetaEW=tbdata.field('H_BETA_EQW')
	    self.hdeltaEW=tbdata.field('H_DELTA_EQW')
	    self.n2EW=tbdata.field('NII_6584_EQW')
	    self.o3EW=tbdata.field('OIII_5007_EQW')
    

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


        self.cdMpc=self.biweightvel/H0
        self.cdcm=self.cdMpc*3.e24
        self.csigma=self.biweightscale
        self.mcl=my.clusterMass(self.csigma,self.cz,h)
        self.AngDistance=my.DA(self.cz,h)


        self.drcut=3.#get galaxies w/in 3 degrees
	self.cutoutpath=cutoutpath+self.prefix+'/'
        #if self.prefix.find('A2063')>-1:
        #    self.dr=5.

#mipsimage=['/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits']
        self.image24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_minus_median_extract.fits'
	self.noise24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_unc.fits'
        self.sdssrim1='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R1.fits'
        self.sdssrim2='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R2.fits'
        self.sdssgim1='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'_g_R1.fits'
        self.sdssgim2='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'_g_R2.fits'
        self.sdssrim1skysub='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R1s.fits'
        self.sdssrim2skysub='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R2s.fits'
        self.rotatedimage24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24umWCS/'+self.prefix+'-WCS-mosaic_minus_median_extract.fits'
#	self.rotatedimage24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+'r'+self.prefix+'-rotated-24.fits'
	if ncl > 6:#for Abell 1367 and Hercules cluster
		self.image24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+self.prefix+'ch1r1_mosaic_minus_median_extract.fits'
		self.noise24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+self.prefix+'ch1r1_mosaic_unc.fits'
		self.sdssrim1='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'_SDSSr_mosaic.fits'
		self.sdssrim2='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'_SDSSr_mosaic.fits'
		self.sdssrim1skysub='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'_SDSSr_mosaics.fits'
		self.sdssrim2skysub='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'_SDSSr_mosaics.fits'

	self.imagepath24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'
	self.sdssimagepath='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'

	t=self.sdssrim1.split('.')
	self.sdsstestcat1=t[0]+'-test.cat'

    	t=self.sdssrim2.split('.')
	self.sdsstestcat2=t[0]+'-test.cat'
	self.testcat24=self.imagepath24+self.prefix+'-24um-test.cat'
	

    def localdensity(self,x,y,v,xref,yref,vref,n1,n2):#find local density, using nearest neighbors n1 through n2
	    DA=my.DA(self.cz,h)
	    sigma=zeros(len(x),'d')
	    for i in range(len(x)):
		    deltav=abs(v[i]-vref)
		    d=sqrt((x[i]-xref)**2+(y[i]-yref)**2)*3600./1000.*DA#d in Mpc
		    #d=d[deltav<1500]#apply a velocity cut of +/- 1500 km/s
		    d.sort()#sort in ascending order, zeroth element is distance from galaxy to itself
		    sig=0
		    if len(d) < n2:
			    print 'not enough points to calculate local density'
			    print 'only have ',len(d),' galaxies w/in 1500 km/s'
			    if len(d) < n1:
				    print 'ut oh!'
			    for j in range(n1,len(d)):
				    sig += (1.*j)/(d[j])**2

		    else:
			    for j in range(n1,n2+1):
				    sig += (1.*j)/(d[j])**2
		    sigma[i]=1./(4.*pi)*sig
	    return sigma

    def readSkidmore(self,maryfile):

	ngal2=0
	outfile2 = open(maryfile, 'r')
	for line in outfile2:
	    #print line
	    if line.startswith('A') or line.startswith(','):
		continue
	    ngal2 += 1
	outfile2.close()
	print "number from Mary's catalog = ",ngal2
	self.agcname=[]
	self.RAHI = zeros(ngal2,'f')
	self.DecHI = zeros(ngal2,'f')
	self.cz = zeros(ngal2,'f')
	self.D3 = zeros(ngal2,'f')
	self.D6 = zeros(ngal2,'f')
	self.den = zeros(ngal2,'f')
	self.btype= zeros(ngal2,'f')
	self.HImass= zeros(ngal2,'f')
	self.HIflag= zeros(ngal2,'i')
	self.g= zeros(ngal2,'f')
	self.r= zeros(ngal2,'f')
	self.i= zeros(ngal2,'f')
	self.mstargr= zeros(ngal2,'f')
	self.mstarri= zeros(ngal2,'f')
	self.mdynam= zeros(ngal2,'f')
	self.Bdiam= zeros(ngal2,'f')


	outfile3=open('/home/rfinn/research/LocalClusters/RegionsFiles/MKW11.HIpos.reg','w')
	s='global color=cyan font="helvetica 10 normal" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source \n'
	outfile3.write(s)
	s='fk5 \n'
	outfile3.write(s)

	outfile2 = open(maryfile,'r')
	i=0
	for line in outfile2:
	    if line.startswith('A') or line.startswith(','):
		continue
	    t=line.split(',')
	    self.agcname.append(t[0])
	    self.RAHI[i] = float(t[1])
	    self.DecHI[i] = float(t[2])
	    self.cz[i] = float(t[3])
	    self.D3[i] = float(t[4])
	    self.D6[i] = float(t[5])
	    self.den[i] = float(t[6])
	    self.btype[i] = float(t[7])
	    try:
		self.HImass[i] = float(t[8])
		self.HIflag[i]=1
	    except:
		self.HImass[i] = -99
		self.HIflag[i]=0
	    self.g[i] = float(t[9])
	    self.r[i] = float(t[10])
	    self.i[i] = float(t[11])
	    self.mstargr[i] = float(t[12])
	    self.mstarri[i] = float(t[13])
	    tm=t[14]
	    if tm.find('NAN') > -1:
		self.mdynam[i] = -99
	    elif tm.find('nan') > -1:
		self.mdynam[i] = -99
	    else:
		self.mdynam[i] = float(t[14])

	    self.Bdiam[i] = float(t[15])
	    s='circle(%12.8f, %12.8f, 10") \n'%(self.RAHI[i],self.DecHI[i])
	    outfile3.write(s)
	    i += 1
	outfile2.close()
	outfile3.close()

        self.id=id
	s=self.prefix+'.radec'
	self.incoords=s
	out1=open(s,'w')
        #print self.sdssRA
	for i in range(len(self.RAHI)):
	    s='%f %f %s\n'%(self.RAHI[i],self.DecHI[i],self.agcname[i])
	    out1.write(s)
	out1.close()





    def mksdssregfile(self):

	s=self.prefix+'.sdssspec.reg'
	out1=open(s,'w')
	out1.write("global color=blue font=\"helvetica 10 normal\" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\nfk5\n")

        n=str(self.prefix)+'galaxy.dat'
        outfile=open(n,'r')
        for line in outfile:
                t=line.split(',')
                string1 = "circle(%12.8f, %12.8f, 10\") \n"%(float(t[1]),float(t[2]))
                out1.write(string1)
        outfile.close()
        out1.close()

    def getsdsscoords(self): #read in sdss catalogs
	    
        #n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and (g.PrimTarget & 0x00000040) > 0 and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)
        ngal=0
        # updating to read in catalogs w/galaxy zoo info
	#z 	n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.dat'
	n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.WithZoo.dat'
        outfile=open(n,'r')
        for line in outfile:
            ngal += 1
        outfile.close()

	print 'got ',ngal,' galaxies in sdss spec catalog'
        self.sdssdr=zeros(ngal,'d')
        self.sdssRA=zeros(ngal,'d')
        self.sdssDec=zeros(ngal,'d')
        self.sdssu=zeros(ngal,'f')
        self.sdssg=zeros(ngal,'f')
        self.sdssr=zeros(ngal,'f')
        self.sdssi=zeros(ngal,'f')
        self.sdssz=zeros(ngal,'f')
        self.sdssspecz=zeros(ngal,'f')
        self.sdssHaEW=zeros(ngal,'f')
        self.sdssHaEWerr=zeros(ngal,'f')
	self.sdssplate=zeros(ngal,'L')
	self.sdssfiberID=zeros(ngal,'L')
	self.sdsstile=zeros(ngal,'L')
	self.sdssobjid=zeros(ngal,'L')

	#added these for some extra SDSS parameters 5/9/12
	self.sdsspetroMag_u=zeros(ngal,'f')
	self.sdsspetroMag_g=zeros(ngal,'f')
	self.sdsspetroMag_r=zeros(ngal,'f')
	self.sdsspetroMag_i=zeros(ngal,'f')
	self.sdsspetroMag_z=zeros(ngal,'f')
	self.sdsspetroRad_u=zeros(ngal,'f')
	self.sdsspetroRad_g=zeros(ngal,'f')
	self.sdsspetroRad_r=zeros(ngal,'f')
	self.sdsspetroRad_i=zeros(ngal,'f')
	self.sdsspetroRad_z=zeros(ngal,'f')
	self.sdsspetroR50_u=zeros(ngal,'f')
	self.sdsspetroR50_g=zeros(ngal,'f')
	self.sdsspetroR50_r=zeros(ngal,'f')
	self.sdsspetroR50_i=zeros(ngal,'f')
	self.sdsspetroR50_z=zeros(ngal,'f')
	self.sdsspetroR90_u=zeros(ngal,'f')
	self.sdsspetroR90_g=zeros(ngal,'f')
	self.sdsspetroR90_r=zeros(ngal,'f')
	self.sdsspetroR90_i=zeros(ngal,'f')
	self.sdsspetroR90_z=zeros(ngal,'f')
	self.sdssisoA_r=zeros(ngal,'f')
	self.sdssisoB_r=zeros(ngal,'f')
	self.sdssisoPhi_r=zeros(ngal,'f')
	self.sdssisoPhiErr_r=zeros(ngal,'f')
	self.sdssdeVRad_r=zeros(ngal,'f')
	self.sdssdeVRadErr_r=zeros(ngal,'f')
	self.sdssdeVPhi_r=zeros(ngal,'f')
	self.sdssdeVPhiErr_r=zeros(ngal,'f')
	self.sdssdeVMag_r=zeros(ngal,'f')
	self.sdssexpRad_r=zeros(ngal,'f')
	self.sdssexpRadErr_r=zeros(ngal,'f')
	self.sdssexpAB_r=zeros(ngal,'f')
	self.sdssexpABErr_r=zeros(ngal,'f')
	self.sdssexpPhi_r=zeros(ngal,'f')
	self.sdssexpPhiErr_r=zeros(ngal,'f')
	self.sdssexpMag_r=zeros(ngal,'f')
	self.sdssexpMagErr_r=zeros(ngal,'f')
	self.sdssextinction_u=zeros(ngal,'f')
	self.sdssextinction_g=zeros(ngal,'f')
	self.sdssextinction_r=zeros(ngal,'f')
	self.sdssextinction_i=zeros(ngal,'f')
	self.sdssextinction_z=zeros(ngal,'f')
	self.sdssdered_u=zeros(ngal,'f')
	self.sdssdered_g=zeros(ngal,'f')
	self.sdssdered_r=zeros(ngal,'f')
	self.sdssdered_i=zeros(ngal,'f')
	self.sdssdered_z=zeros(ngal,'f')

	#added these sdss params on 9.11.12 (to use when grabbing images from DAS)
	self.sdssrun=zeros(ngal,'i')
	self.sdssrerun=zeros(ngal,'i')
	self.sdsscamcol=zeros(ngal,'i')
	self.sdssfield=zeros(ngal,'i')

	#added these sdss params on 11.28.12 (ccd row and col used to reconstruct psf)
	self.sdsserr_u=zeros(ngal,'f')
	self.sdsserr_g=zeros(ngal,'f')
	self.sdsserr_r=zeros(ngal,'f')
	self.sdsserr_i=zeros(ngal,'f')
	self.sdsserr_z=zeros(ngal,'f')
	self.sdssrowc_u=zeros(ngal,'f')
	self.sdssrowc_g=zeros(ngal,'f')
	self.sdssrowc_r=zeros(ngal,'f')
	self.sdssrowc_i=zeros(ngal,'f')
	self.sdssrowc_z=zeros(ngal,'f')
	self.sdsscolc_u=zeros(ngal,'f')
	self.sdsscolc_g=zeros(ngal,'f')
	self.sdsscolc_r=zeros(ngal,'f')
	self.sdsscolc_i=zeros(ngal,'f')
	self.sdsscolc_z=zeros(ngal,'f')


	#added these for galaxy zoo

        self.zooSpecFlag=zeros(ngal,'i')
	self.zooSpecObjID=zeros(ngal,'L')
        self.zooSpecDr7ObjID=zeros(ngal,'L')
        self.zooSpecNvote=zeros(ngal,'f')
        self.zooSpecPel=zeros(ngal,'f')
        self.zooSpecPcw=zeros(ngal,'f')
        self.zooSpecPacw=zeros(ngal,'f')
        self.zooSpecPedge=zeros(ngal,'f')
	# 90 line
        self.zooSpecPdk=zeros(ngal,'f')
        self.zooSpecPmg=zeros(ngal,'f')
        self.zooSpecPcs=zeros(ngal,'f')
        self.zooSpecPelDebiased=zeros(ngal,'f')
	self.zooSpecPcsDebiased=zeros(ngal,'f')
	self.zooSpecSpiral=zeros(ngal,'f')
	self.zooSpecElliptical=zeros(ngal,'f')
	self.zooSpecUncertain=zeros(ngal,'i')

	# IntegerFields=array([12,13,14,15,63,64,65,66,67,69,70],'i')
	# after adding row and column info and magnitude errors
	IntegerFields=array([12,13,14,15,63,64,65,66, 82,84,85],'i')	
	IntegerFields=IntegerFields-1#convert to zero index
	
        outfile=open(n,'r')
        i=0
        id=[]
        for line in outfile:
                t=line.split(',')
                for j in range(len(t)):
			t[j]=float(t[j])
			#			if j in IntegerFields:
			#	print j,t[j]
			#		t[j]=int(t[j])
			#else:

		#print t
		#print 'length of t = ',len(t)
                #(self.sdssdr[i],self.sdssRA[i],self.sdssDec[i],self.sdssu[i],self.sdssg[i],self.sdssr[i],self.sdssi[i],self.sdssz[i],self.sdssspecz[i],self.sdssHaEW[i],self.sdssHaEWerr[i],self.sdssplate[i],self.sdssfiberID[i],self.sdsstile[i],self.sdssobjid[i])=t
		#        query="select n.distance,g.ra,g.dec, 
		(self.sdssdr[i],self.sdssRA[i],self.sdssDec[i])=t[0:3]
		#g.u, g.g, g.r, g.i, g.z, 
		(self.sdssu[i],self.sdssg[i],self.sdssr[i],self.sdssi[i],self.sdssz[i])=t[3:8]
		#s.z,l.ew,l.ewErr, 
		(self.sdssspecz[i],self.sdssHaEW[i],self.sdssHaEWerr[i])=t[8:11]
		#s.plate, s.fiberID, s.tile, g.objID,  
		(self.sdssplate[i],self.sdssfiberID[i],self.sdsstile[i],self.sdssobjid[i])=t[11:15]
		#g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,
		(self.sdsspetroMag_u[i],self.sdsspetroMag_g[i],self.sdsspetroMag_r[i],self.sdsspetroMag_i[i],self.sdsspetroMag_z[i])=t[15:20]
		#g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, 
		(self.sdsspetroRad_u[i],self.sdsspetroRad_g[i],self.sdsspetroRad_r[i],self.sdsspetroRad_i[i],self.sdsspetroRad_z[i])=t[20:25]
		#g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, 
		(self.sdsspetroR50_u[i],self.sdsspetroR50_g[i],self.sdsspetroR50_r[i],self.sdsspetroR50_i[i],self.sdsspetroR50_z[i])=t[25:30]
		#g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, 
		(self.sdsspetroR90_u[i],self.sdsspetroR90_g[i],self.sdsspetroR90_r[i],self.sdsspetroR90_i[i],self.sdsspetroR90_z[i])=t[30:35]
		#g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, 
		(self.sdssisoA_r[i],self.sdssisoB_r[i],self.sdssisoPhi_r[i],self.sdssisoPhiErr_r[i])=t[35:39]
		#g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, 
		(self.sdssdeVRad_r[i],self.sdssdeVRadErr_r[i],self.sdssdeVPhi_r[i],self.sdssdeVPhiErr_r[i],self.sdssdeVMag_r[i])=t[39:44]
		#g.expRad_r, g.expRadErr_r, g.expAB_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, 
		(self.sdssexpRad_r[i],self.sdssexpRadErr_r[i],self.sdssexpAB_r[i],self.sdssexpABErr_r[i],self.sdssexpPhi_r[i],self.sdssexpPhiErr_r[i],self.sdssexpMag_r[i],self.sdssexpMagErr_r[i])=t[44:52]
		#g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z, 
		(self.sdssextinction_u[i],self.sdssextinction_g[i],self.sdssextinction_r[i],self.sdssextinction_i[i],self.sdssextinction_z[i],self.sdssdered_u[i],self.sdssdered_g[i],self.sdssdered_r[i],self.sdssdered_i[i],self.sdssdered_z[i])=t[52:62]
		#g.run, g.rerun, g.camcol, g.field,
		(self.sdssrun[i],self.sdssrerun[i],self.sdsscamcol[i],self.sdssfield[i])=t[62:66]
		#g.err_u,g.err_g,g.err_r,g.err_i,g.err_z, 
		(self.sdsserr_u[i],self.sdsserr_g[i],self.sdsserr_r[i],self.sdsserr_i[i],self.sdsserr_z[i])=t[66:71]
		#g.rowc_u, g.rowc_g, g.rowc_r,g.rowc_i,g.rowc_z,
		(self.sdssrowc_u[i],self.sdssrowc_g[i],self.sdssrowc_r[i],self.sdssrowc_i[i],self.sdssrowc_z[i])=t[71:76]
		 #g.colc_u,g.colc_g,g.colc_r,g.colc_i,g.colc_z from 
		(self.sdsscolc_u[i],self.sdsscolc_g[i],self.sdsscolc_r[i],self.sdsscolc_i[i],self.sdsscolc_z[i])=t[76:81]
		#print len(t[81:]),t[81:]
		#print (t[79:])
		(self.zooSpecFlag[i],distance,self.zooSpecObjID[i],self.zooSpecDr7ObjID[i],self.zooSpecNvote[i],self.zooSpecPel[i],self.zooSpecPcw[i],self.zooSpecPacw[i],self.zooSpecPedge[i],self.zooSpecPdk[i],self.zooSpecPmg[i],self.zooSpecPcs[i],self.zooSpecPelDebiased[i],self.zooSpecPcsDebiased[i],self.zooSpecSpiral[i],self.zooSpecElliptical[i],self.zooSpecUncertain[i])=t[81:]

#		(self.zooSpecFlag[i],distance,self.zooSpecObjID[i],self.zooSpecDr7ObjID[i],self.zooSpecNvote[i])=t[62:67]
#		(self.zooSpecPel[i],self.zooSpecPcw[i],self.zooSpecPacw[i],self.zooSpecPedge[i],self.zooSpecPdk[i],self.zooSpecPmg[i],self.zooSpecPcs[i])=t[67:74]
#		(self.zooSpecPelDebiased[i],self.zooSpecPcsDebiased[i],self.zooSpecSpiral[i],self.zooSpecElliptical[i],self.zooSpecUncertain[i])=t[74:]
        #query="select n.distance, g.objID, z.dr7objid, z.nvote, z.p_el, z.p_cw,z.p_acw,z.p_edge,z.p_dk,z.p_mg,z.p_cs,z.p_el_debiased,z.p_cs_debiased,z.spiral,z.elliptical,z.uncertain from galaxy g, specobj s, zooSpec z, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and z.objid = g.objid and s.z < %5.4f and s.z > %5.4f" % (self.cra,self.cdec,drsearch,zmax,zmin)
		s=str(i)
		id.append(s)

                    #print 'Failure in naming convention'
                i += 1
        outfile.close()
        self.id=id
	s=self.prefix+'.sdss.radec'
	self.incoords=s
	out1=open(s,'w')
        #print self.sdssRA
	for i in range(len(self.sdssRA)):
	    s='%f %f %s\n'%(self.sdssRA[i],self.sdssDec[i],self.id[i])
	    out1.write(s)
	out1.close()

	##### left off here - match to MPA catalogs and add line fluxes and MPAmatchflag to spec data
	matchsdss2mpaflag=zeros(len(self.sdssRA),'i')#keep track of sdss galaxies that are matched to sdss spec
	matchsdss2mpaindex=zeros(len(self.sdssRA),'i')#keep track of index of sdss galaxies that are matched to sdss spec
	delta=0.1 #fiber ID and plate ID must match
        for i in range(len(self.sdssRA)):
		imatch, matchflag,nmatch= findnearest(1.*self.sdssplate[i],1.*self.sdssfiberID[i],1.*mpa.plateid,1.*mpa.fiberid,delta)
		if matchflag > 0.1:
			matchsdss2mpaflag[i]=1
			try:
				matchsdss2mpaindex[i]=imatch
			except ValueError:
				print 'problem w/MPA matching', nmatch,imatch,matchflag
				if nmatch > 0:
					matchsdss2mpaindex[i]=imatch[0]
	self.matchsdss2mpaflag=matchsdss2mpaflag
	self.sdssHalpha=mpa.halpha[matchsdss2mpaindex]
	self.sdssHbeta=mpa.hbeta[matchsdss2mpaindex]
	self.sdssN2=mpa.n2[matchsdss2mpaindex]
	self.sdssO3=mpa.o3[matchsdss2mpaindex]
	self.sdssHdeltaEW=mpa.hdeltaEW[matchsdss2mpaindex]


    def readapexcat(self):
        ngal=0
        n='/home/rfinn/research/LocalClusters/ApexFinalCatalogs/'+str(self.prefix)+'mosaic_extract.tbl'
        outfile=open(n,'r')
        for line in outfile:
		if line.startswith('#'):
			continue
		if line.startswith('\\'):
			continue
		if line.startswith('|'):
			continue
		ngal += 1
        outfile.close()

	print 'got ',ngal,' galaxies in Apex catalog'
	## left off here.  going to read in apex cat, match to agc, include some apex
	## fields in the master table
        self.mipsRA=zeros(ngal,'d')
        self.mipsDec=zeros(ngal,'d')
        self.mipsx=zeros(ngal,'f')
        self.mipsy=zeros(ngal,'f')
        self.mipsflux=zeros(ngal,'f')
        self.mipsfluxerr=zeros(ngal,'f')
        self.mipsap1=zeros(ngal,'f')
        self.mipsap2=zeros(ngal,'f')
        self.mipsap3=zeros(ngal,'f')
	self.mipsap4=zeros(ngal,'f')
	self.mipsap5=zeros(ngal,'f')
        self.mipsap1err=zeros(ngal,'f')
        self.mipsap2err=zeros(ngal,'f')
        self.mipsap3err=zeros(ngal,'f')
	self.mipsap4err=zeros(ngal,'f')
	self.mipsap5err=zeros(ngal,'f')
	self.mipsSNR=zeros(ngal,'f')
	self.mipsdeblend=[]



        outfile=open(n,'r')
        i=0
        id=[]
        for line in outfile:
		if line.startswith('#'):
			continue
		if line.startswith('\\'):
			continue
		if line.startswith('|'):
			continue

                t=line.split()
                for j in range(len(t)) :
			if j == 22:
				continue
			try:
				t[j]=float(t[j])
			except ValueError:
				print "Error converting value to float in apex cat"
				print 'j = ', j,"Value = ",t[j]
				#print line
				   


		self.mipsRA[i]=t[3]
		self.mipsDec[i]=t[5]
		self.mipsx[i]=t[8]
		self.mipsy[i]=t[10]
		self.mipsflux[i]=t[13]
		self.mipsfluxerr[i]=t[14]
		self.mipsSNR[i]=t[18]
		self.mipsdeblend.append(t[22])
		self.mipsap1[i]=t[23]
		self.mipsap2[i]=t[24]
		self.mipsap3[i]=t[25]
		self.mipsap4[i]=t[26]
		self.mipsap5[i]=t[27]
		self.mipsap1err[i]=t[33]
		self.mipsap2err[i]=t[34]
		self.mipsap3err[i]=t[35]
		self.mipsap4err[i]=t[36]
		self.mipsap5err[i]=t[37]
                i += 1
        outfile.close()
	self.mipsdeblend=array(self.mipsdeblend,'string')
        self.id=id
	s=self.prefix+'.mips.radec'
	out1=open(s,'w')
        #print self.sdssRA
	for i in range(len(self.mipsRA)):
	    s='%f %f\n'%(self.mipsRA[i],self.mipsDec[i])
	    out1.write(s)
	out1.close()


    def getsdssphotcoords(self):
        #n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and (g.PrimTarget & 0x00000040) > 0 and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)
        ngal=0
        # updating to include galaxy zoo info
	#z n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.dat'
	n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.photcat.WithZoo.dat'
        outfile=open(n,'r')
        for line in outfile:
		t=line.split(',')
		if float(t[8]) > 1: #keeps only objects w/out spectra
			continue
		
		ngal += 1
        outfile.close()

	print 'got ',ngal,' galaxies in sdss phot catalog'
        self.sdssphotdr=zeros(ngal,'f')
        self.sdssphotRA=zeros(ngal,'d')
        self.sdssphotDec=zeros(ngal,'d')
        self.sdssphotu=zeros(ngal,'f')
        self.sdssphotg=zeros(ngal,'f')
        self.sdssphotr=zeros(ngal,'f')
        self.sdssphoti=zeros(ngal,'f')
        self.sdssphotz=zeros(ngal,'f')
	self.sdssphotobjid=zeros(ngal,'f')



	#added these for some extra SDSSPHOT parameters 5/9/12
	self.sdssphotpetroMag_u=zeros(ngal,'f')
	self.sdssphotpetroMag_g=zeros(ngal,'f')
	self.sdssphotpetroMag_r=zeros(ngal,'f')
	self.sdssphotpetroMag_i=zeros(ngal,'f')
	self.sdssphotpetroMag_z=zeros(ngal,'f')
	self.sdssphotpetroRad_u=zeros(ngal,'f')
	self.sdssphotpetroRad_g=zeros(ngal,'f')
	self.sdssphotpetroRad_r=zeros(ngal,'f')
	self.sdssphotpetroRad_i=zeros(ngal,'f')
	self.sdssphotpetroRad_z=zeros(ngal,'f')
	self.sdssphotpetroR50_u=zeros(ngal,'f')
	self.sdssphotpetroR50_g=zeros(ngal,'f')
	self.sdssphotpetroR50_r=zeros(ngal,'f')
	self.sdssphotpetroR50_i=zeros(ngal,'f')
	self.sdssphotpetroR50_z=zeros(ngal,'f')
	self.sdssphotpetroR90_u=zeros(ngal,'f')
	self.sdssphotpetroR90_g=zeros(ngal,'f')
	self.sdssphotpetroR90_r=zeros(ngal,'f')
	self.sdssphotpetroR90_i=zeros(ngal,'f')
	self.sdssphotpetroR90_z=zeros(ngal,'f')
	self.sdssphotisoA_r=zeros(ngal,'f')
	self.sdssphotisoB_r=zeros(ngal,'f')
	self.sdssphotisoPhi_r=zeros(ngal,'f')
	self.sdssphotisoPhiErr_r=zeros(ngal,'f')
	self.sdssphotdeVRad_r=zeros(ngal,'f')
	self.sdssphotdeVRadErr_r=zeros(ngal,'f')
	self.sdssphotdeVPhi_r=zeros(ngal,'f')
	self.sdssphotdeVPhiErr_r=zeros(ngal,'f')
	self.sdssphotdeVMag_r=zeros(ngal,'f')
	self.sdssphotexpRad_r=zeros(ngal,'f')
	self.sdssphotexpRadErr_r=zeros(ngal,'f')
	self.sdssphotexpAB_r=zeros(ngal,'f')
	self.sdssphotexpABErr_r=zeros(ngal,'f')
	self.sdssphotexpPhi_r=zeros(ngal,'f')
	self.sdssphotexpPhiErr_r=zeros(ngal,'f')
	self.sdssphotexpMag_r=zeros(ngal,'f')
	self.sdssphotexpMagErr_r=zeros(ngal,'f')
	self.sdssphotextinction_u=zeros(ngal,'f')
	self.sdssphotextinction_g=zeros(ngal,'f')
	self.sdssphotextinction_r=zeros(ngal,'f')
	self.sdssphotextinction_i=zeros(ngal,'f')
	self.sdssphotextinction_z=zeros(ngal,'f')
	self.sdssphotdered_u=zeros(ngal,'f')
	self.sdssphotdered_g=zeros(ngal,'f')
	self.sdssphotdered_r=zeros(ngal,'f')
	self.sdssphotdered_i=zeros(ngal,'f')
	self.sdssphotdered_z=zeros(ngal,'f')


	#added these sdss params on 9.11.12 (to use when grabbing images from DAS)
	self.sdssphotrun=zeros(ngal,'i')
	self.sdssphotrerun=zeros(ngal,'i')
	self.sdssphotcamcol=zeros(ngal,'i')
	self.sdssphotfield=zeros(ngal,'i')

	#added these sdss params on 11.28.12 (ccd row and col used to reconstruct psf)
	self.sdssphoterr_u=zeros(ngal,'f')
	self.sdssphoterr_g=zeros(ngal,'f')
	self.sdssphoterr_r=zeros(ngal,'f')
	self.sdssphoterr_i=zeros(ngal,'f')
	self.sdssphoterr_z=zeros(ngal,'f')
	self.sdssphotrowc_u=zeros(ngal,'f')
	self.sdssphotrowc_g=zeros(ngal,'f')
	self.sdssphotrowc_r=zeros(ngal,'f')
	self.sdssphotrowc_i=zeros(ngal,'f')
	self.sdssphotrowc_z=zeros(ngal,'f')
	self.sdssphotcolc_u=zeros(ngal,'f')
	self.sdssphotcolc_g=zeros(ngal,'f')
	self.sdssphotcolc_r=zeros(ngal,'f')
	self.sdssphotcolc_i=zeros(ngal,'f')
	self.sdssphotcolc_z=zeros(ngal,'f')


	self.zooNoSpecFlag=zeros(ngal,'i')
        self.zooNoSpecObjID=zeros(ngal,'f')
        self.zooNoSpecDr7ObjID=zeros(ngal,'L')
        self.zooNoSpecNvote=zeros(ngal,'f')
        self.zooNoSpecPel=zeros(ngal,'f')
        self.zooNoSpecPcw=zeros(ngal,'f')
        self.zooNoSpecPacw=zeros(ngal,'f')
        self.zooNoSpecPedge=zeros(ngal,'f')
        self.zooNoSpecPdk=zeros(ngal,'f')
        self.zooNoSpecPmg=zeros(ngal,'f')
        self.zooNoSpecPcs=zeros(ngal,'f')


        outfile=open(n,'r')
        i=0
        id=[]
        for line in outfile:
                t=line.split(',')
		if float(t[8]) > 1: #keeps only objects w/out spectra
			continue
                for j in range(len(t)):
                    t[j]=float(t[j])

                #self.sdssphotRA[i]=t[0]
		#self.sdssphotDec[i]=t[1]
		#self.sdssphotu[i]=t[2]
		#self.sdssphotg[i]=t[3]
		#self.sdssphotr[i]=t[4]
		#self.sdssphoti[i]=t[5]
		#self.sdssphotz[i]=t[6]
		#self.sdssphotobjid[i]=t[7]

                



		#print len(t)#,len(t[8:])
		#	query="select g.ra, g.dec, g.u, g.g, g.r, g.i, g.z, g.objid, g.specObjID, g.petroMag_u, g.petroMag_g, g.petroMag_r, g.petroMag_i, g.petroMag_z,g.petroRad_u, g.petroRad_g, g.petroRad_r, g.petroRad_i, g.petroRad_z, g.petroR50_u, g.petroR50_g, g.petroR50_r, g.petroR50_i, g.petroR50_z, g.petroR90_u, g.petroR90_g, g.petroR90_r, g.petroR90_i, g.petroR90_z, g.isoA_r, g.isoB_r, g.isoPhi_r, g.isoPhiErr_r, g.deVRad_r, g.deVRadErr_r, g.deVPhi_r, g.deVPhiErr_r, g.deVMag_r, g.expRad_r, g.expRadErr_r, g.expABErr_r, g.expPhi_r, g.expPhiErr_r, g.expMag_r, g.expMagErr_r, g.extinction_u,g.extinction_g,g.extinction_r,g.extinction_i,g.extinction_z, g.dered_u, g.dered_g, g.dered_r, g.dered_i, g.dered_z from galaxy g, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and  (g.PrimTarget & 0x00000040) > 0 and (g.specObjID = 0    )" % (self.cra,self.cdec,drsearch)#changed so that only galaxies w/out spectra are returned
		(self.sdssphotRA[i],self.sdssphotDec[i],self.sdssphotu[i],self.sdssphotg[i],self.sdssphotr[i],self.sdssphoti[i],self.sdssphotz[i],self.sdssphotobjid[i],junk0,self.sdssphotpetroMag_u[i],self.sdssphotpetroMag_g[i],self.sdssphotpetroMag_r[i],self.sdssphotpetroMag_i[i],self.sdssphotpetroMag_z[i],self.sdssphotpetroRad_u[i],self.sdssphotpetroRad_g[i],self.sdssphotpetroRad_r[i],self.sdssphotpetroRad_i[i],self.sdssphotpetroRad_z[i],self.sdssphotpetroR50_u[i],self.sdssphotpetroR50_g[i],self.sdssphotpetroR50_r[i],self.sdssphotpetroR50_i[i],self.sdssphotpetroR50_z[i],self.sdssphotpetroR90_u[i],self.sdssphotpetroR90_g[i],self.sdssphotpetroR90_r[i],self.sdssphotpetroR90_i[i],self.sdssphotpetroR90_z[i],self.sdssphotisoA_r[i],self.sdssphotisoB_r[i],self.sdssphotisoPhi_r[i],self.sdssphotisoPhiErr_r[i],self.sdssphotdeVRad_r[i],self.sdssphotdeVRadErr_r[i],self.sdssphotdeVPhi_r[i],self.sdssphotdeVPhiErr_r[i],self.sdssphotdeVMag_r[i],self.sdssphotexpRad_r[i],self.sdssphotexpRadErr_r[i],self.sdssphotexpAB_r[i],self.sdssphotexpABErr_r[i],self.sdssphotexpPhi_r[i],self.sdssphotexpPhiErr_r[i],self.sdssphotexpMag_r[i],self.sdssphotexpMagErr_r[i],self.sdssphotextinction_u[i],self.sdssphotextinction_g[i],self.sdssphotextinction_r[i],self.sdssphotextinction_i[i],self.sdssphotextinction_z[i],self.sdssphotdered_u[i],self.sdssphotdered_g[i],self.sdssphotdered_r[i],self.sdssphotdered_i[i],self.sdssphotdered_z[i],self.sdssphotrun[i],self.sdssphotrerun[i],self.sdssphotcamcol[i],self.sdssphotfield[i],self.sdssphoterr_u[i],self.sdssphoterr_g[i],self.sdssphoterr_r[i],self.sdssphoterr_i[i],self.sdssphoterr_z[i],self.sdssphotrowc_u[i],self.sdssphotrowc_g[i],self.sdssphotrowc_r[i],self.sdssphotrowc_i[i],self.sdssphotrowc_z[i],self.sdssphotcolc_u[i],self.sdssphotcolc_g[i],self.sdssphotcolc_r[i],self.sdssphotcolc_i[i],self.sdssphotcolc_z[i],self.zooNoSpecFlag[i],distance,gobjid,self.zooNoSpecObjID[i],self.zooNoSpecDr7ObjID[i],self.zooNoSpecNvote[i],self.zooNoSpecPel[i],self.zooNoSpecPcw[i],self.zooNoSpecPacw[i],self.zooNoSpecPedge[i],self.zooNoSpecPdk[i],self.zooNoSpecPmg[i],self.zooNoSpecPcs[i])=t

	#query="select  n.distance, g.objID, z.specobjid, z.dr7objid, z.nvote, z.p_el, z.p_cw,z.p_acw,z.p_edge,z.p_dk,z.p_mg,z.p_cs from galaxy g, zooNoSpec z, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objID = n.objID and z.objid = g.objid " % (self.cra,self.cdec,drsearch)

		s=str(i)
		id.append(s)

                    #print 'Failure in naming convention'
                i += 1
        outfile.close()
        self.photid=id
	s=self.prefix+'.sdssphot.radec'
	self.incoords=s
	out1=open(s,'w')
        #print self.sdssRA
	for i in range(len(self.sdssphotRA)):
	    s='%f %f %s\n'%(self.sdssphotRA[i],self.sdssphotDec[i],self.photid[i])
	    out1.write(s)
	out1.close()


 
    def getcoords(self,infile):

	in1=open(infile,'r')
        ngal=0
	for line in in1:
            if line.find('#') >-1:
                continue
            ngal += 1
        in1.close()

        self.RA=zeros(ngal,'d')
        self.Dec=zeros(ngal,'d')
        self.vr=zeros(ngal,'f')

	in1=open(infile,'r')
        i=0
	for line in in1:
            if line.find('#') >-1:
                continue
            t=line.split()
            self.RA[i]=float(t[1])
            self.Dec[i]=float(t[2])
            self.vr[i]=float(t[3])
            i += 1
        in1.close()


	s=self.prefix+'.MCOmemb.reg'
	out1=open(s,'w')
	out1.write("global color=green font=\"helvetica 10 normal\" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\nfk5\n")
	for i in range(len(self.RA)):
            string1 = "circle(%12.8f, %12.8f, 2\") \n"%(self.RA[i],self.Dec[i])
            out1.write(string1)
	out1.close()



    def match2agc(self): #matches sdss to agc and writes out first version of mastertable
        
	delta=3./3600.#match radius of 60 arcsec
        dagc=sqrt((self.cra-agc.radeg)**2+(self.cdec-agc.decdeg)**2)
        thisclflag=(dagc<self.drcut)#agc galaxis w/in 3 degrees from cluster center
        agcra=agc.radeg[thisclflag]
        agcdec=agc.decdeg[thisclflag]
	agcnumber=agc.agcnumber[thisclflag]
	matchsdss2agcflag=zeros(len(self.sdssRA),'i')#keep track of sdss galaxies that are matched to agc
	matchagc2sdssflag=zeros(len(agcra),'i')#keep track of agc galaxies that are matched to sdss spec
	matchagc2sdssindex=zeros(len(agcra),'i')#keep track of index of agc galaxies that are matched to sdss spec
	matchagc2sdssdelta=zeros(len(agcra),'f')
	matchagc2sdssphotflag=zeros(len(agcra),'i')#keep track of agc galaxies that are matched to sdss phot
	matchagc2sdssphotindex=zeros(len(agcra),'i')#keep track of index of agc galaxies that are matched to sdss spec
	matchagc2sdssdelta=zeros(len(agcra),'f')
        for i in range(len(agcra)):
		imatch, matchflag,nmatch= findnearest(agcra[i],agcdec[i],self.sdssRA,self.sdssDec,delta)
		if matchflag > 0.1:
			matchagc2sdssflag[i]=1
			matchagc2sdssindex[i]=imatch
			if matchsdsdss2agcflag[imatch] > .1:
				print 'warning!  multiple matches to sdss galaxy ',agcnumber[i],
			else:
				matchsdss2agcflag[imatch]=1
			matchagc2sdssdelta[i]=sqrt((agcra[i]-self.sdssRA[imatch])**2+(agcdec[i]-self.sdssDec[imatch)**2)*3600.
	for i in range(len(agcra)):#try matching to SDSS phot cat
		imatch, matchflag,nmatch= findnearest(agcra[i],agcdec[i],self.sdssphotRA,self.sdssphotDec,delta)
		if matchflag>0.1:
			matchagc2sdssphotflag[i]=1
			matchagc2sdssphotindex[i]=imatch
			matchagc2sdssphotdelta[i]=sqrt((agcra[i]-self.sdssphotRA[imatch])**2+(agcdec[i]-self.sdssphotDec[imatch])**2)*3600.
			# check to see if phot object is a closer match than spec object
			# if so, set spec match flags back to zero and print a warning!!!
			if (matchagc2sdssphotdelta[i] < match2gc2sdssdelta[i]):
			
			       print 'Warning! Found phot object closer than spec match for agc ',agcnumber[i]
			       matchagc2sdssflag[i]=0
			       matchsdss2agcflag[matchagc2sdssindex]=0
			     
			
	s='/home/rfinn/research/LocalClusters/MemberCatalogs/'+self.prefix+'AGCSDSS.dat'
	outfile=open(s,'w')
	fakeagcindex=-99999
	fra=[]
	fdec=[]
	coordflag=[]#indicates where ra and dec came from: 0 = agc, 1=sdss spec cat
	coordindex=[]#index in agc/sdss cat
	fakeindex=[]
	for i in range(len(agcra)):
		if (matchagc2sdssflag[i] > 0.1):
			j=matchagc2sdssindex[i]
			#agcra agcdec sdssra sdssdec agcflag sdssflag agcnumber final-ra final-dec
			s='%13.8f %13.8f %13.8f %13.8f 1 1 %6i %13.8f %13.8f \n'%(agcra[i],agcdec[i],self.sdssRA[j],self.sdssDec[j],agcnumber[i],agcra[i],agcdec[i])
			fra.append(agcra[i])
			fdec.append(agcdec[i])
			coordflag.append(0)
			coordindex.append(i)
			outfile.write(s)
		elif (matchagc2sdssflag[i] < 0.1):
			s='%13.8f %13.8f %13.8f %13.8f 1 0 %6i  %13.8f %13.8f \n'%(agcra[i],agcdec[i],-999.,-999.,agcnumber[i],agcra[i],agcdec[i])
			outfile.write(s)
			fra.append(agcra[i])
			fdec.append(agcdec[i])
			coordflag.append(0)
			coordindex.append(i)

	scoordflag=[]
	scoordindex=[]
	for i in range(len(self.sdssRA)):
		if (matchsdss2agcflag[i] < 0.1):
			s='%13.8f %13.8f %13.8f %13.8f 0 1 %6i  %13.8f %13.8f \n'%(-999.,-999.,self.sdssRA[i],self.sdssDec[i],fakeagcindex,self.sdssRA[i],self.sdssDec[i])
			fakeindex.append(fakeagcindex)
			fakeagcindex += 1
			outfile.write(s)
			fra.append(self.sdssRA[i])
			fdec.append(self.sdssDec[i])
			scoordflag.append(1)
			scoordindex.append(i)
	outfile.close()
	fra=array(fra,'f')
	fdec=array(fdec,'f')

	self.fra=fra
	scoordflag=array(scoordflag,'i')
	scoordindex=array(scoordindex,'i')
	ngaltot=len(fra)
	#get sextractor RA and Dec
	infile=self.sdsstestcat1
	hdulist=pyfits.open(infile)
	tbdata=hdulist[2].data
	hdulist.close()
	self.sexsdssRA=tbdata.field('ALPHA_J2000')
	self.sexsdssDec=tbdata.field('DELTA_J2000')
	self.sexsdssMagPetro=tbdata.field('MAG_PETRO')
	self.sexsdssMagAUTO=tbdata.field('MAG_AUTO')
	#match fra and fdec to sextractor r-band coords
	matchagc2sexsdssflag=zeros(len(fra),'i')#keep track of agc galaxies that are matched to sdss spec
	matchagc2sexsdssindex=zeros(len(fra),'i')#keep track of agc galaxies that are matched to sdss spec
        for i in range(len(fra)):
		imatch, matchflag,nmatch= findnearest(fra[i],fdec[i],self.sexsdssRA,self.sexsdssDec,delta)
		if matchflag > 0.1:
			matchagc2sexsdssflag[i]=1
			matchagc2sexsdssindex[i]=imatch

	self.matchagc2sexsdssflag=matchagc2sexsdssflag
	self.matchagc2sexsdssindex=matchagc2sexsdssindex


	delta=1./3600
	matchsdss2sexsdssflag=zeros(len(self.sdssRA))#keep track of SDSS spec galaxies that are matched to SE catalog
	matchsdss2sexsdssindex=zeros(len(self.sdssRA))#keep track of SDSS galaxies that are matched to SE catalog
        for i in range(len(self.sdssRA)):
		imatch, matchflag,nmatch= findnearest(self.sdssRA[i],self.sdssDec[i],self.sexsdssRA,self.sexsdssDec,delta)
		if matchflag > 0.1:
			matchsdss2sexsdssflag[i]=1
			matchsdss2sexsdssindex[i]=imatch

	self.matchsdss2sexsdssflag=matchsdss2sexsdssflag
	self.matchsdss2sexsdssindex=matchsdss2sexsdssindex




	pipeliner=[]
	petror=[]
	magautor=[]
	for i in range(len(self.sdssRA)):
		if matchsdss2sexsdssflag[i] > 0.1:
			pipeliner.append(self.sdssr[i])
			petror.append(self.sexsdssMagPetro[int(matchsdss2sexsdssindex[i])])
			magautor.append(self.sexsdssMagAUTO[int(matchsdss2sexsdssindex[i])])
	self.pipeliner=array(pipeliner,'f')
	self.petror=array(petror,'f')
	self.magautor=array(magautor,'f')

	
	#get sextractor RA and Dec
	infile=self.testcat24
	hdulist=pyfits.open(infile)
	tbdata24=hdulist[2].data
	hdulist.close()
	sex24RA=tbdata24.field('ALPHA_J2000')
	sex24Dec=tbdata24.field('DELTA_J2000')
	


	#match fra and fdec to sextractor 24um coords

	delta=3./3600
	matchagc2sex24flag=zeros(len(fra),'i')#keep track of agc galaxies that are matched to sdss spec
	matchagc2sex24index=zeros(len(fra),'i')#keep track of agc galaxies that are matched to sdss spec
        for i in range(len(fra)):
		imatch, matchflag,nmatch= findnearest(fra[i],fdec[i],sex24RA,sex24Dec,delta)
		if matchflag > 0.1:
			matchagc2sex24flag[i]=1
			matchagc2sex24index[i]=imatch
	self.matchagc2sex24flag=matchagc2sex24flag
	self.matchagc2sex24index=matchagc2sex24index


	#match fra to apex 24um photometry
	matchfra2mipsflag=zeros(len(fra),'i')#keep track of agc+sdss galaxies that are matched to apex-detected galaxies
	matchfra2mipsindex=zeros(len(fra),'i')#keep track of agc+sdss galaxies that are matched to apex-detected galaxies
	delta=3./3600#match radius in degrees
        for i in range(len(fra)):
		imatch, matchflag,nmatch= findnearest(fra[i],fdec[i],self.mipsRA,self.mipsDec,delta)
		if matchflag > 0.1:
			matchfra2mipsflag[i]=1
			matchfra2mipsindex[i]=imatch

	self.matchfra2mipsflag=matchfra2mipsflag
	self.matchfra2mipsindex=matchfra2mipsindex

	#match 


	nagc=len(agcra)
	nsdss=len(fra)-len(agcra)
	agcflag=array((ones(nagc).tolist()+(zeros(nsdss)).tolist()),'i')
	sdssflag=array((matchagc2sdssflag.tolist()+(ones(nsdss)).tolist()),'i')
	sdssphotflag=array((matchagc2sdssphotflag.tolist()+(ones(nsdss)).tolist()),'i')
	#write out a fits table
	c1=pyfits.Column(name='AGCflag',format='L',array=agcflag)
	self.agcflag=agcflag
	c2=pyfits.Column(name='SDSSflag',format='L',array=sdssflag)
	c3=pyfits.Column(name='SDSSphotflag',format='L',array=sdssphotflag)
	c4=pyfits.Column(name='APEXflag',format='L',array=self.matchfra2mipsflag)
	c5=pyfits.Column(name='SEXSDSSflag',format='L',array=matchagc2sexsdssflag)
	c6=pyfits.Column(name='SEX24FLAG',format='L',array=matchagc2sex24flag)
	agcnum=array((agcnumber.tolist()+fakeindex),'i')
	c7=pyfits.Column(name='AGCNUMBER',format='J',array=agcnum)
	c8=pyfits.Column(name='AGCRA',format='D',unit='DEG',array=fra)
	c9=pyfits.Column(name='AGCDEC',format='D',unit='DEG',array=fdec)

	t=array((agc.a100[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c10=pyfits.Column(name='A100',format='I',unit='arcmin*100',array=t)

    	t=array((agc.b100[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c11=pyfits.Column(name='B100',format='I',unit='arcmin*100',array=t)

    	t=array((agc.mag10[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c12=pyfits.Column(name='MAG10',format='I',unit='MAG',array=t)

	t=array((agc.posang[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c13=pyfits.Column(name='POSANG',format='E',unit='DEG',array=t)

	t=array((agc.bsteintype[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c14=pyfits.Column(name='BSTEINTYPE',format='I',array=t)
	t=array((agc.vopt[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c15=pyfits.Column(name='VOPT',format='I',unit='km/s',array=t)
	voptflag=(t>0)
	c16=pyfits.Column(name='AGCVOPTFLAG',format='L',unit='',array=t)


	t=array((agc.verr[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c17=pyfits.Column(name='VERR',format='I',unit='km/s',array=t)
	t=array((agc.vsource[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c18=pyfits.Column(name='VSOURCE',format='I',unit='Jy/(km/s)*100',array=t)
	t=array((agc.flux100[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c19=pyfits.Column(name='FLUX100',format='I',unit='Jy*100',array=t)

	HIflag=(t>0)

	c20=pyfits.Column(name='HIflag',format='L',array=HIflag)
	
	t=array((agc.rms100[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c21=pyfits.Column(name='RMS100',format='I',unit='km/s',array=t)
	t=array((agc.v21[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c22=pyfits.Column(name='V21',format='I',unit='km/s',array=t)
	t=array((agc.width[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c23=pyfits.Column(name='WIDTH',format='I',unit='km/s',array=t)
	t=array((agc.widtherr[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c24=pyfits.Column(name='WIDTHERR',format='I',unit='km/s',array=t)
	#define columns
	#t=array((agc.posang[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	#c21=pyfits.Column(name='POSANG',format='E',unit='DEG',array=t)
	
#	columnsSDSS=['SDSSRA','SDSSDEC','uMAG','gMAG','rMAG','iMAG','zMAG','SPECZ','HAEW','HAEWERR']
#	unitsSDSS=['deg','deg','MAG','MAG','MAG','MAG','MAG','v/cz','angstrom','angstrom']
#	typeSDSS=['D','D','E','E','E','E','E','E','E','E']

	t=array((self.sdssRA[matchagc2sdssindex].tolist()+self.sdssRA[scoordindex].tolist()),'f')
	c25=pyfits.Column(name='SDSSRA',format='E',unit='DEG',array=t)

	print "Checking array lengths - these should aall be the same"
	print "test c22", len(t)

	t=array((self.sdssDec[matchagc2sdssindex].tolist()+self.sdssDec[scoordindex].tolist()),'f')
	c26=pyfits.Column(name='SDSSDEC',format='E',unit='DEG',array=t)


	t=array(((matchagc2sdssflag*self.sdssRA[matchagc2sdssindex]) +(~matchagc2sdssflag*self.sdssphotRA[matchagc2sdssphotindex])).tolist()+self.sdssRA[scoordindex].tolist()),'f')
	c27=pyfits.Column(name='SDSSphotRA',format='E',unit='DEG',array=t)
	t=array(((matchagc2sdssflag*self.sdssDec[matchagc2sdssindex]) +(~matchagc2sdssflag*self.sdssphotDec[matchagc2sdssphotindex])).tolist()+self.sdssDec[scoordindex].tolist()),'f')
	c28=pyfits.Column(name='SDSSphotDEC',format='E',unit='DEG',array=t)


	t0=matchagc2sdssflag*self.sdssu[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotu[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdssu[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssg[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotg[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdssg[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssr[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotr[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdssr[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssi[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphoti[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdssi[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssz[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotz[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdssz[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c29=pyfits.Column(name='SDSSMAG',format='5E',unit='MAG',array=allsdss)


	t0=matchagc2sdssflag*self.sdsserr_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphoterr_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdsserr_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsserr_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphoterr_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdsserr_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsserr_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphoterr_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdsserr_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsserr_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphoterr_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdsserr_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsserr_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphoterr_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdsserr_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c30=pyfits.Column(name='SDSSMAGERR',format='5E',unit='MAG',array=allsdss)
	#c25=pyfits.Column(name='SDSSG',format='E',unit='MAG',array=t)
	#deredened magnitudes

	t0=matchagc2sdssflag*self.sdssdered_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotdered_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdssdered_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssdered_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotdered_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdssdered_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssdered_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotdered_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdssdered_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssdered_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotdered_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdssdered_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssdered_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotdered_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdssdered_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c31=pyfits.Column(name='SDSSDERED',format='5E',unit='mag',array=allsdss)
	#c28n=pyfits.Column(name='SDSSDEREDG',format='E',unit='mag',array=t)
	#c28o=pyfits.Column(name='SDSSDEREDR',format='E',unit='mag',array=t)
	#c28p=pyfits.Column(name='SDSSDEREDI',format='E',unit='mag',array=t)
	#c28q=pyfits.Column(name='SDSSDEREDZ',format='E',unit='mag',array=t)

	# row and column in each filter
	#
	# needed for reconstructing the PSF
	t0=matchagc2sdssflag*self.sdsscolc_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotcolc_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdsscolc_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsscolc_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotcolc_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdsscolc_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsscolc_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotcolc_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdsscolc_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsscolc_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotcolc_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdsscolc_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsscolc_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotcolc_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdsscolc_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c32=pyfits.Column(name='SDSSCOLC',format='5E',unit='pix',array=allsdss)

	t0=matchagc2sdssflag*self.sdssrowc_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotrowc_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdssrowc_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssrowc_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotrowc_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdssrowc_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssrowc_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotrowc_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdssrowc_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssrowc_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotrowc_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdssrowc_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdssrowc_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotrowc_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdssrowc_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c33=pyfits.Column(name='SDSSROWC',format='5E',unit='pix',array=allsdss)


	#c26=pyfits.Column(name='SDSSR',format='E',unit='MAG',array=t)
	#c27=pyfits.Column(name='SDSSI',format='E',unit='MAG',array=t)
	#c28=pyfits.Column(name='SDSSZ',format='E',unit='MAG',array=t)

	t0=matchagc2sdssflag*self.sdsspetroMag_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroMag_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdsspetroMag_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroMag_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroMag_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdsspetroMag_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroMag_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroMag_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdsspetroMag_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroMag_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroMag_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdsspetroMag_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroMag_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroMag_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdsspetroMag_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c34=pyfits.Column(name='SDSSPETROMAG',format='5E',unit='MAG',array=allsdss)

	t0=matchagc2sdssflag*self.sdsspetroRad_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroRad_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdsspetroRad_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroRad_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroRad_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdsspetroRad_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroRad_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroRad_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdsspetroRad_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroRad_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroRad_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdsspetroRad_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroRad_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroRad_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdsspetroRad_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c35=pyfits.Column(name='SDSSPETRORAD',format='5E',unit='arcsec',array=allsdss)

	t0=matchagc2sdssflag*self.sdsspetroR50_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR50_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdsspetroR50_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR50_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR50_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdsspetroR50_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR50_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR50_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdsspetroR50_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR50_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR50_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdsspetroR50_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR50_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR50_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdsspetroR50_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c36=pyfits.Column(name='SDSSPETROR50',format='5E',unit='arcsec',array=allsdss)

	t0=matchagc2sdssflag*self.sdsspetroR90_u[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR90_u[matchagc2sdssphotindex]
	t1=array((t0.tolist()+self.sdsspetroR90_u[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR90_g[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR90_g[matchagc2sdssphotindex]
	t2=array((t0.tolist()+self.sdsspetroR90_g[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR90_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR90_r[matchagc2sdssphotindex]
	t3=array((t0.tolist()+self.sdsspetroR90_r[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR90_i[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR90_i[matchagc2sdssphotindex]
	t4=array((t0.tolist()+self.sdsspetroR90_i[scoordindex].tolist()),'f')
	t0=matchagc2sdssflag*self.sdsspetroR90_z[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotpetroR90_z[matchagc2sdssphotindex]
	t5=array((t0.tolist()+self.sdsspetroR90_z[scoordindex].tolist()),'f')
	allsdss=column_stack((t1,t2,t3,t4,t5))
	c37=pyfits.Column(name='SDSSPETROR90',format='5E',unit='arcsec',array=allsdss)

	t0=matchagc2sdssflag*self.sdssisoA_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotisoA_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssisoA_r[scoordindex].tolist()),'f')
	c38=pyfits.Column(name='SDSSISOAR',format='E',unit='arcsec',array=t)

	t0=matchagc2sdssflag*self.sdssisoB_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotisoB_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssisoB_r[scoordindex].tolist()),'f')
	c39=pyfits.Column(name='SDSSISOBR',format='E',unit='arcsec',array=t)

	t0=matchagc2sdssflag*self.sdssisoPhi_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotisoPhi_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssisoPhi_r[scoordindex].tolist()),'f')
	c40=pyfits.Column(name='SDSSISOPHIR',format='E',unit='deg N of E',array=t)

	t0=matchagc2sdssflag*self.sdssisoPhiErr_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotisoPhiErr_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssisoPhiErr_r[scoordindex].tolist()),'f')
	c41=pyfits.Column(name='SDSSISOPHIERRR',format='E',unit='deg',array=t)

	#exponential fit parameters
	t0=matchagc2sdssflag*self.sdssexpRad_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotexpRad_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssexpRad_r[scoordindex].tolist()),'f')
	c42=pyfits.Column(name='SDSSEXPRADR',format='E',unit='arcsec',array=t)

	t0=matchagc2sdssflag*self.sdssexpAB_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotexpAB_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssexpAB_r[scoordindex].tolist()),'f')
	c43=pyfits.Column(name='SDSSEXPABR',format='E',unit='',array=t)

	t0=matchagc2sdssflag*self.sdssexpABErr_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotexpABErr_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssexpABErr_r[scoordindex].tolist()),'f')
	c44=pyfits.Column(name='SDSSEXPABRERR',format='E',unit='',array=t)

	t0=matchagc2sdssflag*self.sdssexpPhi_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotexpPhi_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssexpPhi_r[scoordindex].tolist()),'f')
	c45=pyfits.Column(name='SDSSEXPPHIR',format='E',unit='deg N of E',array=t)

	t0=matchagc2sdssflag*self.sdssexpPhiErr_r[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotexpPhiErr_r[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssexpPhiErr_r[scoordindex].tolist()),'f')
	c46=pyfits.Column(name='SDSSEXPPHIERRR',format='E',unit='deg',array=t)


	#spec parameters
	t=array((self.sdssspecz[matchagc2sdssindex].tolist()+self.sdssspecz[scoordindex].tolist()),'f')

	c47=pyfits.Column(name='SDSSSPECZ',format='E',unit='v/c',array=t)
	c48=pyfits.Column(name='SDSSVOPT',format='E',unit='km/s',array=t*3.e5)

	t=array((self.sdssHaEW[matchagc2sdssindex].tolist()+self.sdssHaEW[scoordindex].tolist()),'f')
	c49=pyfits.Column(name='SDSSHAEW',format='E',unit='ANGSTROM',array=t)

	t=array((self.sdssHaEWerr[matchagc2sdssindex].tolist()+self.sdssHaEWerr[scoordindex].tolist()),'f')
	c50=pyfits.Column(name='SDSSHAEWERR',format='E',unit='ANGSTROM',array=t)

	t=array((self.matchsdss2mpaflag[matchagc2sdssindex].tolist()+self.matchsdss2mpaflag[scoordindex].tolist()),'f')
	c51=pyfits.Column(name='MPAFLAG',format='L',unit='',array=t)

	t=array((self.sdssHalpha[matchagc2sdssindex].tolist()+self.sdssHalpha[scoordindex].tolist()),'f')
	c52=pyfits.Column(name='MPAHALPHA',format='E',unit='',array=t)

	t=array((self.sdssHbeta[matchagc2sdssindex].tolist()+self.sdssHbeta[scoordindex].tolist()),'f')
	c53=pyfits.Column(name='MPAHBETA',format='E',unit='',array=t)

	t=array((self.sdssO3[matchagc2sdssindex].tolist()+self.sdssO3[scoordindex].tolist()),'f')
	c54=pyfits.Column(name='MPAOIII',format='E',unit='',array=t)

	t=array((self.sdssN2[matchagc2sdssindex].tolist()+self.sdssN2[scoordindex].tolist()),'f')
	c55=pyfits.Column(name='MPANII',format='E',unit='',array=t)
	t=array((self.sdssHdeltaEW[matchagc2sdssindex].tolist()+self.sdssHdeltaEW[scoordindex].tolist()),'f')
	c56=pyfits.Column(name='MPAHDELTAEW',format='E',unit='',array=t)


	#adding plate, fiber, and blah info from sdss spec catalog
	t=array((self.sdssplate[matchagc2sdssindex].tolist()+self.sdssplate[scoordindex].tolist()),'f')
	c57=pyfits.Column(name='SDSSPLATE',format='I',unit='',array=t)

	t=array((self.sdssfiberID[matchagc2sdssindex].tolist()+self.sdssfiberID[scoordindex].tolist()),'f')
	c58=pyfits.Column(name='SDSSFIBERID',format='I',unit='',array=t)

	t=array((self.sdsstile[matchagc2sdssindex].tolist()+self.sdsstile[scoordindex].tolist()),'f')
	c59=pyfits.Column(name='SDSSTILE',format='I',unit='',array=t)

	#adding run, rerun, camcol to use for grabbing sdss images from DAS
	t0=matchagc2sdssflag*self.sdssrun[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotrun[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssrun[scoordindex].tolist()),'i')

	c60=pyfits.Column(name='SDSSRUN',format='I',unit='',array=t)

	t0=matchagc2sdssflag*self.sdssrerun[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotrerun[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssrerun[scoordindex].tolist()),'f')
	c61=pyfits.Column(name='SDSSRERUN',format='I',unit='',array=t)

	t0=matchagc2sdssflag*self.sdsscamcol[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotcamcol[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdsscamcol[scoordindex].tolist()),'f')
	c62=pyfits.Column(name='SDSSCAMCOL',format='I',unit='',array=t)
	print "Checking array lengths - these should aall be the same"
	print "test c133", len(t)

	t0=matchagc2sdssflag*self.sdssfield[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.sdssphotfield[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.sdssfield[scoordindex].tolist()),'f')
	c63=pyfits.Column(name='SDSSFIELD',format='I',unit='',array=t)
	print "Checking array lengths - these should aall be the same"
	print "test c134", len(t)



#	columnsSE=['NUMBER','X_IMAGE','Y_IMAGE','XMIN_IMAGE','XMAX_IMAGE','YMIN_IMAGE','YMAX_IMAGE','ALPHA_J2000','DELTA_J2000','FLUX_ISO','FLUXERR_ISO','MAG_ISO','MAGERR_ISO','FLUX_AUTO','FLUXERR_AUTO','MAG_AUTO','MAGERR_AUTO','FLUX_PETRO','FLUXERR_PETRO','MAG_PETRO','MAGERR_PETRO','KRON_RADIUS','PETRO_RADIUS','FLUX_RADIUS','ISOAREA_IMAGE','A_WORLD','B_WORLD','THETA_IMAGE','ERRTHETA_IMAGE','THETA_J2000','ERRTHETA_J2000','ELONGATION','ELLIPTICITY','FWHM_WORLD','FLAGS','CLASS_STAR']

        t=tbdata.field('NUMBER')
        c64=pyfits.Column(name='NUMBERSER',format='J',array=t[matchagc2sexsdssindex])

        t=tbdata.field('X_IMAGE')
        c65=pyfits.Column(name='XIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('Y_IMAGE')
        c66=pyfits.Column(name='YIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('XMIN_IMAGE')
        c67=pyfits.Column(name='XMINIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('XMAX_IMAGE')
        c68=pyfits.Column(name='XMAXIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('YMIN_IMAGE')
        c69=pyfits.Column(name='YMINIMAGESER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ALPHA_J2000')
        c70=pyfits.Column(name='RASER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])
        t=tbdata.field('DELTA_J2000')
        c71=pyfits.Column(name='DECSER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLUX_ISO')
        c72=pyfits.Column(name='FLUXISOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])
        t=tbdata.field('FLUXERR_ISO')
        c73=pyfits.Column(name='FLUXERRISOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])

        t=tbdata.field('MAG_ISO')
        c74=pyfits.Column(name='MAGISOSER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('MAGERR_ISO')
        c75=pyfits.Column(name='MAGERRISOSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLUX_AUTO')
        c76=pyfits.Column(name='FLUXAUTOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])
        t=tbdata.field('FLUXERR_AUTO')
        c77=pyfits.Column(name='FLUXERRAUTOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])

        t=tbdata.field('MAG_AUTO')
        c78=pyfits.Column(name='MAGAUTOSER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('MAGERR_AUTO')
        c79=pyfits.Column(name='MAGERRAUTOSER',format='E',array=t[matchagc2sexsdssindex])


        t=tbdata.field('FLUX_PETRO')
        c80=pyfits.Column(name='FLUXPETROSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])
        t=tbdata.field('FLUXERR_PETRO')
        c81=pyfits.Column(name='FLUXERRPETROSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])

        t=tbdata.field('MAG_PETRO')
        c82=pyfits.Column(name='MAGPETROSER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('MAGERR_PETRO')
        c83=pyfits.Column(name='MAGERRPETROSER',format='E',array=t[matchagc2sexsdssindex])


        t=tbdata.field('KRON_RADIUS')
        c84=pyfits.Column(name='KRONRADSER',format='E',unit='PIXELS',array=t[matchagc2sexsdssindex])

        t=tbdata.field('PETRO_RADIUS')
        c85=pyfits.Column(name='PETRORADSER',format='E',unit='PIXELS',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLUX_RADIUS')
        c86=pyfits.Column(name='FLUXRADSER',format='E',unit='PIXELS',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ISOAREA_IMAGE')
        c87=pyfits.Column(name='ISOAREASER',format='E',unit='PIXELSSQ',array=t[matchagc2sexsdssindex])

        t=tbdata.field('A_WORLD')
        c88=pyfits.Column(name='AWORLDSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('B_WORLD')
        c89=pyfits.Column(name='BWORLDSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('THETA_IMAGE')
        c90=pyfits.Column(name='THETASER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ERRTHETA_IMAGE')
        c91=pyfits.Column(name='ERRTHETASER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('THETA_J2000')
        c92=pyfits.Column(name='THETAJ2000SER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ERRTHETA_J2000')
        c93=pyfits.Column(name='ERRTHETAJ2000SER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ELONGATION')
        c94=pyfits.Column(name='ELONGATIONSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ELLIPTICITY')
        c95=pyfits.Column(name='ELLIPTICITYSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FWHM_WORLD')
        c96=pyfits.Column(name='FWHMSER',format='E',unit='ARCSEC',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLAGS')
        c97=pyfits.Column(name='FLAGSSER',format='I',unit='',array=t[matchagc2sexsdssindex])

        t=tbdata.field('CLASS_STAR')
        c98=pyfits.Column(name='CLASSSTARSER',format='E',unit='',array=t[matchagc2sexsdssindex])


        t=tbdata24.field('NUMBER')
        c99=pyfits.Column(name='NUMBERSE24',format='I',array=t[matchagc2sex24index])

        t=tbdata24.field('X_IMAGE')
        c100=pyfits.Column(name='XIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('Y_IMAGE')
        c101=pyfits.Column(name='YIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('XMIN_IMAGE')
        c102=pyfits.Column(name='XMINIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('XMAX_IMAGE')
        c103=pyfits.Column(name='XMAXIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('YMIN_IMAGE')
        c104=pyfits.Column(name='YMINIMAGESE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('ALPHA_J2000')
        c105=pyfits.Column(name='RASE24',format='E',unit='DEG',array=t[matchagc2sex24index])
        t=tbdata24.field('DELTA_J2000')
        c106=pyfits.Column(name='DECSE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('FLUX_ISO')
        c107=pyfits.Column(name='FLUXISOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])
        t=tbdata24.field('FLUXERR_ISO')
        c108=pyfits.Column(name='FLUXERRISOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])

        t=tbdata24.field('MAG_ISO')
        c109=pyfits.Column(name='MAGISOSE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('MAGERR_ISO')
        c110=pyfits.Column(name='MAGERRISOSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('FLUX_AUTO')
        c111=pyfits.Column(name='FLUXAUTOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])
        t=tbdata24.field('FLUXERR_AUTO')
        c112=pyfits.Column(name='FLUXERRAUTOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])

        t=tbdata24.field('MAG_AUTO')
        c113=pyfits.Column(name='MAGAUTOSE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('MAGERR_AUTO')
        c114=pyfits.Column(name='MAGERRAUTOSE24',format='E',array=t[matchagc2sex24index])


        t=tbdata24.field('FLUX_PETRO')
        c115=pyfits.Column(name='FLUXPETROSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])
        t=tbdata24.field('FLUXERR_PETRO')
        c116=pyfits.Column(name='FLUXERRPETROSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])

        t=tbdata24.field('MAG_PETRO')
        c117=pyfits.Column(name='MAGPETROSE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('MAGERR_PETRO')
        c118=pyfits.Column(name='MAGERRPETROSE24',format='E',array=t[matchagc2sex24index])


        t=tbdata24.field('KRON_RADIUS')
        c119=pyfits.Column(name='KRONRADSE24',format='E',unit='PIXELS',array=t[matchagc2sex24index])

        t=tbdata24.field('PETRO_RADIUS')
        c120=pyfits.Column(name='PETRORADSE24',format='E',unit='PIXELS',array=t[matchagc2sex24index])

        t=tbdata24.field('FLUX_RADIUS')
        c121=pyfits.Column(name='FLUXRADSE24',format='E',unit='PIXELS',array=t[matchagc2sex24index])

        t=tbdata24.field('ISOAREA_IMAGE')
        c122=pyfits.Column(name='ISOAREASE24',format='E',unit='PIXELSSQ',array=t[matchagc2sex24index])

        t=tbdata24.field('A_WORLD')
        c123=pyfits.Column(name='AWORLDSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('B_WORLD')
        c124=pyfits.Column(name='BWORLDSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('THETA_IMAGE')
        c125=pyfits.Column(name='THETASE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('ERRTHETA_IMAGE')
        c126=pyfits.Column(name='ERRTHETASE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('THETA_J2000')
        c127=pyfits.Column(name='THETAJ2000SE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('ERRTHETA_J2000')
        c128=pyfits.Column(name='ERRTHETAJ2000SE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('ELONGATION')
        c129=pyfits.Column(name='ELONGATIONSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('ELLIPTICITY')
        c130=pyfits.Column(name='ELLIPTICITYSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('FWHM_WORLD')
        c131=pyfits.Column(name='FWHMSE24',format='E',unit='ARCSEC',array=t[matchagc2sex24index])

        t=tbdata24.field('FLAGS')
        c132=pyfits.Column(name='FLAGSSE24',format='J',unit='',array=t[matchagc2sex24index])

        t=tbdata24.field('CLASS_STAR')
        c133=pyfits.Column(name='CLASSSTARSE24',format='E',unit='',array=t[matchagc2sex24index])



        c134=pyfits.Column(name='MIPSRA',format='E',unit='DEG',array=self.mipsRA[matchfra2mipsindex])

	print "Checking array lengths - these should aall be the same"
	print "c106 test", len(self.mipsRA[matchfra2mipsindex])

        c135=pyfits.Column(name='MIPSDEC',format='E',unit='DEG',array=self.mipsDec[matchfra2mipsindex])
	c136=pyfits.Column(name='MIPSFLUX',format='E',unit='1e-6 Jy',array=self.mipsflux[matchfra2mipsindex])
	c137=pyfits.Column(name='MIPSFLUXERR',format='E',unit='1e-6 Jy',array=self.mipsfluxerr[matchfra2mipsindex])
	c138=pyfits.Column(name='MIPSSNR',format='E',unit='',array=self.mipsSNR[matchfra2mipsindex])
	c139=pyfits.Column(name='MIPSDEBLEND',format='A',unit='',array=self.mipsdeblend[matchfra2mipsindex])
	c140=pyfits.Column(name='MIPSFLUXAP1',format='E',unit='1e-6 Jy',array=self.mipsap1[matchfra2mipsindex])
	c141=pyfits.Column(name='MIPSFLUXAP1ERR',format='E',unit='1e-6 Jy',array=self.mipsap1err[matchfra2mipsindex])
	c142=pyfits.Column(name='MIPSFLUXAP2',format='E',unit='1e-6 Jy',array=self.mipsap2[matchfra2mipsindex])
	c143=pyfits.Column(name='MIPSFLUXAP2ERR',format='E',unit='1e-6 Jy',array=self.mipsap2err[matchfra2mipsindex])
	c144=pyfits.Column(name='MIPSFLUXAP3',format='E',unit='1e-6 Jy',array=self.mipsap3[matchfra2mipsindex])
	c145=pyfits.Column(name='MIPSFLUXAP3ERR',format='E',unit='1e-6 Jy',array=self.mipsap3err[matchfra2mipsindex])

	#add galaxy zoo data
	t0=matchagc2sdssflag*self.zooSpecFlag[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecFlag[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecFlag[scoordindex].tolist()),'f')
	c146=pyfits.Column(name='GALZOOFLAG',format='L',unit='',array=t)
	print "Checking array lengths - these should aall be the same"
	print "c118 test", len(t)

	#Nvote
	t0=matchagc2sdssflag*self.zooSpecNvote[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecNvote[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecNvote[scoordindex].tolist()),'f')
	c147=pyfits.Column(name='GALZOONVOTE',format='I',unit='',array=t)
	#Pel
	t0=matchagc2sdssflag*self.zooSpecPel[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecPel[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecPel[scoordindex].tolist()),'f')
	c148=pyfits.Column(name='GALZOOPEL',format='D',unit='',array=t)
	#Pcw
	t0=matchagc2sdssflag*self.zooSpecPcw[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecPcw[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecPcw[scoordindex].tolist()),'f')
	c149=pyfits.Column(name='GALZOOPCW',format='D',unit='',array=t)
	#Pacs
	t0=matchagc2sdssflag*self.zooSpecPacw[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecPacw[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecPacw[scoordindex].tolist()),'f')
	c150=pyfits.Column(name='GALZOOPACW',format='D',unit='',array=t)
	#Pedge
	t0=matchagc2sdssflag*self.zooSpecPedge[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecPedge[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecPedge[scoordindex].tolist()),'f')
	c151=pyfits.Column(name='GALZOOPEDGE',format='D',unit='',array=t)
	#Pdk
	t0=matchagc2sdssflag*self.zooSpecPdk[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecPdk[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecPdk[scoordindex].tolist()),'f')
	c152=pyfits.Column(name='GALZOOPDK',format='D',unit='',array=t)
	#Pmg
	t0=matchagc2sdssflag*self.zooSpecPmg[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecPmg[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecPmg[scoordindex].tolist()),'f')
	c153=pyfits.Column(name='GALZOOPMG',format='D',unit='',array=t)
	#Pmg
	t0=matchagc2sdssflag*self.zooSpecPcs[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*self.zooNoSpecPcs[matchagc2sdssphotindex]
	t=array((t0.tolist()+self.zooSpecPcs[scoordindex].tolist()),'f')
	c154=pyfits.Column(name='GALZOOPCS',format='D',unit='',array=t)
	#PelDebiased (for sdss spec galaxies only)
	t0=matchagc2sdssflag*self.zooSpecPelDebiased[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*zeros(len(self.zooNoSpecPcs[matchagc2sdssphotindex]))
	t=array((t0.tolist()+self.zooSpecPelDebiased[scoordindex].tolist()),'f')
	c155=pyfits.Column(name='GALZOOPELDEBIASED',format='D',unit='',array=t)
	#PcsDebiased (for sdss spec galaxies only)
	t0=matchagc2sdssflag*self.zooSpecPcsDebiased[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*zeros(len(self.zooNoSpecPcs[matchagc2sdssphotindex]))
	t=array((t0.tolist()+self.zooSpecPcsDebiased[scoordindex].tolist()),'f')
	c156=pyfits.Column(name='GALZOOPCSDEBIASED',format='D',unit='',array=t)
	#Spiral (for sdss spec galaxies only)
	t0=matchagc2sdssflag*self.zooSpecSpiral[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*zeros(len(self.zooNoSpecPcs[matchagc2sdssphotindex]))
	t=array((t0.tolist()+self.zooSpecSpiral[scoordindex].tolist()),'f')
	c157=pyfits.Column(name='GALZOOSPIRAL',format='L',unit='',array=t)
	#Elliptical (for sdss spec galaxies only)
	t0=matchagc2sdssflag*self.zooSpecElliptical[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*zeros(len(self.zooNoSpecPcs[matchagc2sdssphotindex]))
	t=array((t0.tolist()+self.zooSpecElliptical[scoordindex].tolist()),'f')
	c158=pyfits.Column(name='GALZOOELLIPTICAL',format='L',unit='',array=t)
	#Uncertain (for sdss spec galaxies only)
	t0=matchagc2sdssflag*self.zooSpecUncertain[matchagc2sdssindex]+((~matchagc2sdssflag)&(matchagc2sdssphotflag))*zeros(len(self.zooNoSpecPcs[matchagc2sdssphotindex]))
	t=array((t0.tolist()+self.zooSpecUncertain[scoordindex].tolist()),'f')
	c159=pyfits.Column(name='GALZOOUNCERTAIN',format='L',unit='',array=t)


	print "Checking array lengths - these should aall be the same"
	print "c130 test", len(t)


	mastertb=pyfits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,c71,c72,c73,c74,c75,c76,c77,c78,c79,c80,c81,c82,c83,c84,c85,c86,c87,c88,c89,c90,c91,c92,c93,c94,c95,c96,c97,c98,c99,c100,c101,c102,c103,c104,c105,c106,c107,c108,c109,c110,c111,c112,c113,c114,c115,c116,c117,c118,c119,c120,c121,c122,c123,c124,c125,c126,c127,c128,c129,c130,c131,c132,c133,c134,c135,c136,c137,c138,c139,c140,c141,c142,c143,c144,c145,c146,c147,c148,c149,c150,c151,c152,c153,c154,c155,c156,c157,c158,c159])
	s='/home/rfinn/research/LocalClusters/MasterTables/'+self.prefix+'mastertable.fits'
	mastertb.writeto(s,clobber='yes')
#	unitsSE=['SE NUMBER','PIXELS','PIXELS','PIXELS','PIXELS','PIXELS','PIXELS','DEG','DEG','uJY','uJY','MAG','MAG','uJY','uJY','MAG','MAG','uJY','uJY','MAG','MAG','PIXELS','PIXELS','PIXELS','PIX^2','DEG','DEG','DEG','DEG','DEG N OF E','DEG','ELONGATION','ELLIPTICITY','FWHM DEG','FLAGS','CLASS_STAR']
#	typeSE=['I','E','E','E','E','E','E','D','D','D','D','E','E','D','D','E','E','E','E','D','D','E','E','E','E','D','D','E','E','E','E','E','E','E','I','E']
#
#	columnsSE24=[]
#	for i in range(len(columnsSE)):
#		columnsSE24.append(columnsSE[i]+'24')
#	columns=columnsFlag+columnsAGC+columnsSDSS+columnsSE+columnsSE24
#	types=typesFlag+typesAGC+typeSDSS+typeSE+typeSE
#	units=unitsFlag+unitsAGC+unitsSDSS+unitsSE+unitsSE
#	#create columns for fits table
#	allcolumns=[]
#	for i in range(len(columns)):
#		allcolumns.append(pyfits.Column(name=columns[i],format=types[i],unit=units[i]))
	#write columns using pyfits.new_table
	#write file using tb.writeto('filename.fits')

    def plotmags(self):
	#plot
	figure()
	plot(self.pipeliner,self.petror-self.pipeliner,'bo',label='MAG_PETRO')
	plot(self.pipeliner,self.magautor-self.pipeliner,'go',label='MAG_AUTO')
	xl=arange(min(self.pipeliner),max(self.pipeliner),.1)
	#plot(xl,zeros(len(xl)),'r-')
	legend()
	xlabel('SDSS Pipeline Mag')
	ylabel('SExtractor Mag - SDSS Pipeline Mag')
	show()


    def readmembcat(self):
	s='/home/rfinn/research/LocalClusters/MemberCatalogs/'+self.prefix+'AGCSDSS.dat'
	outfile=open(s,'r')
	ngal=0
	for line in outfile:
		if line.find('#') > -1:
			continue
		ngal += 1
	outfile.close()
	ara=zeros(ngal,'f')
	adec=zeros(ngal,'f')
	sra=zeros(ngal,'f')
	sdec=zeros(ngal,'f')
	aflag=zeros(ngal,'i')
	sflag=zeros(ngal,'i')
	fra=zeros(ngal,'f')
	fdec=zeros(ngal,'f')
	agcname=[]
	outfile=open(s,'r')
	i=0
	for line in outfile:
		if line.find('#') > -1:
			continue
		t=line.split()
		#print line
		#print t
		#for i in range(len(t)-2):
		#	t[i]=float(t[i])
		#for i in range(len(t)-2,len(t)):
		#	t[i]=int(t[i])
		ara[i]=float(t[0])
		adec[i]=float(t[1])
		sra[i]=float(t[2])
		sdec[i]=float(t[3])
		aflag[i]=int(t[4])
		sflag[i]=int(t[5])
		agcname.append(t[6])
		fra[i]=float(t[7])
		fdec[i]=float(t[8])

		i += 1
	outfile.close()
	#print ara
	#print adec
	#print aflag
	#print fra
	fra=array(fra,'f')
	fdec=array(fdec,'f')
	f='/home/rfinn/research/LocalClusters/RegionsFiles/'+self.prefix+'.SdssAgc.reg'
	outfile3=open(f,'w')
	s='global color=cyan font="helvetica 10 normal" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source \n'
	outfile3.write(s)
	s='fk5 \n'
	outfile3.write(s)
	for i in range(len(fra)):
	    s='circle(%12.8f, %12.8f, 10") \n'%(fra[i],fdec[i])
	    outfile3.write(s)
	outfile3.close()

	self.agcname=agcname
	self.fra=fra
	self.fdec=fdec
	figure()
	clf()
	msize=4
	fsize=8
	subplot(2,2,1)
	plot(ara[aflag>.1],adec[aflag>.1],'bo',markersize=msize)
	xmin,xmax=xlim()
	ymin,ymax=ylim()
	s=self.prefix+' - AGC'
	title(s,fontsize=fsize)
	axis([xmin,xmax,ymin,ymax])
	subplot(2,2,2)
	plot(sra[sflag>.1],sdec[sflag>.1],'ro',markersize=msize)
	title('SDSS Spec',fontsize=fsize)
	axis([xmin,xmax,ymin,ymax])
	subplot(2,2,3)
	plot(ara[sflag<.1],adec[sflag<.1],'bo',markersize=msize)
	title('AGC but not in SDSS Spec',fontsize=fsize)
	axis([xmin,xmax,ymin,ymax])
	subplot(2,2,4)
	title('SDSS Spec but not in AGC',fontsize=fsize)
	plot(sra[aflag<.1],sdec[aflag<.1],'ro',markersize=msize)
	axis([xmin,xmax,ymin,ymax])
	fname='/home/rfinn/research/LocalClusters/MemberCatalogs/'+self.prefix+'check.eps'
	savefig(fname)
	s=self.cutoutpath+self.prefix+'.radec'
	self.incoords=s
	out1=open(s,'w')
        #print self.sdssRA
	for i in range(len(self.fra)):
	    s='%f %f %s\n'%(self.fra[i],self.fdec[i],agcname[i])
	    out1.write(s)
	out1.close()

    def transcoords(self):
	outcoords=str(self.prefix)+'.xy'
        s='rm '+outcoords
        os.system(s)
	iraf.imcoords.wcsctran(image=self.image,input=self.incoords,output=outcoords,inwcs='world',outwcs='logical',verbose='no')
	self.outcoords=outcoords

    def transcoords24(self):
        print 'transforming 24um coords'
	outcoords=self.cutoutpath+str(self.prefix)+'.xy24'
        s='rm '+outcoords
        os.system(s)
	iraf.imcoords.wcsctran(image=self.image24,input=self.incoords,output=outcoords,inwcs='world',outwcs='logical',verbose='no')
	self.outcoords24=outcoords

    def transcoords24rot(self):
        print 'transforming rotated 24um coords'
	outcoords=self.cutoutpath+str(self.prefix)+'.rot.xy24'
        s='rm '+outcoords
        os.system(s)
	iraf.imcoords.wcsctran(image=self.rotatedimage24,input=self.incoords,output=outcoords,inwcs='world',outwcs='logical',verbose='no')
	self.outcoordsrot24=outcoords

    def makesdsscutouts(self,band):
        #sdssim1=self.sdssrim1
        #sdssim2=self.sdssrim2

	if band == 1:
		sdssim1=self.sdssgim1
		sdssim2=self.sdssgim2
	else:
		sdssim1=self.sdssrim1skysub
		sdssim2=self.sdssrim2skysub

        #transform coords using image 1
	outcoords=self.cutoutpath+str(self.prefix)+'.xy1'
        s='rm '+outcoords
        os.system(s)
	iraf.imcoords.wcsctran(image=sdssim1,input=self.incoords,output=outcoords,inwcs='world',outwcs='logical',verbose='no')
	self.outcoords1=outcoords
        xim1=[]
        yim1=[]
	infile=open(outcoords,'r')
	for line in infile:
	    if line.find('#') > -1:
		continue
	    if len(line)<2:
		continue
	    t=line.split()
	    xim1.append(float(t[0]))
	    yim1.append(float(t[1]))
        xim1=array(xim1,'f')
        yim1=array(yim1,'f')
        #get center of image1
	xmin=1.
	ymin=1.
	iraf.imgets(image=sdssim1,param='naxis1')#get RA of image
	xmax1=float(iraf.imgets.value)
        xcenter1=xmax1/2

	iraf.imgets(image=sdssim1,param='naxis2')#get RA of image
	ymax1=float(iraf.imgets.value)
        ycenter1=ymax1/2

        #calculate distance of sources from center of image 1
        distanceim1=sqrt((xim1-xcenter1)**2+(yim1-ycenter1)**2)

        #transform coords using image 2
	outcoords=self.cutoutpath+str(self.prefix)+'.xy2'
	#outcoords=str(self.prefix)+'.xy2'
        s='rm '+outcoords
        os.system(s)
	iraf.imcoords.wcsctran(image=sdssim2,input=self.incoords,output=outcoords,inwcs='world',outwcs='logical',verbose='no')
	self.outcoords2=outcoords

        xim2=[]
        yim2=[]
	infile=open(outcoords,'r')
	for line in infile:
	    if line.find('#') > -1:
		continue
	    if len(line)<2:
		continue
	    t=line.split()
	    xim2.append(float(t[0]))
	    yim2.append(float(t[1]))
        xim2=array(xim2,'f')
        yim2=array(yim2,'f')

        #get center of image2
	iraf.imgets(image=sdssim1,param='naxis1')#get RA of image
	xmax2=float(iraf.imgets.value)
        xcenter2=xmax2/2

	iraf.imgets(image=sdssim1,param='naxis2')#get RA of image
	ymax2=float(iraf.imgets.value)
        ycenter2=ymax2/2

        #calculate distance of sources from center of image 2
        distanceim2=sqrt((xim2-xcenter2)**2+(yim2-ycenter2)**2)

        #loop through ids and get cutout from image in which galaxy is furthest from edge, or closest to center

        iraf.imgets(image=sdssim1,param='CDELT2')#get x plate scale. assumes this is the same for x & y, same for both images
        xplate=abs(float(iraf.imgets.value))#deg/pixel
        dpix=delta/3600./xplate/2.

	#add caveat to make bigger cutouts for galaxies that need them, dpix=2 x dpix

        cutouts=[]
        j=0
        for i in range(len(xim1)):
            if self.cutout24flag[i] < 1:#only make cutout for objects that lie on 24um image
                continue
            if distanceim1[i] < distanceim2[i]:#use image 1
                sdssimage=sdssim1
                x=xim1[i]
                y=yim1[i]
            else:
                sdssimage=sdssim2
                x=xim2[i]
                y=yim2[i]
	    xmin=int(round(x-dpix))
	    xmax=int(round(x+dpix))
	    ymin=int(round(y-dpix))
	    ymax=int(round(y+dpix))
	    imsec=sdssimage+'[%i:%i,%i:%i]'%(xmin,xmax,ymin,ymax)
	    #print s
	    if band == 1:
		    outim=self.cutoutpath+self.prefix+'-'+self.agcname[i]+'-cutout-sdss-g.fits'
	    else:
		    outim=self.cutoutpath+self.prefix+'-'+self.agcname[i]+'-cutout-sdss.fits'
	    #print outim
            iraf.imcopy(imsec,outim)
	    cutouts.append(outim)
            j += 1
	infile.close()
	self.sdsscutouts=cutouts

    def makesdsscutoutsv2(self,band):#for A1367 and Hercules
        if band == 1:
		sdssim1=self.sdssgim1
		sdssim1=self.sdssgim1
	else:
		sdssim1=self.sdssrim1
		sdssim1=self.sdssrim1skysub
        #transform coords using image 1
	outcoords=self.cutoutpath+str(self.prefix)+'.xy1'
        s='rm '+outcoords
        os.system(s)
	iraf.imcoords.wcsctran(image=sdssim1,input=self.incoords,output=outcoords,inwcs='world',outwcs='logical',verbose='no')
	self.outcoords1=outcoords

        xim1=[]
        yim1=[]
	infile=open(outcoords,'r')
	for line in infile:
	    if line.find('#') > -1:
		continue
	    if len(line)<2:
		continue
	    t=line.split()
	    xim1.append(float(t[0]))
	    yim1.append(float(t[1]))
        xim1=array(xim1,'f')
        yim1=array(yim1,'f')
        #get center of image1
	xmin=1.
	ymin=1.
	iraf.imgets(image=sdssim1,param='naxis1')#get RA of image
	xmax1=float(iraf.imgets.value)
        xcenter1=xmax1/2

	iraf.imgets(image=sdssim1,param='naxis2')#get RA of image
	ymax1=float(iraf.imgets.value)
        ycenter1=ymax1/2

        #calculate distance of sources from center of image 1
        distanceim1=sqrt((xim1-xcenter1)**2+(yim1-ycenter1)**2)

        #loop through ids and get cutout from image in which galaxy is furthest from edge, or closest to center
        iraf.imgets(image=sdssim1,param='CDELT2')#get x plate scale. assumes this is the same for x & y, same for both images
        xplate=abs(float(iraf.imgets.value))#deg/pixel
        dpix=delta/3600./xplate/2.

        cutouts=[]
        j=0
        for i in range(len(xim1)):
            if self.cutout24flag[i] < 1:#only make cutout for objects that lie on 24um image
                continue
	    sdssimage=sdssim1
	    x=xim1[i]
	    y=yim1[i]
	    xmin=int(round(x-dpix))
	    xmax=int(round(x+dpix))
	    ymin=int(round(y-dpix))
	    ymax=int(round(y+dpix))
	    imsec=sdssimage+'[%i:%i,%i:%i]'%(xmin,xmax,ymin,ymax)
	    #print s
		
	    if band == 1:
		    outim=self.cutoutpath+self.prefix+'-'+self.agcname[i]+'-cutout-sdss-g.fits'
	    else:
		    outim=self.cutoutpath+self.prefix+'-'+self.agcname[i]+'-cutout-sdss.fits'
	    print outim
            iraf.imcopy(imsec,outim)
	    cutouts.append(outim)
            j += 1
	infile.close()
	self.sdsscutouts=cutouts

    def makecutouts24(self,createimages=0):
	iraf.imgets(image=self.image24,param='CD1_1')#get x plate scale on rotated image
	#print iraf.imgets.value
	xplate=abs(float(iraf.imgets.value))#deg/pixel
	dpix=deltaCutout/3600./xplate/2.
	iraf.imgets(image=self.image24,param='naxis1')#get x value corresponding to RA 
	xpixmax=(int(iraf.imgets.value))#deg/pixel
	iraf.imgets(image=self.image24,param='naxis2')#get x value corresponding to RA 
	ypixmax=(int(iraf.imgets.value))#deg/pixel
	infile=open(self.outcoords24,'r')

	cutouts24=[]
	cutout24flag=ones(len(self.fra),'i')
	print "Checking error on cutout flag 24"
	print "length of cutout24flag = ",len(cutout24flag)
	#print outcoords
        i=-1
        minsize=40

	for line in infile:
	    #print images24[i],line,outcoords
	    if line.find('#') > -1:
		continue
	    if len(line)<2:
		continue
            i=i+1
            #print i,line,self.ediscsID[i]
	    x,y,id=line.split()
	    x=float(x)
	    y=float(y)
	    #deltaCutout=100.
	    xmin=int(round(x-dpix))
	    xmax=int(round(x+dpix))
	    ymin=int(round(y-dpix))
	    ymax=int(round(y+dpix))
	    if xmin < 1:
		xmin=1
	    if ymin < 1:
		ymin=1
	    if xmax > xpixmax:
		xmax=xpixmax
	    if ymax > ypixmax:
		ymax=ypixmax
	    if xmin > xpixmax:
		cutout24flag[i]=0
		print i,"pixel value out of range 1"
		continue
	    if ymin > ypixmax:
		cutout24flag[i]=0
		print i,"pixel value out of range 2"
		continue
	    if xmax < minsize:
		cutout24flag[i]=0
		print i," pixel value out of range 3"
		continue
	    if ymax < minsize:
		cutout24flag[i]=0
		print i," pixel value out of range 4"
		continue
	    if (abs(xmax-xmin)<minsize):
		cutout24flag[i]=0
		print i," image less than 5 pixels wide"
		continue
	    if (abs(ymax-ymin)<minsize):
		cutout24flag[i]=0
		print i," image height less than 5 pixels"
		continue

	    s=self.image24+'[%i:%i,%i:%i]'%(xmin,xmax,ymin,ymax)
	    print s
	    #print ra[i],dec[i]
	    id=self.prefix+'-'+self.agcname[i]+'-'
	    outim=self.cutoutpath+id+'cutout-24.fits'


	    if createimages:
		    try:
			    iraf.imcopy(s,outim)
		    except:
			    cutout24flag[i]=0
			    continue

	    cutouts24.append(outim)
	infile.close()
	self.cutouts24=cutouts24
	self.cutout24flag=cutout24flag
        f='/home/rfinn/research/LocalClusters/MemberCatalogs/'+self.prefix+'cutoutfiles.dat'
        output99=open(f,'w')

	f='/home/rfinn/research/LocalClusters/RegionsFiles/'+self.prefix+'.On24.reg'
	outfile3=open(f,'w')
	s='global color=red font="helvetica 10 normal" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source \n'
	outfile3.write(s)
	s='fk5 \n'
	outfile3.write(s)

	for i in range(len(cutout24flag)):
            s='%f %f %i \n'%(self.fra[i],self.fdec[i],self.cutout24flag[i])
            output99.write(s)
	    if cutout24flag[i] > 0.1:
		    s='circle(%12.8f, %12.8f, 10") \n'%(self.fra[i],self.fdec[i])
		    outfile3.write(s)
        output99.close()
	outfile3.close()

    def appendcutout24flagtomastertable(self):

	    s='/home/rfinn/research/LocalClusters/MasterTables/'+self.prefix+'mastertable.fits'
	    mastertab=pyfits.open(s)
	    tbhdu=mastertab[1]

	    cdefs=tbhdu.columns
	    tbdata=mastertab[1].data
	    mastertab.close()

	    newcol=pyfits.Column(name='On24ImageFlag',format='L',array=self.cutout24flag)
	    print 'checking length of cutout24flag = ',len(self.cutout24flag)
	    cdefs2=cdefs.add_col(newcol)
	    newmaster=pyfits.new_table(cdefs2)

	    testoutput='/home/rfinn/research/LocalClusters/MasterTables/'+self.prefix+'mastertable.fits'
	    newmaster.writeto(testoutput,clobber='yes')


    def getnearestall(self):
	    angdist=my.DA(zfield,h100)#kpc/arcsec
	    #calculate nearest neighbor distances from all z=0.8 spec galaxies, including non-Ha galaxies
	    self.sig5=zeros(len(self.RAall),'f')
	    self.sig10=zeros(len(self.RAall),'f')
	    self.nearest=zeros(len(self.RAall),'f')#distance to nearest neighbor
	    self.dmagnearest=zeros(len(self.RAall),'f')#mag diff b/w spec obj and nearest


	    for i in range(len(self.RAall)):
		    dspec=sqrt((self.RAall[i]-self.RAall)**2+(self.Decall[i]-self.Decall)**2)#sorted array of distances in degrees
		    
		    dspecsort=take(dspec,argsort(dspec))
		    
		    self.sig5[i]=5./(pi)/(sum(dspecsort[1:6])*3600.*angdist/1000.)**2
		    if (len(dspecsort) > 10):
			    self.sig10[i]=10./(pi)/(sum(dspecsort[1:11])*3600.*angdist/1000.)**2
		    else:
			    self.sig10[i]=0.

		    self.nearest[i]=dspecsort[1]*3600.*angdist#first element of array is dist from galaxy to itself


    def readmorphcat(self):
	    infile='/home/rfinn/research/LocalClusters/Morphologies2010/Final.'+self.prefix+'.txt'
	    infile1=open(infile,'r')
	    name=[]
	    morph=[]
	    disturb=[]
	    for line in infile1:
		    t=line.split()
		    n=t[0]
		    myn=n.split('-')
		    if len(myn) > 4:
			    s='-'+myn[2]
			    name.append(s)
			    nname=s
		    else:
			    name.append(myn[1])
			    nname=myn[1]
		    try:
			    morph.append(int(t[1]))
		    except:
			    print self.prefix,': trouble in paradise on morph, ',nname,t[1]
			    morph.append(int('-99'))
		    try:
			    disturb.append(int(t[2]))
		    except:
			    print 'trouble in paradise'
			    disturb.append(int('-99'))
	    return name,morph,disturb

    def appendmorecolstomastertable(self):
	    s='/home/rfinn/research/LocalClusters/MasterTables/'+self.prefix+'mastertable.fits'
	    mastertab=pyfits.open(s)
	    tbhdu=mastertab[1]
	    cdefs=tbhdu.columns
	    tbdata=mastertab[1].data
	    mastertab.close()

	    ###  define super velocity column: SDSSVOPT or AGC VOPT or HI V
	    sdssflag=tbdata.field('SDSSFLAG')
	    sdssvopt=tbdata.field('SDSSVOPT')
	    agcvoptflag=tbdata.field('AGCVOPTFLAG')
	    vopt=tbdata.field('VOPT')
	    HIflag=tbdata.field('HIFLAG')
	    v21=tbdata.field('V21')
	    
	    self.supervopt=sdssflag*sdssvopt+(~sdssflag & agcvoptflag)*vopt+(~sdssflag & ~agcvoptflag & HIflag)*v21


	    ### define super RA and Dec columns

	    sdssra=tbdata.field('SDSSRA')
	    sdssdec=tbdata.field('SDSSDEC')
	    agcra=tbdata.field('AGCRA')
	    agcdec=tbdata.field('AGCDEC')
	    agcflag=tbdata.field('AGCFLAG')
	    self.superra=sdssflag*sdssra+(~sdssflag & agcflag)*agcra
	    self.superdec=sdssflag*sdssdec+(~sdssflag & agcflag)*agcdec


	    ### define membership flag, dv < 3*sigma, dr<R200
	    
	    dr=sqrt((self.superra-self.cra)**2+(self.superdec-self.cdec)**2)
	    dv=self.supervopt-self.biweightvel
	    
	    self.membflag=(dr/self.r200deg < 1) & (abs(dv) < 3.*self.biweightscale)


	    ### stellar mass
	    self.distMpc=self.supervopt/H0#distance in Mpc
	    self.distMpc=array(self.distMpc,'d')
	    sdssmags=tbdata.field('SDSSMAG')
	    sdssu=sdssmags[:,0]
	    sdssr=sdssmags[:,1]
	    sdssg=sdssmags[:,2]
	    sdssi=sdssmags[:,3]
	    sdssz=sdssmags[:,4]
	    
	    self.sdssMr=5-5.*log10(self.distMpc*1.e6)+sdssr
	    self.sdssLr=10.**(-1*(self.sdssMr-SolarMag['r'])/2.5)
	    a,b=bellgr['r']
	    self.stellarmassr=10.**(-1.*a+b*(sdssg-sdssr))*self.sdssLr

	    self.sdssMu=5-5.*log10(self.distMpc*1.e6)+sdssu
	    self.sdssLu=10.**(-1*(self.sdssMu-SolarMag['u'])/2.5)
	    
	    self.sdssMg=5-5.*log10(self.distMpc*1.e6)+sdssg
	    self.sdssLg=10.**(-1*(self.sdssMg-SolarMag['g'])/2.5)

	    self.sdssMi=5-5.*log10(self.distMpc*1.e6)+sdssi
	    self.sdssLi=10.**(-1*(self.sdssMi-SolarMag['i'])/2.5)

	    self.sdssMz=5-5.*log10(self.distMpc*1.e6)+sdssz
	    self.sdssLz=10.**(-1*(self.sdssMz-SolarMag['z'])/2.5)

	    # calculate the abs mag and luminosity if galaxy were at cluster redshift

	    self.sdssMr_cl=5-5.*log10(self.cdMpc*1.e6)+sdssr
	    self.sdssLr_cl=10.**(-1*(self.sdssMr-SolarMag['r'])/2.5)
	    a,b=bellgr['r']

	    self.sdssMu_cl=5-5.*log10(self.cdMpc*1.e6)+sdssu
	    self.sdssLu_cl=10.**(-1*(self.sdssMu-SolarMag['u'])/2.5)
	    
	    self.sdssMg_cl=5-5.*log10(self.cdMpc*1.e6)+sdssg
	    self.sdssLg_cl=10.**(-1*(self.sdssMg-SolarMag['g'])/2.5)

	    self.sdssMi_cl=5-5.*log10(self.cdMpc*1.e6)+sdssi
	    self.sdssLi_cl=10.**(-1*(self.sdssMi-SolarMag['i'])/2.5)

	    self.sdssMz_cl=5-5.*log10(self.cdMpc*1.e6)+sdssz
	    self.sdssLz_cl=10.**(-1*(self.sdssMz-SolarMag['z'])/2.5)
	    self.stellarmassr_cl=10.**(-1.*a+b*(sdssg-sdssr))*self.sdssLr_cl

	    #24um Luminosity


	    # morphologies from Summer 2010
	    print 'matching to summer 2010 morphologies'
	    #create dictionary with agc names
	    self.agcnumber=tbdata.field('AGCNUMBER')
	    self.agcdict=dict((a,b) for a,b in zip(self.agcnumber,arange(len(self.agcnumber))))
	    #read in morphology data
	    (names,morph,disturb)=self.readmorphcat()
	    morph=array(morph,'i')
	    disturb=array(disturb,'i')
	    self.morphflag=zeros(len(self.agcnumber),'bool')
	    self.morph=zeros(len(self.agcnumber),'i')
	    self.disturb=zeros(len(self.agcnumber),'i')
	    for i in range(len(names)):
		    try:
			    t=self.agcdict[int(names[i])]
			    self.morph[t]=morph[i]
			    self.disturb[t]=disturb[i]
			    self.morphflag[t]=1
		    except KeyError:
			    print 'no match for ',names[i],' in new catalog'
	    
	    #nearest neighbor density
	    print 'getting local density'
	    self.localdens=self.localdensity(self.superra,self.superdec,self.supervopt,sdssra,sdssdec,sdssvopt,3,6)
	    #match 2 AGN line flux info

	    ha=tbdata.field('MPAHALPHA')
	    hb=tbdata.field('MPAHBETA')
	    o3=tbdata.field('MPAOIII')
	    n2=tbdata.field('MPANII')
	    x=log10(n2/ha)
	    y=log10(o3/hb)

            #equations for lines from Alissa's work
	    self.AGN1=((y >(.61/(x-.05)+1.3)) | (x > 0))#Kauffman 2003?
	    self.AGN2=(y>(.61/(x-.47)+1.19))#Kewely
	    self.AGN3=(y > ((-30.787+(1.1358*x)+((.27297)*(x)**2))*tanh(5.7409*x))-31.093) #Stasinska 2006	    
	    #HI mass
	    #HI deficiency

	    newcol1=pyfits.Column(name='SUPERVOPT',format='E',unit='km/s',array=self.supervopt)
	    newcol2=pyfits.Column(name='SUPERRA',format='E',unit='DEG',array=self.superra)
	    newcol3=pyfits.Column(name='SUPERDEC',format='E',unit='DEG',array=self.superdec)
	    newcol4=pyfits.Column(name='STELLARMASS',format='E',unit='Solar Masses',array=self.stellarmassr)
	    newcol4a=pyfits.Column(name='STELLARMASS_CL',format='E',unit='Solar Masses',array=self.stellarmassr_cl)
	    t=column_stack((self.sdssMu,self.sdssMg,self.sdssMr,self.sdssMi,self.sdssMz))
	    newcol5=pyfits.Column(name='SDSSABSMAG',format='5E',unit='MAG',array=t)
	    t=column_stack((self.sdssLu,self.sdssLg,self.sdssLr,self.sdssLi,self.sdssLz))
	    newcol6=pyfits.Column(name='SDSSLUM',format='5E',unit='Solar Luminosity',array=t)

	    t=column_stack((self.sdssMu_cl,self.sdssMg_cl,self.sdssMr_cl,self.sdssMi_cl,self.sdssMz_cl))
	    newcol7=pyfits.Column(name='SDSSABSMAG_CL',format='5E',unit='MAG',array=t)
	    t=column_stack((self.sdssLu_cl,self.sdssLg_cl,self.sdssLr_cl,self.sdssLi_cl,self.sdssLz_cl))
	    newcol8=pyfits.Column(name='SDSSLUM_CL',format='5E',unit='Solar Luminosity',array=t)

	    #newcol5=pyfits.Column(name='SDSSMU',format='E',unit='MAG',array=self.sdssMu)
	    #newcol6=pyfits.Column(name='SDSSLU',format='E',unit='Solar Luminosity',array=self.sdssLu)
	    #newcol7=pyfits.Column(name='SDSSMG',format='E',unit='MAG',array=self.sdssMg)
	    #newcol8=pyfits.Column(name='SDSSLG',format='E',unit='Solar Luminosity',array=self.sdssLg)
	    #newcol9=pyfits.Column(name='SDSSMR',format='E',unit='MAG',array=self.sdssMr)
	    #newcol10=pyfits.Column(name='SDSSLR',format='E',unit='Solar Luminosity',array=self.sdssLr)
	    #newcol11=pyfits.Column(name='SDSSMI',format='E',unit='MAG',array=self.sdssMi)
	    #newcol12=pyfits.Column(name='SDSSLI',format='E',unit='Solar Luminosity',array=self.sdssLi)
	    #newcol13=pyfits.Column(name='SDSSMZ',format='E',unit='MAG',array=self.sdssMz)
	    #newcol14=pyfits.Column(name='SDSSLZ',format='E',unit='Solar Luminosity',array=self.sdssLz)
	    newcol9=pyfits.Column(name='MEMBFLAG',format='L',unit='',array=self.membflag)
	    newcol10=pyfits.Column(name='MORPHFLAG',format='L',unit='',array=self.morphflag)
	    newcol11=pyfits.Column(name='MORPH',format='I',unit='',array=self.morph)
	    newcol12=pyfits.Column(name='DISTURB',format='I',unit='',array=self.disturb)
	    newcol13=pyfits.Column(name='LOCALDENS',format='D',unit='',array=self.localdens)
	    newcol14=pyfits.Column(name='AGNKAUFF',format='L',unit='',array=self.AGN1)
	    newcol15=pyfits.Column(name='AGNKEWLEY',format='L',unit='',array=self.AGN2)
	    newcol16=pyfits.Column(name='AGNSTASIN',format='L',unit='',array=self.AGN3)
	    

	    #get info on whether ellipse finished in r-band and 24
	    ellipseflag24=zeros(len(self.AGN1),'i')
	    ellipse24rejectcode=zeros(len(self.AGN1),'i')
	    ellipseflagr=zeros(len(self.AGN1),'i')
	    ellipserrejectcode=zeros(len(self.AGN1),'i')

	    cutoutpath='/home/alissa/LocalClusters/cutouts/' # where processed cutouts are
	    cutoutpath='/home/rfinn/research/LocalClusters/EllipseProcessedImages/' # where processed cutouts are
	    
	    subdirectories=['NearbyObjects','OffCenter','PartialImages','PeculiarGalaxies','Finished','Finished/reject']
	    
	    p=cutoutpath+self.prefix+'/Finished/m*cutout-24-rot.fits'
	    finalellipsefiles=glob.glob(p)
	    sfinalellipsefiles=''.join(finalellipsefiles)

	    p=cutoutpath+self.prefix+'/Finished/m*cutout-sdss.fits'
	    finalellipsefilesr=glob.glob(p)
	    sfinalellipsefilesr=''.join(finalellipsefiles)
	    for i in range(len(self.agcnumber)):
		    name=str(self.agcnumber[i])
		    if sfinalellipsefiles.find(name) > -1:
			    ellipseflag24[i]=1
		    if sfinalellipsefilesr.find(name) > -1:
			    ellipseflagr[i]=1

	    newcol17=pyfits.Column(name='ELLIPSEFLAG24',format='L',unit='',array=ellipseflag24)
	    newcol18=pyfits.Column(name='ELLIPSEFLAGSDSS',format='L',unit='',array=ellipseflagr)
	    t=ellipseflagr & ellipseflag24
	    newcol19=pyfits.Column(name='ELLIPSEFLAG',format='L',unit='',array=t)
	    

	    # galaxy zoo morphologies  12/1/11
	    print 'matching to galaxy zoo morphologies'
	    #create dictionary with objid names
	    self.agcnumber=tbdata.field('AGCNUMBER')
	    self.agcdict=dict((a,b) for a,b in zip(self.agcnumber,arange(len(self.agcnumber))))
	    #read in morphology data
	    (names,morph,disturb)=self.readmorphcat()
	    morph=array(morph,'i')
	    disturb=array(disturb,'i')
	    self.morphflag=zeros(len(self.agcnumber),'bool')
	    self.morph=zeros(len(self.agcnumber),'i')
	    self.disturb=zeros(len(self.agcnumber),'i')
	    for i in range(len(names)):
		    try:
			    t=self.agcdict[int(names[i])]
			    self.morph[t]=morph[i]
			    self.disturb[t]=disturb[i]
			    self.morphflag[t]=1
		    except KeyError:
			    print 'no match for ',names[i],' in new zoo catalog'
	    
	    fluxautose24=tbdata.field('FLUXAUTOSE24')
	    fluxerrautose24=tbdata.field('FLUXERRAUTOSE24')
	    mipsflux=tbdata.field('MIPSFLUX')
	    mipsfluxerr=tbdata.field('MIPSFLUXERR')
	    apexflag=tbdata.field('APEXFLAG')
	    sex24flag=tbdata.field('SEX24FLAG')
	    flux100=tbdata.field('FLUX100')
	    ## these should be added to mastertable
	    conv=4*pi*(self.cdcm**2)*1.e-6*1.e-23*(3.e8/24.e-6)/Lsol
	    print 'conversion from F24 to L24 = ',conv
	    self.L24_cl=mipsflux*conv
	    self.L24err_cl=mipsfluxerr*conv
	    self.Lir_cl=self.L24_cl*8.#approx conv from papovich
	    self.Lirerr_cl=self.L24err_cl*8.#approx conv from papovich
	    self.SFR24_cl=self.Lir_cl*4.5e-44*Lsol#approx conv from papovich
	    self.SFR24err_cl=self.Lirerr_cl*4.5e-44*Lsol#approx conv from papovich
	    self.SFR24se_cl=(fluxautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
	    self.SFR24seerr_cl=(fluxerrautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
	    self.snr24se=abs(self.SFR24se_cl/self.SFR24seerr_cl)
	    self.superSFR24_cl=self.SFR24_cl*apexflag+self.SFR24se_cl*(~apexflag&sex24flag)
	    self.superSFR24err_cl=self.SFR24err_cl*apexflag+self.SFR24seerr_cl*(~apexflag&sex24flag)
	    self.sSFR_cl=self.SFR24_cl/self.stellarmassr_cl	    
	    self.HImass_cl=2.356e5*flux100/100.*self.cdMpc**2

	    newcol20=pyfits.Column(name='L24_CL',format='E',unit='Solar Luminosity',array=self.L24_cl)
	    newcol21=pyfits.Column(name='L24ERR_CL',format='E',unit='Solar Luminosity',array=self.L24err_cl)
	    newcol22=pyfits.Column(name='LIR_CL',format='E',unit='Solar Luminosity',array=self.Lir_cl)
	    newcol23=pyfits.Column(name='LIRERR_CL',format='E',unit='Solar Luminosity',array=self.Lirerr_cl)
	    newcol24=pyfits.Column(name='SFR24_CL',format='E',unit='Msun/yr',array=self.SFR24_cl)
	    newcol25=pyfits.Column(name='SFR24ERR_CL',format='E',unit='Msun/yr',array=self.SFR24err_cl)
	    newcol26=pyfits.Column(name='SUPERSFR24_CL',format='E',unit='Msun/yr',array=self.superSFR24_cl)
	    newcol27=pyfits.Column(name='SUPERSFR24ERR_CL',format='E',unit='Msun/yr',array=self.superSFR24err_cl)
	    newcol28=pyfits.Column(name='HIMASS_CL',format='E',unit='Msun',array=self.HImass_cl)

 	    ## these should be added to mastertable
	    conv=4*pi*((self.distMpc*1.e6*3.08567758e18)**2)*1.e-6*1.e-23*(3.e8/24.e-6)/Lsol
	    #print 'conversion from F24 to L24 = ',conv
	    self.L24=mipsflux*conv
	    self.L24err=mipsfluxerr*conv
	    self.Lir=self.L24*8.#approx conv from papovich
	    self.Lirerr=self.L24err*8.#approx conv from papovich
	    self.SFR24=self.Lir*4.5e-44*Lsol#approx conv from papovich
	    self.SFR24err=self.Lirerr*4.5e-44*Lsol#approx conv from papovich
	    self.SFR24se=(fluxautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
	    self.SFR24seerr=(fluxerrautose24*141*conv)*8.*4.5e-44*Lsol#approx conv from papovich
	    self.snr24se=abs(self.SFR24se/self.SFR24seerr)
	    self.superSFR24=self.SFR24*apexflag+self.SFR24se*(~apexflag&sex24flag)
	    self.superSFR24err=self.SFR24err*apexflag+self.SFR24seerr*(~apexflag&sex24flag)
	    self.sSFR=self.SFR24/self.stellarmassr
	    
	    self.HImass=2.356e5*flux100/100.*self.cdMpc**2
	    newcol29=pyfits.Column(name='L24',format='E',unit='Solar Luminosity',array=self.L24)
	    newcol30=pyfits.Column(name='L24ERR',format='E',unit='Solar Luminosity',array=self.L24err)
	    newcol31=pyfits.Column(name='LIR',format='E',unit='Solar Luminosity',array=self.Lir)
	    newcol32=pyfits.Column(name='LIRERR',format='E',unit='Solar Luminosity',array=self.Lirerr)
	    newcol33=pyfits.Column(name='SFR24',format='E',unit='Msun/yr',array=self.SFR24)
	    newcol34=pyfits.Column(name='SFR24ERR',format='E',unit='Msun/yr',array=self.SFR24err)
	    newcol35=pyfits.Column(name='SUPERSFR24',format='E',unit='Msun/yr',array=self.superSFR24)
	    newcol36=pyfits.Column(name='SUPERSFR24ERR',format='E',unit='Msun/yr',array=self.superSFR24err)
	    newcol37=pyfits.Column(name='HIMASS',format='E',unit='Msun',array=self.HImass)


	    cdefs1=cdefs.add_col(newcol1)
	    cdefs2=cdefs1.add_col(newcol2)
	    cdefs3=cdefs2.add_col(newcol3)
	    cdefs4=cdefs3.add_col(newcol4)
	    cdefs4a=cdefs4.add_col(newcol4a)
	    cdefs5=cdefs4a.add_col(newcol5)
	    cdefs6=cdefs5.add_col(newcol6)
	    cdefs7=cdefs6.add_col(newcol7)
	    cdefs8=cdefs7.add_col(newcol8)
	    cdefs9=cdefs8.add_col(newcol9)
	    cdefs10=cdefs9.add_col(newcol10)
	    cdefs11=cdefs10.add_col(newcol11)
	    cdefs12=cdefs11.add_col(newcol12)
	    cdefs13=cdefs12.add_col(newcol13)
	    cdefs14=cdefs13.add_col(newcol14)
	    cdefs15=cdefs14.add_col(newcol15)
	    cdefs16=cdefs15.add_col(newcol16)
	    cdefs17=cdefs16.add_col(newcol17)
	    cdefs18=cdefs17.add_col(newcol18)
	    cdefs19=cdefs18.add_col(newcol19)
	    cdefs20=cdefs19.add_col(newcol20)
	    cdefs21=cdefs20.add_col(newcol21)
	    cdefs22=cdefs21.add_col(newcol22)
	    cdefs23=cdefs22.add_col(newcol23)
	    cdefs24=cdefs23.add_col(newcol24)
	    cdefs25=cdefs24.add_col(newcol25)
	    cdefs26=cdefs25.add_col(newcol26)
	    cdefs27=cdefs26.add_col(newcol27)
	    cdefs28=cdefs27.add_col(newcol28)
	    cdefs29=cdefs28.add_col(newcol29)
	    cdefs30=cdefs29.add_col(newcol30)
	    cdefs31=cdefs30.add_col(newcol31)
	    cdefs32=cdefs31.add_col(newcol32)
	    cdefs33=cdefs32.add_col(newcol33)
	    cdefs34=cdefs33.add_col(newcol34)
	    cdefs35=cdefs34.add_col(newcol35)
	    cdefs36=cdefs34.add_col(newcol36)
	    cdefs37=cdefs34.add_col(newcol37)


	    
	    
	    newmaster=pyfits.new_table(cdefs36)
	    newmaster.writeto(s,clobber='yes')


    def makecutoutsrot24(self):
        print 'making rotated 24um cutouts'
	cutouts24=[]
	iraf.imgets(image=self.rotatedimage24,param='CD1_1')#get x plate scale on rotated image
	print iraf.imgets.value
	xplate=abs(float(iraf.imgets.value))#deg/pixel
	dpix=deltaCutout/3600./xplate/2.
	iraf.imgets(image=self.rotatedimage24,param='naxis1')#get x value corresponding to RA 
	xpixmax=(int(iraf.imgets.value))#deg/pixel
	iraf.imgets(image=self.rotatedimage24,param='naxis2')#get x value corresponding to RA 
	ypixmax=(int(iraf.imgets.value))#deg/pixel
	infile=open(self.outcoordsrot24,'r')
	
	#print outcoords
        i=-1
        minsize=20

	for line in infile:

	    #print line
	    if line.find('#') > -1:
		continue
	    if len(line)<2:
		continue
            i=i+1
            if (self.cutout24flag[i] < 0.1):
                continue
            #print i,line,self.ediscsID[i]
            x,y,id=line.split()
            x=float(x)
	    y=float(y)
	    #deltaCutout=100.
	    xmin=int(round(x-dpix))
	    xmax=int(round(x+dpix))
	    ymin=int(round(y-dpix))
	    ymax=int(round(y+dpix))
	    if xmin < 1:
		xmin=1
	    if ymin < 1:
		ymin=1
	    if xmax > xpixmax:
		xmax=xpixmax
	    if ymax > ypixmax:
		ymax=ypixmax

	    s=self.rotatedimage24+'[%i:%i,%i:%i]'%(xmin,xmax,ymin,ymax)
	    print s
	    #print ra[i],dec[i]
            id=self.prefix+'-'+self.agcname[i]+'-'
	    outim=self.cutoutpath+id+'cutout-24-rot.fits'
	    print 'Rotated',outim
            iraf.imcopy(s,outim)
	infile.close()

    def makecutoutsrot24old(self):
        print 'making rotated 24um cutouts'
	cutouts24=[]
	iraf.imgets(image=self.rotatedimage24,param='CD1_1')#get x plate scale on rotated image
	print iraf.imgets.value
	xplate=abs(float(iraf.imgets.value))#deg/pixel
	dpix=deltaCutout/3600./xplate/2.
	iraf.imgets(image=self.rotatedimage24,param='naxis1')#get x value corresponding to RA 
	xpixmax=(int(iraf.imgets.value))#deg/pixel
	iraf.imgets(image=self.rotatedimage24,param='naxis2')#get x value corresponding to RA 
	ypixmax=(int(iraf.imgets.value))#deg/pixel
	infile=open(self.outcoordsrot24,'r')
	
	#print outcoords
        i=-1
        minsize=20

	for line in infile:

	    #print line
	    if line.find('#') > -1:
		continue
	    if len(line)<2:
		continue
            i=i+1
            if (self.cutout24flag[i] < 0.1):
                continue
            #print i,line,self.ediscsID[i]
            x,y,id=line.split()
            x=float(x)
	    y=float(y)
	    #deltaCutout=100.
	    xmin=int(round(x-dpix))
	    xmax=int(round(x+dpix))
	    ymin=int(round(y-dpix))
	    ymax=int(round(y+dpix))
	    if xmin < 1:
		xmin=1
	    if ymin < 1:
		ymin=1
	    if xmax > xpixmax:
		xmax=xpixmax
	    if ymax > ypixmax:
		ymax=ypixmax

	    s=self.rotatedimage24+'[%i:%i,%i:%i]'%(xmin,xmax,ymin,ymax)
	    print s
	    #print ra[i],dec[i]
            id=self.prefix+'-'+self.agcname[i]+'-'
	    outim=self.cutoutpath+id+'cutout-24-rot.fits'
	    print 'Rotated',outim
            iraf.imcopy(s,outim)
	infile.close()


    def displaycutouts(self):
        hstfiles=[]
        mipsfiles=[]
        f=self.prefix+'cutoutfiles.dat'
        infile=open(f,'r')
	cutout24flag=[]
        for line in infile:
            t=line.split()
	    cutout24flag.append(float(t[1]))
	    id=t[0]
	    hstfiles.append(id+"cutout.fits")
	    mipsfiles.append(id+"cutout24.fits")
	#hstfiles=self.cutouts
	#mipsfiles=self.cutouts24
	print "number of cutouts = ",len(hstfiles),len(mipsfiles)
	for i in range(len(mipsfiles)):
	    if cutout24flag[i] > 0.1:
		print hstfiles[i],mipsfiles[i]
		print self.ediscsID[i]
		print 'mips flag = ',self.mipsflag[i]
		print '24um flux (SFSR) = ',self.f24[i],self.SFRir[i]
		print 'flux80 flag = ',self.f80flag[i]
		print 'spec memb flag (z) = ',self.specmembflag[i],self.specz[i]
		try:
		    os.system('xpaset -p ds9 frame 2')
		    s='xpaset -p ds9 file '+mipsfiles[i]
		    s='cat '+mipsfiles[i]+' | xpaset ds9 fits '
		    os.system(s)
		    os.system('xpaset -p ds9 frame 1')
		    s='xpaset -p ds9 file '+hstfiles[i]
		    s='cat '+hstfiles[i]+' | xpaset ds9 fits '
		    os.system(s)
		    os.system('xpaset -p ds9 match frames wcs')
		except:
		    print 'problem displaying ',hstfiles[i] 
		#iraf.display(hstfiles[i],frame=1,contrast=0.01,fill='No')
		#iraf.display(mipsfiles[i],frame=2,contrast=.01,fill='No',zscale='no',zrange='no',z1=-.01,z2=.3)
		t=raw_input('hit any key to continue, q to quit \n')
		print t
		t=str(t)
		if t.find('q') > -1:
		    break

    def plotcutouts(self):
	clf()
	cla()
	s=self.prefix+'Cutouts.pdf'
	#pdf=PdfFile('s')
	figure(figsize=(12,16))
	subplots_adjust(left=0.1, right=.95,bottom=.1,top=0.95,wspace=0.001,hspace=0.001)
	nx=6
	ny=8
	nplot=0
	npage=1
	nplotmax=float(nx)*float(ny)
	#hstfiles=glob.glob('EDCSNJ*cutout.fits')
	#mipsfiles=glob.glob('EDCSNJ*cutout24.fits')

        hstfiles=[]
        mipsfiles=[]
        f=self.prefix+'cutoutfiles.dat'
        infile=open(f,'r')
        for line in infile:
            t=line.split()
            if (float(t[1])>.1):#24um image is ok
                id=t[0]
                hstfiles.append(id+"cutout.fits")
                mipsfiles.append(id+"cutout24.fits")
	#hstfiles=self.cutouts
	#mipsfiles=self.cutouts24
	print "number of cutouts = ",len(hstfiles),len(mipsfiles)

	for i in range(len(mipsfiles)):
	    if (self.cutout24flag[i] < 0.5):
		continue
	    if (float(nplot)> (nplotmax-1)):
		f=self.prefix+'AllCutouts'+str(npage)+'.eps'
		npage=npage+1
		#savefig(pdf,format='pdf')
		savefig(f)
		print "starting new plot",npage
		clf()
		cla()
		nplot=0
	    name=hstfiles[i]
	    for j in range(2):
		nplot=nplot+1
		k=nplot
		print k,nplotmax,mipsfiles[i]
		subplot(ny,nx,nplot)
		if (j == 0):
		    fits=pyfits.open(hstfiles[i])
		    name=hstfiles[i]
		    t=name.split('cutout')
		    im=fits[0].data.copy()
		    lim=.07
		    im[where(im>(lim))]=lim
		    nave=3.5
		    im[where(im<0)]=0
		else:
		    fits=pyfits.open(mipsfiles[i])
		    name=mipsfiles[i]
		    im=fits[0].data.copy()
		    im[where(im<0)]=0
		    im[where(im>(.08))]=.08
		    axis([1.,5.,1.,5.])
	
		fits.close()
		axis('equal')
		imshow(-1.*(im),interpolation='nearest',origin='upper',cmap='gray')#,vmin=myvmin,vmax=myvmax)
		ax=gca()
		ax.set_xticklabels(([]))
		ax=gca()
		ax.set_yticklabels(([]))
		col='k'
	    #print col
		t=self.ediscsID[i]
		junk,id=t.split('J')

		text(.5,.82,id,fontsize=10,transform=ax.transAxes,color=col,horizontalalignment='center')
		if (j == 0):
		    if (self.specmembflag[i] > 0.1):
			s="%6.4f"%(self.specz[i])
			text(.6,.75,s,fontsize=10,transform=ax.transAxes,color=col,horizontalalignment='left')
		else:
		    col='r'
		    s="%2.1f"%(self.mipsflag[i])
		    text(.6,.75,s,fontsize=10,transform=ax.transAxes,color=col,horizontalalignment='left')

		    if (self.mipsflag[i] > 0.1):
			s="%5.1f, %6.1f"%(self.f24[i],self.SFRir[i])
			text(.6,.65,s,fontsize=10,transform=ax.transAxes,color=col,horizontalalignment='left')

	f=self.prefix+'AllCutouts'+str(npage)+'.eps'
	savefig(f)
	#savefig(pdf,format='pdf')
	#pdf.close()


    def doall(self):
	self.getcoords()
	#self.transcoords()
	#self.makecutouts()
	#self.transcoords24()
	#self.makecutouts24()
	#self.plotcutouts()
	self.displaycutouts()


    def rotate24(self):
        #iraf.imgets(image=self.image24,param='CROTA2')#[deg] Orientation of axis 2 (W of N, +=CW)
        iraf.imgets(image=self.image24,param='PA')#[deg] Orientation of axis 2 (W of N, +=CW)
        print iraf.imgets.value
        rot24=float(iraf.imgets.value)
        angle=rot24
        iraf.rotate(input=self.image24,output=self.rotatedimage24,rotation=angle,ncols=0,nlines=0)

    def subtractskysdssr(self):
        #move to where SDSS images are

	#sdssimagepath='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'
	os.chdir(self.sdssimagepath)
	#cp sextractor files to this director
	#os.system('cp /home/rfinn/research/LocalClusters/sextractor/default.sex.sdss default.sex')
	os.system('cp /home/rfinn/research/LocalClusters/sextractor/default.param .')
	os.system('cp /home/rfinn/research/LocalClusters/sextractor/default.nnw .')
	#os.system('cp /home/rfinn/research/LocalClusters/sextractor/*.conv .')
        #run sextractor on r-band image 1. 
	# saved as skysubtracted.fits by sextractor
	s='sextractor '+self.sdssrim1+' -c /home/rfinn/research/LocalClusters/sextractor/default.sex.sdss'
	os.system(s)

	#save sky-subtracted image as self.skysubsdssr
	t=self.sdssrim1.split('.')
	self.skysubsdssrim1=t[0]+'s.fits'
	s='cp skysubtracted.fits '+self.skysubsdssrim1
	os.system(s)
	#save test.cat
	s='cp test.cat '+self.sdsstestcat1
	os.system(s)
	self.sdssrim1testcat=t[0]+'-test.cat'
        #run sextractor on r-band image 2. 
	# saved as skysubtracted.fits by sextractor
	s='sextractor '+self.sdssrim2+' -c /home/rfinn/research/LocalClusters/sextractor/default.sex.sdss'
	os.system(s)

	#save sky-subtracted image as self.skysubsdssr
	t=self.sdssrim2.split('.')
	self.skysubsdssrim2=t[0]+'s.fits'
	s='cp skysubtracted.fits '+self.skysubsdssrim2
	os.system(s)
	#save test.cat
	s='cp test.cat '+self.sdsstestcat2
	os.system(s)



    def runsextractor24(self):
        #move to where SDSS images are


	os.chdir(self.imagepath24)
	#cp sextractor files to this director
	os.system('cp /home/rfinn/research/LocalClusters/sextractor/default.param .')
	os.system('cp /home/rfinn/research/LocalClusters/sextractor/default.nnw .')
	#os.system('cp /home/rfinn/research/LocalClusters/sextractor/*.conv .')
	#s='cp '+self.noise24+' weight.fits'
	#os.system(s)
	
        #run sextractor on r-band image 1. 
	# saved as skysubtracted.fits by sextractor
	s='sextractor '+self.image24+' -c /home/rfinn/research/LocalClusters/sextractor/default.sex.24um -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE '+self.noise24
	os.system(s)

	#save test.cat
	s='cp test.cat '+self.testcat24
	os.system(s)

	#save check.fits
	s='mv check.fits '+self.prefix+'-check.fits'
	os.system(s)

#agc=GetAGC()
## commented these two lines while debugging getsdsscoords()
agc=ReadAGCsav.agc()#only need this line if running match2agc
##
mpa=mpacatalog()
ncl=0


names=['MKW11','MKW8','AWM4', 'A2063','A2052','NGC6107','Coma','A1367','Hercules']
#names=['NGC6107', 'Coma','A1367','Hercules']

#clusterRA=[202.38000,220.1796,241.2375, 230.7578, 229.1896, 244.333750,194.9531, 176.1231, 241.3125]
#clusterDec=[11.78861,3.4530, 23.9206, 8.6394, 7.0003, 34.901389, 27.9807, 19.8391, 17.7485]
#clusterz=[.022849,.027,.031755,.034937,.035491,.030658,.023,.028,.037]

#mipsimage=['/home/rfinn/research/LocalClusters/cutouts/MKW11/MKW11R1.fits']
#mkw11=cluster()
#mkw11.getcoords('/home/rfinn/research/LocalClusters/cutouts/MKW11/mkw11MCO.dat')

for ncl in range(len(names)):
#for ncl in range(3):
#these=array([0,3,4,5,7,8],'i')
#for ncl in these:
#for ncl in range(1,len(names)):
#for ncl in range(6,7):#coma only
#for ncl in range(6,len(names)):
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

    #print 'running ',cl.prefix    
    s=cl.cutoutpath
    s2='mkdir '+s
    os.system(s2)
    os.chdir(s)
    #print cl.prefix,': cut for Martha AGC'
    print '%8s RA  range = %12.8f %12.8f'%(cl.prefix,clusterRA[names[ncl]]-3.3/2.,clusterRA[names[ncl]]+3.3/2.)
    print '%8s Dec range = %12.8f %12.8f'%(cl.prefix,clusterDec[names[ncl]]-3.3/2.,clusterDec[names[ncl]]+3.3/2.)
    

#####   Run the following functions to create the new mastertables

    #commenting out these tasks to just run r-band sky subtraction
    print 'starting getsdsscoords'
##
    cl.getsdsscoords()
##
    cl.getsdssphotcoords()
    #uncomment the following line to run sextractor on the sdss r-band images
#    cl.subtractskysdssr()
    #uncomment the following line to run sextractor on the 24um images
#    cl.runsextractor24()


##
    cl.readapexcat()




##
    cl.match2agc()

#    print 'starting readmembcat'
##
    cl.readmembcat()
#    print 'starting transcoords24'
##
    cl.transcoords24()
#    print 'ran transcoords24'

#    cl.rotate24()
    print 'starting makecutous24'
##
    cl.makecutouts24()

#    cl.transcoords24rot()
#    cl.makecutoutsrot24()

# band = 0=u,1=g,2=r,3=i,4=z 
#    band=2
#    if ncl <7:
#	    cl.makesdsscutouts(band)
#    elif ncl > 6:
#	    print 'running makesdsscutoutsv2'
#	    cl.makesdsscutoutsv2(band)
#    band=1 #make g-band cutouts
#    if ncl <7:
#	    cl.makesdsscutouts(band)
#    elif ncl > 6:
#	    print 'running makesdsscutoutsv2'
#	    cl.makesdsscutoutsv2(band)




#    print 'starting match2agc'




##
    cl.appendcutout24flagtomastertable()
##
    cl.appendmorecolstomastertable()

###    End of functions that are needed to create mastertable

    # uncomment the following line to match AGC-SDSS cat with sextractor cats and write master table
#    cl.createmastertable()

#mkw11.getsdsscat()
#mkw11.mksdssregfile()

#mkw11.readSkidmore('/home/rfinn/research/LocalClusters/SkidmoreCatalogs/MKW11regionMCO.v2.csv')
#mkw11.rotate24()
#mkw11.getsdsscoords()
#mkw11.makesdsscutouts()


#hist(mkw11.vr,20)
#savefig('velhist.eps')



def plotagcLCS():
    figure()
    clf()
    cra=[]
    cdec=[]
    for i in range(len(clusterz)):
        print names[i]
        text(clusterRA[names[i]],clusterDec[names[i]],names[i],fontsize=16,color='r')
	cra.append(clusterRA[names[i]])
	cdec.append(clusterDec[names[i]])
    plot(agc.radeg[velflag],agc.decdeg[velflag],'k.')
    plot(cra,cdec,'ro',markersize=16,color='b')
    xlabel('RA (deg)')
    ylabel('Dec (deg)')
    

#	IntegerFields=array([12,13,14,15,63,64,65,66, 82,83,84],'i')	

'''
1 - n.distance,
g.ra,
g.dec,
g.u,
g.g,
g.r,
g.i,
g.z,
s.z,
10 - l.ew,
l.ewErr,
s.plate,
s.fiberID,
s.tile,
g.objID,
g.petroMag_u,
g.petroMag_g,
g.petroMag_r,
g.petroMag_i,
20 - g.petroMag_z,
g.petroRad_u,
g.petroRad_g,
g.petroRad_r,
g.petroRad_i,
g.petroRad_z,
g.petroR50_u,
g.petroR50_g,
g.petroR50_r,
g.petroR50_i,
30 - g.petroR50_z,
g.petroR90_u,
g.petroR90_g,
g.petroR90_r,
g.petroR90_i,
g.petroR90_z,
g.isoA_r,
g.isoB_r,
g.isoPhi_r,
g.isoPhiErr_r,
40 - g.deVRad_r,
g.deVRadErr_r,
g.deVPhi_r,
g.deVPhiErr_r,
g.deVMag_r,
g.expRad_r,
g.expRadErr_r,
g.expAB_r,
g.expABErr_r,
g.expPhi_r,
50 - g.expPhiErr_r,
g.expMag_r,
g.expMagErr_r,
g.extinction_u,
g.extinction_g,
g.extinction_r,
g.extinction_i,
g.extinction_z,
g.dered_u,
g.dered_g,

60 - g.dered_r,
g.dered_i,
g.dered_z,
g.run,
g.rerun,
g.camcol,
g.field,
g.err_u,
g.err_g,
g.err_r,
70 - g.err_i,
g.err_z,
g.rowc_u,
g.rowc_g,
g.rowc_r,
g.rowc_i,
g.rowc_z,
g.colc_u,
g.colc_g,
g.colc_r,
80 - g.colc_i,
g.colc_z 
'''
