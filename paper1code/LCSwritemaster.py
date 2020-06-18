#!/usr/bin/env python
"""
run from /Users/rfinn/clusters/spitzer/GroupLirgs

"""

from pylab import *
from pyraf import iraf
import pyfits
import sqlcl
import glob
import os
import ReadAGCsav
from matplotlib.backends.backend_pdf import PdfFile

delta=100.#width of cutouts in arcsec
ramin=170.
ramax=250.
decmax=38.
zmin=0.0133#min z cut, set by z(Coma) - 3x1000 km/s
zmax=0.0433#max z cut, set by z(A2052)+ 3xsigma km/s
vmin=zmin*3.e5
vmax=zmax*3.e5

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


class cluster:
    def __init__(self):
	self.prefix=names[ncl]
        self.cra=clusterRA[ncl]
        self.cdec=clusterDec[ncl]
        self.cz=clusterz[ncl]
        self.dr=3.#get galaxies w/in 3 degrees
	self.cutoutpath=cutoutpath+self.prefix+'/'
        #if self.prefix.find('A2063')>-1:
        #    self.dr=5.

#mipsimage=['/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits','/home/rfinn/research/LocalClusters/cutouts/MKW11/FullMKW11ch1rf_mosaic_minus_median_extract.fits']
        self.image24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_minus_median_extract.fits'
	self.noise24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/Full'+self.prefix+'ch1rf_mosaic_unc.fits'
        self.sdssrim1='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R1.fits'
        self.sdssrim2='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R2.fits'
        self.sdssrim1skysub='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R1s.fits'
        self.sdssrim2skysub='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/SDSS/'+self.prefix+'R2s.fits'
        self.rotatedimage24='/home/rfinn/research/LocalClusters/Images/'+self.prefix+'/24um/'+'r'+self.prefix+'-rotated-24.fits'
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




    def getsdsscat(self):
        print 'Getting SDSS spec cat for ',self.prefix
        drsearch=3.*60.#search radius in arcmin for sdss query
        #zmin=self.cz-.005
        #zmax=self.cz+.005
        #from this, we will make a field sample and a cluster sample
        query="select n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and (g.PrimTarget & 0x00000040) > 0 and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)
        try:
            lines=sqlcl.query(query).readlines()
        except IOError:
            print "IOError for cluster",self.prefix,i," trying spec query again"
            lines=sqlcl.query(query).readlines()
        print self.prefix,": got number + 1 of spec objects = ",len(lines)
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.dat'
        outfile=open(n,'w')
        j=0
        if (len(lines) > 1.):
            for line in lines[1:]:
                if j < 0:
                    print line
                    j=j+1
                outfile.write(line)
        outfile.close()

 

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

    def getsdsscoords(self):
        #n.distance,g.ra,g.dec, g.u, g.g, g.r, g.i, g.z, s.z,l.ew,l.ewErr from galaxy g, specobj s, specline l, dbo.fGetNearbyObjEq(%12.8f,%12.8f,%8.3f) n where g.objid = s.bestobjid and g.objID = n.objID and l.specobjid = s.specobjid and s.z < %5.4f and s.z > %5.4f and (g.PrimTarget & 0x00000040) > 0 and l.LineId = 6565 order by distance" % (self.cra,self.cdec,drsearch,zmax,zmin)
        ngal=0
        n='/home/rfinn/research/LocalClusters/SDSSCatalogs/'+str(self.prefix)+'galaxy.dat'
        outfile=open(n,'r')
        for line in outfile:
            ngal += 1
        outfile.close()

	print 'got ',ngal,' galaxies in sdss catalog'
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

        outfile=open(n,'r')
        i=0
        id=[]
        for line in outfile:
                t=line.split(',')
                for j in range(len(t)):
                    t[j]=float(t[j])

                (self.sdssdr[i],self.sdssRA[i],self.sdssDec[i],self.sdssu[i],self.sdssg[i],self.sdssr[i],self.sdssi[i],self.sdssz[i],self.sdssspecz[i],self.sdssHaEW[i],self.sdssHaEWerr[i])=t
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


 
    def getcoords(self,infile):

	in1=open(infile,'r')
        ngal=0
	for line in in1:
            if line.find('#') >-1:
                continue
            ngal += 1
        in1.close()

        self.RA=zeros(ngal,'f')
        self.Dec=zeros(ngal,'f')
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

    def match2agc(self):
        
	delta=20/3600.#match radius of 60 arcsec
        dagc=sqrt((self.cra-agc.radeg)**2+(self.cdec-agc.decdeg)**2)
        thisclflag=(dagc<self.dr)#agc galaxis w/in 3 degrees from cluster center
        agcra=agc.radeg[thisclflag]
        agcdec=agc.decdeg[thisclflag]
	agcnumber=agc.agcnumber[thisclflag]
	matchsdss2agcflag=zeros(len(self.sdssRA),'i')#keep track of sdss galaxies that are matched to agc
	matchagc2sdssflag=zeros(len(agcra),'i')#keep track of agc galaxies that are matched to sdss spec
	matchagc2sdssindex=zeros(len(agcra),'i')#keep track of index of agc galaxies that are matched to sdss spec
        for i in range(len(agcra)):
		imatch, matchflag,nmatch= findnearest(agcra[i],agcdec[i],self.sdssRA,self.sdssDec,delta)
		if matchflag > 0.1:
			matchagc2sdssflag[i]=1
			matchagc2sdssindex[i]=imatch
			matchsdss2agcflag[imatch]=1

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

	nagc=len(agcra)
	nsdss=len(fra)-len(agcra)
	agcflag=array((ones(nagc).tolist()+(zeros(nsdss)).tolist()),'i')
	sdssflag=array((matchagc2sdssflag.tolist()+(ones(nsdss)).tolist()),'i')
	#write out a fits table
	c1=pyfits.Column(name='AGCflag',format='L',array=agcflag)
	c2=pyfits.Column(name='SDSSflag',format='L',array=sdssflag)
	c3=pyfits.Column(name='SEXSDSSflag',format='L',array=matchagc2sexsdssflag)
	c4=pyfits.Column(name='SEX24FLAG',format='L',array=matchagc2sex24flag)
	agcnum=array((agcnumber.tolist()+fakeindex),'i')
	c5=pyfits.Column(name='AGCNUMBER',format='J',array=agcnum)
	c6=pyfits.Column(name='AGC-RA',format='D',unit='DEG',array=fra)
	c7=pyfits.Column(name='AGC-DEC',format='D',unit='DEG',array=fdec)

	t=array((agc.a100[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c8=pyfits.Column(name='A100',format='I',unit='arcmin*100',array=t)

    	t=array((agc.b100[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c9=pyfits.Column(name='B100',format='I',unit='arcmin*100',array=t)

    	t=array((agc.mag10[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c10=pyfits.Column(name='MAG10',format='I',unit='MAG',array=t)

	t=array((agc.posang[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c11=pyfits.Column(name='POSANG',format='E',unit='DEG',array=t)

	t=array((agc.bsteintype[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	c12=pyfits.Column(name='BSTEINTYPE',format='I',array=t)
	t=array((agc.vopt[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c13=pyfits.Column(name='VOPT',format='J',unit='km/s',array=t)
	t=array((agc.verr[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c14=pyfits.Column(name='VERR',format='J',unit='km/s',array=t)
	t=array((agc.vsource[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c15=pyfits.Column(name='VSOURCE',format='J',unit='Jy/(km/s)*100',array=t)
	t=array((agc.flux100[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c16=pyfits.Column(name='FLUX100',format='J',unit='mJy*100',array=t)
	t=array((agc.rms100[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c17=pyfits.Column(name='RMS100',format='J',unit='km/s',array=t)
	t=array((agc.v21[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c18=pyfits.Column(name='V21',format='J',unit='km/s',array=t)
	t=array((agc.width[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c19=pyfits.Column(name='WIDTH',format='J',unit='km/s',array=t)
	t=array((agc.widtherr[thisclflag].tolist()+zeros(nsdss).tolist()),'f')
	c20=pyfits.Column(name='WIDTHERR',format='J',unit='km/s',array=t)
	#define columns
	#t=array((agc.posang[thisclflag].tolist()+zeros(nsdss).tolist()),'i')
	#c21=pyfits.Column(name='POSANG',format='E',unit='DEG',array=t)
	
#	columnsSDSS=['SDSSRA','SDSSDEC','uMAG','gMAG','rMAG','iMAG','zMAG','SPECZ','HAEW','HAEWERR']
#	unitsSDSS=['deg','deg','MAG','MAG','MAG','MAG','MAG','v/cz','angstrom','angstrom']
#	typeSDSS=['D','D','E','E','E','E','E','E','E','E']

	t=array((self.sdssRA[matchagc2sdssindex].tolist()+self.sdssRA[scoordindex].tolist()),'f')
	c22=pyfits.Column(name='SDSSRA',format='E',unit='DEG',array=t)

	t=array((self.sdssDec[matchagc2sdssindex].tolist()+self.sdssDec[scoordindex].tolist()),'f')
	c23=pyfits.Column(name='SDSSDEC',format='E',unit='DEG',array=t)

	t=array((self.sdssu[matchagc2sdssindex].tolist()+self.sdssu[scoordindex].tolist()),'f')
	c24=pyfits.Column(name='SDSSU',format='E',unit='MAG',array=t)

	t=array((self.sdssg[matchagc2sdssindex].tolist()+self.sdssg[scoordindex].tolist()),'f')
	c25=pyfits.Column(name='SDSSG',format='E',unit='MAG',array=t)

	t=array((self.sdssr[matchagc2sdssindex].tolist()+self.sdssr[scoordindex].tolist()),'f')
	c26=pyfits.Column(name='SDSSR',format='E',unit='MAG',array=t)

	t=array((self.sdssi[matchagc2sdssindex].tolist()+self.sdssi[scoordindex].tolist()),'f')
	c27=pyfits.Column(name='SDSSI',format='E',unit='MAG',array=t)

	t=array((self.sdssz[matchagc2sdssindex].tolist()+self.sdssz[scoordindex].tolist()),'f')
	c28=pyfits.Column(name='SDSSZ',format='E',unit='MAG',array=t)

	t=array((self.sdssspecz[matchagc2sdssindex].tolist()+self.sdssspecz[scoordindex].tolist()),'f')
	c29=pyfits.Column(name='SDSSSPECZ',format='E',unit='MAG',array=t)

	t=array((self.sdssHaEW[matchagc2sdssindex].tolist()+self.sdssHaEW[scoordindex].tolist()),'f')
	c30=pyfits.Column(name='SDSSHAEW',format='E',unit='ANGSTROM',array=t)

	t=array((self.sdssHaEWerr[matchagc2sdssindex].tolist()+self.sdssHaEWerr[scoordindex].tolist()),'f')
	c31=pyfits.Column(name='SDSSHAEWERR',format='E',unit='ANGSTROM',array=t)


#	columnsSE=['NUMBER','X_IMAGE','Y_IMAGE','XMIN_IMAGE','XMAX_IMAGE','YMIN_IMAGE','YMAX_IMAGE','ALPHA_J2000','DELTA_J2000','FLUX_ISO','FLUXERR_ISO','MAG_ISO','MAGERR_ISO','FLUX_AUTO','FLUXERR_AUTO','MAG_AUTO','MAGERR_AUTO','FLUX_PETRO','FLUXERR_PETRO','MAG_PETRO','MAGERR_PETRO','KRON_RADIUS','PETRO_RADIUS','FLUX_RADIUS','ISOAREA_IMAGE','A_WORLD','B_WORLD','THETA_IMAGE','ERRTHETA_IMAGE','THETA_J2000','ERRTHETA_J2000','ELONGATION','ELLIPTICITY','FWHM_WORLD','FLAGS','CLASS_STAR']

        t=tbdata.field('NUMBER')
        c32=pyfits.Column(name='NUMBERSER',format='J',array=t[matchagc2sexsdssindex])

        t=tbdata.field('X_IMAGE')
        c33=pyfits.Column(name='XIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('Y_IMAGE')
        c34=pyfits.Column(name='YIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('XMIN_IMAGE')
        c35=pyfits.Column(name='XMINIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('XMAX_IMAGE')
        c36=pyfits.Column(name='XMAXIMAGESER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('YMIN_IMAGE')
        c37=pyfits.Column(name='YMINIMAGESER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ALPHA_J2000')
        c38=pyfits.Column(name='RASER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])
        t=tbdata.field('DELTA_J2000')
        c39=pyfits.Column(name='DECSER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLUX_ISO')
        c40=pyfits.Column(name='FLUXISOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])
        t=tbdata.field('FLUXERR_ISO')
        c41=pyfits.Column(name='FLUXERRISOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])

        t=tbdata.field('MAG_ISO')
        c42=pyfits.Column(name='MAGISOSER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('MAGERR_ISO')
        c43=pyfits.Column(name='MAGERRISOSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLUX_AUTO')
        c44=pyfits.Column(name='FLUXAUTOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])
        t=tbdata.field('FLUXERR_AUTO')
        c45=pyfits.Column(name='FLUXERRAUTOSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])

        t=tbdata.field('MAG_AUTO')
        c46=pyfits.Column(name='MAGAUTOSER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('MAGERR_AUTO')
        c47=pyfits.Column(name='MAGERRAUTOSER',format='E',array=t[matchagc2sexsdssindex])


        t=tbdata.field('FLUX_PETRO')
        c48=pyfits.Column(name='FLUXPETROSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])
        t=tbdata.field('FLUXERR_PETRO')
        c49=pyfits.Column(name='FLUXERRPETROSER',format='E',unit='ADU/S',array=t[matchagc2sexsdssindex])

        t=tbdata.field('MAG_PETRO')
        c50=pyfits.Column(name='MAGPETROSER',format='E',array=t[matchagc2sexsdssindex])
        t=tbdata.field('MAGERR_PETRO')
        c51=pyfits.Column(name='MAGERRPETROSER',format='E',array=t[matchagc2sexsdssindex])


        t=tbdata.field('KRON_RADIUS')
        c52=pyfits.Column(name='KRONRADSER',format='E',unit='PIXELS',array=t[matchagc2sexsdssindex])

        t=tbdata.field('PETRO_RADIUS')
        c53=pyfits.Column(name='PETRORADSER',format='E',unit='PIXELS',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLUX_RADIUS')
        c54=pyfits.Column(name='FLUXRADSER',format='E',unit='PIXELS',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ISOAREA_IMAGE')
        c55=pyfits.Column(name='ISOAREASER',format='E',unit='PIXELSSQ',array=t[matchagc2sexsdssindex])

        t=tbdata.field('A_WORLD')
        c56=pyfits.Column(name='AWORLDSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('B_WORLD')
        c57=pyfits.Column(name='BWORLDSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('THETA_IMAGE')
        c58=pyfits.Column(name='THETASER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ERRTHETA_IMAGE')
        c59=pyfits.Column(name='ERRTHETASER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('THETA_J2000')
        c60=pyfits.Column(name='THETAJ2000SER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ERRTHETA_J2000')
        c61=pyfits.Column(name='ERRTHETAJ2000SER',format='E',unit='DEG',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ELONGATION')
        c62=pyfits.Column(name='ELONGATIONSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('ELLIPTICITY')
        c63=pyfits.Column(name='ELLIPTICITYSER',format='E',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FWHM_WORLD')
        c64=pyfits.Column(name='FWHMSER',format='E',unit='ARCSEC',array=t[matchagc2sexsdssindex])

        t=tbdata.field('FLAGS')
        c65=pyfits.Column(name='FLAGSSER',format='J',unit='',array=t[matchagc2sexsdssindex])

        t=tbdata.field('CLASS_STAR')
        c66=pyfits.Column(name='CLASSSTARSER',format='E',unit='',array=t[matchagc2sexsdssindex])


        t=tbdata24.field('NUMBER')
        c67=pyfits.Column(name='NUMBERSE24',format='J',array=t[matchagc2sex24index])

        t=tbdata24.field('X_IMAGE')
        c68=pyfits.Column(name='XIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('Y_IMAGE')
        c69=pyfits.Column(name='YIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('XMIN_IMAGE')
        c70=pyfits.Column(name='XMINIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('XMAX_IMAGE')
        c71=pyfits.Column(name='XMAXIMAGESE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('YMIN_IMAGE')
        c72=pyfits.Column(name='YMINIMAGESE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('ALPHA_J2000')
        c73=pyfits.Column(name='RASE24',format='E',unit='DEG',array=t[matchagc2sex24index])
        t=tbdata24.field('DELTA_J2000')
        c74=pyfits.Column(name='DECSE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('FLUX_ISO')
        c75=pyfits.Column(name='FLUXISOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])
        t=tbdata24.field('FLUXERR_ISO')
        c76=pyfits.Column(name='FLUXERRISOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])

        t=tbdata24.field('MAG_ISO')
        c77=pyfits.Column(name='MAGISOSE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('MAGERR_ISO')
        c78=pyfits.Column(name='MAGERRISOSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('FLUX_AUTO')
        c79=pyfits.Column(name='FLUXAUTOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])
        t=tbdata24.field('FLUXERR_AUTO')
        c80=pyfits.Column(name='FLUXERRAUTOSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])

        t=tbdata24.field('MAG_AUTO')
        c81=pyfits.Column(name='MAGAUTOSE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('MAGERR_AUTO')
        c82=pyfits.Column(name='MAGERRAUTOSE24',format='E',array=t[matchagc2sex24index])


        t=tbdata24.field('FLUX_PETRO')
        c83=pyfits.Column(name='FLUXPETROSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])
        t=tbdata24.field('FLUXERR_PETRO')
        c84=pyfits.Column(name='FLUXERRPETROSE24',format='E',unit='ADU/S',array=t[matchagc2sex24index])

        t=tbdata24.field('MAG_PETRO')
        c85=pyfits.Column(name='MAGPETROSE24',format='E',array=t[matchagc2sex24index])
        t=tbdata24.field('MAGERR_PETRO')
        c86=pyfits.Column(name='MAGERRPETROSE24',format='E',array=t[matchagc2sex24index])


        t=tbdata24.field('KRON_RADIUS')
        c87=pyfits.Column(name='KRONRADSE24',format='E',unit='PIXELS',array=t[matchagc2sex24index])

        t=tbdata24.field('PETRO_RADIUS')
        c88=pyfits.Column(name='PETRORADSE24',format='E',unit='PIXELS',array=t[matchagc2sex24index])

        t=tbdata24.field('FLUX_RADIUS')
        c89=pyfits.Column(name='FLUXRADSE24',format='E',unit='PIXELS',array=t[matchagc2sex24index])

        t=tbdata24.field('ISOAREA_IMAGE')
        c90=pyfits.Column(name='ISOAREASE24',format='E',unit='PIXELSSQ',array=t[matchagc2sex24index])

        t=tbdata24.field('A_WORLD')
        c91=pyfits.Column(name='AWORLDSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('B_WORLD')
        c92=pyfits.Column(name='BWORLDSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('THETA_IMAGE')
        c93=pyfits.Column(name='THETASE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('ERRTHETA_IMAGE')
        c94=pyfits.Column(name='ERRTHETASE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('THETA_J2000')
        c95=pyfits.Column(name='THETAJ2000SE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('ERRTHETA_J2000')
        c96=pyfits.Column(name='ERRTHETAJ2000SE24',format='E',unit='DEG',array=t[matchagc2sex24index])

        t=tbdata24.field('ELONGATION')
        c97=pyfits.Column(name='ELONGATIONSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('ELLIPTICITY')
        c98=pyfits.Column(name='ELLIPTICITYSE24',format='E',array=t[matchagc2sex24index])

        t=tbdata24.field('FWHM_WORLD')
        c99=pyfits.Column(name='FWHMSE24',format='E',unit='ARCSEC',array=t[matchagc2sex24index])

        t=tbdata24.field('FLAGS')
        c100=pyfits.Column(name='FLAGSSE24',format='J',unit='',array=t[matchagc2sex24index])

        t=tbdata24.field('CLASS_STAR')
        c101=pyfits.Column(name='CLASSSTARSE24',format='E',unit='',array=t[matchagc2sex24index])

	mastertb=pyfits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,c71,c72,c73,c74,c75,c76,c77,c78,c79,c80,c81,c82,c83,c84,c85,c86,c87,c88,c89,c90,c91,c92,c93,c94,c95,c96,c97,c98,c99,c100,c101])
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


 



#agc=GetAGC()
agc=ReadAGCsav.agc()
ncl=0
names=['MKW11','MKW8','AWM4', 'A2063','A2052','NGC6107', 'Coma','A1367','Hercules']
clusterRA=[202.38000,220.1796,241.2375, 230.7578, 229.1896, 244.333750,194.9531, 176.1231, 241.3125]
clusterDec=[11.78861,3.4530, 23.9206, 8.6394, 7.0003, 34.901389, 27.9807, 19.8391, 17.7485]
clusterz=[.022849,.027,.031755,.034937,.035491,.030658,.023,.028,.037]

#mipsimage=['/home/rfinn/research/LocalClusters/cutouts/MKW11/MKW11R1.fits']
#mkw11=cluster()
#mkw11.getcoords('/home/rfinn/research/LocalClusters/cutouts/MKW11/mkw11MCO.dat')

for ncl in range(len(names)):

#for ncl in range(1):
#for ncl in range(3,4):
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
    print '%8s RA  range = %12.8f %12.8f'%(cl.prefix,clusterRA[ncl]-3.3/2.,clusterRA[ncl]+3.3/2.)
    print '%8s Dec range = %12.8f %12.8f'%(cl.prefix,clusterDec[ncl]-3.3/2.,clusterDec[ncl]+3.3/2.)
    
    cl.getsdsscat()



    #commenting out these tasks to just run r-band sky subtraction
#    print 'starting getsdsscoords'
#    cl.getsdsscoords()

    #uncomment the following line to run sextractor on the sdss r-band images
#    cl.subtractskysdssr()

    #uncomment the following line to run sextractor on the 24um images
#    cl.runsextractor24()


#    print 'starting match2agc'
#    cl.match2agc()


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
    for i in range(len(clusterz)):
        print names[i]
        text(clusterRA[i],clusterDec[i],names[i],fontsize=16,color='r')
    plot(agc.radeg[velflag],agc.decdeg[velflag],'k.')
    plot(clusterRA,clusterDec,'ro',markersize=16,color='b')
    xlabel('RA (deg)')
    ylabel('Dec (deg)')
    
