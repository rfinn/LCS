#!/usr/bin/env python

'''
generate new mastertables using NSA catalog as the parent sample

benefit over previous matching is that tables contain corrected mags, sersic fit, and GALEX fluxes where available.  GALEX is an important addition b/c it allows for more accurate estimage of SFR using 24um + GALEX fluxes.

basic procedure is to find all galaxies within 3 deg of cluster center and with recession velocity within velocity range
    zmin=0.01366 # min z cut, z(coma)-3 sigma
    zmax=0.04333 # max z cut, z(A2052)+3 sigma


'''

import pyfits
from LCScommon import *
from pylab import *
import os, atpy
import mystuff as my
from LCSreadNSA import *
from LCSReadmasterBase import *

mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'


def drawbox(data,style):#feed in center x,y,dx,dy,rotation E of N
    #xcoords of unrotated box, going around CCW
    xl=array([data[0]-0.5*data[2],data[0]+0.5*data[2],data[0]+0.5*data[2],data[0]-0.5*data[2],data[0]-0.5*data[2]],'d')
    yl=array([data[1]-0.5*data[3],data[1]-0.5*data[3],data[1]+0.5*data[3],data[1]+0.5*data[3],data[1]-0.5*data[3] ],'d')

    xl=array([-0.5*data[2],+0.5*data[2],+0.5*data[2],-0.5*data[2],-0.5*data[2]],'d')
    yl=array([-0.5*data[3],-0.5*data[3],+0.5*data[3],+0.5*data[3],-0.5*data[3] ],'d')

    ang=data[4]*pi/180.*-1.#convert rotation to radians
    #rotate coordinates
    xp=cos(ang)*xl-sin(ang)*yl
    yp=sin(ang)*xl+cos(ang)*yl

    #put back on absolute scale
    xp=data[0]+xp
    yp=data[1]+yp
    #draw rotated box
    plot(xp,yp,style)

class cluster:
    def __init__(self,clustername):
#Get current path so program can tell if this is being run on Becky or Rose's computer
	self.prefix=clustername
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]
	self.biweightvel=clustercbi[self.prefix]
	self.biweightscale=clustersbi[self.prefix]
	self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. # in Mpc
        self.r200deg=self.r200*1000./my.DA(self.cz,h)/3600.

        self.cdMpc=self.biweightvel/H0
        self.cdcm=self.cdMpc*3.e24
        self.csigma=self.biweightscale
        self.mcl=my.clusterMass(self.csigma,self.cz,h)
        self.AngDistance=my.DA(self.cz,h)
        mastertable=homedir+'research/LocalClusters/MasterTables/'+clustername+'mastertable.fits'

    def getNSAgals(self):

        distance=sqrt((self.cra-nsa.ra)**2+(self.cdec-nsa.dec)**2)
        dvflag=abs(nsa.redshift*3.e5-self.biweightvel)/self.biweightscale < 3.
        keepflag =(distance < 3.) & dvflag

        figure(figsize=(10,10))

        cir=Circle((self.cra,self.cdec),radius=self.r200deg,color='0.9',ec='0.2',alpha=0.2)
        #gca().add_patch(cir)
        cir=Circle((self.cra,self.cdec),radius=3,color='r',ec='0.2',alpha=0.2)
        #gca().add_patch(cir)
        if self.prefix.find('MKW11') > -1:
            mymkw11.plotpositionson24()
        elif self.prefix.find('A2052') > -1:
            mya2052.plotpositionson24()
        #plot(amkw11.ra,amkw11.dec,'bo',markerfacecolor='None',mec='b',markersize=10)
        plot(nsa.ra[keepflag],nsa.dec[keepflag],'ko',mfc='None',mec='c',markersize=10)
        #drawbox(cluster24Box[self.prefix],'g-')
        dr=.5
        dd=.8
        axis([self.cra-dr,self.cra+dr,self.cdec-dd,self.cdec+dd])
        #figure()
        #hist(nsa.redshift[keepflag])
class agc_cluster(baseCluster):
    def test(self):
        print self.prefix
    def plotpositionson24(self,racenter=0,deccenter=0):
        cir=Circle((self.clusterra-racenter,self.clusterdec-deccenter),radius=self.r200deg,color='0.9',ec='0.2',alpha=0.2)
        gca().add_patch(cir)

        baseflag = self.dvflag & self.On24ImageFlag
        flag=self.sdssflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'ko',markersize=6,label='SDSS')
        flag = self.HIflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'bo', markerfacecolor='None',markeredgecolor='b',markersize=8,label='HI')
        flag=self.spiralFlag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'g^',markerfacecolor='None',markeredgecolor='g',markersize=10,label='Spiral')
        flag=self.agnflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'c*',mec='0.5',label='AGN',markersize=4)
        flag=self.apexflag & ~self.agnflag & baseflag
        plot(self.ra[flag]-racenter,self.dec[flag]-deccenter,'ro', markerfacecolor='r',markeredgecolor='r',markersize=3,label='_nolabel_')
        #scatter(self.ra[flag]-racenter,self.dec[flag]-deccenter,s=50*self.SFR24[flag],color='r',alpha=0.5,label='24um')
            #plot(ra[flag],dec[flag],'k.')
            #plot(array([self.clusterra])-racenter,array([self.clusterdec])-deccenter,'kx',markersize=15,lw=8)#mark cluster center with a red x



        #legend(loc='upper left',numpoints=1,scatterpoints=1)


        title(self.clustername,fontsize=12)
            #axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #axis('equal')
        axis('equal')

        #xmin,xmax=xlim()
        #print xmin,xmax
        #x1=round(xmin)
        #x2=round(xmax)
        #xt=arange(x1,x2+1)
        #xticks(xt)
        #ymin,ymax=ylim()
        #yticks(arange(round(ymin),ymax,'i'),fontsize=10)

        t=cluster24Box[self.clustername]
        drawbox([t[0]-racenter,t[1]-deccenter,t[2],t[3],t[4]],'g-')


    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
        #s=self.clustername+'.png'

        #savefig(s)






#nsa=NSA()
mkw11=cluster('MKW11')
mymkw11=agc_cluster('MKW11')
a2052=cluster('A2052')
mya2052=agc_cluster('A2052')
