#!/usr/bin/env python

'''
useage:  LCSfitprofile.py

read in ellipse output

plot flux vs radius

fit exponential, I = I_0 exp(-r/R_0), solving for R_0

plot data and fit
'''

import glob
#from pyraf import iraf
from LCScommon import *
from pylab import *
import pyfits
#from scipy import stats
#import os

ellipseOutputPath='/home/rfinn/research/LocalClusters/EllipseTables/'

class Lcluster:
    def __init__(self,clustername):
        self.prefix=clustername
        self.cra=clusterRA[self.prefix]
        self.cdec=clusterDec[self.prefix]
        self.cz=clusterz[self.prefix]
	self.biweightvel=clustercbi[self.prefix]
	self.biweightscale=clustersbi[self.prefix]
	self.r200=2.02*(self.biweightscale)/1000./sqrt(OmegaL+OmegaM*(1.+self.cz)**3)*H0/70. # in Mpc

    #def getSpirals(self):
    def checkplots(self):
        #get list from LCSreadmaster.py
        inf1='/home/rfinn/research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.sdss.dat'
        infile1=open(inf1,'r')
        sfiles=[]
        for line in infile1:
            t=line.rstrip()
            sfiles.append(t)
            #print t
        infile1.close()

        inf1='/home/rfinn/research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.24.dat'
        infile1=open(inf1,'r')
        s24files=[]
        for line in infile1:
            t=line.rstrip()
            s24files.append(t)
            #print t
        infile1.close()

        inf1='/home/rfinn/research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.sdss.dat'
        infile1=open(inf1,'r')
        simages=[]
        for line in infile1:
            t=line.rstrip()
            simages.append(t)
            #print t
        infile1.close()

        inf1='/home/rfinn/research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.24.dat'
        infile1=open(inf1,'r')
        s24images=[]
        for line in infile1:
            t=line.rstrip()
            s24images.append(t)
            #print t
        infile1.close()

        pscale24=2.45#arcsec per pixel

        pscalesdss=1.#arcsec per pixel
        
        nrow=4
        ncol=4
        #for i in range(0,len(sfiles),nrow):
        ngal=0
        ngaltot=1.*len(sfiles)

        ratio=ngaltot/((nrow*ncol)/8.)
        npage=round(ratio)
        if ratio > npage:
            npage += 1
        print "Ngal = ",ngaltot
        print "Npage = ",npage
        for i in range(int(npage)):
            figure(figsize=(15,10))
            subplots_adjust(left=0.1, right=.95,bottom=.15,top=0.9,wspace=0.15,hspace=0.1)
            clf()
            for j in range(0,ncol*nrow,8):
                t=sfiles[ngal]
                t1=t.split('/')
                t2=t1[len(t1)-1].split('-')
                if len(t2) > 4:
                    galname='-'+t2[2]
                else:
                    galname=t2[1]

                t=s24files[ngal]
                t1=t.split('/')
                t2=t1[len(t1)-1].split('-')
                if len(t2) > 4:
                    galname2='-'+t2[2]
                else:
                    galname2=t2[1]


                subplot(nrow,ncol,j+1)#sdss image
                fits=pyfits.open(simages[ngal])
                im=fits[0].data.copy()
                
                fits.close()
                axis('equal')
                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax)                           
                ax=gca()
                ax.set_yticklabels(([]))
                ax.set_xticklabels(([]))
                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                
                subplot(nrow,ncol,j+2)#sdss masked image
                fits=pyfits.open(simages[ngal])
                im=fits[0].data.copy()
                fits.close()
                axis('equal')
                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax)                           
                ax=gca()
                ax.set_yticklabels(([]))
                ax.set_xticklabels(([]))
                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                
                subplot(nrow,ncol,j+3)#sdss profile
                edat=load(sfiles[ngal],usecols=[1,2,3])
                x=edat[:,0]
                y=edat[:,1]
                yerr=edat[:,2]
                plot(x,y,'b.')
                errorbar(x,y,yerr,fmt=None)

                subplot(nrow,ncol,j+4)#sdss ln profile with fit
                xfit=(x[y>5])
                yfit=log(y[y>5])
                m,b=polyfit(xfit,yfit,1)
                plot(xfit,yfit-b,'r^')
                xlabel('r (arcsec)')
                ylabel(r'$ln(flux)-ln(I_0)$')
                #gradient, intercept, r_value, p_value, std_err = stats.linregress(xfit,yfit)

                plot(xfit,m*xfit,'g')
                axvline(-1./m,color='k',ls='--')
                axhline(-1,color='k',ls='--')
                
                subplot(nrow,ncol,j+5)#sdss image
                fits=pyfits.open(s24images[ngal])
                im=fits[0].data.copy()
                
                fits.close()
                axis('equal')
                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax)                           
                ax=gca()
                ax.set_yticklabels(([]))
                ax.set_xticklabels(([]))
                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                
                subplot(nrow,ncol,j+6)#sdss masked image
                fits=pyfits.open(s24images[ngal])
                im=fits[0].data.copy()
                fits.close()
                axis('equal')
                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax)                           
                ax=gca()
                ax.set_yticklabels(([]))
                ax.set_xticklabels(([]))
                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                
                subplot(nrow,ncol,j+7)#sdss profile
                edat=load(s24files[ngal],usecols=[1,2,3])
                x=edat[:,0]
                y=edat[:,1]
                yerr=edat[:,2]
                plot(x*pscale24,y,'b.')
                errorbar(x*pscale24,y,yerr,fmt=None)

                subplot(nrow,ncol,j+8)#sdss ln profile with fit
                xfit=(x[y>.04])
                yfit=log(y[y>.04])

                if len(xfit) > 1:
                    m,b=polyfit(xfit*pscale24,yfit,1)
                    plot(xfit*pscale24,m*xfit*pscale24,'g')
                    plot(xfit*pscale24,yfit-b,'r^')
                else:
                    plot(xfit*pscale24,yfit,'r^')
                xlabel('r (arcsec)')
                ylabel(r'$ln(flux)-ln(I_0)$')
                #gradient, intercept, r_value, p_value, std_err = stats.linregress(xfit,yfit)


                axvline(-1./m,color='k',ls='--')
                axhline(-1,color='k',ls='--')
                

                ngal += 1
                if ngal >= ngaltot:
                    figname=self.prefix+'Profiles'+str(i)+'.png'
                    savefig(figname)
                    break
            figname=self.prefix+'Profiles'+str(i)+'.png'
            savefig(figname)
mkw11=Lcluster('MKW11')
mkw11.checkplots()
