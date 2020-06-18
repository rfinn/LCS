#!/usr/bin/env python
'''

useage

LCSFitProfiles.py # runs for all clusters
LCSFitProfiles.py 0 # runs for cluster 0.  This can help speed up analysis by running 4 jobs simultaneously

'''



import pyfits
from LCScommon import *
from pylab import *
import os,sys
import mystuff as my
from LCSReadmasterBase import *
import numpy as np
def calcError(ellip,x,yerr,indexrange):
    error=pi*(1-ellip)*sqrt((yerr[0]*x[0]**2)**2+sum((yerr[indexrange]*(x[indexrange]**2-x[indexrange-1]**2))**2))
    return error
def calcC90(x,y,yerr,fluxencl,ellip):#radius("),intensity,error,
    #find average using last 3 pts, which should approximate sky
    sky=mean(y[len(y)-3:len(y)])
    #subtract sky from y
    sy=y-sky
    #multiply y by r**2 to account for increasing area
    toty=y*(x**2)
    #sum r=0-36arcsec to get total flux
    totygood=toty[0:len(y)-3]
    rgood=x[0:len(y)-3]
    totflux=sum(totygood)
    #start summing from 39arcsec inward, until it reaches 10% of total flux
    sum10=0
    tenpercent=.1*totflux
    thirty=.3*totflux
    fifty=.5*totflux
    ninety=.9*totflux
    for i in range(len(totygood)):
        index=len(totygood)-1-i
        sum10 += totygood[index]
        #print sky,sum10,tenpercent,totflux
        if sum10 > tenpercent:
            r90=rgood[index]
            break

    #calculate r30
    sum30=0
    for i in range(len(totygood)):
        sum30 += totygood[i]
        if sum30 > thirty:
            r30=rgood[i]
            break
    #calculate r50
    sum50=0
    for i in range(len(totygood)):
        sum50 += totygood[i]
        if sum50 > fifty:
            r50=rgood[i]
            break


    #find max of array
    maxEncFlux=max(fluxencl)
    indexMax=where((fluxencl == maxEncFlux))
    #break index out of array and into a plain integer
    indexMax=indexMax[0]
    #use index of max of array and move inward until value is .9 max
    transitionFlux=0.9*maxEncFlux

    try:
        for i in range(indexMax):
            index=indexMax-i
            if fluxencl[index] < transitionFlux:
                r90FromEncFlux=x[index]
                indexrange=arange(1,index+1)
                F90err=calcError(ellip,x,yerr,indexrange)

                break
    except:
        print "Warning: Could not calculate r90FromEncFlux!"
        F90err=0
    #calculate r30FromEncFlux
    transitionFlux=0.3*maxEncFlux
    try:
        for i in range(indexMax):
            index=i
            if fluxencl[index] > transitionFlux:
                r30FromEncFlux=x[index]
                indexrange=arange(1,index+1)
                F30err=calcError(ellip,x,yerr,indexrange)
                break
    except:
        print "Warning: Could not calculate r30FromEncFlux!"
        F30err=0
    #return radius at C90 (the radius that encloses 90% of the light) and sky

    #calculate r50FromEncFlux
    transitionFlux=0.5*maxEncFlux
    try:
        for i in range(indexMax):
            index=i
            if fluxencl[index] > transitionFlux:
                r50FromEncFlux=x[index]
                indexrange=arange(1,index+1)
                F50err=calcError(ellip,x,yerr,indexrange)
                break
    except:
        print "Warning: Could not calculate r50FromEncFlux!"
        F50err=0
    #return radius at C90 (the radius that encloses 90% of the light) and sky


    try:
        a=r90
    except NameError:
        print "Warning: Could not find R90"
        r90=0
        
    try:
        a=r90FromEncFlux
    except NameError:
        print "Warning: Could not find R90FromEncFlux"
        r90FromEncFlux=0
        F90err=0
    try:
        a=r30
    except NameError:
        print "Warning: Could not find R30"
        r30=0
        
    try:
        a=r30FromEncFlux
    except NameError:
        print "Warning: Could not find R30FromEncFlux"
        r30FromEncFlux=0
        F30err=0
    try:
        a=r50
    except NameError:
        print "Warning: Could not find R50"
        r50=0

        
    try:
        a=r50FromEncFlux
    except NameError:
        print "Warning: Could not find R50FromEncFlux"
        r50FromEncFlux=0
        F50err=0

    return r90,sky,r90FromEncFlux,maxEncFlux,r30,r30FromEncFlux,r50,r50FromEncFlux ,F30err,F50err,F90err

def calcMIPSFromSDSS(x,y,yerr,fluxencl,s30,s50,s90,ellip):#radius("),intensity,error,enclosed flux
    #calculate MIPS flux enclosed w/in sdss values of R30, R50, R90 using enclosed flux
    #find average using last 3 pts, which should approximate sky
    sky=mean(y[len(y)-3:len(y)])
    #subtract sky from y
    sy=y-sky

    #measure F30
    for i in range(len(x)):
        if x[i]>s30:
            F30=fluxencl[i]
            indexrange=arange(1,i+1)
            F30err=calcError(ellip,x,yerr,indexrange)
            break
    #measure F50
    for i in range(len(x)):
        if x[i]>s50:
            F50=fluxencl[i]
            indexrange=arange(1,i+1)
            F50err=calcError(ellip,x,yerr,indexrange)
            break
    #measure F90
    for i in range(len(x)):
        if x[i]>s90:
            F90=fluxencl[i]
            indexrange=arange(1,i+1)
            F90err=calcError(ellip,x,yerr,indexrange) 

            break

    try:
        a=F30
    except NameError:
        print "Warning: Could not find MIPS R30 From SDSS"
        print "s30, s50, s90 = ",s30,s50,s90
        F30=-99
        F30err=-99
    try:
        a=F50
    except NameError:
        print "Warning: Could not find MIPS R50 From SDSS"
        print "s30, s50, s90 = ",s30,s50,s90
        print "x = ",x
        F50=-99
        F50err=-99
    try:
        a=F90
    except NameError:
        print "Warning: Could not find MIPS R90 From SDSS"
        print "s30, s50, s90 = ",s30,s50,s90
        print "x = ",x
        F90=-99
        F90err=-99

        


    return F30, F50, F90,F30err,F50err,F90err

class cluster(baseCluster):

    def __init__(self,clustername):
        baseCluster.__init__(self,clustername)
        self.selectFlag=self.ellipseflag

    def getFilesForProfileFitting(self):
        outfile1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.24.dat'
        outfile2=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.sdss.dat'
        outfile3=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.24.dat'
        outfile4=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.sdss.dat'
        #agcSpiral=self.agcnumber[self.spiralFlag]
        #changing this to include all galaxies that we successfully ran ellipse on
        agcSpiral=self.agcnumber[self.selectFlag]
        out1=open(outfile1,'w')
        out2=open(outfile2,'w')
        out3=open(outfile3,'w')
        out4=open(outfile4,'w')
        cutoutpath=homedir+'research/LocalClusters/cutouts/'+self.prefix+'/'
        for i in range(len(agcSpiral)):
            outim=cutoutpath+self.prefix+'-'+str(agcSpiral[i])+'-cutout-sdss.fits \n'
	    outim24=cutoutpath+self.prefix+'-'+str(agcSpiral[i])+'-cutout-24-rot.fits \n'
            out1.write(outim24)
            out2.write(outim)
            outfile=homedir+'research/LocalClusters/EllipseTables/'+self.prefix+'/'+self.prefix+'-'+str(agcSpiral[i])+'-cutout-sdss.dat \n'
            outfile24=homedir+'research/LocalClusters/Ellipse24UsingSDSS/'+self.prefix+'/'+self.prefix+'-'+str(agcSpiral[i])+'.24UsingSDSS.dat \n'
            out3.write(outfile24)
            out4.write(outfile)
        out1.close()
        out2.close()
        out3.close()
        out4.close()
                 


    def fitprofiles(self):
        #get list from LCSreadmaster.py
        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.sdss.dat'
        infile1=open(inf1,'r')
        sfiles=[]
        for line in infile1:
            t=line.rstrip()
            sfiles.append(t)
            #print t
        infile1.close()

        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.24.dat'
        infile1=open(inf1,'r')
        s24files=[]
        for line in infile1:
            t=line.rstrip()
            s24files.append(t)
            #print t
        infile1.close()

        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.sdss.dat'
        infile1=open(inf1,'r')
        simages=[]
        for line in infile1:
            t=line.rstrip()
            simages.append(t)
            #print t
        infile1.close()

        inf1=homedir+'research/LocalClusters/ProfileFitting/'+self.prefix+'Spirals.Images.24.dat'
        infile1=open(inf1,'r')
        s24images=[]
        for line in infile1:
            t=line.rstrip()
            s24images.append(t)
            #print t
        infile1.close()

        pscale24=2.45#arcsec per pixel

        pscalesdss=1.#arcsec per pixel
        
        nrow=2
        ncol=4
        xticksize=10
        yticksize=10
        ngal=0
        ngaltot=1.*len(sfiles)

        ratio=ngaltot/((nrow*ncol)/8.)
        npage=round(ratio)
        if ratio > npage:
            npage += 1
        npage=ngaltot
        #print "Ngal = ",ngaltot
        #print "Npage = ",npage
        redshift=(self.supervopt[self.selectFlag]-self.clustervel)/self.clustersigma
        member=self.flagmemb[self.selectFlag]
        dr=self.drR200[self.selectFlag]
        mipssnr=self.mipssnr[self.selectFlag]
        agn=self.agn1[self.selectFlag]
        index=arange(len(self.selectFlag))
        spiralIndex=index[self.selectFlag]
        vminsdss=-400
        vmaxsdss=100
        vmin24=-2.1
        vmax24=.5
        self.r0SDSS=zeros(len(spiralIndex),'f')#scale length from exponential fit
        self.r30SDSS=zeros(len(spiralIndex),'f')
        self.r50SDSS=zeros(len(spiralIndex),'f')
        self.r90SDSS=zeros(len(spiralIndex),'f')
        self.skySDSS=zeros(len(spiralIndex),'f')
        self.r30EncFluxSDSS=zeros(len(spiralIndex),'f')
        self.r50EncFluxSDSS=zeros(len(spiralIndex),'f')
        self.r90EncFluxSDSS=zeros(len(spiralIndex),'f')
        self.MaxEncFluxSDSS=zeros(len(spiralIndex),'f')
        self.SDSSF30=zeros(len(spiralIndex),'f')
        self.SDSSF50=zeros(len(spiralIndex),'f')
        self.SDSSF90=zeros(len(spiralIndex),'f')
        self.SDSSF30err=zeros(len(spiralIndex),'f')
        self.SDSSF50err=zeros(len(spiralIndex),'f')
        self.SDSSF90err=zeros(len(spiralIndex),'f')

        self.SDSSEllipsePhi=zeros(len(spiralIndex),'f')
        self.SDSSEllipseEllip=zeros(len(spiralIndex),'f')
        self.SDSSEllipsePhiErr=zeros(len(spiralIndex),'f')
        self.SDSSEllipseEllipErr=zeros(len(spiralIndex),'f')
        #same array for 24um

        
        self.r0F24=zeros(len(spiralIndex),'f')#scale length from exponential fit
        self.r30F24=zeros(len(spiralIndex),'f')
        self.r50F24=zeros(len(spiralIndex),'f')
        self.r90F24=zeros(len(spiralIndex),'f')
        self.skyF24=zeros(len(spiralIndex),'f')
        self.r30EncFluxF24=zeros(len(spiralIndex),'f')
        self.r50EncFluxF24=zeros(len(spiralIndex),'f')
        self.r90EncFluxF24=zeros(len(spiralIndex),'f')
        self.MaxEncFluxF24=zeros(len(spiralIndex),'f')
        self.mipsF30=zeros(len(spiralIndex),'f')
        self.mipsF50=zeros(len(spiralIndex),'f')
        self.mipsF90=zeros(len(spiralIndex),'f')
        self.mipsF30err=zeros(len(spiralIndex),'f')
        self.mipsF50err=zeros(len(spiralIndex),'f')
        self.mipsF90err=zeros(len(spiralIndex),'f')

        for i in range(int(npage)):
        #for i in range(1):

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
                if len(t2) > 5:
                    galname24='-'+t2[2]
                else:
                    galname24=t2[1]

                print 'ngal = ',ngal, galname, galname24
                
                edat=np.loadtxt(sfiles[ngal],usecols=[1,2,3,21,40])#6,7,8,9=Ellip,ellip_err,pa,pa_err
                x=edat[:,0]
                y=edat[:,1]
                yerr=edat[:,2]
                tflux=edat[:,3]
                sarea=edat[:,4]
                f99=open(sfiles[ngal],'r')
                for mytext in f99:
                    words=mytext.split()
                    self.SDSSEllipseEllip[ngal]=float(words[6])
                    self.SDSSEllipseEllipErr[ngal]=float(words[7])
                    self.SDSSEllipsePhi[ngal]=float(words[8])
                    self.SDSSEllipsePhiErr[ngal]=float(words[9])
                    
                    break
                f99.close()
                
                r90,sky,r90EncFlux,MaxEncFlux,r30,r30EncFlux,r50,r50EncFlux,F30err,F50err,F90err=calcC90(x,y,yerr,tflux,self.SDSSEllipseEllip[ngal])

                self.r30SDSS[ngal]=r30
                self.r50SDSS[ngal]=r50
                self.r90SDSS[ngal]=r90
                self.skySDSS[ngal]=sky
                self.r30EncFluxSDSS[ngal]=r30EncFlux
                self.r50EncFluxSDSS[ngal]=r50EncFlux
                self.r90EncFluxSDSS[ngal]=r90EncFlux
                self.MaxEncFluxSDSS[ngal]=MaxEncFlux
                self.SDSSF30[ngal]=.3*MaxEncFlux
                self.SDSSF50[ngal]=.5*MaxEncFlux
                self.SDSSF90[ngal]=.9*MaxEncFlux
                self.SDSSF30err[ngal]=F30err
                self.SDSSF50err[ngal]=F50err
                self.SDSSF90err[ngal]=F90err

                xfit=(x[y>5])
                yfit=log(y[y>5])
                if len(xfit) > 1:
                    m,b=polyfit(xfit,yfit,1)

                    self.r0SDSS[ngal]=-1./m

                else:
                    plot(xfit,yfit,'r^')
                    self.r0SDSS[ngal]=-99

                edat=np.loadtxt(s24files[ngal],usecols=[1,2,3,21])
                try:
                    x=edat[:,0]
                except IndexError:
                    self.spiralFlag[ngal]=0
                    print 'problem with 24um ellipse data for ',galname24
                    ngal +=1
                    break
                y=edat[:,1]
                yerr=edat[:,2]
                tflux=edat[:,3]
                r90,sky,r90EncFlux,MaxEncFlux,r30,r30EncFlux,r50,f50EncFlux,F30err,F50err,F90err=calcC90(x*pscale24,y,yerr,tflux,self.SDSSEllipseEllip[ngal])
                self.r30F24[ngal]=r30
                self.r50F24[ngal]=r50
                self.r90F24[ngal]=r90
                self.skyF24[ngal]=sky
                self.r30EncFluxF24[ngal]=r30EncFlux
                self.r50EncFluxF24[ngal]=r50EncFlux
                self.r90EncFluxF24[ngal]=r90EncFlux
                self.MaxEncFluxF24[ngal]=MaxEncFlux

                F30,F50,F90,F30err,F50err,F90err=calcMIPSFromSDSS(x*pscale24,y,yerr,tflux,self.r30SDSS[ngal],self.r50SDSS[ngal],self.r90SDSS[ngal],self.SDSSEllipseEllip[ngal])
                self.mipsF30[ngal]=F30
                self.mipsF50[ngal]=F50
                self.mipsF90[ngal]=F90
                self.mipsF30err[ngal]=F30err
                self.mipsF50err[ngal]=F50err
                self.mipsF90err[ngal]=F90err


                subplot(nrow,ncol,j+8)#sdss ln profile with fit
                xfit=(x[y>.05])
                yfit=log(y[y>.05])
                if len(xfit) > 1:
                    m,b=polyfit(xfit*pscale24,yfit,1)
                    self.r0F24[ngal]=-1./m
                                    
                else:

                    self.r0F24[ngal]=-99

                ngal += 1
    def writeMasterTable(self):

        s='/home/rfinn/research/LocalClusters/MasterTables/'+self.prefix+'mastertable.fits'
        mastertab=pyfits.open(s)
        tbhdu=mastertab[1]
        cdefs=tbhdu.get_coldefs()
        tbdata=mastertab[1].data
        mastertab.close()


        index=arange(len(self.selectFlag))
        spiralIndex=index[self.selectFlag]

        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r0SDSS
        newcol0=pyfits.Column(name='SPIRALFLAG',format='L',array=self.spiralFlag)
        newcol1=pyfits.Column(name='R0SDSS',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r30SDSS
        newcol2=pyfits.Column(name='R30SDSS',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r50SDSS
        newcol2a=pyfits.Column(name='R50SDSS',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r90SDSS
        newcol3=pyfits.Column(name='R90SDSS',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.skySDSS
        newcol4=pyfits.Column(name='skySDSS',format='E',unit='ADU/s',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r30EncFluxSDSS
        newcol5=pyfits.Column(name='R30EncFluxSDSS',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r50EncFluxSDSS
        newcol5a=pyfits.Column(name='R50EncFluxSDSS',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r90EncFluxSDSS
        newcol6=pyfits.Column(name='R90EncFluxSDSS',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.MaxEncFluxSDSS
        newcol7=pyfits.Column(name='MaxEncFluxSDSS',format='E',unit='ADU/s',array=newarray)


        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r0F24
        newcol8=pyfits.Column(name='R0F24',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r30F24
        newcol9=pyfits.Column(name='R30F24',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r50F24
        newcol9a=pyfits.Column(name='R50F24',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r90F24
        newcol10=pyfits.Column(name='R90F24',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.skyF24
        newcol11=pyfits.Column(name='skyF24',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r30EncFluxF24
        newcol12=pyfits.Column(name='R30EncFluxF24',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r50EncFluxF24
        newcol12a=pyfits.Column(name='R50EncFluxF24',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.r90EncFluxF24
        newcol13=pyfits.Column(name='R90EncFluxF24',format='E',unit='arcsec',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.MaxEncFluxF24
        newcol14=pyfits.Column(name='MaxEncFluxF24',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSF30
        newcol15=pyfits.Column(name='SDSSF30',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSF50
        newcol16=pyfits.Column(name='SDSSF50',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSF90
        newcol17=pyfits.Column(name='SDSSF90',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.mipsF30
        newcol18=pyfits.Column(name='MIPSF30',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.mipsF50
        newcol19=pyfits.Column(name='MIPSF50',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.mipsF90
        newcol20=pyfits.Column(name='MIPSF90',format='E',unit='microJy',array=newarray)

        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSEllipsePhi
        newcol21=pyfits.Column(name='SDSSEllipsePhi',format='E',unit='deg',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSEllipsePhiErr
        newcol22=pyfits.Column(name='SDSSEllipsePhiErr',format='E',unit='deg',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSEllipseEllip
        newcol23=pyfits.Column(name='SDSSEllipseEllip',format='E',unit='',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSEllipseEllipErr
        newcol24=pyfits.Column(name='SDSSEllipseEllipErr',format='E',unit='',array=newarray)

        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSF30err
        newcol25=pyfits.Column(name='SDSSF30err',format='E',unit='',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSF50err
        newcol26=pyfits.Column(name='SDSSF50err',format='E',unit='',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.SDSSF90err
        newcol27=pyfits.Column(name='SDSSF90err',format='E',unit='',array=newarray)

        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.mipsF30err
        newcol28=pyfits.Column(name='mipsF30err',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.mipsF50err
        newcol29=pyfits.Column(name='mipsF50err',format='E',unit='microJy',array=newarray)
        newarray=zeros(len(self.selectFlag),'f')
        newarray[spiralIndex]=self.mipsF90err
        newcol30=pyfits.Column(name='mipsF90err',format='E',unit='microJy',array=newarray)



        cdefs0=cdefs.add_col(newcol0)
        cdefs1=cdefs0.add_col(newcol1)
        cdefs2=cdefs1.add_col(newcol2)
        cdefs2a=cdefs2.add_col(newcol2a)
        cdefs3=cdefs2a.add_col(newcol3)
        cdefs4=cdefs3.add_col(newcol4)
        cdefs5=cdefs4.add_col(newcol5)
        cdefs5a=cdefs5.add_col(newcol5a)
        cdefs6=cdefs5a.add_col(newcol6)
        cdefs7=cdefs6.add_col(newcol7)
        cdefs8=cdefs7.add_col(newcol8)
        cdefs9=cdefs8.add_col(newcol9)
        cdefs9a=cdefs9.add_col(newcol9a)
        cdefs10=cdefs9a.add_col(newcol10)
        cdefs11=cdefs10.add_col(newcol11)
        cdefs12=cdefs11.add_col(newcol12)
        cdefs12a=cdefs12.add_col(newcol12a)
        cdefs13=cdefs12a.add_col(newcol13)
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
        newmaster=pyfits.new_table(cdefs30)
        s='/home/rfinn/research/LocalClusters/MasterTables/'+self.prefix+'mastertable.WithProfileFits.fits'
        newmaster.writeto(s,clobber='yes')



mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'


def checkmorphall():
    for i in range(1,10):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        cl.checkmorph()

def getSpirals():

    mkw11=cluster('MKW11')
    mkw8=cluster('MKW8')
    awm4=cluster('AWM4')
    a2052=cluster('A2052')
    a2063=cluster('A2063')
    ngc=cluster('NGC6107')
    coma=cluster('Coma')
    herc=cluster('Hercules')
    a1367=cluster('A1367')
    
    for i in range(1,10):
    #for i in range(1,3):
        if i == 1:
            cl=mkw11
        if i == 2:
            cl=mkw8
        if i == 3:
            cl=awm4
        if i == 4:
            cl = ngc
        if i == 5:
            cl = a2052
        if i == 6:
            cl = a2063
        if i == 7:
            cl = coma
        if i == 8:
            cl = herc
        if i == 9:
            cl = a1367
        cl.getFilesForProfileFitting()
        cl.fitprofiles()
        cl.writeMasterTable()

def getSpiralsParallel(ncluster):
    i = ncluster
    if i == 1:
        mkw11=cluster('MKW11')
        cl=mkw11
    elif i == 2:
        mkw8=cluster('MKW8')
        cl=mkw8
    elif i == 3:
        awm4=cluster('AWM4')
        cl=awm4
    elif i == 4:
        ngc=cluster('NGC6107')
        cl = ngc
    elif i == 5:
        a2052=cluster('A2052')
        cl = a2052
    elif i == 6:
        a2063=cluster('A2063')
        cl = a2063
    elif i == 7:
        coma=cluster('Coma')
        cl = coma
    elif i == 8:
        herc=cluster('Hercules')
        cl = herc
    elif i == 9:
        a1367=cluster('A1367')
        cl = a1367
    cl.getFilesForProfileFitting()
    cl.fitprofiles()
    cl.writeMasterTable()

if len(sys.argv) > 1:
    getSpiralsParallel(int(sys.argv[1]))
else:
    getSpirals()
print sys.argv, len(sys.argv)

