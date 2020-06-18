#!/usr/bin/env python
'''

useage

LCSPlotProfiles.py # runs for all clusters

'''



import pyfits
from LCScommon import *
from pylab import *
import os,sys
import mystuff as my
#from LCSReadmasterBaseWithProfileFits import *
from LCSReadmasterBaseNSA import *
import numpy as np
import aplpy
from pyraf import iraf

def transcoordspix2wcs(cimage,incoords):
    outcoords=incoords+'.radec'
    s='rm '+outcoords
    os.system(s)
    iraf.imcoords.wcsctran(image=cimage,input=incoords,output=outcoords,outwcs='world',inwcs='logical',verbose='no')
    in1=open(outcoords,'r')
    data=np.loadtxt(outcoords,unpack=True)#usecols=(0,1))
    xra,yra=data
    return xra,yra


class cluster(baseClusterNSA):

    def __init__(self,clustername):
        baseCluster.__init__(self,clustername)
        self.selectFlag=self.ellipseflag

          
    def plotprofiles(self):
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
        ncol=3
        xticksize=10
        yticksize=10
        ngal=0
        ngaltot=1.*len(sfiles)

        ratio=ngaltot/((nrow*ncol)/8.)
        npage=round(ratio)
        if ratio > npage:
            npage += 1
        npage=ngaltot
        redshift=(self.supervopt[self.selectFlag]-self.clustervel)/self.clustersigma
        member=self.flagmemb[self.selectFlag]
        dr=self.drR200[self.selectFlag]
        mipssnr=self.mipssnr[self.selectFlag]
        agn=self.agn1[self.selectFlag]
        index=arange(len(self.selectFlag))
        spiralIndex=index[self.selectFlag]
        vminsdss=-400
        vmaxsdss=100
        vminsdssapl=-100
        vmaxsdssapl=400
        
        vmin24=-2.1
        vmax24=.5
        sky=self.skySDSS[self.selectFlag]
        r50=self.r50SDSS[self.selectFlag]
        r90=self.r90SDSS[self.selectFlag]
        sky24=self.skyF24[self.selectFlag]
        t=arange(len(self.skySDSS))
        masterindex=t[self.selectFlag]
        subplots_adjust(left=0.05, right=.95,bottom=.1,top=0.9,wspace=0.4,hspace=0.6)

        for i in range(len(simages)):
#        for i in range(2):
            mindex=masterindex[i]
            print self.agcnumber[mindex],simages[i]
            if ~(self.ellipseflag[mindex]&self.spiralFlag[mindex]&(self.r50SDSS[mindex] > 6.)):
                 print 'skipping ',simages[i]
                 print self.ellipseflag[mindex], self.spiralFlag[mindex],self.r50SDSS[mindex]
                 ngal +=1
                 continue
            fig=figure(figsize=(15,9))

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
                if len(t2) > 5:
                    galname24='-'+t2[2]
                else:
                    galname24=t2[1]

                subplot(nrow,ncol,j+3)#sdss profile
                print 'sfiles = ',sfiles[ngal]
                #edat=np.loadtxt(sfiles[ngal],usecols=[1,2,3,21,40,10,12])#6,7,8,9=Ellip,ellip_err,pa,pa_err
                edat=np.loadtxt(sfiles[ngal],usecols=[1,2,3,10,12,21,40])#6,7,8,9=Ellip,ellip_err,pa,pa_err
                #print sfiles[ngal]
                #edat=np.loadtxt(sfiles[ngal],usecols=[2,3,4,22,41])#6,7,8,9=Ellip,ellip_err,pa,pa_err
                x=edat[:,0]
                y=edat[:,1]
                yerr=edat[:,2]
                xcenter=edat[0,3]#x0 in pixels
                ycenter=edat[0,4]#y0 in pixels

                tflux=edat[:,5]
                sarea=edat[:,6]
                s='echo %f %f > junk'%(xcenter,ycenter)
                os.system(s)
                #write function to convert pixels to ra,dec
                myra,mydec=transcoordspix2wcs(simages[ngal],'junk')
                plot(x,y,'b.')
                errorbar(x,y,yerr,fmt=None)



                axhline(sky[ngal],color='k',ls=':')
                axvline(r90[ngal],color='k',ls='--')
                axvline(r50[ngal],color='c',ls='--')
                xlabel('$r \ (arcsec)$')
                ylabel('$I(r)$')
                s='$sky = %5.2f, \ r50 = %5.1f, \ r90 = %5.1f$'%(sky[ngal],r50[ngal],r90[ngal])
                title(s,fontsize=10)
                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)

                fig=gcf()
                subf=aplpy.FITSFigure(simages[ngal],figure=fig,subplot=[.1,.55,.23,.35])
                subf.show_grayscale(vmin=-1*vmaxsdss,vmax=-1*vminsdss)
                axratio=(1-self.SDSSEllipseEllip[mindex])
                phi=self.SDSSEllipsePhi[mindex]+90
                subf.tick_labels.set_font(size='x-small')

                s='$'+self.prefix+': \ '+galname+'$'
                subf.add_label(.5,1.05,s,fontsize=12,relative=True)
                s='$ r-band$'
                subf.add_label(.1,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
                #title(s)
                s='AGN Flag = %i'%(agn[ngal])
                subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='red')

                xlabel(r'$100 X 100 \ arcsec^2$',fontsize=10)


                subf=aplpy.FITSFigure(simages[ngal],figure=fig,subplot=[.38,.55,.23,.35])
                subf.tick_labels.hide()
                subf.show_grayscale(vmin=-1*vmaxsdss,vmax=-1*vminsdss)
                subf.show_ellipses(self.sdssra[mindex],self.sdssdec[mindex],self.r50SDSS[mindex]/3600.,self.r50SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='r')
                subf.show_ellipses(self.sdssra[mindex],self.sdssdec[mindex],self.r90SDSS[mindex]/3600.,self.r90SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='r')
                subf.show_ellipses(myra,mydec,self.r50SDSS[mindex]/3600.,self.r50SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='y')
                subf.show_ellipses(myra,mydec,self.r90SDSS[mindex]/3600.,self.r90SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='y')
                subf.tick_labels.set_font(size='x-small')

                s='$ Spiral \ Flag\ = \ '+str(self.spiralFlag[mindex])+'\ (%5.2f)$'%(self.galzoopcsdebiased[mindex])
                subf.add_label(.5,1.05,s,fontsize=12,relative=True)
                s='$Mr = %5.2f,\ log(M_*)=%5.2f$'%(self.sdssMr[mindex],log10(self.stellarmass[mindex]))
                subf.add_label(.1,.9,s,size='large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')
                #title(s)
                #xlabel(r'$100 X 100 \ arcsec^2$',fontsize=10)


#                ylabel('$24 \ \mu m$',fontsize=14)
#                s1='$\Delta v/\sigma = %5.2f, \ \Delta r/R_{200} = %5.2f$'%(redshift[ngal],dr[ngal])
#                title(s1,fontsize=10)


                #ylabel('$100 \ arcsec$')
##                subplot(nrow,ncol,j+2)#sdss masked image
##                fits=pyfits.open(simages[ngal])
##                im=fits[0].data.copy()
##                fits.close()
##                axis('equal')
##                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax,cmap=cm.Greys)
##                ax=gca()
##                ax.set_yticklabels(([]))
##                ax.set_xticklabels(([]))
##                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                
                #print 'subplot 2'

                #text(.1, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                


                #print 'subplot 4'
#                subplot(nrow,ncol,j+4)#24um image
#                fits=pyfits.open(s24images[ngal])
#                im=fits[0].data.copy()
#                fits.close()
#                axis('equal')
#                imshow(-1.*(im),interpolation='nearest',origin='upper',vmin=vmin24,vmax=vmax24,cmap=cm.Greys)
#                ax=gca()
#                ax.set_yticklabels(([]))
#                ax.set_xticklabels(([]))
#                s='SNR(24) = %5.1f, AGN = %i'%(mipssnr[ngal],agn[ngal])
#                text(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
#                ylabel('$24 \ \mu m$',fontsize=14)
#                s1='$\Delta v/\sigma = %5.2f, \ \Delta r/R_{200} = %5.2f$'%(redshift[ngal],dr[ngal])
#                title(s1,fontsize=10)
#                xlabel(r'$100 X 100 \ arcsec^2$',fontsize=10)

##                subplot(nrow,ncol,j+6)#sdss masked image
##                fits=pyfits.open(s24images[ngal])
##                im=fits[0].data.copy()
##                fits.close()
##                axis('equal')
##                imshow(-1.*(im),interpolation='nearest',origin='upper')#,cmap='binary')#,vmin=myvmin,vmax=myvmax,cmap=cm.Greys)
##                ax=gca()
##                ax.set_yticklabels(([]))
##                ax.set_xticklabels(([]))
##                text(.9, .5, galname, horizontalalignment='center', verticalalignment='center',rotation=90, transform=ax.transAxes)
                #print 'subplot 5'
                subplot(nrow,ncol,j+6)#24um profile
                edat=np.loadtxt(s24files[ngal],usecols=[1,2,3,10,12,21,40])#6,7,8,9=Ellip,ellip_err,pa,pa_err
                try:
                    x=edat[:,0]
                except IndexError:
                    print 'problem with 24um ellipse data for ',galname24
                    ngal +=1
                    break
                y=edat[:,1]
                yerr=edat[:,2]
                tflux=edat[:,5]

                xcenter=edat[0,3]#x0 in pixels
                ycenter=edat[0,4]#y0 in pixels
                print 'xcenter, ycenter = ',xcenter,ycenter
                s='echo %f %f > junk'%(xcenter,ycenter)
                os.system(s)
                #write function to convert pixels to ra,dec
                myra24,mydec24=transcoordspix2wcs(s24images[ngal],'junk')


                plot(x*pscale24,y,'b.')
                errorbar(x*pscale24,y,yerr,fmt=None)

                xlabel('$r \ (arcsec)$')
                ylabel('$I(r)$')
                s='$sky = %5.2f, \ r50 = %5.1f, \ r90 = %5.1f$'%(sky24[ngal],r50[ngal],r90[ngal])
                title(s,fontsize=10)
                axhline(sky24[ngal],color='k',ls=':')
                axvline(r90[ngal],color='k',ls='--')
                axvline(r50[ngal],color='c',ls='--')
                xticks(fontsize=xticksize)
                yticks(fontsize=yticksize)
                xlim(0,50)

                #print 'aplpy 24um subplot'
                fig=gcf()
                subf=aplpy.FITSFigure(s24images[ngal],figure=fig,subplot=[.1,.1,.23,.35])
                subf.show_grayscale(vmin=-1*vmax24,vmax=-1*vmin24)
                s='$ dv/\sigma = %5.2f,\ dr/R_{200} = %5.2f$'%(self.dv[mindex],self.drR200[mindex])
                subf.add_label(.5,1.05,s,fontsize=12,relative=True)

                axratio=1-(self.SDSSEllipseEllip[mindex])
                phi=self.SDSSEllipsePhi[mindex]+90
                subf.tick_labels.set_font(size='x-small')
                if self.apexflag[mindex]:
                    s='SNR(24)=%5.1f, SFR=%5.1f'%(mipssnr[ngal],self.SFR24[mindex])
                else:
                    s='ApexFlag=0, SNR(24)=%5.1f, SFR=%5.1f'%(self.snr24se[mindex],self.SFR24se[mindex])
                subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='red')
                s='$ 24-micron$'
                subf.add_label(.1,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')


                subf=aplpy.FITSFigure(s24images[ngal],figure=fig,subplot=[.38,.1,.23,.35])
                subf.tick_labels.hide()
                subf.show_grayscale(vmin=-1*vmax24,vmax=-1*vmin24)
                subf.show_ellipses(self.sdssra[mindex],self.sdssdec[mindex],self.r50SDSS[mindex]/3600.,self.r50SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='r')
                subf.show_ellipses(self.sdssra[mindex],self.sdssdec[mindex],self.r90SDSS[mindex]/3600.,self.r90SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='r')
                subf.show_ellipses(myra24,mydec24,self.r50SDSS[mindex]/3600.,self.r50SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='y')
                subf.show_ellipses(myra24,mydec24,self.r90SDSS[mindex]/3600.,self.r90SDSS[mindex]*axratio/3600.,angle=phi,edgecolor='y')
                subf.tick_labels.set_font(size='x-small')
                #s='SNR(24) = %5.1f, AGN = %i'%(mipssnr[ngal],agn[ngal])
                #subf.add_label(.9, .5, s, horizontalalignment='center', verticalalignment='center',rotation=90,relative=True,color='red')
                if self.HIflag[mindex]:
                    s='$ log_{10}(M_{HI}) = %5.2f$'%(log10(self.HImass[mindex]))
                else:
                    s='$No \ HI$'
                subf.add_label(.1,.9,s,size='x-large',horizontalalignment='left',verticalalignment='center',relative=True,color='white')


                #print 'subplot 3'
                fig.add_subplot(2*nrow,2*ncol,18,position=[0.79,.3,.1,.15])#sSFR
                plot(self.sSFR50[masterindex[i]],self.sSFR5090[masterindex[i]],'bo')
                errorbar(self.sSFR50[masterindex[i]],self.sSFR5090[masterindex[i]],xerr=self.sSFR50err[masterindex[i]],yerr=self.sSFR5090err[masterindex[i]],fmt=None)
                xlabel('$F_{24}/F_r(R_{50})$',fontsize=12)
                ylabel('$F_{24}/F_r(R_{50-90})$',fontsize=12)
                legend(loc='upper left',numpoints=1,scatterpoints=1,markerscale=1)
                axis([-0.01,0.02,-0.01,0.02])
                xmin,xmax=xlim()
                xl=arange(xmin,xmax,.001)
                plot(xl,xl,'k-')
                xticks(visible=False)
                yticks(visible=False)


                ngal += 1
                if ngal >= ngaltot:
                    figname=self.prefix+'Profiles.sSFR.'+str(galname)+'.png'
                    savefig(figname)
                    break
            figname=self.prefix+'Profiles.sSFR.'+str(galname)+'.png'
            savefig(figname)


mypath=os.getcwd()
if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'


def getSpirals():
    for i in range(1,10):
    #for i in range(1,2):
    #for i in range(2,10):
        if i == 1:
            mkw11=cluster('MKW11')
            cl=mkw11
        if i == 2:
            mkw8=cluster('MKW8')
            cl=mkw8
        if i == 3:
            awm4=cluster('AWM4')
            cl=awm4
        if i == 4:
            ngc=cluster('NGC6107')
            cl = ngc
        if i == 5:
            a2052=cluster('A2052')
            cl = a2052
        if i == 6:
            a2063=cluster('A2063')
            cl = a2063
        if i == 7:
            coma=cluster('Coma')
            cl = coma
        if i == 8:
            herc=cluster('Hercules')
            cl = herc
        if i == 9:
            a1367=cluster('A1367')
            cl = a1367
        cl.plotprofiles()


getSpirals()
#coma.plotprofiles()
#herc.plotprofiles()
#a1367.plotprofiles()
