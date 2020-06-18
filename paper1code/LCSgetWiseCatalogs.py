#!/usr/bin/env -python
import urllib2, urllib
import os
import atpy
from LCSReadmasterBase import *

''' 
Goal is to get the WISE catalogs for the LCS clusters (3 deg search radius, if possible)

following example at

http://python4astronomers.github.com/vo/webmodules.html

url = "http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
p = {}
p['spatial'] = "Cone"
p['objstr'] = "MKW 11"
p['outfmt'] = 1    # IPAC table format
p['catalog'] = 'wise_prelim_p3as_psd'
p['radius'] = 300

query = urllib.urlencode(p)
get_url = url + "?" + query
handler = urllib2.urlopen(get_url)
raw = handler.read()
print raw[0:255]

with open('/Users/rfinn/ic348_wise.tbl', 'wb') as f:
    f.write(raw)
'''
mypath=os.getcwd()

if mypath.find('Users') > -1:
    print "Running on Rose's mac pro"
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print "Running on coma"
    homedir='/home/rfinn/'

    
class cluster(baseCluster):
    
    def __init__(self,clustername):
        baseCluster.__init__(self,clustername)

    def printradec(self):#print file containing RA and Dec of all objects on 24micron Image
        outfile=homedir+'research/LocalClusters/WISE/'+self.prefix+'RADec.txt'
        out1=open(outfile,'w')
        #agcname=self.agcnumber[self.On24ImageFlag]
        #ra=self.ra[self.On24ImageFlag]
        #dec=self.dec[self.On24ImageFlag]
        agcname=self.agcnumber
        ra=self.ra
        dec=self.dec
        out1.write('\\fexlen=T\n')
        out1.write('| object  | ra           | dec         | cntr            |\n')
        #            | object | ra           | dec         | cntr            |
        out1.write('|int      |double        |double       |int              |\n')


        for i in range(len(ra)):
            #s='  %i8   %12.8f  %12.8f   %i \n'%(int(agcname[i]),ra[i],dec[i],i+1)
            print ra[i],dec[i]
            s='  '+str(agcname[i]).rjust(8)+'   '+repr(ra[i]).rjust(12)+' '+repr(dec[i]).rjust(12)+'    '+str(i).rjust(3)+'\n'
                                                        
            out1.write(s)
        out1.close()
    def readWiseCatalog(self):#read output from multi-object search
        infile=homedir+'research/LocalClusters/WISE/'+self.prefix+'wise.fits'
        wtable=atpy.Table(infile)
        self.wtable=wtable
        #initialize arrays (set length to the sames as mastertable)
        self.wiseagcnumber=wtable.object_01
        self.w1mpro=wtable.w1mpro
        self.w1sigmpro=wtable.w1sigmpro
        self.w1snr=wtable.w1snr
        self.w2mpro=wtable.w2mpro
        self.w2sigmpro=wtable.w2sigmpro
        self.w2snr=wtable.w2snr
        self.w3mpro=wtable.w3mpro
        self.w3sigmpro=wtable.w3sigmpro
        self.w3snr=wtable.w3snr
        self.w4mpro=wtable.w4mpro
        self.w4sigmpro=wtable.w4sigmpro
        self.w4snr=wtable.w4snr
        w1=wtable.w1mpro
        w2=wtable.w2mpro
        w3=wtable.w3mpro
        w4=wtable.w4mpro
        w1snr=wtable.w1snr
        w2snr=wtable.w2snr
        w3snr=wtable.w3snr
        w4snr=wtable.w4snr
        w1err=w1/w1snr
        w2err=w2/w2snr
        w3err=w3/w3snr
        w4err=w4/w4snr

        self.w12=w1-w2
        self.w23=w2-w3
        self.w34=w3-w4
        #zeropoint fluxes from http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2ab
        self.fw1=309.54*10.**(-1.*wtable.w1mpro/2.5)
        self.fw2=171.787*10.**(-1.*wtable.w2mpro/2.5)
        self.fw3=31.674*10.**(-1.*wtable.w3mpro/2.5)
        self.fw4=8.363*10.**(-1.*wtable.w4mpro/2.5)
        self.fw4=8.363*10.**(-1.*wtable.w4mag_4/2.5)

        self.fw1err=self.fw1/w1snr
        self.fw2err=self.fw2/w2snr
        self.fw3err=self.fw3/w3snr
        self.fw4err=self.fw4/w4snr

        matchindex=[]
        for i in range(len(wtable.object_01)):
            #print wtable.object_u[i], self.agcdict[wtable.object_u[i]]
            matchindex.append(self.agcdict[wtable.object_01[i]])
        matchindex=array(matchindex,'i')
        self.matchindex=matchindex
        figure()
        mipsflux=self.mipsflux[matchindex]
        mipsfluxerr=self.mipsfluxerr[matchindex]
        mipsflag=self.apexflag[matchindex]
        flag=self.apexflag[matchindex]
        y=self.fw4[flag]*1.e6
        wyerr=self.fw4err[flag]*1.e6
        #y=wtable.w4mag[flag]
        #wyerr=wtable.w4sigm[flag]
        plot((mipsflux[flag]),y,'ko',label='_nolegend_')
        errorbar((mipsflux[flag]),y,yerr=wyerr,xerr=mipsfluxerr[flag],fmt=None,label='_nolegend_')
        self.diff24wise=self.mipsflux[matchindex]-self.fw4*1.e6
        xl=arange(0,41000,1000)
        plot(xl,xl,'r-')
        plot(xl,1.15*xl,'r--',label='F22=1.15*F24')
        plot(xl,xl+3000,'c--',label='F22=F24+3000')
        legend(loc='lower right')
        #axis([-50,4000,-100,4700])
        xlabel('MIPS F24 (micro-Jy)')
        ylabel('WISE F22 (micro-Jy)')
        #figure()
        #subplot(2,2,1)
        #plot(self.w12,self.diff24wise,'b.')
        #subplot(2,2,2)
        #plot(self.w23,self.diff24wise,'b.')
        #subplot(2,2,3)
        #plot(self.w34,self.diff24wise,'b.')
        #print wtable.w4mpro
        
    def getWiseCatalog(self):
        
        url = "http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
        p = {}
        p['spatial'] = "Cone"
        #p['objstr'] = self.prefix
        p['objstr'] = str(self.cra)+'+'+str(self.cdec)
        p['outfmt'] = 1    # IPAC table format
        p['catalog'] = 'wise_allsky_4band_p3as_psd'
        p['radius'] = 360
        
        query = urllib.urlencode(p)
        get_url = url + "?" + query
        handler = urllib2.urlopen(get_url)
        raw = handler.read()
        print raw[0:255]
        fname='/Users/rfinn/research/LocalClusters/WISE/'+self.prefix+'WISE.tbl'
        with open(fname, 'wb') as f:
            f.write(raw)
        self.wisefname=fname
    
        
