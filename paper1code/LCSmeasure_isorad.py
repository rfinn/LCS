#!/usr/bin/env python

'''
PURPOSE:

to calculate isophotal radii from ellipse table output
to calculate R90 from ellipse table output



'''

###############

import argparse
parser=argparse.ArgumentParser()
#parser.add_argument("agcnumber",help='agcnumber')
parser.add_argument("-a",'--all',help="run for all clusters",action="store_true")
parser.add_argument("-c",'--cluster',help="cluster name (run one cluster only)",action="store")
parser.add_argument("-s",'--startindex', help="starting index (to continue from previous)",action="store")
parser.add_argument("-n",'--startnsaid', help="starting NSAID (to continue from previous)",action="store")

#parser.add_argument("-d",'--display', help="display result in ds9",action="store_true")
args = parser.parse_args()

###############

from LCScommon import *
import numpy as np
from pylab import *
from astropy.io import fits
import sys
from scipy import interpolate

mosaicpixelscale=.423





#constants
ratio_r2ha=18.43
kpscale=.0015
nsacut=.05
mipscut=.154
# looked up SDSS surface brightness limit
# equal to 26.9 mag/arcsec^2
# this corresponds to intensity of 0.01738 nmgy
# if I use .025, this is sb=26.5 mag/arcsec^2
nsacut=sdss_sb_cut
# use a lower limit for MIPS as well
mipscut=mips_sb_cut
hacut=nsacut/(kpscale*ratio_r2ha)
kprcut=nsacut/kpscale


def get_isorad(data_file,sb,pixelscale=2.45,ynorm=1):
    #print data_file
    data1 = np.genfromtxt(data_file)
    sma = data1[:,1]*pixelscale
    intens, intens_err = data1[:,2]*ynorm, data1[:,3]*ynorm
    f=interpolate.interp1d(intens,sma)
    #riso = f(sb)
    #print 'interpolating!!!'

    try:
        riso = f(sb)
        print 'interpolating in get_isorad!!!'

    except ValueError:
        if sum(intens > sb) == len(intens):
            isoindex=len(intens)-1
            riso=sma[isoindex]
            return riso
        elif sum(intens > sb) > 0:
            y=np.where(intens < sb)
            try:
                isoindex=min(y[0])

            except:
                print y
                print 'intens > sb = ',sum(intens > sb), len(intens),data_file
                
                print 'error in r_iso algorithm'
                sys.exit()

            x1=sma[isoindex]
            #try:
            #    x2=sma[isoindex+1]
            #    y1=intens[isoindex]
            #    y2=intens[isoindex+1]
            #    riso=(sb-y1)*(x2-x1)/(y2-y1) + x1
            #except IndexError:
            #    riso=x1
            riso=x1
            if riso < 0:
                riso = sma[isoindex]
        else:
            riso = 0
            return riso
    # what if profile pops up again due to noise?
    return riso

def get_r90(data_file,total_flux,pixelscale=2.45,ynorm=1,fflux=.9,mkplot=0):
    #print data_file
    data1 = np.genfromtxt(data_file)
    sma = data1[:,1]*pixelscale
    # 21 - TFLUX_E
    # 22 - TFLUX_C

    intens = data1[:,21]
    f=interpolate.interp1d(intens,sma)
    #riso = f(sb)
    #print 'interpolating!!!'
    sb=fflux*total_flux
    try:
        riso = f(sb)
        print 'interpolating in get_r90!!!'

    except ValueError:
        if sum(intens > sb) == 0:
            print 'never got R90!'
            return 0
        elif sum(intens > sb) > 0:
            for i in range(len(intens)):
                if intens[i] > sb:
                    isoindex=i-1
                    break

            x1=0.5*(sma[isoindex]+sma[isoindex+1])
            #try:
            #    x2=sma[isoindex+1]
            #    y1=intens[isoindex]
            #    y2=intens[isoindex+1]
            #    riso=(sb-y1)*(x2-x1)/(y2-y1) + x1
            #except IndexError:
            #    riso=x1
            riso=x1
            if riso < 0:
                riso = sma[isoindex]
        else:
            riso = 0
            return riso
    # what if profile pops up again due to noise?

    if mkplot:
        figure()
        plot(sma,intens)
        axhline(y=total_flux,ls='-')
        axhline(y=fflux*total_flux,ls=':')
        axvline(x=riso,ls='-',color='r')

    return riso


# set up parameters
infile=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_all.fits'
# read in LCSspirals catalog
hdr=fits.open(infile)
spirals=hdr[1].data
hdr.close()
nsadict=dict((a,b) for a,b in zip(spirals.NSAID,arange(len(spirals.NSAID))))
agcdict=dict((a,b) for a,b in zip(spirals.AGCNUMBER,arange(len(spirals.NSAID))))

# intialize arrays

riso=zeros(len(spirals.RA),'f')
mipsiso=zeros(len(spirals.RA),'f')
mipsisoe0=zeros(len(spirals.RA),'f')
haiso=zeros(len(spirals.RA),'f')
kpriso=zeros(len(spirals.RA),'f')
haconviso=zeros(len(spirals.RA),'f')




def measureone(i,mkplot=0):
    print '### ',spirals.NSAID[i],' ###'
    rdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.dat'
    #crdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/NSA/c'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-cutout.dat'

    mipsdat=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.dat'
    mipse0=homedir+'research/LocalClusters/GalfitAnalysis/'+str(spirals.CLUSTER[i])+'/24um/'+str(spirals.CLUSTER[i])+'-'+str(spirals.NSAID[i])+'-galfit-cutout24.e05.dat'
    nsar90=0
    mipsr90=0
    if os.path.isfile(rdat):
        isor=get_isorad(rdat,nsacut,pixelscale=sdsspixelscale)
        nsar90=get_r90(rdat,spirals.NMGY[:,4][i],pixelscale=sdsspixelscale,fflux=.9,mkplot=mkplot)
    else:
        print 'bummer, no NSA ellipse data'
        isor=0
    if os.path.isfile(mipsdat):
        isomips=get_isorad(mipsdat,mipscut,pixelscale=mipspixelscale)
        mipsr90=get_r90(mipsdat,spirals.SE_FLUX_BEST[i],pixelscale=mipspixelscale,fflux=.9,mkplot=mkplot)
    else:
        print 'bummer, no mips ellipse data'
        isomips=0
    if os.path.isfile(mipse0):
        isomipse0=get_isorad(mipse0,mipscut,pixelscale=mipspixelscale)
    else:
        print 'bummer, no mips e0 data'
        isomipse0=0


    if (spirals.CLUSTER[i] == 'MKW11') or (spirals.CLUSTER[i] == 'MKW8'):
        hadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        kprdat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/'+str(int(spirals.AGCNUMBER[i]))+'_R.dat'
        chadat='/Users/rfinn/research/LocalClusters/Halpha/cutouts/'+str(spirals.CLUSTER[i])+'/c'+str(int(spirals.AGCNUMBER[i]))+'_Ha.dat'
        if os.path.isfile(hadat):
            isoha=get_isorad(hadat,hacut,pixelscale=mosaicpixelscale)
        else:
            print 'bummer, no mips ha data'
            isoha=0
        if os.path.isfile(kprdat):
            isokpR=get_isorad(kprdat,kprcut,pixelscale=mosaicpixelscale)
        else:
            print 'bummer, no kp R data'
            isokpR=0
        if os.path.isfile(chadat):
            isohaconv=get_isorad(chadat,hacut,pixelscale=mosaicpixelscale)
        else:
            print 'bummer, no conv ha data'
            isohaconv=0
    else:
        isokpR=0
        isoha=0
        isohaconv=0

    # find r90
    print 'nsar90 = ',nsar90
    print 'mipsr90 = ',mipsr90
    return isor,isomips,isomipse0,isokpR,isoha,isohaconv,nsar90,mipsr90



riso=zeros(len(spirals.RA),'f')
mipsiso=zeros(len(spirals.RA),'f')
mipsisoe0=zeros(len(spirals.RA),'f')
haiso=zeros(len(spirals.RA),'f')
kpriso=zeros(len(spirals.RA),'f')
haconviso=zeros(len(spirals.RA),'f')
nsar90=zeros(len(spirals.RA),'f')
mipsr90=zeros(len(spirals.RA),'f')

def runall(cluster,startindex=0):
    keepflag = (spirals.CLUSTER == cluster) # & (spirals.SERSIC_TH50 > mipspixelscale)
    index=arange(len(spirals.RA))
    keepindex=index[keepflag]
    for i in keepindex:
        riso[i],mipsiso[i],mipsisoe0[i],kpriso[i],haiso[i],haconviso[i],nsar90[i],mipsr90[i]=measureone(i)



def writefits():
    output24=homedir+'research/LocalClusters/NSAmastertables/LCS_Spirals_isorad.fits'
    col0 = fits.Column(name='NSAID',array=spirals.NSAID,format='J')
    col1 = fits.Column(name='CLUSTER',array=spirals.CLUSTER,format='9A')
    col2 = fits.Column(name='NSA',array=riso,format='E')
    col3 = fits.Column(name='MIPS',array=mipsiso,format='E')
    col4 = fits.Column(name='MIPSE0',array=mipsisoe0,format='E')
    col5 = fits.Column(name='HA',array=haiso,format='E')
    col6 = fits.Column(name='HACONV',array=haconviso,format='E')
    col7 = fits.Column(name='KPR',array=kpriso,format='E')
    col8 = fits.Column(name='NSAR90',array=nsar90,format='E')
    col9 = fits.Column(name='MIPSR90',array=mipsr90,format='E')
    if os.path.exists(output24):
        os.remove(output24)


    cols=fits.ColDefs([col0,col1,col2,col3,col4,col5,col6,col7,col8,col9])
    tbhdu=fits.new_table(cols)
    tbhdu.writeto(output24)

if args.all: 
        index=arange(len(spirals.RA))
        if args.startindex:
            startindex=int(args.startindex)
        elif args.startnsaid:
            startindex=nsadict[int(args.startnsaid)]
            print 'starting index = ',startindex
        else:
            startindex=0
        
        for i in range(startindex,len(spirals.RA)):
            # keep clean sample only
            #if spirals.p_cs[i] > 0.8:
            print i,spirals.NSAID[i],spirals.CLUSTER[i]

            riso[i],mipsiso[i],mipsisoe0[i],kpriso[i],haiso[i],haconviso[i],nsar90[i],mipsr90[i]=measureone(i)
        writefits()
if args.cluster:
    runall(args.cluster)
    writefits()

    #for c in clusternames:
    #runall('MKW11')
    #runall('MKW8')
    #runall('AWM4')
    #runall('A2063')
    #runall(c)

