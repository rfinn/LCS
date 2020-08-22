#!/usr/bin/env python

###########################
###### IMPORT MODULES
###########################

import LCSbase as lb
from LCScommon import *

from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.stats as st
from scipy.stats import ks_2samp
import argparse# here is min mass = 9.75

from urllib.parse import urlencode
from urllib.request import urlretrieve


from astropy.io import fits, ascii
from astropy.cosmology import WMAP9 as cosmo
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy import units as u
from astropy.stats import median_absolute_deviation as MAD

from PIL import Image


###########################
##### DEFINITIONS
###########################
homedir = os.getenv("HOME")
plotdir = homedir+'/research/LCS/plots/'

#USE_DISK_ONLY = np.bool(np.float(args.diskonly))#True # set to use disk effective radius to normalize 24um size
USE_DISK_ONLY = True
#if USE_DISK_ONLY:
#    print('normalizing by radius of disk')
minsize_kpc=1.3 # one mips pixel at distance of hercules
#minsize_kpc=2*minsize_kpc

mstarmin=10.#float(args.minmass)
mstarmax=10.8
minmass=mstarmin #log of M*
ssfrmin=-12.
ssfrmax=-9
spiralcut=0.8
truncation_ratio=0.5


zmin = 0.0137
zmax = 0.0433
Mpcrad_kpcarcsec = 2. * np.pi/360./3600.*1000.
mipspixelscale=2.45

exterior=.68
colors=['k','b','c','g','m','y','r','sienna','0.5']
shapes=['o','*','p','d','s','^','>','<','v']
#colors=['k','b','c','g','m','y','r','sienna','0.5']

truncated=np.array([113107,140175,79360,79394,79551,79545,82185,166185,166687,162832,146659,99508,170903,18236,43796,43817,43821,70634,104038,104181],'i')

###########################
##### Functions
###########################
# using colors from matplotlib default color cycle
mycolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

colorblind1='#F5793A' # orange
colorblind2 = '#85C0F9' # light blue
colorblind3='#0F2080' # dark blue
darkblue = colorblind3
darkblue = mycolors[1]
lightblue = colorblind2
#lightblue = 'b'
#colorblind2 = 'c'
colorblind3 = 'k'

def colormass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
              xlabel='$\log_{10}(M_\star/M_\odot) $', ylabel='$(g-i)_{corrected} $', color1=colorblind3,color2=colorblind2,\
              nhistbin=50, alpha1=.1,alphagray=.1):

    '''
    PARAMS:
    -------
    * x1,y1
    * x2,y2
    * name1
    * name2
    * hexbinflag
    * contourflag
    * xmin, xmax
    * ymin, ymax
    * contour_bins, ncontour_levels
    * color1
    * color2
    * nhistbin
    * alpha1
    * alphagray
    

    '''
    fig = plt.figure(figsize=(8,8))
    nrow = 4
    ncol = 4
    
    # for purposes of this plot, only keep data within the 
    # window specified by [xmin:xmax, ymin:ymax]
    
    keepflag1 = (x1 >= xmin) & (x1 <= xmax) & (y1 >= ymin) & (y1 <= ymax)
    keepflag2 = (x2 >= xmin) & (x2 <= xmax) & (y2 >= ymin) & (y2 <= ymax)
    
    x1 = x1[keepflag1]
    y1 = y1[keepflag1]
    
    x2 = x2[keepflag2]
    y2 = y2[keepflag2]
    n1 = sum(keepflag1)
    n2 = sum(keepflag2)
    ax1 = plt.subplot2grid((nrow,ncol),(1,0),rowspan=nrow-1,colspan=ncol-1, fig=fig)
    if hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)

        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75,label='_nolegend_')
    else:
        label=name1+' (%i)'%(n1)
        plt.plot(x1,y1,'ko',color=color1,alpha=alphagray,label=label, zorder=10,mec='k',markersize=8)
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=1,colors=color2, label='__nolegend__')
        #plt.legend()
    else:
        label=name2+' (%i)'%(n2)
        plt.plot(x2,y2,'co',color=color2,alpha=alpha1, label=label,markersize=8,mec='k')
        
        
        #plt.legend()
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()
    
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=26)
    plt.ylabel(ylabel,fontsize=26)
    plt.gca().tick_params(axis='both', labelsize=16)
    #plt.axis([7.9,11.6,-.05,2])
    ax2 = plt.subplot2grid((nrow,ncol),(0,0),rowspan=1,colspan=ncol-1, fig=fig, sharex = ax1, yticks=[])
    minx = min([min(x1),min(x2)])
    maxx = max([max(x1),max(x2)])    
    mybins = np.linspace(minx,maxx,nhistbin)
    t = plt.hist(x1, normed=True, bins=mybins,color=color1,histtype='step',lw=1.5, label=name1+' (%i)'%(n1))
    t = plt.hist(x2, normed=True, bins=mybins,color=color2,histtype='step',lw=1.5, label=name2+' (%i)'%(n2))
    #plt.legend()
    leg = ax2.legend(fontsize=18)
    #for l in leg.legendHandles:
    #    l.set_alpha(1)
    #    l._legmarker.set_alpha(1)
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    miny = min([min(y1),min(y2)])
    maxy = max([max(y1),max(y2)])    
    mybins = np.linspace(miny,maxy,nhistbin)
    
    t=plt.hist(y1, normed=True, orientation='horizontal',bins=mybins,color=color1,histtype='step',lw=1.5, label=name1)
    t=plt.hist(y2, normed=True, orientation='horizontal',bins=mybins,color=color2,histtype='step',lw=1.5, label=name2)
    
    plt.yticks(rotation='horizontal')
    ax3.yaxis.tick_right()
    ax3.tick_params(axis='both', labelsize=16)
    ax2.tick_params(axis='both', labelsize=16)
    #ax3.set_title('$log_{10}(SFR)$',fontsize=20)
    #plt.savefig(figname)

    print('############################################################# ')
    print('KS test comparising galaxies within range shown on the plot')
    print('')
    print('STELLAR MASS')
    t = ks(x1,x2,run_anderson=False)
    print('')
    print('COLOR')
    t = ks(y1,y2,run_anderson=False)
    return ax1,ax2,ax3

def plotsalim07():
    #plot the main sequence from Salim+07 for a Chabrier IMF

    lmstar=np.arange(8.5,11.5,0.1)

    #use their equation 11 for pure SF galaxies
    lssfr = -0.35*(lmstar - 10) - 9.83

    #use their equation 12 for color-selected galaxies including
    #AGN/SF composites.  This is for log(Mstar)>9.4
    #lssfr = -0.53*(lmstar - 10) - 9.87

    lsfr = lmstar + lssfr -.3
    sfr = 10.**lsfr

    plt.plot(lmstar, lsfr, 'w-', lw=4)
    plt.plot(lmstar, lsfr, c='salmon',ls='-', lw=2, label='$Salim+07$')
    plt.plot(lmstar, lsfr-np.log10(5.), 'w--', lw=4)
    plt.plot(lmstar, lsfr-np.log10(5.), c='salmon',ls='--', lw=2)

def mass_match(input_mass,comp_mass,dm=.15,nmatch=20,inputZ=None,compZ=None,dz=.0025):
    '''
    for each galaxy in parent, draw nmatch galaxies in match_mass
    that are within +/-dm

    PARAMS:
    -------
    * parent - parent sample to create mass-matched sample from
    * match - sample to draw matches from
    * dm - mass offset from which to draw matched galaxies from
    * nmatch = number of matched galaxies per galaxy in the input
    * comp_sfr = SFRs of comparison sample, if you want those too 

    RETURNS:
    --------
    * indices of the comp_sample

    '''
    # for each galaxy in parent
    # select nmatch galaxies from comp_sample that have stellar masses
    # within +/- dm
    return_index = np.zeros(len(input_mass)*nmatch,'i')

    comp_index = np.arange(len(comp_mass))
    # hate to do this with a loop,
    # but I can't think of a smarter way right now
    for i in range(len(input_mass)):
        # limit comparison sample to mass of i galaxy, +/- dm
        flag = np.abs(comp_mass - input_mass[i]) < dm
        # if redshifts are provided, also restrict based on redshift offset
        if inputZ is not None:
            flag = flag & (np.abs(compZ - inputZ[i]) < dz)
        # select nmatch galaxies randomly from this limited mass range
        # NOTE: can avoid repeating items by setting replace=False
        if sum(flag) < nmatch:
            print('galaxies in slice < # requested',sum(flag),nmatch,input_mass[i],inputZ[i])
        if sum(flag) == 0:
            print('\truh roh - doubling mass and redshift slices')
            flag = np.abs(comp_mass - input_mass[i]) < 2*dm
            # if redshifts are provided, also restrict based on redshift offset
            if inputZ is not None:
                flag = flag & (np.abs(compZ - inputZ[i]) < 2*dz)
        return_index[int(i*nmatch):int((i+1)*nmatch)] = np.random.choice(comp_index[flag],nmatch,replace=True)

    return return_index

                     
def plotelbaz():
    #plot the main sequence from Elbaz+13
        
    xe=np.arange(8.5,11.5,.1)
    xe=10.**xe

    #I think that this comes from the intercept of the
    #Main-sequence curve in Fig. 18 of Elbaz+11.  They make the
    #assumption that galaxies at a fixed redshift have a constant
    #sSFR=SFR/Mstar.  This is the value at z=0.  This is
    #consistent with the value in their Eq. 13

    #This is for a Salpeter IMF
    ye=(.08e-9)*xe   
        
        
    plt.plot(log10(xe),np.log19(ye),'w-',lw=3)
    plt.plot(log10(xe),np.log10(ye),'k-',lw=2,label='$Elbaz+2011$')
    #plot(log10(xe),(2*ye),'w-',lw=4)
    #plot(log10(xe),(2*ye),'k:',lw=2,label='$2 \ SFR_{MS}$')
    plt.plot(log10(xe),np.log10(ye/5.),'w--',lw=4)
    plt.plot(log10(xe),np.log10(ye/5.),'k--',lw=2,label='$SFR_{MS}/5$')

def getlegacy(ra1,dec1,jpeg=True,imsize=None):
    '''
    imsize is size of desired cutout in arcmin
    '''
    default_image_size = 60
    gra = '%.5f'%(ra1) # accuracy is of order .1"
    gdec = '%.5f'%(dec1)
    galnumber = gra+'-'+gdec
    if imsize is not None:
        image_size=imsize
    else:
        image_size=default_image_size
    cwd = os.getcwd()
    if not(os.path.exists(cwd+'/cutouts/')):
        os.mkdir(cwd+'/cutouts')
    rootname = 'cutouts/legacy-im-'+str(galnumber)+'-'+str(image_size)
    jpeg_name = rootname+'.jpg'

    fits_name = rootname+'.fits'

    # check if images already exist
    # if not download images
    if not(os.path.exists(jpeg_name)):
        print('downloading image ',jpeg_name)
        url='http://legacysurvey.org/viewer/jpeg-cutout?ra='+str(ra1)+'&dec='+str(dec1)+'&layer=dr8&size='+str(image_size)+'&pixscale=1.00'
        urlretrieve(url, jpeg_name)
    if not(os.path.exists(fits_name)):
        print('downloading image ',fits_name)
        url='http://legacysurvey.org/viewer/cutout.fits?ra='+str(ra1)+'&dec='+str(dec1)+'&layer=dr8&size='+str(image_size)+'&pixscale=1.00'
        urlretrieve(url, fits_name)
            
    try:
        t,h = fits.getdata(fits_name,header=True)
    except IndexError:
        print('problem accessing image')
        print(fits_name)
        url='http://legacysurvey.org/viewer/cutout.fits?ra='+str(ra1)+'&dec='+str(dec1)+'&layer=dr8&size='+str(image_size)+'&pixscale=1.00'
        print(url)
        return None
    
    if np.mean(t[1]) == 0:
        return None
    norm = simple_norm(t[1],stretch='asinh',percent=99.5)
    if jpeg:
        t = Image.open(jpeg_name)
        plt.imshow(t,origin='lower')
    else:
        plt.imshow(t[1],origin='upper',cmap='gray_r', norm=norm)
    w = WCS(fits_name,naxis=2)        
    
    return t,w

def sersic(x,Ie,n,Re):
    bn = 1.999*n - 0.327
    return Ie*np.exp(-1*bn*((x/Re)**(1./n)-1))

def plot_models():
    plt.figure(figsize=(8,6))
    plt.subplots_adjust(wspace=.35)

    rmax = 4
    scaleRe = 0.8
    rtrunc = 1.5
    n=1
    # shrink Re
    plt.subplot(2,2,1)
    x = np.linspace(0,rmax,100)
    
    Ie=1
    Re=1
    y = sersic(x,Ie,n,Re)
    plt.plot(x,y,label='sersic n='+str(n),lw=2)
    y2 = sersic(x,Ie,n,scaleRe*Re)
    plt.plot(x,y2,label="Re "+r"$ \rightarrow \ $"+str(scaleRe)+"Re",ls='--',lw=2)
    #plt.legend()
    
    plt.ylabel('Intensity',fontsize=18)
    plt.text(0.1,.85,'(a)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    
    # plot total flux
    plt.subplot(2,2,2)
    dx = x[1]-x[0]
    sum1 = (y*dx*2*np.pi*x)
    sum2 = (y2*dx*2*np.pi*x)
    plt.plot(x,np.cumsum(sum1)/np.max(np.cumsum(sum1)),label='sersic n='+str(n),lw=2)
    plt.plot(x,np.cumsum(sum2)/np.max(np.cumsum(sum1)),label="Re "+r"$ \rightarrow \ $"+str(scaleRe)+"Re",ls='--',lw=2)
    plt.ylabel('Enclosed Flux',fontsize=18)
    plt.grid()
    plt.text(0.1,.85,'(b)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    
    # truncated sersic model
    plt.subplot(2,2,3)
    plt.plot(x,y,label='sersic n='+str(n),lw=2)
    y3 = y.copy()
    flag = x > rtrunc
    y3[flag] = np.zeros(sum(flag))
    plt.plot(x,y3,ls='--',label='Rtrunc =  '+str(rtrunc)+' Re',lw=2)
    plt.legend()
    plt.ylabel('Intensity',fontsize=18)
    #plt.legend()
    #plt.gca().set_yscale('log')
    plt.text(0.1,.85,'(c)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    plt.xlabel('r/Re')
    
    plt.subplot(2,2,4)
    sum3 = (y3*dx*2*np.pi*x)
    plt.plot(x,np.cumsum(sum1)/np.max(np.cumsum(sum1)),label='sersic n='+str(n),lw=2)
    plt.plot(x,np.cumsum(sum3)/np.max(np.cumsum(sum1)),label='Rtrunc =  '+str(rtrunc)+' Re',ls='--',lw=2)
    plt.ylabel('Enclosed Flux',fontsize=18)
    
    plt.grid()
    plt.text(0.1,.85,'(d)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    plt.xlabel('r/Re')
    plt.savefig(plotdir+'/cartoon-models.png')
    plt.savefig(plotdir+'/cartoon-models.pdf')
    pass
    
###########################
##### Plot parameters
###########################

# figure setup
plotsize_single=(6.8,5)
plotsize_2panel=(10,5)
params = {'backend': 'pdf',
          'axes.labelsize': 24,
          'font.size': 20,
          'legend.fontsize': 12,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          #'lines.markeredgecolor'  : 'k',  
          #'figure.titlesize': 20,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'text.usetex': True,
          'figure.figsize': plotsize_single}
plt.rcParams.update(params)
#figuredir = '/Users/rfinn/Dropbox/Research/MyPapers/LCSpaper1/submit/resubmit4/'
figuredir='/Users/grudnick/Work/Local_cluster_survey/Papers/Finn_MS/Plots/'
#figuredir = '/Users/grudnick/Work/Local_cluster_survey/Analysis/MS_paper/Plots/'

#colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

########################################
##### STATISTICS TO COMPARE CORE VS EXTERNAL
########################################

test_statistics = lambda x: (np.mean(x), np.var(x), MAD(x), st.skew(x), st.kurtosis(x))
stat_cols = ['mean','var','MAD','skew','kurt']
###########################
##### START OF GALAXIES CLASS
###########################


class gswlc_full():
    def __init__(self,catalog,cutBT=False):
        if catalog.find('.fits') > -1:
            self.gsw = Table(fits.getdata(catalog))
            self.redshift_field = 'zobs'
            if cutBT:
                self.outfile = catalog.split('.fits')[0]+'-LCS-Zoverlap-BTcut.fits'
            else:
                self.outfile = catalog.split('.fits')[0]+'-LCS-Zoverlap.fits'                        
            
        else:
            self.gsw = ascii.read(catalog)
            if catalog.find('v2') > -1:
                self.redshift_field = 'Z_1'
            else:
                self.redshift_field = 'Z'
            if cutBT:
                self.outfile = catalog.split('.dat')[0]+'-LCS-Zoverlap-BTcut.fits'
            else:
                self.outfile = catalog.split('.dat')[0]+'-LCS-Zoverlap.fits'                            
        self.cut_redshift()
        if cutBT:
            self.cut_BT()
        self.save_trimmed_cat()

    def cut_redshift(self):
        z1 = zmin
        z2 = zmax
        zflag = (self.gsw[self.redshift_field] > z1) & (self.gsw[self.redshift_field] < z2)
        massflag = self.gsw['logMstar'] > 0
        self.gsw = self.gsw[zflag & massflag]

    def cut_BT(self):
        btflag = self.gsw['(B/T)r'] < 0.3
        self.gsw = self.gsw[btflag]
        
    def save_trimmed_cat(self):
        t = Table(self.gsw)

        t.write(self.outfile,format='fits',overwrite=True)
        
# functions that will be applied to LCS-GSWLC catalog and GSWLC catalog
class gswlc_base():
    def base_init(self):
        self.calc_ssfr()
        
    def calc_ssfr(self):
        self.ssfr = self.cat['logSFR'] - self.cat['logMstar']
        
    def plot_ms(self,plotsingle=True,outfile1=None,outfile2=None):
        if plotsingle:
            plt.figure(figsize=(8,6))
        x = self.cat['logMstar']
        y = self.cat['logSFR']
        plt.plot(x,y,'k.',alpha=.1)
        #plt.hexbin(x,y,gridsize=30,vmin=5,cmap='gray_r')
        #plt.colorbar()
        xl=np.linspace(8,12,50)
        ssfr_limit = -11.5
        plt.plot(xl,xl+ssfr_limit,'c-')
        plotsalim07()
        plt.xlabel('logMstar',fontsize=20)
        plt.ylabel('logSFR',fontsize=20)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/sfms-ssfr-cut.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/sfms-ssfr-cut.png')
        else:
            plt.savefig(outfile2)
        
    def plot_positions(self,plotsingle=True, filename=None):
        if plotsingle:
            plt.figure(figsize=(14,8))

        plt.scatter(self.cat['RA'],selfgsw['DEC'],c=np.log10(self.densNN*u.Mpc),vmin=-.5,vmax=1.,s=10)
        plt.colorbar(label='$\log_{10}(N/d_N)$')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.gca().invert_xaxis()
        plt.axhline(y=5)
        plt.axhline(y=60)
        plt.axvline(x=135)
        plt.axvline(x=225)
        #plt.axis([130,230,0,65])
        if filename is not None:
            plt.savefig(filename)
class gswlc(gswlc_base):
    def __init__(self,catalog,cutBT=False):
        self.cat = fits.getdata(catalog) 
        self.base_init()
        self.calc_local_density()
        self.get_field1()
        
    def calc_local_density(self,NN=10):
        try:
            redshift = self.cat['Z']
        except KeyError:
            redshift = self.cat['Z_1']
        pos = SkyCoord(ra=self.cat['RA']*u.deg,dec=self.cat['DEC']*u.deg, distance=redshift*3.e5/70*u.Mpc,frame='icrs')

        idx, d2d, d3d = pos.match_to_catalog_3d(pos,nthneighbor=NN)
        self.densNN = NN/d3d
        self.sigmaNN = NN/d2d
    def get_field1(self):
        ramin=135
        ramax=225
        decmin=5
        decmax=60
        gsw_position_flag = (self.cat['RA'] > ramin) & (self.cat['RA'] < ramax) & (self.cat['DEC'] > decmin) & (self.cat['DEC'] < decmax)
        gsw_density_flag = np.log10(self.densNN*u.Mpc) < .2
        self.field1 = gsw_position_flag & gsw_density_flag
    def plot_field1(self,figname1=None,figname2=None):
        plt.figure(figsize=(14,8))
        plt.scatter(self.cat['RA'],self.cat['DEC'],c=np.log10(self.densNN*u.Mpc),vmin=-.5,vmax=1.,s=10)
        plt.colorbar(label='$\log_{10}(N/d_N)$')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.gca().invert_xaxis()
        plt.axhline(y=5)
        plt.axhline(y=60)
        plt.axvline(x=135)
        plt.axvline(x=225)
        plt.title('Entire GSWLC (LCS z range)')
        plt.axis([130, 230, 0, 65])
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)
    def plot_dens_hist(self,figname1=None,figname2=None):
        plt.figure()
        t = plt.hist(np.log10(self.densNN*u.Mpc), bins=20)
        plt.xlabel('$ \log_{10} (N/d_N)$')

        print('median local density =  %.3f'%(np.median(np.log10(self.densNN*u.Mpc))))
        plt.axvline(x = (np.median(np.log10(self.densNN*u.Mpc))),c='k')
        plt.title('Distribution of Local Density')
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)

#######################################################################
#######################################################################
########## NEW CLASS: LCSGSW
#######################################################################
#######################################################################
class lcsgsw(gswlc_base):
    '''
    functions to operate on catalog that is cross-match between
    LCS and GSWLC 


    '''
    def __init__(self,catalog,sigma_split=600,cutBT=False):
        # read in catalog of LCS matched to GSWLC
        #self.cat = fits.getdata(catalog)
        self.cat = Table.read(catalog)
        self.write_file_for_simulation()        
        self.base_init()
        if cutBT:
            self.cut_BT()
        self.group = self.cat['CLUSTER_SIGMA'] < sigma_split
        self.cluster = self.cat['CLUSTER_SIGMA'] > sigma_split

        #lcspath = homedir+'/github/LCS/'
        #self.lcsbase = lb.galaxies(lcspath)
    def get_DA(self):
        # stole this from LCSbase.py
        #print(self.cat.colnames)
        self.DA=np.zeros(len(self.cat))
        for i in range(len(self.DA)):
            if self.cat['membflag'][i]:
                self.DA[i] = cosmo.angular_diameter_distance(self.cat['CLUSTER_REDSHIFT'][i]).value*Mpcrad_kpcarcsec
            else:
                self.DA[i] = cosmo.angular_diameter_distance(self.cat['ZDIST'][i]).value*Mpcrad_kpcarcsec
        
    def calculate_sizeratio(self):
        self.gim2dflag = self.cat['matchflag']
        # stole this from LCSbase.py
        self.SIZE_RATIO_DISK = np.zeros(len(self.cat))
        a =  self.cat['fcre1'][self.gim2dflag]*mipspixelscale # fcre1 = 24um half-light radius in mips pixels
        b = self.DA[self.gim2dflag]
        c = self.cat['Rd'][self.gim2dflag] # gim2d half light radius for disk in kpc

        # this is the size ratio we use in paper 1
        self.SIZE_RATIO_DISK[self.gim2dflag] =a*b/c
        self.SIZE_RATIO_DISK_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_DISK_ERR[self.gim2dflag] = self.cat['fcre1err'][self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.cat['Rd'][self.gim2dflag]

        self.sizeratio = self.SIZE_RATIO_DISK
        self.sizeratioERR=self.SIZE_RATIO_DISK_ERR
        # size ratio corrected for inclination 
        #self.size_ratio_corr=self.sizeratio*(self.cat.faxisratio1/self.cat.SERSIC_BA)
        
    def write_file_for_simulation(self):
        # need to calculate size ratio
        # for some reason, I don't have this in my table
        # what was I thinking???

        # um, nevermind - looks like it's in the file afterall
        self.get_DA()
        self.calculate_sizeratio()
        # write a file that contains the
        # sizeratio, error, SFR, sersic index, membflag
        # we are using GSWLC SFR
        c1 = Column(self.sizeratio,name='sizeratio')
        c2 = Column(self.sizeratioERR,name='sizeratio_err')
        # using all simard values of sersic fit
        tabcols = [c1,c2,self.cat['membflag'],self.cat['B_T_r'],self.cat['Re'],self.cat['ng'],self.cat['logSFR']]
        tabnames = ['sizeratio','sizeratio_err','membflag','BT','Re','nsersic','logSFR']
        newtable = Table(data=tabcols,names=tabnames)
        newtable = newtable[self.cat['sampleflag']]
        newtable.write(homedir+'/research/LCS/tables/LCS-simulation-data.fits',format='fits',overwrite=True)

    def cut_BT(self):
        btflag = (self.cat['B_T_r'] < 0.3) & (self.cat['matchflag'])
        self.cat = self.cat[btflag]
        self.ssfr = self.ssfr[btflag]
        self.sizeratio = self.sizeratio[btflag]
        self.sizeratioERR = self.sizeratioERR[btflag]                
    def get_mstar_limit(self,rlimit=17.7):
        
        print(rlimit,zmax)
        # assume hubble flow
        
        dmax = zmax*3.e5/70
        # abs r
        # m - M = 5logd_pc - 5
        # M = m - 5logd_pc + 5
        ## r = 22.5 - np.log10(lcsgsw['NMGY'][:,4])
        Mr = rlimit - 5*np.log10(dmax*1.e6) +5
        print(Mr)
        ssfr = self.cat['logSFR'] - self.cat['logMstar']
        flag = (self.cat['logMstar'] > 0) & (ssfr > -11.5)
        Mr = self.cat['ABSMAG'][:,4]+5*np.log10(.7)
        plt.figure()
        plt.plot(Mr[flag],self.cat['logMstar'][flag],'bo',alpha=.2,markersize=3)
        plt.axvline(x=-18.6)
        plt.axhline(y=10)
        plt.axhline(y=9.7,ls='--')        

        plt.xlabel('Mr')
        plt.ylabel('logMstar GSWLC')
        plt.grid(True)        
    def plot_dvdr(self, figname1=None,figname2=None,plotsingle=True):
        # log10(chabrier) = log10(Salpeter) - .25 (SFR estimate)
        # log10(chabrier) = log10(diet Salpeter) - 0.1 (Stellar mass estimates)
        xmin,xmax,ymin,ymax = 0,3.5,0,3.5
        if plotsingle:
            plt.figure(figsize=(8,6))
            ax=plt.gca()
            plt.subplots_adjust(left=.1,bottom=.15,top=.9,right=.9)
            plt.ylabel('$ \Delta v/\sigma $',fontsize=26)
            plt.xlabel('$ \Delta R/R_{200}  $',fontsize=26)
            plt.legend(loc='upper left',numpoints=1)

        if USE_DISK_ONLY:
            clabel=['$R_{24}/R_d$','$R_{iso}(24)/R_{iso}(r)$']
        else:
            clabel=['$R_e(24)/R_e(r)$','$R_{iso}(24)/R_{iso}(r)$']
        
        x=(self.cat['DR_R200'])
        y=abs(self.cat['DELTA_V'])
        plt.hexbin(x,y,extent=(xmin,xmax,ymin,ymax),cmap='gray_r',gridsize=50,vmin=0,vmax=10)
        xl=np.arange(0,2,.1)
        plt.plot(xl,-4./3.*xl+2,'k-',lw=3,color=colorblind1)
        props = dict(boxstyle='square', facecolor='0.8', alpha=0.8)
        plt.text(.1,.1,'CORE',transform=plt.gca().transAxes,fontsize=18,color=colorblind3,bbox=props)
        plt.text(.6,.6,'INFALL',transform=plt.gca().transAxes,fontsize=18,color=colorblind3,bbox=props)        
        #plt.plot(xl,-3./1.2*xl+3,'k-',lw=3)
        plt.axis([xmin,xmax,ymin,ymax])
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)            
    def compare_sfrs(self,shift=None,masscut=None,nbins=20):
        '''
        plot distribution of LCS external and core SFRs

        '''
        if masscut is None:
            masscut = self.minmass
        flag = (self.cat['logSFR'] > -99) & (self.cat['logSFR']-self.cat['logMstar'] > -11.5) & (self.cat['logMstar'] > masscut)
        sfrcore = self.cat['logSFR'][self.cat['membflag'] & flag] 
        sfrext = self.cat['logSFR'][~self.cat['membflag']& flag]
        plt.figure()
        mybins = np.linspace(-2.5,1.5,nbins)

        plt.hist(sfrext,bins=mybins,histtype='step',label='External',lw=3)
        if shift is not None:
            plt.hist(sfrext+np.log10(1-shift),bins=mybins,histtype='step',label='Shifted External',lw=3)
        plt.hist(sfrcore,bins=mybins,histtype='step',label='Core',lw=3)            
        plt.legend()
        plt.xlabel('SFR')
        plt.ylabel('Normalized Counts')
        print('CORE VS EXTERNAL')
        t = ks(sfrcore,sfrext,run_anderson=False)

    def plot_ssfr_sizeratio(self,outfile1='plot.pdf',outfile2='plot.png'):
        '''
        GOAL: 
        * compare sSFR vs sizeratio for galaxies in the paper 1 sample

        INPUT: 
        * outfile1 - usually pdf name for output plot
        * outfile2 - usually png name for output plot

        OUTPUT:
        * plot of sSFR vs sizeratio
        '''

        plt.figure(figsize=(8,6))
        flag = self.cat['sampleflag']
        plt.scatter(self.ssfr[flag],self.sizeratio[flag],c=self.cat['B_T_r'][flag])
        cb = plt.colorbar(label='B/T')
        plt.ylabel('$R_e(24)/R_e(r)$',fontsize=20)
        plt.xlabel('$sSFR / (yr^{-1})$',fontsize=20)
        t = spearmanr(self.sizeratio[flag],self.ssfr[flag])
        print(t)
        plt.savefig(outfile1)
        plt.savefig(outfile2)
        

#######################################################################
#######################################################################
########## NEW CLASS: comp_lcs_gsw
#######################################################################
#######################################################################

class comp_lcs_gsw():
    '''
    class that operates on LCS and GSWLC

    used to compare LCS with field sample constructed from SDSS with GSWLC SFR and Mstar values

    '''
    def __init__(self,lcs,gsw,minmstar = 10, minssfr = -11.5):
        self.lcs = lcs
        self.gsw = gsw
        self.masscut = minmstar
        self.ssfrcut = minssfr
        self.lowssfr_flag = (self.lcs.cat['logMstar']> self.masscut)  &\
            (self.lcs.ssfr > self.ssfrcut) & (self.lcs.ssfr < -11.)
        
        
    def plot_sfr_mstar(self,lcsflag=None,label='LCS core',outfile1=None,outfile2=None,coreflag=True,massmatch=True):
        """
        OVERVIEW:
        * compares ssfr vs mstar of lcs and gswlc field samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:
        * lcsflag 
        - if None, this will select LCS members
        - you can use this to select other slices of the LCS sample, like external
        
        * label
        - name of the LCS subsample being plotted
        - default is 'LCS core'

        * outfile1, outfile2
        - figure name to save as
        - I'm using two to save a png and pdf version

        OUTPUT:
        * creates a plot of ssfr vs mstar
        * creates histograms above x and y axes to compare two samples

        """
        
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        
        flag1 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        # removing field1 cut because we are now using Tempel catalog that only
        # includes galaxies in halo masses logM < 12.5
        flag2 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        print('number in lcs sample = ',sum(flag1))
        print('number in gsw sample = ',sum(flag2))
        # GSWLC sample
        x1 = self.gsw.cat['logMstar'][flag2]
        y1 = self.gsw.cat['logSFR'][flag2]
        z1 = self.gsw.cat['Z_1'][flag2]
        # LCS sample (forgive the switch in indices)
        x2 = self.lcs.cat['logMstar'][flag1]
        y2 = self.lcs.cat['logSFR'][flag1]
        z2 = self.lcs.cat['Z'][flag1]
        # get indices for mass-matched gswlc sample
        if massmatch:
            keep_indices = mass_match(x2,x1,inputZ=z2,compZ=z1,dz=.002)
            x1 = x1[keep_indices]
            y1 = y1[keep_indices]
        
        if coreflag:
            color2=darkblue
        else:
            color2=lightblue
        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'GSWLC',label,'sfr-mstar-gswlc-field.pdf',ymin=-2,ymax=1.6,xmin=8.5,xmax=11.5,nhistbin=10,ylabel='$\log_{10}(SFR)$',contourflag=False,alphagray=.1,hexbinflag=True,color2=color2,color1='0.5',alpha1=1)
        # add marker to figure to show galaxies with size measurements
        flag = self.lcs.cat['sampleflag'] & lcsflag
        ax1.plot(self.lcs.cat['logMstar'][flag],self.lcs.cat['logSFR'][flag],'ks',color='darkmagenta',alpha=.5,markersize=10,mec='w',label='LCS size sample ('+str(sum(flag))+')')
        ax1.legend(loc='upper left')
        
        plt.subplots_adjust(left=.15)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.png')
        else:
            plt.savefig(outfile2)
    def plot_ssfr_mstar(self,lcsflag=None,outfile1=None,outfile2=None,label='LCS core',nbins=20,coreflag=True,massmatch=True):
        """
        OVERVIEW:
        * compares ssfr vs mstar of lcs and gswlc field samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:
        * lcsflag 
        - if None, this will select LCS members
        - you can use this to select other slices of the LCS sample, like external
        
        * label
        - name of the LCS subsample being plotted
        - default is 'LCS core'

        * outfile1, outfile2
        - figure name to save as
        - I'm using two to save a png and pdf version

        OUTPUT:
        * creates a plot of ssfr vs mstar
        * creates histograms above x and y axes to compare two samples

        """
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        
        flag1 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        flag2 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.ssfr[flag1]

        
        x2 = self.gsw.cat['logMstar'][flag2]
        y2 = self.gsw.ssfr[flag2]

        z1 = self.gsw.cat['Z_1'][flag2]
        # LCS sample (forgive the switch in indices)
        x2 = self.lcs.cat['logMstar'][flag1]
        y2 = self.lcs.cat['logSFR'][flag1]
        z2 = self.lcs.cat['Z'][flag1]
        # get indices for mass-matched gswlc sample
        if massmatch:
            keep_indices = mass_match(x2,x1,inputZ=z2,compZ=z1,dz=.002)

        if coreflag:
            color2=darkblue
        else:
            color2=lightblue
            
        # get indices for mass-matched gswlc sample
        if massmatch:
            keep_indices = mass_match(x2,x1)
            x1 = x1[keep_indices]
            y1 = y1[keep_indices]

        
        colormass(x1,y1,x2,y2,'GSWLC',label,'sfr-mstar-gswlc-field.pdf',ymin=-11.6,ymax=-8.75,xmin=9.75,xmax=11.5,nhistbin=nbins,ylabel='$\log_{10}(sSFR)$',contourflag=False,alphagray=.8,hexbinflag=True,color1='0.5',color2=color2)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.png')
        else:
            plt.savefig(outfile2)

    def plot_sfr_mstar_lcs(self,outfile1=None,outfile2=None,nbins=20):
        """
        OVERVIEW:
        * compares ssfr vs mstar within lcs samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:

        * outfile1, outfile2
        - figure name to save as
        - I'm using two to save a png and pdf version

        OUTPUT:
        * creates a plot of ssfr vs mstar
        * creates histograms above x and y axes to compare two samples

        """
        
        baseflag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        flag1 = baseflag & self.lcs.cat['membflag'] 
        flag2 = baseflag & ~self.lcs.cat['membflag']  & (self.lcs.cat['DELTA_V'] < 3.)
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.cat['logSFR'][flag1]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        
        colormass(x1,y1,x2,y2,'LCS core','LCS infall','sfr-mstar-lcs-core-field.pdf',ymin=-2,ymax=1.5,xmin=9.75,xmax=11.5,nhistbin=nbins,ylabel='$\log_{10}(SFR)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-external-sfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-external-sfrmstar.png')
        else:
            plt.savefig(outfile2)

    def plot_ssfr_mstar_lcs(self,outfile1=None,outfile2=None,nbins=20):
        """
        OVERVIEW:
        * compares ssfr vs mstar within lcs samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:

        * outfile1, outfile2
        - figure name to save as
        - I'm using two to save a png and pdf version

        OUTPUT:
        * creates a plot of ssfr vs mstar
        * creates histograms above x and y axes to compare two samples

        """
        
        baseflag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        flag1 = baseflag & self.lcs.cat['membflag'] 
        flag2 = baseflag & ~self.lcs.cat['membflag']  & (self.lcs.cat['DELTA_V'] < 3.)
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.ssfr[flag1]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.ssfr[flag2]
        
        colormass(x1,y1,x2,y2,'LCS core','LCS infall','sfr-mstar-lcs-core-field.pdf',ymin=-11.6,ymax=-8.75,xmin=9.75,xmax=11.5,nhistbin=nbins,ylabel='$\log_{10}(sSFR)$',contourflag=False,alphagray=.8,alpha1=1)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.png')
        else:
            plt.savefig(outfile2)


    def print_lowssfr_nsaids(self,lcsflag=None,ssfrmin=None,ssfrmax=-11):
        if ssfrmin is not None:
            ssfrmin=ssfrmin
        else:
            ssfrmin = -11.5
        lowssfr_flag = (self.lcs.cat['logMstar']> self.masscut)  &\
            (self.lcs.ssfr > ssfrmin) & (self.lcs.ssfr < ssfrmax)

        
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        flag = lcsflag & lowssfr_flag        
        nsaids = self.lcs.cat['NSAID'][flag]
        for n in nsaids:
            print(n)
            
    def get_legacy_images(self,lcsflag=None,ssfrmin=None,ssfrmax=-11):
        if ssfrmin is not None:
            ssfrmin=ssfrmin
        lowssfr_flag = (self.lcs.cat['logMstar']> self.masscut)  &\
            (self.lcs.ssfr > ssfrmin) & (self.lcs.ssfr < ssfrmax)
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        
        flag = lowssfr_flag & lcsflag
        ids = np.arange(len(self.lowssfr_flag))[flag]
        for i in ids:
            # open figure
            plt.figure()
            # get legacy image
            d, w = getlegacy(self.lcs.cat['RA_1'][i],self.lcs.cat['DEC_2'][i])
            plt.title("NSID {0}, sSFR={1:.1f}".format(self.lcs.cat['NSAID'][i],self.lcs.ssfr[i]))
    def lcs_compare_BT(self):
        plt.figure()
        x1 = self.lcs.cat['B_T_r'][self.lcs.cat['membflag']]
        x2 = self.lcs.cat['B_T_r'][~self.lcs.cat['membflag'] &( self.lcs.cat['DELTA_V'] < 3.)]
        plt.hist(x1,label='Core',histtype='step',normed=True,lw=2)
        plt.hist(x2,label='Infall',histtype='step',normed=True,lw=2)
        plt.xlabel('B/T',fontsize=20)
        plt.legend()
        t = ks_2samp(x1,x2)
        print(t)

        # for BT < 0.3 only
        plt.figure()
        flag =  self.lcs.cat['B_T_r'] < 0.3
        x1 = self.lcs.cat['B_T_r'][self.lcs.cat['membflag'] & flag]
        x2 = self.lcs.cat['B_T_r'][~self.lcs.cat['membflag'] &( self.lcs.cat['DELTA_V'] < 3.) & flag]
        plt.hist(x1,label='Core',histtype='step',normed=True,lw=2)
        plt.hist(x2,label='Infall',histtype='step',normed=True,lw=2)
        plt.xlabel('B/T',fontsize=20)
        plt.legend()
        t = ks_2samp(x1,x2)
        print(t)        
        pass

                                  
if __name__ == '__main__':
    ###########################
    ##### SET UP ARGPARSE
    ###########################

    parser = argparse.ArgumentParser(description ='Program to run analysis for LCS paper 2')
    parser.add_argument('--minmass', dest = 'minmass', default = 10., help = 'minimum stellar mass for sample.  default is log10(M*) > 7.9')
    parser.add_argument('--cutBT', dest = 'cutBT', default = False, action='store_true', help = 'Set this to cut the sample by B/T < 0.3.')
    #parser.add_argument('--cutBT', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')    

    args = parser.parse_args()
    
    trimgswlc = True
    if trimgswlc:
        #g = gswlc_full('/home/rfinn/research/GSWLC/GSWLC-X2.dat')
        # 10 arcsec match b/s GSWLC-X2 and Tempel-12.5_v_2 in topcat, best, symmetric
        #g = gswlc_full('/home/rfinn/research/LCS/tables/GSWLC-Tempel-12.5-v2.fits')
        g = gswlc_full(homedir+'/research/GSWLC/GSWLC-Tempel-12.5-v2-Simard2011.fits',cutBT=args.cutBT)        
        g.cut_redshift()
        g.save_trimmed_cat()

    #g = gswlc('/home/rfinn/research/LCS/tables/GSWLC-X2-LCS-Zoverlap.fits')
    if args.cutBT:
        infile=homedir+'/research/GSWLC/GSWLC-Tempel-12.5-v2-Simard2011-LCS-Zoverlap-BTcut.fits'
    else:
        infile = homedir+'/research/GSWLC/GSWLC-Tempel-12.5-v2-Simard2011-LCS-Zoverlap.fits'
    g = gswlc(infile)
    #g.plot_ms()
    #g.plot_field1()
    lcsfile = homedir+'/research/LCS/tables/lcs-gswlc-x2-match.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS_all_size_KE_SFR_GSWLC2_X2.fits'
    lcs = lcsgsw(lcsfile,cutBT=args.cutBT)
    #lcs = lcsgsw('/home/rfinn/research/LCS/tables/LCS_all_size_KE_SFR_GSWLC2_X2.fits',cutBT=args.cutBT)    
    #lcs.compare_sfrs()
    b = comp_lcs_gsw(lcs,g,minmstar=float(args.minmass))
