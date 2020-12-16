#!/usr/bin/env python

###########################
###### IMPORT MODULES
###########################

import LCSbase as lb
import LCScommon as lcscommon

from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.stats as st
from scipy.stats import ks_2samp, anderson_ksamp
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

minsize_kpc=1.3 # one mips pixel at distance of hercules
BTkey = '__B_T_r'
Rdkey = 'Rd_2'
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


# my version, binned median for GSWLC galaxies with vr < 15,0000 
t = Table.read(homedir+'/research/APPSS/GSWLC2-median-ssfr-mstar-vr15k.dat',format='ipac')
log_mstar2 = t['med_logMstar']
log_ssfr2 = t['med_logsSFR']

def get_BV_MS(logMstar):
    ''' get MS fit that BV calculated from GSWLC '''
    return 0.53*logMstar-5.5

def plot_BV_MS(ax,color='mediumblue',ls='-'):
    plt.sca(ax)
    lsfr = log_mstar2+log_ssfr2
    #plt.plot(log_mstar2, lsfr, 'w-', lw=4)
    plt.plot(log_mstar2, lsfr, c='m',ls='-', lw=6, label='Durbala+20')
    
    x1,x2 = 9.6,11.15
    xline = np.linspace(x1,x2,100)
    yline = get_BV_MS(xline)
    ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
    ax.plot(xline,yline,c=color,ls=ls,lw=3,label='Linear Fit')

    # scatter around MS fit
    sigma=0.3
    ax.plot(xline,yline-1.5*sigma,c='w',ls='--',lw=4)
    ax.plot(xline,yline-1.5*sigma,c=color,ls='--',lw=3,label='fit-1.5$\sigma$')

def plot_GSWLC_sssfr(ax=None,ls='-'):
    if ax is None:
        ax = plt.gca()

    ssfr = -11.5
    x1,x2 = 9.6,11.15
    xline = np.linspace(x1,x2,100)
    yline = ssfr+xline
    ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
    ax.plot(xline,yline,c='0.5',ls=ls,lw=3,label='log(sSFR)=-11.5')
    
def colormass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
              xlabel='$\log_{10}(M_\star/M_\odot) $', ylabel='$(g-i)_{corrected} $', color1=colorblind3,color2=colorblind2,\
              nhistbin=50, alpha1=.1,alphagray=.1,lcsflag=False,ssfrlimit=None):

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
    plt.subplots_adjust(left=.15,bottom=.15)
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
        label=name1+' (%i)'%(n1)
        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75)
    else:
        label=name1+' (%i)'%(n1)
        if lcsflag:
        
            plt.plot(x1,y1,'ko',color=color1,alpha=alphagray,label=label, zorder=10,mec='k',markersize=8)
        else:
            plt.plot(x1,y1,'k.',color=color1,alpha=alphagray,label=label, zorder=1,markersize=8)        
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=1,colors=color2, label='__nolegend__')
        #plt.legend()
    else:
        label=name2+' (%i)'%(n2)
        plt.plot(x2,y2,'co',color=color2,alpha=alpha1, label=label,markersize=8,mec='k')
        
        

    plt.legend()
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()
    if ssfrlimit is not None:
        xl=np.linspace(xmin,xmax,100)
        yl =xl + ssfrlimit
        plt.plot(xl,yl,'k--')#,label='sSFR=-11.5')
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=26)
    plt.ylabel(ylabel,fontsize=26)
    plt.gca().tick_params(axis='both', labelsize=16)
    #plt.axis([7.9,11.6,-.05,2])
    ax2 = plt.subplot2grid((nrow,ncol),(0,0),rowspan=1,colspan=ncol-1, fig=fig, sharex = ax1, yticks=[])
    print('just checking ...',len(x1),len(x2))
    print(min(x1))
    print(min(x2))
    minx = min([min(x1),min(x2)])
    maxx = max([max(x1),max(x2)])    
    mybins = np.linspace(minx,maxx,nhistbin)
    t = plt.hist(x1, normed=True, bins=mybins,color=color1,histtype='step',lw=1.5, label=name1+' (%i)'%(n1))
    t = plt.hist(x2, normed=True, bins=mybins,color=color2,histtype='step',lw=1.5, label=name2+' (%i)'%(n2))
    if hexbinflag:
        plt.legend()
    #leg = ax2.legend(fontsize=12,loc='lower left')
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
    t = lcscommon.ks(x1,x2,run_anderson=True)
    #t = anderson_ksamp(x1,x2)
    #print('Anderson-Darling: ',t)
    print('')
    print('COLOR')
    t = lcscommon.ks(y1,y2,run_anderson=True)
    #t = anderson_ksamp(y1,y2)
    #print('Anderson-Darling: ',t)
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
            if sum(flag) == 0:
                print('\truh roh again - tripling mass and redshift slices')
                flag = np.abs(comp_mass - input_mass[i]) < 4*dm
                # if redshifts are provided, also restrict based on redshift offset
                if inputZ is not None:
                    flag = flag & (np.abs(compZ - inputZ[i]) < 4*dz)
                if sum(flag) == 0:
                    print("can't seem to find a match for mass = ",input_mass[i], i)
                    print("skipping this galaxy")                    
                    continue
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
            print('got a fits file')
            self.gsw = Table(fits.getdata(catalog))
            self.redshift_field = 'zobs'
            if cutBT:
                self.outfile = catalog.split('.fits')[0]+'-LCS-Zoverlap-BTcut.fits'
                print('outfile = ',self.outfile)
            else:
                self.outfile = catalog.split('.fits')[0]+'-LCS-Zoverlap.fits'                        
                print('outfile = ',self.outfile)
        else:
            print('reading ascii')
            self.gsw = ascii.read(catalog)
            if catalog.find('v2') > -1:
                self.redshift_field = 'Z_1'
            else:
                self.redshift_field = 'Z'
            if cutBT:
                self.outfile = catalog.split('.dat')[0]+'-LCS-Zoverlap-BTcut.fits'
            else:
                self.outfile = catalog.split('.dat')[0]+'-LCS-Zoverlap.fits'
        #print(self.gsw.colnames[0:10],len(self.gsw.colnames))
        self.cut_redshift()
        if cutBT:
            self.cut_BT()
        self.save_trimmed_cat()

    def cut_redshift(self):
        z1 = zmin
        z2 = zmax
        #print(self.redshift_field)
        #print(self.gsw.colnames[0:10],len(self.gsw.colnames))        
        #print(self.gsw.colnames)
        print(z1,z2)
        zflag = (self.gsw[self.redshift_field] > z1) & (self.gsw[self.redshift_field] < z2)
        massflag = self.gsw['logMstar'] > 0
        self.gsw = self.gsw[zflag & massflag]

    def cut_BT(self):
        btflag = self.gsw[BTkey] < 0.3
        self.gsw = self.gsw[btflag]
        
    def save_trimmed_cat(self):
        t = Table(self.gsw)

        t.write(self.outfile,format='fits',overwrite=True)
        
# functions that will be applied to LCS-GSWLC catalog and GSWLC catalog
class gswlc_base():
    def base_init(self):
        # selecting agn in a different way - from dr10 catalog
        #self.get_agn()
        # cut catalog to remove agn
        #self.remove_agn()
        self.calc_ssfr()
    def get_agn(self):
        self.specflag = ~np.isnan(self.cat['O3FLUX'])
        self.AGNKAUFF= (np.log10(self.cat['O3FLUX']/self.cat['HBFLUX']) > (.61/(np.log10(self.cat['N2FLUX']/self.cat['HAFLUX']-.05))+1.3)) | (np.log10(self.cat['N2FLUX']/self.cat['HAFLUX']) > 0.)  #& (self.s.HAEW > 0.)
        # add calculations for selecting the sample
        
        self.wiseagn=(self.cat['W1MAG_3'] - self.cat['W2MAG_3']) > 0.8
        self.agnflag = (self.AGNKAUFF & self.specflag) | self.wiseagn
        print('fraction of AGN = %.3f (%i/%i)'%(sum(self.agnflag)/len(self.agnflag),sum(self.agnflag),len(self.agnflag)))
    def remove_agn(self):
        print('REMOVING AGN')
        self.cat = self.cat[~self.agnflag]
        
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
        #self.calc_local_density()
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
        #gsw_density_flag = np.log10(self.densNN*u.Mpc) < .2
        #self.field1 = gsw_position_flag & gsw_density_flag
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
        '''
        c1 = Column(self.agnflag,name='agnflag')
        c2 = Column(self.wiseagn,name='wiseagn')
        c3 = Column(self.AGNKAUFF,name='kauffagn')
        c4 = Column(self.cat['N2FLUX']/self.cat['HAFLUX'],name='n2ha')
        c5 = Column(self.cat['O3FLUX']/self.cat['HBFLUX'],name='o3hb')        
        tabcols = [c1,c2,c3,c4,c5,self.cat['HAFLUX'],self.cat['N2FLUX'],self.cat['HBFLUX'],self.cat['O3FLUX'],self.cat['W1MAG_3'],self.cat['W2MAG_3'],self.cat['AGNKAUFF']]
        tabnames = ['agnflag','wiseagn','kauffagn','N2/HA','O3/HB','HAFLUX','N2FLUX','HBFLUX','O3FLUX','W1MAG_3','W2MAG_3','AGNKAUF']
        newtable = Table(data=tabcols,names=tabnames)
        newtable.write(homedir+'/research/LCS/tables/lcs-gsw-noagn.fits',overwrite=True)
        '''
        self.base_init()

        # this doesn't work after adding agn cut
        # getting error that some array lengths are not the same
        # I am not actually using this data file in the simulation
        # so commenting it out for now.
        # I'm sure the error will pop up somewhere else...
        #self.write_file_for_simulation()

        if cutBT:
            self.cut_BT()
        self.nsadict=dict((a,b) for a,b in zip(self.cat['NSAID'],np.arange(len(self.cat))))                    
        self.get_DA()
        self.get_sizeflag()
        self.get_sbflag()
        self.get_galfitflag()
        self.get_membflag()
        self.get_infallflag()
        self.get_sampleflag()
        self.calculate_sizeratio()
        self.group = self.cat['CLUSTER_SIGMA'] < sigma_split
        self.cluster = self.cat['CLUSTER_SIGMA'] > sigma_split
        self.get_NUV24()
    def get_NUV24(self):
        self.NUVr=self.cat['ABSMAG'][:,1] - self.cat['ABSMAG'][:,4]
        self.NUV = 22.5 - 2.5*np.log10(self.cat['NMGY'][:,1])
        self.MAG24 = 2.5*np.log10(3631./(self.cat['FLUX24']*1.e-6))
        self.NUV24 =self.NUV-self.MAG24

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
    def get_sizeflag(self):
        ''' calculate size flag '''
        self.sizeflag=(self.cat['SERSIC_TH50']*self.DA > minsize_kpc)
    def get_sbflag(self):
        '''  surface brightness flag '''
        self.sb_obs = 999*np.ones(len(self.cat))
        mipsflag = self.cat['FLUX24'] > 0
        self.sb_obs[mipsflag]=(self.cat['fcmag1'][mipsflag] + 2.5*np.log10(np.pi*((self.cat['fcre1'][mipsflag]*mipspixelscale)**2)*self.cat['fcaxisratio1'][mipsflag]))
        self.sbflag = self.sb_obs < 20.
        print('got sb flag')
    def get_gim2dflag(self):
        # don't need this b/c the catalog has already been match to the simard table
        self.gim2dflag=(self.cat['SERSIC_TH50']*self.DA > minsize_kpc)
    def get_galfitflag(self):
        ''' calculate galfit flag '''
        
        self.galfitflag = (self.cat['fcmag1'] > .01)  & ~self.cat['fcnumerical_error_flag24']
        
        
        galfit_override = [70588,70696,43791,69673,146875,82170, 82182, 82188, 82198, 99058, 99660, 99675, 146636, 146638, 146659, 113092, 113095, 72623,72631,72659, 72749, 72778, 79779, 146121, 146130, 166167, 79417, 79591, 79608, 79706, 80769, 80873, 146003, 166044,166083, 89101, 89108,103613,162792,162838, 89063,99509,72800,79381,10368]
        for id in galfit_override:
            try:
                self.galfitflag[self.nsadict[int(id)]] = True
                #print('HEY! found a match, just so you know, with NSAID ',id,'\n\tCould be from sources that were removed with AGN/BT/GSWLC matches')
            except KeyError:
                pass
                #print('got a key error, just so you know, with NSAID ',id,'\n\tCould be from sources that were removed with AGN/BT/GSWLC matches')
            except IndexError:
                pass
                #print('WARNING: got an index error in nsadict for NSAID', id,'\n\tCould be from sources that were removed with AGN/BT/GSWLC matches')
        #self.galfitflag = self.galfitflag
        
        self.galfitflag[self.nsadict[79378]] = False

        # bringing this over from LCSbase.py
        self.badfits=np.zeros(len(self.cat),'bool')
        nearbystar=[142655, 143485, 99840, 80878] # bad NSA fit; 24um is ok
        #nearbygalaxy=[103927,143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        # checked after reworking galfit
        nearbygalaxy=[143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        #badNSA=[166185,142655,99644,103825,145998]
        #badNSA = [
        badfits= nearbygalaxy#+nearbystar+nearbygalaxy
        badfits=np.array(badfits,'bool')
        for gal in badfits:
            flag = self.cat['NSAID'] == gal
            if sum(flag) == 1:
                self.badfits[flag]  = True

        # fold badfits into galfit flag
        self.galfitflag = self.galfitflag & ~self.badfits
    def get_membflag(self):
        self.membflag = (abs(self.cat['DELTA_V']) < (-4./3.*self.cat['DR_R200'] + 2))
    def get_infallflag(self):
        self.infallflag = (abs(self.cat['DELTA_V']) < 3) & ~self.membflag
    def get_sampleflag(self):
        print(len(self.galfitflag),len(self.sbflag),len(self.cat['lirflag']),len(self.sizeflag))
        self.sampleflag=  self.sbflag & self.cat['lirflag'] & self.sizeflag  & self.galfitflag
        #& self.cat['galfitflag2'] #& &  &    #~self.cat['AGNKAUFF'] #&  # #& self.cat['gim2dflag'] ##& ~self.cat['fcnumerical_error_flag24'] 


    def calculate_sizeratio(self):
        # all galaxies in the catalog have been matched to simard table 1
        # does this mean they all have a disk scale length?
        self.gim2dflag = np.ones(len(self.cat),'bool')#  self.cat['matchflag'] & self.cat['lirflag'] & self.cat['sizeflag'] & self.cat['sbflag']
        # stole this from LCSbase.py
        self.SIZE_RATIO_DISK = np.zeros(len(self.cat))
        a =  self.cat['fcre1'][self.gim2dflag]*mipspixelscale # fcre1 = 24um half-light radius in mips pixels
        b = self.DA[self.gim2dflag]
        c = self.cat[Rdkey][self.gim2dflag] # gim2d half light radius for disk in kpc

        # this is the size ratio we use in paper 1
        self.SIZE_RATIO_DISK[self.gim2dflag] =a*b/c
        self.SIZE_RATIO_DISK_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_DISK_ERR[self.gim2dflag] = self.cat['fcre1err'][self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.cat[Rdkey][self.gim2dflag]

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
        c3 = Column(self.agnflag,name='agnflag')        
        # using all simard values of sersic fit
        tabcols = [c1,c2,self.cat['membflag'],self.cat[BTkey],self.cat['Re'],self.cat['ng'],self.cat['logSFR'],c3]
        tabnames = ['sizeratio','sizeratio_err','membflag','BT','Re','nsersic','logSFR','agnflag']
        newtable = Table(data=tabcols,names=tabnames)
        newtable = newtable[self.sampleflag]
        newtable.write(homedir+'/research/LCS/tables/LCS-simulation-data.fits',format='fits',overwrite=True)

    def cut_BT(self):
        btflag = (self.cat[BTkey] < 0.3) & (self.cat['matchflag'])
        self.cat = self.cat[btflag]
        self.ssfr = self.ssfr[btflag]
        #self.sizeratio = self.sizeratio[btflag]
        #self.sizeratioERR = self.sizeratioERR[btflag]                
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
            masscut = self.masscut
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
        t = lcscommon.ks(sfrcore,sfrext,run_anderson=False)

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
        flag = self.sampleflag
        plt.scatter(self.ssfr[flag],self.sizeratio[flag],c=self.cat[BTkey][flag])
        cb = plt.colorbar(label='B/T')
        plt.ylabel('$R_e(24)/R_e(r)$',fontsize=20)
        plt.xlabel('$sSFR / (yr^{-1})$',fontsize=20)
        t = lcscommon.spearmanr(self.sizeratio[flag],self.ssfr[flag])
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
    def __init__(self,lcs,gsw,minmstar = 10, minssfr = -11.5,cutBT=False):
        self.lcs = lcs
        self.gsw = gsw
        self.masscut = minmstar
        self.ssfrcut = minssfr
        self.lowssfr_flag = (self.lcs.cat['logMstar']> self.masscut)  &\
            (self.lcs.ssfr > self.ssfrcut) & (self.lcs.ssfr < -11.)
        self.cutBT = cutBT
        
        self.mass_sfr_flag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)         
    def plot_sfr_mstar(self,lcsflag=None,label='LCS core',outfile1=None,outfile2=None,coreflag=True,massmatch=True,hexbinflag=False,lcsinfall=False,lcsmemb=False):
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

        * massmatch
        - set to True to draw a mass-matched sample from GSWLC field sample

        * hexbinflag
        - set to True to use hexbin to plot field sample
        - do this if NOT drawing mass-matched sample b/c number of points is large

        * lcsinfall
        - select if plotting lcs infall (this affects plotting of galaxies with size measurements)
        * lcsmemb
        - select if plotting lcs memb (this affects plotting of galaxies with size measurements)

        OUTPUT:
        * creates a plot of sfr vs mstar
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
        z2 = self.lcs.cat['Z_1'][flag1]
        # get indices for mass-matched gswlc sample
        if massmatch:
            #keep_indices = mass_match(x2,x1,inputZ=z2,compZ=z1,dz=.002)
            keep_indices = mass_match(x2,x1)            
            # if keep_indices == False
            # remove redshift constraint
            if (len(keep_indices) == 1):
                if (keep_indices == False):
                    print("WARNING: Removing the redshift constraint from the mass-matched sample")
                    keep_indices = mass_match(x2,x1)
                
            x1 = x1[keep_indices]
            y1 = y1[keep_indices]
        
        if coreflag:
            color2=darkblue
        else:
            color2=lightblue
        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'GSWLC',label,'sfr-mstar-gswlc-field.pdf',ymin=-2,ymax=1.6,xmin=9.5,xmax=11.25,nhistbin=10,ylabel='$\log_{10}(SFR/(M_\odot/yr))$',contourflag=False,alphagray=.15,hexbinflag=hexbinflag,color2=color2,color1='0.5',alpha1=1,ssfrlimit=-11.5)
        # add marker to figure to show galaxies with size measurements

        #self.plot_lcs_size_sample(ax1,memb=lcsmemb,infall=lcsinfall,ssfrflag=False)
        #ax1.legend(loc='upper left')
        if not hexbinflag:
            ax1.legend(loc='upper left')
        plot_BV_MS(ax1)
        ax1.legend(loc='lower right')
        plt.subplots_adjust(left=.15)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.png')
        else:
            plt.savefig(outfile2)
    def plot_ssfr_mstar(self,lcsflag=None,outfile1=None,outfile2=None,label='LCS core',nbins=20,coreflag=True,massmatch=True,hexbinflag=True,lcsmemb=False,lcsinfall=False):
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
        
        x1 = self.gsw.cat['logMstar'][flag2]
        y1 = self.gsw.ssfr[flag2]
        z1 = self.gsw.cat['Z_1'][flag2]

        # LCS sample (forgive the switch in indices)
        x2 = self.lcs.cat['logMstar'][flag1]
        y2 = self.lcs.cat['logSFR'][flag1] - self.lcs.cat['logMstar'][flag1] 
        z2 = self.lcs.cat['Z_1'][flag1]
        
        if coreflag:
            color2=darkblue
        else:
            color2=lightblue
            
        # get indices for mass-matched gswlc sample
        if massmatch:
            keep_indices = mass_match(x2,x1,inputZ=z2,compZ=z1,dz=.002)
            x1 = x1[keep_indices]
            y1 = y1[keep_indices]
        
            print('AFTER MASS MATCHING')
            print('number of gswlc = ',len(x1))
            print('number of lcs = ',len(x2))

        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'GSWLC',label,'sfr-mstar-gswlc-field.pdf',ymin=-11.6,ymax=-8.75,xmin=9.5,xmax=11.5,nhistbin=nbins,ylabel='$\log_{10}(sSFR)$',\
                                contourflag=False,alphagray=.15,hexbinflag=hexbinflag,color1='0.5',color2=color2,alpha1=1)

        #self.plot_lcs_size_sample(ax1,memb=lcsmemb,infall=lcsinfall,ssfrflag=True)
        ax1.legend(loc='upper left')

        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.png')
        else:
            plt.savefig(outfile2)

    def plot_sfr_mstar_lcs(self,outfile1=None,outfile2=None,nbins=10):
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
        
        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'LCS core','LCS infall','sfr-mstar-lcs-core-field.pdf',ymin=-2,ymax=1.5,xmin=9.5,xmax=11.25,nhistbin=nbins,ylabel='$\log_{10}(SFR)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True,ssfrlimit=-11.5)

        self.plot_lcs_size_sample(ax1,memb=True,infall=True)
        ax1.legend(loc='upper left')
        plot_BV_MS(ax1)
        ax1.legend(loc='lower right')
        plt.subplots_adjust(left=.15)
        
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


        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'LCS core','LCS infall','sfr-mstar-lcs-core-field.pdf',ymin=-11.6,ymax=-8.75,xmin=9.5,xmax=11.5,nhistbin=nbins,ylabel='$\log_{10}(sSFR)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True)


        #self.plot_lcs_size_sample(ax1,memb=True,infall=True,ssfrflag=True)
        ax1.legend(loc='upper left')
        
        plt.subplots_adjust(left=.15)
        
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.png')
        else:
            plt.savefig(outfile2)
    def plot_lcs_size_sample(self,ax,memb=True,infall=True,ssfrflag=False):
        cmemb = colorblind1
        cmemb = 'darkmagenta'
        cinfall = lightblue
        cinfall = 'blueviolet'
        cinfall = 'darkmagenta'
        if ssfrflag:
            y = self.lcs.cat['logSFR'] - self.lcs.cat['logMstar']
        else:
            y = self.lcs.cat['logSFR']
        baseflag = self.lcs.sampleflag& (self.lcs.cat['logMstar'] > self.masscut)
        if memb:
            flag = self.lcs.sampleflag & self.lcs.cat['membflag']  & (self.lcs.cat['logMstar'] > self.masscut)
            #ax.plot(self.lcs.cat['logMstar'][flag],y[flag],'ks',color=cmemb,alpha=.5,markersize=8,label='LCS memb w/size ('+str(sum(flag))+')')
        if infall:
            flag = self.lcs.sampleflag & ~self.lcs.cat['membflag']  & (self.lcs.cat['DELTA_V'] < 3.) & (self.lcs.cat['logMstar'] > self.masscut)
            #ax.plot(self.lcs.cat['logMstar'][flag],y[flag],'k^',color=cinfall,alpha=.5,markersize=10,label='LCS infall w/size ('+str(sum(flag))+')')
        flag = baseflag& (self.lcs.cat['DELTA_V'] < 3.)
        ax.plot(self.lcs.cat['logMstar'][flag],y[flag],'ks',alpha=.6,zorder=1,markersize=10,lw=2,label='LCS w/size ('+str(sum(flag))+')')
    def plot_dsfr_hist(self,nbins=15,outfile1=None,outfile2=None):
        lcsflag = self.lcs.cat['membflag']
        
        flag1 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        # removing field1 cut because we are now using Tempel catalog that only
        # includes galaxies in halo masses logM < 12.5
        flag2 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        # GSWLC
        x1 = self.gsw.cat['logMstar'][flag2]
        y1 = self.gsw.cat['logSFR'][flag2]
        dsfr1 = y1-get_BV_MS(x1)
        #LCS core
        x2 = self.lcs.cat['logMstar'][flag1]
        y2 = self.lcs.cat['logSFR'][flag1]
        dsfr2 = y2-get_BV_MS(x2)
        #LCS infall
        lcsflag = ~self.lcs.cat['membflag'] & (self.lcs.cat['DELTA_V'] < 3.)
        flag3 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)

        x3 = self.lcs.cat['logMstar'][flag3]
        y3 = self.lcs.cat['logSFR'][flag3]
        dsfr3 = y3-get_BV_MS(x3)        
        plt.figure(figsize=(8,6))

        mybins = np.linspace(-1.5,1.5,nbins)
        delta_bin = mybins[1]-mybins[0]
        mybins = mybins + 0.5*delta_bin
        dsfrs = [dsfr1,dsfr2,dsfr3]
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','LCS Core','LCS Infall']
        orders = [1,3,2]
        alphas = [.4,0,.5]        
        hatches = ['/','\\','|']
        for i in range(len(dsfrs)):
            plt.hist(dsfrs[i],bins=mybins,color=colors[i],normed=True,\
                     histtype='stepfilled',lw=3,alpha=alphas[i],zorder=orders[i])#hatch=hatches[i])
            plt.hist(dsfrs[i],bins=mybins,color=colors[i],normed=True,\
                     histtype='step',lw=3,zorder=orders[i],label=labels[i])#hatch=hatches[i])
            
        plt.xlabel('$ \log_{10}SFR - \log_{10}SFR_{MS} \ (M_\odot/yr) $',fontsize=20)
        plt.ylabel('$Normalized \ Distribution$',fontsize=20)
        plt.legend()
        if outfile1 is not None:
            plt.savefig(outfile1)
        if outfile2 is not None:
            plt.savefig(outfile2)

        print('KS STATISTICS: FIELD VS CORE')
        print(ks_2samp(dsfr1,dsfr2))
        print(anderson_ksamp([dsfr1,dsfr2]))
        print('')
        print('KS STATISTICS: FIELD VS INFALL')
        print(ks_2samp(dsfr1,dsfr3))
        print(anderson_ksamp([dsfr1,dsfr3]))
        print('')              
        print('KS STATISTICS: CORE VS INFALL')
        print(ks_2samp(dsfr2,dsfr3))
        print(anderson_ksamp([dsfr2,dsfr3]))
        

    def plot_dsfr_sizeratio(self,nbins=15,outfile1=None,outfile2=None,sampleflag=None):
        if sampleflag is None:
            sampleflag = self.lcs.sampleflag

        print('number in sampleflag = ',sum(sampleflag),len(sampleflag))
        print('number in membflag = ',sum(self.lcs.membflag),len(self.lcs.membflag))
        lcsflag = self.lcs.membflag & sampleflag        
        print('number in both = ',sum(lcsflag))
        print('number in both and in sfr/mstar cut = ',sum(lcsflag & self.mass_sfr_flag))

        flag2 = lcsflag &  self.mass_sfr_flag
        #LCS core
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        z2 = self.lcs.sizeratio[flag2]        
        dsfr2 = y2-get_BV_MS(x2)
        print('fraction of core with dsfr below 0.3dex = {:.3f} ({:d}/{:d})'.format(sum(dsfr2 < -0.3)/len(dsfr2),sum(dsfr2 < -0.3),len(dsfr2)))
        
        #LCS infall
        lcsflag = sampleflag  &self.lcs.infallflag
        flag3 = lcsflag &  self.mass_sfr_flag
        x3 = self.lcs.cat['logMstar'][flag3]
        y3 = self.lcs.cat['logSFR'][flag3]
        z3 = self.lcs.sizeratio[flag3]
        dsfr3 = y3-get_BV_MS(x3)
        print('fraction of core with dsfr below 0.3dex = {:.3f} ({:d}/{:d})'.format(sum(dsfr3 < -0.3)/len(dsfr3),sum(dsfr3 < -0.3),len(dsfr3)) )       


        # make figure
        #plt.figure(figsize=(8,6))
        sizes = [z2,z3]
        dsfrs = [dsfr2,dsfr3]
        colors = [darkblue,lightblue]
        labels = ['LCS Core ({})'.format(sum(flag2)),'LCS Infall ({})'.format(sum(flag3))]
        hatches = ['/','\\','|']
        #for i in range(len(dsfrs)):
        #    plt.plot(sizes[i],dsfrs[i],'bo',c=colors[i],label=labels[i])
            
        #plt.ylabel('$ SFR - SFR_{MS}(M_\star) \ (M_\odot/yr) $',fontsize=20)
        #plt.xlabel('$R_{24}/R_d$',fontsize=20)
        #plt.legend()
        #plt.axhline(y=0,color='k')

        plt.figure()
        nbins=12
        ax1,ax2,ax3 = colormass(z2,dsfr2,z3,dsfr3,'LCS core','LCS infall','temp.pdf',ymin=-1.5,ymax=1.,xmin=-.05,xmax=2,nhistbin=nbins,xlabel='$R_{24}/R_d$',ylabel='$\log_{10}(SFR)-\log_{10}(SFR_{MS})  \ (M_\odot/yr)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True)
        var1 = z2.tolist()+z3.tolist()
        var2 = dsfrs[0].tolist()+dsfrs[1].tolist()
        t = lcscommon.spearmanr(var1,var2)
        print(t)

        # add line to show sb limit from LCS size measurements
        size=np.linspace(0,2)
        sb = .022
        sfr = sb*(4*np.pi*size**2)
        ax1.plot(size,np.log10(sfr),'k--',label='SB limit')
        ax1.axhline(y=.3,ls=':',color='0.5')
        ax1.axhline(y=-.3,ls=':',color='0.5')

        # plot all galfit results
        baseflag = self.mass_sfr_flag & ~sampleflag #& ~self.lcs.cat['agnflag'] 
        flag4 = self.lcs.cat['membflag'] &  baseflag
        x4 = self.lcs.cat['logMstar'][flag4]
        y4 = self.lcs.cat['logSFR'][flag4]
        z4 = self.lcs.sizeratio[flag4]        
        dsfr4 = y4-get_BV_MS(x4)
        ax1.plot(z4,dsfr4,'kx',c=darkblue,markersize=10)
        
        # plot all galfit results
        flag4 = self.lcs.infallflag&   self.mass_sfr_flag & ~sampleflag #& ~self.lcs.cat['agnflag'] 
        x4 = self.lcs.cat['logMstar'][flag4]
        y4 = self.lcs.cat['logSFR'][flag4]
        z4 = self.lcs.sizeratio[flag4]        
        dsfr4 = y4-get_BV_MS(x4)
        ax1.plot(z4,dsfr4,'kx',markersize=10,c=lightblue)
        
        if outfile1 is not None:
            plt.savefig(outfile1)
        if outfile2 is not None:
            plt.savefig(outfile2)
        return ax1,ax2,ax3

    def plot_dsfr_HIdef(self,nbins=15,outfile1=None,outfile2=None):
        lcsflag = self.lcs.cat['membflag'] & self.lcs.sampleflag & self.lcs.cat['HIflag']

        flag2 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut) 
        #LCS core
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        z2 = self.lcs.cat['HIDef'][flag2]        
        dsfr2 = y2-get_BV_MS(x2)
        
        #LCS infall
        lcsflag = self.lcs.sampleflag & self.lcs.cat['HIflag'] & ~self.lcs.cat['membflag'] & (self.lcs.cat['DELTA_V'] < 3.)
        flag3 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        x3 = self.lcs.cat['logMstar'][flag3]
        y3 = self.lcs.cat['logSFR'][flag3]
        z3 = self.lcs.cat['HIDef'][flag3]
        dsfr3 = y3-get_BV_MS(x3)

        # make figure
        #plt.figure(figsize=(8,6))
        sizes = [z2,z3]
        dsfrs = [dsfr2,dsfr3]
        colors = [darkblue,lightblue]
        labels = ['LCS Core ({})'.format(sum(flag2)),'LCS Infall ({})'.format(sum(flag3))]
        hatches = ['/','\\','|']
        #for i in range(len(dsfrs)):
        #    plt.plot(sizes[i],dsfrs[i],'bo',c=colors[i],label=labels[i])
            
        #plt.ylabel('$ SFR - SFR_{MS}(M_\star) \ (M_\odot/yr) $',fontsize=20)
        #plt.xlabel('$R_{24}/R_d$',fontsize=20)
        #plt.legend()
        #plt.axhline(y=0,color='k')

        plt.figure()
        nbins=12
        ax1,ax2,ax3 = colormass(z2,dsfr2,z3,dsfr3,'LCS core','LCS infall','temp.pdf',ymin=-1,ymax=1.,xmin=-.05,xmax=2,nhistbin=nbins,xlabel='$HI \ Deficiency$',ylabel='$\log_{10}(SFR)-\log_{10}(SFR_{MS})  \ (M_\odot/yr)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True)
        var1 = z2.tolist()+z3.tolist()
        var2 = dsfrs[0].tolist()+dsfrs[1].tolist()
        t = lcscommon.spearmanr(var1,var2)
        print(t)

        if outfile1 is not None:
            plt.savefig(outfile1)
        if outfile2 is not None:
            plt.savefig(outfile2)
            
    def plot_phasespace_dsfr(self):
        ''' plot phase space diagram, and mark galaxies with SFR < (MS - 1.5sigma)   '''

        # maybe try size that scales with size ratio
        pass

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
        x1 = self.lcs.cat[BTkey][self.lcs.cat['membflag']]
        x2 = self.lcs.cat[BTkey][~self.lcs.cat['membflag'] &( self.lcs.cat['DELTA_V'] < 3.)]
        plt.hist(x1,label='Core',histtype='step',normed=True,lw=2)
        plt.hist(x2,label='Infall',histtype='step',normed=True,lw=2)
        plt.xlabel('B/T',fontsize=20)
        plt.legend()
        t = ks_2samp(x1,x2)
        print(t)

        # for BT < 0.3 only
        plt.figure()
        flag =  self.lcs.cat[BTkey] < 0.3
        x1 = self.lcs.cat[BTkey][self.lcs.cat['membflag'] & flag]
        x2 = self.lcs.cat[BTkey][~self.lcs.cat['membflag'] &( self.lcs.cat['DELTA_V'] < 3.) & flag]
        plt.hist(x1,label='Core',histtype='step',normed=True,lw=2)
        plt.hist(x2,label='Infall',histtype='step',normed=True,lw=2)
        plt.xlabel('B/T',fontsize=20)
        plt.legend()
        t = ks_2samp(x1,x2)
        print(t)        
        pass
    

    def compute_ks(self):
        '''
        GOAL:
        * compute KS statistics for table in paper comparing 
          - LCS core/field vs GSWLC
          - LCS core vs field
          - with and witout B/T cut
          - with and without mass matching
        '''

        # no BT cut
        # actually, not that easy b/c some cuts are made in plot panels
        # putting this off until later...
        pass
    def compare_lcs_sfr(self,core=True,infall=True):
        '''hist of sfrs of core vs infall lcs galaxies'''
        plt.figure(figsize=(8,5))
        plt.subplots_adjust(bottom=.2,left=.05)
        mybins=np.linspace(-2.5,2,10)
        ssfr = self.lcs.cat['logSFR'] - self.lcs.cat['logMstar']
        flag1 = (self.lcs.cat['logSFR'] > -50) & (ssfr > self.ssfrcut) & (self.lcs.cat['logMstar'] > self.masscut)
        x1 = self.lcs.cat['logSFR'][self.lcs.cat['membflag'] & flag1 ]
        x2 = self.lcs.cat['logSFR'][~self.lcs.cat['membflag'] &( self.lcs.cat['DELTA_V'] < 3.) & flag1]
        print('KS test comparing SFRs:')
        t = ks_2samp(x1,x2)
        print(t)
        vars = [x1,x2]
        labels = ['LCS Core','LCS Infall']
        mycolors = ['r','b']
        myhatch=['/','\\']
        for i in [0,1]:
            #plt.subplot(1,2,i+1)
            if (i == 0) and not(core):
                continue

            if (i == 1) and not(infall):
                continue
            t = plt.hist(vars[i],label=labels[i],lw=2,bins=mybins,hatch=myhatch[i],color=mycolors[i],histtype='step')                
        plt.xlabel('$log_{10}(SFR/(M_\odot/yr))$',fontsize=24)
        #plt.axis([-2.5,2,0,50])
        plt.legend()
        if not(core):
            outfile=homedir+'/research/LCS/plots/lcsinfall-sfrs'
        elif not(infall):
            outfile=homedir+'/research/LCS/plots/lcscore-sfrs'
        else:
            outfile=homedir+'/research/LCS/plots/lcscore-infall-sfrs'
        if self.cutBT:
            plt.savefig(outfile+'-BTcut.png')
            plt.savefig(outfile+'-BTcut.pdf')        
        else:
            plt.savefig(outfile+'.png')
            plt.savefig(outfile+'.pdf')        
if __name__ == '__main__':
    ###########################
    ##### SET UP ARGPARSE
    ###########################

    parser = argparse.ArgumentParser(description ='Program to run analysis for LCS paper 2')
    parser.add_argument('--minmass', dest = 'minmass', default = 9.7, help = 'minimum stellar mass for sample.  default is log10(M*) > 9.7')
    parser.add_argument('--cutBT', dest = 'cutBT', default = False, action='store_true', help = 'Set this to cut the sample by B/T < 0.3.')
    #parser.add_argument('--cutBT', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')    

    args = parser.parse_args()
    
    trimgswlc = True
    # 10 arcsec match b/w GSWLC-X2-NO-DR10-AGN-Simard2011-tab1 and Tempel_gals_below_13.fits in topcat, best,symmetric
    #gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-13-2020Nov11'
    
    # 10 arcsec match b/w GSWLC-X2-NO-DR10-AGN-Simard2011-tab1 and Tempel_gals_below_13_5.fits in topcat, best,symmetric
    #gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-13.5-2020Nov11'
    
    # 10 arcsec match b/w GSWLC-X2-NO-DR10-AGN-Simard2011-tab1 and Tempel_gals_below_12.cat in topcat, best,symmetric
    #gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-12.5-2020Nov11'
    gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-13-2020Nov25'    
    if trimgswlc:
        #g = gswlc_full('/home/rfinn/research/GSWLC/GSWLC-X2.dat')
        # 10 arcsec match b/s GSWLC-X2 and Tempel-12.5_v_2 in topcat, best, symmetric
        #g = gswlc_full('/home/rfinn/research/LCS/tables/GSWLC-Tempel-12.5-v2.fits')
        #g = gswlc_full(homedir+'/research/GSWLC/GSWLC-Tempel-12.5-v2-Simard2011-NSAv0-unwise.fits',cutBT=args.cutBT)
        
        
        g = gswlc_full(gsw_basefile+'.fits',cutBT=args.cutBT)                
        g.cut_redshift()
        g.save_trimmed_cat()

    #g = gswlc('/home/rfinn/research/LCS/tables/GSWLC-X2-LCS-Zoverlap.fits')
    if args.cutBT:
        infile=gsw_basefile+'-LCS-Zoverlap-BTcut.fits'
    else:
        infile = gsw_basefile+'-LCS-Zoverlap.fits'
    g = gswlc(infile)

    #g.plot_ms()
    #g.plot_field1()
    lcsfile = homedir+'/research/LCS/tables/lcs-gswlc-x2-match.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS_all_size_KE_SFR_GSWLC2_X2.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS-KE-SFR-GSWLC-X2-NO-DR10-AGN-Simard2011-tab1.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS-GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-vizier-10arcsec.fits'        
    lcs = lcsgsw(lcsfile,cutBT=args.cutBT)
    #lcs = lcsgsw('/home/rfinn/research/LCS/tables/LCS_all_size_KE_SFR_GSWLC2_X2.fits',cutBT=args.cutBT)    
    #lcs.compare_sfrs()

    b = comp_lcs_gsw(lcs,g,minmstar=float(args.minmass),cutBT=args.cutBT)
