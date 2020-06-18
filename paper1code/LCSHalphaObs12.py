#!/usr/bin/env python

'''
GOAL:
  To plot the positions of galaxies in AGC relative to group centers.
  This will help us to determine best positions for Halpha observations.

PROCEDURE:
  Read in agctotal
  plot positions

  group information is entered in arrays at beginning of code

USEAGE:
  in ipython,
  >%run LCSHalphaObs11.py
  > makeplot(19) # for example, this makes the plot for MKW8

REQUIRED MODULES:
  pylab
  scipy
  os
  numpy
  
REQUIRED FILES:
  agctotal.sav

'''



from pylab import *
from scipy.io.idl import readsav
import os
import matplotlib.patches as mpatches
#import coords as C
from numpy import*

#for marking corners of the 24um image

#these represent full coverage of 24um scan, but S/N is max for delta Dec = 2.5
#MKW824um=array([220.16377,3.4883817,1.3137727,3.029165,12.7456],'f')
#MKW1124um=array([202.36305,11.746882,1.2454248,3.0148808,206.4],'f')
#NGC24um=array([244.30994,34.933704,1.2865442,3.029165,321.317],'f')

#these correpond to area w/more uniform covereage
#MKW824um=array([220.16377,3.4883817,1.3137727,2.5,12.7456],'f')
#MKW1124um=array([202.36305,11.746882,1.2454248,2.5,206.4],'f')
#NGC24um=array([244.30994,34.933704,1.2865442,2.5,321.317],'f')

MKW824um=array([220.16377,3.4883817,1.3137727,2.5,102.7456],'f')
MKW1124um=array([202.36305,11.746882,1.2454248,2.5,116.4],'f')
NGC24um=array([244.30994,34.933704,1.2865442,2.5,231.317],'f')
A136724um=array([176.1019,19.799614,.51080152,.90025557,31.5],'f')
Coma24um=array([194.86318,27.865896,1.5391027,1.976467,29.5002],'f')


def drawbox(data,style):#feed in center x,y,dx,dy,rotation E of N
    #xcoords of unrotated box, going around CCW
    xl=array([data[0]-0.5*data[2],data[0]+0.5*data[2],data[0]+0.5*data[2],data[0]-0.5*data[2],data[0]-0.5*data[2]],'d')
    yl=array([data[1]-0.5*data[3],data[1]-0.5*data[3],data[1]+0.5*data[3],data[1]+0.5*data[3],data[1]-0.5*data[3] ],'d')

    xl=array([-0.5*data[2],+0.5*data[2],+0.5*data[2],-0.5*data[2],-0.5*data[2]],'d')
    yl=array([-0.5*data[3],-0.5*data[3],+0.5*data[3],+0.5*data[3],-0.5*data[3] ],'d')

    ang=data[4]*pi/180.#convert rotation to radians
    #rotate coordinates
    xp=cos(ang)*xl-sin(ang)*yl
    yp=sin(ang)*xl+cos(ang)*yl

    #put back on absolute scale
    xp=data[0]+xp
    yp=data[1]+yp
    #draw rotated box
    plot(xp,yp,style)

#Cluster coordinates for groups/clusters to be targeted with Halpha
namec,rahc,ramc,rasc,decdc,decmc,decsc=loadtxt('group_center',unpack=True)
grouprac=(rahc+ramc/60.+rasc/3600)*15.
groupdecc=decdc+decmc/60.+decsc/3600.

name,rah,ram,ras,decd,decm,decs=loadtxt('group_field1',unpack=True)
groupra=(rah+ram/60.+ras/3600)*15.
groupdec=decd+decm/60.+decs/3600.


name2,rah2,ram2,ras2,decd2,decm2,decs2=loadtxt('group_field2',unpack=True)
groupra2=(rah2+ram2/60.+ras2/3600)*15.
groupdec2=decd2+decm2/60.+decs2/3600.

name3,rah3,ram3,ras3,decd3,decm3,decs3=loadtxt('group_field3',unpack=True)
groupra3=(rah3+ram3/60.+ras3/3600)*15.
groupdec3=decd3+decm3/60.+decs3/3600.

name4,rah4,ram4,ras4,decd4,decm4,decs4=loadtxt('group_field4',unpack=True)
groupra4=(rah4+ram4/60.+ras4/3600)*15.
groupdec4=decd4+decm4/60.+decs4/3600.

name5,rah5,ram5,ras5,decd5,decm5,decs5=loadtxt('group_field5',unpack=True)
groupra5=(rah5+ram5/60.+ras5/3600)*15.
groupdec5=decd5+decm5/60.+decs5/3600.

name,rahh1,ramh1,rash1,decdh1,decmh1,decsh1=loadtxt('group_hfield1',unpack=True)
grouprah1=(rahh1+ramh1/60.+rash1/3600)*15.
groupdech1=decdh1+decmh1/60.+decsh1/3600.

name,rahh2,ramh2,rash2,decdh2,decmh2,decsh2=loadtxt('group_hfield2',unpack=True)
grouprah2=(rahh2+ramh2/60.+rash2/3600)*15.
groupdech2=decdh2+decmh2/60.+decsh2/3600.

#Group names
groupnames=['A1367','Coma','NRGb190','NRGs027','NRGb161','NRGb004','NRGb137','NRGb302','NRGb049','NRGb177','NRGb025','NRGb286','NRGb310','NRGb041','NRGb151','NRGb157','NRGb168','NRGb206','NRGb247','NRGb282','NRGb301','MKW8','NCG5846','NRGs076','NRGs272','NRGs385','NRGb032','NRGb054','NRGb331']#,NGC5746','NGC5364','NGC5775','NGC 5673']
altgroupnames=['NRGb155','NRGb226','NRGb190','NRGs027','NRGb161','NRGb004','NRGb137','NRGb302','NRGb049','NRGb177','NRGb025','NRGb286','NRGb310','WBL226','MKW10','HCG59','WBL368','WBL404','MKW11','Zw1400','WBL509','MKW8','NGC5846','WBL251','WBL477','NGC6107','A779','','WBL 543']#,'NGC5746','NGC5364','NGC5775','NGC 5673']
groupvel=array([6505,7011,7128,8683,7168,8343,6421,8151,8445,7079,6785,5062,4514,3616,6265,3745,4789,7256,7056,6872,4819,8100,1749,8988,9294,9477,7473,6460,6904],'f')#,1651,1108,1749,2150],'f')
groupsigma=array([838,1054,95,562,617,603,69,380,478,421,223,316,83,208,257,457,162,457,269,660,302,325,322,257,204,525,360,145,69],'f')#,200,200,300,200],'f')
groupdist=array([95.8,104,102,124,102,119,92,116,125,106,97,72,65,55,94,56,71,108,105,102,72,119,26.1,137,139,143,110,97,101],'f')
grouplhalpha=6563.*(1+groupvel/3.e5)

groupfilter=[12,16,12,16,12,16,12,16,16,12,12,8,8,8,12,8,8,16,16,12,8,16,4,16,16,16,16,12,12]#,4,4,4,4]

print grouplhalpha,groupfilter

filtercenter={4:6618,8:6658,12:6700,16:6742}
filterwidth=60.
groupfiltcenter=zeros(len(groupfilter))
groupfiltvcenter=zeros(len(groupfilter))
groupfiltvmin=zeros(len(groupfilter))
groupfiltvmax=zeros(len(groupfilter))

groupvmin=groupvel-1000
groupvmax=groupvel+1000
groupvmin=groupvel-3.*groupsigma
groupvmax=groupvel+3.*groupsigma
#calculate vcenter, vmin and vmax for each group according to the filter we will observe it through
for i in range(len(groupfiltcenter)):
    groupfiltcenter[i]=filtercenter[groupfilter[i]]
    groupfiltvcenter[i]=(groupfiltcenter[i]/6563. -1)*3.e5
    groupfiltvmin[i]=((groupfiltcenter[i]-0.5*filterwidth)/6563. -1)*3.e5
    groupfiltvmax[i]=((groupfiltcenter[i]+0.5*filterwidth)/6563. -1)*3.e5

#Get current path so program can tell if this is being run on Becky or Rose's computer
mypath=os.getcwd()
if mypath.find('rfinn') > -1:
    print "Running on Rose's computer"
    #agcfile='/Users/rfinn/idl/programs/idl_alfa/agctotal.sav'
    agcfile='/Users/rfinn/idl/alfalfa/agctotal.sav'
elif mypath.find('koopmanr') > -1:
    print "Running on Becky's computer"
    agcfile='/Users/koopmanr/alfalfa/idl_alfa/agctotal.sav'
#s=idlsave.read(agcfile)
s=readsav(agcfile)


#change this to where your agc file is located


#s=idlsave.read('/Users/koopmanr/idl_alfa/agctotal.sav')

#      Dtype=[(('agcnumber', 'AGCNUMBER'), '|O8'), (('which', 'WHICH'), '|O8'), (('rah', 'RAH'), '|O8'), (('ram', 'RAM'), '|O8'), (('ras10', 'RAS10'), '|O8'), (('sign', 'SIGN'), '|O8'), (('decd', 'DECD'), '|O8'), (('decm', 'DECM'), '|O8'), (('decs', 'DECS'), '|O8'), (('a100', 'A100'), '|O8'), (('b100', 'B100'), '|O8'), (('mag10', 'MAG10'), '|O8'), (('inccode', 'INCCODE'), '|O8'), (('posang', 'POSANG'), '|O8'), (('description', 'DESCRIPTION'), '|O8'), (('bsteintype', 'BSTEINTYPE'), '|O8'), (('vopt', 'VOPT'), '|O8'), (('verr', 'VERR'), '|O8'), (('extrc3', 'EXTRC3'), '|O8'), (('extdirbe', 'EXTDIRBE'), '|O8'), (('vsource', 'VSOURCE'), '|O8'), (('ngcic', 'NGCIC'), '|O8'), (('flux100', 'FLUX100'), '|O8'), (('rms100', 'RMS100'), '|O8'), (('v21', 'V21'), '|O8'), (('width', 'WIDTH'), '|O8'), (('widtherr', 'WIDTHERR'), '|O8'), (('widthcode', 'WIDTHCODE'), '|O8'), (('telcode', 'TELCODE'), '|O8'), (('detcode', 'DETCODE'), '|O8'), (('hisource', 'HISOURCE'), '|O8'), (('statuscode', 'STATUSCODE'), '|O8'), (('snratio', 'SNRATIO'), '|O8'), (('ibandqual', 'IBANDQUAL'), '|O8'), (('ibandsrc', 'IBANDSRC'), '|O8'), (('irasflag', 'IRASFLAG'), '|O8'), (('icluster', 'ICLUSTER'), '|O8'), (('hidata', 'HIDATA'), '|O8'), (('iposition', 'IPOSITION'), '|O8'), (('ipalomar', 'IPALOMAR'), '|O8'), (('rc3flag', 'RC3FLAG'), '|O8'), (('irotcat', 'IROTCAT'), '|O8'), (('newstuff', 'NEWSTUFF'), '|O8')])


ra=15.*(s.agctotal.rah[0]+s.agctotal.ram[0]/60.+s.agctotal.ras10[0]/10./3600.)#Convert to degrees
dec=s.agctotal.decd[0]+s.agctotal.decm[0]/60.+s.agctotal.decs[0]/3600.
agcvopt=s.agctotal.vopt[0]
agcflux100=s.agctotal.flux100[0]
agcmag10=s.agctotal.mag10[0]
agctype=s.agctotal.bsteintype[0]



dr=3.#radial search in degrees
#offset of the field center in degrees, if desired
deltadec=array([0,0,0,0,0,0,0,0,0.,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'f')
deltara=array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'f')

def makeplot(i,makesubplots=True,dr=3.):
    dist=sqrt((grouprac[i]-ra)**2 + (groupdecc[i]-dec)**2)    
    absmag=agcmag10/10-5*log10(groupdist[i]*1e6)+5
    flag = (dist < dr) & (agcvopt > groupfiltvmin[i]) & (agcvopt < groupfiltvmax[i])
    flagHI = (dist < dr) & (agcvopt > groupfiltvmin[i]) & (agcvopt < groupfiltvmax[i]) & (agcflux100 > 0.)
    flagmemb = (dist < dr) & (agcvopt > groupvmin[i]) & (agcvopt < groupvmax[i]) & flag
    flagspiral=(dist < dr) & (agctype > 99) & (agctype < 183) & (absmag < -18) & (agcvopt > groupfiltvmin[i]) & (agcvopt < groupfiltvmax[i])
    flagspiral2=(dist < dr) & (agctype > 299) & (agctype < 313) & (absmag < -18) & (agcvopt > groupfiltvmin[i]) & (agcvopt < groupfiltvmax[i])
    if makesubplots:
        figure(figsize=(12,5))
        subplot(1,2,1)
        subplots_adjust(bottom=.15)
    else:
        figure()
        clf()

    #draw a box
    if groupra[i] > 0: 
     width,height=1.,1.
     xy=groupra[i]-0.5*width+deltara[i],groupdec[i]-0.5*height+deltadec[i]
     p=mpatches.Rectangle(xy,width,height,facecolor='1',edgecolor='k')
     ax=gca()

   #draw box near center
     xl=[groupra[i]-0.5*width+deltara[i],groupra[i]+0.5*width+deltara[i]]
     y1=groupdec[i]-0.5*height+deltadec[i]
     y2=groupdec[i]+0.5*height+deltadec[i]
     plot(xl,[y1,y1],'k')
     plot(xl,[y2,y2],'k')
     plot([xl[0],xl[0]],[y1,y2],'k')
     plot([xl[1],xl[1]],[y1,y2],'k')
     text(xl[0]-0.05,y1-0.05,'1',fontsize=16)
    #ax.add_patch(p)
    #draw()

   #draw a second box
    if groupra2[i] > 0.1: 
     width,height=1.,1.
     xy=groupra2[i]-0.5*width+deltara[i],groupdec2[i]-0.5*height+deltadec[i]
     p=mpatches.Rectangle(xy,width,height,facecolor='1',edgecolor='k')
     ax=gca()

     xl=[groupra2[i]-0.5*width+deltara[i],groupra2[i]+0.5*width+deltara[i]]
     y1=groupdec2[i]-0.5*height+deltadec[i]
     y2=groupdec2[i]+0.5*height+deltadec[i]
     plot(xl,[y1,y1],'k')
     plot(xl,[y2,y2],'k')
     plot([xl[0],xl[0]],[y1,y2],'k')
     plot([xl[1],xl[1]],[y1,y2],'k')
     text(xl[0]-0.05,y1-0.05,'2',fontsize=16)
    #ax.add_patch(p)
    #draw()

   #draw a third box
    if groupra3[i] > 0.1: 
     width,height=1.,1.
     xy=groupra3[i]-0.5*width+deltara[i],groupdec3[i]-0.5*height+deltadec[i]
     p=mpatches.Rectangle(xy,width,height,facecolor='1',edgecolor='k')
     ax=gca()

     xl=[groupra3[i]-0.5*width+deltara[i],groupra3[i]+0.5*width+deltara[i]]
     y1=groupdec3[i]-0.5*height+deltadec[i]
     y2=groupdec3[i]+0.5*height+deltadec[i]
     plot(xl,[y1,y1],'k')
     plot(xl,[y2,y2],'k')
     plot([xl[0],xl[0]],[y1,y2],'k')
     plot([xl[1],xl[1]],[y1,y2],'k')
     text(xl[0]-0.05,y1-0.05,'3',fontsize=16)
    #ax.add_patch(p)
    #draw()

   #draw a fourth box
    if groupra4[i] > 0: 
     width,height=1.,1.
     xy=groupra4[i]-0.5*width+deltara[i],groupdec4[i]-0.5*height+deltadec[i]
     p=mpatches.Rectangle(xy,width,height,facecolor='1',edgecolor='k')
     ax=gca()

     xl=[groupra4[i]-0.5*width+deltara[i],groupra4[i]+0.5*width+deltara[i]]
     y1=groupdec4[i]-0.5*height+deltadec[i]
     y2=groupdec4[i]+0.5*height+deltadec[i]
     plot(xl,[y1,y1],'k')
     plot(xl,[y2,y2],'k')
     plot([xl[0],xl[0]],[y1,y2],'k')
     plot([xl[1],xl[1]],[y1,y2],'k')
     text(xl[0]-0.05,y1-0.05,'4',fontsize=16)
     #ax.add_patch(p)
    #draw()

  #draw a fifth box
    if groupra5[i] > 0: 
     width,height=1.,1.
     xy=groupra5[i]-0.5*width+deltara[i],groupdec5[i]-0.5*height+deltadec[i]
     p=mpatches.Rectangle(xy,width,height,facecolor='1',edgecolor='k')
     ax=gca()

     xl=[groupra5[i]-0.5*width+deltara[i],groupra5[i]+0.5*width+deltara[i]]
     y1=groupdec5[i]-0.5*height+deltadec[i]
     y2=groupdec5[i]+0.5*height+deltadec[i]
     plot(xl,[y1,y1],'k')
     plot(xl,[y2,y2],'k')
     plot([xl[0],xl[0]],[y1,y2],'k')
     plot([xl[1],xl[1]],[y1,y2],'k')
     text(xl[0]-0.05,y1-0.05,'5',fontsize=16)
     #ax.add_patch(p)
    #draw()

   #draw first HDI box
    if grouprah1[i] > 0: 
     widthh,heighth=0.487,0.487
     xy=grouprah1[i]-0.5*widthh+deltara[i],groupdech1[i]-0.5*heighth+deltadec[i]
     p=mpatches.Rectangle(xy,widthh,heighth,facecolor='1',edgecolor='c')
     ax=gca()

     xl=[grouprah1[i]-0.5*widthh+deltara[i],grouprah1[i]+0.5*widthh+deltara[i]]
     y1=groupdech1[i]-0.5*heighth+deltadec[i]
     y2=groupdech1[i]+0.5*heighth+deltadec[i]
     plot(xl,[y1,y1],'r')
     plot(xl,[y2,y2],'r')
     plot([xl[0],xl[0]],[y1,y2],'r')
     plot([xl[1],xl[1]],[y1,y2],'r')
     text(xl[0]-0.05,y1-0.05,'H1',fontsize=16)


   #draw second HDI box
    if grouprah2[i] > 0: 
     widthh,heighth=0.487,0.487
     xy=grouprah2[i]-0.5*widthh+deltara[i],groupdech2[i]-0.5*heighth+deltadec[i]
     p=mpatches.Rectangle(xy,widthh,heighth,facecolor='1',edgecolor='c')
     ax=gca()

     xl=[grouprah2[i]-0.5*widthh+deltara[i],grouprah2[i]+0.5*widthh+deltara[i]]
     y1=groupdech2[i]-0.5*heighth+deltadec[i]
     y2=groupdech2[i]+0.5*heighth+deltadec[i]
     plot(xl,[y1,y1],'r')
     plot(xl,[y2,y2],'r')
     plot([xl[0],xl[0]],[y1,y2],'r')
     plot([xl[1],xl[1]],[y1,y2],'r')
     text(xl[0]-0.05,y1-0.05,'H2',fontsize=16)

    #draw footprint of mips data, if applicable
    if altgroupnames[i].find('MKW8') > -1:
        drawbox(MKW824um,'r-')
    if altgroupnames[i].find('MKW11') > -1:
        drawbox(MKW1124um,'r-')
    if altgroupnames[i].find('NGC6107') > -1:
        drawbox(NGC24um,'r-')
    if altgroupnames[i].find('Coma') > -1:
        drawbox(Coma24um,'r-')
    if altgroupnames[i].find('A1367') > -1:
        drawbox(A136724um,'r-')


    scatter(ra[flag],dec[flag],s=(20-agcmag10[flag]/10)*20+20,color='.8')
    plot(ra[flagmemb],dec[flagmemb],'g^', alpha=0.3,markersize=14)
    plot(ra[flagspiral],dec[flagspiral],'r*',alpha=0.3,markersize=14)
    plot(ra[flagspiral2],dec[flagspiral2],'r*',alpha=0.3,markersize=14)
    plot(ra[flag],dec[flag],'k.')


    plot(ra[flagHI],dec[flagHI],'bo', markerfacecolor='None',markeredgecolor='b',markersize=8)
    plot(ra[flag],dec[flag],'k.')
    plot(grouprac[i],groupdecc[i],'rx',markersize=10,lw=3)#mark cluster center with a red x


    mytitle=groupnames[i]+', '+altgroupnames[i]
    title(mytitle,fontsize=24)
    xlabel('RA (deg)',fontsize=18)
    ylabel('Dec (deg)',fontsize=18)
    axis([grouprac[i]+dr,grouprac[i]-dr,groupdecc[i]-dr,groupdecc[i]+dr])
    axis('equal')
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
    s=groupnames[i]+'.png'
    print i,s
    if makesubplots:
        subplot(1,2,2)
    else:
        savefig(s)
        figure()    
    # make a histogram 
    hist(agcvopt[flagmemb],bins=20)
    axvline(x=groupvel[i],color='r')
    axvline(x=groupfiltvmax[i],color='r',ls='--')
    axvline(x=groupfiltvmin[i],color='r',ls='--')
    if subplots:
        savefig(s)
        
def plotall():
    for i in range(len(grouprac)):
        makeplot(i)
