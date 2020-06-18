#!/usr/bin/env python
from pylab import *
import idlsave
import matplotlib.patches as mpatches

#for marking corners of the 24um image
MKW824um=array([220.16377,3.4883817,1.3137727,3.029165,12.7456],'f')
MKW1124um=array([202.36305,11.746882,1.2454248,3.0148808,206.4],'f')
NGC24um=array([244.30994,34.933704,1.2865442,3.029165,321.317],'f')

def drawbox(data,style):#feed in center x,y,dx,dy,rotation E of N
    #xcoords of unrotated box, going around CCW
    xl=array([data[0]-0.5*data[2],data[0]+0.5*data[2],data[0]+0.5*data[2],data[0]-0.5*data[2]],'d')
    yl=array([data[1]-0.5*data[3],data[1]-0.5*data[3],data[1]+0.5*data[3],data[1]+0.5*data[3] ],'d')

    ang=data[4]*pi/180.#convert rotation to radians
    #rotate coordinates
    xp=cos(ang)*xl-sin(ang)*yl
    yp=sin(ang)*xl+cos(ang)*yl
    #draw rotated box
    plot(xp,yp,style)


#Cluster coordinates for groups/clusters to be targeted with Halpha
rah=array([9,11,11,11,12,13,14,14,14,15,10,13,16],'f')
ram=array([25,42,47,58,24,29,05,28,40,5,6,53,17],'f')
ras=array([12.2,9.4,51.8,30.5,02.6,31.2,57.,1.9,49.1,32.,41.8,14.1,15.4],'f')

decd=array([11,10,13,25,9,11,9,25,3,1,14,25,34],'f')
decm=array([16,16,16,8,21,47,7,34,24,41,25,2,55],'f')
decs=array([49,30,33,23,25,19,58,10,11,48,49,31,0],'f')
groupra=(rah+ram/60.+ras/3600)*15.
groupdec=decd+decm/60.+decs/3600.

#Group names
groupnames=['NRGb041','NRGb151','NRGb157','NRGb168','NRGb206','NRGb247','NRGb282','NRGb301','MKW8','NCG5846','NRGs076','NRGs272','NRGs385']
altgroupnames=['WBL226','MKW10','HCG59','WBL368','WBL404','MKW11','Zw1400','WBL509','WBL518','NGC5846','WBL251','WBL477','NGC6107']
#drawboxflag=array([0,1,0,0,0,1,0,0,0,0,0,0,1],'bool')
groupvel=array([3616,6265,3745,4789,7256,7056,6872,4819,8100,1749,8988,9294,9477],'f')
groupsigma=array([208,257,457,162,457,269,660,302,325,322,257,204,525],'f')
grouplhalpha=6563.*(1+groupvel/3.e5)
groupfilter=[8,12,8,8,16,16,12,8,16,4,16,16,16]

filtercenter={4:6621,8:6654,12:6699,16:6731}
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

#change this to where your agc file is located
s=idlsave.read('/Users/rfinn/idl/programs/idl_alfa/agcnorth.sav')
#      Dtype=[(('agcnumber', 'AGCNUMBER'), '|O8'), (('which', 'WHICH'), '|O8'), (('rah', 'RAH'), '|O8'), (('ram', 'RAM'), '|O8'), (('ras10', 'RAS10'), '|O8'), (('sign', 'SIGN'), '|O8'), (('decd', 'DECD'), '|O8'), (('decm', 'DECM'), '|O8'), (('decs', 'DECS'), '|O8'), (('a100', 'A100'), '|O8'), (('b100', 'B100'), '|O8'), (('mag10', 'MAG10'), '|O8'), (('inccode', 'INCCODE'), '|O8'), (('posang', 'POSANG'), '|O8'), (('description', 'DESCRIPTION'), '|O8'), (('bsteintype', 'BSTEINTYPE'), '|O8'), (('vopt', 'VOPT'), '|O8'), (('verr', 'VERR'), '|O8'), (('extrc3', 'EXTRC3'), '|O8'), (('extdirbe', 'EXTDIRBE'), '|O8'), (('vsource', 'VSOURCE'), '|O8'), (('ngcic', 'NGCIC'), '|O8'), (('flux100', 'FLUX100'), '|O8'), (('rms100', 'RMS100'), '|O8'), (('v21', 'V21'), '|O8'), (('width', 'WIDTH'), '|O8'), (('widtherr', 'WIDTHERR'), '|O8'), (('widthcode', 'WIDTHCODE'), '|O8'), (('telcode', 'TELCODE'), '|O8'), (('detcode', 'DETCODE'), '|O8'), (('hisource', 'HISOURCE'), '|O8'), (('statuscode', 'STATUSCODE'), '|O8'), (('snratio', 'SNRATIO'), '|O8'), (('ibandqual', 'IBANDQUAL'), '|O8'), (('ibandsrc', 'IBANDSRC'), '|O8'), (('irasflag', 'IRASFLAG'), '|O8'), (('icluster', 'ICLUSTER'), '|O8'), (('hidata', 'HIDATA'), '|O8'), (('iposition', 'IPOSITION'), '|O8'), (('ipalomar', 'IPALOMAR'), '|O8'), (('rc3flag', 'RC3FLAG'), '|O8'), (('irotcat', 'IROTCAT'), '|O8'), (('newstuff', 'NEWSTUFF'), '|O8')])


ra=15.*(s.agcnorth.rah[0]+s.agcnorth.ram[0]/60.+s.agcnorth.ras10[0]/10./3600.)#Convert to degrees
dec=s.agcnorth.decd[0]+s.agcnorth.decm[0]/60.+s.agcnorth.decs[0]/3600.
agcvopt=s.agcnorth.vopt[0]
agcflux100=s.agcnorth.flux100[0]
agcmag10=s.agcnorth.mag10[0]

dr=5.#radial search in degrees
#offset of the field center in degrees, if desired
deltadec=array([0.,0,0,0,0,0,0,0,0,0,0,0,0],'f')
deltara=array([.0,0,0,0,0,0,0,0,0,0,0,0,0],'f')

def makeplot(i):
#for i in range(len(groupra)):
#for i in range(1):
    dist=sqrt((groupra[i]-ra)**2 + (groupdec[i]-dec)**2)
    flag = (dist < dr) & (agcvopt > groupfiltvmin[i]) & (agcvopt < groupfiltvmax[i])
    flagHI = (dist < dr) & (agcvopt > groupfiltvmin[i]) & (agcvopt < groupfiltvmax[i]) & (agcflux100 > 0.)
    flagmemb = (dist < dr) & (agcvopt > groupvmin[i]) & (agcvopt < groupvmax[i])
    figure()
    clf()
    #draw a box
    width,height=1.,1.
    xy=groupra[i]-0.5*width+deltara[i],groupdec[i]-0.5*height+deltadec[i]
    p=mpatches.Rectangle(xy,width,height,facecolor='1',edgecolor='k')
    ax=gca()

    xl=[groupra[i]-0.5*width+deltara[i],groupra[i]+0.5*width+deltara[i]]
    y1=groupdec[i]-0.5*height+deltadec[i]
    y2=groupdec[i]+0.5*height+deltadec[i]
    plot(xl,[y1,y1],'k')
    plot(xl,[y2,y2],'k')
    plot([xl[0],xl[0]],[y1,y2],'k')
    plot([xl[1],xl[1]],[y1,y2],'k')
    #ax.add_patch(p)
    #draw()

    scatter(ra[flag],dec[flag],s=(20-agcmag10[flag]/10)*20+20,color='.8')
    plot(ra[flagmemb],dec[flagmemb],'g^', alpha=0.3,markersize=14)
    plot(ra[flag],dec[flag],'k.')

    plot(ra[flagHI],dec[flagHI],'bo', markerfacecolor='1',markeredgecolor='b',markersize=8)
    plot(ra[flag],dec[flag],'k.')
    plot(groupra[i],groupdec[i],'rx',markersize=10,lw=3)#mark cluster center with a red x

    #draw footprint of mips data, if applicable
    if altgroupnames[i].find('MKW8') > -1:
        drawbox(MKW824um,'r-')
    if altgroupnames[i].find('MKW11') > -1:
        drawbox(MKW1124um,'r-')
    if altgroupnames[i].find('NGC6107') > -1:
        drawbox(NGC24um,'r-')


    title(groupnames[i],fontsize=24)
    xlabel('RA (deg)',fontsize=18)
    ylabel('Dec (deg)',fontsize=18)
    axis([groupra[i]+dr,groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
    axis('equal')
    #axis(groupra[i]+dr,[groupra[i]-dr,groupdec[i]-dr,groupdec[i]+dr])
    s=groupnames[i]+'.png'
    print i,s
    savefig(s)


for i in range(len(groupra)):
    makeplot(i)
