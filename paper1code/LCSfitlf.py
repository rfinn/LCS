#!/usr/bin/env python

from pylab import *
from scipy import *
import numpy as np

# define function for calculating luminosity function
# p[0] = phi*
# p[1] = L*
# p[2] = alph
schechter = lambda p, x,dx:p[0]*(x/p[1])**(-1*p[2])*exp(-1*x/p[1])#*dx/p[1]
errfunc= lambda p, x, dx, y, erry: (y - schecter(p,x,dx))**2/erry**2



def simdata():
    dx=5
    x=arange(5,100,dx)
    p=[1.,50.,1.41]
    y=schechter(p,x,dx)
    ysimerr=0.1*ones(len(y))
    
    # add noise to data
    yn=y+.02*np.random.normal(size=len(x))

    # plot sim data
    figure()
    plot(x,y,'k-')
    errorbar(x,yn,yerr=ysimerr)

    yexp=exp(-1*x/p[1])
    plot(x,yexp)

    #ypl=


def plotlirdist(x,label1):#label 1 = All for morph plot
	greyscale='0.5'
	xmin=10.25
	xmax=13.
	nbin=10
	x=pylab.log10(x)
	(xbin,ybin,ybinerr) = my.horizontalhist(x,xmin,xmax,nbin)
	dbin=xbin[1]-xbin[0]
	ybinerrm=pylab.zeros(len(ybinerr),'d')
	ybinerrp=pylab.zeros(len(ybinerr),'d')
	(yberrm,yberrp)=poisson.Poisson_lim(ybin)

	for i in range(len(ybin)):
		if ybin[i] > 0.:
			if yberrm[i] > 0.:
				ybinerrm[i]=pylab.log10(ybin[i]/dbin)-pylab.log10((yberrm[i])/dbin)
			else:
				ybinerrm[i]=999.
			ybinerrp[i]=pylab.log10((yberrp[i])/dbin)-pylab.log10(ybin[i]/dbin)
			ybin[i]=pylab.log10(ybin[i]/dbin)
		else:
			ybin[i]=-10.
	(xp,yp)=my.drawhistpylabnoplot(xbin,(ybin),'k-')
	pylab.plot(xp,yp,c=greyscale,ls='-')
	yerrall=pylab.zeros([2,len(ybinerrm)],'f')
	yerrall[0]=ybinerrm
	yerrall[1]=ybinerrp
	pylab.errorbar(xbin+dbin/2.,ybin,yerr=yerrall,fmt=None,ecolor='0.5')
	pylab.axis([9.8,12.4,0.,3.4])
	ax=pylab.gca()
	s=label1
	pylab.text(.7,.9,s,fontsize=18,color=greyscale,transform=ax.transAxes)
	pylab.text(.7,.8,label2,fontsize=18,color='k',transform=ax.transAxes)


