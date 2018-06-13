#!/usr/bin/env python

'''
GOAL:
- this code contains all of the code to make figures for paper1


REQUIRED MODULES
- LCSbase.py



'''

import LCSbase as lb
from matplotlib import pyplot as plt
import numpy as np

class galaxies(lb.galaxies):
    def plotsizedvdr(self,plotsingle=1,reonly=1,onlycoma=0,plotHI=0,plotbadfits=0,lowmass=0,himass=0,cluster=None,plothexbin=True,hexbinmax=40,scalepoint=0,clustername=None,blueflag=False,plotmembcut=True,colormin=.2,colormax=1,colorbydensity=False,plotoman=False,masscut=None,BTcut=None):
        # log10(chabrier) = log10(Salpeter) - .25 (SFR estimate)
        # log10(chabrier) = log10(diet Salpeter) - 0.1 (Stellar mass estimates)

        if plotsingle:
            figure(figsize=(10,6))
            ax=gca()
            subplots_adjust(left=.1,bottom=.15,top=.9,right=.9)
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            #axis([1.e9,1.e12,5.e-14,5.e-10])
            #axis([9,12,-14.5,-10.5])
            ylabel('$ \Delta v/\sigma $',fontsize=26)
            xlabel('$ \Delta R/R_{200}  $',fontsize=26)
            legend(loc='upper left',numpoints=1)

        colors=self.sizeratio
        if colorbydensity:
            colors=np.log10(self.s.SIGMA_5)
            colormin=-1.5
            colormax=1.5
        cbticks=arange(colormin,colormax+.1,.1)
        if USE_DISK_ONLY:
            clabel=['$R_{24}/R_d$','$R_{iso}(24)/R_{iso}(r)$']
        else:
            clabel=['$R_e(24)/R_e(r)$','$R_{iso}(24)/R_{iso}(r)$']
        cmaps=['jet_r','jet_r']

        v1=[0.2,0.]
        v2=[1.2,2]
        nplot=1
        
        x=(self.s.DR_R200)
        y=abs(self.dv)
        flag=self.sampleflag #& self.dvflag
        if blueflag:
            flag=self.bluesampleflag & self.dvflag
        if clustername != None:
            flag = flag & (self.s.CLUSTER == clustername)
        #flag=self.limitedsample & self.dvflag
        if masscut != None:
            flag = flag & (self.logstellarmass < masscut)
        if BTcut != None:
            flag = flag & (self.gim2d.B_T_r < 0.3)
        if cluster != None:
            flag = flag & (self.s.CLUSTER == cluster)
        hexflag=self.dvflag
        if cluster != None:
            hexflag = hexflag & (self.s.CLUSTER == cluster)
        nofitflag = self.sfsampleflag & ~self.sampleflag & self.dvflag
        nofitflag = self.gim2dflag & (self.gim2d.B_T_r < .2) & self.sfsampleflag & ~self.sampleflag & self.dvflag 
        if cluster != None:
            nofitflag = nofitflag & (self.s.CLUSTER == cluster)
        if lowmass:
            flag = flag & (self.s.CLUSTER_LX < 1.)
            hexflag = hexflag & (self.s.CLUSTER_LX < 1.)
            nofitflag = nofitflag & (self.s.CLUSTER_LX < 1.)
        if himass:
            flag = flag & (self.s.CLUSTER_LX > 1.)
            hexflag = hexflag & (self.s.CLUSTER_LX > 1.)
            nofitflag = nofitflag & (self.s.CLUSTER_LX > 1.)
        if onlycoma:
            flag = flag & (self.s.CLUSTER == 'Coma')
        if plothexbin:
            sp=hexbin(x[hexflag],y[hexflag],gridsize=(30,20),alpha=.7,extent=(0,5,0,10),cmap='gray_r',vmin=0,vmax=hexbinmax)#,C=colors[flag],vmin=v1[i],vmax=v2[i],cmap=cmaps[i],gridsize=5,alpha=0.5,extent=(0,3.,0,2))        
        #flags=[self.sampleflag  & self.dvflag ,self.sampleflag & self.agnflag]
        subplots_adjust(bottom=.15,left=.1,right=.95,top=.95,hspace=.02,wspace=.02)
        #xl=np.array([1.4,.35])
        #yl=np.array([3.0,1.2])
        #plt.plot(xl,yl,'k--',lw=2)
        if plotmembcut:
            xl=np.array([-.2,1,1])
            yl=np.array([3,3,-0.1])
            plt.plot(xl,yl,'k-',lw=2)
        elif plotoman: # line to identify infall galaxies from Oman+2013
            xl=np.arange(0,2,.1)
            plt.plot(xl,-4./3.*xl+2,'k-',lw=3)
            #plt.plot(xl,-3./1.2*xl+3,'k-',lw=3)       
        else: # cut from Jaffe+2011
            xl=np.array([0.01,1.2])
            yl=np.array([1.5,0])
            plt.plot(xl,yl,'k-',lw=2)

        if reonly:
            nplot=1
        else:
            nplot=2
        if scalepoint:
            size=(self.ssfrms[flag]+2)*40
        else:
            size=60
        for i in range(nplot):
            if not(reonly):
                subplot(1,2,nplot)
            nplot +=1
            if plotbadfits:
                scatter(x[nofitflag],y[nofitflag],marker='x',color='k',s=40)#markersize=8,mec='r',mfc='None',label='No Fit')

            ax=gca()
            #flag=flags[i]
            #sp=hexbin(x[flag],y[flag],C=colors[flag],vmin=v1[i],vmax=v2[i],cmap=cmaps[i],gridsize=5,alpha=0.5,extent=(0,3.,0,2))
            if colorbydensity:
                sp=scatter(x[flag],y[flag],c=colors[flag],s=size,cmap=cm.jet,vmin=colormin,vmax=colormax,edgecolors=None,lw=0.)
            else:
                sp=scatter(x[flag],y[flag],c=colors[flag],s=size,cmap=cm.jet_r,vmin=colormin,vmax=colormax,edgecolors=None,lw=0.)
            #sp=scatter(x[flag],y[flag],c=colors[flag],s=size,cmap=cm.jet_r,vmin=colormin,vmax=colormax)
            #print len(x[flag])
            #flag2=self.spiralflag & ~self.sampleflag & self.dvflag & ~self.agnflag
            #plot(x[flag2],y[flag2],'ko',color='0.5')
            #flag2=flag & self.truncflag
            #scatter(x[flag2],y[flag2]-x[flag2],c=colors[i][flag2],marker='*',s=120)
            #scatter(x[self.sampleflag],y[self.outertruncflag]-x[self.outertruncflag],c=colors[i][self.outertruncflag],marker='*',s=120)
            #axhline(y=0,ls='-',color='k')
            axis([-.1,4.5,-.1,5])
            if masscut != None:
                axis([-.1,4.5,-.1,4])
            if i > 0:
                ax.set_yticklabels(([]))
            ax.tick_params(axis='both', which='major', labelsize=16)
            #ax.set_xscale('log')
            #axins1 = inset_axes(ax,
            #        width="5%", # width = 10% of parent_bbox width
            #        height="50%", # height : 50%
            #        bbox_to_anchor=(.9,0.05,1,1),
            #        bbox_transform=ax.transAxes,
            #        borderpad=0,
            #        loc=3)
            if plotsingle:
                cb=colorbar(sp,fraction=0.08,label=clabel[i],ticks=cbticks)#cax=axins1,ticks=cbticks[i])
                #text(.95,.9,clabel[i],transform=ax.transAxes,horizontalalignment='right',fontsize=20)
            if plotHI:
                f=flag & self.HIflag
                plt.plot(x[f],y[f],'bs',mfc='None',mec='b',lw=2,markersize=20)

        if not(reonly):
            ax.text(0,-.1,'$ \Delta R/R_{200} $',fontsize=22,transform=ax.transAxes,horizontalalignment='center')
            ax.text(-1.3,.5,'$\Delta v/\sigma_v $',fontsize=22,transform=ax.transAxes,rotation=90,verticalalignment='center')

        if lowmass:
            figname=homedir+'research/LocalClusters/SamplePlots/sizedvdr-lowLx'
        elif himass:
            figname=homedir+'research/LocalClusters/SamplePlots/sizedvdr-hiLx'
        else:
            figname=homedir+'research/LocalClusters/SamplePlots/sizedvdr'
        if plotsingle:
            if masscut != None:
                plt.savefig(figuredir+'sizedvdr-lowmass-lowBT.eps')
            savefig(figname+'.png')
            savefig(figname+'.eps')
            plt.savefig(figuredir+'fig4.pdf')
    def plotsizehist(self, btcut = None,colorflag=False):
        figure(figsize=(6,6))
        plt.subplots_adjust(left=.15,bottom=.2,hspace=.1)
        axes=[]
        plt.subplot(2,1,1)

        axes.append(plt.gca())

        mybins=arange(0,2,.15)
        if btcut == None:
            flag = self.sampleflag
        else:
            flag = self.sampleflag & (self.gim2d.B_T_r < btcut)
        if colorflag:
            colors = ['r','b']
        else:
            colors = ['k','k']
        flags = [flag & self.membflag & ~self.agnflag,flag & ~self.membflag & ~self.agnflag]
        labels = ['$Core$','$External$']
        for i in range(len(colors)):
            plt.subplot(2,1,i+1)
            print 'median ratio for ',labels[i],' = ',np.median(self.sizeratio[flags[i]])
            hist(self.sizeratio[flags[i]],bins=mybins,histtype='stepfilled',color=colors[i],label=labels[i],lw=1.5,alpha=1)#,normed=True)
            plt.legend(loc='upper right')
            plt.axis([0,2,0,22])
            if i < 1:
                plt.xticks(([]))

        
        plt.text(-.2,1,'$N_{gal}$',transform=gca().transAxes,verticalalignment='center',rotation=90,fontsize=24)
        print 'comparing cluster and exterior SF galaxies'
        ks(self.sizeratio[flag & self.membflag & ~self.agnflag],self.sizeratio[flag & ~self.membflag & ~self.agnflag])
        
        plt.xlabel('$ R_{24}/R_d $')
        if btcut == None:
            #plt.ylim(0,20)
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblue.eps')
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblue.png')
            plt.savefig(figuredir+'fig11a.eps')
            
        else:
            #plt.ylim(0,15)
            plt.subplot(2,1,1)
            plt.title('$ B/T < %2.1f \ Galaxies $'%(btcut),fontsize=20)
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblueBTcut.eps')
            #plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizehistblueBTcut.png')
            plt.savefig(figuredir+'fig11b.eps')
    def plotsize3panel(self,logyscale=False,use_median=True,equal_pop_bins=True):
        figure(figsize=(10,10))
        subplots_adjust(left=.12,bottom=.1,top=.9,wspace=.02,hspace=.4)


        nrow=3
        ncol=3        
        flags=[self.sampleflag, self.sampleflag & self.membflag, self.sampleflag & ~self.membflag]
        flags = flags & (self.s.SIGMA_5 > 0.)

        x=[self.gim2d.B_T_r,np.log10(self.s.SIGMA_5),self.logstellarmass]
        xlabels=['$B/T$','$\log_{10}(\Sigma_5 \ (gal/Mpc^2))$','$\log_{10}(M_\star/M_\odot)$']
        colors=[self.logstellarmass,self.gim2d.B_T_r,self.gim2d.B_T_r]
        cblabel=['$\log(M_\star/M_\odot)$','$B/T$','$B/T$']
        cbticks=[np.arange(8.5,10.8,.4),np.arange(0,1,.2),np.arange(0,1,.2)]
        xticklabels=[np.arange(0,1,.2),np.arange(-1.2,2.2,1),np.arange(8.5,11.5,1)]
        xlims=[(-.05,.9),(-1.1,1.9),(8.3,11.2)]
        v1 = [8.5,0,0]
        v2 = [10.8,0.6,0.6]
        y=self.sizeratio
        yerror=self.sizeratioERR
        

        for i in range(len(x)):
            allax=[]
           
            for j in range(3):
                plt.subplot(nrow,ncol,3.*i+j+1)
                plt.errorbar(x[i][flags[j]],y[flags[j]],yerr=yerror[flags[j]],fmt=None,ecolor='.5',markerfacecolor='white',zorder=1,alpha=.5)
                sp=plt.scatter(x[i][flags[j]],y[flags[j]],c=colors[i][flags[j]],vmin=v1[i],vmax=v2[i],cmap='jet',s=40,label='GALFIT',lw=0,alpha=0.7,zorder=1)
                if j < 3:
                    (rho,p)=spearman_with_errors(x[i][flags[j]],y[flags[j]],yerror[flags[j]])
                    ax=plt.gca()
                    plt.text(.95,.9,r'$\rho = [%4.2f, %4.2f]$'%(np.percentile(rho,16),np.percentile(rho,84)),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
                    plt.text(.95,.8,'$p = [%5.4f, %5.4f]$'%(np.percentile(p,16),np.percentile(p,84)),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
                a=plt.gca()
                #plt.axis(limits)
                allax.append(a)
                if j > 0:
                    a.set_yticklabels(([]))
                if i == 0:
                    if j == 0:
                        plt.title('$All $',fontsize=24)
                    elif j == 1:
                        plt.title('$Core$',fontsize=24)
                        
                    elif j == 2:
                        plt.title('$External$',fontsize=24)
                if j == 1:
                    plt.xlabel(xlabels[i])
                if j == 0:
                    #plt.ylabel('$R_e(24)/Re(r)$')
                    plt.ylabel('$R_{24}/R_d$')
                xbin,ybin,ybinerr, colorbin = binxycolor(x[i][flags[j]],y[flags[j]],colors[i][flags[j]],nbin=5,erry=True,equal_pop_bins=equal_pop_bins,use_median = use_median)
                plt.scatter(xbin,ybin,c=colorbin,s=180,vmin=v1[i],vmax=v2[i],cmap='jet',zorder=5,lw=2,edgecolors='k')
                plt.errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',alpha=0.7)
                if logyscale:
                    a.set_yscale('log')
                    ylim(.08,6)
                else:
                    ylim(-.1,3.3)
                    yticks((np.arange(0,4,1)))
                xticks(xticklabels[i])
                xlim(xlims[i])
                #ylim(-.1,2.8)
                if j == 2:
                    c = np.polyfit(xbin,ybin,1)
                    print 'xbin = ', xbin
                    print 'ybin = ', ybin
                    #c = np.polyfit(x[i][flags[j]],y[flags[j]],1)
                    xl=np.linspace(min(x[i][flags[j]]),max(x[i][flags[j]]),10)
                    yl = np.polyval(c,xl)
                    plt.plot(xl,yl,'k--',lw=2)
                    plt.subplot(nrow,ncol,3.*i+j)
                    xl=np.linspace(min(x[i][flags[j-1]]),max(x[i][flags[j-1]]),10)
                    yl = np.polyval(c,xl)
                    plt.plot(xl,yl,'k--',lw=2)
                    #print xbin,ybin,colorbin
                

        
            #if i == 2:
            #    #text(0.1,0.9,'$External$',transform=a.transAxes,horizontalalignment='left',fontsize=20)
            #    text(-2.3,1.7,'$R_e(24)/Re(r)$',transform=a.transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=26)


            c=colorbar(ax=allax,fraction=.02,ticks=cbticks[i])
            c.ax.text(6,.5,cblabel[i],rotation=-90,verticalalignment='center',fontsize=20)


        savefig(homedir+'research/LocalClusters/SamplePlots/size3panel.png')
        savefig(homedir+'research/LocalClusters/SamplePlots/size3panel.eps')
        #savefig(figuredir+'fig12.eps')
        savefig(figuredir+'fig12.pdf')

    def plotsizestellarmass(self,plotsingle=True,btmax=None,btmin=None,equal_pop_bins=True,use_median=True):
        if plotsingle:
            plt.figure(figsize=(7,6))
            plt.subplots_adjust(bottom=.15,left=.15)
        flags = [self.sampleflag & self.membflag,self.sampleflag & ~self.membflag]
        if btmax != None:
            flags = flags & (self.gim2d.B_T_r < btmax)
        if btmin != None:
            flags = flags & (self.gim2d.B_T_r > btmin)
        colors = ['r','b']
        for i in range(len(flags)):
            #plot(self.logstellarmass[flags[i]],self.sizeratio[flags[i]],'ro',color=colors[i],alpha=0.5)
            plot(self.logstellarmass[flags[i]],self.sizeratio[flags[i]],'ro',color=colors[i],alpha=0.5)
            errorbar(self.logstellarmass[flags[i]],self.sizeratio[flags[i]],self.sizeratioERR[flags[i]],fmt=None,ecolor='0.5',alpha=0.5)
            flag = flags[i]
            if btmax != None:
                flag = flag & (self.logstellarmass > 9.1) & (self.logstellarmass < 10.5)
            xbin,ybin,ybinerr,colorbin = binxycolor(self.logstellarmass[flag],self.sizeratio[flag],self.gim2d.B_T_r[flag],erry=True,nbin=5,equal_pop_bins=equal_pop_bins,use_median=use_median)
            #print xbin
            plot(xbin,ybin,'ro',color=colors[i],markersize=18,mec='k',zorder=5)
            #scatter(xbin,ybin,s=200, c=colorbin,marker='^',vmin=0,vmax=0.6,cmap='jet')
            errorbar(xbin,ybin,ybinerr,fmt=None,ecolor='k',alpha=0.7)
        #colorbar(label='$B/T$')

        xlabel('$ \log_{10}(M_\star /M_\odot) $',fontsize=22)
        ylabel('$ R_{24}/R_d  $',fontsize=22)
        #rho,p=spearman(self.logstellarmass[flag],self.sizeratio[flag])
        #ax=plt.gca()
        #plt.text(.95,.9,r'$\rho = %4.2f$'%(rho),horizontalalignment='right',transform=ax.transAxes,fontsize=18)
        #plt.text(.95,.8,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=18)
        plt.legend(['$Core$','$<Core>$','$External$','$<External>$'],numpoints=1)
        s=''
        if btmax != None:
            s = '$B/T \ <  \  %.2f$'%(btmax)
        if btmin != None:
            s = '$B/T \ >  \  %.2f$'%(btmin)
        if (btmax != None) & (btmin != None):
            s = '$%.2f < B/T \ <  \  %.2f$'%(btmin,btmax)
        plt.title(s,fontsize=20)
        
        plt.axis([8.6,10.9,-.1,2.9])
        plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizestellarmass.pdf')
        plt.savefig(homedir+'research/LocalClusters/SamplePlots/sizestellarmass.png')
        plt.savefig(figuredir+'fig13.pdf')
        
    def plotsizeHIfrac(self,sbcutobs=20.5,isoflag=0,r90flag=0,color_BT=False):
        plt.figure(figsize=plotsize_single)
        plt.subplots_adjust(bottom=.2,left=.15)
        plt.clf()
        flag = self.sampleflag & (self.HIflag) #& self.dvflag #& ~self.agnflag
        print 'number of galaxies = ',sum(flag)
        y=(self.sizeratio[flag & self.membflag])
        x=np.log10(self.s.HIMASS[flag & self.membflag])-self.logstellarmass[flag & self.membflag]
        print 'spearman for cluster galaxies only'
        t = spearman(x,y)
        if color_BT:
            pointcolor = self.gim2d.B_T_r
            v1=0
            v2=0.6
         else:
             pointcolor = self.logstellarmass
             v1=mstarmin
             v2=mstarmax
         #color=self.logstellarmass[flag]
         color=pointcolor[flag & self.membflag]
         sp=scatter(x,y,s=90,c=color,vmin=v1,vmax=v2,label='$Core$',cmap='jet',edgecolor='k')

         y=(self.sizeratio[flag & ~self.membflag])
         x=np.log10(self.s.HIMASS[flag & ~self.membflag])-self.logstellarmass[flag & ~self.membflag]
         print 'spearman for exterior galaxies only'
         t = spearman(x,y)

         #color=self.logstellarmass[flag]
         color=pointcolor[flag & ~self.membflag]
         sp=scatter(x,y,s=90,c=color,vmin=v1,vmax=v2,marker='s',label='$External$',cmap='jet',edgecolor='k')
         y=(self.sizeratio[flag])
         x=np.log10(self.s.HIMASS[flag])-self.logstellarmass[flag]
         plt.legend(loc='upper left',scatterpoints=1)
         errorbar(x,y,self.sizeratioERR[flag],fmt=None,ecolor='.5',zorder=100)
         rho,p=spearman(x,y)

         ax=plt.gca()
         text(.95,.9,r'$\rho = %4.2f$'%(rho),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
         text(.95,.8,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
         print 'spearman for log(M*) < 10.41'
         rho,p=spearman(x[color < 10.41],y[color<10.41])
         cb = plt.colorbar(sp,fraction=.08,ticks=np.arange(8.5,11,.5))
         cb.ax.text(4.,.5,'$\log(M_\star/M_\odot)$',rotation=-90,verticalalignment='center',fontsize=20)
         #plt.ylabel(r'$ R_e(24)/R_e(r)$')
         plt.ylabel('$R_{24}/R_d$')
         plt.xlabel(r'$ \log_{10}(M_{HI}/M_*)$')

         ax.tick_params(axis='both', which='major', labelsize=16)
         plt.axis([-1.8,1.6,0,2.5])
         plt.savefig(figuredir+'fig16a.eps')

    def plotNUVrsize(self):
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(left=.1,wspace=.01,bottom=.2,right=.9)

        BTmin = 0
        BTmax = 0.4
        flags = [self.sampleflag, self.sampleflag & self.membflag,self.sampleflag & ~self.membflag]
        labels = ['$All$','$Core$','$External$']
        allax=[]
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.scatter(self.sizeratio[flags[i]],self.NUVr[flags[i]],c=self.gim2d.B_T_r[flags[i]],s=60,cmap='jet',vmin=BTmin,vmax=BTmax)
            

            if i == 0:
                plt.ylabel('$NUV-r$',fontsize=24)
            else:
                plt.gca().set_yticklabels(([]))
            text(0.98,0.9,labels[i],transform=gca().transAxes,horizontalalignment='right',fontsize=20)
            (rho,p)=spearman_with_errors(self.NUVr[flags[i]],self.sizeratio[flags[i]],self.sizeratioERR[flags[i]])
              
            ax=plt.gca()
    
            plt.text(.05,.08,r'$\rho = [%4.2f, %4.2f]$'%(np.percentile(rho,16),np.percentile(rho,84)),horizontalalignment='left',transform=ax.transAxes,fontsize=14)
            plt.text(.05,.03,'$p = [%5.4f, %5.4f]$'%(np.percentile(p,16),np.percentile(p,84)),horizontalalignment='left',transform=ax.transAxes,fontsize=14)

            plt.axhline(y=4,ls='-',color='0.5')
            plt.axhline(y=4.5,ls='--',color='0.5')
            plt.axhline(y=3.5,ls='--',color='0.5')
            allax.append(plt.gca())
            plt.xticks(np.arange(0,4))
            plt.axis([-0.3,3.1,0,6.2])
        
        colorlabel='$B/T$'
        c=plt.colorbar(ax=allax,fraction=.02,ticks = np.arange(0,.5,.1))
        c.ax.text(3.5,.5,colorlabel,rotation=-90,verticalalignment='center',fontsize=20)
        plt.text(-.51,-.2,'$R_{24}/R_d $',transform=plt.gca().transAxes,fontsize=24,horizontalalignment='center')
        outfile=homedir+'research/LocalClusters/SamplePlots/NUVrsize'
        plt.savefig(outfile+'.png')
        plt.savefig(outfile+'.eps')
        plt.savefig(figuredir+'fig17.eps')

