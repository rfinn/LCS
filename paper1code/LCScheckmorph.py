#!/usr/bin/env python

'''

useage:  LCScheckmorph.py

run from OffCenter subdirectory


display image in ds9 w/contrast=0.01
start imexam
print current morphology
ask user if she wants to reclassify
   if yes, write agcname and new morph class to cluster.RevisedMorph
      this can then be read by LCSmkcutouts.py

   if no, continue to next galaxy
'''

import glob
from pyraf import iraf
from LCScommon import *
import os

def checkmorph(cluster):
    print "Checking morphologies for ",cluster
    infile1='/home/rfinn/research/LocalClusters/MorphologyF2011/'+cluster+'.funnySpirals.dat'
    in1=open(infile1,'r')
    infile2='/home/rfinn/research/LocalClusters/MorphologyF2011/'+cluster+'.funnySpirals'
    in2=open(infile2,'r')
    lines1=in1.readlines()
    lines2=in2.readlines()
    Nfunny=len(lines1)
    bothlines1=zip(lines1,lines2)
    in1.close()
    in2.close()
    infile1='/home/rfinn/research/LocalClusters/MorphologyF2011/'+cluster+'.noBsteinSpirals.dat'
    in1=open(infile1,'r')
    infile2='/home/rfinn/research/LocalClusters/MorphologyF2011/'+cluster+'.noBsteinSpirals'
    in2=open(infile2,'r')
    lines1=in1.readlines()
    lines2=in2.readlines()
    nBstein=len(lines1)
    bothlines2=zip(lines1,lines2)
    in1.close()
    in2.close()
    infile1='/home/rfinn/research/LocalClusters/MorphologyF2011/'+cluster+'.noMorph.dat'
    in1=open(infile1,'r')
    infile2='/home/rfinn/research/LocalClusters/MorphologyF2011/'+cluster+'.noMorph'
    in2=open(infile2,'r')
    lines1=in1.readlines()
    lines2=in2.readlines()
    NnoMorph=len(lines1)
    bothlines3=zip(lines1,lines2)
    in1.close()
    in2.close()
    newmorphagc=[]
    newmorphclass=[]
    allfiles=bothlines1+bothlines2+bothlines3
    print 'Our classification system: \n 1=Elliptical\n 2=S0 \n 3=Spiral\n 4=Irregular\n 5=Peculiar\n 6=No Galaxy\n 7=Low Surface Brightnes\n'
    i=0
    for line1,line2 in allfiles:
        line1=line1.rstrip()
        if i == 0:
            print "Galaxies where our class and Bstein class disagree"
        if i == Nfunny:
            print "Galaxies with no Bstein class"
        if i == (Nfunny+nBstein):
            print "Galaxies that are on our 24um image but not classified"
        iraf.display(line1,contrast=0.01,frame=1)
        print "current classifications: ",line2
        iraf.imexam(line1,frame=1)
        flag=str(raw_input('Are you happy with the our classification?  y=yes n=no x=quit '))
        flag=str(flag)
        print 'this is what I think you typed ',flag
        if flag.find('n') > -1:
            repeatflag=1
            while repeatflag:
                newmorph=str(raw_input('Enter new morphology. \n 1=Elliptical\n 2=S0 \n 3=Spiral\n 4=Irregular\n 5=Peculiar\n 6=No Galaxy\n 7=Low Surface Brightnes\n'))
                print 'this is what I think you typed ',newmorph
                flag2= str(raw_input('Are you happy with this classification?  y=yes n=no x=quit '))
                if flag2.find('y') > -1:
                    repeatflag=0
                if flag2.find('x') > -1:
                    return
            t=line2.split()
            agc=t[0]
            newmorphagc.append(agc)
            newmorphclass.append(newmorph)
        elif flag.find('x') > -1:
            print 'quitting'
            return
        ngal =+ 1
    in1.close()
    in2.close()
    outfile='/home/rfinn/research/LocalClusters/MorphologyF2011/'+cluster+'.F2011morph.dat'
    out1=open(outfile,'w')
    for i in range(len(newmorphagc)):
        s=str(newmorphagc[i])+' '+str(newmorphclass[i])+' \n'
        out1.write(s)
    out1.close()


checkmorph(clusternames[0])
#for cname in clusternames:
#    checkmorph(cname)
