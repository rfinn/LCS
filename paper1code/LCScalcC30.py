#!/usr/bin/env python


'''
starting code on MKW 11 first

goal is to calculate the extent of the image, C30, for example


Before running:
* copy final *.tab to /home/rfinn/research/LocalClusters/EllipseTables/MKW11/.
* run

galTablesV4.py '*.tab' 

to convert *.tab to *.dat files



Steps:

* cull the sample so only galaxies with both sdss and 24um images

* read in mastertable to identify the hubble type of the galaxy

* keep only spiral galaxies

* for each galaxy, read in sdss and 24um profiles

* calculate C30 for each galaxy

* write out results to new mastertable

'''




def movefiles():


def readtable(infile):
    data=asciitable.read(infile,delimiter='\s')
    x=data['col1']
    y=data['col2']
    yerr=data['col3']
