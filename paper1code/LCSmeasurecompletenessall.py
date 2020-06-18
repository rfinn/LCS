#!/usr/bin/env python

import os
#from LCScommon import *

clusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
clusternames=['MKW11','Coma']

for cl in clusternames:
    s="LCSmeasurecompleteness.py "+cl
    os.system(s)
