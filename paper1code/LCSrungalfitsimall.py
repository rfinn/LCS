#!/usr/bin/env python

import os
#from LCScommon import *

clusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
#clusternames=[ 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
#clusternames=['Coma','Hercules']

for cl in clusternames:
    s="~/Dropbox/pythonCode/LCSgalfitsim.py "+cl
    os.system(s)
