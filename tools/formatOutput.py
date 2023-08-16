#! /usr/bin/env python
##
## usage: python formatOutput.py file.txt out.txt
##
## takes the output of milliqan_test.py (Q,m,p,pT,eta,phi,theta,thetaW,thetaV,w,v,pInt)
## and outputs more useful numbers (Q,m,x,y,z,px,py,pz), where the position and 
## momentum are the values upon intersection with the detector plane

import numpy as np
import sys

distanceToDetector = 33.

if len(sys.argv) < 3:
    print "usage: python formatOutput.py <input file> <output_file>"
    exit(1)

fout = open(sys.argv[2],'w')

with open(sys.argv[1]) as fin:
    for line in fin:
        sp = line.strip().split()
        if len(sp)!=12:
            continue
        q = float(sp[0])
        m = float(sp[1])
        x = distanceToDetector
        y = float(sp[10])
        z = float(sp[9])

        th = float(sp[6])
        thW = float(sp[7])
        thV = float(sp[8])
        p = float(sp[11])
        
        px = p*np.cos(th)
        py = p*np.cos(th)*np.tan(thV)
        pz = p*np.cos(th)*np.tan(thW)

        fout.write('{0:f}\t{1:f}\t{2:f}\t{3:f}\t{4:f}\t{5:f}\t{6:f}\t{7:f}\n'.format(q,m,x,y,z,px,py,pz))

fout.close()
