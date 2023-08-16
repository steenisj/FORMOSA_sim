#! /usr/bin/env python

from __future__ import print_function 

try:
    import cPickle as pickle
except:
    import pickle
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

startTime = time.time()

pyVersion = sys.version_info[0]

z = np.arange(-1500, 1500+1e-10, 10)
x = np.arange(-900, 900+1e-10, 10)

Z,X = np.meshgrid(z,x)

#Bx, By, Bz, Bmag = pickle.load(open("bfield.pkl","rb"))   
#Bmag = pickle.load(open("bfield_mag.pkl","rb"))
if pyVersion == 2:
    Bx,By,Bz,Bmag = pickle.load(open("bfield_coarse.pkl","rb"))
else:
    Bx,By,Bz,Bmag = pickle.load(open("bfield_coarse_p3.pkl","rb"))

print("loaded pickle file ({0:.2f} s)".format(time.time()-startTime))

Bx = np.array(Bx)
By = np.array(By)
Bz = np.array(Bz)
Bmag = np.array(Bmag)

print(Bmag.shape)
print(np.amin(Bmag), np.amax(Bmag))

print("converted to numpy array ({0:.2f} s)".format(time.time()-startTime))

Bx   = np.append(  Bx[::-1,:,0],  Bx[1:,:,int(180/5)],0)
By   = np.append(  By[::-1,:,0],  By[1:,:,int(180/5)],0)
Bz   = np.append(  Bz[::-1,:,0],  Bz[1:,:,int(180/5)],0)
Bmag = np.append(Bmag[::-1,:,0],Bmag[1:,:,int(180/5)],0)

print("flattened to 2D array ({0:.2f} s)".format(time.time()-startTime))
print(Bmag)

plt.figure(num=1, figsize=(11.7,7))
bmplot = plt.pcolor(Z,X,Bmag,cmap='bone', vmax = 4.0, vmin = 0.0)
bmcb = plt.colorbar(bmplot)
bmcb.set_label('B (T)',fontsize=14)

plt.xlabel('z (cm)',fontsize=14)
plt.ylabel('x (cm)',fontsize=14)

print(np.min(Bmag[-1,-1]))

k=3
#plt.quiver(Z[::k,::k],X[::k,::k],Bz[::k,::k],Bx[::k,::k], width=0.001, color='k')

plt.axis([-1500,1500,-900,900])

plt.savefig('cms_bfield_coarse.png', bbox_inches='tight')

# dpi = 80.0
# xpixels, ypixels = 3001, 1801
# plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)

# plt.imshow(Bmag, cmap='bone', interpolation='nearest')

# print "created image ({0:.2f} s)".format(time.time()-startTime)

# plt.savefig('cms_bfield_coarse.png')

print("saved as png ({0:.2f} s)".format(time.time()-startTime))

# fig = plt.figure(num=2, figsize=(11.7,7))
# byplot = plt.pcolor(Z,X,Bx, cmap='RdYlGn',vmin=-2.0,vmax=2.0)
# bycb = plt.colorbar(byplot)

# x = np.arange(-900,901,5)
# y = np.arange(-900,901,5)

# X,Y = np.meshgrid(x,y)

# Bmag = np.zeros(X.shape)

# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         x = X[i,j]
#         y = Y[i,j]

#         r = np.sqrt(x**2+y**2)
#         phi = np.arctan2(y,x)*180/np.pi
        
#         if r>900:
#             Bmag[i,j]=0
#             continue

#         nearR = int(10*round(r/10))
#         nearPhi = int(5*round(phi/5))

#         if nearPhi==360:
#             nearPhi = 0

#         ir = r/10
#         iphi = phi/5

#         Bmag[i,j] = Brpmag[iphi,ir]

# plt.figure(num=3, figsize=(8,8))

# plt.pcolor(X,Y,Bmag, vmin=0, vmax=3.9)

# plt.savefig("slices/z{0:04d}.png".format(zslice+1500))
# print zslice


plt.show()




















