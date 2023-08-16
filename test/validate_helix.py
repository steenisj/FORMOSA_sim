#! /usr/bin/python

## simply propagates a particle through a uniform B field and compares
## the computed trajectory to the predicted helix

import numpy as np
import matplotlib.pyplot as plt
import math
from millisim.Environment import Environment
from millisim.Integrator import Integrator

env = Environment(
    mat_setup = None,
    bfield = 'uniform',    
    )

itg = Integrator(
    environ = env,
    Q = 1.0,
    m = 0.5,
    dt = 0.4,
    nsteps = 1000,
    )

p0 = [1000,0,1000]

P = np.linalg.norm(p0)
Pxy = np.linalg.norm(p0[:2])
E = np.sqrt(P**2 + itg.m**2)
Q = 1.
B = 1.
c = 2.9979e-1
R = Pxy/(300*Q*B)
vxy = Pxy/E * 2.9979e-1
T = 2*np.pi*R/vxy
w = 2*np.pi/T

x0 = np.array([0,R,0]+p0)

traj,tvec = itg.propagate(x0)

time = np.arange(0,itg.dt*itg.nsteps+1e-10, itg.dt)

predx = R*np.sin(w*time) 
predy = R*np.cos(w*time)
predz = p0[0]/np.linalg.norm(p0) * c * time

plt.plot(time,predx,'-b', label='x pred')
plt.plot(time[::5], traj[0,::5],'ob', label='x sim')

#plt.plot(time,traj[1,:]-predy,'-r')
plt.plot(time,predy,'-r', label='y pred')
plt.plot(time[::5], traj[1,::5],'or', label='y sim')

plt.xlabel('Time (ns)')
plt.ylabel('Position (m)')
plt.title('x/y vs. t, 1 GeV electron, 1 Tesla B-field')

plt.legend()

plt.show()
