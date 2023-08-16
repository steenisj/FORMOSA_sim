## Drawing.py
## contains various routines to visualize trajectories

import numpy as np
import matplotlib.pyplot as plt
from millisim.Environment import Environment

## 3d view

def Draw3Dtrajs(trajs, colors=None, ax = None, fig=None, subplot=111):
    # trajs is a list of trajectory arrays as returned by the Integrator.rk4 routine
    # colors is an optional array of colors to use
    # ax is the mpl axes to plot to. If None, createsnew axes
    # fig is the figure to use (if ax not specified). Defaults to plt.gcf()
    # subplot is the subplot to plot to

    from mpl_toolkits.mplot3d import Axes3D

    if ax==None:
        if fig==None:
            fig = plt.gcf()
        ax = fig.add_subplot(subplot, projection='3d')

    if colors==None:
        colors = ['r','g','b','c','m','y']
        # colors = ['r','c','y']

    nc = len(colors)
        
    # NOTE: y and z axes are all inverted here fo display purposes
    for i in range(len(trajs)):
        ax.plot3D(xs=trajs[i][0,:], ys=trajs[i][2,:], zs=trajs[i][1,:], color=colors[i%nc])

    sr, sl = Environment.CMS_RADIUS, Environment.CMS_LENGTH
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(xs=sr*np.cos(t), ys=sl/2*np.ones(t.size), zs=sr*np.sin(t), color='k')
    ax.plot(xs=sr*np.cos(t), ys=-sl/2*np.ones(t.size), zs=sr*np.sin(t), color='k')
    for i in range(8):
        th = i * 2*np.pi/8
        x = sr*np.cos(th)
        y = sr*np.sin(th)
        ax.plot(xs=[x,x], ys=[-sl/2,sl/2], zs=[y,y], color='k')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_zlabel('y (m)')

    ax.set_xlim((-9,9))
    ax.set_ylim((-15,15))
    ax.set_zlim((-9,9))


def DrawXYslice(trajs, colors=None, ax=None, fig=None, subplot=111):
    # draws projected trajectories in z=0 plane
    # see above for argument descriptions

    if ax == None:
        if fig==None:
            fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    if colors==None:
        colors = ['r','g','b','c','m','y']

    nc = len(colors)

    # draw solenoid outline
    sr = Environment.CMS_RADIUS
    sl = Environment.CMS_LENGTH
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(sr*np.cos(t),sr*np.sin(t), '-')

    # draw trajectory
    for i in range(len(trajs)):
        ax.plot(trajs[i][0,:],trajs[i][1,:],'-', linewidth=2, color=colors[i%nc])

    ax.set_xlim((-9,9))
    ax.set_ylim((-9,9))

def DrawXZslice(trajs, colors=None, ax=None, fig=None, subplot=111, drawBFieldFromEnviron=None, drawColorbar=False):
    # draws projected trajectories in z=0 plane
    # see above for argument descriptions

    if ax is None:
        if fig is None:
            fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    if colors is None:
        colors = ['r','g','b','c','m','y']
    nc = len(colors)

    if drawBFieldFromEnviron is not None:        
        env = drawBFieldFromEnviron
        # draw B field
        x = np.arange(-env.RMAX, env.RMAX+1e-10, env.DR)/100
        z = np.arange(env.ZMIN, env.ZMAX+1e-10, env.DZ)/100
        Z,X = np.meshgrid(z,x)

        mag = np.append(env.Bmag_cc[::-1,:,0],env.Bmag_cc[1:,:,int(180/env.DPHI)],0)
        bmplot = plt.pcolor(Z,X,mag,cmap='afmhot', vmax = 4.0, vmin = 0.0)
        
        if drawColorbar:
            bmcb = plt.colorbar(bmplot)
            bmcb.set_label('B (T)',fontsize=14)

    # draw trajectories
    for i in range(len(trajs)):
        plt.plot(trajs[i][2,:],trajs[i][0,:],'-', linewidth=2, color=colors[i%nc])

    plt.xlabel("z (m)")
    plt.ylabel("x (m)")

    ax.set_xlim((-15,15))
    ax.set_ylim((-9,9))

def DrawLine(p1, p2, ax=None, is3d=False, **kwargs):
    # draws a line connecting 2 points

    if ax==None:
        ax = plt.gca()

    if is3d:
        # invert y and z for display purposes (see Draw3Dtrajs function)
        ax.plot(xs=[p1[0],p2[0]],ys=[p1[2],p2[2]],zs=[p1[1],p2[1]], **kwargs)
    else:
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]], **kwargs)
    

