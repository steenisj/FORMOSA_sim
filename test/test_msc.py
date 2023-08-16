import numpy as np
import matplotlib.pyplot as plt
import math
from millisim.Environment import Environment
from millisim.Integrator import Integrator
from millisim.Detector import *
import millisim.fast_integrate as fast
import millisim.Drawing as Drawing
from tqdm import tqdm

env = Environment(
    mat_setup = 'unif_fe',
    bfield = 'none',
)

itg = Integrator(
    environ = env,
    Q = 1.0,
    m = 105.,
    dt = 0.1,
    nsteps = 5000,
    cutoff_dist = 5,
    cutoff_axis = 'x',
    use_var_dt = True,
    lowv_dx = 0.01,
    multiple_scatter = 'pdg',
    do_energy_loss = False,
    randomize_charge_sign = False,
    )

x0 = np.array([0, 0, 0, 20000.0, 0, 0])
N = 10000

itg.multiple_scatter = 'pdg'
defl_pdg = []
defly_pdg = []
deflz_pdg = []
disp_pdg = []
fe_pdg = []
for i in tqdm(range(N)):
    traj,tvec = itg.propagate(x0, fast=True)
    p = traj[3:,-1]
    pxy = [p[0], p[1], 0]
    pxz = [p[0], 0, p[2]]
    defl_pdg.append(180./np.pi*np.arccos(p[0]/np.linalg.norm(p)))
    defly_pdg.append(180./np.pi*np.arcsin(p[1]/np.linalg.norm(pxy)))
    deflz_pdg.append(180./np.pi*np.arcsin(p[2]/np.linalg.norm(pxz)))
    disp_pdg.append(np.linalg.norm(traj[1:3,-1]))
    fe_pdg.append(np.linalg.norm(p)/1000)

itg.multiple_scatter = 'kuhn'
defl_kuhn = []
defly_kuhn = []
deflz_kuhn = []
disp_kuhn = []
fe_kuhn = []
for i in tqdm(range(N)):
    traj,tvec = itg.propagate(x0, fast=True)
    p = traj[3:,-1]
    pxy = [p[0], p[1], 0]
    pxz = [p[0], 0, p[2]]
    defl_kuhn.append(180./np.pi*np.arccos(p[0]/np.linalg.norm(p)))
    defly_kuhn.append(180./np.pi*np.arcsin(p[1]/np.linalg.norm(pxy)))
    deflz_kuhn.append(180./np.pi*np.arcsin(p[2]/np.linalg.norm(pxz)))
    disp_kuhn.append(np.linalg.norm(traj[1:3,-1]))
    fe_kuhn.append(np.linalg.norm(p)/1000)

plt.figure(figsize=(16,12))
plt.subplot(221)
plt.hist(defl_pdg, bins=40, range=(0,7), histtype='step', color='b', ls='-')
plt.hist(defl_kuhn, bins=40, range=(0,7), histtype='step', color='b', ls='--')
plt.xlabel("deflection (degrees)")
plt.gca().set_yscale('log')

plt.subplot(222)
plt.hist(defly_pdg, bins=40, range=(-5,5), histtype='step', color='b', ls='-')
plt.hist(deflz_pdg, bins=40, range=(-5,5), histtype='step', color='r', ls='-')
plt.hist(defly_kuhn, bins=40, range=(-5,5), histtype='step', color='b', ls='--')
plt.hist(deflz_kuhn, bins=40, range=(-5,5), histtype='step', color='r', ls='--')
plt.xlabel("deflection (degrees)")
plt.gca().set_yscale('log')

plt.subplot(223)
plt.hist(disp_pdg, bins=40, range=(0,0.25), histtype='step', color='b', ls='-')
plt.hist(disp_kuhn, bins=40, range=(0,0.25), histtype='step', color='b', ls='--')
plt.xlabel("displacement (meters)")

plt.subplot(224)
plt.hist(fe_pdg, bins=40, range=(18,22), histtype='step', color='b', ls='-')
plt.hist(fe_kuhn, bins=40, range=(18,22), histtype='step', color='b', ls='--')
plt.xlabel("displacement (meters)")

plt.show()
