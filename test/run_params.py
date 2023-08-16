import numpy as np

mode = "VIS"
ntrajs = 10
pt_spect_filename = "../p_eta_dist/combined_PtSpect_Eta0p16.root"
dt = 0.1   #timestep in ns
max_nsteps = 5000
cutoff = 42.
use_var_dt = False
bfield_type = "cms"

particleQ = 1.0  # in electron charge units
particleM = 105. # in MEV

distToDetector = 33.
eta = 0.16
rock_begins = distToDetector - 17.

detWidth = 5.0
detHeight = 5.0
detDepth = 1.0

etabounds = (eta-0.08, eta+0.08)
ptCut = 17.
phibounds = (0.00, 0.22)

useCustomMaterialFunction = False
#useCustomIntersectionFunction = False
useCustomOutput = False
