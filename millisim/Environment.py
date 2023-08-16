## Detector.py
## methods relating to detector and environment properties

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

class Environment(object):

    ## materials definition. (atomic num, atomic weight, density (g/cm3), radiation length (m))
    ## found here: http://pdg.lbl.gov/2019/AtomicNuclearProperties/index.html
    materials = { "fe"   : (26, 55.845, 7.874, .01757), 
                  "si"   : (14, 28.0855, 2.329, .0937),
                  "pb"   : (82, 207.2, 11.35, .005612),
                  "air"  : (7.34, 14.719, 1.205e-3, 3.04e2),  
                  "pbwo4": (31.3, 75.8, 8.3, 0.008903),
                  "bc408": (3.37, 6.23, 1.032, 0.4254),
                  "concrete": (11.10, 22.08, 2.3, .1155),
                  "rock": (11.0, 22.0, 2.65, .1002)  }
    
    ## parameters for dEdx (I, a, k, x0, x1, Cbar, delta0)
    ## found here: http://pdg.lbl.gov/2019/AtomicNuclearProperties/index.html
    dEdx_params = { "fe"   : (286.0, 0.14680, 2.9632, -0.0012, 3.1531, 4.2911, 0.12),
                    "si"   : (173.0, 0.14921, 3.2546, 0.2015, 2.8716, 4.4355, 0.14),
                    "pb"   : (823.0, 0.09359, 3.1608, 0.3776, 3.8073, 6.2018, 0.14),
                    "air"  : (85.7, 0.10914, 3.3994, 1.7418, 4.2759, 10.5961, 0.0),
                    "pbwo4": (600.7, 0.22758, 3.0, 0.4068, 3.0023, 5.8528, 0.0),
                    "bc408": (64.7, 0.16101, 3.2393, 0.1464, 2.4855, 3.1997, 0.0),
                    "concrete": (135.2, .07515, 3.5467, .1301, 3.0466, 3.9464, 0.0),
                    "rock": (136.4, .08301, 3.4210, .0492, 3.0549, 3.7738, 0.0) }

    CMS_LENGTH = 15.0
    CMS_RADIUS = 3.6

    ## parameters to load bfield
    ZMIN = -1500
    ZMAX = 1500
    DZ = 10
    RMIN = 0
    RMAX = 900
    DR = 10
    PHIMIN = 0
    PHIMAX = 355
    DPHI = 5
    # parameters to load fine bfield
    ZMINf = -1500
    ZMAXf = 1500
    DZf = 1
    RMINf = 0
    RMAXf = 900
    DRf = 1
    PHIMINf = 0
    PHIMAXf = 355
    DPHIf = 5

    def __init__(self, mat_setup=None, bfield=None, bfield_file=None, mat_function=None, density_mult=1.0,
                 use_fine_bfield=False, interpolate_b=True, rock_begins=999999., rock_ends=999999.):

        self.BFieldLoaded = False

        self.mat_setup = mat_setup
        self.__mat_function = mat_function
        self.bfield = bfield
        self.rock_begins = rock_begins
        self.rock_ends = rock_ends
        self.use_fine_bfield = use_fine_bfield
        self.interpolate_b = interpolate_b
        self.density_mult = density_mult

        if bfield_file is not None:
            if use_fine_bfield:
                self.LoadFineBField(bfield_file)
            else:
                self.LoadCoarseBField(bfield_file)


    @property
    def mat_setup(self):
        return self.__mat_setup

    @mat_setup.setter
    def mat_setup(self, m):
        if m is None:
            m = "none"
        m = m.lower()
        allowed = ["none", "sife", "justrock", "cms"]
        allowed += ["unif_"+mat for mat in Environment.materials]
        if m not in allowed:
            raise Exception("Unrecognized material setup: "+m)
        self.__mat_setup = m

    @property
    def bfield(self):
        return self.__bfield

    @bfield.setter
    def bfield(self, b):
        if b is None:
            b = "none"
        b = b.lower()
        if b not in ["none", "uniform", "updown", "cms"]:
            raise Exception("Unrecognized bfield configuration: "+b)
        self.__bfield = b

    
    def LoadCoarseBField(self, fname, usePickle=True):

        # Claudio: added self.Bmag_cc because it looks like
        # Drawing.py wants to have Bmag but it smells like
        # that is obsolete.  I add it with a "_cc" suffix
        # here and in Drawing.py so that if any other piece
        # of code needs it, it will crash and I will know
        if usePickle:
            Bx,By,Bz,Bmag = pickle.load(open(fname,"rb"))
            self.B = np.stack((Bx,By,Bz),3)
            self.Bmag_cc = np.array(Bmag)  # the np.array is from Bfield.py
            self.BFieldLoaded = True
            return

        NR = self.RMAX/self.DR + 1
        NZ = (self.ZMAX-self.ZMIN)/self.DZ + 1
        NPHI = (self.PHIMAX-self.PHIMIN)/self.DPHI + 1

        self.B   = np.zeros((NR,NZ,NPHI,3))

        with open(fname,'r') as fid:
            for line in fid:
                sp = line.strip().split()
                if len(sp)!=4:
                    continue

                r = float(sp[0])
                z = float(sp[1])
                phi = float(sp[2])
                B = tuple([float (a) for a in sp[3].strip("()").split(",")])

                iz = int((z-self.ZMIN)/self.DZ)
                ir = int(r/self.DR)
                iphi = int((phi-self.PHIMIN)/self.DPHI)

                self.B[ir,iz,iphi,:] = B

        self.BFieldLoaded = True


    def LoadFineBField(fnamex, fnamey=None, fnamez=None):

        if fnamey is None:
            Bxf,Byf,Bzf = pickle.load(open(fname,"rb"))
            self.B = np.stack((Bxf,Byf,Bzf), 3)
            self.BFieldLoaded = True
            return

        Bxf = pickle.load(open(fnamex,"rb"))
        Byf = pickle.load(open(fnamey,"rb"))
        Bzf = pickle.load(open(fnamez,"rb"))
        self.B = np.stack((Bxf,Byf,Bzf), 3)
        self.BFieldLoaded = True

    def GetMaterial(self, x,y,z):

        # support for custom material function
        if self.__mat_function is not None:
            return self.__mat_function(x,y,z)

        if self.mat_setup.startswith("unif"):
            return self.mat_setup.split("_")[1]

        if self.mat_setup == 'sife':
            if x<4:
                return 'si'
            else:
                return 'fe'

        R = np.sqrt(x**2+y**2+z**2)
        if self.mat_setup in ["justrock", "cms"] and self.rock_begins < R < self.rock_ends:
            return 'rock'

        if self.mat_setup == 'justrock':
            return 'air'

        if self.mat_setup == 'cms':

            withinLength = -self.CMS_LENGTH/2.0 < z < self.CMS_LENGTH/2.0
            r = np.sqrt(x**2+y**2)

            if not withinLength:
                return 'air'

            if r < 1.25:
                mat = 'air'
            elif r < 1.50:
                mat = 'pbwo4'
            elif 1.8 < r < 2.8 or\
                 3.15 < r < 3.5 or\
                 3.85 < r < 4.05 or\
                 4.6 < r < 4.95 or\
                 5.35 < r < 5.95 or\
                 6.35 < r < 7.15:
                mat = 'fe'
            else:
                mat = 'air'
                        
            return mat

        raise Exception("ERROR: invalid material setup. Shouldn't get here")


    def GetBField(self, x,y,z):

        if self.bfield == 'none':
            return np.zeros(3)
        
        if self.bfield == 'uniform':
            return np.array([0.,0.,1.])
        
        if self.bfield == 'updown':
            r = np.sqrt(x**2+y**2)
            if r<4:
                return np.array([0,0,3])
            else:
                return np.array([0,0,-3])/r

        if self.bfield == 'cms':
            if not self.BFieldLoaded:
                raise Exception("ERROR: the bfield file must be loaded before calling GetBField")
    
            ## correct for cm usage in bfield file
            x *= 100
            y *= 100
            z *= 100
            r = np.sqrt(x**2+y**2)

            if self.use_fine_bfield:
                ZMIN,ZMAX,DZ,RMIN,RMAX,DR,PHIMIN,PHIMAX,DPHI = \
                    self.ZMINf, self.ZMAXf, self.DZf, \
                    self.RMINf, self.RMAXf, self.DRf, \
                    self.PHIMINf, self.PHIMAXf, self.DPHIf
            else:
                ZMIN,ZMAX,DZ,RMIN,RMAX,DR,PHIMIN,PHIMAX,DPHI = \
                    self.ZMIN, self.ZMAX, self.DZ, \
                    self.RMIN, self.RMAX, self.DR, \
                    self.PHIMIN, self.PHIMAX, self.DPHI

            if z>ZMIN and z<ZMAX and r<RMAX:

                phi = np.arctan2(y,x) * 180/np.pi

                if phi<0:
                    phi += 360
        
                nearZ = int(DZ*round(z/DZ))
                nearPHI = int(DPHI*round(phi/DPHI))
        
                if nearPHI==360:
                    nearPHI = 0

                iz = int( (nearZ-ZMIN)/DZ )
                iphi = int( (nearPHI-PHIMIN)/DPHI )

                ir = r/DR
                irlow = int(np.floor(ir))
                irhigh = int(np.ceil(ir))
                irfrac = ir-irlow

                if not self.interpolate_b:
                    irfrac = 1
                    irhigh = int(np.round(ir))
        
                if self.use_fine_bfield:
                    B = irfrac*self.B[irhigh,iz,iphi,:] + (1-irfrac)*self.B[irlow,iz,iphi,:]
                else:
                    B = irfrac*self.B[irhigh,iz,iphi,:] + (1-irfrac)*self.B[irlow,iz,iphi,:]
            
                return B

            else:
                return np.zeros(3)

        raise Exception("ERROR: invalid bfield config. Shouldn't get here")

