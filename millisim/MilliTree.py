import ROOT
import numpy as np

class MilliTree:
    def __init__(self):
        self.tree = ROOT.TTree("Events","")
        
        self.Bfield = ""
        self.MatSetup = ""
        self.MSCtype = ""
        self.EnergyLossOn = 0
        self.q = 0.
        self.m = 0.
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.px = 0.
        self.py = 0.
        self.pz = 0.
        self.numHits = 0
        self.numSims = 0

        self._Bfield = ROOT.std.string()
        self._MatSetup = ROOT.std.string()
        self._MSCtype = ROOT.std.string()
        self._EnergyLossOn = np.zeros(1, dtype=int)
        self._q = np.zeros(1, dtype=float)
        self._m = np.zeros(1, dtype=float)
        self._x = np.zeros(1, dtype=float)
        self._y = np.zeros(1, dtype=float)
        self._z = np.zeros(1, dtype=float)
        self._px = np.zeros(1, dtype=float)
        self._py = np.zeros(1, dtype=float)
        self._pz = np.zeros(1, dtype=float)

        self.tree.Branch("Bfield",self._Bfield)
        self.tree.Branch("MatSetup",self._MatSetup)
        self.tree.Branch("MSCtype",self._MSCtype)
        self.tree.Branch("EnergyLossOn",self._EnergyLossOn,"EnergyLossOn/I")
        self.tree.Branch("q",self._q,"q/D")
        self.tree.Branch("m",self._m,"m/D")
        self.tree.Branch("x",self._x,"x/D")
        self.tree.Branch("y",self._y,"y/D")
        self.tree.Branch("z",self._z,"z/D")
        self.tree.Branch("px",self._px,"px/D")
        self.tree.Branch("py",self._py,"py/D")
        self.tree.Branch("pz",self._pz,"pz/D")

    def Fill(self):
        
        n = ROOT.std.string.npos
        self._Bfield.replace(0, n, self.Bfield)
        self._MatSetup.replace(0, n, self.MatSetup)
        self._MSCtype.replace(0, n, self.MSCtype)
        self._EnergyLossOn[0] = self.EnergyLossOn
        self._q[0] = self.q
        self._m[0] = self.m
        self._x[0] = self.x
        self._y[0] = self.y
        self._z[0] = self.z
        self._px[0] = self.px
        self._py[0] = self.py
        self._pz[0] = self.pz

        self.tree.Fill()

    # pass the output of Detector.FindIntersection to this, and it will populate the tree values
    def SetValues(self, itg, intersect, pInt):
        self.Bfield = itg.environ.bfield
        self.MatSetup = itg.environ.mat_setup
        self.MSCtype = itg.multiple_scatter
        self.EnergyLossOn = itg.do_energy_loss
        self.q = itg.Q
        self.m = itg.m
        self.x = intersect[0]
        self.y = intersect[1]
        self.z = intersect[2]
        self.px = pInt[0]
        self.py = pInt[1]
        self.pz = pInt[2]

    def Write(self, filename):
        fid = ROOT.TFile(filename,"RECREATE")
        self.tree.Write()
        fid.Close()
