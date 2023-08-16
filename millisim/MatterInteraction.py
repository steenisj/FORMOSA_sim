## MatterInteraction.py
## methods relating to interaction with matter (energy loss, multiple scattering)

import numpy as np
from millisim.Environment import Environment

def doEnergyLoss(itg, x, dt):
    ## returns dx from losing energy according to Bethe-Bloch

    p = x[3:]
    magp = np.linalg.norm(p)
    E = np.sqrt(magp**2 + itg.m**2)
    gamma = E / itg.m
    beta = magp / E;
    me = 0.511;  #electron mass in MeV
    
    Wmax = 2*me*beta**2*gamma**2/(1+2*gamma*me/itg.m + (me/itg.m)**2)
    K = 0.307075  # in MeV cm^2/mol

    mat = itg.environ.GetMaterial(x[0],x[1],x[2])
    Z,A,rho,X0 = Environment.materials[mat]
    rho *= itg.environ.density_mult
    I,a,k,x0,x1,Cbar,delta0 = Environment.dEdx_params[mat]


    I = I/1e6  ## convert from eV to MeV

    xp = np.log10(magp/itg.m)
    if xp>=x1:
        delta = 2*np.log(10)*xp - Cbar
    elif xp>=x0:
        delta = 2*np.log(10)*xp - Cbar + a*(x1-xp)**k
    else:
        delta = delta0*10**(2*(xp-x0))

    # mean energy loss in MeV/cm
    dEdx = K*rho*itg.Q**2*Z/A/beta**2*(0.5*np.log(2*me*beta**2*gamma**2*Wmax/I**2) - beta**2 - delta/2)

    dE = dEdx * beta*2.9979e1 * dt

    if dE>(E-itg.m):
        return np.append(np.zeros(3), -p)

    newmagp = np.sqrt((E-dE)**2-itg.m**2)
    return np.append(np.zeros(3), p*(newmagp/magp-1))

def multipleScatter(itg, x, dt):
    # get the angles/displacements from relevant function and return the
    # net change in x=(x,y,z,px,py,pz)    

    if itg.environ.GetMaterial(x[0],x[1],x[2])=='air':
        return np.zeros(6)

    p = x[3:]
    if itg.multiple_scatter=='pdg':
        thetax, thetay, yx, yy = _getScatterAnglePDG(itg, x, dt)
    elif itg.multiple_scatter=='kuhn':
        thetax, thetay, yx, yy = _getScatterAngleKuhn(itg, x, dt)
        if thetax < 0:
            thetax, thetay, yx, yy = _getScatterAnglePDG(itg, x, dt)
    else:
        raise Exception("Invalid MSC algorithm "+itg.multiple_scatter)

    return _getMSVec(p, thetax, thetay, yx, yy)


def _getMSVec(p, thetax, thetay, yx, yy):
    vx = _getNormVector(p)
    vy = np.cross(vx, p/np.linalg.norm(p))

    # transverse displacement
    disp = yx*vx + yy*vy
    
    # deflection in momentum
    defl = np.linalg.norm(p) * (thetax*vx + thetay*vy)

    return np.append(disp, defl)

def _getNormVector(v):
    # generate and return a random vector in the plane orthogonal to v

    v = np.array(v)
    nv = np.sqrt(v[0]**2+v[1]**2+v[2]**2)

    if nv == 0:
        raise ValueError("Can't generate vector orthogonal to 0 vector!")

    # angular coords of v
    thetav = np.arccos(np.dot(v/nv, [0,0,1]))
    phiv = np.arctan2(v[1],v[0])
    
    angle = 2*np.pi*np.random.rand()
    random_unit = np.array([np.cos(angle), np.sin(angle), 0]).reshape((3,1))

    ct = np.cos(thetav)
    st = np.sin(thetav)
    cp = np.cos(phiv)
    sp = np.sin(phiv)

    # matrix for rotation about y axis by thetav
    rotateY = np.array([[ ct, 0, st],
                        [  0, 1,  0],
                        [-st, 0, ct]])

    # matrix for rotation about z axis by phiv
    rotateZ = np.array([[cp, -sp, 0],
                        [sp,  cp, 0],
                        [ 0,   0, 1]])

    random_unit = np.dot(rotateZ, np.dot(rotateY, random_unit))
    random_unit = random_unit.reshape((3,))

    return random_unit

def _getKuhnScatteringParams(itg, x, dt):

    mat = itg.environ.GetMaterial(x[0],x[1],x[2])

    Z,A,rho,X0 = Environment.materials[mat]
    rho *= itg.environ.density_mult

    z = abs(itg.Q)

    p = x[3:]
    magp = np.linalg.norm(p)
    v = p/np.sqrt(magp**2 + itg.m**2)
    beta = np.linalg.norm(v)

    ds = beta * 2.9979e1 * dt

    Xc = np.sqrt(0.1569 * z**2 * Z*(Z+1) * rho * ds / (magp**2 * beta**2 * A))
    b = np.log(6700*z**2*Z**(1./3)*(Z+1)*rho*ds/A / (beta**2+1.77e-4*z**2*Z**2))

    if b<3:
        return -1,-1,-1

    ## we want to solve the equation B-log(B) = b. Using Newton-Raphson
    B = b
    prevB = 2*B
    
    f = lambda x: x-np.log(x)-b
    fp = lambda x: 1-1./x

    while abs((B-prevB)/prevB)>0.001:
        prevB = B
        B = B - f(B)/fp(B)
        
    # use B+1 for correction at intermediate angles
    return ds, Xc, B+1


def _getScatterAngleKuhn(itg, x, dt):
    # use Kuhn method to multiple scatter

    ds, Xc, B = _getKuhnScatteringParams(itg, x, dt)
    if Xc==-1:
        return -1, -1, -1, -1

    sr2 = np.sqrt(2)

    # width of gaussian
    th1e = Xc*np.sqrt(B-1.25)
    th1esr2 = th1e/sr2

    Ppi = 1 - 0.827/B + Xc**2/4*(1/np.sin(th1esr2)**2 - 1)

    R = np.random.rand() * Ppi;

    if R >= 1-0.827/B:
        R = R-1+0.827/B
        th = 2*np.arcsin(Xc*np.sin(th1esr2)/np.sqrt(Xc**2-4*R*np.sin(th1esr2)**2))
    else:
        th = th1e*np.sqrt(np.log((1-0.827/B)/(1-0.827/B-R)))

    # # generate correlated transverse displacements
    # rho = 0.87
    # z = np.random.normal()
    # yx = z*ds*th1e*np.sqrt((1-rho**2)/3) + th*rho*ds/np.sqrt(3)
    # z = np.random.normal()
    # yy = z*ds*th1e*np.sqrt((1-rho**2)/3)    

    # return th, 0.0, yx, yy
    return th, 0.0, 0.0, 0.0

def _getScatterAnglePDG(itg, x, dt):
    # return thetax, thetay, yx, yy
    # given a velocity and timestep, compute a
    # deflection angle and deviation due to multiple scattering
    # Update the position/velocity and return deflection
    #
    # x is a 6-element vector (x,y,z,px,py,pz)
    # returns deflection angles thetax, thetay and
    # displacements yx, yy in two orthogonal directions to p

    p = x[3:]
    magp = np.linalg.norm(p) # must be in MeV
    E = np.sqrt(magp**2 + itg.m**2)
    v = p/E
    beta = np.linalg.norm(v)

    dx = (beta*2.9979e-1) * dt  # in m
    
    ## in the following we generate a random deflection angle theta
    ## and transverse displacement y for the given momentum. This is taken
    ## from the PDG review chapter on the Passage of Particles through Matter

    mat = itg.environ.GetMaterial(x[0],x[1],x[2])

    X0 = Environment.materials[mat][3]
    X0 /= itg.environ.density_mult

    if X0 <= 0:
        return np.zeros(6)

    # rms of projected theta distribution.
    theta0 = 13.6/(beta*magp) * abs(itg.Q) * np.sqrt(dx/X0) * (1 + 0.038*np.log(dx/X0))
    
    # correlation coefficient between theta_plane and y_plane
    rho = 0.87
    
    getRandom = np.random.normal

    z1 = getRandom()
    z2 = getRandom()
    yx = z1*dx*theta0 * np.sqrt((1-rho**2)/3) + z2*rho*dx*theta0/np.sqrt(3)
    thetax = z2*theta0
    
    z1 = getRandom()
    z2 = getRandom()
    yy = z1*dx*theta0 * np.sqrt((1-rho**2)/3) + z2*rho*dx*theta0/np.sqrt(3)
    thetay = z2*theta0

    return thetax, thetay, yx, yy
