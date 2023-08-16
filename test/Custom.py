import Params
import numpy as np

def getMaterial(x,y,z):
    
    withinLength = -Params.solLength/2 < z < Params.solLength/2
    r = np.sqrt(x**2+y**2)

    # cms detector model
    if not withinLength:
        mat = 'air'
    elif r < 1.29:
        mat = 'air'
    elif r < 1.8:
        mat = 'pbwo4'
    elif r < 2.95:
        mat = 'fe'
    elif r < 4.0:
        mat = 'fe'
    elif r < 7.0:
        mat = 'fe'
    else:
        mat = 'air'

    if np.sqrt(r**2 + z**2) < 32.0 - 17.0:
        return mat
        
    distToDetector = 33.0
    detWidth = 0.11
    detHeight = 0.11
    eta = 0.16
    theta = np.pi/2 - 2*np.arctan(np.exp(-eta))
    xd = distToDetector*np.cos(theta)
    zd = distToDetector*np.sin(theta)
    centerOfDetector = np.array([xd, 0., zd])
    norm = centerOfDetector / distToDetector
    detV = np.array([0., 1., 0.])
    detW = np.cross(norm, detV)

    #u,v,w coordinates
    pos = np.array([x,y,z])
    u = np.dot(pos, norm)
    v = np.dot(pos, detV)
    w = np.dot(pos, detW)

    if 32.0 - 17.0 <= u < 32.0:
        mat = 'rock'
    elif abs(v) > detHeight/2 or abs(w) > detWidth/2:
        mat = 'air'
    elif 33.00 < u < 33.80:
        mat = 'bc408'
    elif 33.975 < u < 34.025:
        mat = 'pb'
    elif 34.20 < u < 35.00:
        mat = 'bc408'
    elif 35.175 < u < 35.225:
        mat = 'pb'
    elif 35.40 < u < 36.20:
        mat = 'bc408'
    else:
        mat = 'air'

    return mat

def findIntersection(traj, tvec, detectorDict):
    # find intersection with rectangular prism detector
    # None if no intersection

    norm = detectorDict["norm"]
    dist = detectorDict["dist"]
    vHat = detectorDict["v"]
    wHat = detectorDict["w"]
    width = detectorDict["width"]
    height = detectorDict["height"]
    depth = detectorDict["depth"]

    hodoSize = 0.45  # side of square hodometer (put 0.0 for no hodometer)

    intersect = None

    for i in range(traj.shape[1]-1):
        p1 = traj[:3,i]
        p2 = traj[:3,i+1]
        
        u = np.dot(p2,norm)
        v = np.dot(p2,vHat)
        w = np.dot(p2,wHat)

        if (dist < u < dist+depth and abs(w) < width/2 and abs(v) < height/2) or \
           (u>=dist and np.dot(p1,norm)<dist and abs(w)<hodoSize/2 and abs(v)<hodoSize/2) or \
           (u>dist+depth and np.dot(p1,norm)<dist+depth and abs(w)<hodoSize/2 and abs(v)<hodoSize/2):

            print "intersect:", u, np.dot(p1,norm), w, v, width, height, hodoSize

            intersect = p2
            t = tvec[i+1]

            unit = (p2-p1)/np.linalg.norm(p2-p1)
            theta = np.arccos(np.dot(unit,norm))
            
            projW = np.dot(unit,wHat)
            projV = np.dot(unit,vHat)
            
            thW = np.arcsin(projW/np.linalg.norm(unit-projV*vHat))
            thV = np.arcsin(projV/np.linalg.norm(unit-projW*wHat))

            # momentum when it hits detector
            pInt = traj[3:,i]

            return intersect,t,theta,thW,thV,pInt

    

    return None, None, None, None, None, None


def trajOutput(traj, detectorDict):

    norm = detectorDict["norm"]
    dist = detectorDict["dist"]
    vHat = detectorDict["v"]
    wHat = detectorDict["w"]
    width = detectorDict["width"]
    height = detectorDict["height"]
    depth = detectorDict["depth"]
    center = norm*dist

    intersect = None
    startInd = -1
    endInd = -1
    tempEndInd = -1

    hodoSize = 0.45
    hitsFrontHodo = False
    hitsBackHodo = False

    for i in range(traj.shape[1]-1):
        p1 = traj[:3,i]
        p2 = traj[:3,i+1]
        
        proj2 = np.dot(p2,norm)

        if proj2>=dist:

            if startInd==-1:
                intersect = p2
                w = np.dot(intersect,wHat)
                v = np.dot(intersect,vHat)
                print "stardInd:", proj2, np.dot(p1, norm), w, v
                if proj2 < dist+0.02 and abs(w) < hodoSize/2 and abs(v) < hodoSize/2:
                    hitsFrontHodo = True
                    startInd = i+1
                elif (abs(w) < width/2 and abs(v) < height/2 and dist < proj2 < dist+depth) or\
                        (proj2>=dist+depth and abs(w)<hodoSize/2 and abs(v)<hodoSize/2):
                    startInd = i+1
            
            w = np.dot(p2,wHat)
            v = np.dot(p2,vHat)

            if startInd != -1 and i+1 >= startInd:
                if abs(w) < width/2 and abs(v) < height/2:
                    tempEndInd = -1
                if (abs(w) > width/2 or abs(v) > height/2) and tempEndInd==-1:
                    tempEndInd = i+1
                if proj2 >= 36.2:
                    endInd = i+1
                    if abs(w) < hodoSize/2 and abs(v) < hodoSize/2:
                        hitsBackHodo = True
                    break                                

    # print startInd, endInd, traj.shape[1]

    if startInd == -1:
        raise Exception("shouldn't get here!")

    if endInd==-1:
        endInd = traj.shape[1]-1

    if not hitsBackHodo and tempEndInd != -1:
        endInd = tempEndInd

    startU = np.dot(intersect, norm)
    startW = np.dot(intersect, wHat)
    startV = np.dot(intersect, vHat)
    startMomU = np.dot(traj[3:, startInd-1], norm)
    startMomW = np.dot(traj[3:, startInd-1], wHat)
    startMomV = np.dot(traj[3:, startInd-1], vHat)
    endU = np.dot(traj[:3, endInd], norm)
    endW = np.dot(traj[:3, endInd], wHat)
    endV = np.dot(traj[:3, endInd], vHat)
    endMomU = np.dot(traj[3:, endInd], norm)
    endMomW = np.dot(traj[3:, endInd], wHat)
    endMomV = np.dot(traj[3:, endInd], vHat)

    crystals = []
    for i in range(startInd, endInd+1):
        pos = traj[:3, i]
        
        u = np.dot(pos, detectorDict["norm"])
        v = np.dot(pos,vHat)
        w = np.dot(pos,wHat)

        if i<endInd and (u < 33. or u > 36.2):
            raise Exception("shouldn't get here!!")

        ind = -1
        if abs(w) < width/2 and abs(v) < height/2:
            if 33.0 < u < 33.8:
                if w<0 and v<0:
                    ind = 0
                if w>=0 and v<0:
                    ind = 1
                if w<0 and v>=0:
                    ind = 2
                if w>=0 and v>=0:
                    ind = 3
            elif 34.2 < u < 35.0:
                if w<0 and v<0:
                    ind = 4
                if w>=0 and v<0:
                    ind = 5
                if w<0 and v>=0:
                    ind = 6
                if w>=0 and v>=0:
                    ind = 7
            elif 35.4 < u < 36.2:
                if w<0 and v<0:
                    ind = 8
                if w>=0 and v<0:
                    ind = 9
                if w<0 and v>=0:
                    ind = 10
                if w>=0 and v>=0:
                    ind = 11
            else:
                ind = -1

        if len(crystals)==0 or crystals[-1] != ind:
            crystals.append(ind)

    if hitsFrontHodo:
        crystals = [12] + crystals
    if hitsBackHodo:
        crystals = crystals + [13]

    cstring = "/".join(str(i) for i in crystals)
    
    return Params.Q, Params.m, startU, startW, startV, startMomU, startMomW, startMomV, endU, endW, endV, endMomU, endMomW, endMomV, cstring
