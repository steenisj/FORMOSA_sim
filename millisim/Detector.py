## Detector.py
## methods relating to detector and environment properties
from __future__ import print_function 
import numpy as np

class PlaneDetector(object):
    def __init__(self, dist_to_origin, eta, phi, width=None, height=None):
        # if width or height are None, detector plane has infinite extent
        # width corresponds to eta-hat direction (self.unit_w)
        # height corresonds to phi-hat direction (self.unit_v)

        self.dist_to_origin = float(dist_to_origin)
        self.eta = float(eta)
        self.phi = float(phi)
        self.width = float(width) if width is not None else None
        self.height = float(height) if height is not None else None

        if eta == float("inf"):
            theta = 0.0
        elif eta == float("-inf"):
            theta = np.pi
        else:
            theta = 2*np.arctan(np.exp(-eta))
        x = dist_to_origin * np.sin(theta) * np.cos(phi)
        y = dist_to_origin * np.sin(theta) * np.sin(phi)
        z = dist_to_origin * np.cos(theta)

        self.center = np.array([x,y,z])
        self.norm = self.center / np.linalg.norm(self.center)

        if np.isinf(eta):
            self.unit_v = np.array([0., 1., 0.])
        else:
            self.unit_v = np.cross(np.array([0., 0., 1.]), self.norm)
        self.unit_v /= np.linalg.norm(self.unit_v)
        self.unit_w = np.cross(self.norm, self.unit_v)
        

    # get the four corners, for drawing purposes
    def get_corners(self):
        if self.width is None or self.height is None:
            raise Exception("Can't get corners of an infinite detector!")

        c1 = self.center + self.unit_w * self.width/2 + self.unit_v * self.height/2
        c2 = self.center - self.unit_w * self.width/2 + self.unit_v * self.height/2
        c3 = self.center - self.unit_w * self.width/2 - self.unit_v * self.height/2
        c4 = self.center + self.unit_w * self.width/2 - self.unit_v * self.height/2
        
        return c1,c2,c3,c4

    def get_line_segments(self):
        c1,c2,c3,c4 = self.get_corners()
        return [(c1,c2),(c2,c3),(c3,c4),(c4,c1)]

    def draw(self, ax, **kwargs):
        if self.width is None or self.height is None:
            raise Exception("Can't draw an infinite detector!")
        if "color" not in kwargs and "c" not in kwargs:
            kwargs["color"] = 'k'
        # NOTE: y and z axes flipped, for consistency with Drawing module (see Drawing.Draw3Dtrajs)
        for p1, p2 in self.get_line_segments():
            ax.plot(xs=[p1[0],p2[0]], ys=[p1[2],p2[2]], zs=[p1[1],p2[1]], **kwargs)

    def transform_from_detcoords(self, v, w, n=None):
        if n is None:
            n = self.dist_to_origin
        return n*self.norm + v*self.unit_v + w*self.unit_w

    def find_intersection(self, traj, tvec=None):
        # find the intersection with a plane with normal norm
        # and distance to origin dist. returns None if no intersection

        npoints = traj.shape[1]
        dists = np.sum(np.tile(self.norm, npoints).reshape(npoints,3).T * traj[:3,:], axis=0)
        idx = np.argmax(dists > self.dist_to_origin)

        if idx == 0:
            return None

        p1 = traj[:3,idx-1]
        p2 = traj[:3,idx]
        
        proj1 = np.dot(p1,self.norm)
        proj2 = np.dot(p2,self.norm)
        
        frac = (self.dist_to_origin-proj1)/(proj2-proj1)
        intersect = p1 + frac * (p2-p1)        
            
        v = np.dot(intersect,self.unit_v)
        w = np.dot(intersect,self.unit_w)
            
        if self.width is None or self.height is None or (abs(w) < self.width/2 and abs(v) < self.height/2):
            unit = (p2-p1)/np.linalg.norm(p2-p1)
            theta = np.arccos(np.dot(unit,self.norm))
            
            projW = np.dot(unit,self.unit_w)
            projV = np.dot(unit,self.unit_v)
            
            thW = np.arcsin(projW / np.linalg.norm(unit-projV*self.unit_v))
            thV = np.arcsin(projV / np.linalg.norm(unit-projW*self.unit_w))

            t = None
            if tvec is not None:
                t = tvec[idx-1] + frac * (tvec[idx] - tvec[idx-1])
            
            pInt = traj[3:,idx-1] + frac * (traj[3:,idx] - traj[3:,idx-1])

            return {
                "x_int" : intersect,
                "t_int" : t,
                "p_int" : pInt,
                "v" : v,
                "w" : w,
                "theta" : theta,
                "theta_w" : thW,
                "theta_v" : thV,
                }

        return None

class Box(object):
    def __init__(self, center, norm1, norm2, width, height, depth):
        # width corresponds to direction norm1 (unit_u)
        # height corresponds to direction norm2 (unit_v)
        # depth corresponds to direction norm1 x norm2 (unit_w)

        if abs(np.dot(norm1, norm2)) > 1e-6:
            raise Exception("norm1 and norm2 must be perpendicular!")

        self.center = center
        self.unit_u = norm1 / np.linalg.norm(norm1)
        self.unit_v = norm2 / np.linalg.norm(norm2)
        self.unit_w = np.cross(self.unit_u, self.unit_v)
        self.unit_w /= np.linalg.norm(self.unit_w)

        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)

    def get_corners(self):
        c1 = self.center - self.depth/2 * self.unit_w - self.width/2 * self.unit_u - self.height/2 * self.unit_v
        c2 = self.center - self.depth/2 * self.unit_w - self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c3 = self.center - self.depth/2 * self.unit_w + self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c4 = self.center - self.depth/2 * self.unit_w + self.width/2 * self.unit_u - self.height/2 * self.unit_v
        c5 = self.center + self.depth/2 * self.unit_w - self.width/2 * self.unit_u - self.height/2 * self.unit_v
        c6 = self.center + self.depth/2 * self.unit_w - self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c7 = self.center + self.depth/2 * self.unit_w + self.width/2 * self.unit_u + self.height/2 * self.unit_v
        c8 = self.center + self.depth/2 * self.unit_w + self.width/2 * self.unit_u - self.height/2 * self.unit_v
        return (c1,c2,c3,c4,c5,c6,c7,c8)

    def get_line_segments(self):
        c1,c2,c3,c4,c5,c6,c7,c8 = self.get_corners()

        return [
            (c1, c2),
            (c2, c3),
            (c3, c4),
            (c4, c1),
            (c1, c5),
            (c2, c6),
            (c3, c7),
            (c4, c8),
            (c5, c6),
            (c6, c7),
            (c7, c8),
            (c8, c5),
            ]

    def draw(self, ax, **kwargs):
        if "color" not in kwargs and "c" not in kwargs:
            kwargs["color"] = 'k'
        # NOTE: y and z axes flipped, for consistency with Drawing module (see Drawing.Draw3Dtrajs)
        for p1, p2 in self.get_line_segments():
            ax.plot(xs=[p1[0],p2[0]], ys=[p1[2],p2[2]], zs=[p1[1],p2[1]], **kwargs)

    def transform_to_boxcoords(self, inp):
        # inp is either a length-3 1D array, or and (n x 3) 2D array, where each row is a vector to be transformed        
        inp = np.array(inp)
        inp_shape = inp.shape
        inp = np.reshape(inp, (-1,3))    
        inp -= np.tile(self.center, inp.shape[0]).reshape(inp.shape[0],3)
        cob = np.array([self.unit_u, self.unit_v, self.unit_w])
        xform = np.dot(cob, inp.T).T
        return np.reshape(xform, inp_shape)

    def transform_from_boxcoords(self, inp):
        # inp is either a length-3 1D array, or and (n x 3) 2D array, where each row is a vector to be transformed        
        inp = np.array(inp)
        inp_shape = inp.shape
        inp = np.reshape(inp, (-1,3))    
        cob = np.array([self.unit_u, self.unit_v, self.unit_w]).T
        xform = np.dot(cob, inp.T).T
        xform += np.tile(self.center, inp.shape[0]).reshape(inp.shape[0],3)
        return np.reshape(xform, inp_shape)
        
    def contains(self,p):
        xf = self.transform_to_boxcoords(p)
        if not -self.width/2 < xf[0] < self.width/2:
            return False
        if not -self.height/2 < xf[1] < self.height/2:
            return False
        if not -self.depth/2 < xf[2] < self.depth/2:
            return False
        return True

    def find_intersections(self, p1, p2):
        # finds intersections of the line segment from p1 to p2 with the box faces (either 0 or 2)
        xf = self.transform_to_boxcoords([p1,p2])
        p1 = xf[0,:]
        p2 = xf[1,:]

        tbounds = np.zeros((3,2))
        dims = [self.width/2, self.height/2, self.depth/2]
        for i in range(3):
            if p1[i] == p2[i]:
                tbounds[i,:] = [np.inf, np.inf]
            else:
                t1 = (-dims[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (dims[i] - p1[i]) / (p2[i] - p1[i])
                tbounds[i,:] = [min(t1,t2),max(t1,t2)]
                
        tmin = np.amax(tbounds[:,0])
        tmax = np.amin(tbounds[:,1])

        if tmin >= tmax:
            return []

        pmin = p1 + tmin*(p2-p1)
        pmax = p1 + tmax*(p2-p1)

        ixf = self.transform_from_boxcoords([pmin,pmax])

        ret = []
        if 0 <= tmin <= 1:
            ret.append(ixf[0,:])
        if 0 <= tmax <= 1:
            ret.append(ixf[1,:])
        return ret

class MilliqanDetector(object):
    def __init__(self, dist_to_origin, eta, phi, 
                 nrows=3, ncols=2, nlayers=3,
                 bar_width = 0.05, bar_height = 0.05, bar_length = 0.86,
                 bar_gap = 0.01, layer_gap = 0.30):

        width = ncols*bar_width + (ncols-1)*bar_gap
        height = nrows*bar_height + (nrows-1)*bar_gap
        self.face = PlaneDetector(dist_to_origin, eta, phi, width, height)

        self.__nrows = nrows
        self.__ncols = ncols
        self.__nlayers = nlayers
        self.__nbars = self.nrows*self.ncols*self.nlayers
        self.__bar_width = bar_width
        self.__bar_height = bar_height
        self.__bar_length = bar_length
        self.__bar_gap = bar_gap
        self.__layer_gap = layer_gap

        mid_layer = (self.nlayers-1)/2.0
        self.center_3d = self.face.center + \
                         ((mid_layer+0.5)*bar_length + mid_layer*layer_gap) * self.face.norm

        self.total_length = self.nlayers*self.bar_length + (self.nlayers-1)*self.layer_gap
        self.containing_box = Box(self.center_3d, self.face.unit_w, self.face.unit_v, 
                                  width, height, self.total_length)

        self.layer_boxes = []
        for ilayer in range(self.nlayers):
            self.layer_boxes.append(Box(
                self.face.center + ((ilayer+0.5)*bar_length + ilayer*layer_gap) * self.face.norm,
                self.face.unit_w, self.face.unit_v, width, height, self.bar_length
            ))

        # bars is an (nlayers x nrows x ncols) array of Box objects
        # counting from near layer to far, top row to bottom, left col to right
        self.bars = []
        for ilayer in range(nlayers):
            self.bars.append([])
            layer = self.bars[-1]
            for irow in range(nrows):
                layer.append([])
                row = layer[-1]
                rows_from_center = -(irow - (nrows-1)/2.0) # negative so we start counting from top
                for icol in range(ncols):
                    cols_from_center = icol - (ncols-1)/2.0
                    center = self.face.center + \
                             ((ilayer+0.5)*bar_length + ilayer*layer_gap) * self.face.norm + \
                             rows_from_center*(bar_height+bar_gap) * self.face.unit_v +\
                             cols_from_center*(bar_width+bar_gap) * self.face.unit_w
                    row.append(Box(
                        center = center,
                        norm1 = self.face.unit_w,
                        norm2 = self.face.unit_v,
                        width = bar_width,
                        height = bar_height,
                        depth = bar_length
                    ))

    def lrc_to_idx(self, ilayer, irow, icol):
        if not 0 <= ilayer <= self.nlayers-1:
            raise Exception("ilayer = {0} is not a valid layer number!".format(ilayer))
        if not 0 <= irow <= self.nrows-1:
            raise Exception("irow = {0} is not a valid row number!".format(irow))
        if not 0 <= icol <= self.ncols-1:
            raise Exception("icol = {0} is not a valid col number!".format(icol))
        return ilayer*(self.nrows*self.ncols) + irow*self.ncols + icol

    def idx_to_lrc(self, idx):
        if not 0 <= idx <= self.nlayers*self.nrows*self.ncols - 1:
            raise Exception("idx = {0} is not in allowed bounds".format(idx))
        ilayer = idx // (self.nrows*self.ncols)
        idx -= ilayer * self.nrows*self.ncols
        irow = idx // self.ncols
        idx -= irow * self.ncols
        return (ilayer, irow, idx)

    def draw(self, ax, draw_containing_box=False, **kwargs):
        for ilayer in range(self.nlayers):
            for irow in range(self.nrows):
                for icol in range(self.ncols):
                    self.bars[ilayer][irow][icol].draw(ax, **kwargs)
        if draw_containing_box:
            kwargs['c'] = 'r'
            self.containing_box.draw(ax, **kwargs)

    def find_entries_exits(self, traj, assume_straight_line=True):
        # returns a list of tuples
        #  ((layer,row,col), entry_point, exit_point)
        # empty list if no intersections
        # if assume_straight_line==True, this assumes that the trajectory is perfectly straight
        # once past R = self.face.dist_to_origin, and can use a faster algorithm

        npoints = traj.shape[1]
        dists = np.sum(np.tile(self.face.norm, npoints).reshape(npoints,3).T * traj[:3,:], axis=0)
        start_idx = np.argmax(dists > self.face.dist_to_origin)
        end_idx = np.argmax(dists > self.face.dist_to_origin + self.nlayers*self.bar_length + (self.nlayers-1)*self.layer_gap)

        if assume_straight_line:
            p1 = traj[:3,start_idx-1]
            p2 = traj[:3,end_idx]
        
            # first check overall containing box. If no intersects, can skip
            if len(self.containing_box.find_intersections(p1, p2)) == 0:
                return []

            points = []
            for ilayer in range(self.nlayers):
                if len(self.layer_boxes[ilayer].find_intersections(p1, p2)) == 0:
                    continue
                for irow in range(self.nrows):
                    for icol in range(self.ncols):
                        isects = self.bars[ilayer][irow][icol].find_intersections(p1,p2)
                        if len(isects)==0:
                            continue
                        i1 = isects[0]
                        i2 = isects[1] if len(isects)>1 else None
                        points.append(((ilayer,irow,icol),i1,i2))

            return sorted(points, key=lambda p:np.dot(self.face.norm,p[1]))

        # if we're not assuming straight line, step through the trajectory point by point
        points = []
        last_isin = None
        for i in range(start_idx, end_idx+1):
            d = dists[i]
            ilayer = int(np.floor((d-self.face.dist_to_origin)/(self.bar_length+self.layer_gap)))
            if not self.containing_box.contains(traj[:3,i]):
                ilayer = -1
            isin = None
            if 0 <= ilayer < self.nlayers:
                for irow in range(self.nrows):
                    for icol in range(self.ncols):
                        if self.bars[ilayer][irow][icol].contains(traj[:3,i]):
                            isin = (ilayer, irow, icol)
                            break
                    if isin is not None:
                        break

            if isin != last_isin:
                if last_isin is not None:
                    exit_point = self.bars[last_isin[0]][last_isin[1]][last_isin[2]].find_intersections(traj[:3,i-1],traj[:3,i])[0]
                    points.append((last_isin, entry_point, exit_point))
                if isin is not None:
                    entry_point = self.bars[isin[0]][isin[1]][isin[2]].find_intersections(traj[:3,i-1],traj[:3,i])[0]

            last_isin = isin

        return points

    def hits_straight_line(self, isects):
        # takes list of isects from find_entries_exits, and tests whether the same
        # (row,col) is hit in every layer
        layer_hits = np.zeros((self.nlayers,self.nrows,self.ncols))
        for isect in isects:
            layer_hits[isect[0]] = 1
        nlayers = np.sum(layer_hits, axis=0)
        if np.amax(nlayers) == self.nlayers:
            return True
        return False

    @property
    def nrows(self):
        return self.__nrows
    @nrows.setter
    def nrows(self, _):
        print("Can't change nrows after initialization")
    @property
    def ncols(self):
        return self.__ncols
    @ncols.setter
    def ncols(self, _):
        print("Can't change ncols after initialization")
    @property
    def nlayers(self):
        return self.__nlayers
    @nlayers.setter
    def nlayers(self, _):
        print("Can't change nlayers after initialization")
    @property
    def nbars(self):
        return self.__nbars
    @nbars.setter
    def nbars(self, _):
        print("Can't change nbars after initialization")
    @property
    def bar_width(self):
        return self.__bar_width
    @bar_width.setter
    def bar_width(self, _):
        print("Can't change bar_width after initialization")
    @property
    def bar_height(self):
        return self.__bar_height
    @bar_height.setter
    def bar_height(self, _):
        print("Can't change bar_height after initialization")
    @property
    def bar_length(self):
        return self.__bar_length
    @bar_length.setter
    def bar_length(self, _):
        print("Can't change bar_length after initialization")
    @property
    def bar_gap(self):
        return self.__bar_gap
    @bar_gap.setter
    def bar_gap(self, _):
        print("Can't change bar_gap after initialization")
    @property
    def layer_gap(self):
        return self.__layer_gap
    @layer_gap.setter
    def layer_gap(self, _):
        print("Can't change layer_gap after initialization")


