import miolo as mlo
import numpy as np

mlo.miolo_type = "float"

class LGC:
    
    """
        Local and global consistency model for transduction.
    """
    
    def __init__(self, connection: mlo.Graph, bias: mlo.Matrix, alpha=0.1):
        self.connection = connection
        self.bias = bias
        self.phi = self.bias
        self.alpha = alpha
    
    def grad(self):
        aux = 2*self.connection.laplacian()
        return aux&self.phi+(2.*(1./self.alpha-1.))*(self.bias-self.phi)
    
    def mode(self):
        return self.phi.argmax().numpy.reshape(self.phi.rows)
    
    def train(self, tmax=1000, precision=0.005, track=None):
        t = 0 
        delta = 1+precision
        if track is None or track == []:
            while t<tmax and delta>precision:
                phiAux = (self.alpha*self.connection)&self.phi + \
                    (1.-self.alpha)*self.bias
                delta = abs(phiAux-self.phi)
                self.phi = phiAux
                t += 1
            return t
        else:
             tracker = [[] for l in range(len(track))]
             while t<tmax and delta>precision:
                phiAux = self.alpha*self.connection&self.phi + \
                    (1-self.alpha)*self.bias
                delta = abs(phiAux-self.phi)
                self.phi = phiAux
                for l in range(len(track)):
                    tracker[l].append(track[l]())
                t += 1
        return t, tracker

class hardLP:
    
    def __init__(self, connection: mlo.Graph, clamped, q=2):
        self.connection = connection
        self.clamped = clamped
        self.phi = mlo.Matrix(len(clamped),q,1./q)
        self.q = q 
    
    def mode(self):
        return self.phi.argmax().numpy.reshape(self.phi.rows)
    
    def train(self, tmax=1000, precision=0.005, track=None):
        t = 0
        delta = 1+precision
        if track is None or track == []:
            while t<tmax and delta>precision:
                phiAux = self.connection.propagate(self.phi,self.clamped)
                mlo.normalize(phiAux)
                delta = abs(phiAux-self.phi)
                self.phi = phiAux
                t += 1
            return t
        else:
            tracker = [[] for l in range(len(track))]
            while t<tmax and delta>precision:
                phiAux = self.connection.propagate(self.phi,self.clamped)
                phiAux.normalize()
                delta = abs(phiAux-self.phi)
                self.phi = phiAux
                for l in range(len(track)):
                    tracker[l].append(track[l]())
                t += 1
        return t, tracker

class softLP:
    
    def __init__(self, connection: mlo.Graph, bias: mlo.Matrix, q=2):
        self.connection = connection
        self.bias = bias
        self.phi = bias
        self.q = q 
    
    def mode(self):
        return self.phi.argmax().numpy.reshape(self.phi.rows)
    
    def train(self, tmax=1000, precision=0.005, track=None):
        t = 0
        delta = 1+precision
        if track is None or track == []:
            while t<tmax and delta>precision:
                phiAux = self.connection&self.phi+self.bias
                phiAux.normalize()
                delta = abs(phiAux-self.phi)
                self.phi = phiAux
                t += 1
            return t
        else:
            tracker = [[] for l in range(len(track))]
            while t<tmax and delta>precision:
                phiAux = self.connection&self.phi+self.bias
                phiAux.normalize()
                delta = abs(phiAux-self.phi)
                self.phi = phiAux
                for l in range(len(track)):
                    tracker[l].append(track[l]())
                t += 1
        return t, tracker
