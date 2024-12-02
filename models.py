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

class Kmeans:
    
    def __init__(self, data, centroids, clamped=None, embedding=mlo.Euclidean()):
        """
            Kmeans model for supervised and semi-supervised clustering.
            data: data matrix
            centroids: each row in this matrix is a centroid for a class. Must
            have same cols as data.
            clamped: set of clamped variables
            embedding: manifold where data is embedded.
        """
        if data.cols!=centroids.cols:
            raise Exception("data and centroids must have same number of cols.")
        self.data = data
        self.centroids = centroids
        self.embedding = embedding
        if clamped is not None:
            if self.clamped.ctype!="int":
                raise Exception("clamped must have int ctype.")
            self.clamped = clamped
        else:
            self.clamped = None
        self.preLabels = None
        self.util = mlo.KmeansUtil()
    
    def clampTo(self, To):
        """
            Define to what class variables will be clamped. This is a necessary
            step if self.clamped is not None.
        """
        if To.ctype!="int":
            raise Exception("To must have int ctype.")
        self.preLabels = To
    
    @property
    def labels(self):
        """
            Returns the labels according to current centroids.
            If self.clamped and self.preLabels are not None, clamped variables
            are set to their preLabels values.
        """
        dists = self.getCentroidDistance()
        labels = dists.argmin()
        if self.preLabels is not None and self.clamped is not None:
            labels.copy(self.preLabels,self.clamped)
        return labels
    
    def getCentroids(self):
        """
            Returns the updated centroids according to self.labels.
        """
        if isinstance(self.embedding,mlo.Euclidean):
            return self.util.euclideanCentroid(self.data,self.labels)
    
    def getCentroidDistance(self):
        if isinstance(self.embedding,mlo.Euclidean):
            dists = self.util.euclideanCentroidDistance(
                self.data,self.centroids
            )
        return dists
    
    def train(self, tmax=1000, precision=0.005, track=None):
        t = 0
        delta = 1+precision
        if track is None or track==[]:
            while t<tmax and delta>precision:
                newCentroid = self.getCentroids()
                delta = abs(newCentroid-self.centroids)
                self.centroids = newCentroid
                t += 1
            return t
        else:
            tracked = [[] for p in track]
            while t<tmax and delta>precision:
                print(t)
                newCentroid = self.getCentroids()
                newCentroid.print()
                self.centroids.print()
                delta = abs(newCentroid-self.centroids)
                self.centroids = newCentroid
                for p in range(len(track)):
                    tracked[p].append(track[p]())
                t += 1
            return t, tracked