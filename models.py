import miolo as mlo
import numpy as np

mlo.miolo_type = "float"

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
        self.data = data
        self.centroids = centroids
        self.embedding = embedding
        self.clamped = clamped
        self.preLabels = None
        self.util = mlo.KmeansUtil()
    
    def seed(self,k_centroids):
        self.centroids = self.util.seed(self.data,k_centroids)
    
    def clampTo(self, To):
        """
            Define to what class variables will be clamped. This is a necessary
            step if self.clamped is not None.
        """
        if To.ctype!="int":
            raise Exception("To must have int ctype.")
        if To.rows!=self.data.rows:
            raise Exception("To must have length equal to self.data.rows.")
        if To.cols!=1:
            raise Exception("To must have a single column.")
        self.preLabels = To
    
    @property
    def mode(self):
        return self.labels.numpy.reshape(self.labels.rows)
    
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
            Returns a Matrix for which each row is the coordinates of the 
            centroids of groups.
        """
        L = self.labels
        for k in range(self.centroids.rows):
            part = self.util.partition(self.data,L,k)
            M = self.embedding.mean(part)
            if k==0:
                cntr = M
            else:
                cntr = mlo.concat(cntr,M)
        return cntr
    
    def getCentroidDistance(self):
        """
            Returns a Matrix for which each row is the distance of a row of 
            self.data to the centroids. 
            Number of cols is equal to self.centroid.cols.
        """
        if isinstance(self.embedding,mlo.Euclidean):
            dists = self.util.euclideanCentroidDistance(
                self.data,self.centroids
            )
        if isinstance(self.embedding,mlo.Sphere):
            dists = self.util.sphereCentroidDistance(
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
                newCentroid = self.getCentroids()
                delta = abs(newCentroid-self.centroids)
                self.centroids = newCentroid
                for p in range(len(track)):
                    tracked[p].append(track[p]())
                t += 1
            return t, tracked
    
class SphericalKmeans:
    
    def __init__(self, data, centroids, clamped=None):
        """
            Kmeans model for supervised and semi-supervised clustering.
            data: data matrix
            centroids: each row in this matrix is a centroid for a class. Must
            have same cols as data.
            clamped: set of clamped variables
            embedding: manifold where data is embedded.
        """
        self.data = data
        self.centroids = centroids
        self.clamped = clamped
        self.preLabels = None
        self.util = mlo.KmeansUtil()
    
    def seed(self,k_centroids):
        self.centroids = self.util.seed(self.data,k_centroids)

    def clampTo(self, To):
        """
            Define to what class variables will be clamped. This is a necessary
            step if self.clamped is not None.
        """
        if To.ctype!="int":
            raise Exception("To must have int ctype.")
        if To.rows!=self.data.rows:
            raise Exception("To must have length equal to self.data.rows.")
        if To.cols!=1:
            raise Exception("To must have a single column.")
        self.preLabels = To
    
    @property
    def mode(self):
        return self.labels.numpy.reshape(self.labels.rows)
    
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
            Returns a Matrix for which each row is a centroid for a group.
        """
        L = self.labels
        sphere = mlo.Sphere()
        eucl = mlo.Euclidean()
        for k in range(self.centroids.rows):
            part = self.util.partition(self.data,L,k)
            M = eucl.mean(part)
            if k==0:
                cntr = M
            else:
                cntr = mlo.concat(cntr,M)
        cntr = sphere.stereographicProjection(cntr)
        return cntr
    
    def getCentroidDistance(self):
        """
            Returns a Matrix for which each row is the distance of a row of 
            self.data to the centroids. 
            Number of cols is equal to self.centroid.cols.
        """
        dists = self.util.sphereCentroidDistance(self.data,self.centroids)
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
                newCentroid = self.getCentroids()
                delta = abs(newCentroid-self.centroids)
                self.centroids = newCentroid
                for p in range(len(track)):
                    tracked[p].append(track[p]())
                t += 1
            return t, tracked

class HyperbolicKmeans:
    
    def __init__(self, data, centroids=None, clamped=None):
        """
            Kmeans model for supervised and semi-supervised clustering on the
            Poincare ball.
            data: data matrix
            centroids: each row in this matrix is a centroid for a class. Must
            have same cols as data.
            clamped: set of clamped variables
        """
        self.data = data
        self.centroids = centroids
        self.clamped = clamped
        self.preLabels = None
        self.util = mlo.KmeansUtil()
    
    def seed(self,k_centroids):
        self.centroids = self.util.seed(self.data,k_centroids)
    
    def clampTo(self, To):
        """
            Define to what class variables will be clamped. This is a necessary
            step if self.clamped is not None.
        """
        if To.ctype!="int":
            raise Exception("To must have int ctype.")
        if To.rows!=self.data.rows:
            raise Exception("To must have length equal to self.data.rows.")
        if To.cols!=1:
            raise Exception("To must have a single column.")
        self.preLabels = To
    
    @property
    def mode(self):
        return self.labels.numpy.reshape(self.labels.rows)
    
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
            Returns a Matrix for which each row is a centroid for a group.
        """
        L = self.labels
        hpbl = mlo.Hyperbolic()
        for k in range(self.centroids.rows):
            part = self.util.partition(self.data,L,k)
            M = hpbl.mean(part)
            if k==0:
                cntr = M
            else:
                cntr = mlo.concat(cntr,M)
        return cntr
    
    def getCentroidDistance(self):
        """
            Returns a Matrix for which each row is the distance of a row of 
            self.data to the centroids. 
            Number of cols is equal to self.centroid.cols.
        """
        dists = self.util.hyperbolicCentroidDistance(self.data,self.centroids)
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
                newCentroid = self.getCentroids()
                delta = abs(newCentroid-self.centroids)
                self.centroids = newCentroid
                for p in range(len(track)):
                    tracked[p].append(track[p]())
                t += 1
            return t, tracked