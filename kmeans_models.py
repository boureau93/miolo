import numpy as np
import miolo as mlo

class Kmeans:
    
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.center = mlo.Matrix(k,data.cols,ctype=data.ctype)
        self.E = mlo.Euclidean()
    
    @property
    def labels(self):
        D = self.E.centroidDistance(self.data,self.center)
        D = D.argmin()
        L = D.rows
        D = D.numpy.reshape(L)
        return D.astype(np.int32)
    
    def seed(self):
        self.center = self.E.kmpp(self.data,self.k)
    
    def updateCenter(self):
        labs = self.labels
        for p in range(self.k):
            aux = self.data.partition(labs,p)
            if aux.null:
                aux = mlo.Matrix(1,self.data.cols,0)
            print("Particionou")
            mean = self.E.mean(aux)
            print("Mediou")
            if p==0:
                self.center = mean
                print("Iniciou")
            else:
                print("Vai concatenar")
                self.center = mlo.concat(self.center,mean)
                print("Concatenou")
    
    def train(self, tmax=1000, precision=0.001, track=None):
        t = 0
        delta = 1+precision
        if track is None or track==[]:
            while t<tmax and delta>precision:
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                t += 1
            return t
        else:
            tracked = [[] for l in range(len(track))]
            while t<tmax and delta>precision:
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                t += 1
                for l in range(len(track)):
                    tracked[l].append(track[l]())
            return t, tracked
        
class SphericalKmeans:
    
    def __init__(self, data, k):
        self.k = k
        self.E = mlo.Euclidean()
        self.S = mlo.Sphere(1)
        self.center = self.S.fromEuclidean(mlo.Matrix(k,data.cols,ctype=data.ctype))
        self.data = self.S.fromEuclidean(data)
    @property
    def labels(self):
        print((self.data.cols,self.center.cols))
        D = self.S.centroidDistance(self.data,self.center)
        D = D.argmin()
        L = D.rows
        D = D.numpy.reshape(L)
        return D.astype(np.int32)
    
    def seed(self):
        print(self.data.cols)
        aux = self.S.toEuclidean(self.data)
        print(aux.cols)
        self.center = self.E.kmpp(aux,self.k)
        self.center = self.S.fromEuclidean(self.center)
    
    def updateCenter(self):
        labs = self.labels
        for p in range(self.k):
            aux = self.data.partition(labs,p)
            if aux.null:
                aux = self.S.fromEuclidean(mlo.Matrix(1,self.data.cols-1,0))
            print("Particionou")
            mean = self.E.mean(aux)
            print("Mediou")
            if p==0:
                self.center = mean
                print("Iniciou")
            else:
                print("Vai concatenar")
                self.center = mlo.concat(self.center,mean)
                print("Concatenou")
        self.center = self.S.stereographicProjection(self.center)
    
    def train(self, tmax=1000, precision=0.001, track=None):
        t = 0
        delta = 1+precision
        if track is None or track==[]:
            while t<tmax and delta>precision:
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                t += 1
            return t
        else:
            tracked = [[] for l in range(len(track))]
            while t<tmax and delta>precision:
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                t += 1
                for l in range(len(track)):
                    tracked[l].append(track[l]())
            return t, tracked

class LorentzianKmeans:
    
    def __init__(self, data, k, beta=0.001):
        self.k = k
        self.E = mlo.Lorentz(beta)
        self.data = self.E.fromEuclidean(data)
        self.center = self.E.fromEuclidean(mlo.Matrix(k,self.data.cols-1))
    
    @property
    def labels(self):
        D = self.E.centroidDistance(self.data,self.center)
        fabs = mlo.fabs()
        D = fabs(D).argmin()
        L = D.rows
        D = D.numpy.reshape(L)
        return D.astype(np.int32)
    
    def seed(self):
        Eucli = mlo.Euclidean()
        self.center = Eucli.kmpp(self.E.toEuclidean(self.data),self.k)
        self.center = self.E.fromEuclidean(self.center)
    
    def updateCenter(self):
        labs = self.labels
        for p in range(self.k):
            aux = self.data.partition(labs,p)
            if aux.null:
                aux = self.E.fromEuclidean(mlo.Matrix(1,self.data.cols-1,0))
            print("Particionou")
            mean = self.E.mean(aux)
            print("Mediou")
            if p==0:
                self.center = mean
                print("Iniciou")
            else:
                print("Vai concatenar")
                self.center = mlo.concat(self.center,mean)
                print("Concatenou")
    
    def train(self, tmax=1000, precision=0.001, track=None):
        t = 0
        delta = 1+precision
        if track is None or track==[]:
            while t<tmax and delta>precision:
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                t += 1
            return t
        else:
            tracked = [[] for l in range(len(track))]
            while t<tmax and delta>precision:
                print(t)
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                print(delta)
                t += 1
                for l in range(len(track)):
                    tracked[l].append(track[l]())
            return t, tracked

class PoincareKmeans:
    
    def __init__(self, data, k, c=1):
        self.k = k
        self.E = mlo.Poincare(c)
        self.zero = mlo.Matrix(data.rows,data.cols)
        self.data = self.E.exp(self.zero,data)
        self.center = mlo.Matrix(k,data.cols)
    
    @property
    def labels(self):
        D = self.E.centroidDistance(self.data,self.center)
        D = D.argmin()
        L = D.rows
        D = D.numpy.reshape(L)
        return D.astype(np.int32)
    
    def seed(self):
        Eucli = mlo.Euclidean()
        self.center = Eucli.kmpp(self.data,self.k)
        aux = mlo.Matrix(self.center.rows,self.center.cols,0)
        self.center = self.E.exp(aux,self.center)
    
    def updateCenter(self):
        labs = self.labels
        for p in range(self.k):
            aux = self.data.partition(labs,p)
            if aux.null:
                aux = mlo.Matrix(1,self.data.cols,0)
            print("Particionou")
            mean = self.E.mean(aux)
            print("Mediou")
            if p==0:
                self.center = mean
                print("Iniciou")
            else:
                print("Vai concatenar")
                self.center = mlo.concat(self.center,mean)
                print("Concatenou")
    
    def train(self, tmax=1000, precision=0.001, track=None):
        t = 0
        delta = 1+precision
        if track is None or track==[]:
            while t<tmax and delta>precision:
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                t += 1
            return t
        else:
            tracked = [[] for l in range(len(track))]
            while t<tmax and delta>precision:
                print(t)
                aux = self.center
                self.updateCenter()
                delta = abs(self.center-aux)
                t += 1
                for l in range(len(track)):
                    tracked[l].append(track[l]())
            return t, tracked