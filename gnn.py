import numpy as np
import miolo as mlo

aux = 1/np.sqrt(np.log(5000))

class GCN:
    
    def __init__(self, featureDimension, numClasses=2):
        self.featureDimension = featureDimension
        self.numClasses = numClasses
        self.layer = mlo.ReLU(featureDimension,numClasses)
        self.layer.mask.numpy = np.random.uniform(-aux,aux,(featureDimension,numClasses))
        self.potential = mlo.logLikelihood()
        
    def __call__(self, connection, features):
        return self.layer(connection,features)
    
    def mode(self, connection, features):
        aux = self(connection,features)
        aux = aux.argmax()
        return aux.numpy.reshape(5000)

    def cost(self,bias,labels):
        return self.potential(bias,labels)

    def grad(self, connection, features, bias):
        A = connection&features
        return self.potential.grad(self.layer,A,features,bias)

    def train(self, connection, features, bias, tmax=100, lr=0.01, track=None):
        t = 0 
        if track is None or track==[]:
            while t<tmax:
                g = self.grad(connection,features,bias)
                self.layer.mask -= lr*g
                t += 1
            return t
        else:
            tracker = [[] for i in track]
            while t<tmax:
                print(t)
                g = self.grad(connection,features,bias)
                self.layer.mask -= lr*g
                for l in range(len(track)):
                    tracker[l].append(track[l]())
                t += 1
            return t, tracker