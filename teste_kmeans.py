import kmeans_models as models
import miolo as mlo
from sklearn.metrics import accuracy_score, rand_score
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import time

X = mlo.txtMatrix("gauss/xgauss.dat")
y = np.loadtxt("gauss/ygauss.dat")

kmeans = models.Kmeans(X,2)
kmeans.seed()

def randIndex():
    return rand_score(kmeans.labels,y)

track = [randIndex]

t, tracked = kmeans.train(tmax=1000,precision=-1,track=track)

print(tracked)

plt.plot(tracked[0])
plt.show()