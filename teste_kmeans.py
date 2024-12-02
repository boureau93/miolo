import models
import miolo as mlo
from sklearn.metrics import accuracy_score, rand_score
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import time

X = mlo.txtMatrix("gauss/xgauss.dat")
y = np.loadtxt("gauss/ygauss.dat")

C = mlo.Matrix(2,2)
C.numpy = [[-0.1,1],[0.2,-0.1]]

kmeans = models.Kmeans(X,C)

def randIndex():
    return rand_score(kmeans.labels.numpy.reshape(5000),y)

track = [randIndex]

t, tracked = kmeans.train(tmax=5,track=track)

print(tracked)

plt.plot(tracked[0])
plt.show()