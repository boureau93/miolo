import models
import miolo as mlo
from sklearn.metrics import rand_score
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import time

str_X = input("Feature Matrix: ")
str_graph = input("Graph file: ")
str_y = input("Ground truth: ")
str_bias = input("Bias folder: ")
str_clamps = input("Clamps folder: ")

X = mlo.txtMatrix(str_X)
G = mlo.txtGraph(str_graph)
y = np.loadtxt(str_y)
q = max(y)+1
biases = listdir(str_bias)
clamps = listdir(str_clamps)

fig, axs = plt.subplots(2,sharex=True)

#-------------------------------------------------------------------------------
# LGC
#-------------------------------------------------------------------------------

def Accuracy():
    return rand_score(y,model.mode)
track = [Accuracy]

acc = []
etime = []

for sp in biases:
    print(sp)
    bias = mlo.txtMatrix(str_bias+"/"+sp)
    model = models.LGC(G,bias)
    start = time.time()
    t, tracked = model.train(tmax=1000,precision=-1,track=track)
    etime.append(time.time()-start)
    acc.append(tracked[0])

n = len(acc)
acc_mean = np.zeros(len(acc[0]))
etime_mean = np.zeros(len(acc[0]))
for i in range(n):
    acc_mean += acc[i]
    etime_mean += etime[i]
acc_mean /= n
etime_mean /= n

axs[0].plot(etime_mean,label="LGC")
axs[1].plot(acc_mean)

#-------------------------------------------------------------------------------
# Kmeans
#-------------------------------------------------------------------------------

def Accuracy():
    return rand_score(y,model.mode)
track = [Accuracy]

def biasToClamp(bias):
    out = np.zeros(len(bias))
    for i in range(len(bias)):
        out[i] = np.sum(bias[i])
    return out.astype(bool)

acc = []
etime = []

yMatrix = mlo.Matrix(len(y),1,ctype="int")
yMatrix.numpy = y.reshape((len(y),1))

for sp in clamps:
    print(sp)
    clamped = np.loadtxt(str_clamps+"/"+sp)
    clamped = clamped.astype(np.uint)
    model = models.Kmeans(X,None,clamped)
    model.seed(q)
    model.clampTo(yMatrix)
    start = time.time()
    t, tracked = model.train(tmax=1000,precision=-1,track=track)
    etime.append(time.time()-start)
    acc.append(tracked[0])

n = len(acc)
acc_mean = np.zeros(len(acc[0]))
etime_mean = np.zeros(len(acc[0]))
for i in range(n):
    acc_mean += acc[i]
    etime_mean += etime[i]
acc_mean /= n
etime_mean /= n

axs[0].plot(etime_mean,label="K-means")
axs[1].plot(acc_mean)

#-------------------------------------------------------------------------------
# Spherical Kmeans
#-------------------------------------------------------------------------------

PI = 3.141592684

def Accuracy():
    return rand_score(y,model.mode)
track = [Accuracy]

def biasToClamp(bias):
    out = np.zeros(len(bias))
    for i in range(len(bias)):
        out[i] = np.sum(bias[i])
    return out.astype(bool)

acc = []
etime = []

yMatrix = mlo.Matrix(len(y),1,ctype="int")
yMatrix.numpy = y.reshape((len(y),1))

sphere = mlo.Sphere(1)
Xsph = sphere.coordinateReady(X)
Xsph = sphere.fromEuclidean(Xsph)

for sp in clamps:
    print(sp)
    clamped = np.loadtxt(str_clamps+"/"+sp)
    clamped = clamped.astype(np.uint)
    model = models.SphericalKmeans(Xsph,None,clamped)
    model.seed(q)
    model.clampTo(yMatrix)
    start = time.time()
    t, tracked = model.train(tmax=1000,precision=-1,track=track)
    etime.append(time.time()-start)
    acc.append(tracked[0])

n = len(acc)
acc_mean = np.zeros(len(acc[0]))
etime_mean = np.zeros(len(acc[0]))
for i in range(n):
    acc_mean += acc[i]
    etime_mean += etime[i]
acc_mean /= n
etime_mean /= n

axs[0].plot(etime_mean,label="Spherical K-means")
axs[1].plot(acc_mean,ls="dashed")

#-------------------------------------------------------------------------------
# Hyperbolic K-means
#-------------------------------------------------------------------------------

PI = 3.141592684

def Accuracy():
    return rand_score(y,model.mode)
track = [Accuracy]

def biasToClamp(bias):
    out = np.zeros(len(bias))
    for i in range(len(bias)):
        out[i] = np.sum(bias[i])
    return out.astype(bool)

acc = []
etime = []

yMatrix = mlo.Matrix(len(y),1,ctype="int")
yMatrix.numpy = y.reshape((len(y),1))

hyper = mlo.Hyperbolic()
zero = mlo.Matrix(X.rows,X.cols,0)
Xhyp = hyper.exp(zero,X)

for sp in clamps:
    print(sp)
    clamped = np.loadtxt(str_clamps+"/"+sp)
    clamped = clamped.astype(np.uint)
    model = models.HyperbolicKmeans(Xhyp,None,clamped)
    model.seed(q)
    model.clampTo(yMatrix)
    start = time.time()
    t, tracked = model.train(tmax=1000,precision=-1,track=track)
    etime.append(time.time()-start)
    acc.append(tracked[0])

n = len(acc)
acc_mean = np.zeros(len(acc[0]))
etime_mean = np.zeros(len(acc[0]))
for i in range(n):
    acc_mean += acc[i]
    etime_mean += etime[i]
acc_mean /= n
etime_mean /= n

axs[0].plot(etime_mean,label="Hyperbolic K-means")
axs[1].plot(acc_mean,ls="dashed")

#-------------------------------------------------------------------------------
# SPH K-means
#-------------------------------------------------------------------------------

PI = 3.141592684

def Accuracy():
    return rand_score(y,model.mode)
track = [Accuracy]

def biasToClamp(bias):
    out = np.zeros(len(bias))
    for i in range(len(bias)):
        out[i] = np.sum(bias[i])
    return out.astype(bool)

acc = []
etime = []

yMatrix = mlo.Matrix(len(y),1,ctype="int")
yMatrix.numpy = y.reshape((len(y),1))

hyper = mlo.Hyperbolic()
zero = mlo.Matrix(X.rows,X.cols,0)
Xhyp = hyper.exp(zero,X)

for sp in clamps:
    print(sp)
    clamped = np.loadtxt(str_clamps+"/"+sp)
    clamped = clamped.astype(np.uint)
    model = models.SphericalKmeans(Xhyp,None,clamped)
    model.seed(q)
    model.clampTo(yMatrix)
    start = time.time()
    t, tracked = model.train(tmax=1000,precision=-1,track=track)
    etime.append(time.time()-start)
    acc.append(tracked[0])

n = len(acc)
acc_mean = np.zeros(len(acc[0]))
etime_mean = np.zeros(len(acc[0]))
for i in range(n):
    acc_mean += acc[i]
    etime_mean += etime[i]
acc_mean /= n
etime_mean /= n

axs[0].plot(etime_mean,label="SHR K-means")
axs[1].plot(acc_mean,ls="dashed")

#-------------------------------------------------------------------------------
# Final plotting adjustments
#-------------------------------------------------------------------------------

axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_xscale("log")

axs[1].set_xlabel("Número de iterações")
axs[1].set_ylabel("Acurácia")
axs[0].set_ylabel("Tempo de execução (s)")
axs[0].legend()

plt.show()