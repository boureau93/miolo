import graph_models as models
import miolo as mlo
from sklearn.metrics import rand_score
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import time

#str_graph = input("Graph file: ")
str_x = input("Features: ")
str_y = input("Ground truth: ")
str_bias = input("Bias folder: ")
str_clamps = input("Clamps folder: ")


#G = mlo.txtGraph(str_graph)
X = mlo.txtMatrix(str_x)
print("aqui1")
E = mlo.Euclidean()
print("aqui2")
D = E.distance(X)
print("aqui3")
k = np.uintc(X.rows*np.log(X.rows)+X.rows)
G = D.knn(k)
print("aqui4")
G = G.similarityJW()
print("aqui5")

y = np.loadtxt(str_y)
q = max(y)+1
biases = listdir(str_bias)
clamps = listdir(str_clamps)

fig, axs = plt.subplots(2,sharex=True)

#-------------------------------------------------------------------------------
# LGC
#-------------------------------------------------------------------------------

def Accuracy():
    return rand_score(y,model.labels)
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

"""
#-------------------------------------------------------------------------------
# Hard LP
#-------------------------------------------------------------------------------
def Accuracy():
    return rand_score(y,model.labels)
track = [Accuracy]

acc = []
etime = []

for sp in clamps:
    print(sp)
    bias = np.loadtxt(str_clamps+"/"+sp)
    model = models.hardLP(G,bias)
    model.clampTo(y)
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

axs[0].plot(etime_mean,label="HLP")
axs[1].plot(acc_mean)
"""
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