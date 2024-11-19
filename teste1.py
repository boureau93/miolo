import models
import miolo as mlo
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import time

str_graph = input("Graph file: ")
str_y = input("Ground truth: ")
str_splits = input("Splits folder: ")

G = mlo.txtGraph(str_graph)
y = np.loadtxt(str_y)
splits = listdir(str_splits)

fig, axs = plt.subplots(2,sharex=True)

#-------------------------------------------------------------------------------
# LGC
#-------------------------------------------------------------------------------

def Accuracy():
    return accuracy_score(y,model.mode())
def simplexBias(bias):
    N, q = bias.shape
    for i in range(N):
        if np.sum(bias[i])<0.5:
            bias[i] = np.ones(q)/q
    return bias

track = [Accuracy]

acc = []
etime = []

for sp in splits:
    print(sp)
    bias = mlo.txtMatrix(str_splits+"/"+sp)
    #bias.numpy = simplexBias(bias.numpy)
    model = models.LGC(G,bias)
    start = time.time()
    t, tracked = model.train(tmax=10,precision=-1,track=track)
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

axs[0].plot(etime_mean,label="LGC, Esparso")
axs[1].plot(acc_mean)

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