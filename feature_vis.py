import numpy as np
import miolo as mlo
import matplotlib.pyplot as plt

mlo.global_ctype = "double"

str_X = input("Features: ")
X = mlo.txtMatrix(str_X)
str_y = input("Labels: ")
y = np.loadtxt(str_y)
y = y.astype(np.intc)

#Euclidean
mexp = mlo.exp()
E = mlo.Euclidean()
X = E.gaussianNormalize(X)
dists = E.distance(X)
G = dists.sparsifyKNN(8,"Graph")
sigma = G.gaussianScale()
G = mlo.hadamard(G,G)
G /= -2*sigma*sigma
G = mexp(G)
print(G.weights)

#plt.axhline(d_max,ls="dashed",color="blue")

plt.xscale("log")
plt.show()