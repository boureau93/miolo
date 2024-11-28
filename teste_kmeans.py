import models
import miolo as mlo
from sklearn.metrics import accuracy_score, rand_score
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import time

mlo.global_ctype="double"

X = mlo.txtMatrix("gauss/xgauss.dat",ctype="double")
C = mlo.Matrix(2,2,ctype="double")
C.numpy = np.array([[-1.,-1.],[1.,1.]],dtype=np.float64)
C.print()
K = mlo.KmeansUtil()
P = K.centroidDistance(X,C)
P.print()