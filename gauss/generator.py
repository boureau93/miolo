import numpy as np
import matplotlib.pyplot as plt

N = 5000
p = 0.5

data = np.zeros((N,2))
labels = np.zeros(N)

for i in range(N):
    if p > np.random.uniform(0,1):
        data[i][0] = np.random.normal(1,0.6)
        data[i][1] = np.random.normal(1,0.5)
        labels[i] = 0
    else:
        data[i][0] = np.random.normal(-1,0.4)
        data[i][1] = np.random.normal(-1,0.6)
        labels[i] = 1

x = np.transpose(data)[0]
y = np.transpose(data)[1]
plt.plot(x,y,linewidth=0,marker='o')
plt.show()

np.savetxt("xgauss.dat",data)
np.savetxt("ygauss.dat",labels)