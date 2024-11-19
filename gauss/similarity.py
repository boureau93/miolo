import numpy as np

str_data = input("Enter dataset for similarity calculation: ")
data = np.loadtxt(str_data)

n, m = data.shape
knn = int(np.log2(n))

dist = np.empty((n,n))

for i in range(n):
    for j in range(n):
        dist[i][j] = np.dot(data[i]-data[j],data[i]-data[j])

sigma = 0
edges = []
for i in range(n):
    aux = np.argsort(dist[i])
    aux_max = 0
    for k in range(1,knn+1):
        if [aux[k],i] not in edges:
            edges.append([i,aux[k]])
            if dist[i][aux[k]]>aux_max:
                aux_max = dist[i][aux[k]]
    sigma += np.sqrt(aux_max) 
sigma /= 3*n

for e in edges:
    e.append(np.exp(-dist[e[0]][e[1]]/(2*sigma*sigma)))

N_edges = len(edges)
np.savetxt("weights.dat",edges)