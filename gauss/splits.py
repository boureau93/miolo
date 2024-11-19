import numpy as np

str_labels = input("Labels: ")
labels = np.loadtxt(str_labels).astype(np.uint)
N = len(labels)
q = int(max(labels)+1)

n_splits = eval(input("Number of splits: "))
r_l = eval(input("Rate of labeled data: "))

for n in range(n_splits):
    bias = np.zeros((N,q))
    N_l = int(r_l*N)
    in_split = []
    k = 0
    while k<N_l:
        i = np.random.randint(0,N)
        if i not in in_split:
            bias[i][int(labels[i])] = 1
            in_split.append(i)
            k += 1
    np.savetxt("bias_"+str(n)+".dat",bias)