import numpy as np
import miolo as mlo

def conjugateGradient(b,A,x_0,tmax=10,precision=0.005):
    res = b-A&x_0
    p = mlo.Matrix(res.rows,res.cols)
    p.copy(res)
    x = mlo.Matrix(res.rows,res.cols)
    x.copy(x_0)
    #Control variables
    t = 0
    delta = mlo.dot(res,res)
    while t<tmax and delta>precision:
        aux = A&p
        alpha = mlo.dot(res,res)/mlo.dot(p,aux)
        x = x+alpha*p
        res = res-alpha*aux
        beta = 1./delta
        delta = mlo.dot(res,res)
        beta *= delta
        p = res+beta*p
        t += 1
    return x
