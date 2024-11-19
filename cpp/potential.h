#include "mtx.h"
#include "graph.h"
#include <cstddef>

#ifndef POTENTIAL_H
#define POTENTIAL_H

template <typename T>
class ufunc{
public:
    T (*f)(T x, T y);
    T(*dfx)(T x, T y);
};

template <typename T>
T sqrdiff(T x, T y){
    return (x-y)*(x-y);
}
template <typename T>
T dsqrdiff(T x, T y){
    return 2*(x-y);
}

template <typename T>
T prod(T x, T y){
    return -x*y;
}
template <typename T>
T dprod(T x, T y){
    return -y;
}

template <typename T>
T loglike(T x, T y){
    if (x>0)
        return y*log(x);
    else{
        cout << "[miolo Warning] ";
        cout << "Preventing invalid logarithm in Potential.Loglikelihood.";
        cout << endl;
        return 0;
    }
}
template <typename T>
T dloglike(T x, T y){
    if (x>0)
        return y/x;
    else{
        cout << "[miolo Warning] ";
        cout << "Preventing invalid gradient in Potential.Loglikelihood.";
        cout << endl;
        return 0;
    }
}

template <typename T>
class phiPotential {
public:

    ufunc<T> interaction;

    T cost(mtx<T> Phi, mtx<T> Bias);
    T cost(mtx<T> Phi, graph<T> G);

    mtx<T>* grad(mtx<T> Phi, mtx<T> Bias);
    mtx<T>* grad(mtx<T> Phi, graph<T> G);

};

template <typename T>
T phiPotential<T>::cost(mtx<T> Phi, mtx<T> Bias){
    T out = 0;
    for (ulong i=0; i<Phi.rows; i++)
        for (ulong j=0; j<Phi.cols; j++){
            out += this->interaction.f(Phi(i,j),Bias(i,j));
        }
    return out;
}

template <typename T>
T phiPotential<T>::cost(mtx<T> Phi, graph<T> G){
    T out = 0;
    for (ulong k=0; k<G.edges; k++)
        for (ulong s=0; s<Phi.cols; s++){
            out += G[k].w*this->interaction.f(
                Phi(G[k].i,s),Phi(G[k].j,s)
            );
        }
    return out;
}

template <typename T>
mtx<T>* phiPotential<T>::grad(mtx<T> Phi, mtx<T> Bias){
    mtx<T>* out = new mtx<T>(Phi.rows,Phi.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<Phi.rows; i++)
        for (ulong j=0; j<Phi.cols; j++){
            (*out)(i,j) = this->interaction.dfx(Phi(i,j),Bias(i,j));
        }
    return out;
}

template <typename T>
mtx<T>* phiPotential<T>::grad(mtx<T> Phi, graph<T> G){
    mtx<T>* out = new mtx<T>(Phi.rows,Phi.cols,0);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<G.edges; k++){
        for (ulong s=0; s<Phi.cols; s++){
            (*out)(G[k].i,s) += G[k].w*this->interaction.dfx(
                Phi(G[k].i,s),Phi(G[k].j,s)
            );
            if (G[k].i!=G[k].j){
                (*out)(G[k].j,s) += G[k].w*this->interaction.dfx(
                    Phi(G[k].j,s),Phi(G[k].i,s)
                );
            }
        }
    }
    return out;
}

/*------------------------------------------------------------------------------
    
------------------------------------------------------------------------------*/

#endif 