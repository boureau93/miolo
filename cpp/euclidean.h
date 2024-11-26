#include "mtx.h"
#include <cmath>

template <typename T>
mtx<T>* mean(mtx<T> M){
    mtx<T>* out = new mtx<T>(1,M.cols,0);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong s=0; s<M.cols; s++){
            (*out)(0,s) += M(i,s);
        }
    }
    for (ulong s=0; s<M.cols; s++)
        (*out)(0,s) /= M.rows;
    return out;
}

template <typename T>
mtx<T>* dot(mtx<T> A){
    mtx<T>* out = new mtx<T>(A.rows,A.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<A.rows; i++){
        for (ulong j=i; A.rows; j++){
            (*out)(i,j) = 0;
            for (ulong s=0; s<A.cols; s++){
                (*out)(i,j) += A(i,s)*A(j,s);
            }
        }
    }
    return out;
}

template <typename T>
mtx<T>* distance(mtx<T> A){
    mtx<T>* out = new mtx<T>(A.rows,A.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<A.rows; i++){
        (*out)(i,i) = 0;
        for (ulong j=i+1; j<A.rows; j++){
            (*out)(i,j) = 0;
            for (ulong s=0; s<A.cols; s++){
                T aux = A(i,s)-A(j,s); 
                (*out)(i,j) = aux*aux;
            }
            (*out)(i,j) = sqrt((*out)(i,j));
            (*out)(j,i) = (*out)(i,j);
        }
    }
    return out;
}