#include "mtx.h"
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

class simplex{
public:

    template <typename T>
    mtx<T>* tangentProjection(mtx<T> base, mtx<T> M);
    template <typename T>
    mtx<T>* softmaxRetraction(mtx<T> at, mtx<T> M);
    
    template <typename T>
    bool isIn(mtx<T> M, T tol);

    template <typename T>
    bool isTangent(mtx<T> at, mtx<T> M, T tol);

};

template <typename T>
mtx<T>* simplex::tangentProjection(mtx<T> base, mtx<T> M){
    mtx<T>* out = new mtx<T>(base.rows,base.cols);
    if (out->null())
        return nullptr;
    T prod = 0;
    for (ulong i=0; i<M.rows*M.cols; i++){
        prod += base.data[i]*M.data[i];
    }
    for (ulong k=0; k<M.rows*M.cols; k++){
        out->data[k] = M.data[k]-prod*base.data[k];
    }
    return out;
}

template <typename T>
mtx<T>* simplex::softmaxRetraction(mtx<T> at, mtx<T> M){
    mtx<T>* out = new mtx<T>(at.rows,at.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<at.rows; i++){
        T z = 0;
        for (ulong j=0; j<at.cols; j++){
            (*out)(i,j) = exp(M(i,j))*at(i,j);
            z += (*out)(i,j);
        }
        for (ulong j=0; j<at.cols; j++){
            (*out)(i,j) /= z;
        }
    }
    return out;
}

template <typename T>
bool simplex::isIn(mtx<T> M, T tol){
    for (ulong i=0; i<M.rows; i++){
        T sum = 0;
        for (ulong s=0; s<M.cols; s++){
            if (M(i,s)<0)
                return false;
            sum += M(i,s);
        }
        if (!(sum<1+tol || sum>1-tol))
            return false;
    }
    return true;
}

template <typename T>
bool simplex::isTangent(mtx<T> at, mtx<T> M, T tolerance){
    for (ulong i=0; i<at.rows; i++){
        T sum = 0;
        for (ulong j=0; j<at.cols; j++){
            sum += at(i,j)*M(i,j);
        }
        if (sum>tolerance || sum<-tolerance)
            return false;
    }
    return true;
}