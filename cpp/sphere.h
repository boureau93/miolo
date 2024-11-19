#include "mtx.h"
#include <cmath>

template <typename T>
class sphere{
public:
    
    T r; //radius

    mtx<T>* stereographicProjection(mtx<T> M);
    mtx<T>* tangentProjection(mtx<T> base, mtx<T> M);

    mtx<T>* distance(mtx<T> M);
    T distance(T); 

    mtx<T>* fromEuclidean(mtx<T> M);
    mtx<T>* toEuclidean(mtx<T> M);

    bool isIn(mtx<T> M, T tolerance);

    bool isTangent(mtx<T> at, mtx<T> M, T tolerance);
};

template <typename T>
mtx<T>* sphere<T>::stereographicProjection(mtx<T> M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    for (ulong i=0; i<M.rows; i++){
        T z = 0;
        for (ulong s=0; s<M.cols; s++){
            z += M(i,s)*M(i,s);
        }
        z = sqrt(z);
        if (z>0)
            for (ulong s=0; s<M.cols; s++){
                (*out)(i,s) = this->r*(M(i,s)/z);
            }
        else{
            cout << cout << "[miolo Warning] Preventing division by zero." << endl;
            cout << "<< This ocurred in Spheres.stereographicProjection at " ;
            cout << "row " << i << " >>" << endl;
        }
    }
    return out;
}

template <typename T>
mtx<T>* sphere<T>::tangentProjection(mtx<T> base, mtx<T> M){
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
mtx<T>* sphere<T>::fromEuclidean(mtx<T> M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols+1,this->r);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.cols+1; j++){
            for (ulong s=0; s<j; s++)
                (*out)(i,j) *= sin(M(i,s));
            if (j<M.cols)
                (*out)(i,j) *= cos(M(i,j));
            else
                (*out)(i,j) *= sin(M(i,j)); 
        }
    }
    return out;
}

template <typename T>
mtx<T>* sphere<T>::toEuclidean(mtx<T> M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols-1.1);
    if (out->null())
        return nullptr;
    T aux = this->r;
    for (ulong i=0; i<M.rows; i++)
        for (ulong j=0; j<M.cols-1; j++){
            aux -= M(i,j)*M(i,j);
            (*out)(i,j) = atan2(sqrt(aux),M(i,j));
        }
    return out;
}

template <typename T>
bool sphere<T>::isIn(mtx<T> M, T tolerance){
    for (ulong i=0; i<M.rows; i++){
        T sum = 0;
        for (ulong j=0; j<M.cols; j++){
            sum += M(i,j)*M(i,j);
        }
        if (sum>this->r+tolerance || sum<this->r-tolerance)
            return false;
    }
    return true;
}

template <typename T>
bool sphere<T>::isTangent(mtx<T> at, mtx<T> M, T tolerance){
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

template <typename T>
mtx<T>* sphere<T>::distance(mtx<T> M){
    mtx<T>* out = new mtx<T>(M.rows,M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        (*out)(i,i) = 0;
        for (ulong j=i+1; j<M.rows; i++){
            T prod = 0;
            for (ulong s=0; s<M.rows; s++)
                prod += M(i,s)*M(j,s);
            (*out)(i,j) = this->r*acos(prod/(this->r*this->r));
            (*out)(j,i) = (*out)(i,j);
        }
    }
    return out;
}
