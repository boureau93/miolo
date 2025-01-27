#include "mtx.h"
#include <cmath>

#ifndef EUCLIDEAN_H
#define EUCLIDEAN_H

class euclidean{
public:

    template <typename T>
    mtx<T>* mean(mtx<T>& M);
    template <typename T>
    mtx<T>* variance(mtx<T>& M);
    template <typename T>
    mtx<T>* dot(mtx<T>& M);
    template <typename T>
    mtx<T>* distance(mtx<T>& M);

    template <typename T>
    mtx<T>* minmaxNormalize(mtx<T>& M);
    template <typename T>
    mtx<T>* gaussianNormalize(mtx<T>& M);
    template <typename T>
    mtx<T>* rowNormalize(mtx<T>& M);
    template <typename T>
    mtx<T>* colNormalize(mtx<T>& M);
};

template <typename T>
mtx<T>* euclidean::mean(mtx<T>& M){
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
mtx<T>* euclidean::variance(mtx<T>& M){
    mtx<T>* mean = this->mean(M);
    if (mean==nullptr)
        return nullptr;
    mtx<T>* var = new mtx<T>(1,M.cols,0);
    for (ulong j=0; j<M.cols; j++){
        for (ulong i=0; i<M.rows; i++){
            T aux = (*mean)(0,j)-M(i,j);
            (*var)(0,j) += aux*aux;
        }
        (*var)(0,j) /= M.rows;
    }
    return var;
}

template <typename T>
mtx<T>* euclidean::dot(mtx<T>& A){
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
mtx<T>* euclidean::distance(mtx<T>& A){
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

template <typename T>
mtx<T>* euclidean::minmaxNormalize(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong j=0; j<M.cols; j++){
        T max = M(0,j);
        T min = M(0,j);
        for (ulong i=0; i<M.rows; i++){
            if (M(i,j)>max)
                max = M(i,j);
            if (M(i,j)<min)
                min = M(i,j);
        }
        if (max==min){
            cout << "[miolo Warning] ";
            cout << "Values for min and max of column coincide in ";
            cout << "Euclidean.minmaxNormalize.";
            cout << " - > Avoiding division by zero and returning nullptr." << endl;
            return nullptr;
        }
        T div = max-min;
        for (ulong i=0; i<M.rows; i++){
            (*out)(i,j) = (M(i,j)-min)/div;
        }
    }
    return out;
}

template <typename T>
mtx<T>* euclidean::rowNormalize(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T z = 0;
        for (ulong j=0; j<M.cols; j++){
            z += M(i,j);
        }
        if (z==0){
            cout << "[miolo Warning] ";
            cout << "Normalizing factor is zero in Euclidean.rowNormalize.";
            cout << " - > Avoiding division by zero and returning nullptr.";
            return nullptr;
        }
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = M(i,j)/z;
        }
    }
    return out;
}

template <typename T>
mtx<T>* euclidean::colNormalize(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong j=0; j<M.cols; j++){
        T z = 0;
        for (ulong i=0; i<M.rows; i++){
            z += M(i,j);
        }
        if (z==0){
            cout << "[miolo Warning] ";
            cout << "Normalizing factor is zero in Euclidean.colNormalize.";
            cout << " - > Avoiding division by zero and returning nullptr.";
            return nullptr;
        }
        for (ulong i=0; i<M.rows; i++){
            (*out)(i,j) = M(i,j)/z;
        }
    }
    return out;
}

template <typename T>
mtx<T>* euclidean::gaussianNormalize(mtx<T> &M){
    mtx<T>* mean = this->mean(M);
    mtx<T>* var = this->variance(M);
    if (mean==nullptr || var == nullptr)
        return nullptr;
    for (ulong j=0; j<M.cols; j++)
        if ((*var)(0,j)==0){
            cout << "[miolo Warning]";
            cout << " Zero variance in Euclidean.gaussianNormalize.";
            cout << " - > Avoiding division by zero and returning nullptr.";
            return nullptr;
        }
    
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.cols; j++)
            (*out)(i,j) = (M(i,j)-(*mean)(0,j))/(*var)(0,j);
    }
    return out;
}

#endif