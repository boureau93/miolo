#include "mtx.h"
#include "graph.h"

#ifndef ALGEBRA_H
#define ALGEBRA_H

template <typename T>
mtx<T>* operator+(mtx<T>& A, mtx<T>& B){
    mtx<T>* out;
    out = A.add(&B);
    return out;
}

template <typename T>
mtx<T>* operator-(mtx<T>& A, mtx<T>& B){
    mtx<T>* out;
    out = A.sub(&B);
    return out;
}

template <typename T>
mtx<T>* operator*(mtx<T>& A, T value){
    mtx<T>* out;
    out = A.smul(value);
    return out;
}

template <typename T>
mtx<T>* operator*(T value, mtx<T>& A){
    mtx<T>* out;
    out = A.smul(value);
    return out;
}

template <typename T>
mtx<T>* operator/(mtx<T>& A, T value){
    if (value==0){
        cout << "[miolo][mtx<T>] Attempting division by zero in division by scalar." << endl;
    }
    mtx<T>* out;
    out = A.smul(1./value);
    return out;
}

template<typename T>
mtx<T>* operator*(mtx<T>& A, mtx<T>& B){
    mtx<T>* out;
    out = A.mmul(&B);
    return out;
}

template <typename T>
mtx<T>* operator*(mtx<T> A, graph<T>& B){
    mtx<T> *out;
    out = B.mmul(&A);
    return out;
}

template <typename T>
mtx<T>* operator*(graph<T>& B, mtx<T> A){
    mtx<T>* out;
    out = B.mmul(&A);
    return out;
}

template <typename T>
graph<T>* operator*(graph<T>& B, T value){
    graph<T>* out;
    out = B.smul(value);
    return out;
}

template <typename T>
graph<T>* operator*(T value, graph<T>& B){
    graph<T>* out;
    out = B.smul(value);
    return out;
}

template <typename T>
T sumAll(mtx<T>& A){
    T out = 0;
    for (ulong k=0; k<A.rows*A.cols; k++){
        out += A[k];
    }
    return out;
}

#endif