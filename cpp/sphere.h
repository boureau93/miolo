#include "mtx.h"
#include "euclidean.h"
#include <cmath>

#ifndef SPHERE_H
#define SPHERE_H

class sphere{
public:

    double r;
    
    template <typename T>
    mtx<T>* stereographicProjection(mtx<T>& M);
    template <typename T>
    mtx<T>* tangentProjection(mtx<T>& base, mtx<T>& M);

    template <typename T>
    mtx<T>* distance(mtx<T>& M);

    template <typename T>
    mtx<T>* fromEuclidean(mtx<T>& M);
    template <typename T>
    mtx<T>* toEuclidean(mtx<T>& M);
    template <typename T>
    mtx<T>* coordinateReady(mtx<T>& M, ulong azimuth);

    template <typename T>
    mtx<T>* exponential(mtx<T>& at, mtx<T>& M);
    template <typename T>
    mtx<T>* exponential(T* at, mtx<T>& M);
    template <typename T>
    mtx<T>* logarithm(mtx<T>& from, mtx<T>& to);
    template <typename T>
    mtx<T>* logarithm(T* from, mtx<T>& to);

    template <typename T>
    bool isIn(mtx<T>& M, T tolerance);

    template <typename T>
    bool isTangent(mtx<T>& at, mtx<T>& M, T tolerance);

    template <typename T>
    mtx<T>* centroidDistance(mtx<T>& data, mtx<T>& center);
};

template <typename T>
mtx<T>* sphere::stereographicProjection(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    for (ulong i=0; i<M.rows; i++){
        T z = 0;
        for (ulong s=0; s<M.cols; s++){
            z += M(i,s)*M(i,s);
        }
        z = sqrt(z)/this->r;
        if (z>0)
            for (ulong s=0; s<M.cols; s++){
                (*out)(i,s) = M(i,s)/z;
            }
        else{
            cout << "[miolo Warning] Preventing division by zero." << endl;
            cout << "<< This ocurred in Sphere.stereographicProjection at " ;
            cout << "row " << i << " >>" << endl;
        }
    }
    return out;
}

template <typename T>
mtx<T>* sphere::fromEuclidean(mtx<T>& M){
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
mtx<T>* sphere::toEuclidean(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols-1);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.cols-1; j++){
            T aux = 0;
            for (ulong k=j+1; k<M.cols; k++){
                aux += M(i,k)*M(i,k);
            }
            (*out)(i,j) = atan2(sqrt(aux),M(i,j));
        }
    }
    return out;
}

template <typename T>
bool sphere::isIn(mtx<T>& M, T tolerance){
    for (ulong i=0; i<M.rows; i++){
        T sum = 0;
        for (ulong j=0; j<M.cols; j++){
            sum += M(i,j)*M(i,j);
        }
        if (sum>this->r*this->r+tolerance || sum<this->r*this->r-tolerance)
            return false;
    }
    return true;
}

template <typename T>
bool sphere::isTangent(mtx<T>& at, mtx<T>& M, T tolerance){
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
mtx<T>* sphere::distance(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        (*out)(i,i) = 0;
        for (ulong j=i+1; j<M.rows; i++){
            T prod = 0;
            for (ulong s=0; s<M.rows; s++)
                prod += M(i,s)*M(j,s);
            (*out)(i,j) = acos(prod);
            (*out)(i,j) = this->r*(*out)(i,j);
            (*out)(j,i) = (*out)(i,j);
        }
    }
    return out;
}

template <typename T>
mtx<T>* sphere::coordinateReady(mtx<T> &M, ulong azimuth){
    euclidean E;
    double PI = 3.141592684;
    mtx<T>* out = E.colNormalize(M);
    if (out==nullptr)
        return nullptr;
    for (ulong i=0; i<out->rows; i++){
        for (ulong j=0; j<out->cols; j++)
            if (j==azimuth)
                (*out)(i,j) *= 2*PI*0.99;
            else 
                (*out)(i,j) *= PI;
    }
    return out;
}

template <typename T>
mtx<T>* sphere::exponential(mtx<T>& at, mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T norm = 0;
        for (ulong j=0; j<M.cols; j++){
            norm += M(i,j)*M(i,j);
        }
        norm = sqrt(norm);
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = cos(norm/this->r)*at(i,j);
            (*out)(i,j) += sin(norm/this->r)*M(i,j)/norm;
            (*out)(i,j) *= this->r;
        }
    }
    return out;
}

template <typename T>
mtx<T>* sphere::logarithm(mtx<T>& from, mtx<T>& to){
    mtx<T>* out = new mtx<T>(from.rows,from.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<from.rows; i++){
        //Inner product
        T prod = 0;
        for (ulong j=0; j<from.cols; j++){
            prod += from(i,j)*to(i,j);
        }
        T dist = acos(prod);
        T proj_norm = 0;
        //Projection
        for (ulong j=0; j<from.cols; j++){
            (*out)(i,j) = dist*(
                to(i,j)-prod*from(i,j)
            );
            proj_norm += (to(i,j)-prod*from(i,j))*(to(i,j)-prod*from(i,j));
        }
        //Normalization
        for (ulong j=0; j<from.cols; j++){
            (*out)(i,j) /= sqrt(proj_norm);
        }
    }
    return out;
}

template <typename T>
mtx<T>* sphere::centroidDistance(mtx<T>& data, mtx<T>& center){
    mtx<T>* out = new mtx<T>(data.rows,center.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<out->rows; i++){
        for (ulong j=0; j<out->cols; j++){
            (*out)(i,j) = 0;
            for (ulong s=0; s<data.cols; s++){
                (*out)(i,j) += data(i,s)*center(j,s);
            }
            (*out)(i,j) = this->r*acos((*out)(i,j));
        }
    }
    return out;
}

#endif