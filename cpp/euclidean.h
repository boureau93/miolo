#include "mtx.h"
#include "graph.h"
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

    template <typename T>
    mtx<T>* centroidDistance(mtx<T>& M, mtx<T>& centroids);
    template <typename T>
    mtx<T>* kmpp(mtx<T>& M, int k);

    template <typename T>
    void conjugateGradient(mtx<T>& M, mtx<T>& b, mtx<T>& x_0, int tmax, double precision);
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
            cout << " Zero variance in Euclidean.gaussianNormalize." << endl;
            cout << " << Avoiding division by zero and returning nullptr. >>";
            cout << endl;
            return nullptr;
        }
    
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.cols; j++)
            (*out)(i,j) = (M(i,j)-(*mean)(0,j))/(*var)(0,j);
    }
    delete mean; delete var;
    return out;
}

template <typename T>
mtx<T>* euclidean::centroidDistance(mtx<T>& data, mtx<T>& center){
    mtx<T>* out = new mtx<T>(data.rows,center.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<out->rows; i++){
        for (ulong j=0; j<out->cols; j++){
            (*out)(i,j) = 0;
            for (ulong s=0; s<data.cols; s++){
                T aux = data(i,s)-center(j,s);
                (*out)(i,j) += aux*aux;
            }
        }
    }
    return out;
}

template <typename T>
bool isIn(T* vec, T value, int size){
    for (int i=0; i<size; i++){
        if (vec[i]==value)
            return true;
    }
    return false;
}

template <typename T>
mtx<T>* euclidean::kmpp(mtx<T> &data, int k){
    mtx<T>* out = new mtx<T>(k,data.cols);
    if (out->null())
        return nullptr;
    srand(time(0));
    //Choose first centroid
    ulong id = rand()%data.rows;
    for (ulong l=0; l<data.cols; l++){
        (*out)(0,l) = data(id,l);
    }
    ulong chosen[k]; chosen[0] = id;
    //Determine next centroids
    int aux_k = 1;
    while (aux_k<k){
        mtx<double> dist(data.rows,1);
        T Z = 0; //normalizing constant
        for (ulong i=0; i<data.rows; i++){
            //If already chosen, zero distance
            if (isIn(chosen,i,aux_k)){
                dist[i] = 0;
            }
            //Else, calculate shortest distance
            else{
                dist[i] = 100000;
                for (int a=0; a<aux_k; a++){
                    T new_dist = 0;
                    for (ulong p=0; p<data.cols; p++){
                        new_dist += 
                            (data(i,p)-(*out)(a,p))*(data(i,p)-(*out)(a,p));
                    }
                    if (new_dist<dist[i])
                        dist[i] = new_dist;
                }
            }
            Z += dist[i];
        }
        //Choose next centroid
        double u = ((double)rand())/((double)RAND_MAX);
        double sum = 0;
        for (ulong i=-1; i<data.rows-1; i++){
            if (u>sum && u<sum+dist[i+1]/Z){
                id = i+1; break;
            }
            else{
                sum += dist[i+1]/Z;
            }
        }
        //Insert centrod in out
        for (ulong l=0; l<data.cols; l++){
            (*out)(aux_k,l) = data(id,l);
        }
        aux_k++;
    }
    return out;
}


#endif