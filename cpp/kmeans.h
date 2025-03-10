#include "mtx.h"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace std;

class kmeans{
public:

    template <typename T>
    mtx<T>* euclideanCentroidDistance(mtx<T>& data, mtx<T>& center);
    template <typename T>
    mtx<T>* sphereCentroidDistance(mtx<T>& data, mtx<T>& center);
    template <typename T>
    mtx<T>* hyperbolicCentroidDistance(mtx<T>& data, mtx<T>& center);
    template <typename T>
    mtx<T>* euclideanCentroid(mtx<T>& data, mtx<int>& labels);
    template <typename T>
    mtx<T>* partition(mtx<T>& data, mtx<int>& labels, int idLabel);
    template <typename T>
    mtx<T>* kmpp(mtx<T>& data, int k);
};

template <typename T>
mtx<T>* kmeans::euclideanCentroidDistance(mtx<T>& data, mtx<T>& center){
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
mtx<T>* kmeans::sphereCentroidDistance(mtx<T>& data, mtx<T>& center){
    mtx<T>* out = new mtx<T>(data.rows,center.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<out->rows; i++){
        for (ulong j=0; j<out->cols; j++){
            (*out)(i,j) = 0;
            for (ulong s=0; s<data.cols; s++){
                (*out)(i,j) += data(i,s)*center(j,s);
            }
            (*out)(i,j) = acos((*out)(i,j));
        }
    }
    return out;
}

template <typename T>
mtx<T>* kmeans::hyperbolicCentroidDistance(mtx<T>& data, mtx<T>& center){
    mtx<T>* out = new mtx<T>(data.rows,center.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<out->rows; i++){
        for (ulong j=0; j<out->cols; j++){
            T aux[3] = {0,0,0};
            for (ulong s=0; s<data.cols; s++){
                aux[0] += (data(i,s)-center(j,s))*(data(i,s)-center(j,s));
                aux[1] += data(i,s)*data(i,s);
                aux[2] += center(j,s)*center(j,s);
            }
            (*out)(i,j) = acosh(1+2*aux[0]/((1-aux[1])*(1-aux[2])));
        }
    }
    return out;
}

template <typename T>
mtx<T>* kmeans::euclideanCentroid(mtx<T>& data, mtx<int>& labels){
    int numLabels = labels.max()+1;
    ulong* labelFreq = new ulong[numLabels];
    mtx<T>* out = new mtx<T>(numLabels,data.cols);
    if (out->null() || labelFreq==nullptr)
        return nullptr;
    for (int p=0; p<numLabels; p++){
        labelFreq[p] = 0;
    }
    for (ulong i=0; i<data.rows; i++){
        for (ulong s=0; s<data.cols; s++){
            (*out)(labels[i],s) += data(i,s);
            labelFreq[labels[i]]++;
        }
    }
    for(ulong k=0; k<out->rows; k++){
        for (ulong s=0; s<out->cols; s++){
            (*out)(k,s) /= labelFreq[k];
        }
    }
    delete[] labelFreq;
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
mtx<T>* kmeans::kmpp(mtx<T> &data, int k){
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