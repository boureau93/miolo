#include "mtx.h"
#include <cmath>

class kmeans{
public:

    template <typename T>
    mtx<T>* euclideanCentroidDistance(mtx<T>& data, mtx<T>& center);
    template <typename T>
    mtx<T>* euclideanCentroid(mtx<T>& data, mtx<int>& labels);
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