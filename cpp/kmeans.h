#include "mtx.h"
#include <complex>

class kmeans{
public:

    mtx<bool>* clamped;
    mtx<bool>* labels;
    unsigned long N, q;

    kmeans();
    kmeans(ulong N, ulong q);
    kmeans(mtx<bool>*, ulong N, ulong q);
    ~kmeans();

    void clamp(int*);
    mtx<int>* getLabels();
    template<typename T> 
    void setLabels(mtx<T> data, mtx<T> means);

    template <typename T>
    mtx<T>* groupFeatures(mtx<T> data, unsigned long gNum);

};

kmeans::kmeans(){
    this->clamped = nullptr;
    this->labels = nullptr;
    this->q = 0;
}

kmeans::kmeans(ulong N, ulong q){
    this->clamped = nullptr;
    this->q = q;
    this->N = N;
    this->labels = new mtx<bool>(N,q);
}

kmeans::kmeans(mtx<bool>* clamped, ulong N, ulong q){
    this->clamped = clamped;
    this->q = q;
    this->N = N;
    this->labels = new mtx<bool>(N,q);
}

kmeans::~kmeans(){
    if (this->clamped!=nullptr)
        delete this->clamped;
    if (this->labels!=nullptr)
        delete this->labels;
}

void kmeans::clamp(int* targets){
    if (this->clamped==nullptr || targets==nullptr || this->labels->null())
        return;
    for (ulong i=0; i<this->clamped->rows; i++){
        if ((*this->clamped)(i,0)){
            for (ulong s=0; s<this->labels->cols; s++){
                if (s==(ulong)targets[i])
                    (*this->labels)(i,s) = true;
                else
                    (*this->labels)(i,s) = false;
            }
        }
    }
}

mtx<int>* kmeans::getLabels(){
    mtx<int>* out = new mtx<int>(this->labels->rows,1);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<this->labels->rows; i++){
        for (ulong s=0; s<this->labels->cols; s++){
            if ((*this->labels)(i,s))
                (*out)(i,0) = s;
        }
    }
    return out;
}

template <typename T>
void kmeans::setLabels(mtx<T> data, mtx<T> means){
    mtx<T> dists(data.rows,means.rows,0);
    if (dists.null())
        return;
    for (ulong i=0; i<data.rows; i++){
        for (ulong s=0; s<means.rows; s++){
            for (ulong d=0; d<means.cols; d++){
                T aux = data(i,d)-means(s,d);
                dists(i,s) += aux*aux;
            }
        }
    }
    mtx<int> arg; arg.wrap(argmin(&dists));
    if (arg.null())
        return;
    if (this->clamped==nullptr){
        for (ulong i=0; i<this->N; i++){
            (*this->labels)(i,arg.data[i]) = true;
        }
    }
    else{
        for (ulong i=0; i<this->N; i++){
            if (!this->clamped->data[i])
               (*this->labels)(i,arg.data[i]) = true;
        }
    }
}

template <typename T>
mtx<T>* kmeans::groupFeatures(mtx<T> data, unsigned long gNum){
    ulong count = 0;
    for (ulong i=0; i<this->labels->rows; i++){
        if ((*this->labels)(i,gNum))
            count++;
    }
    mtx<T>* out = new mtx<T>(count,data.cols);
    if (out->null())
        return nullptr;
    ulong k=0;
    for (ulong i=0; i<data.rows; i++){
        if ((*this->labels)(i,gNum)){
            for (ulong s=0; s<data.cols; s++){
                (*out)(k,s) += data(i,s);
            }
            k++;
        }
    }
    return out;
}