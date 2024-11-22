#include "mtx.h"
#include "graph.h"
#include "digraph.h"

#ifndef DIAGONAL_H
#define DIAGONAL_H

template <typename T>
class diagonal {
public:

    ulong dim;
    T* data;

    diagonal();
    diagonal(ulong dim);
    diagonal(ulong dim, T init);
    ~diagonal();

    bool null(){
        return this->data==nullptr;
    }

    mtx<T>* lmul(mtx<T> M);
    mtx<T>* rmul(mtx<T> M);

    digraph<T>* lmul(digraph<T> G);
    digraph<T>* rmul(digraph<T> G);

    digraph<T>* lmul(graph<T> G);
    digraph<T>* rmul(graph<T> G);

    diagonal<T>* mul(diagonal<T> D);
    diagonal<T>* add(diagonal<T> D);
    diagonal<T>* sub(diagonal<T> D);
    diagonal<T>* smul(T value);

};

template <typename T>
diagonal<T>::diagonal(){
    this->dim = 0;
    this->data = nullptr;
}

template <typename T>
diagonal<T>::diagonal(ulong dim){
    this->dim = dim;
    this->data = new T[dim];
}

template <typename T>
diagonal<T>::diagonal(ulong dim, T init){
    this->dim = dim;
    this->data = new T[dim];
    if (!this->null()){
        for (ulong i=0; i<dim; i++)
            this->data[i] = init;
    }
}

template <typename T>
diagonal<T>::~diagonal(){
    if (this->data!=nullptr)
        delete[] this->data;
}

template <typename T>
mtx<T>* diagonal<T>::lmul(mtx<T> M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++)
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = M(i,j)*this->data[i];
        }
    return out;
}

template <typename T>
mtx<T>* diagonal<T>::rmul(mtx<T> M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++)
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = M(i,j)*this->data[j];
        }
    return out;
}

template <typename T>
digraph<T>* diagonal<T>::lmul(digraph<T> G){
    digraph<T>* out = G.copyStructure();
    if (out->null())
        return nullptr;
    for (ulong i=0; i<G.nodes; i++){
        for (ulong k=0; k<G.hood[i].size(); k++){
            out->hood[i][k].w = G.hood[i][k].w*this->data[i];
        }
    }
    return out;
}

template <typename T>
digraph<T>* diagonal<T>::rmul(digraph<T> G){
    digraph<T>* out = G.copyStructure();
    if (out->null())
        return nullptr;
    for (ulong i=0; i<G.nodes; i++){
        for (ulong k=0; k<G.hood[i].size(); k++){
            out->hood[i][k].w = G.hood[i][k].w*this->data[G.hood[i][k].idx];
        }
    }
    return out;
}

template <typename T>
digraph<T>* diagonal<T>::lmul(graph<T> G){
    digraph<T>* aux = toDigraph(G);
    if (aux==nullptr)
        return nullptr;
    digraph<T>* out = this->lmul(*aux);
    delete aux;
    return out;
}

template <typename T>
digraph<T>* diagonal<T>::rmul(graph<T> G){
    digraph<T>* aux = toDigraph(G);
    if (aux==nullptr)
        return nullptr;
    digraph<T>* out = this->rmul(*aux);
    delete aux;
    return out;
}

template <typename T>
diagonal<T>* diagonal<T>::mul(diagonal<T> D){
    diagonal<T>* out = new diagonal<T>(D.dim);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<D.dim; k++){
        out->data[k] = this->data[k]*D.data[k];
    }
    return out;
}

template <typename T>
diagonal<T>* diagonal<T>::add(diagonal<T> D){
    diagonal<T>* out = new diagonal<T>(D.dim);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<D.dim; k++){
        out->data[k] = this->data[k]+D.data[k];
    }
    return out;
}

template <typename T>
diagonal<T>* diagonal<T>::sub(diagonal<T> D){
    diagonal<T>* out = new diagonal<T>(D.dim);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<D.dim; k++){
        out->data[k] = this->data[k]-D.data[k];
    }
    return out;
}

template <typename T>
diagonal<T>* diagonal<T>::smul(T value){
    diagonal<T>* out = new diagonal<T>(this->dim);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<this->dim; k++){
        out->data[k] = this->data[k]*value;
    }
    return out;
}

#endif