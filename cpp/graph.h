#include "mtx.h"
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef GRAPH_H
#define GRAPH_H

template <typename T>
class edge{
public:
    ulong i, j; //index of nodes in edge
    T w; //weight of edge
};

template <typename T>
class graph{
public:
    
    ulong nodes, edges;
    edge<T>* e;

    graph();
    graph(ulong nodes, ulong edges);
    graph(ulong nodes, ulong edges, T init);
    graph(graph<T>& cp);
    ~graph();

    bool null(){return this->e==nullptr;}
    void wrap(graph<T>* target){
        this->nodes = target->nodes;
        this->edges = target->edges;
        this->e = target->e;
    }

    void operator=(graph<T>& cp);
    
    edge<T>& operator[](ulong k){
        return this->e[k];
    }

    bool isolatedNodes();

    //Algebra
    graph<T>* smul(T value);
    mtx<T>* mmul(mtx<T>* M);
    graph<T>* add(graph<T>* G);
    graph<T>* sub(graph<T>* G);
    graph<T>* hmul(graph<T>* G);

    mtx<T>* propagate(mtx<T>* M, bool* clamped);

    mtx<T>* degree();
    void normalize();
    graph<T>* laplacian();

    mtx<T>* densify();
};

/*------------------------------------------------------------------------------
    Constructors and destructor
------------------------------------------------------------------------------*/

template <typename T>
graph<T>::graph(){
    this->nodes = 0; this->edges = 0;
    this->e = nullptr;
}

template <typename T>
graph<T>::graph(ulong nodes, ulong edges){
    this->nodes = nodes; this->edges = edges;
    this->e = new edge<T>[edges];
}

template <typename T>
graph<T>::graph(ulong nodes, ulong edges, T init){
    this->nodes = nodes; this->edges = edges;
    this->e = new edge<T>[edges];
    if (this->e!=nullptr){
        for (ulong k=0; k<this->edges; k++){
            this->e[k].w = init;
        }
    }
}

template <typename T>
graph<T>::graph(graph<T>& cp){
    this->nodes = cp.nodes; this->edges = cp.edges;
    this->e = new edge<T>[cp.edges];
    if (this->e!=nullptr){
        for (ulong k=0; k<this->edges; k++){
            this->e[k].w = cp[k].w;
            this->e[k].i = cp[k].i;
            this->e[k].j = cp[k].j;
        }
    }
}

template <typename T>
graph<T>::~graph(){
    if (this->e!=nullptr){
        delete[] this->e;
    }
}

template <typename T>
void graph<T>::operator=(graph<T> &cp){
    if (this->nodes!=cp.nodes || this->edges!=cp.edges){
        delete[] this->e;
        this->nodes = cp.nodes; this->edges = cp.edges;
        this->e = new edge<T>[cp.edges];
    }
    for (ulong k=0; k<this->edges; k++){
        this->e[k].w = cp[k].w;
        this->e[k].i = cp[k].i;
        this->e[k].j = cp[k].j;
    }
}

/*------------------------------------------------------------------------------
    Utility
------------------------------------------------------------------------------*/

template <typename T>
mtx<T>* graph<T>::degree(){
    mtx<T>* out = new mtx<T>(this->nodes,1,0);
    for (ulong k=0; k<this->edges; k++){
        out->data[this->e[k].i] += this->e[k].w;
        out->data[this->e[k].j] += this->e[k].w; 
    }
    return out;
}

template <typename T>
void graph<T>::normalize(){
    mtx<T>* D = this->degree();
    for (ulong k=0; k<this->edges; k++){
        this->e[k].w /= sqrt(D->data[this->e[k].i]*D->data[this->e[k].j]);
    }
    delete D;
}

template <typename T>
graph<T>* graph<T>::laplacian(){
    graph<T>* out = new graph<T>(this->nodes,this->edges+this->nodes);
    for (ulong k=0; k<this->edges; k++){
        out->e[k].i = this->e[k].i;
        out->e[k].j = this->e[k].j;
        out->e[k].w = -this->e[k].w;
    }
    for (ulong p=0; p<this->nodes; p++){
        out->e[p+this->edges].i = p;
        out->e[p+this->edges].j = p;
        out->e[p+this->edges].w = 1;
    }
    return out;
}

template <typename T>
bool graph<T>::isolatedNodes(){
    bool* aux = new bool[this->nodes];
    if (aux!=nullptr){
        for (ulong i=0; i<this->nodes; i++){
            aux[i] = false;
        }
        for (ulong k=0; k<this->edges; k++){
            aux[this->e[k].i] = true;
            aux[this->e[k].j] = true;
        }
        for (ulong i=0; i<this->nodes; i++){
            if (!aux[i])
                return true;
        }
        return false;
    }
    else{
        cout << "[miolo Error] Failed to allocate memory." << endl;
        cout << "[This occured in Graph.isolatedNodes.]" << endl;
        return false;
    }
}

/*------------------------------------------------------------------------------
    Algebra
------------------------------------------------------------------------------*/

template <typename T>
graph<T>* graph<T>::smul(T value){
    graph<T>* out = new graph<T>(this->nodes,this->edges);
    for (ulong k=0; k<this->edges; k++){
        out->e[k].w = value*this->e[k].w;
        out->e[k].i = this->e[k].i;
        out->e[k].j = this->e[k].j;
    }
    return out;
}

template <typename T>
mtx<T>* graph<T>::mmul(mtx<T>* A){
    mtx<T>* out = nullptr;
    if (this->nodes==A->rows){
        out = new mtx<T>(A->rows,A->cols,0);
        for (ulong k=0; k<this->edges; k++){
            for (ulong s=0; s<A->cols; s++){
                (*out)(this->e[k].i,s) += this->e[k].w*(*A)(this->e[k].j,s);
                (*out)(this->e[k].j,s) += this->e[k].w*(*A)(this->e[k].i,s);
            }
        }
    }
    else{
        if (this->nodes==A->cols){
            out = new mtx<T>(A->rows,this->nodes,0);
            for (ulong k=0; k<this->edges; k++){
                for (ulong s=0; s<A->rows; s++){
                    (*out)(s,this->e[k].j) += this->e[k].w*(*A)(s,this->e[k].i);
                    (*out)(s,this->e[k].i) += this->e[k].w*(*A)(s,this->e[k].j);
                }
            }
        }
    }
    return out;
}

template <typename T>
graph<T>* graph<T>::add(graph<T>* G){
    graph<T>* out = new graph<T>(this->nodes,this->edges);
    for (ulong k=0; k<this->edges; k++){
        out->e[k].w = G->e[k].w+this->e[k].w;
        out->e[k].i = this->e[k].i;
        out->e[k].j = this->e[k].j;
    }
    return out;
}

template <typename T>
graph<T>* graph<T>::sub(graph<T>* G){
    graph<T>* out = new graph<T>(this->nodes,this->edges);
    for (ulong k=0; k<this->edges; k++){
        out->e[k].w = this->e[k].w-G->e[k].w;
        out->e[k].i = this->e[k].i;
        out->e[k].j = this->e[k].j;
    }
    return out;
}

template <typename T>
graph<T>* graph<T>::hmul(graph<T>* G){
    graph<T>* out = new graph<T>(this->nodes,this->edges);
    for (ulong k=0; k<this->edges; k++){
        out->e[k].w = this->e[k].w*G->e[k].w;
        out->e[k].i = this->e[k].i;
        out->e[k].j = this->e[k].j;
    }
    return out;
}

template <typename T>
mtx<T>* graph<T>::propagate(mtx<T>* M, bool* clamped){
    mtx<T>* out = new mtx<T>(M->rows,M->cols,0);
    for (ulong k=0; k<this->edges; k++){
        for (ulong s=0; s<M->cols; s++){
            if (!clamped[this->e[k].i]){
                (*out)(this->e[k].i,s) += this->e[k].w*(*M)(this->e[k].j,s);
            }
            if (!clamped[this->e[k].j]){
                (*out)(this->e[k].j,s) += this->e[k].w*(*M)(this->e[k].i,s);
            }
        }
    }
    for (ulong i=0; i<M->rows; i++){
        if (clamped[i]){
            for (ulong s=0; s<M->cols; s++){
                (*out)(i,s) = (*M)(i,s);
            }
        }
    }
    return out;
}

template <typename T>
mtx<T>* graph<T>::densify(){
    mtx<T>* out = new mtx<T>(this->nodes,this->nodes,0);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<this->edges; k++){
        (*out)(this->e[k].i,this->e[k].j) = this->e[k].w;
        (*out)(this->e[k].j,this->e[k].i) = this->e[k].w;
    }
    return out;
}

/*------------------------------------------------------------------------------
    Other good stuff
------------------------------------------------------------------------------*/

template <typename T>
vector<T>* slice(T* arr, ulong start, ulong end){
    vector<T>* out = new vector<T>();
    for (ulong p=end-1; p>=start; p--){
        out->push_back(arr[p]);
    } 
    return out;
}

template <typename T>
graph<T>* sparsifyThreshold(mtx<T>* M, T thresh){
    mtx<T>* cop = new mtx<T>(M->rows,M->cols);
    cop->copy(M);
    ulong counter = 0;
    //Symmetrize cop
    for (ulong i=0; i<M->rows; i++)
        for (ulong s=i+1; s<M->rows; s++){
            (*cop)(i,s) = ((*cop)(i,s)+(*cop)(s,i))/2;
            (*cop)(s,i) = (*cop)(i,s);
            if (fabs((*cop)(i,s))>thresh){
                counter++;
            }
        }
    graph<T>* out = new graph<T>(M->rows,counter);
    ulong k = 0;
    if (!out->null()){
        for (ulong i=0; i<M->rows; i++){
            for (ulong j=i+1; j<M->cols; j++){
                if (fabs((*cop)(i,j))>thresh){
                    out->e[k].i = i;
                    out->e[k].j = j;
                    out->e[k].w = cop->data[i*M->cols+j];
                    k++;
                }
            }
        }
    }
    return out;
}

template <typename T>
graph<T>* sparsifyKNN(mtx<T>* M, ulong knn){
    T* cop = new T[M->rows*(M->rows-1)/2+M->rows];
    //Symmetrize M via cop
    ulong counter = 0;
    for (ulong i=0; i<M->rows; i++)
        for (ulong s=i+1; s<M->rows; s++){
            cop[counter] = ((*M)(i,s)+(*M)(s,i))/2;
            counter++;
        }
    sort(&cop[0],&cop[0]+M->rows*(M->rows-1)/2+M->rows);
    graph<T>* out = new graph<T>(M->rows,knn*M->rows);
    counter = 0;
    for (ulong i=0; i<M->rows; i++){
        for (ulong j=0; j<M->cols; j++){
            if (cop[knn]<(*M)(i,j)){
                out->e[counter].i = i;
                out->e[counter].j = j;
                out->e[counter].w = ((*M)(i,j)+(*M)(j,i))/2;
            }
        }
    }
    return out;
}

#endif