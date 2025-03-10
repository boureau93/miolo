#include "mtx.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include "aux.h"
#include <iostream>

using namespace std;

#ifndef GRAPH_H
#define GRAPH_H

template <typename T>
class edge{
public:
    ulong i, j; //index of nodes in edge
    T w; //weight of edge

};

template <typename T>
bool cmpEdge(edge<T>& u, edge<T>& v){
    if (u.w<v.w)
        return true;
    return false;
}

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
    void clean(){delete[] this->e;}

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

    T max();
    T min();

    bool isolatedNodes();

    //Algebra
    graph<T>* smul(T value);
    mtx<T>* mmul(mtx<T>* M);
    graph<T>* add(graph<T>* G);
    graph<T>* sub(graph<T>* G);
    graph<T>* hmul(graph<T>* G);

    mtx<T>* propagate(mtx<T>& M, bool* clamped);

    mtx<T>* degree();
    void normalize();
    graph<T>* laplacian();
    void print();

    T gaussianScale();

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
        delete[] this->e;
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
T graph<T>::max(){
    T out = this->e[0].w;
    for (ulong k=1; k<this->edges; k++){
        if (this->e[k].w>out)
            out = this->e[k].w;
    }
    return out;
}

template <typename T>
T graph<T>::min(){
    T out = this->e[0].w;
    for (ulong k=1; k<this->edges; k++){
        if (this->e[k].w<out)
            out = this->e[k].w;
    }
    return out;
}

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

template <typename T>
void graph<T>::print(){
    for (ulong k=0; k<this->edges; k++){
        cout << "(", 
        cout << this->e[k].i << ",";
        cout << this->e[k].j << ",";
        cout << this->e[k].w << ")" << endl; 
    }
}

/*------------------------------------------------------------------------------
    Similarity
------------------------------------------------------------------------------*/

template <typename T>
T graph<T>::gaussianScale(){
    T* dg = new T[this->nodes];
    if (dg==nullptr)
        return -1;
    for (ulong i=0; i<this->nodes; i++){
        dg[i] = 0;
    }
    //Find most distant neighbor
    for (ulong k=0; k<this->edges; k++){
        if (dg[this->e[k].i]<this->e[k].w)
            dg[this->e[k].i] = this->e[k].w;
        if (dg[this->e[k].j]<this->e[k].w)
            dg[this->e[k].j] = this->e[k].w;
    }
    //Calculate sigma
    T sigma = 0;
    for (ulong i=0; i<this->nodes; i++){
        sigma += dg[i];
    }
    delete[] dg;
    sigma /= 3*this->nodes;
    return sigma;
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
mtx<T>* graph<T>::propagate(mtx<T>& M, bool* clamped){
    mtx<T>* out = new mtx<T>(M.rows,M.cols,0);
    for (ulong k=0; k<this->edges; k++){
        for (ulong s=0; s<M.cols; s++){
            if (!clamped[this->e[k].i]){
                (*out)(this->e[k].i,s) += this->e[k].w*M(this->e[k].j,s);
            }
            if (!clamped[this->e[k].j]){
                (*out)(this->e[k].j,s) += this->e[k].w*M(this->e[k].i,s);
            }
        }
    }
    for (ulong i=0; i<M.rows; i++){
        if (clamped[i]){
            for (ulong s=0; s<M.cols; s++){
                (*out)(i,s) = M(i,s);
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
graph<T>* sparsifyGraphThreshold(mtx<T>& M, double thresh){
    mtx<T>* cop = new mtx<T>(M.rows,M.rows);
    ulong counter = 0;
    //Symmetrize cop
    for (ulong i=0; i<M.rows; i++)
        for (ulong s=i+1; s<M.rows; s++){
            (*cop)(i,s) = (M(i,s)+M(s,i))/2;
            (*cop)(s,i) = (*cop)(i,s);
            if (fabs((*cop)(i,s))<thresh){
                counter++;
            }
        }
    graph<T>* out = new graph<T>(M.rows,counter);
    ulong k = 0;
    if (!out->null()){
        for (ulong i=0; i<M.rows; i++){
            for (ulong j=i+1; j<M.cols; j++){
                if (fabs((*cop)(i,j))<thresh){
                    out->e[k].i = i;
                    out->e[k].j = j;
                    out->e[k].w = cop->data[i*M.cols+j];
                    k++;
                }
            }
        }
    }
    delete cop;
    return out;
}

template <typename T>
graph<T>* mst(mtx<T>& M){
    graph<T>* out = new graph<T>(M.rows,M.rows-1);
    if (out->null())
        return nullptr;
    //Auxiliary array for all edges
    ulong len = (M.rows-1)*M.rows/2;
    edge<T>* aux = new edge<T>[len];
    if (aux==nullptr)
        return nullptr;
    ulong k = 0;
    for (ulong i=0; i<M.rows; i++)
        for (ulong j=i+1; j<M.rows; j++){
            aux[k].i = i;
            aux[k].j = j;
            aux[k].w = M(i,j); 
            k++;
        }
    sort(aux,aux+len,cmpEdge<T>);
    //Kruskal
    DSU D(M.rows);
    if (D.null())
        return nullptr;
    ulong count = 0;
    for (ulong k=0; k<len && count<M.rows-1; k++){
        if (D.find(aux[k].i)!=D.find(aux[k].j)){
            D.unite(aux[k].i,aux[k].j);
            out->e[count].i = aux[k].i;
            out->e[count].j = aux[k].j;
            out->e[count].w = aux[k].w;
            count++; 
        }
    }
    delete[] aux;
    return out;
}

template <typename T>
graph<T>* mst(graph<T>& G){
    graph<T>* out = new graph<T>(G.nodes,G.nodes-1);
    if (out->null())
        return nullptr;
    //Auxiliary array for all edges
    edge<T>* aux = new edge<T>[G.edges];
    if (aux==nullptr)
        return nullptr;
    for (ulong k=0; k<G.edges; k++){
        aux[k].i = G[k].i;
        aux[k].j = G[k].j;
        aux[k].w = G[k].w;
    }
    sort(&aux[0],&aux[0]+G.edges,cmpEdge<T>);
    //Kruskal
    DSU D(G.nodes);
    if (D.null())
        return nullptr;
    ulong count = 0;
    for (ulong k=0; k<G.edges && count<G.nodes-1; k++){
        if (D.find(aux[k].i)!=D.find(aux[k].j)){
            D.unite(aux[k].i,aux[k].j);
            out->e[count].i = aux[k].i;
            out->e[count].j = aux[k].j;
            out->e[count].w = aux[k].w;
            count++;
        }
    }
    delete[] aux;
    return out;
}
#endif