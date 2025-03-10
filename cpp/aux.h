#include <numeric>
#include <algorithm>
#include <utility>
#include <functional>
#include <vector>
#include <array>

using namespace std; 

#ifndef AUX_H
#define AUX_H

typedef unsigned long ulong;

/*------------------------------------------------------------------------------
    Funções para argsort
------------------------------------------------------------------------------*/

template <typename T>
class comparer{
public:
    T* arr;

    bool operator()(size_t i, size_t j){
        return arr[i]<arr[j];
    }
};

template <typename T>
ulong* argsort(T* arr, ulong len){
    ulong* out = new ulong[len];
    if (out==nullptr)
        return nullptr;
    iota(out,out+len,0);
    comparer<T> cmp;
    cmp.arr = arr;
    sort(out,out+len,cmp);
    return out;
}

/*------------------------------------------------------------------------------
    Funções para DSU (Utilizado no algoritmo de Kruskal)
    Source: geeksforgeeks
------------------------------------------------------------------------------*/

class DSU{
    long int* parent;
    long int* rank;
    ulong n;

public:

    DSU(ulong n);
    ~DSU();

    bool null();

    ulong find(ulong k);
    void unite(ulong i, ulong j);
};

DSU::DSU(ulong n){
    this->parent = new long int[n];
    this->rank = new long int[n];
    this->n = n;
    if (this->rank!=nullptr && this->parent!=nullptr){
        for (ulong i=0; i<n; i++){
            this->parent[i] = -1;
            this->rank[i] = -1;
        }
    }
}

DSU::~DSU(){
    delete[] this->parent;
    delete[] this->rank;
}

bool DSU::null(){
    return (this->rank==nullptr || this->parent==nullptr);
}

ulong DSU::find(ulong k){
    ulong p = k;
    while(parent[p]!=-1){
        p = parent[p];
    }
    return p;
}

void DSU::unite(ulong i, ulong j){
    ulong p_i = this->find(i);
    ulong p_j = this->find(j);
    if (p_i!=p_j){
        if (this->rank[p_i]<this->rank[p_j]){
            this->parent[p_i] = this->parent[p_j];
        }
        else if (this->rank[p_i]>this->rank[p_j]){
            this->parent[p_j] = this->parent[p_i];
        }
        else{
            this->parent[p_j] = this->parent[p_i];
            this->rank[p_i]++;
        }
    }
}

#endif