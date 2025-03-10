#include "mtx.h"
#include "graph.h"
#include "digraph.h"
#include "math.h"

class ufunction{
public:
    double (*f)(double);

    template <typename T>
    T operator()(T x);
    template <typename T>
    mtx<T>* mtxApply(mtx<T>& M);
    template <typename T>
    graph<T>* graphApply(graph<T>& G);
    template <typename T>
    digraph<T>* digraphApply(digraph<T>& G);
    
};

template <typename T>
T ufunction::operator()(T x){
    return this->f(x);
}

template <typename T>
mtx<T>* ufunction::mtxApply(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++)
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = this->f(M(i,j));
        }
    return out;
}

template <typename T>
graph<T>* ufunction::graphApply(graph<T>& G){
    graph<T>* out = new graph<T>(G.nodes,G.edges);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<G.edges; k++){
        out->e[k].w = this->f(G[k].w);
        out->e[k].i = G[k].i;
        out->e[k].j = G[k].j;
    }
    return out;
}

template <typename T>
digraph<T>* ufunction::digraphApply(digraph<T>& G){
    digraph<T>* out = new digraph<T>(G.nodes);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<G.nodes; i++){
        for (ulong k=0; k<G.hood[i].size(); k++){
            out->connect(i,G.hood[i][k].idx,this->f(G.hood[i][k].w));
        }
    }
    return out;
}

double reciprocal(double x){
    if (x!=0){
        return 1/x;
    }
    else {
        cout << "[miolo Warning] Preventing division by zero." << endl;
        cout << "<< This ocurred in reciprocal. >> " << endl;
        return 0;
    }
}