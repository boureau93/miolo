#include "mtx.h"
#include "graph.h"
#include "aux.h"
#include <vector>

#ifndef DIGRAPH_H
#define DIGRAPH_H

template <typename T>
class neighbor{
public:
    ulong idx; //index of neighbor in digraph
    T w; //weight of interaction
};
template <typename T>
class neighborhood{
public:
    vector< neighbor<T> >* nb;

    neighborhood(){this->nb = new vector< neighbor<T> >;}
    ~neighborhood(){delete this->nb;}

    void connect(ulong idx, T w){
        neighbor<T> ngb; ngb.idx = idx; ngb.w = w;
        this->nb->push_back(ngb);
    }

    ulong size(){return this->nb->size();}

    neighbor<T>& operator[](ulong k){
        return (*this->nb)[k];
    }

    T weight(ulong idx){
        for (ulong k=0; k<this->size(); k++){
            if ((*this->nb)[k].idx==idx)
                return (*this->nb)[k].w;
        }
        return 0;
    }
};

template <typename T>
class digraph{
public:
    ulong nodes;
    neighborhood<T>* hood; 

    digraph(){this->hood=nullptr;}
    digraph(ulong nodes);
    ~digraph();
    void clean(){delete this->hood;}
    
    bool null();
    bool isNext(ulong i, ulong j);
    bool isPrevious(ulong i, ulong j);
    neighborhood<T>& operator[](ulong k);

    void connect(ulong i, ulong j, T w);
    void connect(ulong i, ulong j);

    void normalize();

    digraph<T>* copyStructure();
    digraph<T>* copy();
    ulong* shape();
    digraph<T>* transpose();
    digraph<T>* symmetrize();
    void connectRegular(mtx<int>&, mtx<T>&);

    digraph<T>* add(digraph<T>& D);
    digraph<T>* sub(digraph<T>& D);
    digraph<T>* hmul(digraph<T>& D);
    digraph<T>* smul(T value);
    mtx<T>* mmul(mtx<T>& D);

    T gaussianScale();
};

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Constructor, destructor and copy
--------------------------------------------------------------------------------
------------------------------------------------------------------------------*/

template <typename T>
digraph<T>::digraph(ulong nodes){
    this->nodes = nodes;
    this->hood = new neighborhood<T>[nodes];
}
template <typename T>
digraph<T>::~digraph(){
    delete[] this->hood;
}

template <typename T>
bool digraph<T>::null(){
    return this->hood==nullptr;
}

template <typename T>
neighborhood<T>& digraph<T>::operator[](ulong k){
    return this->hood[k];
}

template <typename T>
bool digraph<T>::isNext(ulong i, ulong j){
    for (ulong n=0; n<this->hood[i].size(); n++)
        if (this->hood[i][n].idx==j)
            return true;
    return false;
}

template <typename T>
bool digraph<T>::isPrevious(ulong i, ulong j){
    for (ulong n=0; n<this->hood[j].size(); n++)
        if (this->hood[j][n].idx==i)
            return true;
    return false;
}

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Topological utilities
--------------------------------------------------------------------------------
------------------------------------------------------------------------------*/

template <typename T>
void digraph<T>::connect(ulong i, ulong j, T w){
    this->hood[i].connect(j,w);
}

template <typename T>
void digraph<T>::connect(ulong i, ulong j){
    this->connect(i,j,1);
}

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Useful
--------------------------------------------------------------------------------
------------------------------------------------------------------------------*/

template <typename T>
void digraph<T>::normalize(){
    for (ulong i=0; i<this->nodes; i++){
        T z = 0;
        for (ulong k=0; k<this->hood[i].size(); k++){
            z += this->hood[i][k].w;
        }
        if (z!=0){
            for (ulong k=0; k<this->hood[i].size(); k++){
                this->hood[i][k].w /= z;
            }
        }
        else{
            cout << "[miolo Warning] Preventing division by zero." << endl;
            cout << "<< This occured in Digraph.normalize(). >>" << endl;
        }
    }
}

template <typename T>
digraph<T>* digraph<T>::copyStructure(){
    digraph<T>* out = new digraph<T>(this->nodes);
    if (out->null())
        return nullptr;

    for (ulong i=0; i<this->nodes; i++){
        for (ulong n=0; n<this->hood[i].size(); n++){
            out->connect(i,this->hood[i][n].idx,1);
        }
    }
    return out;
}

template <typename T>
digraph<T>* digraph<T>::copy(){
    digraph<T>* out = new digraph<T>(this->nodes);
    if (out->null())
        return nullptr;

    for (ulong i=0; i<this->nodes; i++){
        for (ulong n=0; n<this->hood[i].size(); n++){
            out->connect(i,this->hood[i][n].idx,this->hood[i][n].w);
        }
    }
    return out;
}

template <typename T>
ulong* digraph<T>::shape(){
    ulong* out = new ulong[this->nodes];
    if (out==nullptr)
        return nullptr;
    for (ulong i=0; i<this->nodes; i++){
        out[i] = this->hood[i].size();
    }
    return out;
}

template <typename T>
digraph<T>* digraph<T>::transpose(){
    digraph<T>* out = new digraph<T>(this->nodes);
    if (out->null())
        return nullptr;

    for (ulong i=0; i<this->nodes; i++){
        for (ulong n=0; n<this->hood[i].size(); n++){
            out->connect(this->hood[i][n].idx,i,this->hood[i][n].w);
        }
    }
    return out;
}

template <typename T>
digraph<T>* digraph<T>::symmetrize(){
    digraph<T>* out = new digraph<T>(this->nodes);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<this->nodes; i++){
        for (ulong k=0; k<this->hood[i].size(); k++){
            ulong p = this->hood[i][k].idx;
            if (p>i){
                T aux = (this->hood[i][k].w+this->hood[p].weight(i))/2;
                out->connect(i,p,aux);
                out->connect(p,i,aux);
            }
        }
    }
    return out;
}

template <typename T>
void digraph<T>::connectRegular(mtx<int>& neighbors, mtx<T>& weights){
    for (ulong i=0; i<this->nodes; i++){
        for (ulong p=0; p<neighbors.cols; p++){
            if ((ulong)neighbors(i,p)<this->nodes && (ulong)neighbors(i,p)>=0)
                this->connect(i,neighbors(i,p),weights(i,neighbors(i,p)));
        }
    }
}

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Algebra
--------------------------------------------------------------------------------
------------------------------------------------------------------------------*/

template <typename T>
digraph<T>* digraph<T>::add(digraph<T>& D){
    digraph<T>* out = this->copy();
    if (out==nullptr)
        return nullptr;
    for (ulong i=0; i<this->nodes; i++){
        for (ulong s=0; s<this->hood[i].size(); s++){
            out->hood[i][s].w += D.hood[i][s].w;
        }
    }
    return out;
}

template <typename T>
digraph<T>* digraph<T>::sub(digraph<T>& D){
    digraph<T>* out = this->copy();
    if (out==nullptr)
        return nullptr;
    for (ulong i=0; i<this->nodes; i++){
        for (ulong s=0; s<this->hood[i].size(); s++){
            out->hood[i][s].w -= D.hood[i][s].w;
        }
    }
    return out;
}

template <typename T>
digraph<T>* digraph<T>::hmul(digraph<T>& D){
    digraph<T>* out = this->copy();
    if (out==nullptr)
        return nullptr;
    for (ulong i=0; i<this->nodes; i++){
        for (ulong s=0; s<this->hood[i].size(); s++){
            out->hood[i][s].w *= D.hood[i][s].w;
        }
    }
    return out;
}

template <typename T>
digraph<T>* digraph<T>::smul(T value){
    digraph<T>* out = this->copy();
    if (out==nullptr)
        return nullptr;
    for (ulong i=0; i<this->nodes; i++){
        for (ulong s=0; s<this->hood[i].size(); s++){
            out->hood[i][s].w *= value;
        }
    }
    return out;
}

template <typename T>
mtx<T>* digraph<T>::mmul(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    if (M.rows==this->nodes){
        for (ulong i=0; i<this->nodes; i++){
            for (ulong j=0; j<this->hood[i].size(); j++){
                for (ulong s=0; s<M.cols; s++)
                    (*out)(i,s) += this->hood[i][j].w*M(this->hood[i][j].idx,s);
            }
        }
    }
    else if (M.cols==this->nodes){
        for (ulong i=0; i<this->nodes; i++){
            for (ulong j=0; j<this->hood[i].size(); j++){
                for (ulong s=0; s<M.rows; s++)
                    (*out)(s,i) += this->hood[i][j].w*M(s,this->hood[i][j].idx);
            }
        }
    }
    return out;
}

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Interaction with graphs
--------------------------------------------------------------------------------
------------------------------------------------------------------------------*/

template <typename T>
digraph<T>* toDigraph(graph<T>& G){
    digraph<T>* out = new digraph<T>(G.nodes);
    if (out->null())
        return nullptr;
    for (ulong k=0; k<G.edges; k++){
        out->connect(G[k].i,G[k].j,G[k].w);
        out->connect(G[k].j,G[k].i,G[k].w);
    }
    return out;
}

template <typename T>
graph<T>* toGraph(digraph<T>& G){
    ulong size = 0;
    for (ulong i=0; i<G.nodes; i++){
        for (ulong k=0; k<G.hood[i].size(); k++){
            ulong p = G.hood[i][k].idx;
            if (p>i){
                size++;
            }
        }
    }
    graph<T>* out = new graph<T>(G.nodes,size);
    ulong iter = 0;
    for (ulong i=0; i<G.nodes; i++){
        for (ulong k=0; k<G.hood[i].size(); k++){
            ulong p = G.hood[i][k].idx;
            if (p>i){
                T aux = (G.hood[i][k].w+G.hood[p].weight(i))/2;
                out->e[iter].i = i;
                out->e[iter].j = p;
                out->e[iter].w = aux;
                iter++;
            }
        }
    }
    return out;
}


/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Interaction with mtx
--------------------------------------------------------------------------------
------------------------------------------------------------------------------*/

template <typename T>
digraph<T>* sparsifyDigraphThreshold(mtx<T> &M, double thresh){
    digraph<T>* out = new digraph<T>(M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.cols; j++){
            if (fabs(M(i,j))<thresh){
                out->connect(i,j,M(i,j));
            }
        }
    }
    return out;
}

template <typename T>
digraph<T>* sparsifyDigraphKNN(mtx<T> &M, ulong k){
    digraph<T>* out = new digraph<T>(M.rows);
    if (out->null())
        return nullptr;
    T* cp = new T[M.cols];
        if (cp==nullptr)
            return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.cols; j++)
            cp[j] = M(i,j);
        //argsort
        ulong* args = argsort(cp,M.cols);
        if (args==nullptr)
            return nullptr;
        for (ulong p=0; p<k; p++){
            out->connect(i,args[p+1],M(i,args[p+1]));
        }
        delete[] args;
    }
    delete[] cp;
    return out;
}

template <typename T>
T digraph<T>::gaussianScale(){
    T out = 0;
    for (ulong i=0; i<this->nodes; i++){
        T aux = 0;
        for (ulong k=0; k<this->hood[i].size(); k++){
            if (this->hood[i][k].w>aux)
                aux = this->hood[i][k].w;
        }
        out += aux;
    }
    return out/(3*this->nodes);
}

#endif