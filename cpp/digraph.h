#include "mtx.h"
#include "graph.h"
#include <vector>

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
};

template <typename T>
class digraph{
public:
    ulong nodes;
    neighborhood<T>* hood; 

    digraph(){this->hood=nullptr;}
    digraph(ulong nodes);
    ~digraph();
    bool null();
    bool isNext(ulong i, ulong j);
    bool isPrevious(ulong i, ulong j);
    neighborhood<T>& operator[](ulong k);

    void connect(ulong i, ulong j, T w);
    void connect(ulong i, ulong j);

    digraph<T>* copyStructure();
    digraph<T>* copy();
    ulong* shape();
    digraph<T>* transpose();

    digraph<T>* add(digraph<T> D);
    digraph<T>* sub(digraph<T> D);
    digraph<T>* hmul(digraph<T> D);
    digraph<T>* smul(T value);
    mtx<T>* mmul(mtx<T> D);
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
    if (this->hood!=nullptr)
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
digraph<T>* digraph<T>::copyStructure(){
    digraph<T>* out = new digraph<T>(this->nodes);
    if (out->null())
        return nullptr;

    for (ulong i=0; i<this->nodes; i++){
        for (ulong n=0; n<this->hood[i].size(); n++){
            out->connect(i,this->hood[i][n].idx,0);
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

/*------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    Algebra
--------------------------------------------------------------------------------
------------------------------------------------------------------------------*/

template <typename T>
digraph<T>* digraph<T>::add(digraph<T> D){
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
digraph<T>* digraph<T>::sub(digraph<T> D){
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
digraph<T>* digraph<T>::hmul(digraph<T> D){
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
mtx<T>* digraph<T>::mmul(mtx<T> M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<this->nodes; i++){
        for (ulong j=0; j<this->hood[i].size(); j++){
            for (ulong s=0; s<M.cols; s++)
                (*out)(i,s) += this->hood[i][j].w*M(this->hood[i][j].idx,s);
        }
    }
    return out;
}