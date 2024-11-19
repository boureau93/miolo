#include "mtx.h"
#include <cstddef>
#include <cmath>

#ifndef STIEFEL_H
#define STIEFEL_H

class stiefel{
public:

    template <typename T>
    mtx<T>* qfactor(mtx<T> M);

};

template <typename T>
mtx<T>* stiefel::qfactor(mtx<T> M){
    mtx<T>* Q = new mtx<T>(M.rows,M.cols);
    mtx<T>* R = new mtx<T>(M.cols,M.cols,0);
    mtx<T>* out = new mtx<T>(M.rows,M.cols,0);
    if (Q->null() || R->null() || out->null())
        return nullptr;
    //Gram-Schmidt process
    for (ulong s=0; s<M.cols; s++){
        for (ulong i=0; i<M.rows; i++){
            (*R)(s,s) += M(i,s)*M(i,s);
        }
        (*R)(s,s) = sqrt((*R)(s,s));
        if ((*R)(s,s)>0)
            for (ulong i=0; i<M.rows; i++){
                (*Q)(i,s) = M(i,s)/(*R)(s,s);
            }
        else{
            cout << "[miolo Warning] Preventing division by zero." << endl;
            cout << "<< This ocurred in Stiefel.qfactor and it was ";
            cout << "caused by a zero at the diagonal of the R factor. >>" << endl;
        }
        for (ulong p=s+1; p<M.cols; p++){
            for (ulong i=0; i<M.rows; i++){
                (*R)(s,p) += (*Q)(i,s)*M(i,p);
            }
            for (ulong i=0; i<M.rows; i++){
                (*out)(i,p) = M(i,p)-(*Q)(i,s)*(*R)(s,p);
            }
        }
    }
    delete Q; delete R;
    //Column-normalization
    for (ulong s=0; s<M.cols; s++){
        T z = 0;
        for (ulong i=0; i<M.rows; i++){
            z += (*out)(i,s)*(*out)(i,s);
        }
        z = sqrt(z);
        if (z!=0){
            for (ulong i=0; i<M.rows; i++){
                (*out)(i,s) = (*out)(i,s)/z;
            }
        }
        else {
            cout << "[miolo Warning] Preventing division by zero." << endl;
            cout << "<< This ocurred in Stiefel.qfactor at " ;
            cout << "column " << s << " >>" << endl;
        }
    }
    return out;
}

#endif