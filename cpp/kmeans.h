#include "mtx.h"
#include <cmath>

class kmeans{
public:

    template <typename T>
    mtx<T>* centroidDistance(mtx<T>& data, mtx<T>& center);
};

template <typename T>
mtx<T>* kmeans::centroidDistance(mtx<T>& data, mtx<T>& center){
    mtx<T>* out = new mtx<T>(data.rows,center.rows);
    data.print();
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