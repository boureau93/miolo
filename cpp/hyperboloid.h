#include "mtx.h"

class hyperboloid{

    template <typename T>
    mtx<T>* toEuclidean(mtx<T> M);
    template <typename T>
    mtx<T>* fromEuclidean(mtx<T> M);
};