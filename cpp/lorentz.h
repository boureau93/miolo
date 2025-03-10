#include "mtx.h"
#include "diagonal.h"
#include "math.h"

class lorentz{
public:

    double beta;

    template <typename T>
    mtx<T>* dot(mtx<T>& M);

    template <typename T>
    bool isIn(mtx<T>& M, double tol);

    template <typename T>
    mtx<T>* distance(mtx<T>& M);

    template <typename T>
    mtx<T>* lorentzDistance(mtx<T>& M);

    template <typename T>
    mtx<T>* fromEuclidean(mtx<T>& M);
    template <typename T>
    mtx<T>* toEuclidean(mtx<T>& M);

    template <typename T>
    mtx<T>* exponential(mtx<T>& at, mtx<T>& M);

    template <typename T>
    mtx<T>* tangentProjection(mtx<T>& at, mtx<T>& M);

    template <typename T>
    mtx<T>* centroidDistance(mtx<T>& data, mtx<T>& center);

    template <typename T>
    mtx<T>* mean(mtx<T>& data);
    template <typename T>
    mtx<T>* variance(mtx<T>& data);
};


template <typename T>
mtx<T>* lorentz::dot(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.rows; j++){
            (*out)(i,j) = -M(i,0)*M(j,0);
            for (ulong k=1; k<M.cols; k++){
                (*out)(i,j) += M(i,k)*M(j,k);
            }
        }
    }
    return out;
}

template <typename T>
bool lorentz::isIn(mtx<T> &M, double tolerance){
    for (ulong i=0; i<M.rows; i++){
        T check = 0;
        check = -M(i,0)*M(i,0);
        for (ulong k=1; k<M.cols; k++){
            check += M(i,k)*M(i,k);
        }
        if (check<-this->beta+tolerance and check>-this->beta-tolerance)
            return false;
    }
    return true;
}

template <typename T>
mtx<T>* lorentz::distance(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.rows; j++){
            (*out)(i,j) = -M(i,0)*M(j,0);
            for (ulong k=1; k<M.cols; k++){
                (*out)(i,j) += M(i,k)*M(j,k);
            }
            (*out)(i,j) = sqrt(this->beta)*acosh(-(*out)(i,j)/this->beta);
        } 
    }
    return out;
}

template <typename T>
mtx<T>* lorentz::lorentzDistance(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.rows; j++){
            (*out)(i,j) = -2*this->beta;
            T diff = M(i,0)-M(j,0);
            (*out)(i,j) += 2*diff*diff;
            for (ulong k=1; k<M.cols; k++){
                diff = M(i,k)-M(j,k);
                (*out)(i,j) -= 2*diff*diff;
            }
        }
    }
    return out;
}

template <typename T>
mtx<T>* lorentz::fromEuclidean(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols+1);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        (*out)(i,0) = this->beta;
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,0) += M(i,j)*M(i,j);
            (*out)(i,j+1) = M(i,j);
        }
        (*out)(i,0) = sqrt((*out)(i,0));
    } 
    return out;
}

template <typename T>
mtx<T>* lorentz::toEuclidean(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols-1);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        (*out)(i,0) = this->beta;
        for (ulong j=0; j<M.cols-1; j++){
            (*out)(i,j) = M(i,j+1);
        }
        (*out)(i,0) = sqrt((*out)(i,0));
    } 
    return out;
}

template <typename T>
mtx<T>* lorentz::exponential(mtx<T>& at, mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T norm = -M(i,0)*M(i,0);
        for (ulong s=1; s<M.cols; s++){
            norm += M(i,s)*M(i,s);
        }
        norm = sqrt(norm);
        for (ulong s=0; s<M.cols; s++){
            (*out)(i,s) = cosh(norm)*at(i,s)+sinh(norm)*M(i,s)/norm;
        }
    }
    return out;
}

template <typename T>
mtx<T>* lorentz::tangentProjection(mtx<T> &at, mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows, M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T dot = -at(i,0)*M(i,0);
        for (ulong s=1; s<M.cols; s++){
            dot += at(i,s)*M(i,s);
        }
        for (ulong s=0; s<M.cols; s++){
            (*out)(i,s) = at(i,s)+dot*M(i,s);
        }
    }
    return out;
}

template <typename T>
mtx<T>* lorentz::centroidDistance(mtx<T> &data, mtx<T> &center){
    mtx<T>* out = new mtx<T>(data.rows,center.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<data.rows; i++){
        for (ulong j=0; j<center.rows; j++){
            (*out)(i,j) = -this->beta;
            (*out)(i,j) += (data(i,0)-center(j,0))*(data(i,0)-center(j,0));
            for (ulong s=1; s<data.cols; s++)
                (*out)(i,j) -= (data(i,s)-center(j,s))*(data(i,s)-center(j,s));
            (*out)(i,j) *= 2;
        }
    }
    return out;
}

template <typename T>
mtx<T>* lorentz::mean(mtx<T> &data){
    mtx<T>* out = new mtx<T>(1,data.cols,0);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<data.rows; i++){
        for (ulong s=0; s<data.cols; s++){
            out->data[s] += data(i,s); 
        }
    }
    T norm = -out->data[0]*out->data[0];
    for (ulong s=1; s<data.cols; s++){
        norm += out->data[s]*out->data[s];
    }
    norm = fabs(norm);
    for (ulong s=0; s<data.cols; s++){
        out->data[s] /= norm;
    }
    return out;
}

template <typename T>
mtx<T>* lorentz::variance(mtx<T>& data){
    mtx<T>* mean = this->mean(data);
    if (mean==nullptr)
        return nullptr;
    mtx<T>* aux = new mtx<T>(data.rows,data.cols);
    if (aux->null())
        return nullptr;
    for (ulong i=0; i<data.rows; i++)
        for (ulong s=0; s<data.cols; s++){
            (*aux)(i,s) = (data(i,s)-(*mean)(0,s))*(data(i,s)-(*mean)(0,s));
        }
    mtx<T>* out = this->mean(*aux);
    if (out==nullptr)
        return nullptr;
    return out;    
}