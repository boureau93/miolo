#include "mtx.h"
#include "diagonal.h"
#include "math.h"

class hyperbolic{

public:

    template <typename T>
    mtx<T>* distance(mtx<T>& M);

    template <typename T>
    bool isIn(mtx<T>& M);

    template <typename T>
    mtx<T>* madd(mtx<T>& A, mtx<T>& B);

    template <typename T>
    mtx<T>* exponential(mtx<T>& at, mtx<T>& M);
    template <typename T>
    mtx<T>* logarithm(mtx<T>& start, mtx<T>& end);

    template <typename T>
    mtx<T>* toKlein(mtx<T>& M);
    template <typename T>
    mtx<T>* toPoincare(mtx<T>& M);

    template <typename T>
    mtx<T>* mean(mtx<T>& M);
};

template <typename T>
mtx<T>* hyperbolic::distance(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=i+1; j<M.rows; j++){
            T aux[3] = {0,0,0};
            for (ulong k=0; k<M.cols; k++){
                aux[0] += (M(i,k)-M(j,k))*M(i,k)-M(j,k);
                aux[1] += M(i,k)*M(i,k);
                aux[2] += M(j,k)*M(j,k);
            }
            (*out)(i,j) = acosh(1+2*aux[0]/((1-aux[1])*(1-aux[2])));
            (*out)(j,i) = (*out)(i,j);
        }
    }
    return out;
}

template <typename T>
bool hyperbolic::isIn(mtx<T> &M){
    for (ulong i=0; i<M.rows; i++){
        T aux = 0;
        for (ulong j=0; j<M.cols; j++){
            aux += M(i,j)*M(i,j);
        }
        if (aux>=1)
            return false;
    }
    return true;
}

template <typename T>
mtx<T>* hyperbolic::madd(mtx<T>& A, mtx<T>& B){
    mtx<T>* out = new mtx<T>(A.rows,A.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<A.rows; i++){
        T aux[3] = {0,0,0};
        for (ulong j=0; j<A.cols; j++){
            aux[0] += A(i,j)*B(i,j);
            aux[1] += A(i,j)*A(i,j);
            aux[2] += B(i,j)*B(i,j);
        }
        T z = 1+2*aux[0]+aux[1]*aux[2];
        for (ulong j=0; j<A.cols; j++){
            (*out)(i,j) += (1+2*aux[0]+aux[2])*A(i,j);
            (*out)(i,j) += (1-aux[1])*B(i,j);
            (*out)(i,j) /= z;
        }
    }
    return out;
}

template <typename T>
mtx<T>* hyperbolic::exponential(mtx<T> &at, mtx<T> &M){
    mtx<T>* aux = new mtx<T>(M.rows,M.cols);
    if (aux->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T norm_at = 0;
        T norm_M = 0;
        //Calculate norms
        for (ulong j=0; j<M.cols; j++){
            norm_at += at(i,j);
            norm_M += M(i,j);
        }
        //Calculate right-hand
        T lambda = 2/(1-norm_at);
        for (ulong j=0; j<M.cols; j++){
            (*aux)(i,j) = tanh(lambda*norm_M/2)*M(i,j)/norm_M;
        }
    }
    mtx<T>* out = this->madd(at,*aux);
    delete aux;
    return out;
}

template <typename T>
mtx<T>* hyperbolic::logarithm(mtx<T> &start, mtx<T> &end){
    mtx<T>* aux = start.smul(-1);
    if (aux==nullptr)
        return nullptr;
    mtx<T>* dir = this->madd(start,end);
    if (dir==nullptr)
        return nullptr;
    delete aux;
    diagonal<T> D = diagonal<T>(start.rows);
    for (ulong i=0; i<D.dim; i++){
        T lambda = 0;
        T norm = 0;
        for (ulong j=0; j<start.cols; j++){
            lambda += start(i,j)*start(i,j);
            norm += (*dir)(i,j)*(*dir)(i,j);
        }
        lambda = 2/(1-lambda);
        D[i] = 2*atanh(norm)/(norm*lambda);
    }
    mtx<T>* out = D.lmul(*dir);
    delete dir;
    return out;
}

template <typename T>
mtx<T>* hyperbolic::toKlein(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T norm = 0;
        for (ulong j=0; j<M.cols; j++){
            norm += M(i,j)*M(i,j);
        }
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = 2*M(i,j)/(1+norm);
        }
    }
    return out;
}

template <typename T>
mtx<T>* hyperbolic::toPoincare(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T norm = 0;
        for (ulong j=0; j<M.cols; j++){
            norm += M(i,j)*M(i,j);
        }
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = M(i,j)/(1+sqrt(1-norm));
        }
    }
    return out;
}

template <typename T>
mtx<T>* hyperbolic::mean(mtx<T> &M){
    mtx<T>* out = new mtx<T>(1,M.cols,0);
    mtx<T>* K = this->toKlein(M);
    if (out->null() || K==nullptr)
        return nullptr;
    T z = 0;
    for (ulong i=0; i<M.rows; i++){
        T gamma = 0;
        for (ulong j=0; j<M.cols; j++){
            gamma += (*K)(i,j)*(*K)(i,j);
        }
        gamma = 1/sqrt(1-gamma);
        z += gamma;
        for (ulong j=0; j<M.cols; j++){
            (*out)(0,j) += (*K)(i,j)*gamma;
        }
    }
    for (ulong j=0; j<M.cols; j++){
        (*out)(0,j) /= z;
    }
    mtx<T>* outP = this->toPoincare(*out);
    delete out;
    delete K;
    return outP;
}