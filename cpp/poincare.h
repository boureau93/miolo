#include "mtx.h"
#include "diagonal.h"
#include "math.h"

#ifndef POINCARE_H
#define POINCARE_H 
class poincare{

public:

    double c;

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

    template <typename T>
    double hyperbolicity(mtx<T>& M);

    template <typename T>
    mtx<T>* centroidDistance(mtx<T>& data, mtx<T>& center);
};

template <typename T>
mtx<T>* poincare::distance(mtx<T>& M){
    mtx<T>* out = new mtx<T>(M.rows,M.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        (*out)(i,i) = 0;
        for (ulong j=i+1; j<M.rows; j++){
            T aux0 = 0;
            T aux1 = 0;
            T aux2 = 0;
            for (ulong k=0; k<M.cols; k++){
                aux0 += M(i,k)*M(j,k);
                aux1 += M(i,k)*M(i,k);
                aux2 += M(j,k)*M(j,k);
            }
            T z = 1-2*this->c*aux0+this->c*this->c*aux1*aux2;
            T norm = 0;
            for (ulong k=0; k<M.cols; k++){
                T mobius = -(1-2*this->c+aux0+this->c*aux2)*M(i,k);
                mobius += (1-this->c*aux1)*M(j,k);
                mobius /= z;
                norm += mobius*mobius;
            }
            (*out)(i,j) = atanh(sqrt(this->c*norm));
            (*out)(i,j) *= 2/sqrt(this->c); 
            (*out)(j,i) = (*out)(i,j);
        }
    }
    return out;
}

template <typename T>
bool poincare::isIn(mtx<T> &M){
    for (ulong i=0; i<M.rows; i++){
        T aux = 0;
        for (ulong j=0; j<M.cols; j++){
            aux += M(i,j)*M(i,j);
        }
        if (this->c*aux>=1)
            return false;
    }
    return true;
}

template <typename T>
mtx<T>* poincare::madd(mtx<T>& A, mtx<T>& B){
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
        T z = 1+2*this->c*aux[0]+this->c*this->c*aux[1]*aux[2];
        for (ulong j=0; j<A.cols; j++){
            (*out)(i,j) += (1+2*this->c*aux[0]+this->c*aux[2])*A(i,j);
            (*out)(i,j) += (1-this->c*aux[1])*B(i,j);
            (*out)(i,j) /= z;
        }
    }
    return out;
}

template <typename T>
mtx<T>* poincare::exponential(mtx<T> &at, mtx<T> &M){
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
        T lambda = 2/(1-this->c*norm_at);
        for (ulong j=0; j<M.cols; j++){
            (*aux)(i,j) = tanh(sqrt(this->c)*lambda*norm_M/2);
            (*aux)(i,j) *= M(i,j)/(this->c*norm_M);
        }
    }
    mtx<T>* out = this->madd(at,*aux);
    delete aux;
    return out;
}

template <typename T>
mtx<T>* poincare::logarithm(mtx<T> &start, mtx<T> &end){
    mtx<T>* aux = start.smul(-1);
    if (aux==nullptr)
        return nullptr;
    mtx<T>* dir = this->madd(*aux,end);
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
        lambda = 2/(1-this->c*lambda);
        D[i] = 2*atanh(sqrt(this->c)*norm)/(sqrt(this->c)*norm*lambda);
    }
    mtx<T>* out = D.lmul(*dir);
    delete dir;
    return out;
}

template <typename T>
mtx<T>* poincare::toKlein(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T norm = 0;
        for (ulong j=0; j<M.cols; j++){
            norm += M(i,j)*M(i,j);
        }
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = 2*M(i,j)/(1+this->c*norm);
        }
    }
    return out;
}

template <typename T>
mtx<T>* poincare::toPoincare(mtx<T> &M){
    mtx<T>* out = new mtx<T>(M.rows,M.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<M.rows; i++){
        T norm = 0;
        for (ulong j=0; j<M.cols; j++){
            norm += M(i,j)*M(i,j);
        }
        for (ulong j=0; j<M.cols; j++){
            (*out)(i,j) = M(i,j)/(1+sqrt(1-this->c*norm));
        }
    }
    return out;
}

template <typename T>
mtx<T>* poincare::mean(mtx<T> &M){
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
        gamma = 1/sqrt(1-this->c*gamma);
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

template <typename T>
double poincare::hyperbolicity(mtx<T> &M){
    //Gromov product
    mtx<T> G(M.rows,M.rows);
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.rows; j++){
            T aux[3] = {0,0,0};
            for (ulong k=0; k<M.cols; k++){
                aux[0] += (M(i,k)-M(j,k))*(M(i,k)-M(j,k));
                aux[1] += M(i,k)*M(i,k);
                aux[2] += M(j,k)*M(j,k);
            }
            G(i,j) = (sqrt(aux[1])+sqrt(aux[2])-sqrt(aux[0]))/2;
        }
    }
    //Calculate product, subtract and find maximum
    double out = 0;
    for (ulong i=0; i<M.rows; i++){
        for (ulong j=0; j<M.rows; j++){
            T Aij = -10000000; //Auxiliary variable for matrix entry
            for (ulong k=0; k<M.rows; k++){
                if (G(i,k)>G(k,j)){
                    if (Aij<G(k,j))
                        Aij = G(k,j);
                }
                else{
                    if (Aij<G(i,k))
                        Aij = G(i,k);
                }
            }
            Aij -= G(i,j);
            if ((i==0 && j==0) || out<Aij)
                out = Aij;
        }
    }
    return out;
}

template <typename T>
mtx<T>* poincare::centroidDistance(mtx<T>& data, mtx<T>& center){
    mtx<T>* out = new mtx<T>(data.rows,center.rows);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<out->rows; i++){
        for (ulong j=0; j<out->cols; j++){
            (*out)(i,j) = 0;
            T aux[3] = {0,0,0};
            for (ulong s=0; s<data.cols; s++){
                aux[0] -= (data(i,s)-center(j,s))*(data(i,s)-center(j,s));
                aux[1] += data(i,s)*data(i,s);
                aux[2] += center(j,s)*center(j,s);
            }
            for (ulong s=0; s<data.cols; s++){
                T row = -(1+2*this->c*aux[0]+aux[2])*data(i,s);
                row += (1-this->c*aux[1])*center(j,s);
                row /= 1+2*this->c*aux[0]+this->c*this->c*aux[1]*aux[2];
                (*out)(i,j) += row*row;
            }
            (*out)(i,j) = 2*atanh(sqrt(this->c*(*out)(i,j)))/sqrt(this->c);
        }
    }
    return out;
}

#endif