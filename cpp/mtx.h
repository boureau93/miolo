#include <new>
#include <iostream>
#include <sys/types.h>

using namespace std;

#ifndef MTX_H
#define MTX_H

typedef unsigned long int ulong;

template <typename T>
class mtx {

public:
    ulong rows, cols;
    T* data;

    mtx();
    mtx(ulong rows, ulong cols);
    mtx(ulong rows, ulong cols, T init);
    mtx(mtx<T>& cp);
    ~mtx();

    bool null(){return this->data==nullptr;}
    void wrap(mtx<T>* target){
        this->rows = target->rows;
        this->cols = target->cols;
        this->data = target->data;
    }
    
    T& operator()(ulong i, ulong j){
        return this->data[i*this->cols+j];
    }

    T& operator[](ulong k){
        return this->data[k];
    }

    void print();

    void copy(mtx<T>& cp);
    void copy(mtx<T>& cp, ulong* only, ulong lenOnly);

    mtx<T>* transpose();
    void flatten(bool rows);
    void reshape(unsigned long rows, unsigned long cols);
    mtx<T>* cut(bool* targets);

    //Algebra
    mtx<T>* add(mtx<T>* A);
    mtx<T>* sub(mtx<T>* A);
    mtx<T>* mmul(mtx<T>* A);
    mtx<T>* smul(T value);
    mtx<T>* hmul(mtx<T>* value);

    T max(); 
    T min();

    T dot(mtx<T>* A);
    void normalize();
    T norm();
    T sumAll();

    mtx<T>* symmetrize();

    mtx<T>* rowDistance();
};

/*------------------------------------------------------------------------------
    Constructors and destructor
------------------------------------------------------------------------------*/

template <typename T>
mtx<T>::mtx(){
    this->rows = 0; this->cols = 0;
    this->data = nullptr;
}

template <typename T>
mtx<T>::mtx(ulong rows, ulong cols){
    this->rows = rows; this->cols = cols;
    this->data = new T[rows*cols];
    if (this->data==nullptr){
        cout << "[miolo Error] Failed to allocate memory." << endl;
        cout << "[This is ocurred at the creation of a Matrix]" << endl;
    }
}

template <typename T>
mtx<T>::mtx(ulong rows, ulong cols, T init){
    this->rows = rows; this->cols = cols;
    this->data = new T[rows*cols];
    if (this->data!=nullptr){
        for (ulong k=0; k<rows*cols; k++){
            this->data[k] = init;
        }
    }
    else{
        cout << "[miolo Error] Failed to allocate memory." << endl;
        cout << "[This is ocurred at the creation of a Matrix]" << endl;
    }
}

template <typename T>
mtx<T>::mtx(mtx<T>& cp){
    this->rows = cp.rows; this->cols = cp.cols;
    this->data = new T[this->rows*this->cols];
    for (ulong k=0; k>this->rows*this->cols; k++){
            this->data[k] = cp.data[k];
    }
}

template <typename T>
mtx<T>::~mtx(){
    if (this->data!=nullptr)
        delete [] this->data;
}

/*------------------------------------------------------------------------------
    Misc
------------------------------------------------------------------------------*/

template <typename T>
void mtx<T>::copy(mtx<T>& cp){
    if (this->rows!=cp.rows || this->cols!=cp.cols)
        return;
    for (ulong k=0; k<this->rows*this->cols; k++){
        this->data[k] = cp.data[k];
    }
}

template <typename T>
void mtx<T>::copy(mtx<T>& cp, ulong* only, ulong lenOnly){
    if (this->rows!=cp.rows || this->cols!=cp.cols)
        return;
    for (ulong k=0; k<lenOnly; k++){
        for (ulong s=0; s<this->cols; s++){
            (*this)(only[k],s) = cp(only[k],s);
        }
    }
}
template <typename T>
mtx<T>* mtx<T>::transpose(){
    mtx<T>* out = new mtx<T>(this->cols,this->rows);
    for (ulong i=0; i<this->rows; i++)
        for (ulong j=0; j<this->cols; j++){
            (*out)(j,i) = (*this)(i,j);
        }
    return out;
}

template <typename T>
T mtx<T>::max(){
    T out = this->data[0];
    for (ulong k=1; k<this->rows*this->cols; k++){
        if (out<this->data[k])
            out = this->data[k];
    }
    return out;
}

template <typename T>
T mtx<T>::min(){
    T out = this->data[0];
    for (ulong k=1; k<this->rows*this->cols; k++){
        if (out>this->data[k])
            out = this->data[k];
    }
    return out;
}

template <typename T>
mtx<int>* argmax(mtx<T>* A){
    mtx<int>* out = new mtx<int>(A->rows,1);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<A->rows; i++){
        int max = 0;
        for (ulong j=1; j<A->cols; j++){
            if (A->data[i*A->cols+j]>A->data[i*A->cols+max]){
                max = j;
            }
        }
        out->data[i] = max;
    }
    return out;
}

template <typename T>
mtx<int>* argmin(mtx<T>* A){
    mtx<int>* out = new mtx<int>(A->rows,1);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<A->rows; i++){
        int min = 0;
        for (ulong j=1; j<A->cols; j++){
            if (A->data[i*A->cols+j]<A->data[i*A->cols+min]){
                min = j;
            }
        }
        out->data[i] = min;
    }
    return out;
}

template <typename T>
void mtx<T>::flatten(bool row){
    if (rows){
        this->cols = this->rows*this->cols;
        this->rows = 1;
    }
    else{
        this->rows = this->rows*this->cols;
        this->cols = 1;
    }
}

template <typename T>
void mtx<T>::reshape(unsigned long rows, unsigned long cols){
    this->rows = rows;
    this->cols = cols;
}

template <typename T>
mtx<T>* mtx<T>::cut(bool* targets){
    unsigned long count = 0;
    for (ulong i=0; i<this->rows; i++){
        if (targets[i])
            count++;
    }
    mtx<T>* out = new mtx<T>(count,this->cols);
    if (out->null())
        return nullptr;
    unsigned long k=0;
    for (ulong i=0; i<this->rows; i++){
        if (targets[i]){
            for (ulong s=0; s<this->cols; s++){
                (*out)(k,s) = (*this)(i,s);
            }
            k++;
        }
    }
    return out;
}

template <typename T>
void mtx<T>::print(){
    if (this->null())
        return;
    for (ulong i=0; i<this->rows; i++){
        cout << "[ ";
        for (ulong j=0; j<this->cols; j++){
            cout << this->data[i*this->cols+j] << " ";
        }
        cout << "]"<< endl;
    }
}

/*------------------------------------------------------------------------------
    Algebra
------------------------------------------------------------------------------*/

template <typename T>
mtx<T>* mtx<T>::add(mtx<T>* A){
    mtx<T>* out = nullptr;
    if (A->rows==this->rows && A->cols==this->cols){
        out = new mtx<T>(this->rows,this->cols);
        for (ulong k=0; k<this->rows*this->cols; k++){
            out->data[k] = this->data[k]+A->data[k];
        }
    }
    return out;
}

template <typename T>
mtx<T>* mtx<T>::sub(mtx<T>* A){
    mtx<T>* out = nullptr;
    if (A->rows==this->rows && A->cols==this->cols){
        out = new mtx<T>(this->rows,this->cols);
        for (ulong k=0; k<this->rows*this->cols; k++){
            out->data[k] = this->data[k]-A->data[k];
        }
    }
    return out;
}

template <typename T>
mtx<T>* mtx<T>::smul(T value){
    mtx<T>* out = new mtx<T>(this->rows,this->cols);
    for (ulong k=0; k<this->rows*this->cols; k++){
        out->data[k] = this->data[k]*value;
    }
    return out;
}

template <typename T>
mtx<T>* mtx<T>::mmul(mtx<T>* A){
    mtx<T>* out = nullptr;
    if (A->rows==this->cols){
        out = new mtx<T>(this->rows,A->cols,0);
        for (ulong i=0; i<this->rows; i++)
            for (ulong j=0; j<A->cols; j++)
                for (ulong k=0; k<this->cols; k++){
                    (*out)(i,j) += (*this)(i,k)*(*A)(k,j);
                }
    }
    return out;
}

template <typename T>
mtx<T>* mtx<T>::hmul(mtx<T>* A){
    mtx<T> *out = nullptr;
    if (A->rows==this->rows && A->cols==this->cols){
        out = new mtx<T>(this->rows,this->cols);
        for (ulong k=0; k<this->rows*this->cols; k++){
            out->data[k] = this->data[k]*A->data[k];
        }
    }
    return out;
}

/*------------------------------------------------------------------------------
    Other mathematical operations
------------------------------------------------------------------------------*/

template <typename T>
T mtx<T>::dot(mtx<T>* A){
    T out = 0;
    if (this->rows==A->rows && this->cols==A->cols){
        for (ulong i=0; i<A->cols*A->rows; i++){
            out += this->data[i]*A->data[i];
        }
    }
    return out;
}

template <typename T>
void mtx<T>::normalize(){
    for (ulong i=0; i<this->rows; i++){
        T z = 0;
        for (ulong j=0; j<this->cols; j++){
            z += this->data[i*this->cols+j];
        }
        if (z!=0){
            for (ulong j=0; j<this->cols; j++){
                this->data[i*this->cols+j] /= z;
            }
        }
        else{
            cout << "[miolo Warning] Preventing division by zero." << endl;
            cout << "<< This ocurred in Matrix.normalize at row " << i << \
                " >>." << endl;
            for (ulong j=0; j<this->cols; j++){
                this->data[i*this->cols+j] = 1/this->cols;
            }
        }
    }
}

template <typename T>
T mtx<T>::norm(){
    T out = fabs(this->data[0]);
    for (ulong k=1; k<this->rows*this->cols; k++){
        if (out<fabs(this->data[k])){
            out = fabs(this->data[k]);
        }
    }
    return out;
}

template <typename T>
mtx<T>* mtx<T>::rowDistance(){
    mtx<T>* out = new mtx<T>(this->rows,this->rows,0);
    if (out!=nullptr){
        for (ulong i=0; i<this->rows; i++)
            for (ulong j=i+1; j<this->rows; j++){
                for (ulong s=0; s<this->cols; s++){
                    double diff = (*this)(i,s)-(*this)(j,s);
                    (*out)(i,j) += diff*diff;
                }
                (*out)(j,i) = (*out)(i,j);
            }
    }
    return out;
}

template <typename T>
T mtx<T>::sumAll(){
    T out = 0;
    for (ulong k=0; k<this->rows*this->cols; k++){
        out += this->data[k];
    }
    return out;
}

/*------------------------------------------------------------------------------
    Matrix concatenation
------------------------------------------------------------------------------*/

template <typename T>
mtx<T>* concat(mtx<T> A, mtx<T> B){
    mtx<T>* out = new mtx<T>(A.rows+B.rows,A.cols);
    if (out->null())
        return nullptr;
    for (ulong i=0; i<out->rows; i++){
        if (i<A.rows)
            for (ulong s=0; s<out->cols; s++){
                (*out)(i,s) = A(i,s);
            }
        else
            for (ulong s=0; s<out->cols; s++){
                (*out)(i,s) = B(i-A.rows,s);
            }
    }
    return out;
}

#endif