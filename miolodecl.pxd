
cdef extern from "<new>":
    pass
cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        pass
cdef extern from "<algorithm>":
    pass

cdef extern from "<cmath>":

    double cos(double x)
    double sin(double x)
    double tan(double x)

    double acos(double x)
    double asin(double x)
    double atan(double x)

    double cosh(double x)
    double sinh(double x)
    double tanh(double x)

    double acosh(double x)
    double asinh(double x)
    double atanh(double x)

    double exp(double x)
    double log(double x)

    double fabs(double x)
    double sqrt(double x)

cdef extern from "cpp/mtx.h":
    
    cdef cppclass mtx[T]:
        unsigned long rows, cols
        T* data

        mtx()
        mtx(unsigned long rows, unsigned long cols)
        mtx(unsigned long rows, unsigned long cols, T init)
        mtx(mtx[T]& cp)

        void operator=(mtx[T] cp)

        bint null()
        void wrap(mtx[T]* target)
        
        T& operator()(unsigned long i, unsigned long j)
        T& operator[](unsigned long k)

        mtx[T]* transpose()
        void copy(mtx[T]* cp)
        void flatten(bint rows)
        void reshape(unsigned long rows, unsigned long cols)
        mtx[T]* cut(bint* )

        mtx[T]* add(mtx[T]* A)
        mtx[T]* sub(mtx[T]* A)
        mtx[T]* mmul(mtx[T]* A)
        mtx[T]* smul(T value)
        mtx[T]* hmul(mtx[T]* value)

        T dot(mtx[T]* A)
        void normalize()
        T norm()

        mtx[T]* rowDistance()
        T sumAll()

    mtx[int]* argmax[T](mtx[T]* A)
    mtx[int]* argmin[T](mtx[T]* A)

    mtx[T]* concat[T](mtx[T],mtx[T])

cdef extern from "cpp/graph.h":

    cdef cppclass edge[T]:
        unsigned long i, j
        T w
    
    cdef cppclass graph[T]:
        unsigned long nodes, edges
        edge[T]* e

        graph()
        graph(unsigned long nodes, unsigned long edges)
        graph(unsigned long nodes, unsigned long edges, T init)
        graph(graph[T]& cp)

        bint null()
        void wrap(graph[T]* target)

        graph[T]* smul(T value)
        mtx[T]* mmul(mtx[T]* M)
        graph[T]* add(graph[T]* G)
        graph[T]* sub(graph[T]* G)
        graph[T]* hmul(graph[T]* G)
        mtx[T]* propagate(mtx[T]* M, bint* clamped)

        bint isolatedNodes()

        mtx[T]* degree()
        void normalize()
        graph[T]* laplacian()

        mtx[T]* densify()

    graph[T]* sparsifyThreshold[T](mtx[T]* M, T thresh)
    graph[T]* sparsifyKNN[T](mtx[T]* M, unsigned long knn)

cdef extern from "cpp/digraph.h":

    cdef cppclass neighbor[T]:
        unsigned long idx
        T w
    
    cdef cppclass neighborhood[T]:
        pass
    
    cdef cppclass digraph[T]:
        unsigned long nodes
        neighborhood[T]* hood

        digraph()
        digraph(unsigned long nodes)

        bint null()

        void connect(unsigned long i, unsigned long j, T w)
        void connect(unsigned long i, unsigned long j)

        digraph[T]* copyStructure()
        digraph[T]* copy()
        unsigned long* shape()
        digraph[T]* transpose()
        digraph[T]* symmetrize()

        digraph[T]* add(digraph[T] D)
        digraph[T]* sub(digraph[T] D)
        digraph[T]* hmul(digraph[T] D)
        digraph[T]* smul(T value)
        mtx[T]* mmul(mtx[T] D)
    
    digraph[T]* toDigraph[T](graph[T] G)

cdef extern from "cpp/diagonal.h":

    cdef cppclass diagonal[T]:

        unsigned long dim
        T* data

        diagonal();
        diagonal(unsigned long dim)
        diagonal(unsigned long dim, T init)

        bint null()

        mtx[T]* lmul(mtx[T] M)
        mtx[T]* rmul(mtx[T] M)

        digraph[T]* lmul(digraph[T] G)
        digraph[T]* rmul(digraph[T] G)

        digraph[T]* lmul(graph[T] G)
        digraph[T]* rmul(graph[T] G)

        diagonal[T]* mul(diagonal[T] D)
        diagonal[T]* add(diagonal[T] D)
        diagonal[T]* sub(diagonal[T] D)
        diagonal[T]* smul(T value)

cdef extern from "cpp/euclidean.h":

    mtx[T]* mean[T](mtx[T] M)
    mtx[T]* dot[T](mtx[T] A)
    mtx[T]* distance[T](mtx[T] A)


cdef extern from "cpp/sphere.h":

    cdef cppclass sphere[T]:
        
        T r

        mtx[T]* stereographicProjection(mtx[T] M)
        mtx[T]* tangentProjection(mtx[T] base, mtx[T] M)

        mtx[T]* fromEuclidean(mtx[T] M)
        mtx[T]* toEuclidean(mtx[T] M)

        mtx[T]* distance(mtx[T] M)

        bint isIn(mtx[T],T)
        bint isTangent(mtx[T],mtx[T],T)

cdef extern from "cpp/simplex.h":

    cdef cppclass simplex:

        mtx[T]* softmaxRetraction[T](mtx[T] at, mtx[T] M)
        mtx[T]* tangentProjection[T](mtx[T] base, mtx[T] M)
        
        bint isIn[T](mtx[T],T)
        bint isTangent[T](mtx[T],mtx[T],T)

cdef extern from "cpp/kmeans.h":

    cdef cppclass kmeans:
        mtx[bint]* clamped
        mtx[bint]* labels
        unsigned long N, q

        kmeans()
        kmeans(unsigned long, unsigned long)
        kmeans(mtx[bint]* , unsigned long, unsigned long)

        void clamp(int* targets)
        mtx[int]* getLabels()
        void setLabels[T](mtx[T],mtx[T])
        mtx[T]* groupFeatures[T](mtx[T] data, unsigned long gNum)

