
cdef extern from "<new>":
    pass
cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        pass
cdef extern from "<algorithm>":
    pass
cdef extern from "cpp/aux.h":
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

        void print()
        
        T& operator()(unsigned long i, unsigned long j)
        T& operator[](unsigned long k)

        T max()
        T min()

        mtx[T]* transpose()
        void copy(mtx[T]& cp)
        void copy(mtx[T]& cp, unsigned long* only, unsigned long lenOnly);
        void flatten(bint rows)
        void reshape(unsigned long rows, unsigned long cols)
        mtx[T]* cut(unsigned long rmin, unsigned long rmax, unsigned long cmin, unsigned long cmax)

        mtx[T]* add(mtx[T]* A)
        mtx[T]* sub(mtx[T]* A)
        mtx[T]* mmul(mtx[T]* A)
        mtx[T]* smul(T value)
        mtx[T]* hmul(mtx[T]* value)

        T dot(mtx[T]* A)
        void normalize()
        T norm()
        T trace()

        mtx[T]* rowDistance()
        T sumAll()
        T kthSmallest(unsigned long k)
        mtx[T]* partition(int* labels, int idLabel)

    mtx[int]* argkmax[T](mtx[T]& A, int k)
    mtx[int]* argkmin[T](mtx[T]& A, int k)
    mtx[int]* argmax[T](mtx[T]* A)
    mtx[int]* argmin[T](mtx[T]* A)

    mtx[T]* concat[T](mtx[T]&, mtx[T]&)

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

        T max()
        T min()

        graph[T]* smul(T value)
        mtx[T]* mmul(mtx[T]* M)
        graph[T]* add(graph[T]* G)
        graph[T]* sub(graph[T]* G)
        graph[T]* hmul(graph[T]* G)
        mtx[T]* propagate(mtx[T]& M, bint* clamped)

        bint isolatedNodes()

        mtx[T]* degree()
        void normalize()
        graph[T]* laplacian()
        void print()

        T gaussianScale()

        mtx[T]* densify()

    graph[T]* sparsifyGraphThreshold[T](mtx[T]& M, double thresh)
    graph[T]* sparsifyGraphKNN[T](mtx[T]& M, unsigned long knn)

    graph[T]* mst[T](mtx[T]& M)
    graph[T]* mst[T](graph[T]& M)

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

        void normalize()
        digraph[T]* copyStructure()
        digraph[T]* copy()
        unsigned long* shape()
        digraph[T]* transpose()
        digraph[T]* symmetrize()
        void connectRegular(mtx[int], mtx[T])

        digraph[T]* add(digraph[T]& D)
        digraph[T]* sub(digraph[T]& D)
        digraph[T]* hmul(digraph[T]& D)
        digraph[T]* smul(T value)
        mtx[T]* mmul(mtx[T]& D)

        T gaussianScale()
    
    digraph[T]* toDigraph[T](graph[T]& G)
    graph[T]* toGraph[T](digraph[T]& D)
    digraph[T]* sparsifyDigraphThreshold[T](mtx[T]& M, double Threshold)
    digraph[T]* sparsifyDigraphKNN[T](mtx[T]& M, double Threshold)

cdef extern from "cpp/diagonal.h":

    cdef cppclass diagonal[T]:

        unsigned long dim
        T* data

        diagonal();
        diagonal(unsigned long dim)
        diagonal(unsigned long dim, T init)

        bint null()

        mtx[T]* lmul(mtx[T]& M)
        mtx[T]* rmul(mtx[T]& M)

        digraph[T]* lmul(digraph[T]& G)
        digraph[T]* rmul(digraph[T]& G)

        digraph[T]* lmul(graph[T]& G)
        digraph[T]* rmul(graph[T]& G)

        diagonal[T]* mul(diagonal[T]& D)
        diagonal[T]* add(diagonal[T]& D)
        diagonal[T]* sub(diagonal[T]& D)
        diagonal[T]* smul(T value)

cdef extern from "cpp/euclidean.h":

    cdef cppclass euclidean:
        mtx[T]* mean[T](mtx[T]& M)
        mtx[T]* variance[T](mtx[T]& M)
        mtx[T]* dot[T](mtx[T]& A)
        mtx[T]* distance[T](mtx[T]& A)

        mtx[T]* minmaxNormalize[T](mtx[T]& M)
        mtx[T]* rowNormalize[T](mtx[T]& M)
        mtx[T]* colNormalize[T](mtx[T]& M)
        mtx[T]* gaussianNormalize[T](mtx[T]& M)

        mtx[T]* centroidDistance[T](mtx[T]& M, mtx[T]& center)
        mtx[T]* kmpp[T](mtx[T]& data, int k)

cdef extern from "cpp/sphere.h":

    cdef cppclass sphere:

        double r

        mtx[T]* stereographicProjection[T](mtx[T]& M)

        mtx[T]* fromEuclidean[T](mtx[T]& M)
        mtx[T]* toEuclidean[T](mtx[T]& M)
        mtx[T]* coordinateReady[T](mtx[T]& M, unsigned long azimuth)

        mtx[T]* distance[T](mtx[T]& M)

        bint isIn[T](mtx[T]&,T)
        bint isTangent[T](mtx[T]&,mtx[T]&,T)

        mtx[T]* exponential[T](mtx[T]& at, mtx[T]& M)
        mtx[T]* logarithm[T](mtx[T]& frm, mtx[T]& to)
        
        mtx[T]* centroidDistance[T](mtx[T]& data, mtx[T]& center)

cdef extern from "cpp/poincare.h":

    cdef cppclass poincare:

        double c

        mtx[T]* distance[T](mtx[T]& M)
        bint isIn[T](mtx[T]& M)
        mtx[T]* madd[T](mtx[T]& A, mtx[T]& B)

        mtx[T]* exponential[T](mtx[T]& at, mtx[T]& M)
        mtx[T]* logarithm[T](mtx[T]& start, mtx[T]& end)

        mtx[T]* mean[T](mtx[T]& M)
        mtx[T]* centroidDistance[T](mtx[T]& M, mtx[T]& center)

        double hyperbolicity[T](mtx[T]& M)

cdef extern from "cpp/lorentz.h":

    cdef cppclass lorentz:

        double beta

        bint isIn[T](mtx[T]& M, double tol)
        
        mtx[T]* dot[T](mtx[T]& M)
        mtx[T]* distance[T](mtx[T]& M)
        mtx[T]* lorentzDistance[T](mtx[T]& M)
        
        mtx[T]* fromEuclidean[T](mtx[T]& M)
        mtx[T]* toEuclidean[T](mtx[T]& M)
        
        mtx[T]* exponential[T](mtx[T]& at, mtx[T]& M)
        mtx[T]* tangentProjection[T](mtx[T]& at, mtx[T]& M)

        mtx[T]* centroidDistance[T](mtx[T]& data, mtx[T]& center)
        mtx[T]* mean[T](mtx[T]& data)
        mtx[T]* variance[T](mtx[T]& data)

cdef extern from "cpp/ufunctions.h":
    
    cdef cppclass ufunction:
        double (*f)(double x)
        T operator()[T](T x)
        mtx[T]* mtxApply[T](mtx[T]& M)
        graph[T]* graphApply[T](graph[T]& G)
        digraph[T]* digraphApply[T](digraph[T]& G)
    
    double reciprocal(double x)

cdef extern from "cpp/other_functions.h":

    cdef cppclass classSeparation:
        T operator()[T](mtx[T]& dist, int* labels, bint intra)
        T operator()[T](graph[T]& dist, int* labels, bint intra)
    
    cdef cppclass weightedSquareDistance:
        T operator()[T](mtx[T]& M, graph[T]& G)
        T operator()[T](mtx[T]& M, mtx[T]& W)
    
    cdef cppclass weightedDot:
        T operator()[T](mtx[T]& M, graph[T]& G)
        T operator()[T](mtx[T]& M, mtx[T]& W)
    
    cdef cppclass pottsEnergy:
        T operator()[T](mtx[T]&, int*)
        T operator()[T](graph[T]&, int*)