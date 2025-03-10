#include "mtx.h"
#include "graph.h"

class classSeparation{
public:

    template <typename T>
    T operator()(mtx<T>& dist, int* labels, bool intra){
        T out = 0;
        for (ulong i=0; i<dist.rows; i++)
            for (ulong j=0; j<dist.cols; j++){
                bool check = (labels[i]==labels[j]);
                if (check==intra)
                    out += dist(i,j);
            }
        return out/2;
    }

    template <typename T>
    T operator()(graph<T>& dist, int* labels, bool intra){
        T out = 0;
        for (ulong k=0; k<dist.edges; k++){
            bool check = (labels[dist[k].i]==labels[dist[k].j]);
            if (check==intra)
                out += dist[k].w;
        }
        return out/2;
    }
};

class weightedSquareDistance{
public:

    template <typename T>
    T operator()(mtx<T>& M, graph<T>& G){
        T out = 0;
        for (ulong k=0; k<G.edges; k++){
            T aux = 0;
            for (ulong s=0; s<M.cols; s++){
                aux += (M(G[k].i,s)-M(G[k].j,s))*(M(G[k].i,s)-M(G[k].j,s));
            }
            out += G[k].w*aux;
        }
        return out;
    }

    template <typename T>
    T operator()(mtx<T>& M, mtx<T>& W){
        T out = 0;
        for (ulong i=0; i<W.rows; i++){
            for (ulong j=0; j<W.cols; j++){
                T aux = 0;
                for (ulong s=0; s<M.cols; s++)
                    aux += (M(i,s)-M(j,s))*(M(i,s)-M(j,s));
                out += W(i,j)*aux;
            }
        }
        return out;
    }
};

class weightedDot{
public:

    template <typename T>
    T operator()(mtx<T>& M, graph<T>& G){
        T out = 0;
        for (ulong k=0; k<G.edges; k++){
            T aux = 0;
            for (ulong s=0; s<M.cols; s++){
                aux += M(G[k].i,s)*M(G[k].j,s);
            }
            out += G[k].w*aux;
        }
        return out;
    }

    template <typename T>
    T operator()(mtx<T>& M, mtx<T>& W){
        T out = 0;
        for (ulong i=0; i<W.rows; i++){
            for (ulong j=0; j<W.cols; j++){
                T aux = 0;
                for (ulong s=0; s<M.cols; s++)
                    aux += M(i,s)*M(j,s);
                out += W(i,j)*aux;
            }
        }
        return out;
    }
};

class pottsEnergy{
public:

    template <typename T>
    T operator()(graph<T>& G, int* labels){
        T out = 0;
        for (ulong k=0; k<G.edges; k++){
            if (labels[G[k].i]==labels[G[k].j])
                out -= G[k].w;
        }
        return out;
    }

    template <typename T>
    T operator()(mtx<T>& M, int* labels){
        T out = 0;
        for (ulong i=0; i<M.rows; i++)
            for (ulong j=0; j<M.cols; j++){
                if (labels[i]==labels[j])
                    out -= M(i,j);
            }
        return out;
    }
};