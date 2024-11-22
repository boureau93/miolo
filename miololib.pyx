cimport miolodecl as mld
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from cython.operator cimport dereference as drf

str_memory = "Failed to allocate memory."
global_ctype = "float"

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Wrapper for miolo objects
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class mioloObject:
    """
        A container object for two objects in the miolo library. This is usually
        not useful outside Cython.
    """
    #Matrix
    cdef mld.mtx[int]* mtxInt
    cdef mld.mtx[float]* mtxFloat
    cdef mld.mtx[double]* mtxDouble
    #Graph
    cdef mld.graph[int]* graphInt
    cdef mld.graph[float]* graphFloat
    cdef mld.graph[double]* graphDouble
    #Digraph
    cdef mld.digraph[int]* digraphInt
    cdef mld.digraph[float]* digraphFloat
    cdef mld.digraph[double]* digraphDouble
    #Diagonal matrices
    cdef mld.diagonal[int]* diagonalInt
    cdef mld.diagonal[float]* diagonalFloat
    cdef mld.diagonal[double]* diagonalDouble

    def isMatrix(self):
        """
            Returns true if object is a miolo.Matrix.
        """
        if self.mtxInt is not NULL:
            return True
        if self.mtxFloat is not NULL:
            return True
        if self.mtxDouble is not NULL:
            return True
        return False
    
    def isGraph(self):
        """
            Returns true if object is miolo.Graph.
        """
        if self.graphInt is not NULL:
            return True
        if self.graphFloat is not NULL:
            return True
        if self.graphDouble is not NULL:
            return True
        return False
    
    def isDigraph(self):
        """ 
            Returns true if object is miolo.Digraph
        """
        if self.digraphInt is not NULL:
            return True
        if self.digraphFloat is not NULL:
            return True
        if self.digraphDouble is not NULL:
            return True
        return False
    
    def isDiagonal(self):
        """ 
            Returns true if object is miolo.Digraph
        """
        if self.diagonalInt is not NULL:
            return True
        if self.digraphFloat is not NULL:
            return True
        if self.digraphDouble is not NULL:
            return True
        return False
    
    @property
    def ctype(self):
        pass
    @property
    def rows(self):
        pass
    @property
    def cols(self):
        pass
    @property
    def nodes(self):
        pass
    @property
    def dim(self):
        pass

ctypes = ["int","float","double"]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Matrix
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class Matrix(mioloObject):

    """
        A class for dense matrices. 
        Initializes a C++ object that stores a dense matrix.
        rows: number of rows of Matrix
        cols: number of cols of Matrix
        ctype: underlying C type for Matrix data
    """

    cdef object cType

    def __cinit__(self, unsigned long rows=0, unsigned long cols=0, init=0,
        ctype=global_ctype):
        if ctype in ctypes:
            self.cType = ctype
            if rows>0 and cols>0:
                if ctype=="int":
                    self.mtxInt = new mld.mtx[int](rows,cols,init)
                if ctype=="float":
                    self.mtxFloat = new mld.mtx[float](rows,cols,init)
                if ctype=="double":
                    self.mtxDouble = new mld.mtx[double](rows,cols,init)
            else:
                if ctype=="int":
                    self.mtxInt = NULL
                if ctype=="float":
                    self.mtxFloat = NULL
                if ctype=="double":
                    self.mtxDouble = NULL
        else:
            raise Exception("Unknown ctype.")
    
    def __dealloc__(self):
        if self.ctype=="int":
            del self.mtxInt
        if self.ctype=="float":
            del self.mtxFloat
        if self.ctype=="double":
            del self.mtxDouble

    @property
    def ctype(self):
        return str(self.cType)
    
    @property
    def rows(self):
        if self.ctype=="int":
            return self.mtxInt.rows
        if self.ctype=="float":
            return self.mtxFloat.rows
        if self.ctype=="double":
            return self.mtxDouble.rows
    
    @property
    def cols(self):
        if self.ctype=="int":
            return self.mtxInt.cols
        if self.ctype=="float":
            return self.mtxFloat.cols
        if self.ctype=="double":
            return self.mtxDouble.cols
    
    @property
    def numpy(self):
        """
            Interaction with numpy.
        """
        cdef unsigned long i, j
        out = np.empty((self.rows,self.cols))
        if self.ctype == "int":
            for i in range(self.mtxInt.rows):
                for j in range(self.mtxInt.cols):
                    out[i][j] = self.mtxInt.data[i*self.mtxInt.cols+j]
        if self.ctype == "float":
            for i in range(self.mtxFloat.rows):
                for j in range(self.mtxFloat.cols):
                    out[i][j] = self.mtxFloat.data[i*self.mtxFloat.cols+j]
        if self.ctype == "double":
            for i in range(self.mtxDouble.rows):
                for j in range(self.mtxDouble.cols):
                    out[i][j] = self.mtxDouble.data[i*self.mtxDouble.cols+j]
        return out
    
    @numpy.setter
    def numpy(self, data):
        cdef unsigned long i, j
        if np.shape(data)!=(self.rows,self.cols):
            raise Exception("Incompatible shape for miolo.Matrix data.")
        if self.ctype == "int":
            for i in range(self.mtxInt.rows):
                for j in range(self.mtxInt.cols):
                    self.mtxInt.data[i*self.mtxInt.cols+j] = data[i][j]
        if self.ctype == "float":
            for i in range(self.mtxFloat.rows):
                for j in range(self.mtxFloat.cols):
                    self.mtxFloat.data[i*self.mtxFloat.cols+j] = data[i][j]
        if self.ctype == "double":
            for i in range(self.mtxDouble.rows):
                for j in range(self.mtxDouble.cols):
                    self.mtxDouble.data[i*self.mtxDouble.cols+j] = data[i][j]

    def __len__(self):
        return self.rows*self.cols

    def __getitem__(self, unsigned long k):
        if k >= <unsigned long>self.rows*self.cols:
            raise Exception("Index k is out of bounds.")
        if self.ctype=="int":
            return self.mtxInt.data[k]
        if self.ctype=="float":
            return self.mtxFloat.data[k]
        if self.ctype=="double":
            return self.mtxDouble.data[k]
    
    def __setitem__(self, unsigned long k, double value):
        if k >= <unsigned long>self.rows*self.cols:
            raise Exception("Index k is out of bounds.")
        if self.ctype=="int":
            self.mtxInt.data[k] = <int>value
        if self.ctype=="float":
            self.mtxFloat.data[k] = <float>value
        if self.ctype=="double":
            self.mtxDouble.data[k] = value
    
    def copy(self, Matrix M):
        if M.rows!=self.rows or M.cols!=self.cols:
            raise Exception("Matrices must have same shape.")
        if self.ctype!=M.ctype:
            raise TypeError("Matrices must share same ctype.")
        if self.ctype=="int":
            self.mtxInt.copy(M.mtxInt)
        if self.ctype=="float":
            self.mtxFloat.copy(M.mtxFloat)
        if self.ctype=="double":
            self.mtxDouble.copy(M.mtxDouble)

    #---------------------------------------------------------------------------
    #   Other useful stuff
    #---------------------------------------------------------------------------

    def normalize(self):
        """
            Row normalization in order to elements in a same row sum to 1.
        """
        if self.ctype=="int":
            self.mtxInt.normalize()
        if self.ctype=="float":
            self.mtxFloat.normalize()
        if self.ctype=="double":
            self.mtxDouble.normalize()
    
    def transpose(self):
        """
            Returns the transpose of a Matrix.
        """
        out = Matrix(ctype=self.ctype)
        if self.ctype == "int":
            out.mtxInt = self.mtxInt.transpose()
        if self.ctype == "float":
            out.mtxFloat = self.mtxFloat.transpose()
        if self.ctype == "double":
            out.mtxDouble = self.mtxDouble.transpose()
    

    def flatten(self, bool rows=True):
        """
            Inplace flattening of a Matrix.
            If rows is True, Matrix is flattened to have a single row. If rows
            is False, flattened to a single column.
            NOTE: This only changes a view on the Matrix data. No changes are
            made on stored data.
        """
        if self.ctype=="int":
            self.mtxInt.flatten(rows)
        if self.ctype=="float":
            self.mtxFloat.flatten(rows)
        if self.ctype=="int":
            self.mtxDouble.flatten(rows)
    
    
    def reshape(self, unsigned long rows, unsigned long cols):
        """
            Inplace reshape of a Matrix. Can be done only if rows*cols is equal
            to self.rows*self.cols.
            NOTE: This only changes a view on the Matrix data. No changes are
            made on stored data.
        """
        if self.rows*self.cols!=rows*cols:
            raise Exception("Invalid new shape")
        if self.ctype=="int":
            self.mtxInt.reshape(rows,cols)
        if self.ctype=="float":
            self.mtxFloat.reshape(rows,cols)
        if self.ctype=="int":
            self.mtxDouble.reshape(rows,cols)
        
    #---------------------------------------------------------------------------
    #   Algebra
    #---------------------------------------------------------------------------

    def __add__(self, Matrix A):
        if self.ctype!=A.ctype:
            raise Exception("Matrix operations require same ctype.")
        if self.rows!=A.rows or self.cols!=A.cols:
            raise Exception("Incompatible shape for Matrix addition.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt.wrap(self.mtxInt.add(A.mtxInt))
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat.wrap(self.mtxFloat.add(A.mtxFloat))
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble.wrap(self.mtxDouble.add(A.mtxDouble))
            return out
    
    def __sub__(self, Matrix A):
        if self.ctype!=A.ctype:
            raise Exception("Matrix operations require same ctype.")
        if self.rows!=A.rows or self.cols!=A.cols:
            raise Exception("Incompatible shape for Matrix subtraction.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt.wrap(self.mtxInt.sub(A.mtxInt))
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat.wrap(self.mtxFloat.sub(A.mtxFloat))
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble.wrap(self.mtxDouble.sub(A.mtxDouble))
            return out
    
    def __mul__(self, value):
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt.wrap(self.mtxInt.smul(value))
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat.wrap(self.mtxFloat.smul(value))
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble.wrap(self.mtxDouble.smul(value))
            return out
    
    def __rmul__(self, value):
        return self*value
    
    def __truediv__(self, value):
        if value==0:
            raise ValueError("Attempting division by zero.")
        return self*(1./value)
    
    def __and__(self, Matrix A):
        if self.ctype!=A.ctype:
            raise Exception("Matrix operations require same ctype.")
        if self.cols!=A.rows:
            raise Exception("Incompatible shape for Matrix product.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt.wrap(self.mtxInt.mmul(A.mtxInt))
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat.wrap(self.mtxFloat.mmul(A.mtxFloat))
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble.wrap(self.mtxDouble.mmul(A.mtxDouble))
            return out

    def __mod__(self, Matrix A):
        if self.ctype!=A.ctype:
            raise Exception("Matrix operations require same ctype.")
        if self.rows!=A.rows or self.cols!=A.cols:
            raise Exception("Incompatible shape for Hadamard product.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt.wrap(self.mtxInt.hmul(A.mtxInt))
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat.wrap(self.mtxFloat.hmul(A.mtxFloat))
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble.wrap(self.mtxDouble.hmul(A.mtxDouble))
            return out
    
    def __abs__(self):
        if self.ctype=="int":
            return self.mtxInt.norm()
        if self.ctype=="float":
            return self.mtxFloat.norm()
        if self.ctype=="double":
            return self.mtxDouble.norm()

    #---------------------------------------------------------------------------
    #   Useful
    #---------------------------------------------------------------------------

    def argmax(self):
        out = Matrix(ctype="int")
        if self.ctype=="int":
            out.mtxInt = mld.argmax(self.mtxInt)
        if self.ctype=="float":
            out.mtxInt = mld.argmax(self.mtxFloat)
        if self.ctype=="double":
            out.mtxInt = mld.argmax(self.mtxDouble)
        return out
    
    def argmin(self):
        out = Matrix(ctype="int")
        if self.ctype=="int":
            out.mtxInt = mld.argmin(self.mtxInt)
        if self.ctype=="float":
            out.mtxInt = mld.argmin(self.mtxFloat)
        if self.ctype=="double":
            out.mtxInt = mld.argmin(self.mtxDouble)
        return out
    
    def rowDistance(self):
        """
            Returns the self.rows x self.rows Matrix for which each entry (i,j)
            is the euclidean distance between rows i and j of self.
        """
        out = Matrix(ctype=self.ctype)
        if self.ctype=="int":
            out.mtxInt = self.mtxInt.rowDistance()
        if self.ctype=="float":
            out.mtxFloat = self.mtxFloat.rowDistance()
        if self.ctype=="double":
            out.mtxDouble = self.mtxDouble.rowDistance()
        return out
    
    def sumAll(self):
        """
            Returns the sum of all elements in a Matrix.
        """
        if self.ctype=="int":
            return self.mtxInt.sumAll()
        if self.ctype=="float":
            return self.mtxFloat.sumAll()
        if self.ctype=="double":
            return self.mtxDouble.sumAll()

def dot(Matrix A, Matrix B):
    """
        Frobenius dot product between two matrices.
    """
    if A.ctype!=B.ctype:
        raise TypeError("Matrices must have the same ctype.")
    if A.ctype=="int":
        return A.mtxInt.dot(B.mtxInt)
    if A.ctype=="float":
        return A.mtxFloat.dot(B.mtxFloat)
    if A.ctype=="double":
        return A.mtxDouble.dot(B.mtxDouble)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Graph
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class Graph(mioloObject):

    """
        A class for undirected graphs. 
        Initializes a C++ object that stores a sparse weighted adjacency 
        matrix.
        nodes: number of nodes in Graph
        edges: number of edges in Graph
        ctype: underlying C type for Graph weights
    """
    
    cdef object cType

    def __cinit__(self, unsigned long nodes=0, unsigned long edges=0, 
        ctype=global_ctype):
        if ctype in ctypes:
            self.cType = ctype
            if nodes>0 and edges>0:
                if ctype=="int":
                    self.graphInt = new mld.graph[int](nodes,edges,1)
                if ctype=="float":
                    self.graphFloat = new mld.graph[float](nodes,edges,1)
                if ctype=="double":
                    self.graphDouble = new mld.graph[double](nodes,edges,1)
            else:
                if ctype=="int":
                    self.graphInt = NULL
                if ctype=="float":
                    self.graphFloat = NULL
                if ctype=="double":
                    self.graphDouble = NULL
        else:
            raise Exception("Unknown ctype.")
    
    def __dealloc__(self):
        if self.ctype=="int":
            del self.graphInt
        if self.ctype=="float":
            del self.graphFloat
        if self.ctype=="double":
            del self.graphDouble
    
    @property
    def ctype(self):
        return str(self.cType)

    @property
    def nodes(self):
        if self.ctype=="int":
            return self.graphInt.nodes
        if self.ctype=="float":
            return self.graphFloat.nodes
        if self.ctype=="double":
            return self.graphDouble.nodes
    
    @property
    def edges(self):
        if self.ctype=="int":
            return self.graphInt.edges
        if self.ctype=="float":
            return self.graphFloat.edges
        if self.ctype=="double":
            return self.graphDouble.edges
    
    @property
    def structure(self):
        """
            Returns the pairs of existing edges in graph as a numpy array. This
            works both as a getter and a setter.
        """
        cdef unsigned long k
        out = np.empty((self.edges,2),np.ulong)
        if self.ctype == "int":
            for k in range(self.graphInt.edges):
                out[k][0] = self.graphInt.e[k].i
                out[k][1] = self.graphInt.e[k].j
        if self.ctype == "float":
            for k in range(self.graphFloat.edges):
                out[k][0] = self.graphFloat.e[k].i
                out[k][1] = self.graphFloat.e[k].j
        if self.ctype == "double":
            for k in range(self.graphDouble.edges):
                out[k][0] = self.graphDouble.e[k].i
                out[k][1] = self.graphDouble.e[k].j
        return out
    
    @structure.setter
    def structure(self, data):
        cdef unsigned long k
        if np.shape(data)!=(self.edges,2):
            raise Exception("Incompatible shape for Graph data.")
        if self.ctype == "int":
            for k in range(self.graphInt.edges):
                self.graphInt.e[k].i = data[k][0]
                self.graphInt.e[k].j = data[k][1]
        if self.ctype == "float":
            for k in range(self.graphFloat.edges):
                self.graphFloat.e[k].i = data[k][0]
                self.graphFloat.e[k].j = data[k][1]
        if self.ctype == "double":
            for k in range(self.graphDouble.edges):
                self.graphDouble.e[k].i = data[k][0]
                self.graphDouble.e[k].j = data[k][1]
    
    @property
    def weights(self):
        """
            Returns the pairs weights of edges in graph as a numpy array. This
            works both as a getter and a setter.
        """
        cdef unsigned long k
        if self.ctype == "int":
            out = np.empty(self.edges,dtype=np.single)
            for k in range(self.graphInt.edges):
                out[k] = self.graphInt.e[k].w
        if self.ctype == "float":
            for k in range(self.graphFloat.edges):
                out[k] = self.graphFloat.e[k].w
        if self.ctype == "double":
            for k in range(self.graphDouble.edges):
                out[k][0] = self.graphDouble.e[k].w
        return out
    
    @weights.setter
    def weights(self, data):
        cdef unsigned long k
        if len(data)!=self.edges:
            raise Exception("Incompatible shape for Graph data.")
        if self.ctype == "int":
            for k in range(self.graphInt.edges):
                self.graphInt.e[k].w = data[k]
        if self.ctype == "float":
            for k in range(self.graphFloat.edges):
                self.graphFloat.e[k].w = data[k]
        if self.ctype == "double":
            for k in range(self.graphDouble.edges):
                self.graphDouble.e[k].w = data[k]
    
    def __len__(self):
        return self.edges
    
    def __getitem__(self, unsigned long k):
        if k >= <unsigned long>self.edges:
            raise Exception("Index is out of bounds.")
        if self.ctype=="int":
            return (self.graphInt.e[k].i,self.graphInt.e[k].j,self.graphInt.e[k].w)
        if self.ctype=="float":
            return (self.graphFloat.e[k].i,self.graphFloat.e[k].j,self.graphFloat.e[k].w)
        if self.ctype=="double":
            return (self.graphDouble.e[k].i,self.graphDouble.e[k].j,self.graphDouble.e[k].w)
    
    def isolatedNodes(self):
        if self.ctype=="int":
            return self.graphInt.isolatedNodes()
        if self.ctype=="float":
            return self.graphFloat.isolatedNodes()
        if self.ctype=="double":
            return self.graphDouble.isolatedNodes()
    
    #---------------------------------------------------------------------------
    #   Algebra
    #---------------------------------------------------------------------------

    def propagate(self, Matrix M, bool[:] clamped=None):
        """
            When clamped is None, returns the matrix product of self and M.
            When clamped is not None, it acts as a copy indicator: if clamped[i]
            is True, row i of M is copied to output. If clamped[i] is False, 
            matrix multiplication is done as usual.
            clamped must have length equal to M.rows.
        """
        if clamped is not None:
            if M.rows!=clamped.size:
                raise Exception("clamped must have length equal to Matrix rows.")
            if M.rows!=self.nodes:
                raise Exception("Number of Matrix rows must be equal to number of nodes.")
            if M.ctype!=self.ctype:
                raise TypeError("Graph and Matrix must have same ctype.")
            out = Matrix()
            if self.ctype=="int":
                out.mtxInt = self.graphInt.propagate(M.mtxInt,&clamped[0])
            if self.ctype=="float":
                out.mtxFloat = self.graphFloat.propagate(M.mtxFloat,&clamped[0])
            if self.ctype=="int":
                out.mtxDouble = self.graphDouble.propagate(M.mtxDouble,&clamped[0])
            return out
        else:
            return self&M

    def __add__(self, Graph G):
        if self.ctype!=G.ctype:
            raise TypeError("Graphs must share same ctype.")
        if self.edges!=G.edges:
            raise Exception("Graphs must have same number of edges")
        out = Graph()
        if self.ctype=="int":
            out.graphInt = self.graphInt.add(G.graphInt)
        if self.ctype=="float":
            out.graphFloat = self.graphFloat.add(G.graphFloat)
        if self.ctype=="double":
            out.graphDouble = self.graphDouble.add(G.graphDouble)
        return out
    
    def __sub__(self, Graph G):
        if self.ctype!=G.ctype:
            raise TypeError("Graphs must share same ctype.")
        if self.edges!=G.edges:
            raise Exception("Graphs must have same number of edges")
        out = Graph()
        if self.ctype=="int":
            out.graphInt = self.graphInt.sub(G.graphInt)
        if self.ctype=="float":
            out.graphFloat = self.graphFloat.sub(G.graphFloat)
        if self.ctype=="double":
            out.graphDouble = self.graphDouble.sub(G.graphDouble)
        return out
    
    def __mul__(self,value):
        if self.ctype=="int":
            out = Graph(ctype="int")
            out.graphInt = self.graphInt.smul(value)
            return out
        if self.ctype=="float":
            out = Graph(ctype="float")
            out.graphFloat = self.graphFloat.smul(value)
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.graphDouble = self.graphDouble.smul(value)
            return out
    
    def __rmul__(self,value):
        return self*value
    
    def __truediv__(self, value):
        if value==0:
            raise ValueError("Attempting division by zero.")
        return self*(1./value)
    
    def __and__(self, Matrix A):
        if self.ctype!=A.ctype:
            raise Exception("Graph-Matrix operations require same ctype.")
        if self.nodes!=A.rows:
            raise Exception("Incompatible shape for product.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt = self.graphInt.mmul(A.mtxInt)
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat = self.graphFloat.mmul(A.mtxFloat)
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble = self.graphDouble.mmul(A.mtxDouble)
            return out
    
    def __rand__(self, Matrix A):
        if self.ctype!=A.ctype:
            raise Exception("Graph-Matrix operations require same ctype.")
        if self.nodes!=A.cols:
            raise Exception("Incompatible shape for product.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt = self.graphInt.mmul(A.mtxInt)
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat = self.graphFloat.mmul(A.mtxFloat)
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble = self.graphDouble.mmul(A.mtxDouble)
            return out
    
    #---------------------------------------------------------------------------
    #   Useful stuff
    #---------------------------------------------------------------------------

    def normalize(self):
        """
            Symmetric normalization of edge weights.
        """
        if self.ctype=="int":
            self.graphInt.normalize()
        if self.ctype=="float":
            self.graphFloat.normalize()
        if self.ctype=="double":
            self.graphDouble.normalize()
    
    def degree(self):
        """
            Returns a column Matrix where each entry is the degree of the
            corresponding verterx.
        """
        out = Matrix(ctype=self.ctype)
        if self.ctype=="int":
            out.mtxInt = self.graphInt.degree()
        if self.ctype=="float":
            out.mtxFloat = self.graphFloat.degree()
        if self.ctype=="double":
            out.mtxDouble = self.graphDouble.degree()
        return out
    
    def laplacian(self):
        """
            Returns the graph corresponding to the normalized laplacian of 
            current graph.
        """
        out = Graph(ctype=self.ctype)
        if self.ctype=="int":
            out.graphInt = self.graphInt.laplacian()
        if self.ctype=="float":
            out.graphFloat = self.graphFloat.laplacian()
        if self.ctype=="double":
            out.graphDouble = self.graphDouble.laplacian()
        return out
    
    def densify(self):
        """
            Returns a Matrix corresponding to the dense representation of self.
        """
        out = Matrix(ctype=self.ctype)
        if self.ctype=="int":
            out.mtxInt = self.graphInt.densify()
        if self.ctype=="float":
            out.mtxFloat = self.graphFloat.densify()
        if self.ctype=="double":
            out.mtxDouble = self.graphDouble.densify()
        return out

#---------------------------------------------------------------------------
#   Other graph-related functions
#---------------------------------------------------------------------------

def hadamard(Graph G, Graph H):
    """
        Hadamard (element-wise) product between weights in Graphs. Both graphs 
        must have same number of edges. Structure of G will be the structure of
        the returning graph.
    """
    if H.ctype!=G.ctype:
        raise TypeError("Graphs must share same ctype.")
    if H.edges!=G.edges:
        raise Exception("Graphs must have same number of edges")
    out = Graph()
    if G.ctype=="int":
        out.graphInt = G.graphInt.hmul(H.graphInt)
    if G.ctype=="float":
        out.graphFloat = G.graphFloat.hmul(H.graphFloat)
    if G.ctype=="double":
        out.graphDouble = G.graphDouble.hmul(H.graphDouble)
    return out

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Digraph
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class Digraph(mioloObject):

    """
        A class for directed graphs. 
    """

    cdef object cType

    def __cinit__(self, unsigned long nodes=0, ctype=global_ctype):
        if ctype in ctypes:
            self.cType = ctype
            if nodes>0:
                if ctype=="int":
                    self.digraphInt = new mld.digraph[int](nodes)
                if ctype=="float":
                    self.digraphFloat = new mld.digraph[float](nodes)
                if ctype=="double":
                    self.digraphDouble = new mld.digraph[double](nodes)
            else:
                if ctype=="int":
                    self.digraphInt = NULL
                if ctype=="float":
                    self.digraphFloat = NULL
                if ctype=="double":
                    self.digraphDouble = NULL
        else:
            raise Exception("Unknown ctype.")
    
    def __dealloc__(self):
        if self.ctype=="int":
            del self.digraphInt
        if self.ctype=="float":
            del self.digraphFloat
        if self.ctype=="double":
            del self.digraphDouble
    
    @property
    def ctype(self):
        return str(self.cType)

    @property
    def nodes(self):
        if self.ctype=="int":
            return self.digraphInt.nodes
        if self.ctype=="float":
            return self.digraphFloat.nodes
        if self.ctype=="double":
            return self.digraphDouble.nodes
    
    @property
    def null(self):
        if self.ctype=="int":
            return self.digraphInt.null()
        if self.ctype=="float":
            return self.digraphFloat.null()
        if self.ctype=="double":
            return self.digraphDouble.null()

    def connect(self, unsigned long i, unsigned long j, double value=0):
        """
            Create connection from i to j weighted by value.
        """
        if i >= self.nodes or i<0:
            raise Exception("i is not a valid index for Digraph.")
        if j >= self.nodes or j<0:
            raise Exception("i is not a valid index for Digraph.")
        if self.ctype=="int":
            self.digraphInt.connect(i,j,<int>value)
        if self.ctype=="float":
            self.digraphFloat.connect(i,j,<float>value)
        if self.ctype=="double":
            self.digraphDouble.connect(i,j,value)
    
    def copy(self, mode="whole"):
        """
            Returns a copy of self. 
            if mode is whole, returns a copy of the whole Digraph. If mode is 
            structure, returns a Digraph with same structure of self, but with
            all weights set to zero.
        """
        out = Digraph(ctype=self.ctype)
        if mode=="whole":
            if self.ctype=="int":
                out.digraphInt = self.digraphInt.copy()
            if self.ctype=="float":
                out.digraphFloat = self.digraphFloat.copy()
            if self.ctype=="double":
                out.digraphDouble = self.digraphDouble.copy()
        if mode=="structure":
            if self.ctype=="int":
                out.digraphInt = self.digraphInt.copyStructure()
            if self.ctype=="float":
                out.digraphFloat = self.digraphFloat.copyStructure()
            if self.ctype=="double":
                out.digraphDouble = self.digraphDouble.copyStructure()
        return out
    
    def shape(self):
        """
            Returns the size number of elements of each row in the adjacency
            matrix.
        """
        cdef unsigned long* view 
        cdef unsigned long[:] aux
        cdef unsigned long k, n
        out = np.empty(self.nodes,dtype=np.ulong)
        aux = out
        if self.ctype=="int":
            view = self.digraphInt.shape()
            n = self.digraphInt.nodes
        if self.ctype=="float":
            view = self.digraphFloat.shape()
            n = self.digraphFloat.nodes
        if self.ctype=="double":
            view = self.digraphDouble.shape()
            n = self.digraphDouble.nodes
        for k in range(n):
            aux[k] = view[k]
        return out
    
    def transpose(self):
        out = Digraph(ctype=self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.transpose()
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.transpose()
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.transpose()
        return out
    
    def sameShape(self, Digraph G):
        cdef unsigned long k, n
        if self.nodes!=G.nodes:
            return False
        n = self.nodes
        ashape = self.shape()
        gshape = G.shape()
        for k in range(n):
            if ashape[k]!=gshape[k]:
                return False
        return True
    
    #---------------------------------------------------------------------------
    #   Algebra
    #---------------------------------------------------------------------------

    def __add__(self, Digraph G):
        if self.ctype!=G.ctype:
            raise TypeError("Digraphs must share same ctype.")
        if not self.sameShape(G):
            raise Exception("Digraphs must have same shapes.")
        out = Digraph(self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.add(drf(G.digraphInt))
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.add(drf(G.digraphFloat))
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.add(drf(G.digraphDouble))
        return out
    
    def __sub__(self, Digraph G):
        if self.ctype!=G.ctype:
            raise TypeError("Digraphs must share same ctype.")
        if not self.sameShape(G):
            raise Exception("Digraphs must have same shapes.")
        out = Digraph(self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.sub(drf(G.digraphInt))
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.sub(drf(G.digraphFloat))
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.sub(drf(G.digraphDouble))
        return out
    
    def __mul__(self, value):
        out = Digraph(self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.smul(value)
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.smul(value)
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.smul(value)
        return out
    
    def __and__(self, Matrix M):
        if self.nodes!=Matrix.rows:
            raise Exception("Incompatible shape for matrix multiplication.")
        if self.ctype!=M.ctype:
            raise TypeError("Digraph and Matrix must share same ctype.")
        out = Matrix(self.ctype)
        if self.ctype=="int":
            out.mtxInt = self.digraphInt.mmul(drf(M.mtxInt))
        if self.ctype=="float":
            out.mtxFloat = self.digraphFloat.mmul(drf(M.mtxFloat))
        if self.ctype=="double":
            out.mtxDouble = self.digraphDouble.mmul(drf(M.mtxDouble))
        return out

    def __rand__(self, Matrix M):
        if self.nodes!=Matrix.cols:
            raise Exception("Incompatible shape for matrix multiplication.")
        if self.ctype!=M.ctype:
            raise TypeError("Digraph and Matrix must share same ctype.")
        out = Matrix(self.ctype)
        if self.ctype=="int":
            out.mtxInt = self.digraphInt.mmul(drf(M.mtxInt))
        if self.ctype=="float":
            out.mtxFloat = self.digraphFloat.mmul(drf(M.mtxFloat))
        if self.ctype=="double":
            out.mtxDouble = self.digraphDouble.mmul(drf(M.mtxDouble))
        return out

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Diagonal matrices
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class Diagonal(mioloObject):

    """
        A class for dense matrices. 
        Initializes a C++ object that stores a diagonal matrix.
        dim: dimension of square matrix (number of rows is equal to cols)
        Elements of can be acessed via the __getitem__ method, ie Diagonal[k]
    """

    cdef object cType

    def __cinit__(self, unsigned long dim=0, init=0, ctype=global_ctype):
        if ctype in ctypes:
            self.cType = ctype
            if dim>0:
                if ctype=="int":
                    self.diagonalInt = new mld.diagonal[int](dim,init)
                if ctype=="float":
                    self.diagonalFloat = new mld.diagonal[float](dim,init)
                if ctype=="double":
                    self.diagonalDouble = new mld.diagonal[double](dim,init)
            else:
                if ctype=="int":
                    self.mtxInt = NULL
                if ctype=="float":
                    self.mtxFloat = NULL
                if ctype=="double":
                    self.mtxDouble = NULL
        else:
            raise Exception("Unknown ctype.")
    
    def __dealloc__(self):
        if self.ctype=="int":
            del self.diagonalInt
        if self.ctype=="float":
            del self.diagonalFloat
        if self.ctype=="double":
            del self.diagonalDouble

    @property
    def ctype(self):
        return str(self.cType)
    
    @property
    def rows(self):
        if self.ctype=="int":
            return self.diagonalInt.dim
        if self.ctype=="float":
            return self.diagonalFloat.dim
        if self.ctype=="double":
            return self.diagonalDouble.dim
    
    @property
    def cols(self):
        return self.rows

    def __len__(self):
        return self.rows

    def __getitem__(self, unsigned long k):
        if k >= <unsigned long>self.rows:
            raise Exception("Index k is out of bounds.")
        if self.ctype=="int":
            return self.diagonalInt.data[k]
        if self.ctype=="float":
            return self.diagonalFloat.data[k]
        if self.ctype=="double":
            return self.diagonalDouble.data[k]
    
    def __setitem__(self, unsigned long k, value):
        if k >= <unsigned long>self.rows*self.cols:
            raise Exception("Index k is out of bounds.")
        if self.ctype=="int":
            self.diagonalInt.data[k] = <int>value
        if self.ctype=="float":
            self.diagonalFloat.data[k] = <float>value
        if self.ctype=="double":
            self.diagonalDouble.data[k] = value

    #---------------------------------------------------------------------------
    #   Algebra
    #---------------------------------------------------------------------------

    def __add__(self, Diagonal D):
        if len(self)!=len(D):
            raise Exception("Incompatible shapes for addition")
        if self.ctype!=D.ctype:
            raise TypeError("Diagonals must have same ctype")
        out = Diagonal(ctype=D.ctype)
        if self.ctype=="int":
            out.diagonalInt = self.diagonalInt.add(drf(D.diagonalInt))
        if self.ctype=="float":
            out.diagonalFloat = self.diagonalFloat.add(drf(D.diagonalFloat))
        if self.ctype=="double":
            out.diagonalDouble = self.diagonalDouble.add(drf(D.diagonalDouble))
        return out
    
    def __sub__(self, Diagonal D):
        if len(self)!=len(D):
            raise Exception("Incompatible shapes for subtraction")
        if self.ctype!=D.ctype:
            raise TypeError("Diagonals must have same ctype")
        out = Diagonal(ctype=D.ctype)
        if self.ctype=="int":
            out.diagonalInt = self.diagonalInt.sub(drf(D.diagonalInt))
        if self.ctype=="float":
            out.diagonalFloat = self.diagonalFloat.sub(drf(D.diagonalFloat))
        if self.ctype=="double":
            out.diagonalDouble = self.diagonalDouble.sub(drf(D.diagonalDouble))
        return out

    def __and__(self, mioloObject D):
        if D.ctype!=self.ctype:
            raise TypeError("Objects must share same ctype.")
        if isinstance(D,Matrix):
            if (D.rows!=self.dim):
                raise Exception("Number of rows must be equal to Diagonal.cols")
            out = Matrix(self.ctype)
            if self.ctype=="int":
                out.mtxInt = self.diagonalInt.lmul(drf(D.mtxInt))
            if self.ctype=="float":
                out.mtxFloat = self.diagonalFloat.lmul(drf(D.mtxFloat))
            if self.ctype=="double":
                out.mtxDouble = self.diagonalDouble.lmul(drf(D.mtxDouble))
            return out
        if isinstance(D,Graph):
            if (D.nodes!=self.dim):
                raise Exception("Number of nodes must be equal to Diagonal.cols")
            out = Digraph(self.ctype)
            if self.ctype=="int":
                out.digraphInt = self.diagonalInt.lmul(drf(D.graphInt))
            if self.ctype=="float":
                out.digraphFloat = self.diagonalFloat.lmul(drf(D.graphFloat))
            if self.ctype=="double":
                out.digraphDouble = self.diagonalDouble.lmul(drf(D.graphDouble))
            return out
        if isinstance(D,Digraph):
            if (D.nodes!=self.dim):
                raise Exception("Number of nodes must be equal to Diagonal.cols")
            out = Digraph(self.ctype)
            if self.ctype=="int":
                out.digraphInt = self.diagonalInt.lmul(drf(D.digraphInt))
            if self.ctype=="float":
                out.digraphFloat = self.diagonalFloat.lmul(drf(D.digraphFloat))
            if self.ctype=="double":
                out.digraphDouble = self.diagonalDouble.lmul(drf(D.digraphDouble))
            return out
        if isinstance(D,Diagonal):
            if (D.dim!=self.dim):
                raise Exception("Dimension of diagonals must be equal.")
            out = Diagonal(self.ctype)
            if self.ctype=="int":
                out.diagonalInt = self.diagonalInt.mul(drf(D.diagonalInt))
            if self.ctype=="float":
                out.diagonalFloat = self.diagonalFloat.mul(drf(D.diagonalFloat))
            if self.ctype=="double":
                out.diagonalDouble = self.diagonalDouble.mul(drf(D.diagonalDouble))
            return out
    
    def __rand__(self, mioloObject D):
        if D.ctype!=self.ctype:
            raise TypeError("Objects must share same ctype.")
        if isinstance(D,Matrix):
            if (D.cols!=self.dim):
                raise Exception("Number of rows must be equal to Diagonal.cols")
            out = Matrix(self.ctype)
            if self.ctype=="int":
                out.mtxInt = self.diagonalInt.rmul(drf(D.mtxInt))
            if self.ctype=="float":
                out.mtxFloat = self.diagonalFloat.rmul(drf(D.mtxFloat))
            if self.ctype=="double":
                out.mtxDouble = self.diagonalDouble.rmul(drf(D.mtxDouble))
            return out
        if isinstance(D,Graph):
            if (D.nodes!=self.dim):
                raise Exception("Number of nodes must be equal to Diagonal.cols")
            out = Digraph(self.ctype)
            if self.ctype=="int":
                out.digraphInt = self.diagonalInt.rmul(drf(D.graphInt))
            if self.ctype=="float":
                out.digraphFloat = self.diagonalFloat.rmul(drf(D.graphFloat))
            if self.ctype=="double":
                out.digraphDouble = self.diagonalDouble.rmul(drf(D.graphDouble))
            return out
        if isinstance(D,Digraph):
            if (D.nodes!=self.dim):
                raise Exception("Number of nodes must be equal to Diagonal.cols")
            out = Digraph(self.ctype)
            if self.ctype=="int":
                out.digraphInt = self.diagonalInt.rmul(drf(D.digraphInt))
            if self.ctype=="float":
                out.digraphFloat = self.diagonalFloat.rmul(drf(D.digraphFloat))
            if self.ctype=="double":
                out.digraphDouble = self.diagonalDouble.rmul(drf(D.digraphDouble))
            return out
        if isinstance(D,Diagonal):
            if (D.dim!=self.dim):
                raise Exception("Dimension of diagonals must be equal.")
            out = Diagonal(self.ctype)
            if self.ctype=="int":
                out.diagonalInt = self.diagonalInt.mul(drf(D.diagonalInt))
            if self.ctype=="float":
                out.diagonalFloat = self.diagonalFloat.mul(drf(D.diagonalFloat))
            if self.ctype=="double":
                out.diagonalDouble = self.diagonalDouble.mul(drf(D.diagonalDouble))
            return out

    def __mul__(self, value):
        out = Diagonal(ctype=self.ctype)
        if self.ctype=="int":
            out.diagonalInt = self.diagonalInt.smul(value)
        if self.ctype=="float":
            out.diagonalFloat = self.diagonalFloat.smul(value)
        if self.ctype=="double":
            out.diagonalDouble = self.diagonalDouble.smul(value)
        return out
    
    def __truediv__(self, value):
        if (value==0):
            raise ValueError("Cannot divide by zero.")
        return self*(1/value)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Input Matrix, Graph or Digraph from txt
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def txtMatrix(filename,ctype=global_ctype):
    m = np.loadtxt(filename)
    out = Matrix(m.shape[0],m.shape[1],0,ctype)
    out.numpy = m
    return out

def txtGraph(filename,ctype=global_ctype):
    g = np.transpose(np.loadtxt(filename))
    max_0 = max(g[0])
    max_1 = max(g[1])
    N = max([max_0,max_1])+1
    E = len(g[0])
    out = Graph(N,E,ctype)
    out.structure = np.transpose(g[0:2])
    out.weights = g[2]
    return out

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Manifolds
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class Sphere:
    """
        This class treats each row of a Matrix as a unit sphere. Therefore, it
        views a Matrix as the product manifold of Matrix.rows unit spheres 
        embedded in the euclidean space of dimension Matrix.cols.
    """

    cdef mld.sphere[int] sphereInt
    cdef mld.sphere[float] sphereFloat
    cdef mld.sphere[double] sphereDouble

    def __init__(self, double radius=1):
        self.radius = radius 
    
    @property
    def radius(self):
        return self.sphereDouble.r
    @radius.setter
    def radius(self, double value):
        if value<=0:
            raise ValueError("WTF radius is less than or equal to zero.")
        self.sphereInt.r = <int>value
        self.sphereFloat.r = <float>value
        self.sphereDouble.r = value
    
    def stereographicProjection(self, Matrix M):
        """
            Returns the matrix for which each row is the stereographic 
            projection of the corresponding row in M over the unit sphere.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.sphereInt.stereographicProjection(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.sphereFloat.stereographicProjection(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.sphereDouble.stereographicProjection(drf(M.mtxDouble))
        return out
    
    def tangentProjection(self, Matrix at, Matrix M):
        """
            Returns the matrix for which each row r is the projection of the 
            r-th row in M onto the tangent space of the r-th row of at.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.sphereInt.tangentProjection(drf(at.mtxInt),drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.sphereFloat.tangentProjection(drf(at.mtxFloat),drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.sphereDouble.tangentProjection(drf(at.mtxDouble),drf(M.mtxDouble))
        return out
    
    def fromEuclidean(self, Matrix M):
        """
            Returns the Matrix for which each row r has M.cols+1 columns and is
            the mapping of the r row of M to the sphere embedded in the euclidean
            space of dimension M.cols+1. 
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.sphereInt.fromEuclidean(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.sphereFloat.fromEuclidean(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.sphereDouble.fromEuclidean(drf(M.mtxDouble))
        return out
    
    def toEuclidean(self, Matrix M):
        """
            Returns the Matrix for which each row r has M.cols-1 columns and is
            the mapping of the r row of M to the sphere embedded in the euclidean
            space of dimension M.cols+1.  
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.sphereInt.toEuclidean(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.sphereFloat.toEuclidean(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.sphereDouble.toEuclidean(drf(M.mtxDouble))
        return out
    
    def distance(self, Matrix M):
        """
            Return a square Matrix for which each entry is the geodesic distance
            between the pair of (i,j) rows of M.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.sphereInt.distance(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.sphereFloat.distance(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.sphereDouble.distance(drf(M.mtxDouble))
        return out
    
    def isIn(self, Matrix M, double tolerance=0.001):
        if M.ctype=="int":
            return self.sphereInt.isIn(drf(M.mtxInt),<int>tolerance)
        if M.ctype=="float":
            return self.sphereFloat.isIn(drf(M.mtxFloat),<float>tolerance)
        if M.ctype=="double":
            return self.sphereDouble.isIn(drf(M.mtxDouble),tolerance)
    
    def isTangent(self, Matrix at, Matrix M, double tolerance=0.001):
        if at.rows!=M.rows or at.cols!=M.cols:
            raise Exception("at and M must have same shape.")
        if at.ctype!=M.ctype:
            raise TypeError("at and M must have same ctype.")
        if M.ctype=="int":
            return self.sphereInt.isTangent(drf(at.mtxInt),drf(M.mtxInt),<int>tolerance)
        if M.ctype=="float":
            return self.sphereFloat.isTangent(drf(at.mtxFloat),drf(M.mtxFloat),<float>tolerance)
        if M.ctype=="double":
            return self.sphereDouble.isTangent(drf(at.mtxDouble),drf(M.mtxDouble),tolerance)

cdef class Simplex:
    """
        This class treats each row of a Matrix as a simplex. Therefore, it
        views a Matrix as the product manifold of Matrix.rows unit simplexes 
        embedded in the euclidean space of dimension Matrix.cols.
    """

    cdef mld.simplex sp
    
    def softmaxRetraction(self, Matrix at, Matrix M):
        """
            A retraction based on the softmax function.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.sp.softmaxRetraction(drf(at.mtxInt),drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.sp.softmaxRetraction(drf(at.mtxFloat),drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.sp.softmaxRetraction(drf(at.mtxDouble),drf(M.mtxDouble))
        return out
    
    def tangentProjection(self, Matrix at, Matrix M):
        """
            Returns the matrix for which each row r is the projection of the 
            r-th row in M onto the tangent space of the r-th row of at.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.sp.tangentProjection(drf(at.mtxInt),drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.sp.tangentProjection(drf(at.mtxFloat),drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.sp.tangentProjection(drf(at.mtxDouble),drf(M.mtxDouble))
        return out
    
    def isIn(self, Matrix M, double tolerance=0.001):
        if M.ctype=="int":
            return self.sp.isIn(drf(M.mtxInt),<int>tolerance)
        if M.ctype=="float":
            return self.sp.isIn(drf(M.mtxFloat),<float>tolerance)
        if M.ctype=="double":
            return self.sp.isIn(drf(M.mtxDouble),tolerance)
    
    def isTangent(self, Matrix at, Matrix M, double tolerance=0.001):
        if at.rows!=M.rows or at.cols!=M.cols:
            raise Exception("at and M must have same shape.")
        if at.ctype!=M.ctype:
            raise TypeError("at and M must have same ctype.")
        if M.ctype=="int":
            return self.sp.isTangent(drf(at.mtxInt),drf(M.mtxInt),<int>tolerance)
        if M.ctype=="float":
            return self.sp.isTangent(drf(at.mtxFloat),drf(M.mtxFloat),<float>tolerance)
        if M.ctype=="double":
            return self.sp.isTangent(drf(at.mtxDouble),drf(M.mtxDouble),tolerance)