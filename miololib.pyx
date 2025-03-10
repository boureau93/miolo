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
        A container object for four objects in the miolo library. This is usually
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
            Returns true if object is miolo.Diagonal
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

        Operators +-*/ are defined between matrices and scalars to replicate 
        vector space operations. Matrix multiplication is done via the __and__
        and __rand__ operators (&). Absolute value operator __abs__ is also
        overloaded and returns the max absolute value of elements of matrix.
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
    def null(self):
        if not self.isMatrix():
            return True
        else:
            if self.mtxInt is not NULL and self.mtxInt.null():
                return True
            if self.mtxFloat is not NULL and self.mtxFloat.null():
                return True
            if self.mtxDouble is not NULL and self.mtxDouble.null():
                return True
            return False

    @property
    def ctype(self):
        return str(self.cType)
    
    @property
    def rows(self):
        if self.null:
            pass
        if self.ctype=="int":
            return self.mtxInt.rows
        if self.ctype=="float":
            return self.mtxFloat.rows
        if self.ctype=="double":
            return self.mtxDouble.rows
    
    @property
    def cols(self):
        if self.null:
            pass
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
                    self.mtxInt.data[i*self.mtxInt.cols+j] = <int>data[i][j]
        if self.ctype == "float":
            for i in range(self.mtxFloat.rows):
                for j in range(self.mtxFloat.cols):
                    self.mtxFloat.data[i*self.mtxFloat.cols+j] = <float>data[i][j]
        if self.ctype == "double":
            for i in range(self.mtxDouble.rows):
                for j in range(self.mtxDouble.cols):
                    self.mtxDouble.data[i*self.mtxDouble.cols+j] = <double>data[i][j]

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
    
    def copy(self, Matrix M, unsigned long[:] only=None):
        """
            Copies M into self if both have same shape.
            only: if not None, copies only rows in only.
        """
        cdef unsigned long n
        if M.rows!=self.rows or M.cols!=self.cols:
            raise Exception("Matrices must have same shape.")
        if self.ctype!=M.ctype:
            raise TypeError("Matrices must share same ctype.")
        if only is None:
            if self.ctype=="int":
                self.mtxInt.copy(drf(M.mtxInt))
            if self.ctype=="float":
                self.mtxFloat.copy(drf(M.mtxFloat))
            if self.ctype=="double":
                self.mtxDouble.copy(drf(M.mtxDouble))
        else:
            if np.max(only)+1>self.rows:
                raise Exception("Entries of only exceed self.rows.")
            n = only.size
            if self.ctype=="int":
                self.mtxInt.copy(drf(M.mtxInt),&only[0],n)
            if self.ctype=="float":
                self.mtxFloat.copy(drf(M.mtxFloat),&only[0],n)
            if self.ctype=="double":
                self.mtxDouble.copy(drf(M.mtxDouble),&only[0],n)
    
    def print(self):
        if self.ctype=="int":
            self.mtxInt.print()
        if self.ctype=="float":
            self.mtxFloat.print()
        if self.ctype=="double":
            self.mtxDouble.print()
    
    def cut(self, rowMin, rowMax, colMin,colMax):
        """
            Cuts the window [rowMin,rowMax) x [colMin,colMax) from self and 
            returns a new Matrix.
        """
        if rowMin<0 or rowMax>self.rows:
            raise Exception("Row range is out of bounds.")
        if colMin<0 or colMax>self.cols:
            raise Exception("Col range is out of bounds.")
        out = Matrix(ctype=self.ctype)
        if self.ctype=="int":
            out.mtxInt = self.mtxInt.cut(rowMin,rowMax,colMin,colMax)
        if self.ctype=="float":
            out.mtxFloat = self.mtxFloat.cut(rowMin,rowMax,colMin,colMax)
        if self.ctype=="double":
            out.mtxDouble = self.mtxDouble.cut(rowMin,rowMax,colMin,colMax)
        return out
        
    #---------------------------------------------------------------------------
    #   Other useful stuff
    #---------------------------------------------------------------------------

    def max(self):
        """
            Returns max value in matrix.
        """
        if self.ctype=="int":
            return self.mtxInt.max()
        if self.ctype=="float":
            return self.mtxFloat.max()
        if self.ctype=="double":
            return self.mtxDouble.max()

    def min(self):
        """
            Returns min value in matrix.
        """
        if self.ctype=="int":
            return self.mtxInt.min()
        if self.ctype=="float":
            return self.mtxFloat.min()
        if self.ctype=="double":
            return self.mtxDouble.min()
    
    def argkmax(self, k=1):
        """
            Returns index of k largest values in each row.
        """
        out = Matrix(ctype="int")
        if self.ctype=="int":
            out.mtxInt = mld.argkmax(drf(self.mtxInt),k)
        if self.ctype=="float":
            out.mtxInt = mld.argkmax(drf(self.mtxFloat),k)
        if self.ctype=="double":
            out.mtxInt = mld.argkmax(drf(self.mtxDouble),k)
        return out
    
    def argkmin(self, k=1):
        """
            Returns index of k smallest values in each row.
        """
        out = Matrix(ctype="int")
        if self.ctype=="int":
            out.mtxInt = mld.argkmin(drf(self.mtxInt),k)
        if self.ctype=="float":
            out.mtxInt = mld.argkmin(drf(self.mtxFloat),k)
        if self.ctype=="double":
            out.mtxInt = mld.argkmin(drf(self.mtxDouble),k)
        return out

    def normalize(self):
        """
            Row normalization in to make elements in the same row sum to 1.
            This operation is done inplace.
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
        return out 
    

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
            NOTE: This only changes a 'view' on the Matrix data. No changes are
            made on stored data.
        """
        if self.rows*self.cols!=rows*cols:
            raise Exception("Invalid new shape.")
        if self.ctype=="int":
            self.mtxInt.reshape(rows,cols)
        if self.ctype=="float":
            self.mtxFloat.reshape(rows,cols)
        if self.ctype=="int":
            self.mtxDouble.reshape(rows,cols)
    
    def sparsifyThreshold(self, Threshold, output="Graph"):
        """
            If self is a square matrix, returns a Graph with only the entries
            that are smaller than Threshold. 
            Args:
                Threshold: lower bound on entries.
                output: can return Graph or Digraph.
            Returns:
                Graph, Digraph or None in case output is unknown.
        """
        if self.rows!=self.cols:
            raise Exception("Only square matrices can be sparsified.")
        if output=="Graph":
            out = Graph(ctype=self.ctype)
            if out.ctype=="int":
                out.graphInt = mld.sparsifyGraphThreshold(drf(self.mtxInt),Threshold)
            if out.ctype=="float":
                out.graphFloat = mld.sparsifyGraphThreshold(drf(self.mtxFloat),Threshold)
            if out.ctype=="double":
                out.graphDouble = mld.sparsifyGraphThreshold(drf(self.mtxDouble),Threshold)
            return out
        if output=="Digraph":
            out = Digraph(ctype=self.ctype)
            if out.ctype=="int":
                out.digraphInt = mld.sparsifyDigraphThreshold(drf(self.mtxInt),Threshold)
            if out.ctype=="float":
                out.digraphFloat = mld.sparsifyDigraphThreshold(drf(self.mtxFloat),Threshold)
            if out.ctype=="double":
                out.digraphDouble = mld.sparsifyDigraphThreshold(drf(self.mtxDouble),Threshold)
            return out
    
    def sparsifyKNN(self, k, output="Digraph"):
        if self.rows!=self.cols:
            raise Exception("Only square matrices can be sparsified.")
        out = Digraph(ctype=self.ctype)
        if out.ctype=="int":
            out.digraphInt = mld.sparsifyDigraphKNN(drf(self.mtxInt),k)
        if out.ctype=="float":
            out.digraphFloat = mld.sparsifyDigraphKNN(drf(self.mtxFloat),k)
        if out.ctype=="double":
            out.digraphDouble = mld.sparsifyDigraphKNN(drf(self.mtxDouble),k)
        if output=="Digraph":
            return out
        else:
            return out.toGraph()

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
            out.mtxInt = self.mtxInt.add(A.mtxInt)
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat = self.mtxFloat.add(A.mtxFloat)
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble = self.mtxDouble.add(A.mtxDouble)
            return out
    
    def __sub__(self, Matrix A):
        if self.ctype!=A.ctype:
            raise Exception("Matrix operations require same ctype.")
        if self.rows!=A.rows or self.cols!=A.cols:
            raise Exception("Incompatible shape for Matrix subtraction.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt = self.mtxInt.sub(A.mtxInt)
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat = self.mtxFloat.sub(A.mtxFloat)
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble = self.mtxDouble.sub(A.mtxDouble)
            return out
    
    def __mul__(self, value):
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt = self.mtxInt.smul(value)
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat = self.mtxFloat.smul(value)
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble = self.mtxDouble.smul(value)
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
            out.mtxInt = self.mtxInt.mmul(A.mtxInt)
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat = self.mtxFloat.mmul(A.mtxFloat)
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble = self.mtxDouble.mmul(A.mtxDouble)
            return out

    def __mod__(self, Matrix A):
        """
            Hadamard (element-wise) product.
        """
        if self.ctype!=A.ctype:
            raise Exception("Matrix operations require same ctype.")
        if self.rows!=A.rows or self.cols!=A.cols:
            raise Exception("Incompatible shape for Hadamard product.")
        if self.ctype=="int":
            out = Matrix(ctype="int")
            out.mtxInt = self.mtxInt.hmul(A.mtxInt)
            return out
        if self.ctype=="float":
            out = Matrix(ctype="float")
            out.mtxFloat = (self.mtxFloat.hmul(A.mtxFloat))
            return out
        if self.ctype=="double":
            out = Matrix(ctype="double")
            out.mtxDouble = (self.mtxDouble.hmul(A.mtxDouble))
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
        """
            Returns the argmax of each row.
        """
        out = Matrix(ctype="int")
        if self.ctype=="int":
            out.mtxInt = mld.argmax(self.mtxInt)
        if self.ctype=="float":
            out.mtxInt = mld.argmax(self.mtxFloat)
        if self.ctype=="double":
            out.mtxInt = mld.argmax(self.mtxDouble)
        return out
    
    def argmin(self):
        """
            Returns the argmin of each row.
        """
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
    
    def partition(self, int[:] labels, int targetLabel):
        """
            Returns the rows of self with label equal to target. Useful for 
            implementing k-means.
            @params:
                labels: array of labels. Must have length equal to self.rows.
                targetLabel: targeted label for extraction.
        """
        if labels.size!=self.rows:
            raise Exception("labels must have size equal to self.rows.")
        out = Matrix(ctype=self.ctype)
        if out.ctype=="int":
            out.mtxInt = self.mtxInt.partition(&labels[0],targetLabel)
        if out.ctype=="float":
            out.mtxFloat = self.mtxFloat.partition(&labels[0],targetLabel)
        if out.ctype=="double":
            out.mtxDouble = self.mtxDouble.partition(&labels[0],targetLabel)
        return out
    
    def trace(self):
        """
            Sum of the elements with row index equal to column index. This is
            a generalization of trace operation to non-square matrices.
        """
        if self.ctype=="int":
            return self.mtxInt.trace()
        if self.ctype=="float":
            return self.mtxFloat.trace()
        if self.ctype=="double":
            return self.mtxDouble.trace()

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

def concat(Matrix A, Matrix B):
    """
        Concatenates A and B if both have same number of columns and share same
        ctype. Rowise concatenation is performed.
    """
    if A.ctype!=B.ctype:
        raise TypeError("A and B must have same ctype.")
    if A.cols!=B.cols:
        raise Exception("A and B must have same number of columns.")
    if A.null:
        return B
    if B.null:
        return A
    out = Matrix(ctype=A.ctype)
    if out.ctype=="int":
        out.mtxInt = mld.concat(drf(A.mtxInt),drf(B.mtxInt))
    if out.ctype=="float":
        out.mtxFloat = mld.concat(drf(A.mtxFloat),drf(B.mtxFloat))
    if out.ctype=="double":
        out.mtxDouble = mld.concat(drf(A.mtxDouble),drf(B.mtxDouble))
    return out

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

        Vector space operators +- are defined for graphs with same number of 
        edges. Multiplication and division by scalar is defined.
        Matrix multiplication can be done with objects of type Matrix, returning
        another matrix.
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
    def null(self):
        if self.ctype=="int":
            return self.graphInt.null()
        if self.ctype=="float":
            return self.graphFloat.null()
        if self.ctype=="double":
            return self.graphDouble.null()

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
            out = np.empty(self.edges,dtype=int)
            for k in range(self.graphInt.edges):
                out[k] = self.graphInt.e[k].w
        if self.ctype == "float":
            out = np.empty(self.edges,dtype=np.single)
            for k in range(self.graphFloat.edges):
                out[k] = self.graphFloat.e[k].w
        if self.ctype == "double":
            out = np.empty(self.edges,dtype=np.double)
            for k in range(self.graphDouble.edges):
                out[k] = self.graphDouble.e[k].w
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
    
    def print(self):
        if self.ctype=="int":
            return self.graphInt.print()
        if self.ctype=="float":
            return self.graphFloat.print()
        if self.ctype=="double":
            return self.graphDouble.print()
    
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
                out.mtxInt = self.graphInt.propagate(drf(M.mtxInt),&clamped[0])
            if self.ctype=="float":
                out.mtxFloat = self.graphFloat.propagate(drf(M.mtxFloat),&clamped[0])
            if self.ctype=="int":
                out.mtxDouble = self.graphDouble.propagate(drf(M.mtxDouble),&clamped[0])
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
    
    def toMatrix(self):
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
    
    def toDigraph(self):
        """
            Returns a Digraph representation of self.
        """
        out = Digraph(ctype=self.ctype)
        if self.ctype=="int":
            out.digraphInt = mld.toDigraph(drf(self.graphInt))
        if self.ctype=="float":
            out.digraphFloat = mld.toDigraph(drf(self.graphFloat))
        if self.ctype=="double":
            out.digraphDouble = mld.toDigraph(drf(self.graphDouble))
        return out

    def max(self):
        if self.ctype=="int":
            return self.graphInt.max()
        if self.ctype=="float":
            return self.graphFloat.max()
        if self.ctype=="double":
            return self.graphDouble.max()
    
    def min(self):
        if self.ctype=="int":
            return self.graphInt.min()
        if self.ctype=="float":
            return self.graphFloat.min()
        if self.ctype=="double":
            return self.graphDouble.min()
    
    #---------------------------------------------------------------------------
    #   Similarity calculation
    #---------------------------------------------------------------------------

    def gaussianScale(self):
        """
            Returns a Graph corresponding to the gaussian similarity of self.
            Entries of self are treated as distances and the scale parameter is
            calculated according to Jebara and Wang paper.
        """
        out = Graph(ctype=self.ctype)
        if out.ctype=="int":
            return self.graphInt.gaussianScale()
        if out.ctype=="float":
            return self.graphFloat.gaussianScale()
        if out.ctype=="double":
            return self.graphDouble.gaussianScale()
        return out
    
    #---------------------------------------------------------------------------
    #   toDigraph
    #---------------------------------------------------------------------------

    def toDigraph(self):
        out = Digraph(ctype=self.ctype)
        if out.ctype=="int":
            out.digraphInt = mld.toDigraph(drf(self.graphInt))
        if out.ctype=="float":
            out.digraphFloat = mld.toDigraph(drf(self.graphFloat))
        if out.ctype=="double":
            out.digraphDouble = mld.toDigraph(drf(self.graphDouble))
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

def MST(mioloObject O, symmetrize=True):
    """
        Returns a minimum spanning tree using Kruskal's Algorithm.
        symmetrize: if O is a Matrix, symmetrize in-place.
    """
    if isinstance(O,Matrix):
        if O.rows!=O.cols:
            raise Exception("Matrix must be square.")
        if symmetrize:
            O = (O+O.transpose())/2
        out = Graph(ctype=O.ctype)
        if out.ctype=="int":
            out.graphInt = mld.mst(drf(O.mtxInt))
        if out.ctype=="float":
            out.graphFloat = mld.mst(drf(O.mtxFloat))
        if out.ctype=="int":
            out.graphDouble = mld.mst(drf(O.mtxDouble))
        return out
    if isinstance(O,Graph):
        out = Graph(ctype=O.ctype)
        if out.ctype=="int":
            out.graphInt = mld.mst(drf(O.graphInt))
        if out.ctype=="float":
            out.graphFloat = mld.mst(drf(O.graphFloat))
        if out.ctype=="int":
            out.graphDouble = mld.mst(drf(O.graphDouble))
        return out

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Digraph
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class Digraph(mioloObject):

    """
        A class for directed graphs.

        nodes: number of nodes in graph
        ctype: underlying ctype. 

        Vector space operators +- are defined for graphs with same number of 
        edges. Multiplication and division by scalar is defined.
        Matrix multiplication can be done with objects of type Matrix, returning
        another matrix.
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

    def connect(self, i, j, value=0):
        """
            Create connection from i to j weighted by value.
        """
        if i >= self.nodes or i<0:
            raise Exception("i is not a valid index for Digraph.")
        if j >= self.nodes or j<0:
            raise Exception("j is not a valid index for Digraph.")
        if self.ctype=="int":
            self.digraphInt.connect(i,j,value)
        if self.ctype=="float":
            self.digraphFloat.connect(i,j,value)
        if self.ctype=="double":
            self.digraphDouble.connect(i,j,value)
    
    def connectRegular(self, Matrix Neighbors, Matrix Weights):
        """
            Connects arcs in self using Neighbors as topology and Weights as 
            weights.
            Neighbors: the i-th row corresponds to the neighbors of i-th node.
            Weights: square matrix for weighting.
        """
        if Neighbors.ctype!="int":
            raise TypeError("Neighbors must have int ctype.")
        if Neighbors.rows!=self.nodes or Weights.rows!=self.nodes:
            raise Exception("Neighbors and Weights must have rows equals to self.nodes.")
        if Weights.ctype!=self.ctype:
            raise TypeError("Weights and self must have same ctype.")
        if Weights.rows!=Weights.cols:
            raise Exception("Weights must be a square Matrix.")
        if self.ctype=="int":
            self.digraphInt.connectRegular(drf(Neighbors.mtxInt),drf(Weights.mtxInt))
        if self.ctype=="float":
            self.digraphFloat.connectRegular(drf(Neighbors.mtxInt),drf(Weights.mtxFloat))
        if self.ctype=="double":
            self.digraphDouble.connectRegular(drf(Neighbors.mtxInt),drf(Weights.mtxDouble))
    
    def normalize(self):
        """
            Normalize weights of each node in order to sum to 1.
        """
        if self.ctype=="int":
            self.digraphInt.normalize()
        if self.ctype=="float":
            self.digraphFloat.normalize()
        if self.ctype=="double":
            self.digraphDouble.normalize()
    
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
            Returns the number of nonzero elements of each row in the adjacency
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
        """
            Matrix transposition of Digraphs.
        """
        out = Digraph(ctype=self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.transpose()
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.transpose()
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.transpose()
        return out
    
    def sameShape(self, Digraph G):
        """
            Checks if both Digraphs have the same shapes.
        """
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
    
    def symmetrize(self):
        """
            Returns a Digraph with the symmetrization of self.
        """
        out = Digraph(ctype=self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.symmetrize()
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.symmetrize()
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.symmetrize()
        return out
    
    def gaussianScale(self):
        """
            Returns the scale parameter for gaussian similarity.
        """
        if self.ctype=="int":
            return self.digraphInt.gaussianScale()
        if self.ctype=="float":
            return self.digraphFloat.gaussianScale()
        if self.ctype=="double":
            return self.digraphDouble.gaussianScale()
            
    #---------------------------------------------------------------------------
    #   Algebra
    #---------------------------------------------------------------------------

    def __add__(self, Digraph G):
        if self.ctype!=G.ctype:
            raise TypeError("Digraphs must share same ctype.")
        if not self.sameShape(G):
            raise Exception("Digraphs must have same shapes.")
        out = Digraph(ctype=self.ctype)
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
        out = Digraph(ctype=self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.sub(drf(G.digraphInt))
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.sub(drf(G.digraphFloat))
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.sub(drf(G.digraphDouble))
        return out
    
    def __mul__(self, value):
        out = Digraph(ctype=self.ctype)
        if self.ctype=="int":
            out.digraphInt = self.digraphInt.smul(value)
        if self.ctype=="float":
            out.digraphFloat = self.digraphFloat.smul(value)
        if self.ctype=="double":
            out.digraphDouble = self.digraphDouble.smul(value)
        return out
    
    def __truediv__(self, value):
        if value==0:
            raise Exception("Avoiding division by zero.")
        return self*(1/value)
    
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

    #---------------------------------------------------------------------------
    #   toGraph
    #---------------------------------------------------------------------------

    def toGraph(self, symmetrize=True):
        if symmetrize:
            self = self.symmetrize()
        out = Graph(ctype=self.ctype)
        if out.ctype=="int":
            out.graphInt = mld.toGraph(drf(self.digraphInt))
        if out.ctype=="float":
            out.graphFloat = mld.toGraph(drf(self.digraphFloat))
        if out.ctype=="double":
            out.graphDouble = mld.toGraph(drf(self.digraphDouble))
        return out
    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Diagonal matrices
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class Diagonal(mioloObject):

    """
        A class for diagonal matrices. 
        Initializes a C++ object that stores a diagonal matrix.
        dim: dimension of square matrix (number of rows is equal to cols)
        Elements can be acessed via the __getitem__ method, ie Diagonal[k]

        Vector space +- are defined. Scalar multiplication and division are also
        defined via */ operator. Matrix multiplication can be done for Matrix,
        Graph and Digraph objects via the & operator.
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
    """
        Loads Matrix from txt file. Uses numpy.
    """
    m = np.loadtxt(filename)
    out = Matrix(m.shape[0],m.shape[1],0,ctype)
    out.numpy = m
    return out

def txtGraph(filename,ctype=global_ctype):
    """
        Loads Graph from txt file. Uses numpy.
        It is expected a file consisting of rows of triplets (i,j,w), the two
        first being unsigned integers and w the weight of corresponde edge.
    """
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

cdef class Manifold:
    pass

cdef class Euclidean(Manifold):

    cdef mld.euclidean view
    
    def __init__(self):
        pass
    
    def dot(self, Matrix A):
        """
            Dot product between rows of A. This is different from miolo.dot, 
            that computes the frobenius dot product between matrices.
        """
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.dot(drf(A.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.dot(drf(A.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.dot(drf(A.mtxDouble))
        return out

    def distance(self, Matrix A):
        """
            Euclidean distance between rows of A.
        """
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.distance(drf(A.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.distance(drf(A.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.distance(drf(A.mtxDouble))
        return out

    def mean(self, Matrix A):
        """
            Returns a row Matrix which is the mean of the rows of A.
        """
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.mean(drf(A.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.mean(drf(A.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.mean(drf(A.mtxDouble))
        return out
    
    def variance(self, Matrix A):
        """
            Returns a row Matrix which is the variance of the rows of A.
        """
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.variance(drf(A.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.variance(drf(A.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.variance(drf(A.mtxDouble))
        return out
    
    def minmaxNormalize(self, Matrix A):
        """
            For each column, calculates min and max values, and then returns
            a matrix for which each element is (A_ij-min_j)/(max_j-min_j).
        """
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.minmaxNormalize(drf(A.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.minmaxNormalize(drf(A.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.minmaxNormalize(drf(A.mtxDouble))
        return out

    def rowNormalize(self, Matrix A):
        """
            Returns A with rows normalized to sum 1.
        """
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.rowNormalize(drf(A.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.rowNormalize(drf(A.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.rowNormalize(drf(A.mtxDouble))
        return out

    def gaussianNormalize(self, Matrix A):
        """
            Returns A normalized to have columns with mean 0 and variance 1.
        """
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.gaussianNormalize(drf(A.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.gaussianNormalize(drf(A.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.gaussianNormalize(drf(A.mtxDouble))
        return out
    
    def centroidDistance(self, Matrix M, Matrix center):
        """
            Returns the distance of rows in M to rows in center.
            Useful for k-means.
        """
        if M.ctype!=center.ctype:
            raise TypeError("M and center must have same ctype.")
        if M.cols!=center.cols:
            raise Exception("M.cols must be equal to center.cols.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.centroidDistance(drf(M.mtxInt),drf(center.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.centroidDistance(drf(M.mtxFloat),drf(center.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.centroidDistance(drf(M.mtxDouble),drf(center.mtxDouble))
        return out
    
    def kmpp(self, Matrix M, int k):
        """
            K-means++ algorithm for seeding.
        """
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.kmpp(drf(M.mtxInt),k)
        if out.ctype=="float":
            out.mtxFloat = self.view.kmpp(drf(M.mtxFloat),k)
        if out.ctype=="double":
            out.mtxDouble = self.view.kmpp(drf(M.mtxDouble),k)
        return out

cdef class Sphere(Manifold):

    """
        This class treats each row of a Matrix as sphere. Therefore, it
        views a Matrix as the product manifold of Matrix.rows unit spheres 
        embedded in the euclidean space of dimension Matrix.cols. 
    """

    cdef mld.sphere view

    def __init__(self, radius=1):
        self.view.r = radius
    
    @property
    def radius(self):
        return self.view.r
    @radius.setter
    def radius(self, r):
        if r<=0:
            raise Exception("Sphere radius must be greater than 0.")
        self.view.r = r
    
    def stereographicProjection(self, Matrix M):
        """
            Returns the matrix for which each row is the stereographic 
            projection of the corresponding row in M over the sphere.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.view.stereographicProjection(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.view.stereographicProjection(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.view.stereographicProjection(drf(M.mtxDouble))
        return out
    
    def fromEuclidean(self, Matrix M):
        """
            Returns the Matrix for which each row r has M.cols+1 columns and is
            the mapping of the r row of M to the sphere embedded in the euclidean
            space of dimension M.cols+1. 
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.view.fromEuclidean(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.view.fromEuclidean(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.view.fromEuclidean(drf(M.mtxDouble))
        return out
    
    def toEuclidean(self, Matrix M):
        """
            Returns the Matrix for which each row r has M.cols-1 columns and is
            the mapping of the r row of M to the sphere embedded in the euclidean
            space of dimension M.cols+1.  
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.view.toEuclidean(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.view.toEuclidean(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.view.toEuclidean(drf(M.mtxDouble))
        return out
    
    def coordinateReady(self, Matrix M, unsigned long azimuth=0):
        """
            Transforms M in order to make its rows suitable spherical coordinates.
            First, minmax normalization is done and then columns are multiplied
            by PI, with the exception of azimuth, which is multiplied by 2*PI.
        """
        if azimuth>=M.cols:
            raise ValueError("azimuth must be smaller than M.cols.")
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.view.coordinateReady(drf(M.mtxInt),azimuth)
        if M.ctype=="float":
            out.mtxFloat = self.view.coordinateReady(drf(M.mtxFloat),azimuth)
        if M.ctype=="double":
            out.mtxDouble = self.view.coordinateReady(drf(M.mtxDouble),azimuth)
        return out
    
    def distance(self, Matrix M):
        """
            Return a square Matrix for which each entry is the geodesic distance
            between the pair of (i,j) rows of M.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.view.distance(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.view.distance(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.view.distance(drf(M.mtxDouble))
        return out
    
    def isIn(self, Matrix M, double tolerance=0.001):
        """
            Checks if each row of M is on the sphere.
            tolerance: numerical tolerance for acceptance.
        """
        if M.ctype=="int":
            return self.view.isIn(drf(M.mtxInt),<int>tolerance)
        if M.ctype=="float":
            return self.view.isIn(drf(M.mtxFloat),<float>tolerance)
        if M.ctype=="double":
            return self.view.isIn(drf(M.mtxDouble),tolerance)
    
    def isTangent(self, Matrix at, Matrix M, double tolerance=0.001):
        """
            Checks if each row of M is on the tangent space of the corresponding
            row of at.
            tolerance: numerical tolerance for acceptance.
        """
        if at.rows!=M.rows or at.cols!=M.cols:
            raise Exception("at and M must have same shape.")
        if at.ctype!=M.ctype:
            raise TypeError("at and M must have same ctype.")
        if M.ctype=="int":
            return self.view.isTangent(drf(at.mtxInt),drf(M.mtxInt),<int>tolerance)
        if M.ctype=="float":
            return self.view.isTangent(drf(at.mtxFloat),drf(M.mtxFloat),<float>tolerance)
        if M.ctype=="double":
            return self.view.isTangent(drf(at.mtxDouble),drf(M.mtxDouble),tolerance)
    
    def exp(self, Matrix at, Matrix tangent):
        """
            Exponential map on the sphere.
            @params:
                at: point on the sphere
                tangent: a tangent vector of at. 
            Note: no checking is done for either at or tangent. Ensure before
            applying.
        """
        if at.cols!=tangent.cols:
            raise Exception("at and tangent must have same number of cols.")
        if at.rows!=1 and at.rows!=tangent.rows:
            raise Exception("Invalid shape for at.")
        if at.ctype!=tangent.ctype:
            raise TypeError("at and tanget must have same ctype.")
        out = Matrix(ctype=at.ctype)
        if at.rows==tangent.rows:
            if out.ctype=="int":
                out.mtxInt = self.view.exponential(
                    drf(at.mtxInt),drf(tangent.mtxInt)
                )
            if out.ctype=="float":
                out.mtxFloat = self.view.exponential(
                    drf(at.mtxFloat),drf(tangent.mtxFloat)
                )
            if out.ctype=="double":
                out.mtxDouble = self.view.exponential(
                    drf(at.mtxDouble),drf(tangent.mtxDouble)
                )
        return out
    
    def centroidDistance(self, Matrix M, Matrix center):
        """
            Returns the distance of each row in M to each row of center.
            Useful for implementing k-means.
        """
        if M.ctype!=center.ctype:
            raise TypeError("M and center must have same ctype.")
        if M.cols!=center.cols:
            raise Exception("M.cols must be equal to center.cols.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.centroidDistance(drf(M.mtxInt),drf(center.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.centroidDistance(drf(M.mtxFloat),drf(center.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.centroidDistance(drf(M.mtxDouble),drf(center.mtxDouble))
        return out

cdef class Poincare(Manifold):

    """
        This class treats each row of a Matrix as a point in the Poincare disk.
        Convertions to and from the Klein model are also available.
    """

    cdef mld.poincare view

    def __init__(self, curvature=1):
        self.view.c = curvature
    
    @property
    def curvature(self):
        return self.view.c
    @curvature.setter
    def curvature(self, curv):
        self.view.c = curv

    def distance(self, Matrix M):
        """
            Return a square Matrix for which each entry is the geodesic distance
            between the pair of (i,j) rows of M.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.view.distance(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.view.distance(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.view.distance(drf(M.mtxDouble))
        return out
    
    def isIn(self, Matrix M):
        """
            Check if rows of M belong to the M.cols-dimensional Poincare Disk.
        """
        if M.ctype=="int":
            return self.view.isIn(drf(M.mtxInt))
        if M.ctype=="float":
            return self.view.isIn(drf(M.mtxFloat))
        if M.ctype=="double":
            return self.view.isIn(drf(M.mtxDouble))
    
    def add(self, Matrix A, Matrix B):
        """
            Mobius addition.
        """
        if A.rows!=B.rows or A.cols!=B.cols:
            raise Exception("Matrices must have same shape.")
        if A.ctype!=B.ctype:
            raise TypeError("Matrices must have same ctype.")
        out = Matrix(ctype=A.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.madd(drf(A.mtxInt),drf(B.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.madd(drf(A.mtxFloat),drf(B.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.madd(drf(A.mtxDouble),drf(B.mtxDouble))
        return out
    
    def exp(self, Matrix at, Matrix M):
        """
            Mobius addition.
        """
        if at.rows!=M.rows or at.cols!=M.cols:
            raise Exception("Matrices must have same shape.")
        if at.ctype!=M.ctype:
            raise TypeError("Matrices must have same ctype.")
        out = Matrix(ctype=at.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.exponential(drf(at.mtxInt),drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.exponential(drf(at.mtxFloat),drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.exponential(drf(at.mtxDouble),drf(M.mtxDouble))
        return out
    
    def log(self, Matrix start, Matrix end):
        """
            Mobius addition.
        """
        if start.rows!=end.rows or start.cols!=end.cols:
            raise Exception("Matrices must have same shape.")
        if start.ctype!=end.ctype:
            raise TypeError("Matrices must have same ctype.")
        out = Matrix(ctype=start.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.logarithm(drf(start.mtxInt),drf(end.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.logarithm(drf(start.mtxFloat),drf(end.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.logarithm(drf(start.mtxDouble),drf(end.mtxDouble))
        return out
    
    def mean(self, Matrix M):
        """
            Returns the Einstein Midpoint of rows of M.
        """
        out = Matrix(ctype=M.ctype)
        if M.ctype=="int":
            out.mtxInt = self.view.mean(drf(M.mtxInt))
        if M.ctype=="float":
            out.mtxFloat = self.view.mean(drf(M.mtxFloat))
        if M.ctype=="double":
            out.mtxDouble = self.view.mean(drf(M.mtxDouble))
        return out
    
    def centroidDistance(self, Matrix M, Matrix center):
        """
            Returns the distance of each row in M to each row of center.
            Useful for implementing k-means.
        """
        if M.ctype!=center.ctype:
            raise TypeError("M and center must have same ctype.")
        if M.cols!=center.cols:
            raise Exception("M.cols must be equal to center.cols.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.centroidDistance(drf(M.mtxInt),drf(center.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.centroidDistance(drf(M.mtxFloat),drf(center.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.centroidDistance(drf(M.mtxDouble),drf(center.mtxDouble))
        return out
    
    def hyperbolicity(self, Matrix M):
        """
            Returns the Gromov hiperbolicity of the rows of M.
        """
        if M.ctype=="int":
            return self.view.hyperbolicity(drf(M.mtxInt))
        if M.ctype=="float":
            return self.view.hyperbolicity(drf(M.mtxFloat))
        if M.ctype=="double":
            return self.view.hyperbolicity(drf(M.mtxDouble))

cdef class Lorentz(Manifold):

    """
        This class treats each row of a Matrix as a point in the Lorentz model.
        Matrices passed as arguments must have dimension greater than 1.
    """

    cdef mld.lorentz view

    def __init__(self, beta=1):
        self.view.beta = beta
    
    @property
    def beta(self):
        return self.beta
    @beta.setter
    def beta(self, b):
        self.view.beta = b

    def isIn(self, Matrix M, tolerance):
        """
            Checks if rows of M belong to the manifold according to specified
            tolerance.
        """
        if M.ctype=="int":
            return self.view.isIn(drf(M.mtxInt),tolerance)
        if M.ctype=="float":
            return self.view.isIn(drf(M.mtxFloat),tolerance)
        if M.ctype=="double":
            return self.view.isIn(drf(M.mtxDouble),tolerance)

    def dot(self, Matrix M):
        """
            Lorentz dot product between rows of M. Returns a square matrix.
        """
        if M.cols<2:
            raise Exception("Number of cols must be greater than 1.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.dot(drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.dot(drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.dot(drf(M.mtxDouble))
        return out
    
    def metricDistance(self, Matrix M):
        """
            Distance in the lorentz model. Returns a square matrix.
        """
        if M.cols<2:
            raise Exception("Number of cols must be greater than 1.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.distance(drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.distance(drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.distance(drf(M.mtxDouble))
        return out
    
    def lorentzDistance(self, Matrix M):
        """
            Lorentz norm with respect to the lorentzian inner product. 
            Returns a square matrix.
        """
        if M.cols<2:
            raise Exception("Number of cols must be greater than 1.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.lorentzDistance(drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.lorentzDistance(drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.lorentzDistance(drf(M.mtxDouble))
        return out
    
    def fromEuclidean(self, Matrix M):
        """
            Maps from euclidean n-dimensional space to the space of timelike
            vectors in the n-dimensional Lorentz model embedded in the 
            n+1-dimensional euclidean space. 
        """
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.fromEuclidean(drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.fromEuclidean(drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.fromEuclidean(drf(M.mtxDouble))
        return out
    
    def toEuclidean(self, Matrix M):
        """
            Maps from lorentz n-dimensional manifold embedded in (n+1)-dimensional
            euclidean space to the n-dimensional euclidean space.
        """
        if M.cols<2:
            raise Exception("Number of cols must be greater than 1.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.toEuclidean(drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.toEuclidean(drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.toEuclidean(drf(M.mtxDouble))
        return out
        
    def exp(self, Matrix at, Matrix M):
        """
            Exponential map in the Lorentz model.
        """
        if at.cols!=M.cols:
            raise Exception("at and M must have same number of cols.")
        if at.rows!=M.rows:
            raise Exception("at and M must have same number of rows.")
        if at.ctype!=M.ctype:
            raise TypeError("at and M must have same ctype.")
        if M.cols<2:
            raise Exception("Number of cols must be greater than 1.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.exponential(drf(at.mtxInt),drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.exponential(drf(at.mtxFloat),drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.exponential(drf(at.mtxDouble),drf(M.mtxDouble))
        return out
    
    def tangentProjection(self, Matrix at, Matrix M):
        """
            Projection of M on the the tangent space.
        """
        if at.cols!=M.cols:
            raise Exception("at and M must have same number of cols.")
        if at.rows!=M.rows:
            raise Exception("at and M must have same number of rows.")
        if at.ctype!=M.ctype:
            raise TypeError("at and M must have same ctype.")
        if M.cols<2:
            raise Exception("Number of cols must be greater than 1.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.tangentProjection(drf(at.mtxInt),drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.tangentProjection(drf(at.mtxFloat),drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.tangentProjection(drf(at.mtxDouble),drf(M.mtxDouble))
        return out
    
    def centroidDistance(self, Matrix M, Matrix center):
        """
            Returns the distance of each row in M to each row of center.
            Useful for implementing k-means.
        """
        if M.ctype!=center.ctype:
            raise TypeError("M and center must have same ctype.")
        if M.cols!=center.cols:
            raise Exception("M.cols must be equal to center.cols.")
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.centroidDistance(drf(M.mtxInt),drf(center.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.centroidDistance(drf(M.mtxFloat),drf(center.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.centroidDistance(drf(M.mtxDouble),drf(center.mtxDouble))
        return out
    
    def mean(self, Matrix M):
        """
            Returns the mean of rows in M in lorentzian space.
        """
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.mean(drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.mean(drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.mean(drf(M.mtxDouble))
        return out
    
    def mean(self, Matrix M):
        """
            Returns the variance of rows in M in lorentzian space.
        """
        out = Matrix(ctype=M.ctype)
        if out.ctype=="int":
            out.mtxInt = self.view.variance(drf(M.mtxInt))
        if out.ctype=="float":
            out.mtxFloat = self.view.variance(drf(M.mtxFloat))
        if out.ctype=="double":
            out.mtxDouble = self.view.variance(drf(M.mtxDouble))
        return out

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Unitary functions
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class ufunction:

    """
        Basic mathematical functions. These are implemented as subclasses of 
        ufunction.
    """

    cdef mld.ufunction view

    def __init__(self):
        pass

    def __call__(self, mioloObject X):
        if isinstance(X,Matrix):
            out = Matrix(ctype=X.ctype)
            if out.ctype=="int":
                out.mtxInt = self.view.mtxApply(drf(X.mtxInt))
            if out.ctype=="float":
                out.mtxFloat = self.view.mtxApply(drf(X.mtxFloat))
            if out.ctype=="double":
                out.mtxDouble = self.view.mtxApply(drf(X.mtxDouble))
            return out
        if isinstance(X,Graph):
            out = Graph(ctype=X.ctype)
            if out.ctype=="int":
                out.graphInt = self.view.graphApply(drf(X.graphInt))
            if out.ctype=="float":
                out.graphFloat = self.view.graphApply(drf(X.graphFloat))
            if out.ctype=="double":
                out.graphDouble = self.view.graphApply(drf(X.graphDouble))
            return out
        if isinstance(X,Digraph):
            out = Digraph(ctype=X.ctype)
            if out.ctype=="int":
                out.digraphInt = self.view.digraphApply(drf(X.digraphInt))
            if out.ctype=="float":
                out.digraphFloat = self.view.digraphApply(drf(X.digraphFloat))
            if out.ctype=="double":
                out.digraphDouble = self.view.digraphApply(drf(X.digraphDouble))
            return out

cdef class sin(ufunction):
    def __init__(self):
        self.view.f = mld.sin
cdef class cos(ufunction):
    def __init__(self):
        self.view.f = mld.cos
cdef class tan(ufunction):
    def __init__(self):
        self.view.f = mld.tan

cdef class asin(ufunction):
    def __init__(self):
        self.view.f = mld.asin
cdef class acos(ufunction):
    def __init__(self):
        self.view.f = mld.acos
cdef class atan(ufunction):
    def __init__(self):
        self.view.f = mld.atan

cdef class cosh(ufunction):
    def __init__(self):
        self.view.f = mld.cosh
cdef class sinh(ufunction):
    def __init__(self):
        self.view.f = mld.sinh
cdef class tanh(ufunction):
    def __init__(self):
        self.view.f = mld.tanh

cdef class acosh(ufunction):
    def __init__(self):
        self.view.f = mld.acosh
cdef class asinh(ufunction):
    def __init__(self):
        self.view.f = mld.asinh
cdef class atanh(ufunction):
    def __init__(self):
        self.view.f = mld.atanh

cdef class exp(ufunction):
    def __init__(self):
        super().__init__()
        self.view.f = mld.exp
cdef class log(ufunction):
    def __init__(self):
        self.view.f = mld.log

cdef class fabs(ufunction):
    def __init__(self):
        self.view.f = mld.fabs
cdef class sqrt(ufunction):
    def __init__(self):
        self.view.f = mld.sqrt
cdef class recp(ufunction):
    def __init__(self):
        self.view.f = mld.reciprocal

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#   Special functions
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cdef class classSeparation:

    cdef mld.classSeparation view 

    def intra(self, mioloObject O, int[:] labels):
        """
            Computes intraclass distance.
            Dist: distance matrix between features. Must be square.
            labels: array with labels for each instance.
        """
        if isinstance(O,Matrix):
            if O.rows!=O.cols:
                raise Exception("Only square matrices are allowed.")
            if O.rows!=labels.size:
                raise Exception("Dist.rows must be equal to labels length.")
            if O.ctype=="int":
                return self.view(drf(O.mtxInt),&labels[0],True)
            if O.ctype=="float":
                return self.view(drf(O.mtxFloat),&labels[0],True)
            if O.ctype=="double":
                return self.view(drf(O.mtxDouble),&labels[0],True)
        if isinstance(O,Graph):
            if O.nodes!=labels.size:
                raise Exception("Dist.rows must be equal to labels length.")
            if O.ctype=="int":
                return self.view(drf(O.graphInt),&labels[0],True)
            if O.ctype=="float":
                return self.view(drf(O.graphFloat),&labels[0],True)
            if O.ctype=="double":
                return self.view(drf(O.graphDouble),&labels[0],True)
    
    def extra(self, mioloObject O, int[:] labels):
        """
            Computes intraclass distance.
            Dist: distance matrix between features. Must be square.
            labels: array with labels for each instance.
        """
        if isinstance(O,Matrix):
            if O.rows!=O.cols:
                raise Exception("Only square matrices are allowed.")
            if O.rows!=labels.size:
                raise Exception("Dist.rows must be equal to labels length.")
            if O.ctype=="int":
                return self.view(drf(O.mtxInt),&labels[0],False)
            if O.ctype=="float":
                return self.view(drf(O.mtxFloat),&labels[0],False)
            if O.ctype=="double":
                return self.view(drf(O.mtxDouble),&labels[0],False)
        if isinstance(O,Graph):
            if O.nodes!=labels.size:
                raise Exception("Dist.rows must be equal to labels length.")
            if O.ctype=="int":
                return self.view(drf(O.graphInt),&labels[0],False)
            if O.ctype=="float":
                return self.view(drf(O.graphFloat),&labels[0],False)
            if O.ctype=="double":
                return self.view(drf(O.graphDouble),&labels[0],False)

cdef class weightedSquareDistance:

    cdef mld.weightedSquareDistance view

    def __call__(self, Matrix M, mioloObject Weights):
        """
            Computes the sum of distances of rows in M weighted by Weights. M and
            Weights must have same number of rows.
            M: feature Matrix.
            Weights: weight matrix between rows. Must be square Matrix or Graph.
        """
        if M.ctype!=Weights.ctype:
            raise TypeError("M and Weights must have same ctype.")
        if isinstance(Weights,Matrix):
            if Weights.rows!=Weights.cols:
                raise Exception("Only square weight matrices are allowed.")
            if M.rows!=Weights.rows:
                raise Exception("M and Weights must have same number of rows.")
            if M.ctype=="int":
                return self.view(drf(M.mtxInt),drf(Weights.mtxInt))
            if M.ctype=="float":
                return self.view(drf(M.mtxFloat),drf(Weights.mtxFloat))
            if M.ctype=="double":
                return self.view(drf(M.mtxDouble),drf(Weights.mtxDouble))
        if isinstance(Weights,Graph):
            if Weights.nodes!=M.rows:
                raise Exception("Weights.nodes must be equal to M.rows.")
            if M.ctype=="int":
                return self.view(drf(M.mtxInt),drf(Weights.graphInt))
            if M.ctype=="float":
                return self.view(drf(M.mtxFloat),drf(Weights.graphFloat))
            if M.ctype=="double":
                return self.view(drf(M.mtxDouble),drf(Weights.graphDouble))
    
cdef class weightedDotProduct:

    cdef mld.weightedDot view

    def __call__(self, Matrix M, mioloObject Weights):
        """
            Computes the sum of dot products of rows in M weighted by Weights. M and
            Weights must have same number of rows.
            M: feature Matrix.
            Weights: weight matrix between rows. Must be square Matrix or Graph.
        """
        if M.ctype!=Weights.ctype:
            raise TypeError("M and Weights must have same ctype.")
        if isinstance(Weights,Matrix):
            if Weights.rows!=Weights.cols:
                raise Exception("Only square weight matrices are allowed.")
            if M.rows!=Weights.rows:
                raise Exception("M and Weights must have same number of rows.")
            if M.ctype=="int":
                return self.view(drf(M.mtxInt),drf(Weights.mtxInt))
            if M.ctype=="float":
                return self.view(drf(M.mtxFloat),drf(Weights.mtxFloat))
            if M.ctype=="double":
                return self.view(drf(M.mtxDouble),drf(Weights.mtxDouble))
        if isinstance(Weights,Graph):
            if Weights.nodes!=M.rows:
                raise Exception("Weights.nodes must be equal to M.rows.")
            if M.ctype=="int":
                return self.view(drf(M.mtxInt),drf(Weights.graphInt))
            if M.ctype=="float":
                return self.view(drf(M.mtxFloat),drf(Weights.graphFloat))
            if M.ctype=="double":
                return self.view(drf(M.mtxDouble),drf(Weights.graphDouble))

cdef class pottsEnergy:

    cdef mld.pottsEnergy view

    def __call__(self, mioloObject Weights, int[:] labels):
        """
            Computes the sum of pairwise similarities between nodes with the
            same label.
        """
        if isinstance(Weights,Matrix):
            if Weights.rows!=Weights.cols:
                raise Exception("Only square weight matrices are allowed.")
            if Weights.rows!=labels.size:
                raise Exception("Size of labels must be equal to Weights.rows.")
            if Weights.ctype=="int":
                return self.view(drf(Weights.mtxInt),&labels[0])
            if Weights.ctype=="float":
                return self.view(drf(Weights.mtxFloat),&labels[0])
            if Weights.ctype=="double":
                return self.view(drf(Weights.mtxDouble),&labels[0])
        if isinstance(Weights,Graph):
            if Weights.nodes!=labels.size:
                raise Exception("Size of labels must be equal to Weights.nodes")
            if Weights.ctype=="int":
                return self.view(drf(Weights.graphInt),&labels[0])
            if Weights.ctype=="float":
                return self.view(drf(Weights.graphFloat),&labels[0])
            if Weights.ctype=="double":
                return self.view(drf(Weights.graphDouble),&labels[0])