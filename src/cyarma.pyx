# cython: language_level=2
import numpy as np
cimport numpy as np
cimport cython

from cython.operator cimport dereference as deref

from libcpp cimport bool

cdef extern from "armadillo" namespace "arma" nogil:
    # matrix class (double)
    cdef cppclass mat:
        mat(double * aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict) nogil
        mat(double * aux_mem, int n_rows, int n_cols) nogil
        mat(int n_rows, int n_cols) nogil
        mat() nogil
        # attributes
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        int n_nonzero
        # fuctions
        mat i() nogil #inverse
        mat t() nogil #transpose
        vec diag() nogil
        vec diag(int) nogil
        fill(double) nogil
        void raw_print(char*) nogil
        void raw_print() nogil
        vec unsafe_col(int) nogil
        vec col(int) nogil
        #print(char)
        #management
        mat reshape(int, int) nogil
        mat resize(int, int) nogil
        double * memptr() nogil
        # opperators
        double& operator[](int) nogil
        double& operator[](int,int) nogil
        double& at(int,int) nogil
        double& at(int) nogil
        mat operator*(mat) nogil
        mat operator%(mat) nogil
        vec operator*(vec) nogil
        mat operator+(mat) nogil
        mat operator-(mat) nogil
        mat operator*(double) nogil
        mat operator-(double) nogil
        mat operator+(double) nogil
        mat operator/(double) nogil
        #etc

    cdef cppclass cube:
        cube(double * aux_mem, int n_rows, int n_cols, int n_slices, bool copy_aux_mem, bool strict) nogil
        cube(double * aux_mem, int n_rows, int n_cols, int n_slices) nogil
        cube(int, int, int) nogil
        cube() nogil
        #attributes
        int n_rows
        int n_cols
        int n_elem
        int n_elem_slices
        int n_slices
        int n_nonzero
        double * memptr() nogil
        void raw_print(char*) nogil
        void raw_print() nogil
        

    # vector class (double)
    cdef cppclass vec:
        cppclass iterator:
            double& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            bint operator==(iterator)
            bint operator!=(iterator)
            bint operator<(iterator)
            bint operator>(iterator)
            bint operator<=(iterator)
            bint operator>=(iterator)
        cppclass reverse_iterator:
            double& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator>=(reverse_iterator)
        vec(double * aux_mem, int number_of_elements, bool copy_aux_mem, bool strict) nogil
        vec(double * aux_mem, int number_of_elements) nogil
        vec(int) nogil
        vec() nogil
        # attributes
        int n_elem
        # opperators
        double& operator[](int)
        double& at(int)
        vec operator%(vec)
        vec operator+(vec)
        vec operator/(vec)
        vec operator*(mat)
        vec operator*(double)
        vec operator-(double)
        vec operator+(double)
        vec operator/(double)
        iterator begin()
        iterator end()
        reverse_iterator rbegin()
        reverse_iterator rend()


        # functions
        double * memptr()
        void raw_print(char*) nogil
        void raw_print() nogil

    # Armadillo Linear Algebra tools
    cdef bool chol(mat R, mat X) nogil # preallocated result
    cdef mat chol(mat X) nogil # new result
    cdef bool inv(mat R, mat X) nogil
    cdef mat inv(mat X) nogil
    cdef bool solve(vec x, mat A, vec b) nogil
    cdef vec solve(mat A, vec b) nogil
    cdef bool solve(mat X, mat A, mat B) nogil
    cdef mat solve(mat A, mat B) nogil
    cdef bool eig_sym(vec eigval, mat eigvec, mat B) nogil
    cdef bool svd(mat U, vec s, mat V, mat X, method) nogil
    cdef bool lu(mat L, mat U, mat P, mat X) nogil
    cdef bool lu(mat L, mat U, mat X) nogil
    cdef mat pinv(mat A) nogil
    cdef bool pinv(mat B, mat A) nogil
    cdef bool qr(mat Q, mat R, mat X) nogil
    cdef float dot(vec a, vec b) nogil
    cdef mat arma_cov "cov"(mat X) nogil
    cdef vec arma_mean "mean"(mat X, int dim) nogil
    cdef mat arma_var "var"(mat X, int norm_type, int dim) nogil

## TODO: refactor to return pointer, but other function derefed. 

##### Tools to convert numpy arrays to armadillo arrays ######
cdef mat * numpy_to_mat(np.ndarray[np.double_t, ndim=2] X):
    if not (X.flags.f_contiguous or X.flags.owndata):
        X = X.copy(order="F")
    cdef mat *aR_p  = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
    return aR_p

cdef mat numpy_to_mat_d(np.ndarray[np.double_t, ndim=2] X):
    cdef mat * aR_p = numpy_to_mat(X)
    cdef mat aR = deref(aR_p)
    del aR_p
    return aR

cdef cube * numpy_to_cube(np.ndarray[np.double_t, ndim=3] X):
    cdef cube *aR_p
    if not X.flags.c_contiguous:
        raise ValueError("For Cube, numpy array must be C contiguous")
    aR_p  = new cube(<double*> X.data, X.shape[2], X.shape[1], X.shape[0], False, True)
    return aR_p

cdef cube numpy_to_cube_d(np.ndarray[np.double_t, ndim=3] X):
    cdef cube * aR_p = numpy_to_cube(X)
    cdef cube aR = deref(aR_p)
    del aR_p
    return aR
    
cdef vec * numpy_to_vec(np.ndarray[np.double_t, ndim=1] x):
    if not (x.flags.f_contiguous or x.flags.owndata):
        x = x.copy()
    cdef vec *ar_p = new vec(<double*> x.data, x.shape[0], False, True)
    return ar_p

cdef vec numpy_to_vec_d(np.ndarray[np.double_t, ndim=1] x):
    cdef vec *ar_p = numpy_to_vec(x)
    cdef vec ar = deref(ar_p)
    del ar_p
    return ar
    

#### Get subviews #####
cdef vec * mat_col_view(mat * x, int col) nogil:
    cdef vec * ar_p = new vec(x.memptr()+x.n_rows*col, x.n_rows, False, True)
    return ar_p

cdef vec mat_col_view_d(mat * x, int col) nogil:
    cdef vec * ar_p = mat_col_view(x, col)
    cdef vec ar = deref(ar_p)
    del ar_p
    return ar

cdef mat * cube_slice_view(cube * x, int slice) nogil:
    cdef mat *ar_p = new mat(x.memptr() + x.n_rows*x.n_cols*slice,
                           x.n_rows, x.n_cols, False, True)
    return ar_p

cdef mat cube_slice_view_d(cube * x, int slice) nogil:
    cdef mat * ar_p = cube_slice_view(x, slice)
    cdef mat ar = deref(ar_p)
    del ar_p
    return ar



##### Converting back to python arrays, must pass preallocated memory or None
# all data will be copied since numpy doesn't own the data and can't clean up
# otherwise. Maybe this can be improved. #######
@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=2] mat_to_numpy(const mat & X, np.ndarray[np.double_t, ndim=2] D):
    cdef const double * Xptr = X.memptr()
    
    if D is None:
        D = np.empty((X.n_rows, X.n_cols), dtype=np.double, order="F")
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_rows*X.n_cols):
        Dptr[i] = Xptr[i]
    return D

@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=1] vec_to_numpy(const vec & X, np.ndarray[np.double_t, ndim=1] D):
    cdef const double * Xptr = X.memptr()
    
    if D is None:
        D = np.empty(X.n_elem, dtype=np.double)
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_elem):
        Dptr[i] = Xptr[i]
    return D

### A few wrappers for much needed numpy linalg functionality using armadillo
# cpdef np_chol(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     # wrap them up in armidillo arrays
#     cdef mat *aX = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef mat *aR  = new mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
    
#     chol(deref(aR), deref(aX))
    
#     return R

# cpdef np_inv(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     # wrap them up in armidillo arrays
#     cdef mat *aX = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef mat *aR  = new mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
    
#     inv(deref(aR), deref(aX))
    
#     return R


# def np_eig_sym(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     cdef np.ndarray[np.double_t, ndim=1] v = \
#          np.empty(X.shape[0], dtype=np.double)
#     # wrap them up in armidillo arrays
#     cdef mat *aX = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef mat *aR  = new mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
#     cdef vec *av = new vec(<double*> v.data, v.shape[0], False, True)

#     eig_sym(deref(av), deref(aR), deref(aX))

#     return [v, R]


# def example(np.ndarray[np.double_t, ndim=2] X):
#     cdef mat aX = numpy_to_mat(X)
#     cdef mat XX = aX.t() * aX
#     cdef mat ch = chol(XX)
#     ch.raw_print()
#     cdef np.ndarray[np.double_t,ndim=2] Y = mat_to_numpy(ch, None)
#     return Y
