from libcpp cimport bool
cimport numpy as np

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

cdef mat * numpy_to_mat(np.ndarray[np.double_t, ndim=2] X)

cdef mat numpy_to_mat_d(np.ndarray[np.double_t, ndim=2] X)

cdef cube * numpy_to_cube(np.ndarray[np.double_t, ndim=3] X)

cdef cube numpy_to_cube_d(np.ndarray[np.double_t, ndim=3] X)

cdef vec * numpy_to_vec(np.ndarray[np.double_t, ndim=1] x)

cdef vec numpy_to_vec_d(np.ndarray[np.double_t, ndim=1] x)

cdef vec * mat_col_view(mat * x, int col) nogil

cdef vec mat_col_view_d(mat * x, int col) nogil

cdef mat * cube_slice_view(cube * x, int slice) nogil

cdef mat cube_slice_view_d(cube * x, int slice) nogil

cdef np.ndarray[np.double_t, ndim=2] mat_to_numpy(const mat & X, np.ndarray[np.double_t, ndim=2] D)

cdef np.ndarray[np.double_t, ndim=1] vec_to_numpy(const vec & X, np.ndarray[np.double_t, ndim=1] D)
