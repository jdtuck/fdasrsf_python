double          c8_abs ( double complex c );
double complex  c8_acos ( double complex c1 );
double complex  c8_acosh ( double complex c1 );
double complex  c8_add ( double complex c1, double complex c2 );
double          c8_arg ( double complex c );
double complex  c8_asin ( double complex c1 );
double complex  c8_asinh ( double complex c1 );
double complex  c8_atan ( double complex c1 );
double complex  c8_atanh ( double complex c1 );
double complex  c8_conj ( double complex c1 );
void            c8_copy ( double complex c1, double complex c2 );
double complex  c8_cos ( double complex c1 );
double complex  c8_cosh ( double complex c1 );
double complex  c8_cube_root ( double complex c1 );
double complex  c8_div ( double complex c1, double complex c2 );
double complex  c8_div_r8 ( double complex c1, double r );
double complex  c8_exp ( double complex c1 );
double complex  c8_i ( void );
double          c8_imag ( double complex c );
double complex  c8_inv ( double complex c1 );
int             c8_le_l1 ( double complex x, double complex y );
int             c8_le_l2 ( double complex x, double complex y );
int             c8_le_li ( double complex x, double complex y );
double complex  c8_log ( double complex c1 );
double          c8_mag ( double complex c );
double complex  c8_mul ( double complex c1, double complex c2 );
double complex  c8_neg ( double complex c1 );
double complex  c8_nint ( double complex c1 );
double          c8_norm_l1 ( double complex x );
double          c8_norm_l2 ( double complex x );
double          c8_norm_li ( double complex x );
double complex  c8_normal_01 ( int *seed );
double complex  c8_one ( void );
void            c8_print ( double complex a, char *title );
double          c8_real ( double complex c );
double complex  c8_sin ( double complex c1 );
double complex  c8_sinh ( double complex c1 );
double complex  c8_sqrt ( double complex c1 );
double complex  c8_sub ( double complex c1, double complex c2 );
void            c8_swap ( double complex c1, double complex c2 );
double complex  c8_tan ( double complex c1 );
double complex  c8_tanh ( double complex c1 );
void            c8_to_cartesian ( double complex c, double *x, double *y );
void            c8_to_polar ( double complex c, double *r, double *theta );
double complex  c8_uniform_01 ( int *seed );
double complex  c8_zero ( void );
void            c8mat_add ( int m, int n, double complex alpha, double complex a[],
                double complex beta, double complex b[], double complex c[] );
void            c8mat_add_r8 ( int m, int n, double alpha, double complex a[],
                double beta, double complex b[], double complex c[] );
void            c8mat_copy ( int m, int n, double complex a[], double complex b[] );
double complex *c8mat_copy_new ( int m, int n, double complex a1[] );
void            c8mat_fss ( int n, double complex a[], int nb, double complex x[] );
double complex *c8mat_fss_new ( int n, double complex a[], int nb, 
                double complex b[] );
double complex *c8mat_identity_new ( int n );
double complex *c8mat_indicator_new ( int m, int n );
void            c8mat_minvm ( int n1, int n2, double complex a[], 
                double complex b[], double complex e[] );
double complex *c8mat_minvm_new ( int n1, int n2, double complex a[], 
                double complex b[] );
void            c8mat_mm ( int n1, int n2, int n3, double complex a[], double complex b[], 
                double complex c[] );
double complex *c8mat_mm_new ( int n1, int n2, int n3, double complex a[], 
                double complex b[] );
void            c8mat_nint ( int m, int n, double complex a[] );
double          c8mat_norm_fro ( int m, int n, double complex a[] );
double          c8mat_norm_l1 ( int m, int n, double complex a[] );
double          c8mat_norm_li ( int m, int n, double complex a[] );
void            c8mat_print ( int m, int n, double complex a[], char *title );
void            c8mat_print_some ( int m, int n, double complex a[], int ilo, int jlo, 
                int ihi, int jhi, char *title );
void            c8mat_scale ( int m, int n, double complex alpha, double complex a[] );
void            c8mat_scale_r8 ( int m, int n, double alpha, double complex a[] );
double complex *c8mat_uniform_01 ( int m, int n, int *seed );
double complex *c8mat_zero_new ( int m, int n );
void            c8vec_copy ( int n, double complex a[], double complex b[] );
double complex *c8vec_copy_new ( int n, double complex a1[] );
void            c8vec_nint ( int n, double complex a[] );
double          c8vec_norm_l2 ( int n, double complex a[] );
void            c8vec_print ( int n, double complex a[], char *title );
void            c8vec_print_part ( int n, double complex a[], int max_print, char *title );
void            c8vec_sort_a_l2 ( int n, double complex x[] );
double complex *c8vec_spiral_new ( int n, int m, double complex c1, 
                double complex c2 );
double complex *c8vec_uniform_01_new ( int n, int *seed );
double complex *c8vec_unity ( int n );
double complex  cartesian_to_c8 ( double x, double y );
double complex  polar_to_c8 ( double r, double theta );
double complex  r8_csqrt ( double x );
void            r8poly2_root ( double a, double b, double c, double complex *r1,
                double complex *r2 );
void            r8poly3_root ( double a, double b, double c, double d, double complex *r1,
                double complex *r2, double complex *r3 );
void            r8poly4_root ( double a, double b, double c, double d, double e,
                double complex *r1, double complex *r2, double complex *r3,
                double complex *r4 );
