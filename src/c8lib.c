# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <complex.h>
# include <time.h>

# include "c8lib.h"
# include "r8lib.h"

/******************************************************************************/

double c8_abs ( double complex c )

/******************************************************************************/
/*
  Purpose:

    C8_ABS returns the absolute value of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C, the argument.

    Output, double C8_ABS, the function value.
*/
{
  double value;

  value = sqrt ( pow ( creal ( c ), 2 ) 
               + pow ( cimag ( c ), 2 ) );

  return value;
}
/******************************************************************************/

double complex c8_acos ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_ACOS evaluates the inverse cosine of a C8.

  Discussion:

    Here we use the relationship:

      C8_ACOS ( Z ) = pi/2 - C8_ASIN ( Z ).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_ACOS, the function value.
*/
{
  double complex c2;
  double c2_imag;
  double c2_real;
  double r8_pi_half = 1.57079632679489661923;

  c2 = c8_asin ( c1 );

  c2_real = r8_pi_half - creal ( c2 );
  c2_imag =            - cimag ( c2 );

  c2 = c2_real + c2_imag * I;

  return c2;
}
/******************************************************************************/

double complex c8_acosh ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_ACOSH evaluates the inverse hyperbolic cosine of a C8.

  Discussion:

    Here we use the relationship:

      C8_ACOSH ( Z ) = i * C8_ACOS ( Z ).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_ACOSH, the function value.
*/
{
  double complex c2;

  c2 = c8_i ( ) * c8_acos ( c1 );
  
  return c2;
}
/******************************************************************************/

double complex c8_add ( double complex c1, double complex c2 )

/******************************************************************************/
/*
  Purpose:

    C8_ADD adds two C8's.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, C2, the arguments.

    Output, double complex C8_ADD, the sum of C1 and C2.
*/
{
  double complex c3;
  double c3_imag;
  double c3_real;

  c3_real = creal ( c1 ) + creal ( c2 );
  c3_imag = cimag ( c1 ) + cimag ( c2 );

  c3 = c3_real + c3_imag * I;

  return c3;
}
/******************************************************************************/

double c8_arg ( double complex c )

/******************************************************************************/
/*
  Purpose:

    C8_ARG returns the argument of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C, the complex number.

    Output, double C8_ARG, the function value.
*/
{
  double value;

  if ( cimag ( c ) == 0.0 && creal ( c ) == 0.0 )
  {
    value = 0.0;
  }
  else
  {
    value = atan2 ( cimag ( c ), creal ( c ) );
  }
  return value;
}
/******************************************************************************/

double complex c8_asin ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_ASIN evaluates the inverse sine of a C8.

  Discussion:

    Here we use the relationship:

      C8_ASIN ( Z ) = - i * log ( i * z + sqrt ( 1 - z * z ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_ASIN, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex ce;

  c2 = c8_i ( );
  c3 = c8_sqrt ( 1.0 - c1 * c1 );
  c4 = c8_log ( c3 + c2 * c1 );
  ce = - c2 * c4;

  return ce;
}
/******************************************************************************/

double complex c8_asinh ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_ASINH evaluates the inverse hyperbolic sine of a C8.

  Discussion:

    Here we use the relationship:

      C8_ASINH ( Z ) = - i * C8_ASIN ( i * Z ).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_ASINH, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;

  c2 = c8_i ( );
  c3 = c2 * c1;
  c4 = c8_asin ( c3 );
  c5 = c2 * c4;
  c6 = - c5;

  return c6;
}
/******************************************************************************/

double complex c8_atan ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_ATAN evaluates the inverse tangent of a C8.

  Discussion:

    Here we use the relationship:

      C8_ATAN ( Z ) = ( i / 2 ) * log ( ( 1 - i * z ) / ( 1 + i * z ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_ATAN, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;
  double complex c7;
  double complex c8;
  double complex c9;
  double complex cx;

  c2 = c8_i ( );
  c3 = c8_one ( );
  c4 = c8_mul ( c2, c1 );
  c5 = c8_sub ( c3, c4 );
  c6 = c8_add ( c3, c4 );
  c7 = c8_div ( c5, c6 );

  c8 = c8_log ( c7 );
  c9 = c8_mul ( c2, c8 );
  cx = c9 / 2.0;

  return cx;
}
/******************************************************************************/

double complex c8_atanh ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_ATANH evaluates the inverse hyperbolic tangent of a C8.

  Discussion:

    Here we use the relationship:

      C8_ATANH ( Z ) = - i * C8_ATAN ( i * Z ).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_ATANH, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;

  c2 = c8_i ( );

  c3 = c8_mul ( c2, c1 );
  c4 = c8_atan ( c3 );
  c5 = c8_mul ( c2, c4 );
  c6 = c8_neg ( c5 );

  return c6;
}
/******************************************************************************/

double complex c8_conj ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_CONJ conjugates a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_CONJ, the function value.
*/
{
  double c1_norm;
  double complex c2;
  double c2_imag;
  double c2_real;

  c2 = conj ( c1 );

  return c2;
}
/******************************************************************************/

void c8_copy ( double complex c1, double complex c2 )

/******************************************************************************/
/*
  Purpose:

    C8_COPY copies a C8.

  Discussion:

    The order of the arguments may seem unnatural, but it is arranged so
    that the call

      c8_copy ( c1, c2 )

    mimics the assignment

      c1 = c2.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Output, double complex C1, the copy of C2.

    Input, double complex C2, the value to be copied.
*/
{
  c1 = c2;

  return;
}
/******************************************************************************/

double complex c8_cos ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_COS evaluates the cosine of a C8.

  Discussion:

    We use the relationship:

      C8_COS ( C ) = ( C8_EXP ( i * C ) + C8_EXP ( - i * C ) ) / 2

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_COS, the function value.
*/
{
  double complex c2;
  double r;

  c2 = ( cexp ( c1 * I ) + cexp ( - c1 * I ) ) / 2.0;

  return c2;
}
/******************************************************************************/

double complex c8_cosh ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_COSH evaluates the hyperbolic cosine of a C8.

  Discussion:

    A C8 is a C8_COMPLEX value.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_COSH, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;

  c2 = c8_exp ( c1 );

  c3 = c8_neg ( c1 );
  c4 = c8_exp ( c3 );

  c5 = c8_add ( c2, c4 );
  c6 = c8_div_r8 ( c5, 2.0 );

  return c6;
}
/******************************************************************************/

double complex c8_cube_root ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_CUBE_ROOT computes the principal cube root of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_CUBE_ROOT, the function value.
*/
{
  double complex c2;
  double r;
  double t;

  c8_to_polar ( c1, &r, &t );

  r = pow ( r, 1.0 / 3.0 );
  t = t / 3.0;

  c2 = polar_to_c8 ( r, t );

  return c2;
}
/******************************************************************************/

double complex c8_div ( double complex c1, double complex c2 )

/******************************************************************************/
/*
  Purpose:

    C8_DIV divides two C8's.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, C2, the arguments.

    Output, double complex C8_DIV, the function value.
*/
{
  double c2_norm;
  double complex c3;
  double c3_imag;
  double c3_real;

  c2_norm = c8_abs ( c2 );

  c3_real = ( creal ( c1 ) * creal ( c2 ) 
            + cimag ( c1 ) * cimag ( c2 ) ) / c2_norm / c2_norm;

  c3_imag = ( cimag ( c1 ) * creal ( c2 ) 
            - creal ( c1 ) * cimag ( c2 ) ) / c2_norm / c2_norm;

  c3 = c3_real + c3_imag * I;

  return c3;
}
/******************************************************************************/

double complex c8_div_r8 ( double complex c1, double r )

/******************************************************************************/
/*
  Purpose:

    C8_DIV_R8 divides a C8 by an R8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the value to be divided.

    Input, double R, the divisor.

    Output, double complex C8_DIV_R8, the function value.
*/
{
  double complex c2;

  c2 = c1 / r;

  return c2;
}
/******************************************************************************/

double complex c8_exp ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_EXP exponentiates a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_EXP, the function value.
*/
{
  double complex c2;

  c2 = cexp ( c1 );

  return c2;
}
/******************************************************************************/

double complex c8_i ( void )

/******************************************************************************/
/*
  Purpose:

    C8_I returns the value of I as a C8

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Output, double complex C8_I, the value of I.
*/
{
  double complex c1;

  c1 = I;

  return c1;
}
/******************************************************************************/

double c8_imag ( double complex c )

/******************************************************************************/
/*
  Purpose:

    C8_IMAG returns the imaginary part of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C, the argument.

    Output, double C8_IMAG, the function value.
*/
{
  double value;

  value = cimag ( c );

  return value;
}
/******************************************************************************/

double complex c8_inv ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_INV inverts a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_INV, the function value;
*/
{
  double complex c2;

  c2 = 1.0 / c1;

  return c2;
}
/******************************************************************************/

int c8_le_l1 ( double complex x, double complex y )

/******************************************************************************/
/*
  Purpose:

    C8_LE_L1 := X <= Y for C8 values, and the L1 norm.

  Discussion:

    A C8 is a double complex value.

    The L1 norm can be defined here as:

      C8_NORM_L1(X) = abs ( real (X) ) + abs ( imag (X) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, double complex X, Y, the values to be compared.

    Output, int C8_LE_L1, is 1 if X <= Y.
*/
{
  int value;

  if ( r8_abs ( creal ( x ) ) + r8_abs ( cimag ( x ) ) <= 
       r8_abs ( creal ( y ) ) + r8_abs ( cimag ( y ) ) )
  {
    value = 1;
  }
  else
  {
    value = 0;
  }
  return value;
}
/******************************************************************************/

int c8_le_l2 ( double complex x, double complex y )

/******************************************************************************/
/*
  Purpose:

    C8_LE_L2 := X <= Y for C8 values, and the L2 norm.

  Discussion:

    A C8 is a double complex value.

    The L2 norm can be defined here as:

      C8_NORM_L2(X) = sqrt ( ( real (X) )^2 + ( imag (X) )^2 )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2012

  Author:

    John Burkardt

  Parameters:

    Input, double complex X, Y, the values to be compared.

    Output, int C8_LE_L2, is 1 if X <= Y.
*/
{
  int value;

  if ( pow ( creal ( x ), 2 ) + pow ( cimag ( x ), 2 ) <= 
       pow ( creal ( y ), 2 ) + pow ( cimag ( y ), 2 ) )
  {
    value = 1;
  }
  else
  {
    value = 0;
  }
  return value;
}
/******************************************************************************/

int c8_le_li ( double complex x, double complex y )

/******************************************************************************/
/*
  Purpose:

    C8_LE_LI := X <= Y for C8 values, and the L-oo norm.

  Discussion:

    A C8 is a double complex value.

    The L-oo norm can be defined here as:

      C8_NORM_LI(X) = max ( abs ( real (X) ), abs ( imag (X) ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, double complex X, Y, the values to be compared.

    Output, int C8_LE_LI, is 1 if X <= Y.
*/
{
  int value;

  if ( r8_max ( r8_abs ( creal ( x ) ), r8_abs ( cimag ( x ) ) ) <= 
       r8_max ( r8_abs ( creal ( y ) ), r8_abs ( cimag ( y ) ) ) )
  {
    value = 1;
  }
  else
  {
    value = 0;
  }
  return value;
}
/******************************************************************************/

double complex c8_log ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_LOG evaluates the logarithm of a C8.

  Discussion:

    Here we use the relationship:

      C8_LOG ( Z ) = LOG ( MAG ( Z ) ) + i * ARG ( Z )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_LOG, the function value.
*/
{
  double arg;
  double complex c2;
  double mag;

  arg = c8_arg ( c1 );
  mag = c8_mag ( c1 );

  c2 = log ( mag ) + arg * I;

  return c2;
}
/******************************************************************************/

double c8_mag ( double complex c )

/******************************************************************************/
/*
  Purpose:

    C8_MAG returns the magnitude of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C, the argument.

    Output, double C8_MAG, the function value.
*/
{
  double value;

  value = sqrt ( pow ( creal ( c ), 2 ) 
               + pow ( cimag ( c ), 2 ) );

  return value;
}
/******************************************************************************/

double complex c8_mul ( double complex c1, double complex c2 )

/******************************************************************************/
/*
  Purpose:

    C8_MUL multiplies two C8's.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, C2, the arguments.

    Output, double complex C8_MUL, the function value.
*/
{
  double complex c3;

  c3 = c1 * c2;

  return c3;
}
/******************************************************************************/

double complex c8_neg ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_NEG negates a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_NEG, the function value.
*/
{
  double complex c2;

  c2 = - c1;

  return c2;
}
/******************************************************************************/

double complex c8_nint ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_NINT returns the nearest complex integer of a C8.

  Discussion:

    A C8 is a double complex value.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the value to be NINT'ed.

    Output, double complex C8_NINT, the NINT'ed value.
*/
{
  double r;
  double r_min;
  double x;
  double x_min;
  double xc;
  double y;
  double y_min;
  double yc;
  double complex value;

  xc = creal ( c1 );
  yc = cimag ( c1 );
/*
  Lower left.
*/
  x = r8_floor ( creal ( c1 ) );
  y = r8_floor ( cimag ( c1 ) );
  r = pow ( x - xc, 2 ) + ( y - yc, 2 );
  r_min = r;
  x_min = x;
  y_min = y;
/*
  Lower right.
*/
  x = r8_floor ( creal ( c1 ) ) + 1.0;
  y = r8_floor ( cimag ( c1 ) );
  r = pow ( x - xc, 2 ) + ( y - yc, 2 );
  if ( r < r_min )
  {
    r_min = r;
    x_min = x;
    y_min = y;
  }
/*
  Upper right.
*/
  x = r8_floor ( creal ( c1 ) ) + 1.0;
  y = r8_floor ( cimag ( c1 ) ) + 1.0;
  r = pow ( x - xc, 2 ) + ( y - yc, 2 );
  if ( r < r_min )
  {
    r_min = r;
    x_min = x;
    y_min = y;
  }
/*
  Upper left.
*/
  x = r8_floor ( creal ( c1 ) );
  y = r8_floor ( cimag ( c1 ) ) + 1.0;
  r = pow ( x - xc, 2 ) + ( y - yc, 2 );
  if ( r < r_min )
  {
    r_min = r;
    x_min = x;
    y_min = y;
  }

  value = x_min + y_min * I;

  return value;
}
/******************************************************************************/

double c8_norm_l1 ( double complex x )

/******************************************************************************/
/*
  Purpose:

    C8_NORM_L1 evaluates the L1 norm of a C8.

  Discussion:

    A C8 is a double complex value.

    Numbers of equal norm lie along diamonds centered at (0,0).

    The L1 norm can be defined here as:

      C8_NORM_L1(X) = abs ( real (X) ) + abs ( imag (X) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, double complex X, the value whose norm is desired.

    Output, double C8_NORM_L1, the norm of X.
*/
{
  double value;

  value = fabs ( creal ( x ) )
        + fabs ( cimag ( x ) );

  return value;
}
/******************************************************************************/

double c8_norm_l2 ( double complex x )

/******************************************************************************/
/*
  Purpose:

    C8_NORM_L2 evaluates the L2 norm of a C8.

  Discussion:

    A C8 is a double complex value.

    Numbers of equal norm lie on circles centered at (0,0).

    The L2 norm can be defined here as:

      C8_NORM_L2(X) = sqrt ( ( real (X) )^2 + ( imag ( X ) )^2 )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, double complex X, the value whose norm is desired.

    Output, double C8_NORM_L2, the 2-norm of X.
*/
{
  double value;

  value = sqrt ( pow ( creal ( x ), 2 )
               + pow ( cimag ( x ), 2 ) );

  return value;
}
/******************************************************************************/

double c8_norm_li ( double complex x )

/******************************************************************************/
/*
  Purpose:

    C8_NORM_LI evaluates the L-infinity norm of a C8.

  Discussion:

    A C8 is a double complex value.

    Numbers of equal norm lie along squares whose centers are at (0,0).

    The L-infinity norm can be defined here as:

      C8_NORM_LI(X) = max ( abs ( real (X) ), abs ( imag (X) ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, double complex X, the value whose norm is desired.

    Output, double C8_NORM_LI, the infinity norm of X.
*/
{
  double value;

  value = r8_max ( fabs ( creal ( x ) ), fabs ( cimag ( x ) ) );

  return value;
}
/******************************************************************************/

double complex c8_normal_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    C8_NORMAL_01 returns a unit pseudonormal C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input/output, int *SEED, a seed for the random number generator.

    Output, double complex C8_NORMAL_01, a unit pseudonormal value.
*/
{
  double r8_pi = 3.141592653589793;
  double v1;
  double v2;
  double complex value;
  double x_c;
  double x_r;

  v1 = r8_uniform_01 ( seed );
  v2 = r8_uniform_01 ( seed );

  x_r = sqrt ( - 2.0 * log ( v1 ) ) * cos ( 2.0 * r8_pi * v2 );
  x_c = sqrt ( - 2.0 * log ( v1 ) ) * sin ( 2.0 * r8_pi * v2 );

  value = cartesian_to_c8 ( x_r, x_c );

  return value;
}
/******************************************************************************/

double complex c8_one ( void )

/******************************************************************************/
/*
  Purpose:

    C8_ONE returns the value of 1 as a C8

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Output, double complex C8_ONE, the value of 1.
*/
{
  double complex c1;

  c1 = 1.0;

  return c1;
}
/******************************************************************************/

void c8_print ( double complex a, char *title )

/******************************************************************************/
/*
  Purpose:

    C8_PRINT prints a C8.

  Discussion:

    A C8 is a double complex value.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, double complex A, the value to be printed.

    Input, char *TITLE, a title.
*/
{
  printf ( "%s  ( %g, %g )\n", creal ( a ), cimag ( a ) );

  return;
}
/******************************************************************************/

double c8_real ( double complex c )

/******************************************************************************/
/*
  Purpose:

    C8_REAL returns the real part of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C, the complex number.

    Output, double C8_REAL, the function value.
*/
{
  double value;

  value = creal ( c );

  return value;
}
/******************************************************************************/

double complex c8_sin ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_SIN evaluates the sine of a C8.

  Discussion:

    We use the relationship:

      C8_SIN ( C ) = - i * ( C8_EXP ( i * C ) - C8_EXP ( - i * C ) ) / 2

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_SIN, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;
  double complex c7;
  double complex c8;
  double complex c9;
  double complex cx;
  double r;

  c2 = c8_i ( );

  c3 = c8_mul ( c2, c1 );
  c4 = c8_exp ( c3 );

  c5 = c8_neg ( c3 );
  c6 = c8_exp ( c5 );

  c7 = c8_sub ( c4, c6 );

  r = 2.0;
  c8 = c8_div_r8 ( c7, r );
  c9 = c8_mul ( c8, c2 );
  cx = c8_neg ( c9 );

  return cx;
}
/******************************************************************************/

double complex c8_sinh ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_SINH evaluates the hyperbolic sine of a C8.

  Discussion:

    We use the relationship:

      C8_SINH ( C ) = ( C8_EXP ( C ) - C8_EXP ( - i * C ) ) / 2

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_SINH, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;
  double r;

  c2 = c8_exp ( c1 );

  c3 = c8_neg ( c1 );
  c4 = c8_exp ( c3 );

  c5 = c8_sub ( c2, c4 );

  r = 2.0;
  c6 = c8_div_r8 ( c5, r );

  return c6;
}
/******************************************************************************/

double complex c8_sqrt ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_SQRT computes a square root of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_SQRT, the function value.
*/
{
  double complex c2;
  double r;
  double t;

  c8_to_polar ( c1, &r, &t );

  r = sqrt ( r );
  t = t / 2.0;

  c2 = polar_to_c8 ( r, t );

  return c2;
}
/******************************************************************************/

double complex c8_sub ( double complex c1, double complex c2 )

/******************************************************************************/
/*
  Purpose:

    C8_SUB subtracts two C8's.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, C2, the arguments.

    Output, double complex C8_SUB, the function value.
*/
{
  double complex c3;

  c3 = c1 - c2;

  return c3;
}
/******************************************************************************/

void c8_swap ( double complex c1, double complex c2 )

/******************************************************************************/
/*
  Purpose:

    C8_SWAP swaps two C8's.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input/output, double complex C1, C2, the arguments.
*/
{
  double complex c3;

  c3 = c1;
  c1 = c2;
  c2 = c3;

  return;
}
/******************************************************************************/

double complex c8_tan ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_TAN evaluates the tangent of a C8.

  Discussion:

    We use the relationship:

      C8_TAN ( C ) = - i * ( C8_EXP ( i * C ) - C8_EXP ( - i * C ) ) 
                         / ( C8_EXP ( I * C ) + C8_EXP ( - i * C ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_TAN, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;
  double complex c7;
  double complex c8;
  double complex c9;
  double complex cx;
  double complex ce;

  c2 = c8_i ( );
  c3 = c8_mul ( c2, c1 );
  c4 = c8_neg ( c3 );
  
  c5 = c8_exp ( c3 );
  c6 = c8_exp ( c4 );

  c7 = c8_sub ( c5, c6 );
  c8 = c8_add ( c5, c6 );

  c9 = c8_div ( c7, c8 );
  cx = c8_mul ( c2, c9 );
  ce = c8_neg ( cx );

  return ce;
}
/******************************************************************************/

double complex c8_tanh ( double complex c1 )

/******************************************************************************/
/*
  Purpose:

    C8_TANH evaluates the hyperbolic tangent of a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C1, the argument.

    Output, double complex C8_TANH, the function value.
*/
{
  double complex c2;
  double complex c3;
  double complex c4;
  double complex c5;
  double complex c6;
  double complex c7;

  c2 = c8_exp ( c1 );

  c3 = c8_neg ( c1 );
  c4 = c8_exp ( c3 );

  c5 = c8_sub ( c2, c4 );
  c6 = c8_add ( c2, c4 );

  c7 = c8_div ( c5, c6 );

  return c7;
}
/******************************************************************************/

void c8_to_cartesian ( double complex c, double *x, double *y )

/******************************************************************************/
/*
  Purpose:

    C8_TO_CARTESIAN converts a C8 to Cartesian form.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C, the argument.

    Output, double *X, *Y, the Cartesian form.
*/
{
  *x = creal ( c );
  *y = cimag ( c );

  return;
}
/******************************************************************************/

void c8_to_polar ( double complex c, double *r, double *theta )

/******************************************************************************/
/*
  Purpose:

    C8_TO_POLAR converts a C8 to polar form.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double complex C, the argument.

    Output, double *R, *THETA, the polar form.
*/
{
  *r = c8_abs ( c );
  *theta = c8_arg ( c );

  return;
}
/******************************************************************************/

double complex c8_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    C8_UNIFORM_01 returns a unit pseudorandom C8.

  Discussion:

    The angle should be uniformly distributed between 0 and 2 * PI,
    the square root of the radius uniformly distributed between 0 and 1.

    This results in a uniform distribution of values in the unit circle.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Second Edition,
    Springer, 1987,
    ISBN: 0387964673,
    LC: QA76.9.C65.B73.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, December 1986, pages 362-376.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation,
    edited by Jerry Banks,
    Wiley, 1998,
    ISBN: 0471134031,
    LC: T57.62.H37.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, Number 2, 1969, pages 136-143.

  Parameters:

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, double complex C8_UNIFORM_01, a pseudorandom complex value.
*/
{
  int i4_huge = 2147483647;
  int k;
  double r;
  double r8_pi = 3.141592653589793;
  double theta;
  double complex value;

  if ( *seed == 0 )
  {
    printf ( "\n" );
    printf ( "C8_UNIFORM_01 - Fatal error!\n" );
    printf ( "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  r = sqrt ( ( ( double ) ( *seed ) * 4.656612875E-10 ) );

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  theta = 2.0 * r8_pi * ( ( double ) ( *seed ) * 4.656612875E-10 );

  value = r * cos ( theta ) + r * sin ( theta ) * I;
  
  return value;
}
/******************************************************************************/

double complex c8_zero ( void )

/******************************************************************************/
/*
  Purpose:

    C8_ZERO returns the value of 0 as a C8

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Output, double complex C8_ZERO, the value of 0.
*/
{
  double complex c1;

  c1 = 0.0;

  return c1;
}
/******************************************************************************/

void c8mat_add ( int m, int n, double complex alpha, double complex a[],
  double complex beta, double complex b[], double complex c[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_ADD combines two C8MAT's with complex scale factors.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, double complex ALPHA, the first scale factor.

    Input, double complex A[M*N], the first matrix.

    Input, double complex BETA, the second scale factor.

    Input, double complex B[M*N], the second matrix.

    Output, double complex C[M*N], the result.
*/
{
  int i;
  int j;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      c[i+j*m] = alpha * a[i+j*m] + beta * b[i+j*m];
    }
  }
  return;
}
/******************************************************************************/

void c8mat_add_r8 ( int m, int n, double alpha, double complex a[],
  double beta, double complex b[], double complex c[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_ADD_R8 combines two C8MAT's with real scale factors.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, double ALPHA, the first scale factor.

    Input, double complex A[M*N], the first matrix.

    Input, double BETA, the second scale factor.

    Input, double complex B[M*N], the second matrix.

    Output, double complex C[M*N], the result.
*/
{
  int i;
  int j;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      c[i+j*m] = alpha * a[i+j*m] + beta * b[i+j*m];
    }
  }
  return;
}
/******************************************************************************/

void c8mat_copy ( int m, int n, double complex a[], double complex b[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_COPY copies one C8MAT to another.

  Discussion:

    A C8MAT is a doubly dimensioned array of C8's, which
    may be stored as a vector in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, double complex A[M*N], the matrix to be copied.

    Output, double complex B[M*N], the copy.
*/
{
  int i;
  int j;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      b[i+j*m] = a[i+j*m];
    }
  }

  return;
}
/******************************************************************************/

double complex *c8mat_copy_new ( int m, int n, double complex a1[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_COPY_NEW copies one C8MAT to a "new" C8MAT.

  Discussion:

    A C8MAT is a doubly dimensioned array of C8's, which
    may be stored as a vector in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, double complex A1[M*N], the matrix to be copied.

    Output, double complex C8MAT_COPY_NEW[M*N], the copy of A1.
*/
{
  double complex *a2;
  int i;
  int j;

  a2 = ( double complex * ) malloc ( m * n * sizeof ( double complex ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      a2[i+j*m] = a1[i+j*m];
    }
  }

  return a2;
}
/******************************************************************************/

void c8mat_fss ( int n, double complex a[], int nb, double complex x[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_FSS factors and solves a system with multiple right hand sides.

  Discussion:

    This routine uses partial pivoting, but no pivot vector is required.

    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
    in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    05 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the matrix.
    N must be positive.

    Input/output, double complex A[N*N].
    On input, A is the coefficient matrix of the linear system.
    On output, A is in unit upper triangular form, and
    represents the U factor of an LU factorization of the
    original coefficient matrix.

    Input, int NB, the number of right hand sides.

    Input/output, double complex X[N*NB], on input, the right hand sides of the
    linear systems.  On output, the solutions of the linear systems.
*/
{
  int i;
  int ipiv;
  int j;
  int jcol;
  double piv;
  double complex t;

  for ( jcol = 1; jcol <= n; jcol++ )
  {
/*
  Find the maximum element in column I.
*/
    piv = c8_abs ( a[jcol-1+(jcol-1)*n] );
    ipiv = jcol;
    for ( i = jcol+1; i <= n; i++ )
    {
      if ( piv < c8_abs ( a[i-1+(jcol-1)*n] ) )
      {
        piv = c8_abs ( a[i-1+(jcol-1)*n] );
        ipiv = i;
      }
    }

    if ( piv == 0.0 )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "C8MAT_FSS - Fatal error!\n" );
      fprintf ( stderr, "  Zero pivot on step %d\n", jcol );
      exit ( 1 );
    }
/*
  Switch rows JCOL and IPIV, and X.
*/
    if ( jcol != ipiv )
    {
      for ( j = 1; j <= n; j++ )
      {
        t                 = a[jcol-1+(j-1)*n];
        a[jcol-1+(j-1)*n] = a[ipiv-1+(j-1)*n];
        a[ipiv-1+(j-1)*n] = t;
      }
      for ( j = 0; j < nb; j++ )
      {
        t            = x[jcol-1+j*n];
        x[jcol-1+j*n] = x[ipiv-1+j*n];
        x[ipiv-1+j*n] = t;
      }
    }
/*
  Scale the pivot row.
*/
    t = a[jcol-1+(jcol-1)*n];
    a[jcol-1+(jcol-1)*n] = 1.0;
    for ( j = jcol+1; j <= n; j++ )
    {
      a[jcol-1+(j-1)*n] = a[jcol-1+(j-1)*n] / t;
    }
    for ( j = 0; j < nb; j++ )
    {
      x[jcol-1+j*n] = x[jcol-1+j*n] / t;
    }
/*
  Use the pivot row to eliminate lower entries in that column.
*/
    for ( i = jcol+1; i <= n; i++ )
    {
      if ( a[i-1+(jcol-1)*n] != 0.0 )
      {
        t = - a[i-1+(jcol-1)*n];
        a[i-1+(jcol-1)*n] = 0.0;
        for ( j = jcol+1; j <= n; j++ )
        {
          a[i-1+(j-1)*n] = a[i-1+(j-1)*n] + t * a[jcol-1+(j-1)*n];
        }
        for ( j = 0; j < nb; j++ )
        {
          x[i-1+j*n] = x[i-1+j*n] + t * x[jcol-1+j*n];
        }
      }
    }
  }
/*
  Back solve.
*/
  for ( jcol = n; 2 <= jcol; jcol-- )
  {
    for ( i = 1; i < jcol; i++ )
    {
      for ( j = 0; j < nb; j++ )
      {
        x[i-1+j*n] = x[i-1+j*n] - a[i-1+(jcol-1)*n] * x[jcol-1+j*n];
      }
    }
  }

  return;
}
/******************************************************************************/

double complex *c8mat_fss_new ( int n, double complex a[], int nb, 
  double complex b[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_FSS_NEW factors and solves a system with multiple right hand sides.

  Discussion:

    This routine uses partial pivoting, but no pivot vector is required.

    A C8MAT is a doubly dimensioned array of C8 values, stored as a vector
    in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the matrix.
    N must be positive.

    Input/output, double complex A[N*N].
    On input, A is the coefficient matrix of the linear system.
    On output, A is in unit upper triangular form, and
    represents the U factor of an LU factorization of the
    original coefficient matrix.

    Input, int NB, the number of right hand sides.

    Input, double complex B[N*NB], the right hand sides of the linear systems.

    Output, double complex C8MAT_FSS_NEW[N*NB], the solutions of the 
    linear systems.
*/
{
  int i;
  int ipiv;
  int j;
  int jcol;
  double piv;
  double complex t;
  double complex *x;

  x = ( double complex * ) malloc ( n * nb * sizeof ( double complex ) );

  for ( j = 0; j < nb; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      x[i+j*n] = b[i+j*n];
    }
  }
  for ( jcol = 1; jcol <= n; jcol++ )
  {
/*
  Find the maximum element in column I.
*/
    piv = c8_abs ( a[jcol-1+(jcol-1)*n] );
    ipiv = jcol;
    for ( i = jcol + 1; i <= n; i++ )
    {
      if ( piv < c8_abs ( a[i-1+(jcol-1)*n] ) )
      {
        piv = c8_abs ( a[i-1+(jcol-1)*n] );
        ipiv = i;
      }
    }

    if ( piv == 0.0 )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "C8MAT_FSS_NEW - Fatal error!\n" );
      fprintf ( stderr, "  Zero pivot on step %d\n", jcol );
      exit ( 1 );
    }
/*
  Switch rows JCOL and IPIV, and X.
*/
    if ( jcol != ipiv )
    {
      for ( j = 1; j <= n; j++ )
      {
        t                 = a[jcol-1+(j-1)*n];
        a[jcol-1+(j-1)*n] = a[ipiv-1+(j-1)*n];
        a[ipiv-1+(j-1)*n] = t;
      }
      for ( j = 0; j < nb; j++ )
      {
        t            = x[jcol-1+j*n];
        x[jcol-1+j*n] = x[ipiv-1+j*n];
        x[ipiv-1+j*n] = t;
      }
    }
/*
  Scale the pivot row.
*/
    t = a[jcol-1+(jcol-1)*n];
    a[jcol-1+(jcol-1)*n] = 1.0;
    for ( j = jcol + 1; j <= n; j++ )
    {
      a[jcol-1+(j-1)*n] = a[jcol-1+(j-1)*n] / t;
    }
    for ( j = 0; j < nb; j++ )
    {
      x[jcol-1+j*n] = x[jcol-1+j*n] / t;
    }
/*
  Use the pivot row to eliminate lower entries in that column.
*/
    for ( i = jcol+1; i <= n; i++ )
    {
      if ( a[i-1+(jcol-1)*n] != 0.0 )
      {
        t = - a[i-1+(jcol-1)*n];
        a[i-1+(jcol-1)*n] = 0.0;
        for ( j = jcol+1; j <= n; j++ )
        {
          a[i-1+(j-1)*n] = a[i-1+(j-1)*n] + t * a[jcol-1+(j-1)*n];
        }
        for ( j = 0; j < nb; j++ )
        {
          x[i-1+j*n] = x[i-1+j*n] + t * x[jcol-1+j*n];
        }
      }
    }
  }
/*
  Back solve.
*/
  for ( jcol = n; 2 <= jcol; jcol-- )
  {
    for ( i = 1; i < jcol; i++ )
    {
      for ( j = 0; j < nb; j++ )
      {
        x[i-1+j*n] = x[i-1+j*n] - a[i-1+(jcol-1)*n] * x[jcol-1+j*n];
      }
    }
  }

  return x;
}
/******************************************************************************/

double complex *c8mat_identity_new ( int n )

/******************************************************************************/
/*
  Purpose:

    C8MAT_IDENTITY_NEW sets a C8MAT to the identity.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the matrix.

    Output, double complex C8MAT_IDENTITY_NEW[N*N], the matrix.
*/
{
  double complex *a;
  int i;
  int j;

  a = ( double complex * ) malloc ( n * n * sizeof ( double complex ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      if ( i == j )
      {
        a[i+j*n] = 1.0;
      }
      else
      {
        a[i+j*n] = 0.0;
      }
    }
  }
  return a;
}
/******************************************************************************/

double complex *c8mat_indicator_new ( int m, int n )

/******************************************************************************/
/*
  Purpose:

    C8MAT_INDICATOR_NEW returns the C8MAT indicator matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Output, double complex C8MAT_INDICATOR_NEW[M*N], the matrix.
*/
{
  double complex *a;
  int i;
  int j;

  a = ( double complex * ) malloc ( m * n * sizeof ( double complex ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      a[i+j*m] = i + j * I;
    }
  }
  return a;
}
/******************************************************************************/

void c8mat_minvm ( int n1, int n2, double complex a[], 
  double complex b[], double complex c[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_MINVM returns inverse(A) * B for C8MAT's.

  Discussion:

    A C8MAT is a doubly dimensioned array of C8 values, stored as a vector
    in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    05 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N1, N2, the order of the matrices.

    Input, double complex A[N1*N1], B[N1*N2], the matrices.

    Output, double complex C[N1*N2], the result, C = inverse(A) * B.
*/
{
  double complex *alu;

  alu = c8mat_copy_new ( n1, n1, a );

  c8mat_copy ( n1, n2, b, c );

  c8mat_fss ( n1, alu, n2, c );
 
  free ( alu );

  return;
}
/******************************************************************************/

double complex *c8mat_minvm_new ( int n1, int n2, double complex a[], 
  double complex b[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_MINVM_NEW returns inverse(A) * B for C8MAT's.

  Discussion:

    A C8MAT is a doubly dimensioned array of C8 values, stored as a vector
    in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N1, N2, the order of the matrices.

    Input, double complex A[N1*N1], B[N1*N2], the matrices.

    Output, double complex C8MAT_MINVM_NEW[N1*N2], the result, C = inverse(A) * B.
*/
{
  double complex *alu;
  double complex *c;

  alu = c8mat_copy_new ( n1, n1, a );

  c = c8mat_fss_new ( n1, alu, n2, b );
 
  free ( alu );

  return c;
}
/******************************************************************************/

void c8mat_mm ( int n1, int n2, int n3, double complex a[], double complex b[], 
  double complex c[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_MM multiplies two C8MAT's.

  Discussion:

    A C8MAT is a doubly dimensioned array of C8 values, stored as a vector
    in column-major order.

    For this routine, the result is returned as the function value.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N1, N2, N3, the order of the matrices.

    Input, double complex A[N1*N2], double complex B[N2*N3], 
    the matrices to multiply.

    Output, double complex C[N1*N3], the product matrix C = A * B.
*/
{
  double complex *c1;
  int i;
  int j;
  int k;

  c1 = ( double complex * ) malloc ( n1 * n3 * sizeof ( double complex ) );

  for ( i = 0; i < n1; i ++ )
  {
    for ( j = 0; j < n3; j++ )
    {
      c1[i+j*n1] = 0.0;
      for ( k = 0; k < n2; k++ )
      {
        c1[i+j*n1] = c1[i+j*n1] + a[i+k*n1] * b[k+j*n2];
      }
    }
  }

  c8mat_copy ( n1, n3, c1, c );

  free ( c1 );

  return;
}
/******************************************************************************/

double complex *c8mat_mm_new ( int n1, int n2, int n3, double complex a[], 
  double complex b[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_MM_NEW multiplies two C8MAT's.

  Discussion:

    A C8MAT is a doubly dimensioned array of C8 values, stored as a vector
    in column-major order.

    For this routine, the result is returned as the function value.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int N1, N2, N3, the order of the matrices.

    Input, double complex A[N1*N2], double complex B[N2*N3], 
    the matrices to multiply.

    Output, double complex C8MAT_MM_NEW[N1*N3], the product matrix C = A * B.
*/
{
  double complex *c;
  int i;
  int j;
  int k;

  c = ( double complex * ) malloc ( n1 * n3 * sizeof ( double complex ) );

  for ( i = 0; i < n1; i ++ )
  {
    for ( j = 0; j < n3; j++ )
    {
      c[i+j*n1] = 0.0;
      for ( k = 0; k < n2; k++ )
      {
        c[i+j*n1] = c[i+j*n1] + a[i+k*n1] * b[k+j*n2];
      }
    }
  }
  return c;
}
/******************************************************************************/

void c8mat_nint ( int m, int n, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_NINT rounds the entries of a C8MAT.

  Discussion:

    A C8MAT is an array of double complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns of A.

    Input/output, double complex A[M*N], the matrix to be NINT'ed.
*/
{
  int i;
  int j;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      a[i+j*m] = c8_nint ( a[i+j*m] );
    }
  }
  return;
}
/******************************************************************************/

double c8mat_norm_fro ( int m, int n, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_NORM_FRO returns the Frobenius norm of a C8MAT.

  Discussion:

    A C8MAT is an array of C8 values.

    The Frobenius norm is defined as

      C8MAT_NORM_FRO = sqrt (
        sum ( 1 <= I <= M ) Sum ( 1 <= J <= N ) |A(I,J)| )

    The matrix Frobenius-norm is not derived from a vector norm, but
    is compatible with the vector L2 norm, so that:

      c8vec_norm_l2 ( A*x ) <= c8mat_norm_fro ( A ) * c8vec_norm_l2 ( x ).

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the order of the matrix.

    Input, double complex A[M*N], the matrix.

    Output, double C8MAT_NORM_FRO, the Frobenius norm of A.
*/
{
  int i;
  int j;
  double value;

  value = 0.0;
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      value = value + pow ( creal ( a[i+j*m] ), 2 )
                    + pow ( cimag ( a[i+j*m] ), 2 );
    }
  }
  value = sqrt ( value );

  return value;
}
/******************************************************************************/

double c8mat_norm_l1 ( int m, int n, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_NORM_L1 returns the matrix L1 norm of a C8MAT.

  Discussion:

    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].

    The matrix L1 norm is defined as:

      C8MAT_NORM_L1 = max ( 1 <= J <= N )
        sum ( 1 <= I <= M ) abs ( A(I,J) ).

    The matrix L1 norm is derived from the vector L1 norm, and
    satisifies:

      c8vec_norm_l1 ( A * x ) <= c8mat_norm_l1 ( A ) * c8vec_norm_l1 ( x ).

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, the number of rows in A.

    Input, int N, the number of columns in A.

    Input, double complex A[M*N], the matrix whose L1 norm is desired.

    Output, double C8MAT_NORM_L1, the L1 norm of A.
*/
{
  double col_sum;
  int i;
  int j;
  double value;

  value = 0.0;

  for ( j = 0; j < n; j++ )
  {
    col_sum = 0.0;
    for ( i = 0; i < m; i++ )
    {
      col_sum = col_sum + c8_abs ( a[i+j*m] );
    }
    value = r8_max ( value, col_sum );
  }

  return value;
}
/******************************************************************************/

double c8mat_norm_li ( int m, int n, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_NORM_LI returns the L-infinity norm of a C8MAT.

  Discussion:

    A C8MAT is a doubly dimensioned array of C8 values, stored as a vector
    in column-major order.

    The matrix L-oo norm is defined as:

      C8MAT_NORM_LI =  max ( 1 <= I <= M ) sum ( 1 <= J <= N ) abs ( A(I,J) ).

    The matrix L-oo norm is derived from the vector L-oo norm,
    and satisifies:

      c8vec_norm_li ( A * x ) <= c8mat_norm_li ( A ) * c8vec_norm_li ( x ).

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the order of the matrix.

    Input, double complex A[M*N], the matrix.

    Output, double C8MAT_NORM_LI, the L-infinity norm of A.
*/
{
  int i;
  int j;
  double row_sum;
  double value;

  value = 0.0;
  for ( i = 0; i < m; i++ )
  {
    row_sum = 0.0;
    for ( j = 0; j < n; j++ )
    {
      row_sum = row_sum + c8_abs ( a[i+j*m] );
    }
    value = r8_max ( value, row_sum );
  }
  return value;
}
/******************************************************************************/

void c8mat_print ( int m, int n, double complex a[], char *title )

/******************************************************************************/
/*
  Purpose:

    C8MAT_PRINT prints a C8MAT.

  Discussion:

    A C8MAT is a matrix of double precision complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns in the matrix.

    Input, double complex A[M*N], the matrix.

    Input, char *TITLE, a title.
*/
{
  c8mat_print_some ( m, n, a, 1, 1, m, n, title );

  return;
}
/******************************************************************************/

void c8mat_print_some ( int m, int n, double complex a[], int ilo, int jlo, 
  int ihi, int jhi, char *title )

/******************************************************************************/
/*
  Purpose:

    C8MAT_PRINT_SOME prints some of a C8MAT.

  Discussion:

    A C8MAT is a matrix of double precision complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns in the matrix.

    Input, double complex A[M*N], the matrix.

    Input, int ILO, JLO, IHI, JHI, the first row and
    column, and the last row and column to be printed.

    Input, char *TITLE, a title.
*/
{
  double complex c;
  int i;
  int i2hi;
  int i2lo;
  int inc;
  int incx = 4;
  int j;
  int j2;
  int j2hi;
  int j2lo;

  printf ( "\n" );
  printf ( "%s\n", title );
/*
  Print the columns of the matrix, in strips of INCX.
*/
  for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + incx )
  {
    j2hi = j2lo + incx - 1;
    j2hi = i4_min ( j2hi, n );
    j2hi = i4_min ( j2hi, jhi );

    inc = j2hi + 1 - j2lo;

    printf ( "\n" );
    printf ( "  Col: " );
    for ( j = j2lo; j <= j2hi; j++ )
    {
      j2 = j + 1 - j2lo;
      printf ( "          %10d", j );
    }
    printf ( "\n" );
    printf ( "  Row\n" );
    printf ( "  ---\n" );
/*
  Determine the range of the rows in this strip.
*/
    i2lo = i4_max ( ilo, 1 );
    i2hi = i4_min ( ihi, m );

    for ( i = i2lo; i <= i2hi; i++ )
    {
/*
  Print out (up to) INCX entries in row I, that lie in the current strip.
*/
      for ( j2 = 1; j2 <= inc; j2++ )
      {
        j = j2lo - 1 + j2;
        c = a[i-1+(j-1)*m];
        printf ( "  %8g  %8g", creal ( c ), cimag ( c ) );
      }
      printf ( "\n" );
    }
  }
  return;
}
/******************************************************************************/

void c8mat_scale ( int m, int n, double complex alpha, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_SCALE scales a C8MAT by a complex scale factor.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, double complex ALPHA, the scale factor.

    Input/output, double complex A[M*N], the matrix to be scaled.
*/
{
  int i;
  int j;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      a[i+j*m] = a[i+j*m] * alpha;
    }
  }
  return;
}
/******************************************************************************/

void c8mat_scale_r8 ( int m, int n, double alpha, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8MAT_SCALE_R8 scales a C8MAT by a real scale factor

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, double ALPHA, the scale factor.

    Input/output, double complex A[M*N], the matrix to be scaled.
*/
{
  int i;
  int j;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      a[i+j*m] = a[i+j*m] * alpha;
    }
  }
  return;
}
/******************************************************************************/

double complex *c8mat_uniform_01 ( int m, int n, int *seed )

/******************************************************************************/
/*
  Purpose:

    C8MAT_UNIFORM_01 returns a unit pseudorandom C8MAT.

  Discussion:

    A C8MAT is a matrix of double precision complex values.

    The angles should be uniformly distributed between 0 and 2 * PI,
    the square roots of the radius uniformly distributed between 0 and 1.

    This results in a uniform distribution of values in the unit circle.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns in the matrix.

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, double complex C[M*N], the pseudorandom complex matrix.
*/
{
  double complex *c;
  int i;
  int j;
  double r;
  int k;
  double pi = 3.141592653589793;
  double theta;

  c = ( double complex * ) malloc ( m * n * sizeof ( double complex ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      k = *seed / 127773;

      *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

      if ( *seed < 0 )
      {
        *seed = *seed + 2147483647;
      }

      r = sqrt ( ( double ) ( *seed ) * 4.656612875E-10 );

      k = *seed / 127773;

      *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

      if ( *seed < 0 )
      {
        *seed = *seed + 2147483647;
      }

      theta = 2.0 * pi * ( ( double ) ( *seed ) * 4.656612875E-10 );

      c[i+j*m] = r * ( cos ( theta ) + sin ( theta ) * I );
    }
  }

  return c;
}
/******************************************************************************/

double complex *c8mat_zero_new ( int m, int n )

/******************************************************************************/
/*
  Purpose:

    C8MAT_ZERO_NEW returns a new zeroed C8MAT.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Output, double complex C8MAT_ZERO[M*N], the new zeroed matrix.
*/
{
  double complex *a;
  int i;
  int j;

  a = ( double complex * ) malloc ( m * n * sizeof ( double complex ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      a[i+j*m] = 0.0;
    }
  }
  return a;
}
/******************************************************************************/

void c8vec_copy ( int n, double complex a[], double complex b[] )

/******************************************************************************/
/*
  Purpose:

    C8VEC_COPY copies a C8VEC to another.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the vectors.

    Input, double complex A[N], the vector to be copied.

    Output, double complex B[N], the copy.
*/
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    b[i] = a[i];
  }
  return;
}
/******************************************************************************/

double complex *c8vec_copy_new ( int n, double complex a1[] )

/******************************************************************************/
/*
  Purpose:

    C8VEC_COPY_NEW copies a C8VEC to a "new" C8VEC.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    27 August 2008

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the vectors.

    Input, double complex A1[N], the vector to be copied.

    Output, double complex C8VEC_COPY_NEW[N], the copy of A1.
*/
{
  double complex *a2;
  int i;

  a2 = ( double complex * ) malloc ( n * sizeof ( double complex ) );

  for ( i = 0; i < n; i++ )
  {
    a2[i] = a1[i];
  }
  return a2;
}
/******************************************************************************/

void c8vec_nint ( int n, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8VEC_NINT rounds the entries of a C8VEC.

  Discussion:

    A C8VEC is a vector of double complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of components of the vector.

    Input/output, double complex A[N], the vector to be nint'ed.
*/
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    a[i] = c8_nint ( a[i] );
  }

  return;
}
/******************************************************************************/

double c8vec_norm_l2 ( int n, double complex a[] )

/******************************************************************************/
/*
  Purpose:

    C8VEC_NORM_L2 returns the L2 norm of a C8VEC.

  Discussion:

    The vector L2 norm is defined as:

      value = sqrt ( sum ( 1 <= I <= N ) conjg ( A(I) ) * A(I) ).

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in A.

    Input, double complex A[N], the vector whose L2 norm is desired.

    Output, double C8VEC_NORM_L2, the L2 norm of A.
*/
{
  int i;
  double value;

  value = 0.0;
  for ( i = 0; i < n; i++ )
  {
    value = value 
          + creal ( a[i] ) * creal ( a[i] ) 
          + cimag ( a[i] ) * cimag ( a[i] );
  }
  value = sqrt ( value );

  return value;
}
/******************************************************************************/

void c8vec_print ( int n, double complex a[], char *title )

/******************************************************************************/
/*
  Purpose:

    C8VEC_PRINT prints a C8VEC.

  Discussion:

    A C8VEC is a vector of double complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    27 January 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of components of the vector.

    Input, double complex A[N], the vector to be printed.

    Input, char *TITLE, a title.
*/
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );

  for ( i = 0; i < n; i++ )
  {
    fprintf ( stdout, "  %8d: %14f  %14f\n", i, creal( a[i] ), cimag ( a[i] ) );
  }

  return;
}
/******************************************************************************/

void c8vec_print_part ( int n, double complex a[], int max_print, char *title )

/******************************************************************************/
/*
  Purpose:

    C8VEC_PRINT_PART prints "part" of a C8VEC.

  Discussion:

    The user specifies MAX_PRINT, the maximum number of lines to print.

    If N, the size of the vector, is no more than MAX_PRINT, then
    the entire vector is printed, one entry per line.

    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
    followed by a line of periods suggesting an omission,
    and the last entry.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    09 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries of the vector.

    Input, double complex A[N], the vector to be printed.

    Input, int MAX_PRINT, the maximum number of lines
    to print.

    Input, char *TITLE, a title.
*/
{
  int i;

  if ( max_print <= 0 )
  {
    return;
  }

  if ( n <= 0 )
  {
    return;
  }

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );

  if ( n <= max_print )
  {
    for ( i = 0; i < n; i++ )
    {
      fprintf ( stdout, "  %8d: (%14f,  %14f)\n", i, creal ( a[i] ), cimag ( a[i] ) );
    }
  }
  else if ( 3 <= max_print )
  {
    for ( i = 0; i < max_print - 2; i++ )
    {
      fprintf ( stdout, "  %8d: (%14f,  %14f)\n", i, creal( a[i] ), cimag ( a[i] ) );
    }
    fprintf ( stdout, "  ........   ..............   ..............\n" );
    i = n - 1;
    fprintf ( stdout, "  %8d: (%14f,  %14f)\n", i, creal( a[i] ), cimag ( a[i] ) );
  }
  else
  {
    for ( i= 0; i < max_print - 1; i++ )
    {
      fprintf ( stdout, "  %8d: (%14f,  %14f)\n", i, creal( a[i] ), cimag ( a[i] ) );
    }
    i = max_print - 1;
    fprintf ( stdout, "  %8d: (%14f,  %14f )  ...more entries...\n", 
      i, creal( a[i] ), cimag ( a[i] ) );
  }

  return;
}
/******************************************************************************/

void c8vec_print_some ( int n, double complex a[], int i_lo, int i_hi, 
  char *title )

/******************************************************************************/
/*
  Purpose:

    C8VEC_PRINT_SOME prints "some" of a C8VEC.

  Discussion:

    A C8VEC is a vector of C8's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    03 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries of the vector.

    Input, double complex A[N], the vector to be printed.

    Input, integer I_LO, I_HI, the first and last indices to print.
    The routine expects 1 <= I_LO <= I_HI <= N.

    Input, char *TITLE, a title.
*/
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );
  for ( i = i4_max ( 1, i_lo ); i <= i4_min ( n, i_hi ); i++ )
  {
    fprintf ( stdout, "  %8d: (%14f, %14f)\n", i, creal( a[i] ), cimag ( a[i] ) );
  }

  return;
}
/******************************************************************************/

void c8vec_sort_a_l2 ( int n, double complex x[] )

/******************************************************************************/
/*
  Purpose:

    C8VEC_SORT_A_L2 ascending sorts a double complex array by L2 norm.

  Discussion:

    The L2 norm of A+Bi is sqrt ( A * A + B * B ).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    27 January 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, length of input array.

    Input/output, double complex X[N].
    On input, an unsorted array.
    On output, X has been sorted.
*/
{
  int i;
  int indx;
  int isgn;
  int j;
  double normsq_i;
  double normsq_j;
  double complex temp;

  i = 0;
  indx = 0;
  isgn = 0;
  j = 0;

  for ( ; ; )
  {
    sort_heap_external ( n, &indx, &i, &j, isgn );

    if ( 0 < indx )
    {
      temp = x[i-1];
      x[i-1] = x[j-1];
      x[j-1] = temp;
    }
    else if ( indx < 0 )
    {
      normsq_i = pow ( creal ( x[i-1] ), 2 )
               + pow ( cimag ( x[i-1] ), 2 );

      normsq_j = pow ( creal ( x[j-1] ), 2 )
               + pow ( cimag ( x[j-1] ), 2 );

      if ( normsq_i < normsq_j )
      {
        isgn = -1;
      }
      else
      {
        isgn = +1;
      }
    }
    else if ( indx == 0 )
    {
      break;
    }
  }

  return;
}
/******************************************************************************/

double complex *c8vec_spiral ( int n, int m, double complex c1, 
  double complex c2 )

/******************************************************************************/
/*
  Purpose:

    C8VEC_SPIRAL returns N points on a spiral between C1 and C2.

  Discussion:

    A C8VEC is a vector of C8's.

    Let the polar form of C1 be ( R1, T1 ) and the polar form of C2 
    be ( R2, T2 ) where, if necessary, we increase T2 by 2*PI so that T1 <= T2.
    
    Then the polar form of the I-th point C(I) is:

      R(I) = ( ( N - I     ) * R1 
             + (     I - 1 ) * R2 ) 
              / ( N    - 1 )

      T(I) = ( ( N - I     ) * T1 
             + (     I - 1 ) * ( T2 + M * 2 * PI ) ) 
             / ( N     - 1 )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of points on the spiral.

    Input, int M, the number of full circuits the 
    spiral makes.

    Input, double complex C1, C2, the first and last points 
    on the spiral.

    Output, double complex C8VEC_SPIRAL_NEW[N], the points.
*/
{
  double complex *c;
  int i;
  double r1;
  double r2;
  double ri;
  double r8_pi = 3.141592653589793;
  double t1;
  double t2;
  double ti;

  c = ( double complex * ) malloc ( n * sizeof ( double complex ) );

  r1 = c8_abs ( c1 );
  r2 = c8_abs ( c2 );

  t1 = c8_arg ( c1 );
  t2 = c8_arg ( c2 );

  if ( m == 0 )
  {
    if ( t2 < t1 )
    {
      t2 = t2 + 2.0 * r8_pi;
    }
  }
  else if ( 0 < m )
  {
    if ( t2 < t1 )
    {
      t2 = t2 + 2.0 * r8_pi;
    }
    t2 = t2 + ( double ) ( m ) * 2.0 * r8_pi;
  }
  else if ( m < 0 )
  {
    if ( t1 < t2 )
    {
      t2 = t2 - 2.0 * r8_pi;
    }
    t2 = t2 - ( double ) ( m ) * 2.0 * r8_pi;
  }

  for ( i = 0; i < n; i++ )
  {
    ri = ( ( double ) ( n - i - 1 ) * r1
         + ( double ) (     i     ) * r2 )
         / ( double ) ( n     - 1 );

    ti = ( ( double ) ( n - i - 1 ) * t1
         + ( double ) (     i     ) * t2 )
         / ( double ) ( n     - 1 );

    c[i] = polar_to_c8 ( ri, ti );
  }

  return c;
}
/******************************************************************************/

double complex *c8vec_uniform_01_new ( int n, int *seed )

/******************************************************************************/
/*
  Purpose:

    C8VEC_UNIFORM_01_NEW returns a unit pseudorandom C8VEC.

  Discussion:

    The angles should be uniformly distributed between 0 and 2 * PI,
    the square roots of the radius uniformly distributed between 0 and 1.

    This results in a uniform distribution of values in the unit circle.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    09 July 2011

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Second Edition,
    Springer, 1987,
    ISBN: 0387964673,
    LC: QA76.9.C65.B73.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, December 1986, pages 362-376.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation,
    edited by Jerry Banks,
    Wiley, 1998,
    ISBN: 0471134031,
    LC: T57.62.H37.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, Number 2, 1969, pages 136-143.

  Parameters:

    Input, int N, the number of values to compute.

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, double complex C8VEC_UNIFORM_01_NEW[N], the pseudorandom vector.
*/
{
  double complex *c;
  int i;
  int i4_huge = 2147483647;
  double r;
  int k;
  double pi = 3.141592653589793;
  double theta;

  if ( *seed == 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "C8VEC_UNIFORM_01_NEW - Fatal error!\n" );
    fprintf ( stderr, "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  c = ( double complex * ) malloc ( n * sizeof ( double complex ) );

  for ( i = 0; i < n; i++ )
  {
    k = *seed / 127773;

    *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

    if ( *seed < 0 )
    {
      *seed = *seed + i4_huge;
    }

    r = sqrt ( ( double ) ( *seed ) * 4.656612875E-10 );

    k = *seed / 127773;

    *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

    if ( *seed < 0 )
    {
      *seed = *seed + i4_huge;
    }

    theta = 2.0 * pi * ( ( double ) ( *seed ) * 4.656612875E-10 );

    c[i] = r * cos ( theta ) + r * sin ( theta ) * I;
  }

  return c;
}
/******************************************************************************/

double complex *c8vec_unity ( int n )

/******************************************************************************/
/*
  Purpose:

    C8VEC_UNITY returns the N roots of unity in a C8VEC.

  Discussion:

    A C8VEC is a vector of C8's.

    X(1:N) = exp ( 2 * PI * (0:N-1) / N )

    X(1:N)^N = ( (1,0), (1,0), ..., (1,0) ).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    07 September 2010

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of elements of A.

    Output, double complex C8VEC_UNITY[N], the N roots of unity.
*/
{
  double complex *a;
  int i;
  double pi = 3.141592653589793;
  double theta;

  a = ( double complex * ) malloc ( n * sizeof ( double complex ) );

  for ( i = 0; i < n; i++ )
  {
    theta = pi * ( double ) ( 2 * i ) / ( double ) ( n );
    a[i] = cos ( theta ) + sin ( theta ) * I;
  }

  return a;
}
/******************************************************************************/

double complex cartesian_to_c8 ( double x, double y )

/******************************************************************************/
/*
  Purpose:

    CARTESIAN_TO_C8 converts a Cartesian form to a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double X, Y, the Cartesian form.

    Output, double complex CARTESIAN_TO_C8, the complex number.
*/
{
  double complex c;

  c = x + y * I;

  return c;
}
/******************************************************************************/

double complex polar_to_c8 ( double r, double theta )

/******************************************************************************/
/*
  Purpose:

    POLAR_TO_C8 converts a polar form to a C8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, double R, THETA, the polar form.

    Output, double complex POLAR_TO_C8, the complex number.
*/
{
  double complex c;

  c = r * cos ( theta ) + r * sin ( theta ) * I;

  return c;
}
/******************************************************************************/

double complex r8_csqrt ( double x )

/******************************************************************************/
/*
  Purpose:

    R8_CSQRT returns the complex square root of an R8.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    24 October 2005

  Author:

    John Burkardt

  Parameters:

    Input, double X, the number whose square root is desired.

    Output, double complex R8_CSQRT, the square root of X:
*/
{
  double argument;
  double magnitude;
  double pi = 3.141592653589793;
  double complex value;

  if ( 0.0 < x )
  {
    magnitude = x;
    argument = 0.0;
  }
  else if ( 0.0 == x )
  {
    magnitude = 0.0;
    argument = 0.0;
  }
  else if ( x < 0.0 )
  {
    magnitude = -x;
    argument = pi;
  }

  magnitude = sqrt ( magnitude );
  argument = argument / 2.0;

  value = magnitude * ( double complex ) ( cos ( argument ), sin ( argument ) );

  return value;
}
/******************************************************************************/

void r8poly2_root ( double a, double b, double c, double complex *r1,
  double complex *r2 )

/******************************************************************************/
/*
  Purpose:

    R8POLY2_ROOT returns the two roots of a quadratic polynomial.

  Discussion:

    The polynomial has the form:

      A * X^2 + B * X + C = 0

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    23 October 2005

  Parameters:

    Input, double A, B, C, the coefficients of the polynomial.
    A must not be zero.

    Output, double complex *R1, *R2, the roots of the polynomial, which
    might be real and distinct, real and equal, or complex conjugates.
*/
{
  double disc;
  double complex q;

  if ( a == 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8POLY2_ROOT - Fatal error!\n" );
    fprintf ( stderr, "  The coefficient A is zero.\n" );
    exit ( 1 );
  }

  disc = b * b - 4.0 * a * c;
  q = -0.5 * ( b + r8_sign ( b ) * r8_csqrt ( disc ) );
  *r1 = q / a;
  *r2 = c / q;

  return;
}
/******************************************************************************/

void r8poly3_root ( double a, double b, double c, double d,
  double complex *r1, double complex *r2, double complex *r3 )

/******************************************************************************/
/*
  Purpose:

    R8POLY3_ROOT returns the three roots of a cubic polynomial.

  Discussion:

    The polynomial has the form

      A * X^3 + B * X^2 + C * X + D = 0

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 August 2011

  Parameters:

    Input, double A, B, C, D, the coefficients of the polynomial.
    A must not be zero.

    Output, double complex *R1, *R2, *R3, the roots of the polynomial, which
    will include at least one real root.
*/
{
  double complex i;
  double pi = 3.141592653589793;
  double q;
  double r;
  double s1;
  double s2;
  double temp;
  double theta;

  if ( a == 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8POLY3_ROOT - Fatal error!\n" );
    fprintf ( stderr, "  A must not be zero.\n" );
    exit ( 1 );
  }

  i = ( double complex ) ( 0.0, 1.0 );

  q = ( pow ( b / a, 2 ) - 3.0 * ( c / a ) ) / 9.0;

  r = ( 2.0 * pow ( b / a, 3 ) - 9.0 * ( b / a ) * ( c / a )
      + 27.0 * ( d / a ) ) / 54.0;

  if ( r * r < q * q * q )
  {
    theta = acos ( r / sqrt ( pow ( q, 3 ) ) );
    *r1 = - 2.0 * sqrt ( q ) * cos (   theta              / 3.0 );
    *r2 = - 2.0 * sqrt ( q ) * cos ( ( theta + 2.0 * pi ) / 3.0 );
    *r3 = - 2.0 * sqrt ( q ) * cos ( ( theta + 4.0 * pi ) / 3.0 );
  }
  else if ( q * q * q <= r * r )
  {
    temp = - r + sqrt ( r * r - q * q * q );
    s1 = r8_sign ( temp ) * pow ( r8_abs ( temp ), 1.0 / 3.0 );

    temp = - r - sqrt ( r * r - q * q * q );
    s2 = r8_sign ( temp ) * pow ( r8_abs ( temp ), 1.0 / 3.0 );

    *r1 = s1 + s2;
    *r2 = -0.5 * ( s1 + s2 ) + i * 0.5 * sqrt ( 3.0 ) * ( s1 - s2 );
    *r3 = -0.5 * ( s1 + s2 ) - i * 0.5 * sqrt ( 3.0 ) * ( s1 - s2 );
  }

  *r1 = *r1 - b / ( 3.0 * a );
  *r2 = *r2 - b / ( 3.0 * a );
  *r3 = *r3 - b / ( 3.0 * a );

  return;
}
/******************************************************************************/

void r8poly4_root ( double a, double b, double c, double d, double e,
  double complex *r1, double complex *r2, double complex *r3,
  double complex *r4 )

/******************************************************************************/
/*
  Purpose:

    R8POLY4_ROOT returns the four roots of a quartic polynomial.

  Discussion:

    The polynomial has the form:

      A * X^4 + B * X^3 + C * X^2 + D * X + E = 0

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 August 2011

  Parameters:

    Input, double A, B, C, D, the coefficients of the polynomial.
    A must not be zero.

    Output, double complex *R1, *R2, *R3, *R4, the roots of the polynomial.
*/
{
  double a3;
  double a4;
  double b3;
  double b4;
  double c3;
  double c4;
  double d3;
  double d4;
  double complex p;
  double complex q;
  double complex r;
  double complex zero;

  zero = 0.0;

  if ( a == 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8POLY4_ROOT - Fatal error!\n" );
    fprintf ( stderr, "  A must not be zero.\n" );
    exit ( 1 );
  }

  a4 = b / a;
  b4 = c / a;
  c4 = d / a;
  d4 = e / a;
/*
  Set the coefficients of the resolvent cubic equation.
*/
  a3 = 1.0;
  b3 = -b4;
  c3 = a4 * c4 - 4.0 * d4;
  d3 = -a4 * a4 * d4 + 4.0 * b4 * d4 - c4 * c4;
/*
  Find the roots of the resolvent cubic.
*/
  r8poly3_root ( a3, b3, c3, d3, r1, r2, r3 );
/*
  Choose one root of the cubic, here R1.

  Set R = sqrt ( 0.25 * A4^2 - B4 + R1 )
*/
  r = c8_sqrt ( 0.25 * a4 * a4 - b4  + (*r1) );

  if ( creal ( r ) != 0.0 || cimag ( r ) != 0.0 )
  {
    p = c8_sqrt ( 0.75 * a4 * a4 - r * r - 2.0 * b4
        + 0.25 * ( 4.0 * a4 * b4 - 8.0 * c4 - a4 * a4 * a4 ) / r );

    q = c8_sqrt ( 0.75 * a4 * a4 - r * r - 2.0 * b4
        - 0.25 * ( 4.0 * a4 * b4 - 8.0 * c4 - a4 * a4 * a4 ) / r );
  }
  else
  {
    p = c8_sqrt ( 0.75 * a4 * a4 - 2.0 * b4
      + 2.0 * c8_sqrt ( (*r1) * (*r1) - 4.0 * d4 ) );

    q = c8_sqrt ( 0.75 * a4 * a4 - 2.0 * b4
      - 2.0 * c8_sqrt ( (*r1) * (*r1) - 4.0 * d4 ) );
  }
/*
  Set the roots.
*/
  *r1 = -0.25 * a4 + 0.5 * r + 0.5 * p;
  *r2 = -0.25 * a4 + 0.5 * r - 0.5 * p;
  *r3 = -0.25 * a4 - 0.5 * r + 0.5 * q;
  *r4 = -0.25 * a4 - 0.5 * r - 0.5 * q;

  return;
}

