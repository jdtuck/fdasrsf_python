#include "dp_nbhd.h"

#include <stdlib.h>

/**
 * @brief Greatest common divisor.
 *
 * Computes the greatest common divisor between a and b using the Euclidean
 * algorithm.
 *
 * @param[in] a First positive number.
 * @param[in] b Second positive number.
 *
 * @return Greatest common divisor of @a a and @a b.
 */
static unsigned int gcd(unsigned int a, unsigned int b) {

	unsigned int temp;

    /* Swap if b > a */
    if(b > a) {
        temp = a;
        a = b;
        b = temp;
    }

    /* Iterative Euclidean algorithm */
    while (b != 0)
    {
        a %= b;
        temp = a;
        a = b;
        b = temp;
    }
    return a;
}

/**
 * @brief Computes the number of elements in the nbhd grid.
 *
 * This is the number of elements in the set
 * @f[
 *              \{ (i,j) : \text{gcd}(i,j) = 1 & 1 \leq i,j \leq n \}
 * @f]
 *
 * This number corresponds with the OEIS A018805 sequence and can be computed
 * using the following formula:
 * @f[
 *               a(n) = n^2 - \sum_{j=2}^n a(floor(n/j))
 * @f]
 *
 * @param[in] n Number of points in each axis of the grid.
 * @param[out] states Array of size @a n, where the computed number of
 *                    elements will be placed recursively for each number
 *                    less than @a n, in order to not repeat computations.
 *
 * @return Number of elements in the set for the input @a n.
 */
static size_t compute_nbhd_count_rec(size_t n, int * states) {

    if (states[n] != -1) {
        return states[n];
    }

    size_t an = n * n;

    for(size_t j = 2; j <= n; j++) {
        an -= compute_nbhd_count_rec(n / j, states);
    }

    states[n] = an;

    return an;
}

/**
 * @brief Computes the number of elements in the nbhd grid.
 *
 * This is the number of elements in the set
 * @f[
 *              \{ (i,j) : \text{gcd}(i,j) = 1 & 1 \leq i,j \leq n \}
 * @f]
 *
 * This number corresponds with the OEIS A018805 sequence and can be computed
 * using the following formula:
 * @f[
 *               a(n) = n^2 - \sum_{j=2}^n a(floor(n/j))
 * @f]
 *
 * @param[in] n Number of points in each axis of the grid.
 * @param[out] count Number of elements in the set, on success.
 *
 * @return 0 on success, -1 if the scratch buffer could not be allocated.
 */
static int compute_nbhd_count(size_t n, size_t * count) {

    int * states = malloc((n + 1) * sizeof(*states));
    if(states == NULL)
    {
    	return -1;
    }

    for(size_t i = 0; i < n + 1; states[i++] = -1);

    *count = compute_nbhd_count_rec(n, states);

    free(states);

    return 0;
}

/**
 * @brief Creates the nbhd grid.
 *
 * @param[in] nbhd_dim Number of points in each grid axis.
 * @param[out] nbhd_count Number of points in the set, or 0 on failure.
 *
 * @return Set of points, or NULL if it could not be allocated.  Callers are
 *         responsible for reporting the failure: this is linked into a MEX
 *         file, where aborting would take down the host process.
 */
Pair * dp_generate_nbhd(size_t nbhd_dim, size_t * nbhd_count) {

	size_t k = 0;

    *nbhd_count = 0;

    if(compute_nbhd_count(nbhd_dim, nbhd_count) != 0)
    {
    	return NULL;
    }

    /* Allocate memory for the partition, using the exact amount of we can use
    ~60% of memory that if we use nbhd_dim^2 */
    Pair * dp_nbhd = malloc((*nbhd_count) * sizeof(*dp_nbhd));
    if(dp_nbhd == NULL)
    {
    	*nbhd_count = 0;
    	return NULL;
    }

    for(size_t i = 1; i <= nbhd_dim; i++) {
        for(size_t j = 1; j <= nbhd_dim; j++) {
            /* If irreducible fraction add as a coordinate */
            if (gcd(i, j) == 1) {
                dp_nbhd[k][0] = i;
                dp_nbhd[k][1] = j;
                k++;
            }
        }
    }

    return dp_nbhd;
}
