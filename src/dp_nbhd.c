#include "dp_nbhd.h"

#include <stdio.h>
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
 *
 * @return Number of elements in the set.
 */
static size_t compute_nbhd_count(size_t n) {

    int * states = malloc((n + 1) * sizeof(*states));
    if(states == NULL)
    {
    	fprintf(stderr, "Error allocating memory in compute_nbhd_count\n");
    	abort();
    }

    for(size_t i = 0; i < n + 1; states[i++] = -1);

    size_t an = compute_nbhd_count_rec(n, states);

    free(states);

    return an;
}

/**
 * @brief Creates the nbhd grid.
 *
 * @param[in] nbhd_dim Number of points in each grid axis.
 * @param[out] nbhd_count Number of points in the set.
 *
 * @return Set of points.
 */
Pair * dp_generate_nbhd(size_t nbhd_dim, size_t * nbhd_count) {

	size_t k = 0;

    *nbhd_count = compute_nbhd_count(nbhd_dim) ;

    /* Allocate memory for the partition, using the exact amount of we can use
    ~60% of memory that if we use nbhd_dim^2 */
    Pair * dp_nbhd = malloc((*nbhd_count) * sizeof(*dp_nbhd));
    if(dp_nbhd == NULL)
    {
    	fprintf(stderr, "Error allocating memory in dp_generate_nbhd\n");
    	abort();
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
