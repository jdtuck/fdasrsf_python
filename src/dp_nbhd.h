#ifndef DP_NBHD_H
#define DP_NBHD_H 1

#include <stddef.h>

typedef int Pair[2];

Pair * dp_generate_nbhd(size_t nbhd_dim, size_t * nbhd_count);

#endif  /* DP_NBHD_H */

