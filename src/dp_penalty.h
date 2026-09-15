#ifndef DP_PENALTY_H
#define DP_PENALTY_H 1

/**
 * Warping penalties understood by the dynamic-programming solvers, i.e. by
 * the pen argument of \c DP() and \c DynamicProgrammingQ2().
 *
 * A dynamic program can only carry a penalty that is additive over the edges
 * of the path, i.e. one of the form \f$\int p(\dot\gamma(t))\,dt\f$, so only
 * these three are available.  The other two penalties offered by the RBFGS
 * solvers are not of that form and have no DP counterpart:
 *
 *  - roughness, \f$\int \ddot\gamma(t)^2\,dt\f$, vanishes inside every edge of
 *    a piecewise-linear path and concentrates on the knots, so no edge weight
 *    can represent it;
 *  - geodesic, \f$\arccos\left(\int \psi(t)\,dt\right)^2\f$, is a nonlinear
 *    function of a global integral, so it is not additive either.
 */
#define DP_PEN_NONE  0
#define DP_PEN_L2GAM 1
#define DP_PEN_L2PSI 2

#endif /* DP_PENALTY_H */
