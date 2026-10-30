#include <iostream>
#include "armadillo"
#include "rbfgs.h"

using namespace arma;
using namespace std;

class rlbfgs {
    // Riemannian limited-memory BFGS for the warping that aligns q2 to q1, as
    // in Huang, Gallivan, Srivastava and Absil, "Riemannian optimization for
    // registration of curves in elastic shape analysis", restricted to open
    // functions (no rotation and no shift).  The iterate is l = sqrt(gamma')
    // on the unit sphere of L2[0,1], gamma = cumtrapz(l^2), and the cost is
    //   ||q1 - q2(gamma) l||^2 + lam*penalty(l) + omega*B(l),
    // where B is the barrier of their eq. (15), whose weight omega starts at
    // 1/100 and is cut by 0.8 each iteration.  Steps are taken with the
    // exponential map, and the stored BFGS pairs are moved to the new tangent
    // space by parallel translation (their Lemma 1).
    private:
        vec time;      // time
        vec q1;        // srvf1
        vec q2;        // srvf2
        vec q2M;       // second derivatives of the cubic spline of q2
        uword T;         // size of time
        double lamScale; // converts lam to the unit L2 norm SRVFs
        struct options{
            double tolgradnorm;
            int memory;
            int ls_max_steps;
            int maxiter;
            int miniter;
            double tolrelcost;
            int relcostcount;
            double omega0;
            double omegarate;
        };

        // the cost split into the part that stays fixed (data and penalty) and
        // the barrier, whose weight changes each iteration
        struct costparts{
            double fixedcost;
            double barrier;
            vec fixedgrad;
            vec barriergrad;
        };

  public:
        vec gammaOpt;
        vec q2Opt;
        double cost;
        // why the solver stopped: 1 step size, 2 gradient norm, 3 iterations,
        // 4 relative change of the cost
        int stopReason;
        int iterations;
        double gradNorm;
        // constructor
        rlbfgs(vec q1i, vec q2i, vec timei) {
        time = timei;

        T = timei.n_elem;

        // unit L2 norm on time rather than unit Euclidean norm, so the cost,
        // its gradient and so tolgradnorm do not scale with the grid size
        q1 = q1i / norm2(q1i);
        q2 = q2i / norm2(q2i);

        // the data term used to be that of the unit Euclidean norm SRVFs;
        // lamScale keeps lam weighing the penalty against it as before
        lamScale = 1 / (norm2(normalise(q1i, 2)) * norm2(normalise(q2i, 2)));

        // q2 is composed with gamma, so it is represented by an interpolatory
        // cubic spline, which makes the cost twice differentiable as the
        // convergence theory of the quasi-Newton method needs
        q2M = spline_second_derivs(q2);
        }

        // gam0, when not empty, is the warping to start from instead of the
        // identity, sampled on time
        void solve(int maxiter=30, double lam=0.0, int penalty=0, vec gam0=vec()){
            lam = lam * lamScale;
            options option;
            // a safeguard; the paper's own test is on the change of the cost
            option.tolgradnorm = 1e-10;
            // number of previous iterations the program remembers
            option.memory = 30;
            option.ls_max_steps = 25;
            option.maxiter = maxiter;
            // the paper stops once the relative change of the cost has stayed
            // below 1e-3 for more than 5 iterations, after at least 10
            option.miniter = 10;
            option.tolrelcost = 1e-3;
            option.relcostcount = 5;
            option.omega0 = 1.0/100;
            option.omegarate = 0.8;
            vec l = arma::ones(T);
            if (!gam0.is_empty()){
                l = srvf_of_warp(gam0);
            }
            l = l / norm2(l);

            arma::field<vec> sHistory(option.memory);
            arma::field<vec> yHistory(option.memory);
            vec rhoHistory(option.memory);

            // number of stored pairs
            int j = 0;
            // Total number of BFGS iterations
            int k = 0;
            // scaling of initial matrix, Barzilai-Borwein
            double scaleFactor = 1;
            int stop = 0;
            int smallchange = 0;
            int failedsteps = 0;
            double omega = option.omega0;

            costparts cur = costgrad(l, lam, penalty);
            double f = cur.fixedcost + omega * cur.barrier;
            vec g = riemannian_grad(l, cur, omega);
            double gnorm = norm2(g);

            while (true){
                if (gnorm <= option.tolgradnorm){
                    stop = 2;
                    break;
                }
                if (k >= option.maxiter){
                    stop = 3;
                    break;
                }

                // compute BFGS direction; fall back to steepest descent if the
                // quasi-Newton direction is not a descent direction
                vec p = getDirection(g, sHistory, yHistory, rhoHistory, scaleFactor, j);
                double df0 = inner(g, p);
                if (df0 >= 0){
                    j = 0;
                    scaleFactor = 1;
                    p = -g;
                    df0 = -gnorm*gnorm;
                }

                // Armijo backtracking along the exponential map
                double alpha = 1;
                vec lnew = exp(l, p, alpha);
                costparts nxt = costgrad(lnew, lam, penalty);
                double fnew = nxt.fixedcost + omega * nxt.barrier;
                int evals = 1;
                while (arma::any(lnew <= 0) || !std::isfinite(fnew) ||
                       fnew > f + 1e-4 * alpha * df0){
                    alpha *= 0.5;
                    lnew = exp(l, p, alpha);
                    nxt = costgrad(lnew, lam, penalty);
                    fnew = nxt.fixedcost + omega * nxt.barrier;
                    evals += 1;
                    if (evals >= option.ls_max_steps) break;
                }
                if (arma::any(lnew <= 0) || !std::isfinite(fnew) || fnew > f){
                    // no acceptable step: restart the memory once, then stop
                    failedsteps += 1;
                    if (failedsteps >= 2 || j == 0){
                        stop = 1;
                        break;
                    }
                    j = 0;
                    scaleFactor = 1;
                    continue;
                }
                failedsteps = 0;

                // the gradient at the new point with this iteration's omega,
                // so that the BFGS pair is that of one cost function
                vec gnew = riemannian_grad(lnew, nxt, omega);

                // parallel translation of tangent vectors from l to lnew
                vec step = alpha * p;
                vec sk = transp(l, lnew, step);
                vec yk = gnew - transp(l, lnew, g);
                for (int i = 0; i < j; i++){
                    sHistory(i) = transp(l, lnew, sHistory(i));
                    yHistory(i) = transp(l, lnew, yHistory(i));
                }

                // cautious update
                double inner_sk_yk = inner(sk, yk);
                double inner_sk_sk = inner(sk, sk);
                if (inner_sk_sk > 0 && inner_sk_yk / inner_sk_sk >= strict_inc_func(gnorm)){
                    if (j >= option.memory){
                        for (int i = 0; i < option.memory - 1; i++){
                            sHistory(i) = sHistory(i+1);
                            yHistory(i) = yHistory(i+1);
                            rhoHistory(i) = rhoHistory(i+1);
                        }
                        j = option.memory - 1;
                    }
                    sHistory(j) = sk;
                    yHistory(j) = yk;
                    rhoHistory(j) = 1 / inner_sk_yk;
                    scaleFactor = inner_sk_yk / inner(yk, yk);
                    j += 1;
                }

                // relative change of the cost, both values at this omega
                double relchange = std::abs(f - fnew) / std::max(std::abs(f), 1e-300);
                smallchange = (relchange < option.tolrelcost) ? smallchange + 1 : 0;

                l = lnew;
                cur = nxt;
                k += 1;

                // the barrier fades as the iterations proceed
                omega *= option.omegarate;
                f = cur.fixedcost + omega * cur.barrier;
                g = riemannian_grad(l, cur, omega);
                gnorm = norm2(g);

                if (k >= option.miniter && smallchange > option.relcostcount){
                    stop = 4;
                    break;
                }
            }

            vec gam = cumtrapz(time, arma::pow(l, 2));
            gammaOpt = (gam - gam.min()) / (gam.max() - gam.min());
            vec Q, Qdot;
            spline_eval(q2, q2M, gam, Q, Qdot);
            q2Opt = Q % l;
            // the cost of the warping returned, without the barrier
            cost = cur.fixedcost;
            stopReason = stop;
            iterations = k;
            gradNorm = gnorm;
        }

        // Value of the penalty term at the warping function h.  h is the
        // total warping accumulated from the identity, not an increment: each
        // of the penalties below measures h against the identity and so is
        // only meaningful for the cumulative warping.
        double penalty_cost(vec h, int penalty){
            double pen = 0;

            // roughness
            if (penalty == 0){
                vec time1 = arma::linspace<vec>(0,1,h.n_elem);
                vec b = arma::diff(time1);
                double binsize = mean(b);
                vec hdot = gradient(arma::pow(h, 2), binsize);
                arma::mat pen1 = arma::trapz(time1, arma::pow(hdot, 2));
                pen = pen1(0);
            }
            // l2gam
            if (penalty == 1){
                vec tmp = arma::ones(T);
                pen = normL2(arma::pow(h,2)-tmp);
                pen = pow(pen, 2);
            }
            // l2psi
            if (penalty == 2){
                vec tmp = arma::ones(T);
                pen = normL2(h-tmp);
                pen = pow(pen, 2);
            }
            // geodesic
            if (penalty == 3){
                vec time1 = arma::linspace<vec>(0,1,h.n_elem);
                arma::mat pen1 = arma::trapz(time1, h);
                double q1dotq2 = pen1(0);
                if (q1dotq2 > 1){
                    q1dotq2 = 1;
                } else if (q1dotq2 < -1)
                {
                    q1dotq2 = -1;
                }
                pen = pow(real(acos(q1dotq2)),2);
            }

            return pen;
        }

        // Partial derivatives of penalty_cost with respect to each sample of h.
        // These differentiate the discretised penalties above exactly, so they
        // agree with penalty_cost on the grid and not merely in the continuum
        // limit, which matters because the roughness penalty differentiates to
        // a second difference.
        vec penalty_grad(vec h, int penalty){
            vec time1 = arma::linspace<vec>(0,1,h.n_elem);
            vec w = trapz_weights(time1);
            vec dP = arma::zeros(T);

            // roughness: pen = trapz(d(h^2)/dt ^ 2), differentiated through
            // gradient() and then through the square
            if (penalty == 0){
                vec b = arma::diff(time1);
                double binsize = mean(b);
                vec hdot = gradient(arma::pow(h, 2), binsize);
                dP = 2 * (h % gradient_adjoint(2 * (w % hdot), binsize));
            }
            // l2gam: pen = ||h^2-1||^2
            if (penalty == 1){
                vec tmp = arma::ones(T);
                dP = 4 * (h % (arma::pow(h,2) - tmp)) % w;
            }
            // l2psi: pen = ||h-1||^2
            if (penalty == 2){
                vec tmp = arma::ones(T);
                dP = 2 * (h - tmp) % w;
            }
            // geodesic: pen = acos(<1,h>)^2, so the gradient is a multiple of
            // the constant function
            if (penalty == 3){
                arma::mat pen1 = arma::trapz(time1, h);
                double q1dotq2 = pen1(0);
                if (q1dotq2 > 1){
                    q1dotq2 = 1;
                } else if (q1dotq2 < -1)
                {
                    q1dotq2 = -1;
                }
                // acos(c)/sqrt(1-c^2) tends to 1 as c tends to 1, where both
                // factors vanish; h is positive and of unit norm, so c lies in
                // (0,1] and the singularity at c = -1 is out of reach
                double scale = 1;
                double c2 = pow(q1dotq2, 2);
                if (1 - c2 > 1e-12){
                    scale = acos(q1dotq2) / sqrt(1 - c2);
                }
                dP = -2 * scale * w;
            }

            return dP;
        }

        // Cost parts and their partial derivatives with respect to each
        // sample of l, differentiating the discretised cost exactly: gamma is
        // cumtrapz(l^2), q2 is evaluated on its cubic spline and the integrals
        // are trapezoidal sums
        costparts costgrad(const vec& l, double lam, int penalty){
            costparts c;
            vec w = trapz_weights(time);
            vec gam = cumtrapz(time, arma::pow(l, 2));
            vec Q, Qdot;
            spline_eval(q2, q2M, gam, Q, Qdot);
            vec r = q1 - Q % l;
            c.fixedcost = dot(w, r % r);
            // r depends on l directly and through gamma
            vec a = -2 * (w % r % Qdot % l);
            c.fixedgrad = -2 * (w % r % Q) + 2 * (l % cumtrapz_adjoint(time, a));

            if (lam != 0){
                c.fixedcost += lam * penalty_cost(l, penalty);
                c.fixedgrad += lam * penalty_grad(l, penalty);
            }

            // B = trapz((l^2 + 1/l^2) sqrt(1 + l^4)), eq. (15) of the paper;
            // it grows where gamma is nearly flat or vertical and is
            // symmetric under inverting gamma
            vec l2 = arma::pow(l, 2);
            vec l4 = arma::pow(l, 4);
            vec rt = arma::sqrt(1 + l4);
            vec u = l2 + 1 / l2;
            c.barrier = dot(w, u % rt);
            vec du = 2 * l - 2 / arma::pow(l, 3);
            vec drt = 2 * arma::pow(l, 3) / rt;
            c.barriergrad = w % (du % rt + u % drt);
            return c;
        }

        // Riemannian gradient under the L2 metric inner() uses: the partial
        // derivatives divided by the quadrature weights, projected onto the
        // tangent space of the sphere at l
        vec riemannian_grad(const vec& l, const costparts& c, double omega){
            vec w = trapz_weights(time);
            vec v = (c.fixedgrad + omega * c.barriergrad) / w;
            return v - inner(v, l) * l;
        }

        // l = sqrt(gamma') of the warping gam, sampled on time, from central
        // differences; on the spline cost this keeps a DP warping's cost (an
        // inversion of the trapezoid rule cell by cell oscillates instead),
        // and a small floor keeps l inside the barrier
        vec srvf_of_warp(vec gam){
            double hb = 1.0/(T-1);
            gam = (gam - gam(0)) / (gam(T-1) - gam(0));
            return arma::sqrt(arma::clamp(gradient(gam, hb), 1e-6, datum::inf));
        }

        // solves the tridiagonal system with sub-diagonal a, diagonal b and
        // super-diagonal c (a(0) and c(T-1) unused)
        vec thomas(vec a, vec b, vec c, vec d){
            uword n = b.n_elem;
            for (uword i = 1; i < n; i++){
                double m = a(i) / b(i-1);
                b(i) -= m * c(i-1);
                d(i) -= m * d(i-1);
            }
            vec x(n);
            x(n-1) = d(n-1) / b(n-1);
            for (int i = (int)n-2; i >= 0; i--){
                x(i) = (d(i) - c(i) * x(i+1)) / b(i);
            }
            return x;
        }

        // second derivatives of the natural cubic spline interpolating y on
        // the uniform grid time
        vec spline_second_derivs(const vec& y){
            vec M = arma::zeros(T);
            if (T < 3) return M;
            double hb = 1.0/(T-1);
            uword n = T - 2;
            vec a(n), b(n), c(n), d(n);
            for (uword i = 0; i < n; i++){
                a(i) = 1; b(i) = 4; c(i) = 1;
                d(i) = 6 * (y(i+2) - 2*y(i+1) + y(i)) / (hb*hb);
            }
            M(arma::span(1, T-2)) = thomas(a, b, c, d);
            return M;
        }

        // value and derivative at x of the natural cubic spline of y, whose
        // second derivatives are M
        void spline_eval(const vec& y, const vec& M, const vec& x, vec& val, vec& dval){
            double hb = 1.0/(T-1);
            uword n = x.n_elem;
            val.set_size(n);
            dval.set_size(n);
            for (uword i = 0; i < n; i++){
                double xi = std::min(std::max(x(i), 0.0), 1.0);
                uword k = std::min((uword) std::floor(xi / hb), T-2);
                double A = (k+1)*hb - xi;   // distance to the right node
                double B = xi - k*hb;       // distance to the left node
                double cA = y(k)/hb - M(k)*hb/6;
                double cB = y(k+1)/hb - M(k+1)*hb/6;
                val(i) = M(k)*A*A*A/(6*hb) + M(k+1)*B*B*B/(6*hb) + cA*A + cB*B;
                dval(i) = -M(k)*A*A/(2*hb) + M(k+1)*B*B/(2*hb) - cA + cB;
            }
        }

        vec getDirection(vec g, arma::field<vec>& sHistory, arma::field<vec>& yHistory, vec& rhoHistory, double scaleFactor, int j){
            vec q = g;
            vec inner_s_q = arma::zeros(j);

            for (int i = j; i > 0; i--) {
                inner_s_q(i-1) = rhoHistory(i-1) * inner(sHistory(i-1), q);
                q = q - inner_s_q(i-1) * yHistory(i-1);
            }

            vec r = scaleFactor * q;

            double om;
            for (int i=0; i < j; i++){
                om = rhoHistory(i) * inner(yHistory(i), r);
                r = r + (inner_s_q(i) - om) * sHistory(i);
            }

            return -1 * r;
        }

        // parallel translation of v from the tangent space at l to that at
        // lnew = exp(l, u), Lemma 1 of the paper
        vec transp(const vec& l, const vec& lnew, const vec& v){
            vec sum = l + lnew;
            double n2 = inner(sum, sum);
            if (n2 <= 0) return v;
            return v - (2 * inner(v, lnew) / n2) * sum;
        }

        double strict_inc_func(double t){
            // the cautious step needs a real function that has value 0 at t=0
            return 1e-4*t;
        }

        double normL2(vec f){
            double val1 = innerProdL2(f, f);
            double val = sqrt(val1);

            return val;
        }

        double innerProdL2(vec f1, vec f2){
            vec tmp = f1 % f2;
            arma::mat tmp1 = arma::trapz(time, tmp);
            double val = tmp1(0);
            
            return val;
        }

        double dist(vec f1, vec f2){
            double temp = inner(f1, f2);
            if (temp > 1){
                temp = 1;
            } else if (temp < -1){
                temp = -1;
            }
            double d = acos(temp);
            
            return d;
        }

        double typicaldist(){
            double out = M_PI/2;

            return out;
        }

        vec proj(vec f, vec v){
            arma::mat tmp1 = arma::trapz(time, f%v);
            vec out = v - f * tmp1(0);

            return out;
        }

        vec log(vec f1, vec f2){
            vec v = proj(f1, f2 - f1);
            double di = dist(f1, f2);
            if (di > 1e-6){
                double nv = norm2(v);
                v = v * (di/nv);
            }

            return v;
        }

        vec exp(vec f1, vec v, double delta=1){
            vec vd = delta * v;
            double nrm_vd = norm2(vd);

            vec f2;
            if (nrm_vd > 0){
                f2 = f1 * cos(nrm_vd) + vd * (sin(nrm_vd)/nrm_vd);
            } else {
                f2 = f1;
            }

            return f2;
        }

        double inner(vec v1, vec v2){
            arma::mat M = arma::trapz(time, v1 % v2);
            double val = M(0);

            return val;
        }

        double norm2(vec f){
            arma::mat tmp1 = arma::trapz(time, pow(f,2));
            double out = sqrt(tmp1(0));

            return out;
        }

        // Trapezoidal quadrature weights for the grid x, so that
        // trapz(x, y) equals sum(w % y).
        vec trapz_weights(vec x){
            uword n = x.n_elem;
            vec w = arma::zeros(n);
            vec dx = arma::diff(x) / 2.0;

            w(arma::span(0, n-2)) += dx;
            w(arma::span(1, n-1)) += dx;

            return w;
        }

        // Adjoint of gradient(): the transpose of the finite difference
        // operator gradient() applies, accumulated stencil by stencil so that
        // the two stay in step.
        vec gradient_adjoint(vec y, double binsize){
            vec out = arma::zeros(T);

            out(1) += y(0) / binsize;
            out(0) -= y(0) / binsize;

            out(T-1) += y(T-1) / binsize;
            out(T-2) -= y(T-1) / binsize;

            for (uword i = 1; i <= T-2; i++){
                out(i+1) += y(i) / (2 * binsize);
                out(i-1) -= y(i) / (2 * binsize);
            }

            return out;
        }

        vec gradient(vec f, double binsize){
            vec g = arma::zeros(T);
            g(0) = (f(1) - f(0)) / binsize;
            g(T-1) = (f(T-1) - f(T-2)) / binsize;
            
            g(arma::span(1, T-2)) = (f(arma::span(2, T-1)) - f(arma::span(0, T-3))) / (2 * binsize);

            return g;
        }

        // Adjoint of cumtrapz(): the transpose of the running trapezoidal sum,
        // built from the suffix sums of b so that the two stay in step.
        vec cumtrapz_adjoint(vec x, vec b){
            vec out = arma::zeros(T);
            double acc = 0;

            for (uword i = T-1; i >= 1; i--){
                acc += b(i);
                double dt = (x(i) - x(i-1)) / 2.0;
                out(i) += dt * acc;
                out(i-1) += dt * acc;
            }

            return out;
        }

        vec cumtrapz(vec x, vec y){
            vec z = arma::zeros(T);

            vec dt = arma::diff(x)/2.0;
            vec tmp = dt % (y(arma::span(1, T-1)) + y(arma::span(0,T-2)));
            z(arma::span(1,T-1)) = cumsum(tmp);

            return z;
        }
};

// info holds the iterations run, the stopping reason, the final cost and the
// final gradient norm
vec rlbfgs_optim(vec q1, vec q2, vec time, int maxiter, double lam, int penalty,
                 vec gam0, vec& info){
    uword T = time.n_elem;
    vec time1 = arma::linspace(0, 1, T);

    rlbfgs myObj(q1, q2, time1);
    myObj.solve(maxiter, lam, penalty, gam0);

    info = {double(myObj.iterations), double(myObj.stopReason), myObj.cost,
            myObj.gradNorm};

    return myObj.gammaOpt;
}

vec rlbfgs_optim(vec q1, vec q2, vec time, int maxiter, double lam, int penalty){
    vec info;

    return rlbfgs_optim(q1, q2, time, maxiter, lam, penalty, vec(), info);
}

int main() {
    uword T = 101;
    vec time = arma::linspace(0, 2*M_PI, T);
    vec q1 = sin(time);
    vec q2 = cos(time);
    vec time1 = arma::linspace(0, 1, T);

    rlbfgs myObj(q1, q2, time1); 
    myObj.solve();

    myObj.gammaOpt.print();

    return 0;
}
