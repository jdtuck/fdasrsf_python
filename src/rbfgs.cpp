#include <iostream>
#include "armadillo"
//#include "rbfgs.h"

using namespace arma;
using namespace std;

class rlbfgs {      
    private:             
        vec time;      // time
        vec q1;        // srvf1
        vec q2;        // srvf2
        uword T;         // size of time
        double hCurCost;
        vec hCurGradient;
        struct options{
            double tolgradnorm;
            double maxtime;
            int memory;
            int ls_max_steps;
            int maxiter;
            double minstepsize;
        };

        struct stats{
            int iter;
            double cost;
            double gradnorm;
            double stepsize;
            bool accepted;
        };

        struct lstats{
        double stepsize;
        vec newh;
    };

  public:
        vec gammaOpt;
        vec q2Opt;
        double cost;
        // constructor
        rlbfgs(vec q1i, vec q2i, vec timei) {  
        q1 = normalise( q1i, 2 );
        q2 = normalise( q2i, 2 );
        time = timei;

        T = timei.n_elem;
        }

        void solve(int maxiter=30, int verb=0, double lam=0.0, int penalty=0){
            // run solver
            options option;
            // terminates if the norm of the gradient drops below this
            option.tolgradnorm = 1e-3;
            // terminates if more than seconds elapsed
            option.maxtime = datum::inf;
            // number of previous iterations the program remembers
            option.memory = 30;
            option.ls_max_steps = 25;
            option.maxiter = maxiter;
            // minimum norm of tangent vector that points from current to next
            option.minstepsize = 1e-10;

            // Initialization of Variables
            vec htilde = ones(T);
            vec q2tilde = q2;

            // list to store step vectors
            arma::field<vec> tmp(option.memory);
            arma::field<vec> sHistory(option.memory);
            arma::field<vec> yHistory(option.memory);
            vec rhoHistory(option.memory);
            vec tmp_vec(option.memory);

            // number of iterations since last restart
            int j = 0;
            // Total number of BFGS iterations
            int k = 0;
            // scaling of direction given by getDirection
            double alpha = 1;
            // scaling of initial matrix, Barzilai-Borwein
            double scaleFactor = 1;
            // Norm of the step
            double stepsize = 1;

            bool accepted = true;
            int stop;

            double hCurCost;
            vec hCurGradient;
            lstats lstat;
            vec hNext;
            alignment_costgrad(q2tilde, htilde, hCurCost, hCurGradient, lam, penalty);
            double hCurGradNorm = norm2(hCurGradient);

            bool ultimatum = false;

            stats stat;
            stat.iter = k;
            stat.cost = hCurCost;
            stat.gradnorm = hCurGradNorm;
            stat.stepsize = datum::nan;
            stat.accepted = false;

            vec p;
            double in_prod;
            vec step;
            double hNextCost;
            vec hNextGradient;
            vec sk;
            vec yk;
            double norm_sk;
            double inner_sk_yk;
            double inner_sk_sk;
            double cap, rhok;
            while (true){
                stop = stoppingcriterion(option, stat);

                if (stop == 0){
                    if (stat.stepsize < option.minstepsize){
                        if (!ultimatum){
                            j = 0;
                            ultimatum = true;
                        } else {
                            stop = 1;
                        }
                    } else {
                        ultimatum = false;
                    }
                }

                if (stop > 0){
                    break;
                }

                // compute BFGS direction
                p = getDirection(hCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, min(j, option.memory));

                // execute line search
                in_prod = inner(hCurGradient, p);
                lstat = linesearch_hint(p, hCurCost, in_prod, q2tilde, lam, penalty);

                stepsize = lstat.stepsize;
                hNext = lstat.newh;

                // iterative update
                htilde = group_action_SRVF(htilde, hNext);
                q2tilde = group_action_SRVF(q2tilde, hNext);

                // record the BGFS step multiplier
                alpha = stepsize / norm2(p);
                step = alpha * p;

                // query cost and gradient at the candidate new point
                alignment_costgrad(q2tilde, hNext, hNextCost, hNextGradient, lam, penalty);

                // compute sk and yk
                sk = step;
                yk = hNextGradient - hCurGradient;

                // computation of the BFGS step
                norm_sk = norm2(sk);
                sk = sk / norm_sk;
                yk = yk / norm_sk;

                inner_sk_yk = inner(sk, yk);
                inner_sk_sk = pow(norm2(sk),2);  // ensures nonnegativity

                // cautious step
                cap = strict_inc_func(hCurGradNorm);
                if ((inner_sk_sk != 0) & ((inner_sk_yk/inner_sk_sk) >= cap)){
                    accepted = true;

                    rhok = 1/inner_sk_yk;

                    scaleFactor = inner_sk_yk / pow(norm2(yk),2);

                    if (j >= option.memory){
                        if (option.memory > 1){
                            tmp.subfield(0, 0, option.memory-2, 0) = sHistory.subfield(1, 0, option.memory-1, 0);
                            tmp(option.memory-1) = sHistory(0);
                            sHistory = tmp;

                            tmp.subfield(0, 0, option.memory-2, 0) = yHistory.subfield(1, 0, option.memory-1, 0);
                            tmp(option.memory-1) = yHistory(0);
                            yHistory = tmp;

                            tmp_vec(arma::span(0,option.memory-1)) = rhoHistory(arma::span(1,option.memory-2));
                            tmp_vec(option.memory) = rhoHistory(0);
                            rhoHistory = tmp_vec;
                        }
                        if (option.memory > 0){
                            sHistory(option.memory-1) = sk;
                            yHistory(option.memory-1) = yk;
                            rhoHistory(option.memory-1) = rhok;
                        }
                    } else {
                        sHistory(j) = sk;
                        yHistory(j) = yk;
                        rhoHistory(j) = rhok;
                    }

                    j += 1;
                } else {
                    accepted = false;
                }

                hCurGradient = hNextGradient;
                hCurGradNorm = norm2(hNextGradient);
                hCurCost = hNextCost;

                k += 1;

                stat.iter = k;
                stat.cost = hCurCost;
                stat.gradnorm = hCurGradNorm;
                stat.stepsize = datum::nan;
                stat.accepted = accepted;
            }

            gammaOpt = arma::zeros(T);
            gammaOpt(arma::span(1, T)) = cumtrapz(time, pow(htilde,2));
            gammaOpt = (gammaOpt - gammaOpt.min()) / (gammaOpt.max() - gammaOpt.min());
            q2Opt = q2tilde;
            cost = hCurCost;
        }

        double alignment_cost(vec h, vec q2k, double lam = 0, int penalty = 0){
            vec q2new = group_action_SRVF(q2k, h);

            double pen = 0;
            if (penalty == 0){
                vec time1 = arma::linspace<vec>(0,1,h.n_elem);
                vec b = arma::diff(time1);
                double binsize = mean(b);
                vec g = gradient(arma::pow(h, 2), binsize); 
                arma::mat pen1 = arma::trapz(time1, g);
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

            double f = normL2(q1-q2k);
            f = pow(f,2) + lam * pen;

            return f;
        }

        void alignment_costgrad(vec q2k, vec h, double f, vec g, double lam = 0, int penalty = 0){
            // roughness
            double pen = 0;
            if (penalty == 0){
                vec time1 = arma::linspace<vec>(0,1,h.n_elem);
                vec b = arma::diff(time1);
                double binsize = mean(b);
                vec g = gradient(arma::pow(h, 2), binsize); 
                arma::mat pen1 = arma::trapz(time1, g);
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

            // compute cost 
            f = normL2(q1-q2k);
            f = pow(f,2) + lam * pen;

            // compute cost gradient
            double binsize = 1.0/(T-1);
            vec q2kdot = gradient(q2k, binsize);
            vec dq = q1 - q2k;
            vec v = arma::zeros(T);
            vec tmp = dq % q2kdot;
            vec tmp1 = dq % q2k;
            v(arma::span(1,T-1)) = 2 * cumtrapz(time, tmp);
            v = v - tmp1;

            mat val = arma::trapz(time, v);
            g = v - val(0);

            return;

        }

        vec getDirection(vec hCurGradient, arma::field<vec> sHistory, arma::field<vec> yHistory, vec rhoHistory, double scaleFactor, int j){
            vec q = hCurGradient;
            vec inner_s_q = arma::zeros(j);

            for (int i = j; i > 0; i--) {
                inner_s_q(i-1) = rhoHistory(i-1) * inner(sHistory(i-1), q);
                q = q - inner_s_q(i-1) * yHistory(i-1);
            } 

            vec r = scaleFactor * q;

            double omega;
            for (int i=0; i < j; i++){
                omega = rhoHistory(i) * inner(yHistory(i), r);
                r = r + (inner_s_q(i) - omega) * sHistory(i);
            }

            vec direction = -1 * r;

            return direction;
        }

        lstats linesearch_hint(vec d, double f0, double df0, vec q2k, double lam=0, int penalty=0){
            // Armijo line-search based on the line-search hint in the problem

            double contraction_factor = 0.5;
            double suff_decr = 1e-6;
            int max_ls_steps = 25;
            bool ls_backtrack = true;
            bool ls_force_decrease = true;

            double alpha = 1;
            vec hid = arma::ones(T);  // identity element

            vec newh = exp(hid, d, alpha);
            double newf = alignment_cost(newh, q2k, lam, penalty);
            int cost_evaluations = 1;

            uvec tst = newh <= 0;
            while (ls_backtrack & (newf > (f0 + suff_decr*alpha*df0)) || arma::sum(tst) > 0){
                alpha *= contraction_factor;

                newh = exp(hid, d, alpha);
                newf = alignment_cost(newh, q2k, lam, penalty);
                cost_evaluations += 1;
                tst = newh <= 0;

                if (cost_evaluations >= max_ls_steps){
                    break;
                }
            }

            if (ls_force_decrease & (newf > f0)){
                alpha = 0;
                newh = hid;
                newf = f0;
            }

            double norm_d = norm2(d);
            double stepsize = alpha * norm_d;

            lstats lstat;
            lstat.stepsize = stepsize;
            lstat.newh = newh;

            return lstat;
        }

        int stoppingcriterion(options option, stats stat){
            int stop = 0;
            if (stat.gradnorm <= option.tolgradnorm){
                stop = 2;
            }

            if (stat.iter >= option.maxiter){
                stop = 3;
            }

            return stop;
        }

        vec group_action_SRVF(vec q, vec h){
            vec gamma = cumtrapz(time, arma::pow(h,2));
            gamma = gamma / gamma.back();
            vec time1 = arma::linspace<vec>(0,1,h.n_elem);
            vec b = arma::diff(time1);
            double binsize = mean(b);
            vec h1 = gradient(gamma, binsize);
            h1 = sqrt(h1);
            vec qnew;
            arma::interp1(time, q, gamma, qnew);
            qnew = qnew % h;

            return qnew;
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
            double d = real(acos(temp));
            
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

        vec transp(vec f1, vec f2, vec v){
            // isometric vector transport
            vec w = log(f1, f2);
            double dist_f1f2 = norm2(w);

            vec Tv;
            if (dist_f1f2 > 0){
                vec u = w / dist_f1f2;
                double utv = inner(u, v);
                Tv = v + (cos(dist_f1f2) - 1) * utv * u - sin(dist_f1f2);
            } else{
                Tv = v;
            }

            return Tv;
        }

        vec gradient(vec f, double binsize){
            vec g = arma::zeros(T);
            g(0) = (f(1) - f(0)) / binsize;
            g(T-1) = (f(T-1) - f(T-2)) / binsize;
            
            g(arma::span(1, T-2)) = (f(arma::span(2, T-1)) - f(arma::span(0, T-3))) / (2 * binsize);

            return g;
        }

        vec cumtrapz(vec x, vec y){
        vec z = arma::zeros(T);

        vec dt = arma::diff(x)/2.0;
        vec tmp = dt % (y(arma::span(0, T-2))- y(arma::span(1,T-1)));
        z(arma::span(1,T-1)) = cumsum(tmp);

        return z;
    }
};


int main() {
    uword T = 101;
    vec time = arma::linspace(0, 2*M_PI, T);
    vec q1 = sin(time);
    vec q2 = cos(time);

    rlbfgs myObj(q1, q2, time); 

    return 0;
}
