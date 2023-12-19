#include <iostream>
#include "armadillo"
#include "rbfgs.h"

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


    // constructor
    rlbfgs(vec q1i, vec q2i, vec timei) {  
      q1 = normalise( q1i, 2 );
      q2 = normalise( q2i, 2 );
      time = timei;

      T = timei.n_elem;
    }

    void solve(int maxiter=30, int verb=0, double lam=0.0, int penalty){
        // run solver

        struct {
            double tolgradnorm;
            double maxtime;
            int memory;
            int ls_max_steps;
            int maxiter;
            double minstepsize;
        } options;

        // terminates if the norm of the gradient drops below this
        options.tolgradnorm = 1e-3;
        // terminates if more than seconds elapsed
        options.maxtime = datum::inf;
        // number of previous iterations the program remembers
        options.memory = 30;
        options.ls_max_steps = 25;
        options.maxiter = maxiter;
        // minimum norm of tangent vector that points from current to next
        options.minstepsize = 1e-10;

        // Initialization of Variables
        vec htilde = ones(T);
        vec q2tilde = q2;

        // list to store step vectors
        arma::field<vec> sHistory(options.memory);
        arma::field<vec> yHistory(options.memory);
        vec rhoHistory(options.memory);

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

        alignment_costgrad(q2tilde, htilde, lam, penalty);
    }

    void alignment_costgrad(vec q2k, vec h, double lam = 0, int penalty = 0){
        // roughness
        if (penalty == 0){
            vec time1 = arma::linspace<vec>(0,1,h.n_elem);
            vec b = arma::diff(time1);
            double binsize = mean(b);
            vec g = gradient(pow(h, 2), binsize); 
            mat pen = arma::trapz(time1, g);
        }

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
        mat tmp1 = trapz(time, tmp);
        double val = tmp1(0);
        
        return val;
    }

    vec gradient(vec f, double binsize){
        vec g = arma::zeros(T);
        g(0) = (f(1) - f(0)) / binsize;
        g(T-1) = (f(T-1) - f(T-2)) / binsize;
        
        g(arma::span(1, T-2)) = (f(arma::span(2, T-1)) - f(arma::span(0, T-3))) / (2 * binsize);

        return g;
    }
};

