
#ifndef ELASTICCURVESREPARAM_H
#define ELASTICCURVESREPARAM_H

#include <fstream>
#include <DriverElasticCurvesRO.h>
#include <def.h>

void optimum_reparam(double *C1, double *C2, int n, int d, double w,
                     bool onlyDP, bool rotated, bool isclosed, int skipm,
                     int autoselectC, double *opt, bool swap, double *fopts,
                     double *comtime);

#endif // end of TESTELASTICCURVESRO_H
