#ifdef _USE_COMBBLAS_

#ifndef SAENA_COMBBLAS_FUNCTIONS_H
#define SAENA_COMBBLAS_FUNCTIONS_H

#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

int combblas_matmult_DoubleBuff(const string &Aname, const string &Bname);
int combblas_matmult_Synch(const string &Aname, const string &Bname);
int combblas_matmult_experiment(const string &Aname, const string &Bname, MPI_Comm comm);
int combblas_GalerkinNew();

#endif //SAENA_COMBBLAS_FUNCTIONS_H

#endif //_USE_COMBBLAS_