#ifndef SAENA_COMBBLAS_FUNCTIONS_H
#define SAENA_COMBBLAS_FUNCTIONS_H

#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;
#define ITERATIONS 1

// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
    typedef SpDCCols < int, NT > DCCols;
    typedef SpParMat < int, NT, DCCols > MPI_DCCols;
};

#define ElementType double

int combblas_matmult_DoubleBuff(const string &Aname, const string &Bname);
int combblas_matmult_Synch(const string &Aname, const string &Bname);
int combblas_matmult_experiment(const string &Aname, const string &Bname, MPI_Comm comm);
int combblas_GalerkinNew();

#endif //SAENA_COMBBLAS_FUNCTIONS_H
