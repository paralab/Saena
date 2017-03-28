#ifndef SAENA_AMGCLASS_H
#define SAENA_AMGCLASS_H

#include "coomatrix.h"
#include "strengthmatrix.h"

class AMGClass {
public:
    int levels;
    int vcycle_num;
    double relTol;
    string smoother;
    int preSmooth;
    int postSmooth;
    float connStrength; // connection strength parameter
    float tau; // is used during making aggregates.

//    As:: Array{SparseMatrixCSC{Float64}}
//    Ps:: Array{SparseMatrixCSC{Float64}}
//    Rs:: Array{SparseMatrixCSC{Float64}}
//    relaxPrecs
//    LU

//    double* A;
    double* Ac;
    long* P;
    long* R;

    AMGClass(int levels, int vcycle_num, double relTol, string relaxType, int preSmooth, int postSmooth, float connStrength, float tau);
    ~AMGClass();
    int AMGsetup(COOMatrix* A, bool doSparsify);
    int findAggregation(COOMatrix* A);
    int createStrengthMatrix(COOMatrix* A, StrengthMatrix* S);
    int Aggregation(StrengthMatrix* S, long* aggregate);
};


#endif //SAENA_AMGCLASS_H
