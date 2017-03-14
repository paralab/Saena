//
// Created by abaris on 3/13/17.
//

#ifndef SAENA_AMGCLASS_H
#define SAENA_AMGCLASS_H


#include "coomatrix.h"

class AMGClass {
public:
    int levels;
    int vcycle_num;
    double relTol;
    char* relaxType;
    int preSmooth;
    int postSmooth;
    float connStrength; // connection strength parameter

//    As:: Array{SparseMatrixCSC{Float64}}
//    Ps:: Array{SparseMatrixCSC{Float64}}
//    Rs:: Array{SparseMatrixCSC{Float64}}
//    relaxPrecs
//    LU

    double* A;
    double* Ac;
    int* P;
    int* R;

    AMGClass(int levels, int vcycle_num, double relTol, string relaxType, int preSmooth, int postSmooth, float connStrength, bool doSparsify);
    int AMGsetup(COOMatrix* A, bool doSparsify);
    ~AMGClass();
    int findAggregation(COOMatrix* A, float connStrength, double* S);
    int createStrengthMatrix(COOMatrix* A, double* S);

};


#endif //SAENA_AMGCLASS_H
