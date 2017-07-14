#ifndef SAENA_AMGCLASS_H
#define SAENA_AMGCLASS_H

#include "coomatrix.h"
#include "strengthmatrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"

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
    int AMGSetup(COOMatrix* A, bool doSparsify, MPI_Comm comm);
    int findAggregation(COOMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew, MPI_Comm comm);
    int createStrengthMatrix(COOMatrix* A, StrengthMatrix* S, MPI_Comm comm);
    int Aggregation(StrengthMatrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew, MPI_Comm comm);
    int createProlongation(COOMatrix* A, std::vector<unsigned long>& aggregate, prolongMatrix* P, MPI_Comm comm);
//    int createRestriction(prolongMatrix* P, restrictMatrix* R, MPI_Comm comm);
    int coarsen(COOMatrix* A, prolongMatrix* P, restrictMatrix* R, COOMatrix* Ac, MPI_Comm comm);
    int solveCoarsest(COOMatrix* A, std::vector<double>& u, std::vector<double>& b, int& maxIter, double& tol, MPI_Comm comm);
    };


#endif //SAENA_AMGCLASS_H