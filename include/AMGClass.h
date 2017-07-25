#ifndef SAENA_AMGCLASS_H
#define SAENA_AMGCLASS_H

#include "coomatrix.h"
#include "strengthmatrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"
#include "grid.h"

class AMGClass {
public:
    int maxLevel;
    int vcycle_num;
    double relTol;
    string smoother;
    int preSmooth;
    int postSmooth;
    float connStrength; // connection strength parameter
    float tau; // is used during making aggregates.
    bool doSparsify;

//    As:: Array{SparseMatrixCSC{Float64}}
//    Ps:: Array{SparseMatrixCSC{Float64}}
//    Rs:: Array{SparseMatrixCSC{Float64}}
//    relaxPrecs
//    LU

    AMGClass(int levels, int vcycle_num, double relTol, string relaxType, int preSmooth, int postSmooth, float connStrength, float tau, bool doSparsify);
    ~AMGClass();
    int levelSetup(Grid* grid, MPI_Comm comm);
    int AMGSetup(Grid* grids, COOMatrix* A, MPI_Comm comm);
    int findAggregation(COOMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew, MPI_Comm comm);
    int createStrengthMatrix(COOMatrix* A, StrengthMatrix* S, MPI_Comm comm);
    int Aggregation(StrengthMatrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew, MPI_Comm comm);
    int createProlongation(COOMatrix* A, std::vector<unsigned long>& aggregate, prolongMatrix* P, MPI_Comm comm);
//    int createRestriction(prolongMatrix* P, restrictMatrix* R, MPI_Comm comm);
    int coarsen(COOMatrix* A, prolongMatrix* P, restrictMatrix* R, COOMatrix* Ac, MPI_Comm comm);
    int residual(COOMatrix* A, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res, MPI_Comm comm);
    int dotProduct(std::vector<double>& r, std::vector<double>& s, double* dot, MPI_Comm comm);
    int solveCoarsest(COOMatrix* A, std::vector<double>& u, std::vector<double>& rhs, int& maxIter, double& tol, MPI_Comm comm);
    int vcycle(Grid* grid, std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);
    int AMGSolve(Grid* grid, std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);

    int writeMatrixToFile(COOMatrix* A, MPI_Comm comm);
};

#endif //SAENA_AMGCLASS_H