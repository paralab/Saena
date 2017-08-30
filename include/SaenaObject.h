#ifndef SAENA_SaenaObject_H
#define SAENA_SaenaObject_H

#include "SaenaMatrix.h"
#include "strengthmatrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"
#include "grid.h"

class SaenaObject {
public:
    int maxLevel;
    int vcycle_num;
    double relTol;
    string smoother;
    int preSmooth;
    int postSmooth;
    float connStrength = 0.5; // connection strength parameter
    bool doSparsify = false;
    std::vector<Grid> grids;

//    As:: Array{SparseMatrixCSC{Float64}}
//    Ps:: Array{SparseMatrixCSC{Float64}}
//    Rs:: Array{SparseMatrixCSC{Float64}}
//    relaxPrecs
//    LU

    SaenaObject(int levels, int vcycle_num, double relTol, string relaxType, int preSmooth, int postSmooth);
    ~SaenaObject();
    int Destroy();
    int levelSetup(Grid* grid, MPI_Comm comm);
    int Setup(SaenaMatrix* A, MPI_Comm comm);
    int findAggregation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew, MPI_Comm comm);
    int createStrengthMatrix(SaenaMatrix* A, StrengthMatrix* S, MPI_Comm comm);
    int Aggregation(StrengthMatrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew, MPI_Comm comm);
    int createProlongation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, prolongMatrix* P, MPI_Comm comm);
//    int createRestriction(prolongMatrix* P, restrictMatrix* R, MPI_Comm comm);
    int coarsen(SaenaMatrix* A, prolongMatrix* P, restrictMatrix* R, SaenaMatrix* Ac, MPI_Comm comm);
    int residual(SaenaMatrix* A, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res, MPI_Comm comm);
    int dotProduct(std::vector<double>& r, std::vector<double>& s, double* dot, MPI_Comm comm);
    int solveCoarsest(SaenaMatrix* A, std::vector<double>& u, std::vector<double>& rhs, int& maxIter, double& tol, MPI_Comm comm);
    int vcycle(Grid* grid, std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);
    int Solve(std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);

    int writeMatrixToFileA(SaenaMatrix* A, string name, MPI_Comm comm);
    int writeMatrixToFileP(prolongMatrix* P, string name, MPI_Comm comm);
    int writeMatrixToFileR(restrictMatrix* R, string name, MPI_Comm comm);
    int writeVectorToFiled(std::vector<double>& v, unsigned long vSize, string name, MPI_Comm comm);
    int writeVectorToFileul(std::vector<unsigned long>& v, unsigned long vSize, string name, MPI_Comm comm);
//    template <class T>
//    int writeVectorToFile(std::vector<T>& v, unsigned long vSize, string name, MPI_Comm comm);
//    template <class T>
//    int test(std::vector<T>& v);
    int changeAggregation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew, MPI_Comm comm);
};

#endif //SAENA_SaenaObject_H