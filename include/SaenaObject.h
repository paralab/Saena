#ifndef SAENA_SaenaObject_H
#define SAENA_SaenaObject_H

#include "SaenaMatrix.h"
#include "strengthmatrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"
#include "auxFunctions.h"
#include "grid.h"

class SaenaObject {
public:
    int maxLevel = 1; // fine grid is level 0.
    int vcycle_num;
    double relative_tolerance;
    std::string smoother;
    int preSmooth;
    int postSmooth;
    float connStrength = 0.5; // connection strength parameter
    bool doSparsify = false;
    std::vector<Grid> grids;

//    SaenaObject(int levels, int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
    SaenaObject();
    ~SaenaObject();
    int destroy();
    void set_parameters(int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
    int levelSetup(Grid* grid);
    int setup(SaenaMatrix* A);
    int findAggregation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int createStrengthMatrix(SaenaMatrix* A, StrengthMatrix* S);
    int aggregation(StrengthMatrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int createProlongation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, prolongMatrix* P);
    int coarsen(SaenaMatrix* A, prolongMatrix* P, restrictMatrix* R, SaenaMatrix* Ac);
    int solveCoarsest(SaenaMatrix* A, std::vector<double>& u, std::vector<double>& rhs, int& maxIter, double& tol);
    int vcycle(Grid* grid, std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);
    int solve(std::vector<double>& u, std::vector<double>& rhs);
    int residual(SaenaMatrix* A, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res);

    int writeMatrixToFileA(SaenaMatrix* A, std::string name);
    int writeMatrixToFileP(prolongMatrix* P, std::string name);
    int writeMatrixToFileR(restrictMatrix* R, std::string name);
    int writeVectorToFiled(std::vector<double>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int writeVectorToFileul(std::vector<unsigned long>& v, unsigned long vSize, std::string name, MPI_Comm comm);
//    template <class T>
//    int writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int changeAggregation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
};

#endif //SAENA_SaenaObject_H