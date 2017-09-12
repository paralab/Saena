#ifndef SAENA_SAENA_OBJECT_H
#define SAENA_SAENA_OBJECT_H

#include "saena_matrix.h"
#include "strengthmatrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"
#include "auxFunctions.h"
#include "grid.h"

class saena_object {
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
    saena_object();
    ~saena_object();
    int destroy();
    void set_parameters(int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
    int levelSetup(Grid* grid);
    int setup(saena_matrix* A);
    int findAggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int createStrengthMatrix(saena_matrix* A, StrengthMatrix* S);
    int aggregation(StrengthMatrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int createProlongation(saena_matrix* A, std::vector<unsigned long>& aggregate, prolongMatrix* P);
    int coarsen(saena_matrix* A, prolongMatrix* P, restrictMatrix* R, saena_matrix* Ac);
    int solveCoarsest(saena_matrix* A, std::vector<double>& u, std::vector<double>& rhs, int& maxIter, double& tol);
    int vcycle(Grid* grid, std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);
    int solve(std::vector<double>& u, std::vector<double>& rhs);
    int residual(saena_matrix* A, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res);

    int writeMatrixToFileA(saena_matrix* A, std::string name);
    int writeMatrixToFileP(prolongMatrix* P, std::string name);
    int writeMatrixToFileR(restrictMatrix* R, std::string name);
    int writeVectorToFiled(std::vector<double>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int writeVectorToFileul(std::vector<unsigned long>& v, unsigned long vSize, std::string name, MPI_Comm comm);
//    template <class T>
//    int writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int changeAggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
};

#endif //SAENA_SAENA_OBJECT_H