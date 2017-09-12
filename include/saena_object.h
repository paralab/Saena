#ifndef SAENA_SAENA_OBJECT_H
#define SAENA_SAENA_OBJECT_H

#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "aux_functions.h"
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

    saena_object();
    ~saena_object();
    int destroy();
    void set_parameters(int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
    int level_setup(Grid* grid);
    int setup(saena_matrix* A);
    int find_aggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int create_strength_matrix(saena_matrix* A, strength_matrix* S);
    int aggregation(strength_matrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int create_prolongation(saena_matrix* A, std::vector<unsigned long>& aggregate, prolong_matrix* P);
    int coarsen(saena_matrix* A, prolong_matrix* P, restrict_matrix* R, saena_matrix* Ac);
    int solve_coarsest(saena_matrix* A, std::vector<double>& u, std::vector<double>& rhs, int& maxIter, double& tol);
    int vcycle(Grid* grid, std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);
    int solve(std::vector<double>& u, std::vector<double>& rhs);
    int residual(saena_matrix* A, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res);

    int writeMatrixToFileA(saena_matrix* A, std::string name);
    int writeMatrixToFileP(prolong_matrix* P, std::string name);
    int writeMatrixToFileR(restrict_matrix* R, std::string name);
    int writeVectorToFiled(std::vector<double>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int writeVectorToFileul(std::vector<unsigned long>& v, unsigned long vSize, std::string name, MPI_Comm comm);
//    template <class T>
//    int writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int change_aggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
};

#endif //SAENA_SAENA_OBJECT_H
