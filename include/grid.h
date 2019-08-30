#ifndef SAENA_GRID_H
#define SAENA_GRID_H

#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "saena_matrix_dense.h"
#include "saena_matrix.h"
#include "saena_vector.h"

#include <vector>
#include <mpi.h>

//class prolong_matrix;
//class restrict_matrix;
//class saena_matrix;
//class saena_matrix_dense;
//class saena_object;

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;


class Grid{
public:
    saena_matrix* A     = nullptr;
    saena_matrix* A_new = nullptr; // for solve_pcg_update() experiment
    saena_matrix  Ac;
//    saena_matrix_dense* A_d; // dense matrix
    saena_matrix_dense Ac_d; // dense matrix
    prolong_matrix P;
    restrict_matrix R;
    std::vector<value_t> rhs;
    saena_vector *rhs_orig = nullptr;
    int currentLevel, maxLevel;
    float row_reduction_min = 0;
    Grid* coarseGrid = nullptr;

    std::vector<int> rcount;
    std::vector<int> scount;
    std::vector<int> rdispls;
    std::vector<int> sdispls;

    std::vector<int> rcount2;
    std::vector<int> scount2;
    std::vector<int> rdispls2;
    std::vector<int> sdispls2;

//    std::vector<int> rcount3;
//    std::vector<int> scount3;
//    std::vector<int> rdispls3;
//    std::vector<int> sdispls3;

    bool active = false;
//    MPI_Comm comm;

    Grid();
    Grid(saena_matrix* A, int maxLevel, int currentLevel);
    ~Grid();
};

#endif //SAENA_GRID_H
