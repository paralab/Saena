#ifndef SAENA_GRID_H
#define SAENA_GRID_H

#include <mpi.h>
#include "saena_matrix.h"
#include "saena_matrix_dense.h"
#include "restrict_matrix.h"
#include "prolong_matrix.h"

class Grid{
public:
    saena_matrix* A;
    saena_matrix* A_new; // for solve_pcg_update() experiment
    saena_matrix  Ac;
//    saena_matrix_dense* A_d; // dense matrix
    saena_matrix_dense Ac_d; // dense matrix
    prolong_matrix P;
    restrict_matrix R;
    std::vector<value_t> rhs;
    int currentLevel, maxLevel;
    Grid* coarseGrid;
    std::vector<int> rcount;
    std::vector<int> scount;
    std::vector<int> rdispls;
    std::vector<int> sdispls;
    std::vector<int> rcount2;
    std::vector<int> scount2;
    std::vector<int> rdispls2;
    std::vector<int> sdispls2;
    bool active = false;
//    MPI_Comm comm;

    Grid();
    Grid(saena_matrix* A, int maxLevel, int currentLevel);
    ~Grid();
};

#endif //SAENA_GRID_H
