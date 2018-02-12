#ifndef SAENA_GRID_H
#define SAENA_GRID_H

#include <mpi.h>
#include "saena_matrix.h"
#include "restrict_matrix.h"
#include "prolong_matrix.h"


class Grid{
public:
    saena_matrix* A;
    saena_matrix* A_new; // for solve_pcg_update() experiment
    saena_matrix  Ac;
    prolong_matrix P;
    restrict_matrix R;
    std::vector<double> rhs;
    int currentLevel, maxLevel;
    Grid* coarseGrid;
    std::vector<int> rcount;
    std::vector<int> scount;
    bool active = false;
//    MPI_Comm comm;

    Grid();
    Grid(saena_matrix* A, int maxLevel, int currentLevel);
    ~Grid();
};

#endif //SAENA_GRID_H
