#ifndef SAENA_GRID_H
#define SAENA_GRID_H

#include "saena_matrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"

class Grid{
private:

public:
    saena_matrix* A;
    saena_matrix  Ac;
    prolongMatrix P;
    restrictMatrix R;
    int currentLevel, maxLevel;
    Grid* coarseGrid;

    MPI_Comm comm;

    Grid();
    Grid(saena_matrix* A, int maxLevel, int currentLevel);
    ~Grid();
};

#endif //SAENA_GRID_H
