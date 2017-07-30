#ifndef SAENA_GRID_H
#define SAENA_GRID_H

#include "coomatrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"

class Grid{
private:

public:
    COOMatrix* A;
    COOMatrix  Ac;
    prolongMatrix P;
    restrictMatrix R;
    int currentLevel, maxLevel;
    Grid* coarseGrid;

    Grid();
    Grid(COOMatrix* A, int maxLevel, int currentLevel);
    ~Grid();
};

#endif //SAENA_GRID_H
