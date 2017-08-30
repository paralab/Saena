#ifndef SAENA_GRID_H
#define SAENA_GRID_H

#include "SaenaMatrix.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"

class Grid{
private:

public:
    SaenaMatrix* A;
    SaenaMatrix  Ac;
    prolongMatrix P;
    restrictMatrix R;
    int currentLevel, maxLevel;
    Grid* coarseGrid;

    Grid();
    Grid(SaenaMatrix* A, int maxLevel, int currentLevel);
    ~Grid();
};

#endif //SAENA_GRID_H
