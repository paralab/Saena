
#include <grid.h>

Grid::Grid(COOMatrix* A1, int maxLev, int currentLev){
    A = A1;
    maxLevel     = maxLev;
    currentLevel = currentLev;
}


Grid::~Grid(){
}


int Grid::setupCoarse() {
    coarseGrid->maxLevel     = maxLevel;
    coarseGrid->currentLevel = currentLevel - 1;
    return 0;
}