#include <grid.h>


Grid::Grid() {}


Grid::Grid(COOMatrix* A1, int maxLev, int currentLev){
    A = A1;
    maxLevel     = maxLev;
    currentLevel = currentLev;
}


Grid::~Grid(){
//    printf("Grid destructor!!!\n");
}


//int Grid::setupCoarse() {
//    coarseGrid->maxLevel     = maxLevel;
//    coarseGrid->currentLevel = currentLevel - 1;
//    return 0;
//}