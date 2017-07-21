#include <grid.h>

Grid::Grid() {}


Grid::Grid(COOMatrix* A1, int maxLev, int currentLev){
    A = A1;
    maxLevel     = maxLev;
    currentLevel = currentLev;
}


Grid::~Grid(){}
