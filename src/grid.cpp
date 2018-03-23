#include <grid.h>

Grid::Grid(){
}


Grid::Grid(saena_matrix* A1, int maxLev, int currentLev){
    A            = A1;
    maxLevel     = maxLev;
    currentLevel = currentLev;
    active       = A1->active;
}


Grid::~Grid(){}