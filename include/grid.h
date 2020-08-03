#ifndef SAENA_GRID_H
#define SAENA_GRID_H

#include "data_struct.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "saena_matrix.h"
#include "saena_vector.h"


class Grid{
public:
    saena_matrix*   A     = nullptr;
    saena_matrix*   A_new = nullptr; // for solve_pcg_update() experiment
    saena_matrix    Ac;
    prolong_matrix  P;
    restrict_matrix R;

//    saena_matrix_dense* A_d; // dense matrix
//    saena_matrix_dense Ac_d; // dense matrix

    std::vector<value_t> rhs;
    saena_vector         *rhs_orig = nullptr;

    Grid* coarseGrid = nullptr;
    int   level      = 0;
    bool  active     = false;

    std::vector<int> rcount;
    std::vector<int> scount;
    std::vector<int> rdispls;
    std::vector<int> sdispls;

    std::vector<int> rcount2;
    std::vector<int> scount2;
    std::vector<int> rdispls2;
    std::vector<int> sdispls2;

    std::vector<value_t> res;
    std::vector<value_t> uCorr;
//    std::vector<value_t> res_coarse;
//    std::vector<value_t> uCorrCoarse;

    Grid() = default;

    Grid(saena_matrix* A1, int lev){
        A     = A1;
        level = lev;
    }

    ~Grid() = default;
};

#endif //SAENA_GRID_H
