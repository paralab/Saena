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

    saena_vector *rhs_orig = nullptr;
    value_t      *rhs      = nullptr;

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

    std::vector<int> rcount3; // store nonzero rcounts2
    std::vector<int> scount3; // store nonzero scounts2
    std::vector<int> rproc_id; // store the index of nonzero rcounts2
    std::vector<int> sproc_id; // store the index of nonzero scounts2

    std::vector<MPI_Request> requests;  // used in repart_u() and repart_back_u()
    std::vector<value_t> u_old;         // used in repart_u() and repart_back_u()

    std::vector<value_t> res;
    std::vector<value_t> uCorr;
//    std::vector<value_t> res_coarse;
//    std::vector<value_t> uCorrCoarse;

    Grid() = default;

    Grid(saena_matrix* A1, int lev){
        A     = A1;
        level = lev;
    }

    ~Grid(){
        if(rhs != nullptr){
            free(rhs);
            rhs = nullptr;
        }
    }

    void repart_u_prepare();
    void repart_u(std::vector<value_t> &u);
    void repart_back_u(std::vector<value_t> &u);
};

#endif //SAENA_GRID_H
