#pragma once

#include "saena.hpp"
//#include "data_struct.h"
#include "saena_matrix.h"

namespace saena {
    // ==========================
    // Matrix Generator Functions
    // ==========================

    int laplacian2D(saena::matrix* A, index_t mx, index_t my, bool scale = true);
    int laplacian2D_set_rhs(std::vector<double> &rhs, index_t mx, index_t my, MPI_Comm comm);
    int laplacian2D_check_solution(std::vector<double> &u, index_t mx, index_t my, MPI_Comm comm);

    int laplacian3D(saena::matrix* A, index_t mx, index_t my, index_t mz);
    int laplacian3D_no_boundary(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale = true);
    int laplacian3D_no_boundary_lower(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale = true);
    int laplacian3D_set_rhs(value_t *&rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);
    int laplacian3D_check_solution(std::vector<double> &u, index_t mx, index_t my, index_t mz, MPI_Comm comm);

    // second argument is degree-of-freedom on each processor
    int laplacian2D_old(saena::matrix* A, index_t dof_local);

    int laplacian3D_old(saena::matrix* A, index_t dof_local);

    int laplacian3D_old2(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale = true);
    int laplacian3D_set_rhs_old2(std::vector<double> &rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);

    int laplacian3D_old3(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale = true);
    int laplacian3D_set_rhs_old3(std::vector<double> &rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);

    int laplacian3D_set_rhs_zero(std::vector<double> &rhs, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm);

    int band_matrix(saena::matrix &A, index_t M, unsigned int bandwidth, bool use_dense = false);

    int random_symm_matrix(saena::matrix &A, index_t M, float density);

    // ==========================

    int read_vector_file(std::vector<value_t>& v, saena::matrix &A, char *file, MPI_Comm comm);

    index_t find_split(index_t loc_size, index_t &my_split, MPI_Comm comm);
}