#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"

#include "petsc_functions.h"
//#include "combblas_functions.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <omp.h>
#include <dollar.hpp>
#include "mpi.h"


#include "grid.h"


int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

    if(argc != 3){
        if(rank == 0){
//            std::cout << "This is how to make a 3DLaplacian: ./Saena <x grid size> <y grid size> <z grid size>" << std::endl;
            std::cout << "This is how to make a 3DLaplacian: ./Saena <local size> <bandwidth>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // *************************** initialize the matrix: Laplacian ****************************

    double t1 = MPI_Wtime();

    int M(std::stoi(argv[1]));
    int band(std::stoi(argv[2]));

    saena::matrix A(comm);
    saena::band_matrix(A, M, band);

    // ********** print matrix and time **********

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
//    print_time(t1, t2, "Matrix Assemble:", comm);

//    A.print(0);
//    A.get_internal_matrix()->print_info(0);
//    A.get_internal_matrix()->writeMatrixToFile("writeMatrix");

//    petsc_viewer(A.get_internal_matrix());

    // *************************** set rhs: Laplacian ****************************
/*
    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<double> rhs;
    saena::laplacian3D_set_rhs(rhs, mx, my, mz, comm);

    // ********** print rhs **********

//    print_vector(rhs, -1, "rhs", comm);

    // *************************** set u0 ****************************

    std::vector<double> u(num_local_row, 0);

    // *************************** AMG - Setup ****************************

    t1 = MPI_Wtime();

//    int max_level             = 2; // this is moved to saena_object.
    int vcycle_num            = 400;
    double relative_tolerance = 1e-12;
    std::string smoother      = "chebyshev"; // choices: "jacobi", "chebyshev"
    int preSmooth             = 3;
    int postSmooth            = 3;

    saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
//    saena::options opts((char*)"options001.xml");
//    saena::options opts;
    saena::amg solver;
    solver.set_verbose(verbose); // set verbose at the beginning of the main function.
//    solver.set_multigrid_max_level(0); // 0 means only use direct solver, so no multigrid will be used.

//    if(rank==0) printf("usage: ./Saena x_size y_size z_size sparse_epsilon \n");
//    double sp_epsilon(std::atof(argv[4]));
//    if(rank==0) printf("\nsp_epsilon = %f \n", sp_epsilon);
//    solver.get_object()->sparse_epsilon = sp_epsilon;

    // receive sparsifivation factor from input and set it.
//    double sm_sz_prct(std::stof(argv[4]));
//    if(rank==0) printf("sm_sz_prct = %f \n", sm_sz_prct);
//    solver.set_sample_sz_percent(sm_sz_prct);

    solver.set_matrix(&A, &opts);
    solver.set_rhs(rhs);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Setup:", comm);
//    print_time(t1, t2, "Setup:", comm);

//    print_vector(solver.get_object()->grids[0].A->entry, -1, "A", comm);
//    print_vector(solver.get_object()->grids[0].rhs, -1, "rhs", comm);
*/
    // *************************** AMG - Solve ****************************
/*
    t1 = MPI_Wtime();

//    solver.solve(u, &opts);
    solver.solve_pcg(u, &opts);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Solve:", comm);
    print_time(t1, t2, "Solve:", comm);

//    print_vector(u, -1, "u", comm);
*/
    // *************************** matrix-matrix product ****************************

    double matmat_time = 0;
    int matmat_iter_warmup = 1;
    int matmat_iter = 2;

    saena::amg solver;
//    saena::matrix C(comm);

    // warm-up
    for(int i = 0; i < matmat_iter_warmup; i++){
        solver.matmat_ave(&A, &A, matmat_time);
    }

    matmat_time = 0;
    for(int i = 0; i < matmat_iter; i++){
        solver.matmat_ave(&A, &A, matmat_time);
    }

    if(!rank) printf("\nSaena matmat:\n%f\n", matmat_time / matmat_iter);

//    petsc_viewer(A.get_internal_matrix());
//    petsc_viewer(C.get_internal_matrix());
//    saena_object *obj1 = solver.get_object();

//    petsc_matmat_ave(A.get_internal_matrix(), A.get_internal_matrix(), matmat_iter);
    petsc_matmat(A.get_internal_matrix(), A.get_internal_matrix());
//    petsc_check_matmat(A.get_internal_matrix(), A.get_internal_matrix(), C.get_internal_matrix());

    // *************************** CombBLAS ****************************

//    combblas_matmult_DoubleBuff();
//    int combblas_matmult_Synch();
//    int combblas_GalerkinNew();

    // *************************** finalize ****************************

//    if(rank==0) dollar::text(std::cout);

    MPI_Finalize();
    return 0;
}