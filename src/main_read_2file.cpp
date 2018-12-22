#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"
//#include "petsc_functions.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <omp.h>
#include <dollar.hpp>
#include "mpi.h"


int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

    if(argc != 4){
        if(rank == 0) {
            std::cout << "Usage: ./Saena <MatrixA> <MatrixB>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // *************************** initialize the matrix ****************************

    double t1 = MPI_Wtime();

    // ******** 1 - initialize the matrix: laplacian *************
/*
    int mx(std::stoi(argv[1]));
    int my(std::stoi(argv[2]));
    int mz(std::stoi(argv[3]));

    if(verbose){
        MPI_Barrier(comm);
        if(rank==0) printf("3D Laplacian: grid size: x = %d, y = %d, z = %d \n", mx, my, mz);
        MPI_Barrier(comm);}


    saena::matrix A(comm);
    saena::laplacian3D(&A, mx, my, mz);
//    saena::laplacian2D_old(&A, mx);
//    saena::laplacian3D_old(&A, mx);
*/
    // ******** 2 - initialize the matrix: read from file *************

    char* file_name(argv[1]);
    saena::matrix A (comm);
    A.read_file(file_name);
//    A.read_file(file_name, "triangle");
    A.assemble();
//    A.assemble_writeToFile("writeMatrix");

    char* file_name2(argv[2]);
    saena::matrix A2 (comm);
    A2.read_file(file_name2);
    A2.assemble();

    // ********** print matrix and time **********

    double t2 = MPI_Wtime();
//    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
//    print_time(t1, t2, "Matrix Assemble:", comm);

//    A.print(0);
//    A.get_internal_matrix()->print_info(0);
//    A.get_internal_matrix()->writeMatrixToFile("writeMatrix");

//    petsc_viewer(A.get_internal_matrix());

    // *************************** set rhs ****************************

    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<double> rhs;

    // ********** 1 - set rhs: random **********

//    rhs.resize(num_local_row);
//    generate_rhs_old(rhs);

    // ********** 2 - set rhs: ordered: 1, 2, 3, ... **********

//    rhs.resize(num_local_row);
//    for(index_t i = 0; i < A.get_num_local_rows(); i++)
//        rhs[i] = i + 1 + A.get_internal_matrix()->split[rank];

    // ********** 3 - set rhs: Laplacian **********

    // don't set the size for this method
//    saena::laplacian3D_set_rhs(rhs, mx, my, mz, comm);

    // ********** 4 - set rhs: read from file **********

    char* Vname(argv[3]);
    saena::read_vector_file(rhs, A, Vname, comm);
//    read_vector_file(rhs, A.get_internal_matrix(), Vname, comm);

    // set rhs
//    A.get_internal_matrix()->matvec(v, rhs);
//    rhs = v;

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

    solver.set_matrix(&A, &opts);
    solver.set_rhs(rhs);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Setup:", comm);
    print_time(t1, t2, "Setup:", comm);

//    print_vector(solver.get_object()->grids[0].A->entry, -1, "A", comm);
//    print_vector(solver.get_object()->grids[0].rhs, -1, "rhs", comm);

    // *************************** AMG - Solve ****************************

//    t1 = MPI_Wtime();

//    solver.solve(u, &opts);
    solver.solve_pcg(u, &opts);

//    t2 = MPI_Wtime();
//    print_time(t1, t2, "Solve:", comm);

//    print_vector(u, -1, "u", comm);

    // *************************** lazy update ****************************

//    t1 = MPI_Wtime();
    solver.update2(&A2);
    solver.solve_pcg(u, &opts);

//    t2 = MPI_Wtime();
//    print_time(t1, t2, "Solve update:", comm);

    // *************************** test for lazy update functions ****************************
/*
    saena_matrix* A_saena = A.get_internal_matrix();
    std::vector<index_t> rown(A.get_local_nnz());
    std::vector<index_t> coln(A.get_local_nnz());
    std::vector<value_t> valn(A.get_local_nnz());
    for(nnz_t i = 0; i < A.get_local_nnz(); i++){
        rown[i] = A_saena->entry[i].row;
        coln[i] = A_saena->entry[i].col;
        valn[i] = 2 * A_saena->entry[i].val;
//        valn[i] = 0.33;
//        if(i<50 && rank==1) printf("%f \t%f \n", A_saena->entry[i].val, valn[i]);
    }

    saena::matrix A_new(comm);
    A_new.set(&rown[0], &coln[0], &valn[0], rown.size());
    A_new.assemble();
//    A_new.assemble_no_scale();
//    solver.update1(&A_new);

//    solver.get_object()->matrix_diff(*solver.get_object()->grids[0].A, *A_new.get_internal_matrix());

    if(rank==0){
        for(nnz_t i = 0; i < 50; i++){
//            std::cout << A.get_internal_matrix()->entry[i] << "\t" << A_new.get_internal_matrix()->entry[i] << std::endl;
            std::cout << A_saena->entry[i] << "\t" << A_new.get_internal_matrix()->entry[i] << std::endl;
        }
    }
*/
    // *************************** finalize ****************************

//    if(rank==0) dollar::text(std::cout);

//    A.destroy();
//    solver.destroy();
    MPI_Finalize();
    return 0;
}