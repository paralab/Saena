#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"

#include "combblas_functions.h"
#include "petsc_functions.h"
//#include <petsc_functions.h>

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

    if(argc != 3){
        if(!rank){
//            std::cout << "This is how you can generate a 3DLaplacian:\n./Saena_zfp <x grid size> <y grid size> <z grid size>" << std::endl;
//            std::cout << "This is how you can generate a banded matrix:\n./Saena_zfp <local size> <bandwidth>" << std::endl;
            std::cout << "This is how you can generate a random symmetric matrix:\n./Saena_zfp <local size> <density>" << std::endl;
//            std::cout << "This is how you can read a matrix from a file:\n./Saena_zfp <MatrixA>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

#pragma omp parallel default(none) shared(rank, nprocs)
    if(!rank && omp_get_thread_num()==0)
        printf("\nnumber of processes = %d\nnumber of threads   = %d\n\n", nprocs, omp_get_num_threads());

    // *************************** initialize the matrix ****************************

    double t1 = MPI_Wtime();

    // *************************** option 1: banded ****************************
/*
    int M(std::stoi(argv[1]));
    int band(std::stoi(argv[2]));

    saena::matrix A(comm);
    saena::band_matrix(A, M, band);

//    A.print(-1, "A");
//    A.get_internal_matrix()->print_info(-1, "A");
*/
    // *************************** option 2: random symmetric ****************************

    int M(std::stoi(argv[1]));
    float dens(std::stof(argv[2]));

    saena::matrix A(comm);
    saena::random_symm_matrix(A, M, dens);

//    A.print(-1, "A");
//    A.get_internal_matrix()->print_info(-1, "A");

    // *************************** option 3: read from file ****************************
/*
    char *Aname(argv[1]);
    saena::matrix A(comm);
//    A.read_file(Aname);
    A.read_file(Aname, "triangle");
    A.assemble();

//    saena::matrix B(A);
*/
    // ********** print matrix info and time **********

    double t2 = MPI_Wtime();
    if (verbose) print_time(t1, t2, "Matrix Assemble:", comm);
//    print_time(t1, t2, "Matrix Assemble:", comm);

//    A.print(0);
//    A.get_internal_matrix()->print_info(0);
//    A.get_internal_matrix()->writeMatrixToFile("matrix_folder/matrix");
//    petsc_viewer(A.get_internal_matrix());
//    print_vector(A.get_internal_matrix()->split, 0, "split", comm);

    if (!rank) {
        printf("A.Mbig = %u,\tA.nnz = %ld\n", A.get_internal_matrix()->Mbig, A.get_internal_matrix()->nnz_g);
//        printf("threshold1 = %u,\tthreshold2 = %u\n", solver.get_object()->matmat_size_thre1, solver.get_object()->matmat_size_thre2);
    }

    // *************************** set rhs_std ****************************

    saena::vector rhs(comm);
    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<double> rhs_std;

    // ********** 1 - set rhs_std: random **********

    rhs_std.resize(num_local_row);
    generate_rhs_old(rhs_std);

    std::vector<double> tmp1 = rhs_std;
    A.get_internal_matrix()->matvec(tmp1, rhs_std);

    index_t my_split;
    saena::find_split((index_t)rhs_std.size(), my_split, comm);

    rhs.set(&rhs_std[0], (index_t)rhs_std.size(), my_split);
    rhs.assemble();

    // ********** 2 - set rhs_std: ordered: 1, 2, 3, ... **********

//    rhs_std.resize(num_local_row);
//    for(index_t i = 0; i < A.get_num_local_rows(); i++)
//        rhs_std[i] = i + 1 + A.get_internal_matrix()->split[rank];

    // ********** 3 - set rhs_std: Laplacian **********

    // don't set the size for this method
//    saena::laplacian3D_set_rhs(rhs_std, mx, my, mz, comm);

    // ********** 4 - set rhs_std: read from file **********
/*
    char* Vname(argv[2]);
//    saena::read_vector_file(rhs_std, A, Vname, comm);
    read_vector_file(rhs_std, A.get_internal_matrix(), Vname, comm);

    // set rhs_std
//    A.get_internal_matrix()->matvec(v, rhs_std);
//    rhs_std = v;

    index_t my_split;
    saena::find_split((index_t)rhs_std.size(), my_split, comm);

    rhs.set(&rhs_std[0], (index_t)rhs_std.size(), my_split);
    rhs.assemble();
*/
    // ********** print right-hand side **********

//    print_vector(rhs_std, -1, "rhs_std", comm);
//    rhs.print_entry(-1);

    // *************************** set u0 ****************************

    std::vector<double> u(num_local_row, 0);

    // *************************** AMG - Setup ****************************

    t1 = MPI_Wtime();

//    int max_level             = 2; // this is moved to saena_object.
    int vcycle_num            = 400;
    double relative_tolerance = 1e-8;
    std::string smoother      = "chebyshev"; // choices: "jacobi", "chebyshev"
    int preSmooth             = 3;
    int postSmooth            = 3;

    saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
//    saena::options opts((char*)"options001.xml");
    saena::amg solver;
    solver.set_verbose(verbose); // set verbose at the beginning of the main function.

    solver.set_matrix(&A, &opts);
    solver.set_rhs(rhs);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Setup:", comm);
//    print_time(t1, t2, "Setup:", comm);

//    print_vector(solver.get_object()->grids[0].A->entry, -1, "A", comm);
//    print_vector(solver.get_object()->grids[0].rhs, -1, "rhs", comm);
//    print_vector(A.get_internal_matrix()->split, 0, "split", comm);

    // *************************** zfp ****************************

    int matvec_warmup_iter = 3;
    int matvec_iter = 3;

//    solver.get_object()->grids[0].A->allocate_zfp(solver.get_object()->grids[0].rhs);
    saena_matrix *B = solver.get_object()->grids[0].A;

    std::vector<double> v(B->split[rank+1] - B->split[rank], 0);
    std::vector<double> w(B->split[rank+1] - B->split[rank], 0);

//    print_vector(B->split, 0, "B.split", comm);
//    print_vector(solver.get_object()->grids[0].rhs, -1, "rhs", comm);
//    print_vector(v, -1, "v", comm);

    for(int i = 0; i < matvec_warmup_iter; ++i){
        B->matvec_sparse(solver.get_object()->grids[0].rhs, v);
    }

    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    for(int i = 0; i < matvec_iter; ++i){
        B->matvec_sparse(solver.get_object()->grids[0].rhs, v);
    }
    t1 = MPI_Wtime() - t1;
    print_time(t1 / matvec_iter, "matvec original:", comm);


    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    for(int i = 0; i < matvec_iter; ++i){
        B->matvec_sparse_zfp(solver.get_object()->grids[0].rhs, w);
    }
    t1 = MPI_Wtime() - t1;
    print_time(t1 / matvec_iter, "matvec zfp:", comm);

    B->zfp_print_time();

    // *************************** zfp: check the accuracy ****************************

//    if(rank == 1){
//        for(int i = 0; i < v.size(); ++i){
//            if(fabs(w[i] - v[i]) > 1e-8)
//                std::cout << std::setprecision(10) << v[i] << "\t" << w[i] << "\t" << v[i] - w[i] << std::endl;
//        }
//    }

    // *************************** AMG - Solve ****************************
/*
    t1 = MPI_Wtime();

//    solver.solve(u, &opts);
    solver.solve_pcg(u, &opts);
//    solver.solve_pGMRES(u, &opts);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Solve:", comm);
    print_time(t1, t2, "Solve:", comm);

//    print_vector(u, -1, "u", comm);
*/
// *************************** finalize ****************************

//    if(rank==0) dollar::text(std::cout);

    MPI_Finalize();
    return 0;
}