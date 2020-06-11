#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena_matrix_dense.h"
#include "saena.hpp"

#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <dollar.hpp>

#ifdef _USE_COMBBLAS_
#include "combblas_functions.h"
#endif

#ifdef _USE_PETSC_
#include "petsc_functions.h"
#endif

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

    if(argc != 2){
        if(!rank){
//            std::cout << "This is how you can generate a 3DLaplacian:\n./zfp_dense <x grid size> <y grid size> <z grid size>" << std::endl;
//            std::cout << "This is how you can generate a banded matrix:\n./zfp_dense <local size> <bandwidth>" << std::endl;
            std::cout << "This is how you can generate a random symmetric matrix:\n./zfp_dense <local size>" << std::endl;
//            std::cout << "This is how you can read a matrix from a file:\n./zfp_dense <MatrixA>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // set the number of OpenMP threads at run-time
    omp_set_num_threads(1);

#pragma omp parallel default(none) shared(rank, nprocs)
    if(!rank && omp_get_thread_num()==0)
        printf("\nnumber of processes = %d\nnumber of threads   = %d\n\n", nprocs, omp_get_num_threads());

    // *************************** initialize the matrix ****************************

    double t1 = MPI_Wtime();

    // *************************** option 1: dense ****************************

    int M(std::stoi(argv[1]));
    index_t Nbig = nprocs * M;
    saena_matrix_dense A(M, Nbig, comm);
    index_t ofst = rank * M * Nbig;
    for(index_t i = 0; i < M; ++i) {
        for(index_t j = 0; j < Nbig; ++j) {
            A.set(i, j, static_cast<value_t>( ofst + (i * Nbig + j) ));
//            if(rank==1) cout << i << "\t" << j << "\t" << A.get(i, j) << endl;
        }
    }

    A.use_zfp = true;
    A.assemble();

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
        printf("rank %d: A.M = %d, A.Nbig = %d, local A.nnz = %d, global A.nnz = %d\n",
                rank, A.M, A.Nbig, A.M * A.Nbig, A.Nbig * A.Nbig);
//        printf("threshold1 = %u,\tthreshold2 = %u\n", solver.get_object()->matmat_size_thre1, solver.get_object()->matmat_size_thre2);
    }

    // *************************** set rhs_std ****************************

//    saena::vector rhs(comm);
    std::vector<double> rhs(M);

    // ********** set rhs_std: ordered: 1, 2, 3, ... **********

    ofst = rank * M + 1;
    for(index_t i = 0; i < M; i++){
        rhs[i] = i + ofst;
//        if(rank==1) cout << i << "\t" << rhs[i] << endl;
    }

    // ********** print right-hand side **********

//    print_vector(rhs_std, -1, "rhs_std", comm);

    // *************************** warmup ****************************

    int matvec_warmup_iter = 2;
    int matvec_iter = 4;

    std::vector<double> v(M, 0);
    std::vector<double> w(M, 0);

//    print_vector(B->split, 0, "B.split", comm);
//    print_vector(solver.get_object()->grids[0].rhs, -1, "rhs", comm);
//    print_vector(v, -1, "v", comm);

    for(int i = 0; i < matvec_warmup_iter; ++i){
        A.matvec_test(rhs, v);
        A.matvec_comp(rhs, w);
    }

    // *************************** check the correctness ****************************

    double thrshld = 1e-10;
    bool bool_correct = true;
    MPI_Barrier(comm);
    if(rank == 1){
        cout << MAGENTA << "\n******************************************************\n" << COLORRESET;
        std::cout << "\nChecking the correctness:" << std::endl;
        for(int i = 0; i < v.size(); ++i){
            if(fabs(w[i] - v[i]) > thrshld){
                std::cout << i << "\t" << std::setprecision(10) << v[i] << "\t" << w[i] << "\t" << v[i] - w[i] << std::endl;
                bool_correct = false;
            }
        }

        if(bool_correct){
            cout << "\nThe solution is correct! (threshold = " << thrshld << ")\n" << endl;
            cout << MAGENTA << "\n******************************************************\n" << COLORRESET;
        }else{
            cout << "\nThe solution is " << RED << "NOT" << COLORRESET << " correct! (threshold = " << thrshld << ")\n" << endl;
            cout << MAGENTA << "******************************************************\n" << COLORRESET;
        }
    }

    // *************************** normal matvec ****************************

    A.matvec_time_init();

    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    for(int i = 0; i < matvec_iter; ++i){
        A.matvec_test(rhs, v);
    }
    t1 = MPI_Wtime() - t1;
    print_time(t1 / matvec_iter, "matvec original:", comm);

    A.matvec_time_print();

    // *************************** compressed matvec ****************************

    A.matvec_time_init();

    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    for(int i = 0; i < matvec_iter; ++i){
        A.matvec_comp(rhs, w);
    }
    t1 = MPI_Wtime() - t1;
    print_time(t1 / matvec_iter, "matvec zfp:", comm);

    A.matvec_time_print();

// *************************** finalize ****************************

    MPI_Finalize();
    return 0;
}