#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"


int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

    if(argc != 3){
        if(rank == 0){
            std::cout << "Pass two matrices: ./Saena <matrix1> <matrix2>" << std::endl;
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if(!rank) std::cout << "nprocs = " << nprocs << std::endl;

    // *************************** initialize the matrices: read from files ****************************

    double t1 = MPI_Wtime();

    char* file_name(argv[1]);
    saena::matrix A (comm);
    A.read_file(file_name);
//    A.read_file(file_name, "triangle");
    A.assemble();

    char* file_name2(argv[2]);
    saena::matrix B (comm);
    B.read_file(file_name);
//    B.read_file(file_name, "triangle");
    B.assemble();

    // ********** print matrix and time **********

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
//    print_time(t1, t2, "Matrix Assemble:", comm);

//    A.print(0);
//    A.get_internal_matrix()->print_info(0);
//    A.get_internal_matrix()->writeMatrixToFile("matrix_folder/matrix");

//    petsc_viewer(A.get_internal_matrix());
//    petsc_viewer(B.get_internal_matrix());

// *************************** checking the correctness of matrix-matrix product ****************************

    saena::amg solver;
    saena::matrix C(comm);
    solver.matmat(&A, &B, &C, true);

    // view A and C
    petsc_viewer(A.get_internal_matrix());
    petsc_viewer(C.get_internal_matrix());

    // check the correctness with PETSc
//    petsc_check_matmat(A.get_internal_matrix(), A.get_internal_matrix(), C.get_internal_matrix());

// *************************** matrix-matrix product ****************************

/*
    double matmat_time = 0;
    int matmat_iter_warmup = 5;
    int matmat_iter = 5;

    saena::amg solver;
//    saena::matrix C(comm);

    // warm-up
    for(int i = 0; i < matmat_iter_warmup; i++){
        solver.matmat_ave(&A, &A, matmat_time);
    }

    MPI_Barrier(comm);
    matmat_time = 0;
    for(int i = 0; i < matmat_iter; i++){
        solver.matmat_ave(&A, &A, matmat_time);
    }

    if(!rank) printf("Saena matmat:\n%f\n", matmat_time / matmat_iter);
*/

    // *************************** PETSc ****************************

//    petsc_matmat_ave(A.get_internal_matrix(), A.get_internal_matrix(), matmat_iter);
//    petsc_matmat(A.get_internal_matrix(), A.get_internal_matrix());

    // *************************** CombBLAS ****************************

//    combblas_matmult_DoubleBuff();
//    int combblas_matmult_Synch();
//    int combblas_GalerkinNew();

    // *************************** finalize ****************************

//    if(rank==0) dollar::text(std::cout);

    MPI_Finalize();
    return 0;
}