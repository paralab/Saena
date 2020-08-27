#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"

#ifdef _USE_PETSC_
#include "petsc_functions.h"
#endif

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

    if(argc != 2){
        if(!rank){
//            std::cout << "This is how you can generate a 3DLaplacian: ./Saena <x grid size> <y grid size> <z grid size>" << std::endl;
//            std::cout << "This is how you can generate a banded matrix: ./Saena <local size> <bandwidth>" << std::endl;
//            std::cout << "This is how you can generate a random symmetric matrix: ./Saena <local size> <density>" << std::endl;
            std::cout << "This is how you can read a matrix from a file: ./Saena <MatrixA>" << std::endl;
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

    // *************************** option 1: banded ****************************
/*
    int M(std::stoi(argv[1]));
    int band(std::stoi(argv[2]));

    saena::matrix A(comm);
    saena::band_matrix(A, M, band);

//    A.print(-1, "A");
//    A.get_internal_matrix()->print_info(-1, "A");

    saena::matrix B(A);
*/
    // *************************** option 2: random symmetric ****************************
/*
    int M(std::stoi(argv[1]));
    float dens(std::stof(argv[2]));

    saena::matrix A(comm);
    saena::random_symm_matrix(A, M, dens);

//    A.print(-1, "A");
//    A.get_internal_matrix()->print_info(-1, "A");

    saena::matrix B(A);

//    saena::matrix B(comm);
//    saena::random_symm_matrix(B, M, dens);

//    B.print(-1, "B");
//    B.get_internal_matrix()->print_info(-1, "B");
*/
    // *************************** option 3: read from file ****************************

    char *Aname(argv[1]);

    {

        saena::matrix A(comm);
//        A.read_file(Aname);
        A.read_file(Aname, "triangle");
        A.assemble();
//        A.assemble_no_scale();

        saena::matrix B(A);

        // ********** print matrix and time **********

        double t2 = MPI_Wtime();
        if (verbose) print_time(t1, t2, "Matrix Assemble:", comm);
    //    print_time(t1, t2, "Matrix Assemble:", comm);

//        A.print(0);
//        A.get_internal_matrix()->print_info(2);
//        A.get_internal_matrix()->writeMatrixToFile("matrix_folder/matrix");
//        print_vector(A.get_internal_matrix()->split, 1, "split", comm);
//        print_vector(A.get_internal_matrix()->row_local, 0, "rows", comm);
//        petsc_viewer(A.get_internal_matrix());

        // *************************** print info ****************************

        saena::amg solver;

        if (!rank) {
            printf("\nA.Mbig = %u,\tA.nnz = %ld\nB.Mbig = %u,\tB.nnz = %ld\n",
                   A.get_internal_matrix()->Mbig, A.get_internal_matrix()->nnz_g,
                   B.get_internal_matrix()->Mbig, B.get_internal_matrix()->nnz_g);
            printf("threshold1 = %u, threshold2 = %u\n\n", solver.get_object()->matmat_thre1, solver.get_object()->matmat_thre2);
        }

        // *************************** checking the correctness of matrix-matrix product ****************************

        {
//            saena::amg solver;
            saena::matrix C(comm);
            solver.matmat(&A, &B, &C);

            // check the correctness with PETSc
            petsc_check_matmat(A.get_internal_matrix(), B.get_internal_matrix(), C.get_internal_matrix());

//            C.get_internal_matrix()->print_info(0);
//            C.print(-1);

            // view A, B and C
//            petsc_viewer(A.get_internal_matrix());
//            petsc_viewer(B.get_internal_matrix());
//            petsc_viewer(C.get_internal_matrix());

//            if(!rank) printf("=====================\n\n");
        }

        // *************************** Saena matmat experiment ****************************

//        solver.matmat(&A, &B, nullptr, false, true);

        // *************************** PETSc matmat experiment ****************************

//        petsc_matmat_ave(A.get_internal_matrix(), B.get_internal_matrix(), matmat_iter);
//        petsc_matmat(A.get_internal_matrix(), B.get_internal_matrix());

    }

    // *************************** CombBLAS ****************************
/*
    {
//        combblas_matmult_DoubleBuff(Aname, Aname);
//        combblas_matmult_Synch(Aname, Aname);
        combblas_matmult_experiment(Aname, Aname, comm);
//        combblas_GalerkinNew();
    }
*/
    // *************************** finalize ****************************

    MPI_Finalize();
    return 0;
}