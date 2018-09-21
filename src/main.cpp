#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"

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

//#pragma omp parallel
//    if(rank==0 && omp_get_thread_num()==0) printf("\nnumber of processes = %d, number of threads = %d\n\n", nprocs, omp_get_num_threads());

/*
    if(argc != 4){
        if(rank == 0)
            std::cout << "This is how to make a 3DLaplacian: ./Saena <x grid size> <y grid size> <z grid size>" << std::endl;
        MPI_Finalize();
        return -1;}
*/
/*
    if(argc != 2){
        if(rank == 0){
            std::cout << "Usage: ./Saena <MatrixA>" << std::endl;
            std::cout << "Matrix file should be in triples format." << std::endl;}
        MPI_Finalize();
        return -1;}
*/
/*
    if(argc != 3){
        if(rank == 0){
            std::cout << "Usage: ./Saena <MatrixA> <rhs_vec>" << std::endl;
            std::cout << "Matrix file should be in triples format." << std::endl;}
        MPI_Finalize();
        return -1;}
*/
/*
    if(argc != 3){
        if(rank == 0){
            std::cout << "Usage: ./Saena <MatrixA> <MatrixA_new>" << std::endl;
            std::cout << "Matrix file should be in triples format." << std::endl;}
        MPI_Finalize();
        return -1;}
*/
/*
    if(argc != 4){
        if(rank == 0){
            std::cout << "Usage: ./Saena <MatrixA> <rhs_vec> <MatrixA_new>" << std::endl;
            std::cout << "Matrix file should be in triples format." << std::endl;}
        MPI_Finalize();
        return -1;}
*/
    // *************************** initialize the matrix ****************************

    // ******** 1 - initialize the matrix: laplacian *************

    int mx(std::stoi(argv[1]));
    int my(std::stoi(argv[2]));
    int mz(std::stoi(argv[3]));

    if(verbose){
        MPI_Barrier(comm);
        if(rank==0) printf("3D Laplacian: grid size: x = %d, y = %d, z = %d \n", mx, my, mz);
        MPI_Barrier(comm);}

    // timing the matrix setup phase
    double t1 = MPI_Wtime();

    saena::matrix A(comm);
    saena::laplacian3D(&A, mx, my, mz);
//    saena::laplacian2D_old(&A, mx);
//    saena::laplacian3D_old(&A, mx);
//    A.print_entry(-1);

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);

    // ******** 2 - initialize the matrix: read from file *************
/*
    double t1 = MPI_Wtime();

    char* file_name(argv[1]);
    saena::matrix A (file_name, comm);
    A.assemble();

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
    print_time(t1, t2, "Matrix Assemble:", comm);

//    A.print(0);
//    A.get_internal_matrix()->print_info(0);
*/
    // *************************** set rhs ****************************

    unsigned int num_local_row = A.get_num_local_rows();
//    std::vector<double> rhs(num_local_row);

    // ********** 1 - set rhs: random **********

//    generate_rhs_old(rhs);
//    print_vector(rhs, -1, "rhs", comm);

    // ********** 2 - set rhs: ordered: 1, 2, 3, ... **********

//    for(index_t i = 0; i < A.get_num_local_rows(); i++)
//        rhs[i] = i + 1 + A.get_internal_matrix()->split[rank];
//    print_vector(rhs, -1, "rhs", comm);

    // ********** 3 - set rhs: Laplacian **********

    std::vector<double> rhs; // don't set the size for this method
    saena::laplacian3D_set_rhs(rhs, mx, my, mz, comm);
//    print_vector(rhs, -1, "rhs", comm);

    // ********** 4 - set rhs: read from file **********
/*
    char* Vname(argv[2]);
//    char* Vname(argv[3]);

    // check if the size of rhs match the number of rows of A
    struct stat st;
    stat(Vname, &st);
    unsigned int rhs_size = st.st_size / sizeof(double);
    if(rhs_size != A.get_internal_matrix()->Mbig){
        if(rank==0) printf("Error: Size of RHS does not match the number of rows of the LHS matrix!\n");
        if(rank==0) printf("Number of rows of LHS = %d\n", A.get_internal_matrix()->Mbig);
        if(rank==0) printf("Size of RHS = %d\n", rhs_size);
        MPI_Finalize();
        return -1;
    }

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, Vname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) std::cout << "Unable to open the rhs vector file!" << std::endl;
        MPI_Finalize();
        return -1;
    }

    // define the size of v as the local number of rows on each process
    std::vector <double> v(num_local_row);
    double* vp = &(*(v.begin()));

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A.get_internal_matrix()->split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, vp, num_local_row, MPI_DOUBLE, &status);

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

    // set rhs
//    A.get_internal_matrix()->matvec(v, rhs);
    rhs = v;
*/
    // *************************** set u0 ****************************

    std::vector<double> u(num_local_row, 0);

    // *************************** AMG - Setup ****************************

    t1 = MPI_Wtime();

//    int max_level             = 2; // this is moved to saena_object.
    int vcycle_num            = 100;
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
//    solver.set_rhs(rhs);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Setup:", comm);
    print_time(t1, t2, "Setup:", comm);

//    print_vector(solver.get_object()->grids[0].A->entry, -1, "A", comm);
//    print_vector(solver.get_object()->grids[0].rhs, -1, "rhs", comm);

    // *************************** AMG - Solve ****************************
/*
    t1 = MPI_Wtime();

//    solver.solve(u, &opts);
    solver.solve_pcg(u, &opts);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Solve:", comm);
    print_time(t1, t2, "Solve:", comm);
*/
//    print_vector(u, -1, "u", comm);

    // write the Laplacian matrix to file.
//    std::string mat_name = "3DLap";
//    mat_name += std::to_string(mx);
//    mat_name += "-";
//    mat_name += std::to_string(my);
//    mat_name += "-";
//    mat_name += std::to_string(mz);
//    solver.get_object()->writeMatrixToFileA(A.get_internal_matrix(), mat_name);

    // *************************** check correctness of the solution ****************************
    // A is scaled. read it from the file and don't scale.
/*
    saena::matrix AA (file_name, comm);
    AA.assemble_no_scale();
    saena_matrix *AAA = AA.get_internal_matrix();
    std::vector<double> Au(num_local_row, 0);
    std::vector<double> sol = u;
    AAA->matvec(sol, Au);

    bool bool_correct = true;
    if(rank==0){
        printf("\nChecking the correctness of the Saena solution by Saena itself:\n");
        printf("Au \t\trhs \t\tAu-rhs \n");
        for(index_t i = 0; i < num_local_row; i++){
            if(fabs(Au[i] - rhs[i]) > 1e-10){
                bool_correct = false;
                printf("%.10f \t%.10f \t%.10f \n", Au[i], rhs[i], Au[i] - rhs[i]);
            }
        }
        if(bool_correct)
            printf("\n******* The solution was correct! *******\n\n");
        else
            printf("\n******* The solution was NOT correct! *******\n\n");
    }
*/
    // *************************** finalize ****************************

    if(rank==0) dollar::text(std::cout);

//    A.destroy();
//    solver.destroy();
    MPI_Finalize();
    return 0;
}