// to run this code on two generated examples from homg matrices and random righ-hand sides:
// ./Saena ../data/81s4x8o1mu1.bin ../data/vectors/v81.bin
// ./Saena ../data/289s8x16o1mu1.bin ../data/vectors/v289.bin

#include <iostream>
#include "mpi.h"

#include "grid.h"
#include "saena.hpp"

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(argc != 3)
    {
        if(rank == 0)
        {
            cout << "Usage: ./Saena <MatrixA> <rhs>" << endl;
            cout << "Matrix file should be in COO format in column-major order." << endl;
        }
        MPI_Finalize();
        return -1;
    }

    // *************************** set the scaling factor ****************************

    bool scale = true;

    // *************************** initialize the matrix ****************************
    // there are two ways to create a matrix:

    // 1- read from file: pass as an argument in the command line
    // example: ./Saena ../data/81s4x8o1mu1.bin
    char* file_name(argv[1]);
    saena::matrix A (comm);
    A.read_file(file_name);

    // set the p_order of the input matrix A.
    int p_order = 2;
    A.set_p_order(p_order);

    // 2- use the set functions
//    saena::matrix A(comm); // comm: MPI_Communicator
//    A.add_duplicates(true); // in case of duplicates add the values.

    // 2-I) pass the entries one by one
//    A.set(0, 0, 0.1);
//    A.set(0, 1, 0.2);
//    A.set(1, 0, 0.3);
//    A.set(1, 1, 0.4);

    // 2-II) three vectors (or arrays) can be passed to the set functions
    // I: row values of type unsinged
    // J: column values of type unsinged
    // V: entry values of type double
//    std::vector<unsigned> I, J;
//    std::vector<double> V;
//    I = {0,0,1,1}; J = {0,1,0,1}; V = {0.1, 0.2, 0.3, 0.4};
//    A.set(&I[0], &J[0], &V[0], I.size());

    // after setting the matrix entries, the assemble function should be called.
    // pass false to not scale the matrix. default is true.
    A.assemble(scale);

    // the print function can be used to print the matrix entries on a specific processor (pass the
    // processor rank to the print function), or on all the processors (pass -1).
//    A.print(0);

    // *************************** set rhs: read from file ****************************

    // vector should have the following format:
    // first line shows the value in row 0, second line shows the value in row 1, ...
    // entries should be in double.

    // set the size of rhs as the local number of rows on each process
    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<double> rhs_std;

    char* Vname(argv[2]);
    read_from_file_rhs(rhs_std, A.get_internal_matrix(), Vname, comm);

//    write_to_file_vec(rhs_std, "rhs_std", comm);

    // set rhs_std
//    A.get_internal_matrix()->matvec(v, rhs_std);
//    rhs_std = v;

    index_t my_split;
    saena::find_split((index_t)rhs_std.size(), my_split, comm);

    saena::vector rhs(comm);
    rhs.set(&rhs_std[0], (index_t)rhs_std.size(), my_split);
    rhs.assemble();

    // *************************** set u0 ****************************

    // u is the initial guess. at the end, it will be the solution.
    std::vector<double> u(num_local_row, 0); // initial guess = 0

    // *************************** AMG - Setup ****************************
    // There are 3 ways to set options:

    // 1- set them manually
    int    solver_max_iter    = 1000;
    double relative_tolerance = 1e-12;
    std::string smoother      = "chebyshev";
    int    preSmooth          = 3;
    int    postSmooth         = 3;
    saena::options opts(solver_max_iter, relative_tolerance, smoother, preSmooth, postSmooth);

    // 2- read the options from an xml file
//    saena::options opts((char*)"options001.xml");

    // 3- use the default options
//    saena::options opts;

    saena::amg solver;
    solver.set_multigrid_max_level(1);
    solver.set_matrix(&A, &opts);

    solver.set_rhs(rhs, scale);

    // *************************** AMG - Solve ****************************
    // solve the system Au = rhs

    // solve the system using AMG as the solver
    std::vector<double> u_direct(num_local_row, 0); // initial guess = 0
//    solver.solve(u_direct, &opts, scale);

    // solve the system, using AMG as the preconditioner. this is preconditioned conjugate gradient (PCG).
//    solver.solve_pcg(u, &opts, scale);

    // solve the system, using AMG as the preconditioner. this is preconditioned GMRES.
    solver.solve_pGMRES(u, &opts, scale);

    // *************************** print or write the solution ****************************

//    print_vector(u, -1, "solution", comm);
//    print_vector(u_direct, -1, "solution", comm);
//    write_to_file_vec(u, "solution", comm);

    // *************************** check correctness of the solution 2 ****************************

    // A is scaled. read it from the file and don't scale.

    saena::matrix AA (comm);
    AA.read_file(file_name);
    AA.assemble(false);

    saena_matrix *AAA = AA.get_internal_matrix();
    std::vector<double> Au(num_local_row, 0);
    std::vector<double> sol = u;
//    std::vector<double> sol = u_direct; // the SuperLU solution
    AAA->matvec(sol, Au);

    bool bool_correct = true;
    if(rank==0){
        printf("\n******************************************************\n");
        printf("Checking the correctness of the solution:\n");
//        printf("Au \t\trhs_std \t\tAu - rhs_std \n");
        for(index_t i = 0; i < num_local_row; i++){
            if(fabs(Au[i] - rhs_std[i]) > 1e-12){
                bool_correct = false;
//                printf("%.12f \t%.12f \t%.12f \n", Au[i], rhs_std[i], Au[i] - rhs_std[i]);
            }
        }
        if(bool_correct){
            printf("\nThe solution is correct!\n");
            printf("\n******************************************************\n");
        }
        else{
            printf("\nThe solution is NOT correct!\n");
            printf("\n******************************************************\n");
        }
    }

    // *************************** check correctness of the solution ****************************
/*
    bool_correct = true;
    if(rank==0){
        printf("Checking the correctness of GMRES with SuperLU:\n");
//        printf("Correct u \t\tGMRES u\n");
//        for(index_t i = 0; i < num_local_row; i++){
        for(index_t i = 30; i < 40; i++){ //TODO fix the range
//            if(fabs(u_direct[i] - u[i]) > 1e-12){
                bool_correct = false;
                printf("%.6f \t%.6f \t%.6f\n", u_direct[i], u[i], u_direct[i] - u[i]);
//            }
        }
        if(bool_correct){
            printf("\nGMRES matches SuperLU!\n");
            printf("\n******************************************************\n");
        }
        else{
            printf("\nGMRES does NOT match SuperLU!\n");
            printf("\n******************************************************\n");
        }
    }
*/
    // *************************** Destroy ****************************

    A.destroy();
    solver.destroy();
    MPI_Finalize();
    return 0;
}