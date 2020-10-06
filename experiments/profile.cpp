// to run this code on two generated examples from homg matrices and random righ-hand sides:
// ./Saena ../data/81s4x8o1mu1.bin ../data/vectors/v81.bin
// ./Saena ../data/289s8x16o1mu1.bin ../data/vectors/v289.bin

#include "grid.h"
#include "saena.hpp"
#include "data_struct.h"

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(argc != 4) {
        if(rank == 0) {
            cout << "Usage: ./profile mx my mz" << endl;
//            cout << "Usage: ./profile mx" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    // set the number of OpenMP threads at run-time
    omp_set_num_threads(1);

    // *************************** set the scaling factor ****************************

    bool scale = true;

    // *************************** initialize the matrix ****************************

    int mx(std::stoi(argv[1]));
    int my(std::stoi(argv[2]));
    int mz(std::stoi(argv[3]));

//    if(verbose){
        MPI_Barrier(comm);
        if(rank==0) printf("3D Laplacian: grid size: x = %d, y = %d, z = %d \n", mx, my, mz);
        MPI_Barrier(comm);
//    }

    // timing the matrix setup phase
    double t1 = omp_get_wtime();

    saena::matrix A(comm);
//    saena::laplacian2D_old(&A, mx);
//    saena::laplacian3D_old(&A, mx);
    saena::laplacian3D(&A, mx, my, mz);

    double t2 = omp_get_wtime();
//    print_time(t1, t2, "Matrix Assemble:", comm);

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

    saena::laplacian3D_set_rhs(rhs_std, mx, my, mz, comm);

    index_t my_split = 0;
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
    double relative_tolerance = 1e-8;
    std::string smoother      = "chebyshev";
    int    preSmooth          = 2;
    int    postSmooth         = 2;
    saena::options opts(solver_max_iter, relative_tolerance, smoother, preSmooth, postSmooth);

    // 2- read the options from an xml file
//    saena::options opts((char*)"options001.xml");

    // 3- use the default options
//    saena::options opts;

    t1 = omp_get_wtime();

    saena::amg solver;
    solver.set_multigrid_max_level(4);
    solver.set_scale(scale);
    solver.set_matrix(&A, &opts);
    solver.set_rhs(rhs);

    t2 = omp_get_wtime();
    print_time(t1, t2, "Setup:", comm);

    // *************************** AMG - Solve ****************************
    // solve the system Au = rhs

    t1 = omp_get_wtime();

    // solve the system using AMG as the solver
//    solver.solve(u, &opts);

    // solve the system, using pure CG.
//    solver.solve_CG(u, &opts);

    // solve the system, using AMG as the preconditioner. this is preconditioned conjugate gradient (PCG).
    for(int i = 0; i < 2; ++i)
        solver.solve_pCG(u, &opts);

    // solve the system, using pure GMRES.
//    solver.solve_GMRES(u, &opts);

    // solve the system, using AMG as the preconditioner. this is preconditioned GMRES.
//    solver.solve_pGMRES(u, &opts);

    t2 = omp_get_wtime();
//    print_time(t1, t2, "Solve:", comm);

    // *************************** print or write the solution ****************************

//    print_vector(u, -1, "solution", comm);
//    print_vector(u_direct, -1, "solution", comm);
//    write_to_file_vec(u, "solution", comm);

    // *************************** check correctness of the solution 1 ****************************

    // A is scaled. read it from the file and don't scale.
/*
    saena::matrix AA (comm);
    AA.read_file(file_name);
    AA.assemble(false);

    saena_matrix *AAA = AA.get_internal_matrix();
    std::vector<double> Au(num_local_row, 0);
    std::vector<double> sol = u;
//	std::vector<double> sol = u_direct; // the SuperLU solution
    AAA->matvec(sol, Au);

    bool bool_correct = true;
    if(rank==0){
        std::stringstream buf;
        print_sep();
        printf("Checking the correctness of the solution:\n");
//        printf("Au \t\trhs_std \t\tAu - rhs_std \n");
        for(index_t i = 0; i < num_local_row; ++i){
            if(fabs(Au[i] - rhs_std[i]) > 1e-8){
                bool_correct = false;
                break;
//                printf("%.12f \t%.12f \t%.12f \n", Au[i], rhs_std[i], Au[i] - rhs_std[i]);
            }
        }
        if(bool_correct){
            buf << "\nThe solution is correct!\n";
            std::cout << buf.str();
            print_sep();
        }
        else{
            buf << "\nThe solution is " << RED << "NOT" << COLORRESET << " correct!\n";
            std::cout << buf.str();
            print_sep();
        }
    }
*/

/*
    std::vector<double> res(num_local_row,0);
    for(index_t i = 0; i < num_local_row; ++i){
        res[i] = Au[i] - rhs_std[i];
//        printf("%.12f \t%.12f \t%.12f \n", Au[i], rhs_std[i], Au[i] - rhs_std[i]);
    }

    float norm1 = pnorm(res, comm);
    if(!rank) std::cout << "norm(Au-b)     = " << norm1 << "\n";
*/
    // *************************** check correctness of the solution 2 ****************************

/*
    bool_correct = true;
    if(rank==0){
        printf("Checking the correctness of GMRES with SuperLU:\n");
//        printf("Correct u \t\tGMRES u\n");
        for(index_t i = 0; i < num_local_row; i++){
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