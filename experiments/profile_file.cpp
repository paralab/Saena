// to run this code on two generated examples from homg matrices and random righ-hand sides:
// ./Saena ../data/81s4x8o1mu1.bin ../data/vectors/v81.bin
// ./Saena ../data/289s8x16o1mu1.bin ../data/vectors/v289.bin

#include "grid.h"
#include "saena.hpp"
#include "data_struct.h"
#include "petsc_functions.h"
#include "aux_functions2.h"

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(argc < 3 || argc > 4) {
        if(!rank) {
            cout << "Usage: ./profile matrix_file rhs_file <optional: options file>" << endl;
//            cout << "Usage: ./profile matrix_file rhs_file max_level" << endl;
//            cout << "Usage: ./profile matrix_file rhs_file" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    // set the number of OpenMP threads at run-time
//    omp_set_num_threads(1);

    // *************************** set the scaling factor ****************************

    bool scale = false;

    bool use_petsc = false;
    {
        saena::options opts_tmp;
        if (argc == 4) {
            const string optsfile(argv[3]);
            opts_tmp.set_from_file(optsfile);
            if (!opts_tmp.get_petsc_solver().empty()) {
                use_petsc = true;
            }
        }
    }

    // *************************** initialize the matrix ****************************

    // timing the matrix setup phase
    double t1 = omp_get_wtime();

    saena::matrix A(comm);
    char* file_name(argv[1]);
    if(!rank) printf("matrix file:  %s\n", file_name);
    A.set_remove_boundary(!use_petsc); // if using petsc dont remove boundary, otherwise remove
    A.read_file(file_name);
    A.assemble(scale);

    double t2 = omp_get_wtime();
//    print_time(t1, t2, "Matrix Assemble:", comm);

    // the print function can be used to print the matrix entries on a specific processor (pass the
    // processor rank to the print function), or on all the processors (pass -1).
//    A.print(0);

    // write matrix to a file. pass the name as the argument.
//    A.writeMatrixToFile("mat");
//    petsc_viewer(A.get_internal_matrix());

    // *************************** set rhs: read from file ****************************

    // vector should have the following format:
    // first line shows the value in row 0, second line shows the value in row 1, ...
    // entries should be in double.

    auto orig_split = A.get_orig_split();
    const nnz_t orig_sz = orig_split[rank + 1] - orig_split[rank];

    // read rhs from file
    //-------------------
    auto *rhs_std = saena_aligned_alloc<value_t>(orig_sz);
    assert(rhs_std);
    char* Vname(argv[2]);
    if(!rank) printf("rhs file:     %s\n", Vname);
    read_from_file_rhs(rhs_std, orig_split, Vname, comm);
    //-------------------

    // generate random rhs
    //--------------------
//    auto *rhs_std = saena_aligned_alloc<value_t>(A.get_num_local_rows());
//    assert(rhs_std);
//    generate_rhs_old(rhs_std);
    //--------------------

//    print_vector(rhs_std, -1, "rhs_std", comm);

    saena::vector rhs(comm);
    rhs.set(rhs_std, orig_sz, orig_split[rank]);
    rhs.assemble();

//    rhs.print_entry(-1);

    // *************************** set u0 ****************************

//    std::vector<double> u;
    value_t *u = nullptr;

    // *************************** AMG - Setup ****************************
    // There are 3 ways to set options:

    // 1- set them manually
//    int    solver_max_iter    = 200;
//    double relative_tolerance = 1e-8;
//    std::string smoother      = "chebyshev";
//    int    preSmooth          = 3;
//    int    postSmooth         = 3;
//    saena::options opts(solver_max_iter, relative_tolerance, smoother, preSmooth, postSmooth);

    // 2- use the default options
    saena::options opts;

    // 3- read the options from an xml file
    if(argc == 4){
        const string optsfile(argv[3]);
        if(!rank) printf("options file: %s\n", optsfile.c_str());
        opts.set_from_file(optsfile);
        A.set_eig(optsfile); // set eigenvalue from the options file, if it is provided.
    }

    t1 = omp_get_wtime();

    bool free_amg = false;
    saena::amg solver;
//    solver.set_dynamic_levels(true);
//    int max_level(std::stoi(argv[3]));
//    solver.set_multigrid_max_level(max_level);
    solver.set_scale(scale);
    solver.set_matrix(&A, &opts); free_amg = true;
    solver.set_rhs(rhs);

    t2 = omp_get_wtime();
    print_time(t1, t2, "Setup:", comm);

    // *************************** AMG - Solve ****************************
    // solve the system Au = rhs

    if(use_petsc){
        solver.solve_petsc(u, &opts);
        A.destroy();
        if(free_amg)
            solver.destroy();
        saena_free(rhs_std);
        saena_free(u);
        MPI_Finalize();
        return 0;
    }

    int warmup_iter = 5;
    int solve_iter  = 10;
    
    // warm-up
    if(warmup_iter != 0){
        solver.solve_pCG(u, &opts);
        for(int i = 1; i < warmup_iter; ++i)
            solver.solve_pCG(u, &opts, false);
        saena_free(u);
    }

    t1 = omp_get_wtime();

    // solve the system using AMG as the solver
//    solver.solve(u, &opts);

    // solve the system, using pure CG.
//    solver.solve_CG(u, &opts);

    // solve the system, using AMG as the preconditioner. this is preconditioned conjugate gradient (PCG).
    for(int i = 0; i < solve_iter; ++i)
        solver.solve_pCG(u, &opts, false);

    // solve the system, using pure GMRES.
//    solver.solve_GMRES(u, &opts);

    // solve the system, using AMG as the preconditioner. this is preconditioned GMRES.
//    solver.solve_pGMRES(u, &opts);

    t2 = omp_get_wtime();
    print_time(t1 / solve_iter, t2 / solve_iter, "Solve:", comm);

    // *************************** print or write the solution ****************************

//    print_vector(u, -1, "solution", comm);
//    print_vector(u_direct, -1, "solution", comm);
//    write_to_file_vec(u, "solution", comm);

    // *************************** profile solve and matvecs ****************************

    saena_free(u);
    solver.solve_pCG_profile(u, &opts);

    // profile matvec times on all multigrid levels
    solver.profile_matvecs();
//    solver.profile_matvecs_breakdown();

    // *************************** check correctness of the solution 1 ****************************

    // A is scaled. read it from the file and don't scale.
#if 0
    index_t num_local_row = A.get_num_local_rows();
    saena::matrix Ap(comm);
    saena::laplacian3D(&Ap, mx, my, mz, false);

    std::vector<double> u_petsc(num_local_row);
    petsc_solve(Ap.get_internal_matrix(), rhs_std, u_petsc, relative_tolerance);
    bool bool_correct = true;
    if(rank==0){
        std::stringstream buf;
        print_sep();
        printf("Checking the correctness of the solution:\n");
//        printf("Au \t\trhs_std \t\tAu - rhs_std \n");
        for(index_t i = 0; i < num_local_row; ++i){
            if(fabs(u[i] - u_petsc[i]) > 1e-5){
                bool_correct = false;
                break;
//                printf("%.16f \t%.16f \t%.16f \n", u[i], u_petsc[i], u[i] - u_petsc[i]);
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
#endif

    // *************************** check correctness of the solution 2 ****************************

    // A is scaled. read it from the file and don't scale.
#if 0
    saena::matrix AA(comm);
    saena::laplacian3D(&AA, mx, my, mz, false);

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
#endif

/*
    std::vector<double> res(num_local_row,0);
    for(index_t i = 0; i < num_local_row; ++i){
        res[i] = Au[i] - rhs_std[i];
//        printf("%.12f \t%.12f \t%.12f \n", Au[i], rhs_std[i], Au[i] - rhs_std[i]);
    }

    float norm1 = pnorm(res, comm);
    if(!rank) std::cout << "norm(Au-b)     = " << norm1 << "\n";
*/
    // *************************** check correctness of the solution 3 ****************************

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
    if(free_amg)
        solver.destroy();
    saena_free(rhs_std);
    saena_free(u);
    MPI_Finalize();
    return 0;
}