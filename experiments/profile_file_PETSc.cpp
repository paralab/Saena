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

    if(argc != 4) {
        if(!rank) {
            cout << "Usage: ./profile_f_petsc matrix_file rhs_file options_file" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    // set the number of OpenMP threads at run-time
//    omp_set_num_threads(1);

    // *************************** check the petsc flag ****************************

    {
        saena::options opts_test(argv[3]);
        if(opts_test.get_petsc_solver().empty()){
            if(!rank) printf("petsc part of the options file should be set!\n");
            return 0;
        }
    }

    // *************************** set the scaling factor ****************************

    bool scale = false;

    // *************************** initialize the matrix ****************************

    // timing the matrix setup phase
    double t1 = omp_get_wtime();

    saena::matrix A(comm);
    char* file_name(argv[1]);
    A.set_remove_boundary(false);
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
        opts.set_from_file(optsfile);
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

    // *************************** PETSc - Solve ****************************
    // solve the system Au = rhs

    if(opts.get_petsc_solver().empty()){
        if(!rank) printf("petsc part of the options file should be set!\n");
    }else{
        solver.solve_petsc(u, &opts);
    }

    // *************************** print or write the solution ****************************

//    print_vector(u, -1, "solution", comm);
//    write_to_file_vec(u, "solution", comm);

    // *************************** Destroy ****************************

    A.destroy();
    if(free_amg)
        solver.destroy();
    saena_free(rhs_std);
    saena_free(u);
    MPI_Finalize();
    return 0;
}