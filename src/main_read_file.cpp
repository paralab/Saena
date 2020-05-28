#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"

#ifdef _USE_PETSC_
#include "petsc_functions.h"
#endif

#include <iostream>
#include <vector>
#include <omp.h>
#include "mpi.h"


int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

   	/*saena_object so;
	so.pcoarsen();
	exit(0); */

    if(argc != 3){
        if(rank == 0) {
            std::cout << "Usage: ./Saena <MatrixA> <rhs>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // *************************** Setup timing parameters ****************************

    std::vector<double> setup_time_loc, solve_time_loc;

    // *************************** initialize the matrix ****************************

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();

    // ******** initialize the matrix: read from file *************

    char* file_name(argv[1]);
    saena::matrix A (comm);
    A.read_file(file_name);
//    A.read_file(file_name, "triangle");
    A.assemble();
//    A.assemble_writeToFile("matrix_folder");

    // ********** print matrix info and time **********

    t1 = MPI_Wtime() - t1;
    if(verbose) print_time(t1, "Matrix Assemble:", comm);
    print_time(t1, "Matrix Assemble:", comm);
    setup_time_loc.emplace_back(t1);

//    A.print(0);
//    A.get_internal_matrix()->print_info(2);
//    A.get_internal_matrix()->writeMatrixToFile("matrix_folder/matrix");
//    print_vector(A.get_internal_matrix()->split, 1, "split", comm);
//    petsc_viewer(A.get_internal_matrix());

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

    // ********** 3 - set rhs: read from file **********
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
    // ********** print rhs **********

//    print_vector(rhs_std, -1, "rhs_std", comm);
//    rhs.print_entry(-1);

    // *************************** set initial guess u0 ****************************

    std::vector<double> u(num_local_row, 0);

    // *************************** AMG - Setup - Set Parameters ****************************

    MPI_Barrier(comm);
    t1 = MPI_Wtime();

//    int max_level             = 2; // this is moved to saena_object.
    int vcycle_num            = 400;
    double relative_tolerance = 1e-14;
    std::string smoother      = "chebyshev"; // choices: "jacobi", "chebyshev"
    int preSmooth             = 3;
    int postSmooth            = 3;

    saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
//    saena::options opts((char*)"options001.xml");
//    saena::options opts;
    saena::amg solver;
    solver.set_verbose(verbose); // set verbose at the beginning of the main function.
//    solver.set_multigrid_max_level(0); // 0 means only use direct solver, so no multigrid will be used.

    // *************************** Setup - Sparsification ****************************

//    if(rank==0) printf("usage: ./Saena x_size y_size z_size sparse_epsilon \n");
//    double sp_epsilon(std::atof(argv[4]));
//    if(rank==0) printf("\nsp_epsilon = %f \n", sp_epsilon);
//    solver.get_object()->sparse_epsilon = sp_epsilon;

    // receive sparsifivation factor from input and set it.
//    double sm_sz_prct(std::stof(argv[4]));
//    if(rank==0) printf("sm_sz_prct = %f \n", sm_sz_prct);
//    solver.set_sample_sz_percent(sm_sz_prct);

    // *************************** Setup - Perform Setup ****************************

    solver.set_matrix(&A, &opts);
    solver.set_rhs(rhs);

    t1 = MPI_Wtime() - t1;
    if(solver.verbose) print_time(t1, "Setup:", comm);
    print_time(t1, "Setup:", comm);
    setup_time_loc.front() += t1; // add matrix assemble time and AMG setup time to the first entry of setup_time_loc.

//    print_vector(solver.get_object()->grids[0].A->entry, -1, "A", comm);
//    print_vector(solver.get_object()->grids[0].rhs_std, -1, "rhs_std", comm);

    // *************************** AMG - Solve ****************************

    MPI_Barrier(comm);
    t1 = MPI_Wtime();

    // AMG as a solver
//    solver.solve(u, &opts);

    // AMG as a preconditioner for CG
    solver.solve_pcg(u, &opts);

    t1 = MPI_Wtime() - t1;
    if(solver.verbose) print_time(t1, "Solve:", comm);
    print_time(t1, "Solve:", comm);
    solve_time_loc.emplace_back(t1);

    // print solution
//    print_vector(u, -1, "u", comm);

    // *************************** check correctness of the solution ****************************
/*
    // A is scaled. read it from the file and don't scale.

    saena::matrix AA (comm);
    AA.read_file(file_name);
    AA.assemble_no_scale();
    saena_matrix *AAA = AA.get_internal_matrix();
    std::vector<double> Au(num_local_row, 0);
    std::vector<double> sol = u;
    AAA->matvec(sol, Au);

    bool bool_correct = true;
    if(rank==0){
        printf("\nChecking the correctness of the Saena solution by Saena itself:\n");
        printf("Au \t\trhs_std \t\tAu-rhs_std \n");
        for(index_t i = 0; i < num_local_row; i++){
            if(fabs(Au[i] - rhs_std[i]) > 1e-10){
                bool_correct = false;
                printf("%.10f \t%.10f \t%.10f \n", Au[i], rhs_std[i], Au[i] - rhs_std[i]);
            }
        }
        if(bool_correct)
            printf("\n******* The solution was correct! *******\n\n");
        else
            printf("\n******* The solution was NOT correct! *******\n\n");
    }
*/

    // *************************** finalize ****************************

//    if(rank==0) dollar::text(std::cout);

//    A.destroy();
//    solver.destroy();
    MPI_Finalize();
    return 0;
}
