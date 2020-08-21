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
        if(rank == 0) {
            std::cout << "Usage: ./Saena <MatrixA> <rhs>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // set the number of OpenMP threads at run-time
    omp_set_num_threads(1);

    // *************************** Ssetup timing parameters ****************************

    std::vector<double> setup_time_loc, solve_time_loc;

    // *************************** initialize the matrix ****************************

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();

    // ******** 1 - initialize the matrix: laplacian *************
/*
    int mx(std::stoi(argv[1]));
    int my(std::stoi(argv[2]));
    int mz(std::stoi(argv[3]));

    if(verbose){
        MPI_Barrier(comm);
        if(rank==0) printf("3D Laplacian: grid size: x = %d, y = %d, z = %d \n", mx, my, mz);
        MPI_Barrier(comm);}


    saena::matrix A(comm);
    saena::laplacian3D(&A, mx, my, mz);
//    saena::laplacian2D_old(&A, mx);
//    saena::laplacian3D_old(&A, mx);
*/
    // ******** 2 - initialize the matrix: read from file *************

    char* file_name(argv[1]);
    saena::matrix A (comm);
    A.read_file(file_name);
//    A.read_file(file_name, "triangle");
    A.assemble();
//    A.assemble_writeToFile("writeMatrix");

    // ********** print matrix and time **********

    t1 = MPI_Wtime() - t1;
    if(verbose) print_time(t1, "Matrix Assemble:", comm);
    print_time(t1, "Matrix Assemble:", comm);
    setup_time_loc.emplace_back(t1);

//    A.print(0);
//    A.get_internal_matrix()->print_info(0);
//    A.get_internal_matrix()->writeMatrixToFile("writeMatrix");

//    petsc_viewer(A.get_internal_matrix());

    // *************************** set rhs_std ****************************

    saena::vector rhs(comm);
    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<double> rhs_std;

    // ********** 1 - set rhs_std: random **********
/*
    rhs_std.resize(num_local_row);
    generate_rhs_old(rhs_std);

    std::vector<double> tmp1 = rhs_std;
    A.get_internal_matrix()->matvec(tmp1, rhs_std);

    index_t my_split;
    saena::find_split((index_t)rhs_std.size(), my_split, comm);

    rhs.set(&rhs_std[0], (index_t)rhs_std.size(), my_split);
    rhs.assemble();
*/
    // ********** 2 - set rhs_std: ordered: 1, 2, 3, ... **********

//    rhs_std.resize(num_local_row);
//    for(index_t i = 0; i < A.get_num_local_rows(); i++)
//        rhs_std[i] = i + 1 + A.get_internal_matrix()->split[rank];

    // ********** 3 - set rhs_std: Laplacian **********

    // don't set the size for this method
//    saena::laplacian3D_set_rhs(rhs_std, mx, my, mz, comm);

    // ********** 4 - set rhs_std: read from file **********

    char* Vname(argv[2]);
//    saena::read_vector_file(rhs_std, A, Vname, comm);
    read_from_file_rhs(rhs_std, A.get_internal_matrix(), Vname, comm);

    // set rhs_std
//    A.get_internal_matrix()->matvec(v, rhs_std);
//    rhs_std = v;

    index_t my_split = 0;
    saena::find_split((index_t)rhs_std.size(), my_split, comm);

    rhs.set(&rhs_std[0], (index_t)rhs_std.size(), my_split);
    rhs.assemble();

    // ********** print rhs_std **********

//    print_vector(rhs_std, -1, "rhs_std", comm);
//    rhs.print_entry(-1);

    // *************************** set u0 ****************************

    std::vector<double> u(num_local_row, 0);

    // *************************** AMG - Setup ****************************

    MPI_Barrier(comm);
    t1 = MPI_Wtime();

//    int max_level             = 2; // this is moved to saena_object.
    int vcycle_num            = 400;
    double relative_tolerance = 1e-14;
    std::string smoother      = "chebyshev"; // choices: "jacobi", "chebyshev"
    int preSmooth             = 1;
    int postSmooth            = 1;

    saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
//    saena::options opts((char*)"options001.xml");
//    saena::options opts;
    saena::amg solver;
    solver.set_verbose(verbose); // set verbose at the beginning of the main function.
    solver.set_multigrid_max_level(3); // 0 means only use direct solver, so no multigrid will be used.

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

//    solver.solve(u, &opts);
    solver.solve_pCG(u, &opts);

    t1 = MPI_Wtime() - t1;
    if(solver.verbose) print_time(t1, "Solve:", comm);
    print_time(t1, "Solve:", comm);
    solve_time_loc.emplace_back(t1);

//    print_vector(u, -1, "u", comm);

    // *************************** lazy-update ****************************

    std::string file_name2 = file_name;
    char        file_name3[100];
    std::size_t length = file_name2.copy(file_name3, strlen(file_name)-5, 0);
    file_name3[length] = '\0';

    size_t      extIndex       = file_name2.find_last_of('.');
    std::string file_extension = file_name2.substr(extIndex + 1, 3);

//    std::cout << "file name: " << file_name << ", file_name2: " << file_name2 << ", file_name3: " << file_name3 << std::endl;

    saena::matrix B (comm);
    int lazy_step = 0;

    int update_method = 1;
    if(rank==0) printf("================================================\n\nupdate method: %d\n", update_method);

//    char pause1[10];
//    MPI_Barrier(comm);
//    if(!rank){
//        printf("Enter any letter (then press enter) to continue!");
//        scanf("%s", pause1);
//    }
//    MPI_Barrier(comm);

    for(int i = 2; i <= ITER_LAZY; i++){
        std::string file_name_update = file_name3;
        file_name_update            += std::to_string(i);
        file_name_update            += '.';
        file_name_update            += file_extension;

//        std::cout << "file_name_update: " << file_name_update << std::endl;

        MPI_Barrier(comm);
        t1 = MPI_Wtime();
        if( lazy_step % 2 == 1) {
            A.erase_no_shrink_to_fit();
//            A.add_duplicates(true);

            A.read_file(file_name_update.c_str());

            if(update_method == 1) {
                A.assemble();
                solver.update1(&A); // update the AMG hierarchy
            } else if(update_method == 2) {
                A.assemble();
                solver.update2(&(A)); // update the AMG hierarchy
            } else if(update_method == 3) {
                A.assemble(false);
                solver.update3(&(A)); // update the AMG hierarchy
            } else {
                printf("Error: Wrong update_method is set! Options: 1, 2, 3\n");
            }

            lazy_step++;

        } else {

            B.erase_no_shrink_to_fit();
//            B.add_duplicates(true);

            B.read_file(file_name_update.c_str());

            if(update_method == 1) {
                B.assemble();
                solver.update1(&(B)); // update the AMG hierarchy
            } else if(update_method == 2) {
                B.assemble();
                solver.update2(&(B)); // update the AMG hierarchy
            } else if(update_method == 3) {
                B.assemble(false);
                solver.update3(&(B)); // update the AMG hierarchy
            } else {
                printf("Error: Wrong update_method is set! Options: 1, 2, 3\n");
            }

            ++lazy_step;
        }

        t1 = MPI_Wtime() - t1;
        print_time(t1, "Setup:", comm);
        setup_time_loc.emplace_back(t1);

//        MPI_Barrier(comm);
//        if(!rank){
//            printf("Enter any letter (then press enter) to continue!");
//            scanf("%s", pause1);
//        }
//        MPI_Barrier(comm);

        MPI_Barrier(comm);
        t1 = MPI_Wtime();
        solver.solve_pCG(u, &opts);
        t1 = MPI_Wtime() - t1;
        print_time(t1, "Solve:", comm);
        solve_time_loc.emplace_back(t1);

//        MPI_Barrier(comm);
//        if(!rank){
//            printf("Enter any letter (then press enter) to continue!");
//            scanf("%s", pause1);
//        }
//        MPI_Barrier(comm);
    }

    std::vector<double> setup_time(setup_time_loc.size()), solve_time(solve_time_loc.size());
    MPI_Reduce(&setup_time_loc[0], &setup_time[0], setup_time_loc.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&solve_time_loc[0], &solve_time[0], solve_time_loc.size(), MPI_DOUBLE, MPI_SUM, 0, comm);

    if(!rank){
        double ave_setup_time = 0, ave_solve_time = 0;

        for(int i = 0; i < setup_time_loc.size(); i++){
            setup_time[i] /= nprocs; // to print the right number in the following print_vector().
            solve_time[i] /= nprocs;
            ave_setup_time += setup_time[i];
            ave_solve_time += solve_time[i];
        }

        print_vector(setup_time, 0, "setup_time", comm);
        print_vector(solve_time, 0, "solve_time", comm);

        ave_setup_time /= setup_time.size();
        ave_solve_time /= solve_time.size();
        printf("ave_setup_time = %f\nave_solve_time = %f\n", ave_setup_time, ave_solve_time);
    }

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

    // *************************** test for lazy update functions ****************************
/*
    saena_matrix* A_saena = A.get_internal_matrix();
    std::vector<index_t> rown(A.get_local_nnz());
    std::vector<index_t> coln(A.get_local_nnz());
    std::vector<value_t> valn(A.get_local_nnz());
    for(nnz_t i = 0; i < A.get_local_nnz(); i++){
        rown[i] = A_saena->entry[i].row;
        coln[i] = A_saena->entry[i].col;
        valn[i] = 2 * A_saena->entry[i].val;
//        valn[i] = 0.33;
//        if(i<50 && rank==1) printf("%f \t%f \n", A_saena->entry[i].val, valn[i]);
    }

    saena::matrix A_new(comm);
    A_new.set(&rown[0], &coln[0], &valn[0], rown.size());
    A_new.assemble();
//    A_new.assemble_no_scale();
//    solver.update1(&A_new);

//    solver.get_object()->matrix_diff(*solver.get_object()->grids[0].A, *A_new.get_internal_matrix());

    if(rank==0){
        for(nnz_t i = 0; i < 50; i++){
//            std::cout << A.get_internal_matrix()->entry[i] << "\t" << A_new.get_internal_matrix()->entry[i] << std::endl;
            std::cout << A_saena->entry[i] << "\t" << A_new.get_internal_matrix()->entry[i] << std::endl;
        }
    }
*/

    // *************************** finalize ****************************

    A.destroy();
    solver.destroy();
    MPI_Finalize();
    return 0;
}