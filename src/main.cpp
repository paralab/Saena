#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"

#include "petsc_functions.h"
//#include "combblas_functions.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <omp.h>
#include <dollar.hpp>
#include "mpi.h"


#include "grid.h"


int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

    if(argc != 4){
        if(rank == 0)
            std::cout << "This is how to make a 3DLaplacian: ./Saena <x grid size> <y grid size> <z grid size>" << std::endl;
        MPI_Finalize();
        return -1;
    }

    // *************************** initialize the matrix: Laplacian ****************************

    double t1 = MPI_Wtime();

    int mx(std::stoi(argv[1]));
    int my(std::stoi(argv[2]));
    int mz(std::stoi(argv[3]));

    if(verbose){
        MPI_Barrier(comm);
        if(rank==0) printf("3D Laplacian: grid size: x = %d, y = %d, z = %d \n", mx, my, mz);
        MPI_Barrier(comm);
    }

    saena::matrix A(comm);
    saena::laplacian3D(&A, mx, my, mz);
//    saena::laplacian2D_old(&A, mx);
//    saena::laplacian3D_old(&A, mx);

    // ********** print matrix and time **********

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
//    print_time(t1, t2, "Matrix Assemble:", comm);

//    A.print(0);
//    A.get_internal_matrix()->print_info(0);
//    A.get_internal_matrix()->writeMatrixToFile("writeMatrix");

//    petsc_viewer(A.get_internal_matrix());

    // *************************** set rhs: Laplacian ****************************

    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<double> rhs;
    saena::laplacian3D_set_rhs(rhs, mx, my, mz, comm);

    // ********** print rhs **********

//    print_vector(rhs, -1, "rhs", comm);

    // *************************** set u0 ****************************

    std::vector<double> u(num_local_row, 0);

    // *************************** AMG - Setup ****************************

    t1 = MPI_Wtime();

//    int max_level             = 2; // this is moved to saena_object.
    int vcycle_num            = 400;
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

//    if(rank==0) printf("usage: ./Saena x_size y_size z_size sparse_epsilon \n");
//    double sp_epsilon(std::atof(argv[4]));
//    if(rank==0) printf("\nsp_epsilon = %f \n", sp_epsilon);
//    solver.get_object()->sparse_epsilon = sp_epsilon;

    // receive sparsifivation factor from input and set it.
//    double sm_sz_prct(std::stof(argv[4]));
//    if(rank==0) printf("sm_sz_prct = %f \n", sm_sz_prct);
//    solver.set_sample_sz_percent(sm_sz_prct);

    solver.set_matrix(&A, &opts);
    solver.set_rhs(rhs);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Setup:", comm);
//    print_time(t1, t2, "Setup:", comm);

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

    // *************************** experiment for compute_coarsen ****************************
/*
    {
        saena_object *obj1 = solver.get_object();
        Grid *g1 = &obj1->grids[0];

        index_t row_thres;
//        row_thres = g1->A->Mbig;
//        row_thres = g1->P.Nbig;

        obj1->mempool1 = new value_t[obj1->matmat_size_thre];
        obj1->mempool2 = new index_t[g1->A->Mbig * 4];

        Grid g2;
        g2.P = g1->P;
        g2.R = g1->R;
        saena_matrix B(comm);
        g2.A = &B;

        B.Mbig = g1->A->Mbig;
        B.M = g1->A->M;
        B.split = g1->A->split;
        B.comm = g1->A->comm;

        printf("row\tAP\tR(AP)\t\ttotal");
        for(index_t j = 1; j < 20; j*=2){
            row_thres = g1->A->Mbig / j;
//            printf("\n=======================================\n\nrow_thres = %u \n", row_thres);
            printf("\n%u\t", row_thres);

            B.entry.clear();
            for (int i = 0; i < g1->A->entry.size(); i++) {
                if (g1->A->entry[i].row < row_thres) {
                    B.entry.emplace_back(g1->A->entry[i]);
                }
            }
            obj1->compute_coarsen_test(&g2);

        }
        printf("\n");

        delete []obj1->mempool1;
        delete []obj1->mempool2;

    }
*/


    {
        saena_object *obj1 = solver.get_object();
        Grid *g1 = &obj1->grids[0];

        index_t row_thres;
//        row_thres = g1->A->Mbig;
//        row_thres = g1->P.Nbig;

        obj1->mempool1 = new value_t[obj1->matmat_size_thre];
        obj1->mempool2 = new index_t[g1->A->Mbig * 4];

//        Grid g2;
//        g2.P = g1->P;
//        g2.R = g1->R;
//        saena_matrix B(comm);
//        g2.A = &B;
//
//        B.Mbig = g1->A->Mbig;
//        B.M = g1->A->M;
//        B.split = g1->A->split;
//        B.comm = g1->A->comm;

        std::vector<cooEntry_row> RAP_row_sorted;

        int matmat_times = 1;
        MPI_Barrier(comm);
        double t11 = MPI_Wtime();

        for(index_t j = 0; j < matmat_times; j++){
//            obj1->compute_coarsen_test(g1);
            obj1->triple_mat_mult_basic(g1, RAP_row_sorted);
        }

        double t22 = MPI_Wtime();
//        print_time_ave(t22-t11, "triple_mat_mult_test: ", grid->A->comm);
        printf("\naverage coarsen time for %d times: \n%f \n\n", matmat_times, (t22 - t11) / matmat_times);

        delete []obj1->mempool1;
        delete []obj1->mempool2;

    }

    // *************************** CombBLAS ****************************

//    combblas_matmult_DoubleBuff();
//    int combblas_matmult_Synch();
//    int combblas_GalerkinNew();

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

    // *************************** matvec on different coarse levels of a matrix ****************************
/*
    int matvec_iter = 300;
    int time_num = 4;
    std::vector<double> time_e1(time_num, 0); // array for timing matvec
    std::vector<std::vector<double>> time_total; // array for keeping all levels timings
//    double average1, average2, average3;

    saena_object* amg = solver.get_object();
    saena_matrix *B;
    int levels = amg->max_level;

    // warm-up
    // -------
    B = amg->grids[0].A;
    num_local_row = B->M;
    rhs.resize(num_local_row);
    u.resize(num_local_row);
    time_e1.assign(time_e1.size(), 0);
    for (int i = 0; i < 50; i++) {
        B->matvec_timing1(rhs, u, time_e1);
        rhs.swap(u);
    }

//    if (rank == 0) std::cout << "\nlocal loop, remote loop and communication (including <<set vSend>>) times of matvec"
//                                " are being printed for different levels of the multigrid hierarchy:" << std::endl;

    if (rank == 0) printf("\n#####################################\n\n");
    for(int l = 0; l < levels+1; l++) {
        if (rank == 0) printf("start level %d of %d \n", l, levels);

        if (amg->grids[l].active) {
            B = amg->grids[l].A;
            num_local_row = B->M;
//            printf("level = %d, num_local_row = %d \n", l, num_local_row);
            rhs.resize(num_local_row);
            u.resize(num_local_row);
//            if (rank == 0) printf("level %d of %d step1! \n", l, levels);

            // *************************** matvec1 ****************************

            generate_rhs_old(rhs);
            u.assign(num_local_row, 0);
            time_e1.assign(time_e1.size(), 0);
//            printf("rank %d: level %d of %d step3! \n", rank, l, levels);

            MPI_Barrier(B->comm);
//            t1 = omp_get_wtime();
            for (int i = 0; i < matvec_iter; i++) {
                B->matvec_timing1(rhs, u, time_e1);
                rhs.swap(u);
            }
//            t2 = omp_get_wtime();

//        average1 = print_time(t1/double(matvec_iter), t2/double(matvec_iter), "matvec1:", comm);
//        if (rank==0) printf("_________________________________\n\n");
//        if (rank==0) printf("local matvec level %d of %d \n", l, levels);
//        if (rank==0) std::cout << time_e1[1]/(matvec_iter) << std::endl;

//            if (rank == 0) {
//              std::cout << "\n1- Saena matvec total time:\n" << (time_e1[0]+time_e1[3])/(matvec_iter) << std::endl;
//              std::cout << std::endl << "matvec1:" << std::endl;
//                std::cout << time_e1[1] / matvec_iter << std::endl; // local loop
//                std::cout << time_e1[2] / matvec_iter << std::endl; // remote loop
//                std::cout << ( time_e1[0] + time_e1[3] - time_e1[1] - time_e1[2]) / matvec_iter << std::endl; // communication including "set vSend"
//            }

        }
        time_total.push_back(time_e1);
    }

    // *************************** print time results ****************************

    // print on output
    if(rank==0){
        std::cout << "\ntime results:\n" << std::endl;
        std::cout << "level \tlocal \t\tremote \t\tcomm \t\ttotal" << std::endl;
        for(int i = 0; i < time_total.size(); i++)
            std::cout << i << "\t"
                      << time_total[i][1]/matvec_iter << "\t"
                      << time_total[i][2]/matvec_iter << "\t"
                      << (time_total[i][0] + time_total[i][3] - time_total[i][1] - time_total[i][2])/matvec_iter << "\t"
                      << (time_total[i][0] + time_total[i][3])/matvec_iter << std::endl;
    }
*/
/*
    // wrtie to file
    if(rank==0){

        if(rank==0) {
            std::string input_filename_ext = argv[1];
            size_t extIndex = input_filename_ext.find_last_of(".");
            std::string file_name = "./shrink_";
            file_name += input_filename_ext.substr(0, extIndex);
            file_name += ".txt";
            std::ofstream outFile(file_name);

            outFile << "average time for " << matvec_iter << " matvec iterations" << std::endl;
            outFile << "matrix name   = " << argv[1] << "\nprocessors    = " << nprocs << std::endl;
#pragma omp parallel
            if (rank == 0 && omp_get_thread_num() == 0)
                outFile << "OpenMP thread = " << omp_get_num_threads() << std::endl;

            outFile << "\ntime results:\n" << std::endl;
            outFile << "level \tlocal \tremote \tcomm" << std::endl;
            for (int i = 0; i < time_total.size(); i++)
                outFile << i << "\t"
                          << time_total[i][1] / matvec_iter << "\t"
                          << time_total[i][2] / matvec_iter << "\t"
                          << (time_total[i][0] + time_total[i][3] - time_total[i][1] - time_total[i][2]) / matvec_iter
                          << std::endl;

            outFile.close();
        }
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

//    if(rank==0) dollar::text(std::cout);

//    A.destroy();
//    solver.destroy();
    MPI_Finalize();
    return 0;
}