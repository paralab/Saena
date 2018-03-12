#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include "mpi.h"
#include <vector>

#include "grid.h"
#include "saena.hpp"
#include <saena_object.h>
#include <saena_matrix.h>
#include <omp.h>
//#include "El.hpp"

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    int assert1, assert2, assert3;
//    unsigned long i;
    bool verbose = false;

#pragma omp parallel
    if(rank==0 && omp_get_thread_num()==0) printf("\nnumber of processes = %d, number of threads = %d\n\n", nprocs, omp_get_num_threads());

/*
    if(argc != 4){
        if(rank == 0)
            std::cout << "This is how to make a 3DLaplacian: ./Saena <x grid size> <y grid size> <z grid size>" << std::endl;
        MPI_Finalize();
        return -1;}
*/

    if(argc != 2){
        if(rank == 0){
            std::cout << "Usage: ./Saena <MatrixA>" << std::endl;
            std::cout << "Matrix file should be in triples format." << std::endl;}
        MPI_Finalize();
        return -1;}

/*
    if(argc != 3){
        if(rank == 0){
            std::cout << "Usage: ./Saena <MatrixA> <rhs_vec>" << std::endl;
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
/*
    int mx(std::stoi(argv[1]));
    int my(std::stoi(argv[2]));
    int mz(std::stoi(argv[3]));

    if(verbose){
        MPI_Barrier(comm);
        if(rank==0) printf("3D Laplacian: grid size: x = %d, y = %d, z = %d \n", mx, my, mz);
        MPI_Barrier(comm);}

    // timing the matrix setup phase
    double t1 = omp_get_wtime();

    saena::matrix A(comm);
//    saena::laplacian2D_old(&A, mx, comm);
//    saena::laplacian3D_old(&A, mx, comm);
    saena::laplacian3D(&A, mx, my, mz, comm);

    double t2 = omp_get_wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
*/
    // ******** 2 - initialize the matrix: read from file *************

    char* file_name(argv[1]);
//    MPI_Barrier(comm);
    double t1 = omp_get_wtime();

    saena::matrix A (file_name, comm);
    A.assemble();

    double t2 = omp_get_wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);

    // *************************** set rhs ****************************
    // ********** 1 - set rhs: generate randomly **********

    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<value_t> rhs(num_local_row);
    generate_rhs_old(rhs);

    // ********** 2 - set rhs: read from file **********
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
    // ********** print rhs **********

//    if(rank==0)
//        for(index_t i = 0; i < rhs.size(); i++)
//            std::cout << rhs[i] << std::endl;

    // *************************** set u0 ****************************

    std::vector<value_t> u(num_local_row, 0);

    //**************************** compare matvec, matvec2, matvec3 and matvec4 results ****************************
/*
    int time_num = 5;
    std::vector<double> time_e1(time_num, 0);

    std::vector<double> r_saena(num_local_row);
    std::vector<double> r_saena2(num_local_row);
    std::vector<double> r_saena3(num_local_row);
    std::vector<double> r_saena4(num_local_row);
//    std::vector<double> r_saena5(num_local_row);

    saena_matrix *B = A.get_internal_matrix();

    B->matvec(rhs,  r_saena);
    B->matvec2(rhs, r_saena2);
    B->matvec3(rhs, r_saena3);
    B->matvec4(rhs, r_saena4);
//    B->matvec5(rhs, r_saena5);

//    for(long i = 0; i< num_local_row; i++)
//        if(abs(r_saena5[i]-r_saena[i]) > 1e-4)
//            printf("%lu \t%f \t%f \t%f \n", i, r_saena[i], r_saena5[i], r_saena5[i]-r_saena[i]);

//    for(long i = 0; i< num_local_row; i++)
//        if(r_saena3[i]-r_saena2[i] > 1e-4)
//            printf("%lu \t%f \t%f \t%f \n", i, r_saena2[i], r_saena3[i], r_saena3[i]-r_saena2[i]);

//    for(long i = 0; i< num_local_row; i++)
//        printf(" %lu \t%f \t%f \t%f \t%f \t%f \n", i, r_saena[i], r_saena2[i], r_saena3[i], r_saena4[i], r_saena3[i]-r_saena2[i]);
*/
    //******************************* compare matvec_timing1, 2, 3, 4 *******************************

    int matvec_iter = 200;
    int time_num = 4;
    std::vector<double> time_e1(time_num, 0); // array for timing matvec
//    double average1, average2, average3;

//    saena_object* amg = solver.get_object();
    saena_matrix *B = A.get_internal_matrix();
    num_local_row = B->M;
//    if (rank == 0) printf("\n#####################################\n\n");
    if (rank == 0) std::cout << "local loop times are being printed for matvec1, 2, 3, 4, respectively:" << std::endl;

    // *************************** matvec1 ****************************
    // warm-up
    // -------
    generate_rhs_old(rhs);
    u.assign(num_local_row, 0);
    time_e1.assign(time_e1.size(), 0);

    for (int i = 0; i < 50; i++) {
        B->matvec_timing1(rhs, u, time_e1);
        rhs.swap(u);
    }
//            if (rank == 0) printf("level %d of %d step2! \n", l, levels);

    generate_rhs_old(rhs);
    u.assign(num_local_row, 0);
    time_e1.assign(time_e1.size(), 0);
//            printf("rank %d: level %d of %d step3! \n", rank, l, levels);

    MPI_Barrier(B->comm);
//    t1 = omp_get_wtime();
    for (int i = 0; i < matvec_iter; i++) {
        B->matvec_timing1(rhs, u, time_e1);
        rhs.swap(u);
    }
//    t2 = omp_get_wtime();

    if (rank == 0) {
//        std::cout << "\n1- Saena matvec total time:\n" << (time_e1[0]+time_e1[3])/(matvec_iter) << std::endl;
//        std::cout << std::endl << "matvec1:" << std::endl;
        std::cout << time_e1[1] / matvec_iter << std::endl; // local loop
//        std::cout << time_e1[2] / matvec_iter << std::endl; // remote loop
//        std::cout << ( time_e1[0] + time_e1[3] - time_e1[1] - time_e1[2]) / matvec_iter << std::endl; // communication including "set vSend"
    }

    // *************************** matvec2 ****************************

    generate_rhs_old(rhs);
    u.assign(num_local_row, 0);
    time_e1.assign(time_e1.size(), 0);

    MPI_Barrier(B->comm);
//    t1 = omp_get_wtime();
    for (int i = 0; i < matvec_iter; i++) {
        B->matvec_timing2(rhs, u, time_e1);
        rhs.swap(u);
    }
//    t2 = omp_get_wtime();

    if (rank == 0) {
//        std::cout << "\n1- Saena matvec total time:\n" << (time_e1[0]+time_e1[3])/(matvec_iter) << std::endl;
//        std::cout << std::endl << "matvec2:" << std::endl;
        std::cout << time_e1[1] / matvec_iter << std::endl; // local loop
//        std::cout << time_e1[2] / matvec_iter << std::endl; // remote loop
//        std::cout << ( time_e1[0] + time_e1[3] - time_e1[1] - time_e1[2]) / matvec_iter << std::endl; // comm including "set vSend"
    }

    // *************************** matvec3 ****************************

    generate_rhs_old(rhs);
    u.assign(num_local_row, 0);
    time_e1.assign(time_e1.size(), 0);

    MPI_Barrier(B->comm);
//    t1 = omp_get_wtime();
    for (int i = 0; i < matvec_iter; i++) {
        B->matvec_timing3(rhs, u, time_e1);
        rhs.swap(u);
    }
//    t2 = omp_get_wtime();

    if (rank == 0) {
//        std::cout << "\n1- Saena matvec total time:\n" << (time_e1[0]+time_e1[3])/(matvec_iter) << std::endl;
//        std::cout << std::endl << "matvec3:" << std::endl;
        std::cout << time_e1[1] / matvec_iter << std::endl; // local loop
//        std::cout << time_e1[2] / matvec_iter << std::endl; // remote loop
//        std::cout << ( time_e1[0] + time_e1[3] - time_e1[1] - time_e1[2]) / matvec_iter << std::endl; // comm including "set vSend"
    }

    // *************************** matvec4 ****************************

    generate_rhs_old(rhs);
    u.assign(num_local_row, 0);
    time_e1.assign(time_e1.size(), 0);

    MPI_Barrier(B->comm);
//    t1 = omp_get_wtime();
    for (int i = 0; i < matvec_iter; i++) {
        B->matvec_timing4(rhs, u, time_e1);
        rhs.swap(u);
    }
//    t2 = omp_get_wtime();

    if (rank == 0) {
//        std::cout << "\n1- Saena matvec total time:\n" << (time_e1[0]+time_e1[3])/(matvec_iter) << std::endl;
//        std::cout << std::endl << "matvec4:" << std::endl;
        std::cout << time_e1[1] / matvec_iter << std::endl; // local loop
//        std::cout << time_e1[2] / matvec_iter << std::endl; // remote loop
//        std::cout << ( time_e1[0] + time_e1[3] - time_e1[1] - time_e1[2]) / matvec_iter << std::endl; // comm including "set vSend"
    }

    // *************************** finalize ****************************

//    A.destroy();
    MPI_Finalize();
    return 0;
}
