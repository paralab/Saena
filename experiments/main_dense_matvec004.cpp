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
#include <saena_matrix_dense.h>
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

#pragma omp parallel
    if(rank==0 && omp_get_thread_num()==0) printf("\nnumber of processes = %d, number of threads = %d\n\n", nprocs, omp_get_num_threads());

    if(argc != 3) {
        if (rank == 0) {
            std::cout << "Usage: ./Saena <local number of rows> <number of different bandwidth values>" << std::endl;
            MPI_Finalize();
            return -1;
        }
    }

    unsigned int local_row_num = atoi(argv[1]);
    index_t Mbig = nprocs * local_row_num;

    // sparse matrix: read from file
    // -------------
//    char* file_name(argv[1]);
//    saena::matrix A (file_name, comm);
//    A.assemble();

    // sparse matrix: generate a band matrix
    // -------------
    saena::matrix A(comm);
    saena_matrix *B;

    // dense matrix
    // -------------
    saena_matrix_dense C;

    int matvec_iter = 300;
    std::vector<value_t> v1(local_row_num, 0);
    std::vector<value_t> v2(local_row_num, 0);

    std::vector<nnz_t> nnz_vec;
    std::vector<double> density;

    // time parameters
    // ---------------
    double t1, t2, t_dif, ave;
    std::vector<value_t> ave_sparse;
    std::vector<value_t> ave_dense;

    // warm-up
    // -------
    saena::band_matrix(A, local_row_num, 0);
    B = A.get_internal_matrix();
//    B->print(-1);
    generate_rhs_old(v1);
    v2.assign(local_row_num, 0);
    for (int i = 0; i < 50; i++) {
        B->matvec(v1, v2);
        v1.swap(v2);
    }
//    A.erase();
    B->erase2();
    
    index_t b_size = atoi(argv[2]);
    index_t b_step = Mbig / b_size;
    std::vector<index_t> bandwidth = {b_step}; //bandwidth
    while(bandwidth.size() < b_size+1){
//        if(rank==0) printf("\nbandwidth = %u \n", bandwidth.back());

        saena::band_matrix(A, local_row_num, bandwidth.back());
//        A.print(-1);
        B = A.get_internal_matrix();
        C.convert_saena_matrix(B);

        nnz_vec.push_back(B->nnz_g);
        density.push_back(  (B->nnz_g / double(B->Mbig))  / B->Mbig  );

        // sparse matvec
        // ------------
//        if(rank==0) printf("\nstart of sparse matvec \n");
        generate_rhs_old(v1);
        v2.assign(local_row_num, 0);
        MPI_Barrier(comm);
        t1 = omp_get_wtime();
        for (int i = 0; i < matvec_iter; i++) {
            B->matvec(v1, v2);
            v1.swap(v2);
        }
        t2 = omp_get_wtime();

        t_dif = t2 - t1;
        MPI_Reduce(&t_dif, &ave, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        if(rank==0) ave_sparse.push_back(ave/(nprocs*matvec_iter));

        // dense matvec
        // ------------
//        if(rank==0) printf("\nstart of dense matvec \n");
        generate_rhs_old(v1);
        v2.assign(local_row_num, 0);
        MPI_Barrier(comm);
        t1 = omp_get_wtime();
        for (int i = 0; i < matvec_iter; i++) {
            C.matvec(v1, v2);
            v1.swap(v2);
        }
        t2 = omp_get_wtime();

        t_dif = t2 - t1;
        MPI_Reduce(&t_dif, &ave, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        if(rank==0) ave_dense.push_back(ave/(nprocs*matvec_iter));

        B->erase2();


        bandwidth.push_back( bandwidth.back() + b_step );

        // add the completely dense matrix here
//        if(bandwidth.back() >= Mbig)
//            bandwidth.back() = Mbig-1;
    }

    // throw away the last bandwidth value if it is not used.
    if(bandwidth.back() >= Mbig)
        bandwidth.pop_back();

//    print_vector(bandwidth, 0, "bandwidth", comm);
//    print_vector(ave_dense, 0, "dense", comm);
//    print_vector(ave_sparse, 0, "sparse", comm);

    // write to file
    // -------------
    if(rank==0){
        std::string file_name = "matvec_size";
        file_name += std::to_string(Mbig);
        file_name += "_np";
        file_name += std::to_string(nprocs);
        file_name += ".txt";
        std::ofstream outFile(file_name);

        outFile << "average time for " << matvec_iter << " matvec iterations" << std::endl;

        outFile << "\nmatrix size    = " << Mbig << " x " << Mbig
                << "\nprocessors     = " << nprocs
//                << "\nOpenMP threads = " << omp_get_num_threads()
                << std::endl;

        // this one is better readable in the file, the next one in excel.
//        outFile << "\nbandwidth \tsparse \t\tdense \t\tnnz \tdensity \n" << std::endl;
//        for(index_t i = 0; i < ave_sparse.size(); i++)
//            outFile << bandwidth[i] << "\t\t" << ave_sparse[i] << "\t" << ave_dense[i] << "\t" << nnz_vec[i]
//                    << "\t" << density[i] << std::endl;

        outFile << "\nbandwidth \tsparse \tdense \tnnz \tdensity" << std::endl;
        for(index_t i = 0; i < ave_sparse.size(); i++)
            outFile << bandwidth[i] << "\t" << ave_sparse[i] << "\t" << ave_dense[i] << "\t" << nnz_vec[i]
                    << "\t" << density[i] << std::endl;

        outFile.close();
    }

    MPI_Finalize();
    return 0;
}