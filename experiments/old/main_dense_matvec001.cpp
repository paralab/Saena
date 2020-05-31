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

//    int assert1, assert2, assert3;
//    unsigned long i;
    bool verbose = false;

#pragma omp parallel
    if(rank==0 && omp_get_thread_num()==0) printf("\nnumber of processes = %d, number of threads = %d\n\n", nprocs, omp_get_num_threads());

    index_t M;
    if(rank==1)
        M = 5;
    else
        M = 3;

    index_t Nbig = 11;
    saena_matrix_dense A(M, Nbig, comm);

    std::vector<index_t> sp(nprocs+1);
    sp[0] = 0;
    sp[1] = 3;
    sp[2] = 8;
    sp[3] = 11;
    A.split = sp;

    for(index_t i = 0; i < A.M; i++)
        for(index_t j = 0; j < A.Nbig; j++)
            A.entry[i][j] = (i + A.split[rank] + 1) * (j+1);

//    MPI_Barrier(comm); A.print(-1); MPI_Barrier(comm);

    std::vector<value_t> v(A.M, 0);
    for(index_t i = 0; i < A.M; i++)
        v[i] = i + 1 + A.split[rank];

//    print_vector(v, -1, "v", comm);

    std::vector<value_t> w(A.M, 0);
    A.matvec(v, w);

//    print_vector(w, -1, "w", comm);

    MPI_Finalize();
    return 0;
}
