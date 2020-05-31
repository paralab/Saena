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

    // sparse matrix
    // -------------
    char* file_name(argv[1]);
    saena::matrix A (file_name, comm);
    A.assemble();
    saena_matrix *B = A.get_internal_matrix();

    // dense matrix
    // -------------
    saena_matrix_dense C;
    C.convert_saena_matrix(B);
//    MPI_Barrier(comm); C.print(-1); MPI_Barrier(comm);

    std::vector<value_t> v(C.M, 0);
    for(index_t i = 0; i < C.M; i++)
        v[i] = i + 1 + C.split[rank];

//    print_vector(v, -1, "v", comm);

    // sparse matvec
    // -------------
    std::vector<value_t> w(B->M, 0);
    B->matvec(v, w);

//    print_vector(w, -1, "w", comm);

    // dense matvec
    // ------------
    std::vector<value_t> w2(C.M, 0);
    C.matvec(v, w2);

//    print_vector(w2, -1, "w2", comm);

    // find the difference between two matvecs
    // --------------------
    bool equal_local = true;
    for(index_t i = 0; i < w.size(); i++)
        if(fabs(w[i] - w2[i]) > 1e-14){
            equal_local = false;
            printf("rank %d: %d \t%.9f \t%.9f \t%e \n", rank, i, w[i], w2[i], w[i] - w2[i]);
        }

    bool equal;
    MPI_Reduce(&equal_local, &equal, 1, MPI_CXX_BOOL, MPI_LOR, 0, comm);

    if(rank==0) printf("\nAre sparse and dense matvecs equal = %d \n", equal);

    MPI_Finalize();
    return 0;
}
