#ifndef LAMLAN_SAENA_H
#define LAMLAN_SAENA_H

#include "saena_matrix.h"
#include "lambda_lanczos.hpp"

//#include <iostream>
//#include <iomanip>
//#include <cstdlib>

using lambda_lanczos::LambdaLanczos;

int find_eig_lamlan(saena_matrix &A){
    // this function uses IETL library to find the biggest eigenvalue.
    // IETL is modified to work in parallel (only ietl/interface/ublas.h).

    MPI_Comm comm = A.comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    const bool verbose_eig = false;

#ifdef __DEBUG1__
    {
        if(verbose_eig) {
            MPI_Barrier(comm);
            if(rank==0) printf("\nfind_eig: start\n");
            MPI_Barrier(comm);
        }

//    A.print_entry(-1);
//    A.print_info(-1);
    }
#endif

    // the matrix-vector multiplication routine
    auto mv_mul = [&](const vector<value_t>& in, vector<value_t>& out) {
        A.matvec(&in[0], &out[0]);
//        for(int i = 0;i < matrix.size();i++) {
//            out[matrix[i].r] += matrix[i].value*in[matrix[i].c];
//        }
    };

    const size_t n = A.M;
//    if(!rank) printf("n = %ld\n", n);

    // max_iteration is set to 20 in lambda_lanzcos.hpp
    // eps (Convergence threshold) can be set in lambda_lanzcos.hpp
    LambdaLanczos<value_t> engine(mv_mul, n, true, A.comm); // true means to calculate the smallest eigenvalue.
    value_t eigenvalue = 0.0;

    // computing the eigenvector is commented out. Uncomment it at the end of run() if needed.
    vector<value_t> eigenvector;

    int itern = engine.run(eigenvalue, eigenvector);

    // NOTE: the computed eigenvalue slightly fluctuates in each execution. Since an upperbound is needed for Chebyshev,
    // upscale it slightly.
    A.eig_max_of_invdiagXA = 1.0001 * eigenvalue;

#ifdef __DEBUG1__
//    if(!rank) printf("Iteration count = %d, Eigenvalue = %f\n", itern, eigenvalue);

//    cout << "Eigen vector: ";
//    for(int i = 0;i < n;i++) {
//        cout << eigenvector[i] << " ";
//    }
//    cout << endl;

    if(verbose_eig) {
//        if(rank==0) printf("the biggest eigenvalue of D^{-1}*A is %f (IETL) \n", A.eig_max_of_invdiagXA);
        MPI_Barrier(comm);
        if(rank==0) printf("find_eig: end\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}

#endif //LAMLAN_SAENA_H
