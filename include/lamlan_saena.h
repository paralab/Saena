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

//    MPI_Barrier(A.comm);
//    for(unsigned long i = 0; i < A.nnz_l_local; i++) {
//        if(rank==0) printf("%lu \t%u \t%f \tietl, local \n", i, A.row_local[i], (A.val_local[i]*A.inv_diag[A.row_local[i]] - A.entry[i].val * A.inv_diag[A.entry[i].row - A.split[rank]]));
//        if(rank==0) printf("%lu \t%u \t%f \t%f \t%f \tietl, local \n", i, A.row_local[i]+A.split[rank], A.val_local[i], A.inv_diag[A.row_local[i]], A.val_local[i]*A.inv_diag[A.row_local[i]]);
//        A.val_local[i] *= A.inv_diag[A.row_local[i]];
//    }

//    MPI_Barrier(A.comm);
//    for(unsigned long i = 0; i < A.nnz_l_remote; i++) {
//        if(rank==0) printf("%lu \t%u \t%f \t%f \t%f \tietl, remote \n", i, A.row_remote[i], A.val_remote[i], A.inv_diag[A.row_remote[i]], A.val_remote[i]*A.inv_diag[A.row_remote[i]]);
//        A.val_remote[i] *= A.inv_diag[A.row_remote[i]];
//    }
//    MPI_Barrier(A.comm);
    }
#endif

    // the matrix-vector multiplication routine
    auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
        A.matvec(&in[0], &out[0]);
//        for(int i = 0;i < matrix.size();i++) {
//            out[matrix[i].r] += matrix[i].value*in[matrix[i].c];
//        }
    };

    const index_t n = A.Mbig;
    LambdaLanczos<double> engine(mv_mul, n, true); // true means to calculate the smallest eigenvalue.
    double eigenvalue = 0.0;
    vector<double> eigenvector(n);
    int itern = engine.run(eigenvalue, eigenvector);

    cout << "Iteration count: " << itern << endl;
    cout << "Eigen value: " << setprecision(12) << eigenvalue << endl;

//    cout << "Eigen vector: ";
//    for(int i = 0;i < n;i++) {
//        cout << eigenvector[i] << " ";
//    }
//    cout << endl;

    A.eig_max_of_invdiagXA = eigenvalue;

#ifdef __DEBUG1__
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
