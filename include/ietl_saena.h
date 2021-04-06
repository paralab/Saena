#ifndef IETL_SAENA_H
#define IETL_SAENA_H

#include "saena_matrix.h"

#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random.hpp>
#include <boost/limits.hpp>

#include "ietl/interface/ublas.h"
#include "ietl/vectorspace.h"
#include "ietl/lanczos.h"

#include <cmath>
#include <limits>

typedef saena_matrix Matrix;
typedef boost::numeric::ublas::vector<value_t> Vector;


int find_eig_ietl(Matrix& A){
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
//        if(rank==0) printf("%lu \t%u \t%f \t%f \t%f \tietl, remote \n", i, A.row_remote[i], A.values_remote[i], A.inv_diag[A.row_remote[i]], A.values_remote[i]*A.inv_diag[A.row_remote[i]]);
//        A.values_remote[i] *= A.inv_diag[A.row_remote[i]];
//    }
//    MPI_Barrier(A.comm);
    }
#endif

    typedef ietl::vectorspace<Vector> Vecspace;
    typedef boost::lagged_fibonacci607 Gen;

    Vecspace vec(A.Mbig);
    Gen mygen;
    ietl::lanczos<Matrix, Vecspace> lanczos(A, vec);

#ifdef __DEBUG1__
    if(verbose_eig) {
        MPI_Barrier(comm);
        if(rank==0) printf("find_eig: after lanczos\n");
        MPI_Barrier(comm);
    }
#endif

    // Creation of an iteration object:
    int    max_iter       = 50;         // default was 10 * N
    int    n_highest_eigv = 1;
    double rel_tol        = 500 * std::numeric_limits<double>::epsilon();          // default: 500 * epsilon
    double abs_tol        = std::pow(std::numeric_limits<double>::epsilon(), 2. / 3);
    ietl::lanczos_iteration_nhighest<double> iter(max_iter, n_highest_eigv, rel_tol, abs_tol);

#ifdef __DEBUG1__
//    if(rank==0) std::cout << "-----------------------------------\n\n";
//    if(rank==0) std::cout << "\nComputation of " << n_highest_eigenval << " highest converged eigenvalues\n\n";

    if(verbose_eig) {
        MPI_Barrier(comm);
        if(rank==0) printf("find_eig: after lanczos_iteration_nhighest \n");
        MPI_Barrier(comm);
    }
#endif

    std::vector<double> eigen;

//    try{
        lanczos.calculate_eigenvalues(iter, mygen);
        eigen = lanczos.eigenvalues();
//    } catch (std::runtime_error& e) {
//        std::cout << e.what() << "\n";
//    }

#ifdef __DEBUG1__
    // Printing eigenvalues with error & multiplicities:
    // -------------------------------------------------
//    if(!rank) {
//        std::vector<double> err          = lanczos.errors();
//        std::vector<int>    multiplicity = lanczos.multiplicities();
//        std::cout.precision(10);
//
//        printf("\nnumber of iterations to compute the biggest eigenvalue: %d\n", iter.iterations());
//        printf("rank %d: the biggest eigenvalue: %.14f (IETL) \n", rank, eigen.back());
//        printf("\n#        eigenvalue            error         multiplicity\n");
//        for (auto i = eigen.size()-1; i > eigen.size()-1-n_highest_eigv; --i){
//            std::cout << i << "\t" << eigen[i] << "\t" << err[i] << "\t\t" << multiplicity[i] << "\n";
//        }
//    }
#endif

    // The number of max_iter for Lanzcos is set low for performance reasons, so there may be error for the largest
    // eigenvalue. Since we need an upper bound for the largest eigenvalue in Chebyshev, multiply it to slightly
    // up-scale it.
    A.eig_max_of_invdiagXA = eigen.back();

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

#endif //IETL_SAENA_H
