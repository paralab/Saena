#ifndef MATVEC003_IETL_SAENA_H
#define MATVEC003_IETL_SAENA_H

#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <ietl/interface/ublas.h>
#include <ietl/vectorspace.h>
#include <ietl/lanczos.h>
#include <boost/random.hpp>
#include <boost/limits.hpp>
#include <cmath>
#include <limits>

typedef saena_matrix Matrix;
typedef boost::numeric::ublas::vector<value_t> Vector;

int find_eig(Matrix& A){
    // this function uses IETL library to find the biggest eigenvalue.
    // IETL is modified to work in parallel (only ietl/interface/ublas.h).

    int nprocs, rank;
    MPI_Comm_size(A.comm, &nprocs);
    MPI_Comm_rank(A.comm, &rank);

//    MPI_Barrier(A.comm);
//    for(unsigned long i = 0; i < A.nnz_l_local; i++) {
//        if(rank==0) printf("%lu \t%u \t%f \tietl, local \n", i, A.row_local[i], (A.values_local[i]*A.invDiag[A.row_local[i]] - A.entry[i].val * A.invDiag[A.entry[i].row - A.split[rank]]));
//        if(rank==0) printf("%lu \t%u \t%f \t%f \t%f \tietl, local \n", i, A.row_local[i]+A.split[rank], A.values_local[i], A.invDiag[A.row_local[i]], A.values_local[i]*A.invDiag[A.row_local[i]]);
//        A.values_local[i] *= A.invDiag[A.row_local[i]];
//        if(rank==0) printf("%lu \t%u \t%f \tietl, local \n", i, A.row_local[i], A.values_local[i]);
//    }

//    MPI_Barrier(A.comm);
//    for(unsigned long i = 0; i < A.nnz_l_remote; i++) {
//        if(rank==0) printf("%lu \t%u \t%f \t%f \t%f \tietl, remote \n", i, A.row_remote[i], A.values_remote[i], A.invDiag[A.row_remote[i]], A.values_remote[i]*A.invDiag[A.row_remote[i]]);
//        A.values_remote[i] *= A.invDiag[A.row_remote[i]];
//    }
//    MPI_Barrier(A.comm);

    typedef ietl::vectorspace<Vector> Vecspace;
    typedef boost::lagged_fibonacci607 Gen;

    const int N = A.Mbig;
    Vecspace vec(N);
    Gen mygen;
    ietl::lanczos<Matrix,Vecspace> lanczos(A,vec);

    // Creation of an iteration object:
    int max_iter = 8*N;
    double rel_tol = 500*std::numeric_limits<double>::epsilon();
    double abs_tol = std::pow(std::numeric_limits<double>::epsilon(),2./3);
    int n_highest_eigenval = 1;
    ietl::lanczos_iteration_nhighest<double>
            iter(max_iter, n_highest_eigenval, rel_tol, abs_tol);

//    if(rank==0) std::cout << "-----------------------------------\n\n";
//    if(rank==0) std::cout << "\nComputation of " << n_highest_eigenval << " highest converged eigenvalues\n\n";

    std::vector<double> eigen;
    std::vector<double> err;
    std::vector<int> multiplicity;

    try{
        lanczos.calculate_eigenvalues(iter,mygen);
        eigen = lanczos.eigenvalues();
        err = lanczos.errors();
        multiplicity = lanczos.multiplicities();
        if(rank==0) std::cout<<"\nnumber of iterations to compute the biggest eigenvalue: "<<iter.iterations()<<"\n";
    }
    catch (std::runtime_error& e) {
        std::cout << e.what() << "\n";
    }

    // Printing eigenvalues with error & multiplicities:
//    if(rank==0) std::cout << "#        eigenvalue            error         multiplicity\n";
//    std::cout.precision(10);
//    if(rank==0) {
//        for (int i = 0; i < eigen.size(); i++)
//            std::cout << i << "\t" << eigen[i] << "\t" << err[i] << "\t"
//                      << multiplicity[i] << "\n";}
//    if(rank==0) {
//        for (int i = eigen.size()-1; i > eigen.size()-1-n_highest_eigenval; --i)
//            std::cout << i << "\t" << eigen[i] << "\t" << err[i] << "\t"
//                      << multiplicity[i] << "\n";}

    if(rank==0)
        printf("the biggest eigenvalue is %f (IETL) \n", eigen.back());

    A.eig_max_of_invdiagXA = eigen.back();

//    MPI_Barrier(A.comm);
//    for(unsigned long i = 0; i < A.nnz_l_local; i++)
//        A.values_local[i] /= A.invDiag[A.row_local[i]];
//
//    for(unsigned long i = 0; i < A.nnz_l_remote; i++)
//        A.values_remote[i] /= A.invDiag[A.row_remote[i]];

    return 0;
}

#endif //MATVEC003_IETL_SAENA_H
