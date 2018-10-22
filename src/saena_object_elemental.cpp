#include "saena_object.h"
#include "saena_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "dollar.hpp"
//#include "El.hpp"


// saena_object::find_eig_Elemental
/*
int saena_object::find_eig_Elemental(saena_matrix& A) {

    int argc = 0;
    char** argv = {NULL};
//    El::Environment env( argc, argv );
    El::Initialize( argc, argv );

    int rank, nprocs;
    MPI_Comm_rank(A.comm, &rank);
    MPI_Comm_size(A.comm, &nprocs);

    const El::Int n = A.Mbig;

    // *************************** serial ***************************

//    El::Matrix<double> A(n,n);
//    El::Zero( A );
//    for(unsigned long i = 0; i<nnz_l; i++)
//        A(entry[i].row, entry[i].col) = entry[i].val * inv_diag[entry[i].row];

//    El::Print( A, "\nGlobal Elemental matrix (serial):\n" );

//    El::Matrix<El::Complex<double>> w(n,1);

    // *************************** parallel ***************************

    El::DistMatrix<value_t> B(n,n);
    El::Zero( B );
    B.Reserve(A.nnz_l);
    for(nnz_t i = 0; i < A.nnz_l; i++){
//        if(rank==0) printf("%lu \t%u \t%f \t%f \t%f \telemental\n",
//                           i, A.entry[i].row, A.entry[i].val, A.inv_diag[A.entry[i].row - A.split[rank]], A.entry[i].val*A.inv_diag[A.entry[i].row - A.split[rank]]);
//        B.QueueUpdate(A.entry[i].row, A.entry[i].col, A.entry[i].val * A.inv_diag[A.entry[i].row - A.split[rank]]); // this is not A! each entry is multiplied by the same-row diagonal value.
        B.QueueUpdate(A.entry[i].row, A.entry[i].col, A.entry[i].val);
    }
    B.ProcessQueues();
//    El::Print( A, "\nGlobal Elemental matrix:\n" );

    El::DistMatrix<El::Complex<value_t>> w(n,1);

    // *************************** common part between serial and parallel ***************************

    El::SchurCtrl<double> schurCtrl;
    schurCtrl.time = false;
//    schurCtrl.hessSchurCtrl.progress = true;
//    El::Schur( A, w, V, schurCtrl ); //  eigenvectors will be saved in V.

//    printf("before Schur!\n");
    El::Schur( B, w, schurCtrl ); // eigenvalues will be saved in w.
//    printf("after Schur!\n");
//    MPI_Barrier(comm); El::Print( w, "eigenvalues:" ); MPI_Barrier(comm);

//    A.eig_max_of_invdiagXA = w.Get(0,0).real();
//    for(unsigned long i = 1; i < n; i++)
//        if(w.Get(i,0).real() > A.eig_max_of_invdiagXA)
//            A.eig_max_of_invdiagXA = w.Get(i,0).real();

    // todo: if the matrix is not symmetric, the eigenvalue will be a complex number.
    A.eig_max_of_invdiagXA = fabs(w.Get(0,0).real());
    for(index_t i = 1; i < n; i++) {
//       std::cout << i << "\t" << w.Get(i, 0) << std::endl;
        if (fabs(w.Get(i, 0).real()) > A.eig_max_of_invdiagXA)
            A.eig_max_of_invdiagXA = fabs(w.Get(i, 0).real());
    }

//    if(rank==0) printf("\nthe biggest eigenvalue of A is %f (Elemental) \n", A.eig_max_of_invdiagXA);

    El::Finalize();

    return 0;
}
*/


// saena_object::solve_coarsest_Elemental
/*
int saena_object::solve_coarsest_Elemental(saena_matrix *A_S, std::vector<value_t> &u, std::vector<value_t> &rhs){

    int argc = 0;
    char** argv = {NULL};
//    El::Environment env( argc, argv );
    El::Initialize( argc, argv );

    int rank, nprocs;
    MPI_Comm_rank(A_S->comm, &rank);
    MPI_Comm_size(A_S->comm, &nprocs);

//    printf("solve_coarsest_Elemental!\n");

    const El::Unsigned n = A_S->Mbig;
//    printf("size = %d\n", n);

    // set the matrix
    // --------------
    El::DistMatrix<value_t> A(n,n);
    El::Zero( A );
    A.Reserve(A_S->nnz_l);
    for(nnz_t i = 0; i < A_S->nnz_l; i++){
//        if(rank==1) printf("%lu \t%lu \t%f \n", A_S->entry[i].row, A_S->entry[i].col, A_S->entry[i].val);
        A.QueueUpdate(A_S->entry[i].row, A_S->entry[i].col, A_S->entry[i].val);
    }
    A.ProcessQueues();
//    El::Print( A, "\nGlobal Elemental matrix:\n" );

    // set the rhs
    // --------------
    El::DistMatrix<value_t> w(n,1);
    El::Zero( w );
    w.Reserve(n);
    for(index_t i = 0; i < rhs.size(); i++){
//        if(rank==0) printf("%lu \t%f \n", i+A_S->split[rank], rhs[i]);
        w.QueueUpdate(i+A_S->split[rank], 0, rhs[i]);
    }
    w.ProcessQueues();
//    El::Print( w, "\nrhs (w):\n" );

    // solve the system
    // --------------
    // w is the rhs. after calling the solve function, it will be the solution.
//    El::DistMatrix<double> C(n,n);
//    El::SymmetricSolve(El::LOWER, El::NORMAL, &A, &);
    El::LinearSolve(A, w);
//    El::Print( w, "\nsolution (w):\n" );

//    double temp;
//    if(rank==1) printf("w solution:\n");
//    for(unsigned long i = A_S->split[rank]; i < A_S->split[rank+1]; i++){
//        if(rank==1) printf("before: %lu \t%f \n", i, w.Get(i,0));
//        temp = w.Get(i,0);
//        u[i-A_S->split[rank]] = temp;
//        if(rank==0) printf("rank = %d \t%lu \t%f \n", rank, i, u[i-A_S->split[rank]]);
//        if(rank==1) printf("rank = %d \t%lu \t%f \n", rank, i, u[i-A_S->split[rank]]);
//        if(rank==0) printf("rank = %d \t%lu \t%f \n", rank, i, temp);
//        if(rank==1) printf("rank = %d \t%lu \t%f \n", rank, i, temp);
//    }

    std::vector<value_t> temp(n);
    for(index_t i = 0; i < n; i++){
        temp[i] = w.Get(i,0);
//        if(rank==1) printf("rank = %d \t%lu \t%f \n", rank, i, temp[i]);
    }

    for(index_t i = A_S->split[rank]; i < A_S->split[rank+1]; i++)
        u[i-A_S->split[rank]] = temp[i];

    El::Finalize();

    return 0;
}
*/
