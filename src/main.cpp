#include <iostream>
#include "coomatrix.h"
#include "cscmatrix.h"
#include "csrmatrix.h"
#include "mpi.h"

using namespace std;

int main(int argc, char** argv) {

    MPI_Init(NULL, NULL);
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M = 6/p;
    int N = 6;

    double ** A = (double **)malloc(sizeof(double*)*M);
    for(unsigned int i=0;i<M;i++)
        A[i]=(double*) malloc(sizeof(double)*N);

    double* v = (double*) malloc(sizeof(double)*M);
    double* w = (double*) malloc(sizeof(double)*M);

    if (rank ==0){
        A[0][0] = 2;
        A[0][1] = 3;
        A[1][1] = 6;
        A[0][3] = 2;
        A[0][4] = 3;
        A[1][4] = 1;

        v[0] = 2;
        v[1] = 5;
    } else if (rank == 1){
        A[0][1] = 2;
        A[1][1] = 5;
        A[0][2] = 6;
        A[1][2] = 5;
        A[0][3] = 7;
        A[0][5] = 6;

        v[0] = 1;
        v[1] = 3;
    } else{
        A[0][1] = 1;
        A[1][3] = 5;
        A[0][4] = 9;
        A[0][5] = 1;
        A[1][5] = 7;

        v[0] = 7;
        v[1] = 2;
    }

    COOMatrix B (M, N, A);

    B.matvec(v, w, M, N);

    for(unsigned int i=0;i<M;i++)
        cout << w[i] << endl;

    free(v);
    free(w);

    for(unsigned int i=0;i<M;i++)
        delete [] A[i];

    delete [] A;

    MPI_Finalize();
    return 0;
}
