#include <iostream>
#include "cscmatrix.h"
#include "csrmatrix.h"
#include "mpi.h"

using namespace std;

int main() {

    MPI_Init(NULL, NULL);
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M = 2;
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

    CSCMatrix B (M,N,A);

    int vSize = M/p;
    int vStart = rank * vSize;
    int vEnd = vStart + vSize;

    int source;
    double v_remote;

    if (rank == 0){
        B.matvec(v, w, M, N);

        source = 0;
        for(int i=vEnd;i<N;i++){
            //update source for receive
            if(i%vSize) source++;

            if(B.colIndex[i] - B.colIndex[i-1] > 0){
                MPI_Recv(&v_remote, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j=B.colIndex[i]; j<B.colIndex[i+1]; j++)
                    w[B.rows[j]] += B.values[j] * v_remote;
            }
        }
    } else if (rank == 1){
        source = -1;
        for(int i=0;i<vStart;i++){
            //update source for receive
            if(i%vSize) source++;

            if(B.colIndex[i] - B.colIndex[i-1] > 0){
                MPI_Recv(&v_remote, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j=B.colIndex[i]; j<B.colIndex[i+1]; j++)
                    w[B.rows[j]] += B.values[j] * v_remote;
            }
        }
        B.matvec(v, w, M, N);
    }else{
        source = -1;
        for(int i=0;i<vStart;i++){
            //update source for receive
            if(i%vSize) source++;

            if(B.colIndex[i] - B.colIndex[i-1] > 0){
                MPI_Recv(&v_remote, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j=B.colIndex[i]; j<B.colIndex[i+1]; j++)
                    w[B.rows[j]] += B.values[j] * v_remote;
            }
        }

        B.matvec(v, w, M, N);

        source += vSize;
        for(int i=vEnd;i<N;i++){
            //update source for receive
            if(i%vSize) source++;

            if(B.colIndex[i] - B.colIndex[i-1] > 0){
                MPI_Recv(&v_remote, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j=B.colIndex[i]; j<B.colIndex[i+1]; j++)
                    w[B.rows[j]] += B.values[j] * v_remote;
            }
        }
    }


    /*
    for(unsigned int i=0;i<M;i++)
        for(unsigned int j=0;j<N;j++)
            A[i][j]=i+j;

    for(unsigned int i=0;i<N;i++)
        v[i] = i+1;

    B.matvec(v, w, M, N);

    for(unsigned int i=0;i<N;i++)
        cout << "w = " << w[i] << endl;
*/


    free(v);
    free(w);

    for(unsigned int i=0;i<M;i++)
        delete [] A[i];

    delete [] A;

    MPI_Finalize();
    return 0;
}
