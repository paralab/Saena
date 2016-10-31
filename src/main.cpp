#include <iostream>
#include <fstream>
#include <algorithm>
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

//    if (rank==0){
        //string filePath = "/home/abaris/Dropbox/Projects/Saena/data/LFAT5/LFAT5/LFAT5.mtx";
        string filePath = "/home/abaris/Dropbox/Projects/Saena/data/example.mtx";

        COOMatrix B (filePath);

        //B.print();

//    } //if rank






/*    if (rank ==0){
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
    }*/

/*
    if (rank ==0){
        for(unsigned int i=0;i<M;i++)
            for(unsigned int j=0;j<N;j++)
                A[i][j]=i+j;
        for(unsigned int i=0;i<M;i++)
            v[i] = i;
    } else if (rank == 1){
        for(unsigned int i=0;i<M;i++)
            for(unsigned int j=0;j<N;j++)
                A[i][j]=i+j+4;
        for(unsigned int i=0;i<M;i++)
            v[i] = i+4;
    } else if (rank == 2){
        for(unsigned int i=0;i<M;i++)
            for(unsigned int j=0;j<N;j++)
                A[i][j]=i+j+8;
        for(unsigned int i=0;i<M;i++)
            v[i] = i+8;
    }else{
        for(unsigned int i=0;i<M;i++)
            for(unsigned int j=0;j<N;j++)
                A[i][j]=i+j+12;
        for(unsigned int i=0;i<M;i++)
            v[i] = i+12;
    }
*/

/*    int a;
    if (rank == 0) {
        cout << endl<< "result should be: " << endl;
        for(unsigned int i=0;i<N;i++){
            a = 0;
            for(unsigned int j=0;j<N;j++)
                a += (i+j) * j;
            cout << a << endl;
        }
    }

    COOMatrix B (M, N, A);

    B.matvec(v, w, M, N);

    if (rank == 0) cout << endl << "result of matvec: " << endl;
    for(unsigned int i=0;i<M;i++)
        cout << w[i] << endl;

    free(v);
    free(w);*/

    MPI_Finalize();
    return 0;
}
