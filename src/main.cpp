#include <iostream>
#include <fstream>
#include <algorithm>
#include "coomatrix.h"
//#include "cscmatrix.h"
//#include "csrmatrix.h"
#include <time.h>
#include "mpi.h"

using namespace std;

int main(int argc, char** argv) {

    MPI_Init(NULL, NULL);
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const char* filePath = "/home/abaris/Dropbox/Projects/Saena/data/ch3-3-b1/ch3-3-b1.bin";
    char* filePath2 = "/home/abaris/Dropbox/Projects/Saena/data/ch3-3-b1/ch3-3-b1.bin";

    COOMatrix B (filePath, filePath2);

    srand (time(NULL));

    long Mbig = 18;
    //long Nbig = 9;
    double* v = (double*) malloc(sizeof(double) * B.M);
    double* w = (double*) malloc(sizeof(double) * B.M);

    for (long i=0; i<B.M; i++){
        v[i] = rand();
    }

    B.matvec(v, w);

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

    B.matvec(v, w, M, N);

    if (rank == 0) cout << endl << "result of matvec: " << endl;
    for(unsigned int i=0;i<M;i++)
        cout << w[i] << endl;
*/
    free(v);
    free(w);

    //free(filePath);
    //free(filePath2);

    MPI_Finalize();
    return 0;
}
