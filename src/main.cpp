#include <iostream>
#include "cscmatrix.h"
#include "csrmatrix.h"
using namespace std;


int main() {

    int M = 20;
    int N = M;

    double ** A = (double **)malloc(sizeof(double*)*M);
    for(unsigned int i=0;i<M;i++)
        A[i]=(double*) malloc(sizeof(double)*N);

    for(unsigned int i=0;i<M;i++)
        for(unsigned int j=0;j<N;j++)
            A[i][j]=i+j;

    CSCMatrix B (M,N,A);

    double* v = (double*) malloc(sizeof(double)*N);
    double* w = (double*) malloc(sizeof(double)*N);

    for(unsigned int i=0;i<N;i++)
        v[i] = i+1;

    B.matvec(v, w, M, N);

    for(unsigned int i=0;i<N;i++)
        cout << "w = " << w[i] << endl;


    for(unsigned int i=0;i<M;i++)
        delete [] A[i];

    delete [] A;
    return 0;
}