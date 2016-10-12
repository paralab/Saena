#include <iostream>
#include "cssmatrix.h"
using namespace std;


int main() {

    int M = 3;
    int N = M;

    double ** A = (double **)malloc(sizeof(double*)*M);
    for(unsigned int i=0;i<M;i++)
        A[i]=(double*) malloc(sizeof(double)*N);

    for(unsigned int i=0;i<M;i++)
        for(unsigned int j=0;j<N;j++)
            A[i][j]=i+j;


    CSSMatrix B (M,N,A);


    for(unsigned int i=0;i<M;i++)
        delete [] A[i];

    delete [] A;
    return 0;
}