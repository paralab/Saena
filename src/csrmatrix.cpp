//
// Created by abaris on 10/12/16.
//

#include "csrmatrix.h"

CSRMatrix::CSRMatrix(int s1, int s2, double** A){

    M = s1;
    N = s2;
    unsigned int nz = 0;

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if (A[i][j] > matrixTol){
                nz++;
            }
        }
    }
    nnz = nz;

    values = (double*)malloc(sizeof(double)*nnz);
    columns = (int*)malloc(sizeof(int)*nnz);
    rowIndex = (int*)malloc(sizeof(int)*(N+1));

    rowIndex[0] = 0;
    int iter = 0;
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            if (A[i][j] > matrixTol){
                values[iter] = A[i][j];
                columns[iter] = j;
                iter++;
            }
        } //for i
        rowIndex[i+1] = iter;
    } //for j

}

CSRMatrix::~CSRMatrix()
{
    free(values);
    free(columns);
    free(rowIndex);
}

// matvec is not written correctly
void CSRMatrix::matvec(double* v, double* w, int M, int N){
    for(unsigned int i=0;i<nnz;i++) {
        w[columns[i]] += values[i] * v[columns[i]];
        //w[i] += A[i][j] * v[j];
    }
}
