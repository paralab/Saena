//
// Created by abaris on 10/14/16.
//

#include "coomatrix.h"

CSCMatrix::CSCMatrix(int s1, int s2, double** A){

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
    row = (int*)malloc(sizeof(int)*nnz);
    col = (int*)malloc(sizeof(int)*nnz);

    int iter = 0;
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if (A[i][j] > matrixTol){
                values[iter] = A[i][j];
                row[iter] = i;
                col[iter] = j;
                iter++;
            }
        } //for j
    } //for i
}

CSCMatrix::~CSCMatrix()
{
    free(values);
    free(row);
    free(col);
}

void CSCMatrix::matvec(double* v, double* w, int M, int N){
    for(unsigned int i=0;i<nnz;i++) {
        w[row[i]] += values[i] * col[i];
        //w[i] += values[j] * v[row[j]];
    }
}

void CSCMatrix::valprint(){
    for(unsigned int i=0;i<nnz;i++) {
        cout << values[i] << endl;
    }
}

void CSCMatrix::rowprint(){
    for(unsigned int i=0;i<nnz;i++) {
        cout << row[i] << endl;
    }
}

void CSCMatrix::colprint(){
    for(unsigned int i=0;i<nnz;i++) {
        cout << col[i] << endl;
    }
}

void CSCMatrix::print(){
    for(unsigned int i=0;i<nnz;i++) {
        cout << "(" << row[i] << " , " << col[i] << " , " << values[i] << ")" << endl;
    }
}