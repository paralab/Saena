//
// Created by abaris on 10/12/16.
//

#ifndef SAENA_CSRMATRIX_H
#define SAENA_CSRMATRIX_H

#include <iostream>
using namespace std;

#define matrixTol 1.e-6
/**
 * @author Majid
 * @breif Contains the basic structure to define csr matrices
 *
 * */
class CSRMatrix {

public:

    unsigned int nnz;
    int M;
    int N;

    double *values;
    int *columns;
    int *rowIndex;

    //CSRMatrix();
    /**
     * @param[in] M Number of rows in the matrix
     * @param[in] N Number of columns in the matrix
     * @param[in] A is a pointer to the matrix
     * */
    CSRMatrix(int M, int N, double** A);
    ~CSRMatrix();

    // matvec is not written correctly
    void matvec(double* v, double* w, int M, int N);

    //int& rowIndex(int i) { return ; }
    //int& colPtr(int i) { return ;}

    //CSRMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

};

#endif //SAENA_CSRMATRIX_H
