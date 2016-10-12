//
// Created by abaris on 10/12/16.
//

#ifndef SAENA_CSCMATRIX_H
#define SAENA_CSCMATRIX_H

#include <iostream>
using namespace std;

#define matrixTol 1.e-6
/**
 * @author Majid
 * @breif Contains the basic structure to define csc matrices
 *
 * */
class CSCMatrix {

public:

    unsigned int nnz;
    int M;
    int N;

    double *values;
    int *rows;
    int *colIndex;


    //CSSMatrix();
    /**
     * @param[in] M Number of rows in the matrix
     * @param[in] N Number of columns in the matrix
     * @param[in] A is a pointer to the matrix
     * */
    CSCMatrix(int M, int N, double** A);
    ~CSCMatrix();

    void matvec(double* v, double* w, int M, int N);

    //int& rowIndex(int i) { return ; }
    //int& colPtr(int i) { return ;}

    //CSCMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

};

#endif //SAENA_CSCMATRIX_H
