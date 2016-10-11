//
// Created by abaris on 10/11/16.
//

#ifndef SAENA_CSSMATRIX_H
#define SAENA_CSSMATRIX_H

#include <iostream>
using namespace std;

#define matrixTol 1.e-6
/**
 * @author Majid
 * @breif Contains the basic structure to define css matrices
 *
 * */
class CSSMatrix {

public:

    unsigned int nnz;
    int M;
    int N;
    double *values ;
    int *rows ;
    int *pointerB ;
    int *pointerE;


    //CSSMatrix();
    /**
     * @param[in] M Number of rows in the matrix
     * @param[in] N Number of columns in the matrix
     * @param[in] A is a pointer to the matrix
     * */
    CSSMatrix(int M, int N, double** A);
    ~CSSMatrix();

    //int& rowIndex(int i) { return ; }
    //int& colPtr(int i) { return ;}

    //CSSMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

/*
    double* matvec(double v);
*/

};

#endif //SAENA_CSSMATRIX_H
