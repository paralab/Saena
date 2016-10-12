//
// Created by abaris on 10/12/16.
//

#ifndef SAENA_CSCMATRIX2_H
#define SAENA_CSCMATRIX2_H

#include <iostream>
using namespace std;

#define matrixTol 1.e-6
/**
 * @author Majid
 * @breif Contains the basic structure to define css matrices
 *
 * */
class CSC2Matrix {

public:

    unsigned int nnz;
    int M;
    int N;
    double *values ;
    int *rows ;
    int *pointerB ;
    int *pointerE;


    //CSS2Matrix();
    /**
     * @param[in] M Number of rows in the matrix
     * @param[in] N Number of columns in the matrix
     * @param[in] A is a pointer to the matrix
     * */
    CSC2Matrix(int M, int N, double** A);
    ~CSC2Matrix();

    //int& rowIndex(int i) { return ; }
    //int& colPtr(int i) { return ;}

    //CSC2Matrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

/*
    double* matvec(double v);
*/

};

#endif //SAENA_CSCMATRIX2_H
