//
// Created by abaris on 10/14/16.
//

#ifndef SAENA_COOMATRIX_H
#define SAENA_COOMATRIX_H

#include <iostream>
using namespace std;

#define matrixTol 1.e-6
/**
 * @author Majid
 * @breif Contains the basic structure to define csc matrices
 *
 * */
class COOMatrix {

private:
    long *vIndex;
    int *vIndexCount;
    int *recvCount;
    long *split;
    //int findProcess(long a, int procNo, int p);

public:
    long M;
    //long N;
    long nnz_l;

    double *values;
    long *row;
    long *col;
    long *vElement;
    long vElementSize;
    long *vElementRep;
    void valprint();
    void rowprint();
    void colprint();
    void vElementprint();
    void vElementRepprint();
    void print();
    void matvec(double* v, double* w);

    //COOMatrix();
    /**
     * @param[in] M Number of rows in the matrix
     * @param[in] N Number of columns in the matrix
     * @param[in] A is a pointer to the matrix
     * */
    COOMatrix(const char* filePath, char* filePath2);
    ~COOMatrix();

    // for sparsifying:
    //COOMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

};

#endif //SAENA_COOMATRIX_H
