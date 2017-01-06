#ifndef SAENA_COOMATRIX_H
#define SAENA_COOMATRIX_H

#include <iostream>
#include <vector>
using namespace std;

#define matrixTol 1.e-6
/**
 * @author Majid
 * @breif Contains the basic structure to define coo matrices
 *
 * */
class COOMatrix {

private:
    long *vIndex;
    int *vIndexCount;
    int *recvCount;
    std::vector<int> splitOffset;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    int vIndexSize;
    int recvSize;

public:
    long M;
    //long N;
    unsigned long nnz_l;

    int nprocs, rank;

    std::vector<double> values;
    std::vector<long> row;
    std::vector<long> col;

    long *vElement;
    long vElementSize;
    long *vElementRep;
    std::vector<long> split;
    double *vSend;
    double* vecValues;

    void matvec(double* v, double* w);

    // print functions
    void valprint();
    void rowprint();
    void colprint();
    void vElementprint();
    void vElementRepprint();
    void print();

    //COOMatrix();
    /**
     * @param[in] Aname is the pointer to the matrix
     * @param[in] Mbig Number of rows in the matrix
     * */
    COOMatrix(char* Aname, long Mbig);
    ~COOMatrix();

    // for sparsifying:
    //COOMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

};

#endif //SAENA_COOMATRIX_H
