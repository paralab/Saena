#ifndef SAENA_CSRMATRIX_H
#define SAENA_CSRMATRIX_H

#include <vector>

class CSRMatrix {
public:
    std::vector<long>   rowIndex;
    std::vector<long>   col;
    std::vector<double> values;
//    long*   rowIndex;
//    long*   col;
//    double* values;

    long* split;

    long M;
    long Mbig;
    long nnz_l;
    long nnz_g;
    double average_sparsity;

//    CSRMatrix(){}
    int CSRMatrixSet(long* row, long* col, double* values, long M, long Mbig, long nnzl, long* split);
    ~CSRMatrix();
    void print(int rank);
};

#endif //SAENA_CSRMATRIX_H
