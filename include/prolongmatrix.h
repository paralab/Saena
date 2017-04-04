#ifndef SAENA_PROLONGMATRIX_H
#define SAENA_PROLONGMATRIX_H

#include <vector>

class prolongMatrix {
private:

public:
    long Mbig;
    long Nbig;
    long nnz_g;
    long nnz_l;

    std::vector<unsigned long> row;
    std::vector<unsigned long> col;
    std::vector<double> values;

    prolongMatrix();
    prolongMatrix(long Mbig, long Nbig, long nnz_g, long nnz_l, long* row, long* col, double* values);
    ~prolongMatrix();
};


#endif //SAENA_PROLONGMATRIX_H
