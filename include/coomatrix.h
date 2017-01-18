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
    int *recvCount;
    int *vIndexCount;
    long *vIndex;
    std::vector<int> splitOffset;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    int recvSize;
    int vIndexSize;
    int numRecvProc;
    int numSendProc;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    std::vector<double> values_local;
    std::vector<double> values_remote;
    std::vector<long> row_local;
    std::vector<long> row_remote;
    std::vector<long> col_local;
    std::vector<long> col_remote;
    long num_threads;
//    int thread_id;
    long* iter_local_array;
    long* iter_remote_array;

public:
    long M;
    //long N;
    unsigned long nnz_l;

    int nprocs, rank;

    std::vector<double> values;
    std::vector<long> row;
    std::vector<long> col;

    //long *vElement;
    //std::vector<long> vElement_local;
    std::vector<long> vElement_remote;
    //long vElementSize;
    //long vElementSize_local;
    //long vElementSize_remote;
    //long *vElementRep;
    std::vector<long> vElementRep_local;
    std::vector<long> vElementRep_remote;

    std::vector<long> split;
    double *vSend;
    double* vecValues;

    void matvec(double* v, double* w);

    double time[4];
    double totalTime=0;

    // print functions
    void valprint();
    void rowprint();
    void colprint();
    //void vElementprint();
    //void vElementprint_local();
    void vElementprint_remote();
    //void vElementRepprint();
    void vElementRepprint_local();
    void vElementRepprint_remote();
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
