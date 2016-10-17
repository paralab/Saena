//
// Created by abaris on 10/14/16.
//

#include "coomatrix.h"
#include "mpi.h"

COOMatrix::COOMatrix(int s1, int s2, double** A) {

    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    M = s1;
    N = s2;

    int vSize = M;
    int vStart = rank * vSize;
    //int vEnd = vStart + vSize;

    nnz = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i][j] > matrixTol) {
                nnz++;
            }
        }
    }

    values = (double *) malloc(sizeof(double) * nnz);
    row = (int *) malloc(sizeof(int) * nnz);
    col = (int *) malloc(sizeof(int) * nnz);
    proc = (int *) malloc(sizeof(int) * nnz);
    recvCount = (int*)malloc(sizeof(int)*p);

    int procNo = -1;
    int iter = 0;
    for (int j = 0; j < N; j++) {
        if(j%vSize == 0) {procNo++; recvCount[procNo]=0;}
        for (int i = 0; i < M; i++) {
            if (A[i][j] > matrixTol) {
                values[iter] = A[i][j];
                row[iter] = i;
                col[iter] = j;
                proc[iter] = procNo;
                recvCount[procNo] = recvCount[procNo] + 1;
                iter++;
            }
        } //for i
    } //for j

    vIndexCount = (int*)malloc(sizeof(int)*p);

    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, MPI_COMM_WORLD);

    int vIndexSize = 0;
    for (unsigned int i=0; i<p; i++)
        vIndexSize += vIndexCount[i];

    vIndex = (int*)malloc(sizeof(int)*vIndexSize);
    int* recvBuf = (int*)malloc(sizeof(int)*nnz);

    for (unsigned int i=0; i<nnz; i++)
        recvBuf[i] = col[i]%vSize;

    int* vdispls = (int*)malloc(sizeof(int)*p);
    int* rdispls = (int*)malloc(sizeof(int)*p);
    vdispls[0] = 0;
    rdispls[0] = 0;

    for (int i=1; i<p; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }

    MPI_Alltoallv(recvBuf, recvCount, rdispls, MPI_INT, vIndex, vIndexCount, vdispls, MPI_INT, MPI_COMM_WORLD);


    vElement = (int *) malloc(sizeof(int) * nnz);
    vElement[0] = col[0];
    vElementSize = 1;

    vProcess = (int *) malloc(sizeof(int) * nnz);
    // fix this part:  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    vProcess[0] = 0;

    for (unsigned int coliter = 1; coliter < nnz; ++coliter) {
        if (vElement[vElementSize-1] != col[coliter]){
            vElement[vElementSize] = col[coliter];
            vProcess[vElementSize] = proc[coliter];
            vElementSize++;
        }
    }

    free(vdispls);
    free(rdispls);
    free(recvBuf);

}

COOMatrix::~COOMatrix()
{
    free(values);
    free(row);
    free(col);
    free(vElement);
    free(vProcess);
}

void COOMatrix::matvec(double* v, double* w, int M, int N){
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int vIndexSize = 0;
    for (unsigned int i=0; i<p; i++)
        vIndexSize += vIndexCount[i];

    double *vSend = (double*)malloc(sizeof(double) * vIndexSize);
    for(unsigned int i=0;i<vIndexSize;i++){
        vSend[i] = v[vIndex[i]];
    }

    int* vdispls = (int*)malloc(sizeof(int)*p);
    int* rdispls = (int*)malloc(sizeof(int)*p);
    vdispls[0] = 0;
    rdispls[0] = 0;
    for (int i=1; i<p; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }

    double *vValues = (double *) malloc(sizeof(double) * nnz);
    MPI_Alltoallv(vSend, vIndexCount, vdispls, MPI_DOUBLE, vValues, recvCount, rdispls, MPI_DOUBLE, MPI_COMM_WORLD);

    for(unsigned int i=0;i<nnz;i++) {
        w[row[i]] += values[i] * vValues[i];
        //w[row[i]] += values[i] * col[i];
        //w[i] += values[j] * v[row[j]];
    }

    free(vSend);
}

void COOMatrix::valprint(){
    cout << "val:" << endl;
    for(unsigned int i=0;i<nnz;i++) {
        cout << values[i] << endl;
    }
}

void COOMatrix::rowprint(){
    cout << "row:" << endl;
    for(unsigned int i=0;i<nnz;i++) {
        cout << row[i] << endl;
    }
}

void COOMatrix::colprint(){
    cout << "col:" << endl;
    for(unsigned int i=0;i<nnz;i++) {
        cout << col[i] << endl;
    }
}

void COOMatrix::vElementprint(){
    cout << "vElement:" << endl;
    for(unsigned int i=0;i<vElementSize;i++) {
        cout << vElement[i] << endl;
    }
}

void COOMatrix::vProcessprint(){
    cout << "vProcess:" << endl;
    for(unsigned int i=0;i<vElementSize;i++) {
        cout << vProcess[i] << endl;
    }
}

void COOMatrix::print(){
    cout << "triple:" << endl;
    for(unsigned int i=0;i<nnz;i++) {
        cout << "(" << row[i] << " , " << col[i] << " , " << values[i] << ")" << endl;
    }
}