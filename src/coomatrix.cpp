//
// Created by abaris on 10/14/16.
//
#include <fstream>
#include <algorithm>

#include "coomatrix.h"
#include "mpi.h"

COOMatrix::COOMatrix(string filePath) {

    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream inFile;
    inFile.open(filePath, ios::binary);

    if (!inFile.is_open()){
        cout << "Couldn't read the file!" << endl;
    }

    // Ignore headers and comments:
    while (inFile.peek() == '%') inFile.ignore(2048, '\n');

    long int Mbig; // number of rows of the original matrix before partitioning.
    long int Nbig; // number of rows of the original matrix before partitioning.
    //nnz  number of nonzeros in the original matrix

    // Read defining parameters:
    inFile.read((char*)&Mbig, sizeof(long int));
    inFile.read((char*)&Nbig, sizeof(long int));
    inFile.read((char*)&nnz, sizeof(long int));
    //cout << "M = " << Mbig << ", N = " << Nbig << ", nnz = " << nnz << endl;

    M = Mbig/p;
    N = Nbig;

    values = (double *) malloc(sizeof(double) * nnz);
    row = (int *) malloc(sizeof(long int) * nnz);
    col = (int *) malloc(sizeof(long int) * nnz);
    vElement = (int *) malloc(sizeof(long int) * nnz);
    vElementRep = (int *) malloc(sizeof(long int) * nnz);
    vElementSize = 0;
    recvCount = (int*)malloc(sizeof(long int)*p);
    std::fill(recvCount, recvCount + p, 0.);
    int procNo = 0;

    inFile.read((char*)&row[0], sizeof(long int));
    inFile.read((char*)&col[0], sizeof(long int));
    inFile.read((char*)&values[0], sizeof(double));

    vElement[0] = col[0];
    vElementSize = 1;
    vElementRep[0] = 1;
    recvCount[findProcess(col[0], procNo, p)] = 1;

    for (long int i = 1; i < nnz; i++) {
        inFile.read((char*)&row[i], sizeof(long int));
        inFile.read((char*)&col[i], sizeof(long int));
        inFile.read((char*)&values[i], sizeof(double));

        if(col[i] == col[i-1]){
            vElementRep[vElementSize-1] = vElementRep[vElementSize-1] + 1;
        }else{
            vElement[vElementSize] = col[i];
            vElementRep[vElementSize] = 1;
            vElementSize++;

            procNo = findProcess(col[i], procNo, p);
            recvCount[procNo] = recvCount[procNo] + 1;
            cout << "recvCount[procNo] = " << recvCount[procNo] << endl;
        }
        //cout << "i = " << i << ", row = " << row[i] << endl;
    }


/*    vIndexCount = (int*)malloc(sizeof(int)*p);

    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, MPI_COMM_WORLD);

    int vIndexSize = 0;
    for (long int i=0; i<p; i++)
        vIndexSize += vIndexCount[i];

    vIndex = (int*)malloc(sizeof(int)*vIndexSize);
    int* vBuf = (int*)malloc(sizeof(int)*nnz);

    for (long int i=0; i<vElementSize; i++)
        vBuf[i] = vElement[i]%vSize;

    int* vdispls = (int*)malloc(sizeof(int)*p);
    int* rdispls = (int*)malloc(sizeof(int)*p);
    vdispls[0] = 0;
    rdispls[0] = 0;

    for (int i=1; i<p; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }

    MPI_Alltoallv(vBuf, recvCount, rdispls, MPI_INT, vIndex, vIndexCount, vdispls, MPI_INT, MPI_COMM_WORLD);

    free(vdispls);
    free(rdispls);
    free(vBuf);*/

}

COOMatrix::~COOMatrix() {
    free(values);
    free(row);
    free(col);
    free(vElement);
    free(vElementRep);
    free(recvCount);
}

void COOMatrix::matvec(double* v, double* w, long int M, long int N){
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long int vIndexSize = 0;
    for (long int i=0; i<p; i++)
        vIndexSize += vIndexCount[i];

    double *vSend = (double*)malloc(sizeof(double) * vIndexSize);
    for(long int i=0;i<vIndexSize;i++){
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

    double *vValuesCompressed = (double *) malloc(sizeof(double) * vIndexSize);
    MPI_Alltoallv(vSend, vIndexCount, vdispls, MPI_DOUBLE, vValuesCompressed, recvCount, rdispls, MPI_DOUBLE, MPI_COMM_WORLD);

    long int iter = 0;
    double *vValues = (double *) malloc(sizeof(double) * nnz);
    for (long int i=0; i<vElementSize; i++){
        for (long int j=0; j<vElementRep[i]; j++) {
            vValues[iter] = vValuesCompressed[i];
            iter++;
        }
    }

    // the following and the above loops can be combined to have only one nested for loop.
    for(long int i=0;i<nnz;i++) {
        w[row[i]] += values[i] * vValues[i];
        //w[row[i]] += values[i] * col[i];
        //w[i] += values[j] * v[row[j]];
    }

    free(vSend);
    free(vdispls);
    free(rdispls);
    free(vValuesCompressed);
    free(vValues);
}

void COOMatrix::valprint(){
    cout << "val:" << endl;
    for(long int i=0;i<nnz;i++) {
        cout << values[i] << endl;
    }
}

void COOMatrix::rowprint(){
    cout << "row:" << endl;
    for(long int i=0;i<nnz;i++) {
        cout << row[i] << endl;
    }
}

void COOMatrix::colprint(){
    cout << endl << "col:" << endl;
    for(long int i=0;i<nnz;i++) {
        cout << col[i] << endl;
    }
}

void COOMatrix::vElementprint(){
    cout << endl << "vElement:" << endl;
    for(long int i=0;i<vElementSize;i++) {
        cout << vElement[i] << endl;
    }
}

void COOMatrix::vElementRepprint(){
    cout << endl << "vElementRep:" << endl;
    for(long int i=0;i<vElementSize;i++) {
        cout << vElementRep[i] << endl;
    }
}

void COOMatrix::print(){
    cout << endl << "triple:" << endl;
    for(long int i=0;i<nnz;i++) {
        cout << "(" << row[i] << " , " << col[i] << " , " << values[i] << ")" << endl;
    }
}

int COOMatrix::findProcess(int a, int procNo, int p) {
    while(procNo < p){
        if (a >= procNo*M && a < (procNo+1)*M)
            return procNo;

        procNo++;
    }
    return procNo;
}