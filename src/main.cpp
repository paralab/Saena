#include <iostream>
using namespace std;

double matrixTol = 1.e-6;

class CSSMatrix {

private:
    unsigned int nnz;
    int M;
    int N;

public:
    //CSSMatrix();
    CSSMatrix(int M, int N, double** A);

    ~CSSMatrix() {};

    //int& rowIndex(int i) { return ; }
    //int& colPtr(int i) { return ;}

    //CSSMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

    double *values = (double*)malloc(sizeof(double)*nnz);
    int *rows = (int*)malloc(sizeof(int)*nnz);
    int *pointerB = (int*)malloc(sizeof(int)*N);
    int *pointerE = (int*)malloc(sizeof(int)*N);

/*
    double* matvec(double v);
*/

};

CSSMatrix::CSSMatrix(int M, int N, double** A){

    cout<<"constructor!"<<endl;

    M = M;
    N = N;
    unsigned int nz = 0;

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if (A[i][j] > matrixTol){
                nz++;
            }
        }
    }
    nnz = nz;

    pointerB[0] = 0;

    int iter = 0;
    for(int i=0; i<M; i++){
        pointerB[i] = iter;

        for(int j=0; j<N; j++){
            if (A[i][j] > matrixTol){
                values[iter] = A[i][j];
                rows[iter] = i;
                iter++;
            }
        } //for j

        pointerE[i] = iter;
    } //for i

}



int main() {

    int M = 3;
    int N = M;

    double ** A = (double **)malloc(sizeof(double*)*M);
    for(unsigned int i=0;i<M;i++)
        A[i]=(double*) malloc(sizeof(double)*N);

    for(unsigned int i=0;i<M;i++)
        for(unsigned int j=0;j<N;j++)
            A[i][j]=i+j;


    CSSMatrix B (M,N,A);


    for(unsigned int i=0;i<M;i++)
        delete [] A[i];

    delete [] A;
    return 0;
}