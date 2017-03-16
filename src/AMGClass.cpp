//
// Created by abaris on 3/14/17.
//

#include <cstdio>
#include <algorithm>
#include <mpi.h>

#include "AMGClass.h"
//#include "coomatrix.h"
//#include "csrmatrix.h"


// sort indices and store the ordering.
class sort_indices
{
private:
    long* mparr;
public:
    sort_indices(long* parr) : mparr(parr) {}
    bool operator()(long i, long j) const { return mparr[i]<mparr[j]; }
};

// binary search tree using the lower bound
template <class T>
T lower_bound2(T *left, T *right, T val) {
    T* first = left;
    while (left < right) {
        T *middle = left + (right - left) / 2;
        if (*middle < val){
            left = middle + 1;
        }
        else{
            right = middle;
        }
    }
    if(val == *left){
        return distance(first, left);
    }
    else
        return distance(first, left-1);
}

AMGClass::AMGClass(int l, int vcycle_n, double relT, string sm, int preSm, int postSm, float connStr, float ta){
    levels = l;
    vcycle_num = vcycle_n;
    relTol  = relT;
    smoother = sm;
    preSmooth = preSm;
    postSmooth = postSm;
    connStrength = connStr;
    tau = ta;
} //AMGClass

AMGClass::~AMGClass(){}

int AMGClass::AMGsetup(COOMatrix* A, bool doSparsify){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    findAggregation(A);

    return 0;
}

int AMGClass::findAggregation(COOMatrix* A){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CSRMatrix S;
    createStrengthMatrix(A, &S);

//    unsigned int i;
//    if (rank==0)
//        for(i=0; i<S.M; i++){
//            for(long j=S.rowIndex[i]; j<S.rowIndex[i+1]; j++)
//                cout << "[" << i+1 << "," << S.col[j]+1 << "] = " << S.values[j] << endl;
//        }
//    S.print(0);

    Aggregation(&S);
    return 0;
}

int AMGClass::createStrengthMatrix(COOMatrix* A, CSRMatrix* S){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    if(rank==0) cout << "M = " << A->M << ", nnz_l = " << A->nnz_l << endl;

    unsigned int i;
    double maxPerRow[A->M];
    fill(&maxPerRow[0], &maxPerRow[A->M], 0);
    for(i=0; i<A->nnz_l; i++){
        if( A->row[i] != A->col[i] ){
            if(maxPerRow[A->row[i] - A->split[rank]] == 0)
                maxPerRow[A->row[i] - A->split[rank]] = -A->values[i];
            else if(maxPerRow[A->row[i] - A->split[rank]] < -A->values[i])
                maxPerRow[A->row[i] - A->split[rank]] = -A->values[i]; // use split to convert the index from global to local.
        }
    }

/*    if(rank==0)
        for(i=0; i<A->M; i++)
            cout << i << "\t" << maxPerRow[i] << endl;*/

    std::vector<long> Si;
    std::vector<long> Sj;
    std::vector<double> Sval;
    for(i=0; i<A->nnz_l; i++){
        if(A->row[i] == A->col[i]) {
            Si.push_back(A->row[i]);
            Sj.push_back(A->col[i]);
            Sval.push_back(1);
        }
        else if(maxPerRow[A->row[i] - A->split[rank]] != 0) {
//            if ( -A->values[i] / (maxPerRow[A->row[i] - A->split[rank]] ) > connStrength) {
                Si.push_back(A->row[i]);
                Sj.push_back(A->col[i]);
                Sval.push_back(  -A->values[i] / (maxPerRow[A->row[i] - A->split[rank]])  );
//                if (rank==0) cout << Sval[Sval.size()-1] << "\t" << connStrength << endl;
//                if(rank==1) cout << "index = " << A->row[i] - A->split[rank] << ", max = " << maxPerRow[A->row[i] - A->split[rank]] << endl;
//                if(rank==0) cout << "A.val = " << -A->values[i] << ", max = " << maxPerRow[A->row[i] - A->split[rank]] << ", divide = " << (-A->values[i] / (maxPerRow[A->row[i] - A->split[rank]])) << endl;
//            }
        }
    }

/*    if(rank==0)
        for (i=0; i<Si.size(); i++)
            cout << "val = " << Sval[i] << endl;*/

    double local_maxPerCol[A->Mbig];
    fill(&local_maxPerCol[0], &local_maxPerCol[A->Mbig], 0);
    for(i=0; i<A->nnz_l; i++){
        if( A->row[i] != A->col[i] ){
            if(local_maxPerCol[A->col[i]] == 0)
                local_maxPerCol[A->col[i]] = -A->values[i];
            else if(local_maxPerCol[A->col[i]] < -A->values[i])
                local_maxPerCol[A->col[i]] = -A->values[i];
        }
    }

    double maxPerCol[A->Mbig];
    MPI_Allreduce(&local_maxPerCol, &maxPerCol, A->Mbig, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//    if(rank==0)
//        for(i=0; i<A->Mbig; i++)
//            cout << i << "\t" << maxPerCol[i] << endl;

    std::vector<long> STi;
    std::vector<long> STj;
    std::vector<double> STval;
    for(i=0; i<A->nnz_l; i++){
        if(A->row[i] == A->col[i]) {
            STi.push_back(A->row[i]);
            STj.push_back(A->col[i]);
            STval.push_back(1);
        }
        else{
//            if ( (-A->values[i] / maxPerCol[A->col[i]]) > connStrength) {
                STi.push_back(A->row[i]);
                STj.push_back(A->col[i]);
                STval.push_back( -A->values[i] / maxPerCol[A->col[i]] );
//            }
        }
    }

//    if(rank==1)
//        for(i=0; i<STi.size(); i++){
//            cout << "S:  " << "[" << Si[i]+1 << "," << Sj[i]+1 << "] = " << Sval[i] << endl;
//            cout << "ST: " << "[" << STi[i]+1 << "," << STj[i]+1 << "] = " << STval[i] << endl;
//        }

    // *************************** make S symmetric and apply the connection strength parameter ****************************

    std::vector<long> Si2;
    std::vector<long> Sj2;
    std::vector<double> Sval2;

    for(i=0; i<STi.size(); i++){
        if (Sval[i] <= connStrength && STval[i] <= connStrength)
            continue;
        else if (Sval[i] > connStrength && STval[i] <= connStrength){
            Si2.push_back(Si[i]);
            Sj2.push_back(Sj[i]);
            Sval2.push_back(0.5*Sval[i]);
        }
        else if (Sval[i] <= connStrength && STval[i] > connStrength){
            Si2.push_back(Si[i]);
            Sj2.push_back(Sj[i]);
            Sval2.push_back(0.5*STval[i]);
        }
        else{
            Si2.push_back(Si[i]);
            Sj2.push_back(Sj[i]);
            Sval2.push_back(0.5*(Sval[i] + STval[i]));
        }

//        if(rank==1) cout << "S:  " << "[" << Si[i]+1 << "," << Sj[i]+1 << "] = " << Sval[i] << endl;
//        if(rank==1) cout << "ST: " << "[" << STi[i]+1 << "," << STj[i]+1 << "] = " << STval[i] << endl;
//        Si2.push_back(Si[i]);
//        Sj2.push_back(Sj[i]);
//        Sval2.push_back(0.5*(Sval[i] + STval[i]));
    }

//    if(rank==0)
//        for(i=0; i<Si2.size(); i++)
//            cout << "S:  " << "[" << Si2[i]+1 << "," << Sj2[i]+1 << "] = " << Sval2[i] << endl;

    S->CSRMatrixSet(&(*(Si2.begin())), &(*(Sj2.begin())), &(*(Sval2.begin())), A->M, A->Mbig, Si2.size(), &(*(A->split.begin())));
    return 0;
}

int AMGClass::Aggregation(CSRMatrix* S){

    return 0;
};