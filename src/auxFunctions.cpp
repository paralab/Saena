#include <iostream>
#include <random>
#include <auxFunctions.h>
#include <strengthmatrix.h>
//#include <algorithm>

using namespace std;

// binary search tree using the lower bound
template <class T>
T lower_bound2(T *left, T *right, T val){
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


int randomVector(std::vector<unsigned long>& V, long size, StrengthMatrix* S, MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    unsigned long max_weight = ( (1UL<<63) - 1);

    unsigned long i;
    unsigned int max_degree_local = 0;
    for(i=0; i<S->M; i++){
        if(S->nnzPerRow[i] > max_degree_local)
            max_degree_local = S->nnzPerRow[i];
    }

    unsigned int max_degree;
    MPI_Allreduce(&max_degree_local, &max_degree, 1, MPI_UNSIGNED, MPI_MAX, comm);
    max_degree++;
//    printf("rank = %d, max degree local = %lu, max degree = %lu \n", rank, max_degree_local, max_degree);

    //Type of random number distribution
//    std::uniform_real_distribution<float> dist(-1.0,1.0); //(min, max)
    unsigned int max_rand = ( (1UL<<32) - 1);
    std::uniform_int_distribution<unsigned int> dist(0,max_rand); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    std::vector<double> rand(S->M);
    for (i = 0; i < V.size(); i++){
        V[i] = ((unsigned long)(max_degree - S->nnzPerRow[i])<<32) + dist(rng);
//        V[i] = V[i] << 32;
//        V[i] += dist(rng);
//        if(rank==0) cout << i << "\tnnzPerRow = " << S->nnzPerRow[i] << "\t weight = " << V[i] << endl;
    }

    // to have one node with the highest weight possible, so that node will be a root and consequently P and R won't be zero matrices.
    // the median index is being chosen here.
    if (V.size() != 0)
        V[ floor(V.size()/2) ] = max_weight;

    return 0;
}

int randomVector2(std::vector<double>& V){

//    int rank;
//    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    //Type of random number distribution
    std::uniform_real_distribution<double> dist(0.0,1.0); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i=0; i<V.size(); i++){
        V[i] = dist(rng);
    }

    return 0;
}


int randomVector3(std::vector<unsigned long>& V, long size, StrengthMatrix* S, MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    unsigned long i;
    unsigned long max_degree_local = 0;
    for(i=0; i<S->M; i++){
        if(S->nnzPerRow[i] > max_degree_local)
            max_degree_local = S->nnzPerRow[i];
    }

    unsigned long max_degree;
    MPI_Allreduce(&max_degree_local, &max_degree, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    max_degree++;
//    printf("rank = %d, max degree local = %lu, max degree = %lu \n", rank, max_degree_local, max_degree);

    //Type of random number distribution
    std::uniform_real_distribution<double> dist(-1.0,1.0); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    std::vector<double> rand(S->M);
    for (i = 0; i < V.size(); i++){
        V[i] = max_degree - S->nnzPerRow[i];
//        if(rank==0) cout << i << "\tnnzPerRow = " << S->nnzPerRow[i] << "\t weight = " << V[i] << endl;
    }
//        rand[i] = dist(rng);

    // to have one node with the highest weight possible, so that node will be a root and consequently P and R won't be zero matrices.
    // the median index is being chosen here.
    // todo: fix this later. doing this as follows will affect the aggregation in a bad way.
    if (V.size() != 0)
        V[ floor(V.size()/2) ] = size + 1;

    return 0;
}

int randomVector4(std::vector<unsigned long>& V, long size) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    //Type of random number distribution
    std::uniform_int_distribution<unsigned long> dist(1, size); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i = 0; i < V.size(); i++)
        V[i] = dist(rng);

    // to have one node with the highest weight possible, so that node will be a root and consequently P and R won't be zero matrices.
    // the median index is being chosen here.
    if (V.size() != 0)
        V[ floor(V.size()/2) ] = size + 1;

    return 0;
}


//template <class T>
//float myNorm(std::vector<T>& v){
//    float norm = 0;
//    for(long i=0; i<v.size(); i++)
//        norm += v[i] * v[i];
//
//    norm = sqrt(norm);
//    return  norm;
//}

/*
double myNorm(std::vector<double>& v){
    double norm = 0;
    for(long i=0; i<v.size(); i++)
        norm += v[i] * v[i];

    norm = sqrt(norm);
    return  norm;
}
*/
