#include <iostream>
#include <random>

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


int randomVector(unsigned long* V, unsigned long size){

//    int rank;
//    MPI_Comm_rank(comm, &rank);

    //Type of random number distribution
    std::uniform_int_distribution<unsigned long> dist(1, size); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i=0; i<size; i++)
        V[i] = dist(rng);

//    if(rank==0){
//        V[6] = 100;
//        V[11] = 100;
//    }

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


double myNorm(std::vector<double>& v){
    double norm = 0;
    for(long i=0; i<v.size(); i++)
        norm += v[i] * v[i];

    norm = sqrt(norm);
    return  norm;
}