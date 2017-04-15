#include <iostream>

using namespace std;

// sort indices and store the ordering.
class sort_indices
{
private:
    unsigned long* mparr;
public:
    sort_indices(unsigned long* parr) : mparr(parr) {}
    bool operator()(unsigned long i, unsigned long j) const { return mparr[i]<mparr[j]; }
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