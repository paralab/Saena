#include <cmath>

#include "saena_object.h"
#include "saena_matrix.h"
#include "aux_functions.h"
#include "dollar.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <random>
#include <mpi.h>

#include <trsl/is_picked_systematic.hpp>
#include <trsl/ppfilter_iterator.hpp>
//#include <numeric> // accumulate
//#include <cassert>


typedef std::vector<cooEntry>::iterator population_iterator;

typedef trsl::is_picked_systematic<cooEntry> is_picked;

typedef trsl::persistent_filter_iterator
<is_picked, population_iterator> sample_iterator;

typedef trsl::ppfilter_iterator<
is_picked, std::vector<cooEntry>::const_iterator
> sample_iterator2;


int saena_object::sparsify_trsl1(std::vector<cooEntry>& A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm) { $

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if(rank==0) std::cout << "\n" << __func__ << ":" << std::endl;

    nnz_t population_size = A.size();
//    nnz_t sample_size = nnz_t(2 * population_size);
    printf("\npopulation_size = %lu, sample_size = %lu \n", population_size, sample_size);

//    std::vector<cooEntry> const& const_pop = A;

    {
        //----------------------------//
        // Sample from the population //
        //----------------------------//

        auto populationIteratorBegin = A.begin(), // type is population_iterator
                populationIteratorEnd   = A.end();

        is_picked predicate(sample_size, norm_frob_sq, &cooEntry::get_val_sq);
//        is_picked predicate(sample_size, norm_frob_sq, std::ptr_fun(spars_prob));

        sample_iterator sampleIteratorBegin(predicate,
                                            populationIteratorBegin,
                                            populationIteratorEnd);
        sample_iterator sampleIteratorEnd(predicate,
                                          populationIteratorEnd,
                                          populationIteratorEnd);

//        std::cout << "\nSample of " << sample_size << " elements:" << std::endl;
//        std::copy(sampleIteratorBegin,
//                  sampleIteratorEnd,
//                  std::ostream_iterator<cooEntry>(std::cout, "\n"));
//        std::cout << std::endl;

//        std::cout << "sample vector" << std::endl;
        std::vector<cooEntry> A_spars_dup;
        for (sample_iterator
                     sb = sampleIteratorBegin,
                     si = sb,
                     se = sampleIteratorEnd;
             si != se; ++si){

//            std::cout << std::distance(sb, se) << "\t" << *si << std::endl;
            A_spars_dup.emplace_back(*si);
        }

        std::sort(A_spars_dup.begin(), A_spars_dup.end());
//        print_vector(A_spars_dup, -1, "A_spars_dup", comm);

        // remove duplicates
        for(nnz_t i=0; i<A_spars_dup.size(); i++){
            A_spars.emplace_back( A_spars_dup[i] );
            while(i<A_spars_dup.size()-1 && A_spars_dup[i] == A_spars_dup[i+1]){ // values of entries with the same row and col should be added.
                A_spars.back().val += A_spars_dup[i+1].val;
                i++;
            }
        }

    }

    return 0;
}


int saena_object::sparsify_trsl2(std::vector<cooEntry>& A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm) {

    // function is not complete!

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if(rank==0) std::cout << "\n" << __func__ << ":" << std::endl;

    nnz_t population_size = A.size();
//    nnz_t sample_size = nnz_t(0.95 * population_size);
//    nnz_t sample_size = 9;
//    printf("population_size = %lu, sample_size = %lu\n", population_size, sample_size);

//    for(nnz_t i = 0; i < A.size(); i++){
//        A[i].val = A[i].val * A[i]. val / norm_frob_sq;
//    }

    std::vector<cooEntry> const& const_pop = A;
    std::vector<cooEntry> sample;
    // Create the systematic sampling functor.
    is_picked predicate(sample_size, norm_frob_sq, &cooEntry::get_val_sq);
//    is_picked predicate(sample_size, 1.0, &cooEntry::get_val);

//    std::cout << "sample vector" << std::endl;
    for (sample_iterator2
                 sb = sample_iterator2(predicate, const_pop.begin(), const_pop.end()),
                 si = sb,
                 se = sample_iterator2(predicate, const_pop.end(),   const_pop.end());
         si != se; ++si){

//        std::cout << std::distance(sb, se) << "\t" << *si << std::endl;
        A_spars.emplace_back(*si);
    }

    return 0;
}


int saena_object::sparsify_drineas(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if(rank==0) std::cout << "\n" << __func__ << ":" << std::endl;

    // s = 28nln(sqrt(2)*n) / epsilon^2
//    nnz_t sample_size = nnz_t( (double)28 * Ac->Mbig * log(sqrt(2) * Ac->Mbig) * norm_frob_sq / (sparse_epsilon * sparse_epsilon) );
    if(rank==0) printf("sample size \t\t\t\t= %lu\n", sample_size);
//    if(rank==0) printf("norm_frob_sq = %f, \tsparse_epsilon = %f, \tAc->Mbig = %u \n", norm_frob_sq, sparse_epsilon, Ac->Mbig);

    std::uniform_real_distribution<double> dist(0.0,1.0); //(min, max). Type of random number distribution
    std::mt19937 rng; //Mersenne Twister: Good quality random number generator
    rng.seed(std::random_device{}()); //Initialize with non-deterministic seeds

    std::vector<cooEntry> Ac_sample(sample_size);
    double norm_temp = 0, criteria;
    for(nnz_t i = 0; i < A.size(); i++){
        norm_temp += A[i].val * A[i].val;

        criteria = (A[i].val * A[i].val) / norm_temp;
        for(nnz_t j = 0; j < sample_size; j++){
            if(dist(rng) < criteria){
//                std::cout << "dist(rng) = " << dist(rng) << "\tcriteria = " << criteria << "\tAc_sample[j] = " << Ac_sample[j] << std::endl;
//                Ac_sample[j] = cooEntry(A[i].row, A[i].col, A[i].val);
                Ac_sample[j] = A[i];
            }
        }
    }

//    if(rank==0) printf("Ac_sample.size() = %lu\n", Ac_sample.size());
//    print_vector(Ac_sample, -1, "Ac_sample", A->comm);
    std::sort(Ac_sample.begin(), Ac_sample.end());

    // remove duplicates and change the values based on Algorithm 1 of Drineas' paper.
    cooEntry temp;
    double factor = norm_frob_sq / sample_size;
    for(nnz_t i=0; i<Ac_sample.size(); i++){
        temp = Ac_sample[i];
        while(i<Ac_sample.size()-1 && Ac_sample[i] == Ac_sample[i+1]){ // values of entries with the same row and col should be added.
            temp.val += Ac_sample[i+1].val;
            i++;
        }
//        Ac->entry.emplace_back( cooEntry(Ac_sample[i].row, Ac_sample[i].col, factor / val_temp) );
        A_spars.emplace_back( temp );
    }

    if(rank==0) printf("Ac size after sparsification \t\t= %lu\n", A_spars.size());

    Ac_sample.clear();
    Ac_sample.shrink_to_fit();

    return 0;
}


int saena_object::sparsify_majid(std::vector<cooEntry>& A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Barrier(comm);
    if(rank==0){
        std::cout << "\n" << __func__ << ":" << std::endl;
        printf("original size = %lu, sample_size = %lu, norm_frob_sq = %f\n", A.size(), sample_size, norm_frob_sq);
    }
    MPI_Barrier(comm);
//    print_vector(A, -1, "A", comm);

    std::uniform_real_distribution<double> dist(0.0,1.0); //(min, max). Type of random number distribution
    std::mt19937 rng; //Mersenne Twister: Good quality random number generator
    rng.seed(std::random_device{}()); //Initialize with non-deterministic seeds

    std::vector<bool> chosen(A.size(), false);
    nnz_t iter, i = 0;

    for(iter = 0; iter < A.size(); iter++){
        if(A[iter].row == A[iter].col){
            A_spars.emplace_back(A[iter]);
            chosen[iter] = true;
            i++;
        }
    }

    index_t A_passes = 2; // one pass was done to add the diagonal entries.

    if(rank==0) printf("A_pass %u: selected: %lu (diagonal entries) \n", A_passes-1, i);

    double max_prob_inv = norm_frob_sq / max_val / max_val;
    double prob, rand, rand_factor = 10;
    iter = 0;
    while(i < sample_size){

        if( !chosen[iter] && A[iter].row < A[iter].col ){ // upper triangle

            rand = dist(rng) * rand_factor;
            prob = max_prob_inv * ( A[iter].val * ( A[iter].val / norm_frob_sq ));
//            if(rank==0 && !chosen[iter]) printf("prob = %.8f, \trand = %.8f, \tA.row = %u, \tA.col = %u \n",
//                                            prob, rand, A[iter].row, A[iter].col);
            if(rand < prob){
                A_spars.emplace_back(A[iter]);
                A_spars.emplace_back(A[iter].col, A[iter].row, A[iter].val);
                i += 2;
                chosen[iter] = true;
            }

        }

        iter++;
        if(iter >= A.size()){
            if(rank==0) printf("A_pass %u: selected: %lu \n", A_passes, i);
            iter -= A.size();
            A_passes++;
            rand_factor /= 10;
        }

    }

    std::sort(A_spars.begin(), A_spars.end());

    if(rank==0){
        printf("final: \nA_pass %u: selected: %lu\n           original: %lu\n           percent:  %f \n",
                A_passes, A_spars.size(), A.size(), sample_sz_percent);
    }
//    print_vector(A_spars, -1, "A_spars", comm);

    return 0;
}


int saena_object::sparsify_majid_with_dup(std::vector<cooEntry>& A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if(rank==0) std::cout << "\n" << __func__ << ":" << std::endl;
//    if(rank==0) printf("Ac size before sparsification = %lu\n", A.size());

    std::uniform_real_distribution<double> dist(0.0,1.0); //(min, max). Type of random number distribution
    std::mt19937 rng; //Mersenne Twister: Good quality random number generator
    rng.seed(std::random_device{}()); //Initialize with non-deterministic seeds

    double max_prob_inv = norm_frob_sq / max_val / max_val;
    index_t A_passes = 1;
    nnz_t iter = 0, i = 0;
    std::vector<cooEntry> A_spars_dup;
    while(i < sample_size){
        if( dist(rng) < max_prob_inv * ( A[iter].val * ( A[iter].val / norm_frob_sq )) ){
            A_spars_dup.emplace_back(A[iter]);
            i++;
        }

        iter++;
        if(iter >= A.size()){
            iter -= A.size();
            A_passes++;
        }
    }

//    if(rank==0) printf("A_spars_dup.size() = %lu\n", A_spars_dup.size());
//    print_vector(A_spars_dup, -1, "A_spars_dup", comm);
    std::sort(A_spars_dup.begin(), A_spars_dup.end());

    // remove duplicates and change the values based on Algorithm 1 of Drineas' paper.
    cooEntry temp;
    double factor = norm_frob_sq / sample_size;
    for(nnz_t i=0; i<A_spars_dup.size(); i++){
        temp = A_spars_dup[i];
        while(i<A_spars_dup.size()-1 && A_spars_dup[i] == A_spars_dup[i+1]){ // values of entries with the same row and col should be added.
            temp.val += A_spars_dup[i+1].val;
            i++;
        }
//        Ac->entry.emplace_back( cooEntry(A_spars_dup[i].row, A_spars_dup[i].col, factor / val_temp) );
        A_spars.emplace_back( temp );
    }

//    print_vector(A_spars, -1, "A_spars", comm);
    if(rank==0) printf("sparsified size = %lu\n", A_spars.size());
    if(rank==0) printf("A_passes = %u\n", A_passes);

    A_spars_dup.clear();
    A_spars_dup.shrink_to_fit();

    return 0;
}


double saena_object::spars_prob(cooEntry a){

    if(a.row == a.col){
        return 10000000;
    } else{
        return a.val * a.val;
    }

}
