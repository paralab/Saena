#include <cmath>
#include <aux_functions.h>
#include "data_struct.h"


std::ostream & operator<<(std::ostream & stream, const cooEntry & item) {
    stream << item.row << "\t" << item.col << "\t" << item.val;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const cooEntry_row & item) {
    stream << item.row << "\t" << item.col << "\t" << item.val;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const vecEntry & item) {
    stream << item.row << "\t" << item.val;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const tuple1 & item) {
    stream << item.idx1 << "\t" << item.idx2;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const vecCol & item) {
    stream << item.rv->row << "\t" << *item.c << "\t" << item.rv->val;
    return stream;
}


bool row_major (const cooEntry& node1, const cooEntry& node2)
{
    if(node1.row < node2.row)
        return (true);
    else if(node1.row == node2.row)
        return(node1.col <= node2.col);
    else
        return false;
}


bool vecCol_col_major (const vecCol& node1, const vecCol& node2)
{
    if(*node1.c < *node2.c)
        return (true);
    else if(*node1.c == *node2.c)
        return((*node1.rv).row <= (*node2.rv).row);
    else
        return false;
}


int CSCMat::compress_prep(){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int rank_ver = 0;

    compress_prep_compute(row, nnz, comp_row);
    compress_prep_compute(col_scan, col_sz+1, comp_col);

//    print_vector(comp_row.ks, rank_ver, "ks", comm);
//    print_vector(comp_row.qs, rank_ver, "qs", comm);

    // compute max compress buffer size on all procs: q_size * sizeof(short) + #compressed values * (k+2)
    // do this for both rows and col_scan, then add them together
    comp_row.max_tot = 0;
    comp_col.max_tot = 0;
    nnz_t proc_col_sz, row_buf_sz, col_buf_sz;

    max_comp_sz = 0;
    for(int i = 0; i < nprocs; ++i){
//        row_buf_sz = comp_row.qs[i] * sizeof(short) + (nnz_list[i] * (comp_row.ks[i] + 2));
        row_buf_sz = tot_sz(nnz_list[i], comp_row.ks[i], comp_row.qs[i]);

        proc_col_sz = split[i + 1] - split[i];
//        col_buf_sz = comp_col.qs[i] * sizeof(short) + (proc_col_sz + 1) * (comp_col.ks[i] + 2);
        col_buf_sz = tot_sz(proc_col_sz + 1, comp_col.ks[i], comp_col.qs[i]);

        if(row_buf_sz + col_buf_sz > max_comp_sz){
            comp_row.max_tot = row_buf_sz;
            comp_col.max_tot = col_buf_sz;
            max_comp_sz = row_buf_sz + col_buf_sz;
        }
    }

    return 0;
}


int CSCMat::compress_prep_compute(index_t *v, index_t v_sz, GR_sz &comp_sz){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int k_start = 4, k_end = 14;

    int q_sz = 0, r_sz;
//    unsigned int M;

    comp_sz.tot = INT32_MAX;

//    int rank_ver = 0;
    for(int k = k_start; k < k_end; k+=2) {

//        M = 1U << k;
        r_sz = rem_sz(v_sz, k);

        // first element
        q_sz = 0;
        if(v[0] >> k != 0){
            ++q_sz;
        }

        // the rest of v
        for (int i = 1; i < v_sz; ++i) {
//            int q = abs(v[i] - v[i - 1]) >> k;
//            int q = diff >> k;
            if(abs(v[i] - v[i - 1]) >> k != 0){
//                if(rank==rank_ver) std::cout << "v[i] - v[i - 1]: " << (float)v[i] - v[i - 1] << ", M: " << M << ", q: " << q << std::endl;
                ++q_sz;
            }
        }

#ifdef __DEBUG1__
        {
//            if(rank==rank_ver) std::cout << "k: " << k << ", M: " << M << std::endl;
//            if(rank==rank_ver) std::cout << "r_sz: " << r_sz << std::endl;
//            if(rank==rank_ver) std::cout << "q_sz: " << q_sz << std::endl;
        }
#endif

        if(r_sz + q_sz < comp_sz.tot){
            comp_sz.k   = k;
            comp_sz.r   = r_sz; // in bytes
            comp_sz.q   = q_sz; // short or int
//            comp_sz.tot = r_sz + q_sz * sizeof(short) ; // in bytes
            comp_sz.tot = tot_sz(v_sz, k, q_sz) ; // in bytes
        }

    }

    comp_sz.ks.resize(nprocs);
    comp_sz.qs.resize(nprocs);

    // TODO: check if MPI_SHORT works as the datatype for the following commands.
    MPI_Allgather(&comp_sz.k, 1, MPI_INT, &comp_sz.ks[0], 1, MPI_INT, comm);
    MPI_Allgather(&comp_sz.q, 1, MPI_INT, &comp_sz.qs[0], 1, MPI_INT, comm);

#ifdef __DEBUG1__
    {
//        MPI_Barrier(comm);
//        printf("rank %d: comp_sz.q = %d \n", rank, comp_sz.q);
//        MPI_Barrier(comm);

//        if(rank==rank_ver) std::cout << "comp_sz.k: " << comp_sz.k << ", comp_sz.r: " << comp_sz.r
//                    << ", comp_sz.q: " << comp_sz.q << ", comp_sz.tot: " << comp_sz.tot << std::endl;
//        if(rank==rank_ver) std::cout  << std::endl;
//        print_vector(comp_sz.ks, rank_ver, "ks", comm);
//        print_vector(comp_sz.qs, rank_ver, "qs", comm);
    }
#endif

    return 0;
}