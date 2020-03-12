#include <cmath>
#include <aux_functions.h>
#include <iomanip>
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

    unsigned long orig_sz = (nnz + col_sz+1) * sizeof(index_t);
    unsigned long comp_sz = comp_row.tot + comp_col.tot;
    float comp_rate_loc = 1.0f - (static_cast<float>(comp_sz) / orig_sz);
    float comp_rate;

    MPI_Reduce(&comp_rate_loc, &comp_rate, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    if(!rank) std::cout << "GR:  orig sz (rank0) = " << orig_sz << ", comp sz (rank0) = " << comp_sz
                        << ", saving %" << std::setprecision(2) << 100 * comp_rate / nprocs << " (average)" << std::endl;

#ifdef __DEBUG1__
    if(rank==rank_ver && verbose_prep){
        std::cout << "row:\nk: " << std::setw(2) << comp_row.k   << ", M: " << std::setw(5) << (1U << comp_row.k)
                  << ", v_sz: "  << std::setw(5) << nnz          << " ("    << std::setw(5) << nnz*sizeof(index_t) << ")"
                  << ", r_sz: "  << std::setw(5) << comp_row.r
                  << ", q_sz: "  << std::setw(5) << comp_row.q   << " ("    << std::setw(5) << comp_row.q*sizeof(short) << ")"
                  << ", tot: "   << std::setw(5) << comp_row.tot << "\n";

        std::cout << "cols:\nk: " << std::setw(2) << comp_col.k   << ", M: " << std::setw(5) << (1U << comp_col.k)
                  << ", v_sz: "   << std::setw(5) << col_sz+1     << " ("    << std::setw(5) << nnz*sizeof(index_t) << ")"
                  << ", r_sz: "   << std::setw(5) << comp_col.r
                  << ", q_sz: "   << std::setw(5) << comp_col.q   << " ("    << std::setw(5) << comp_col.q*sizeof(short) << ")"
                  << ", tot: "    << std::setw(5) << comp_col.tot << "\n";

//        unsigned long orig_sz = (nnz + col_sz+1) * sizeof(index_t);
//        unsigned long comp_sz = comp_row.tot + comp_col.tot;
//        std::cout << "orig_sz = " << orig_sz << ", comp_sz = " << comp_sz << ", compression rate:" << 1 - (comp_sz / static_cast<double>(orig_sz)) << std::endl;
    }

//    print_vector(comp_row.ks, rank_ver, "ks", comm);
//    print_vector(comp_row.qs, rank_ver, "qs", comm);
#endif

    // compute max compress buffer size on all procs.
    // do this for both rows and col_scan, then add them together.
    nnz_t proc_col_sz, row_buf_sz, col_buf_sz;
    comp_row.max_tot = 0;
    comp_col.max_tot = 0;
    max_comp_sz      = 0;

    for(int i = 0; i < nprocs; ++i){
        row_buf_sz = tot_sz(nnz_list[i], comp_row.ks[i], comp_row.qs[i]);
        proc_col_sz = split[i + 1] - split[i];
        col_buf_sz = tot_sz(proc_col_sz + 1, comp_col.ks[i], comp_col.qs[i]);

        if(row_buf_sz + col_buf_sz > max_comp_sz){
            comp_row.max_tot = row_buf_sz;
            comp_col.max_tot = col_buf_sz;
            max_comp_sz      = row_buf_sz + col_buf_sz;
        }
    }

#ifdef __DEBUG1__
//    if(rank==rank_ver) printf("comp_row.max_tot = %lu, comp_col.max_tot = %lu, max_comp_sz = %lu, \n",
//                               comp_row.max_tot, comp_col.max_tot, max_comp_sz);
#endif

    return 0;
}


int CSCMat::compress_prep_compute(const index_t *v, index_t v_sz, GR_sz &comp_sz){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    int rank_ver = 0;
//    unsigned int M;
#endif

    int k_start = 7, k_end = 8;
    int q_sz = 0, r_sz, tot;
    comp_sz.tot = INT32_MAX;

    for(unsigned int k = k_start; k < k_end; k++) {

//        M = 1U << k;
        r_sz = rem_sz(v_sz, k);

        // first element
        q_sz = 0;
        if( (v[0] >> k) != 0){
            ++q_sz;
        }

        // the rest of v
        for (int i = 1; i < v_sz; ++i) {
//            int diff = static_cast<int>(v[i] - v[i - 1]);
//            int q = diff >> k;
//            if(rank==rank_ver) std::cout << "v[i]: " << v[i] << ", v[i-1]: " << v[i-1] << ", diff: " << diff << ", M: " << (1U << k) << ", q: " << q << std::endl;

            if( (static_cast<int>(v[i] - v[i - 1]) >> k) != 0){
//            if(q != 0){
//                if(rank==rank_ver) std::cout << "v[i]: " << v[i] << ", v[i-1]: " << v[i-1] << ", diff: " << (int)(v[i] - v[i - 1]) << ", M: " << (1U << k) << ", q: " << q << std::endl;
                ++q_sz;
            }

#ifdef __DEBUG1__
            if(static_cast<int>(v[i] - v[i - 1]) > INT16_MAX){
                printf("Datatype short is enough to store the difference for compression!\n");
                exit(EXIT_FAILURE);
            }
#endif

        }

        tot = tot_sz(v_sz, k, q_sz);
        if(tot < comp_sz.tot){
            comp_sz.k   = k;
            comp_sz.r   = r_sz; // in bytes
            comp_sz.q   = q_sz; // number of short numbers
            comp_sz.tot = tot;  // in bytes
        }

#ifdef __DEBUG1__
        if(verbose_prep_compute && rank==rank_ver){
            std::cout << "k: "      << std::setw(2) << k    << ", M: " << std::setw(5) << (1U << k)
                      << ", v_sz: " << std::setw(5) << v_sz << " ("    << std::setw(5) << v_sz*sizeof(index_t) << ")"
                      << ", r_sz: " << std::setw(5) << r_sz
                      << ", q_sz: " << std::setw(5) << q_sz << " ("    << std::setw(5) << q_sz*sizeof(short) << ")"
                      << ", tot: "  << std::setw(5) << tot << std::endl;
        }
#endif

    }

#ifdef __DEBUG1__
    if(verbose_prep_compute && rank==rank_ver){
        std::cout << "k: "      << std::setw(2) << comp_sz.k    << ", M: " << std::setw(5) << (1U << comp_sz.k)
                  << ", v_sz: " << std::setw(5) << v_sz << " (" << std::setw(5) << v_sz*sizeof(index_t) << ")"
                  << ", r_sz: " << std::setw(5) << comp_sz.r
                  << ", q_sz: " << std::setw(5) << comp_sz.q    << " (" << std::setw(5) << q_sz*sizeof(short) << ")"
                  << ", tot: "  << std::setw(5) << comp_sz.tot  << "\n";
    }
#endif

    comp_sz.ks.resize(nprocs);
    comp_sz.qs.resize(nprocs);

    fill(comp_sz.ks.begin(), comp_sz.ks.end(), comp_sz.k);

    // TODO: combine these together.
    // TODO: check if MPI_SHORT works as the datatype for the following commands.
//    MPI_Allgather(&comp_sz.k, 1, MPI_INT, &comp_sz.ks[0], 1, MPI_INT, comm);
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

int CSCMat::compress_prep_compute2(const index_t *v, index_t v_sz, GR_sz &comp_sz){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    int rank_ver = 0;
//    unsigned int M;
#endif

    int k_start = 4, k_end = 8;
    int q_sz = 0, r_sz, tot;
    comp_sz.tot = INT32_MAX;

    for(unsigned int k = k_start; k < k_end; k++) {

//        M = 1U << k;
        r_sz = rem_sz(v_sz, k);

        // first element
        q_sz = 0;
        if( (v[0] >> k) != 0){
            ++q_sz;
        }

        // the rest of v
        for (int i = 1; i < v_sz; ++i) {
//            int diff = static_cast<int>(v[i] - v[i - 1]);
//            int q = diff >> k;
//            if(rank==rank_ver) std::cout << "v[i]: " << v[i] << ", v[i-1]: " << v[i-1] << ", diff: " << diff << ", M: " << (1U << k) << ", q: " << q << std::endl;

            if( (static_cast<int>(v[i] - v[i - 1]) >> k) != 0){
//            if(q != 0){
//                if(rank==rank_ver) std::cout << "v[i]: " << v[i] << ", v[i-1]: " << v[i-1] << ", diff: " << (int)(v[i] - v[i - 1]) << ", M: " << (1U << k) << ", q: " << q << std::endl;
                ++q_sz;
            }

#ifdef __DEBUG1__
            if(static_cast<int>(v[i] - v[i - 1]) > INT16_MAX){
                printf("Datatype short is enough to store the difference for compression!\n");
                exit(EXIT_FAILURE);
            }
#endif

        }

        tot = tot_sz(v_sz, k, q_sz);
        if(tot < comp_sz.tot){
            comp_sz.k   = k;
            comp_sz.r   = r_sz; // in bytes
            comp_sz.q   = q_sz; // number of short numbers
            comp_sz.tot = tot;  // in bytes
        }

#ifdef __DEBUG1__
        if(verbose_prep_compute && rank==rank_ver){
            std::cout << "k: "      << std::setw(2) << k    << ", M: " << std::setw(5) << (1U << k)
                      << ", v_sz: " << std::setw(5) << v_sz << " ("    << std::setw(5) << v_sz*sizeof(index_t) << ")"
                      << ", r_sz: " << std::setw(5) << r_sz
                      << ", q_sz: " << std::setw(5) << q_sz << " ("    << std::setw(5) << q_sz*sizeof(short) << ")"
                      << ", tot: "  << std::setw(5) << tot << std::endl;
        }
#endif

    }

#ifdef __DEBUG1__
    if(verbose_prep_compute && rank==rank_ver){
        std::cout << "k: "      << std::setw(2) << comp_sz.k    << ", M: " << std::setw(5) << (1U << comp_sz.k)
                  << ", v_sz: " << std::setw(5) << v_sz << " (" << std::setw(5) << v_sz*sizeof(index_t) << ")"
                  << ", r_sz: " << std::setw(5) << comp_sz.r
                  << ", q_sz: " << std::setw(5) << comp_sz.q    << " (" << std::setw(5) << q_sz*sizeof(short) << ")"
                  << ", tot: "  << std::setw(5) << comp_sz.tot  << "\n";
    }
#endif

    comp_sz.ks.resize(nprocs);
    comp_sz.qs.resize(nprocs);

    // TODO: combine these together.
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