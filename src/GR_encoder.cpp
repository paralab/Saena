#include <aux_functions.h>
#include <bitset>
#include <cassert>
#include <iomanip>
#include "GR_encoder.h"

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

using namespace std;

//int rank_ver = 10;

inline void GR_encoder::put_bit(uchar *buf, uchar b){

//    MPI_Comm comm = MPI_COMM_WORLD;
//    int rank, nprocs;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);
//    int rank_ver = 1;

    buf[buf_iter] = buf[buf_iter] | ((b & 1) << filled);

//    if(rank == rank_ver) {
//        cout << "bit: " << static_cast<unsigned>(b) << ", bitset(buf): " << std::bitset<8>(buf[buf_iter])
//             << ", filled: " << filled << ", buf_iter: " << buf_iter << endl;
//    }

    if (filled == 0){
        ++buf_iter;
        filled = 8;
    }
    --filled;

}


inline index_t GR_encoder::get_bit(uchar *buf){

//    MPI_Comm comm = MPI_COMM_WORLD;
//    int rank, nprocs;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);
//    int rank_ver = 0;

    uchar tmp = (buf[buf_iter] >> filled) & 1U;

//    if(rank == rank_ver) {
//        cout << "bit: " << static_cast<int>(tmp) <<  ", bitset(tmp): " << std::bitset<8>(tmp) << ", bitset(buf): "
//           << std::bitset<8>(buf[buf_iter]) << ", filled: " << filled << ", bit op: " << static_cast<index_t>(buf[buf_iter] >> filled) << endl;
//    }

    if (filled == 0){
        ++buf_iter;
        filled = 8;
    }
    --filled;

//    if(rank == rank_ver){
//        printf("%d", static_cast<int>(tmp));
//    }

    return static_cast<index_t>(tmp);
}


int GR_encoder::compress(index_t *v, index_t v_sz, index_t k, uchar *buf){

#ifdef __DEBUG1__
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int rank_ver = 0;

//    print_array(v, v_sz, rank_ver, "v compress", comm);
//    unsigned int M = 1U << k;
#endif

    filled    = 7;
    buf_iter  = 0;
    int qiter = 0, diff;
    short q;

    auto r_sz = rem_sz(v_sz, k);
    auto qs   = reinterpret_cast<short*>(&buf[r_sz]);

#ifdef __DEBUG1__
    if(verbose_comp && rank==rank_ver){
        std::cout << "\nrank " << rank << ": compress:   k: " << k << ", M: " << (1U << k) << ", r_sz: " << r_sz << ", v_sz: " << v_sz << std::endl;
    }
#endif

    // ======================================
    // encode rows
    // ======================================

    // 1- encode v[0]
    // ======================================
    // Since we want to encode the difference of the values (v[i] - v[i-1]), we first perform the encoding on the
    // first element here.

    diff = v[0];
    put_bit(buf, 0); // the first entry is a positive number, not a difference, so it cannot be negative.

    q = diff >> k;

    if(q){
        qs[qiter++] = q;
        put_bit(buf, 1);
    }else{
        put_bit(buf, 0);
    }

    for(int j = k-1; j >= 0; --j) {
//        std::cout << ((diff >> j) & 1);
        put_bit(buf, (diff >> j) & 1);
    }
//    cout << endl;

#ifdef __DEBUG1__
    if(verbose_comp && rank==rank_ver){
        std::cout << 0 << "\tdiff: " << diff << ", v[i]: " << v[0] << ", q: " << q << std::endl;
    }
#endif

    // 2- encode the rest of v
    // ======================================

    for(int i = 1; i < v_sz; ++i){
        assert(buf_iter <= r_sz);

        diff = v[i] - v[i-1];

        if(diff < 0){
            diff = -diff;
            put_bit(buf, 1);
        }else{
            put_bit(buf, 0);
        }

        q = diff >> k;

        if(q){
            qs[qiter++] = q; // q can be negative (if v[i] - v[i-1] is negative).
            put_bit(buf, 1);
        }else{
            put_bit(buf, 0);
        }

        for(int j = k-1; j >= 0; --j) {
//            if(rank==rank_ver) std::cout << ((diff >> j) & 1);
            put_bit(buf, (diff >> j) & 1);
        }
//        if(rank==rank_ver) std::cout << std::endl;

#ifdef __DEBUG1__
        if(verbose_comp && rank==rank_ver){
//            std::cout << i << "\t" << v[i] << "\t" << diff << std::endl;
            std::cout << i << "\tdiff: " << diff << ", v[i]: " << v[i] << ", v[i-1]: " << v[i-1] << ", q: " << q << std::endl;
        }
#endif
    }

#ifdef __DEBUG1__
    {
//        print_array(qs, qiter, 0, "qs before", MPI_COMM_WORLD);
//        for(int i = 0; i < qiter; ++i){
//            qs[i] = static_cast<short>(i);
//            if(rank==0) std::cout << i << "\t" << qs[i] << std::endl;
//        }
    }
#endif

    return 0;
}


int GR_encoder::decompress(index_t *v, index_t v_sz, index_t k, int q_sz, uchar *buf) {

#ifdef __DEBUG1__
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int rank_ver = 0;

//    unsigned int M = 1U << k;
#endif

    auto r_sz = rem_sz(v_sz, k);
    auto qs   = reinterpret_cast<short*>(&buf[r_sz]);

#ifdef __DEBUG1__
    if(verbose_decomp && rank==rank_ver){
//        std::cout << "\nrank " << rank << ": decompress: k: " << k << ", M: " << (1U << k) << ", r_sz: " << r_sz << ", q_sz: " << q_sz << std::endl;
//        print_array(qs, q_sz, 0, "qs after", comm);
    }
#endif

    index_t qiter = 0, viter = 0, x;
    short q;

    // 1- decode v[0]
    // ======================================

    filled   = 7;
    buf_iter = 0;
    bool neg = false; // flag for negavtive

    // get the sign bit
    // if(get_bit == 1) -> negative
    if(get_bit(buf)) {
        printf("decompress: error: the first entry is a row index not difference, so it cannot be negative.\n");
//        printf("rank %d: decompress: error: the first entry is a row index not difference, so it cannot be negative.\n", rank);
        exit(EXIT_FAILURE);
    }

    q = 0;
    if(get_bit(buf)){
        q = qs[qiter++];
    }

    x = q << k;

//    for (index_t j = 0; j < k; ++j) {
    for (int j = k-1; j >= 0; --j) {
//        std::cout << "\niter: " << iter << "\t" << std::bitset<1>(buf[iter]) << "\t" << (buf[iter] << j) << std::endl;
        x = x | (get_bit(buf) << j);
//        if(rank==rank_ver) std::cout << "x = " << x << std::endl;
    }
    v[viter++] = x;

#ifdef __DEBUG1__
    if(verbose_decomp && rank==rank_ver){
//        std::cout << viter << ": v[viter] = " << v[viter-1] << ", diff = " << x << ", q = " << q << " (buf_iter: " << buf_iter << ")\n";
//        print_array(qs, q_sz, 0, "qs after", comm);
    }
#endif

    // 2- decode the rest of v
    // ======================================

    while(viter < v_sz){
//    while(buf_iter < r_sz){

        assert(viter <= v_sz);
//        if(!rank) ASSERT(filled%(k+2) == 0, "filled: " << filled << ", k+2: " << k+2);
//        if(!rank) cout << "filled: " << filled << ", k+2: " << k+2 << endl;

        neg = false;
        if(get_bit(buf)) {
            neg = true;
        }

        q = 0;
        if(get_bit(buf)){
            q = qs[qiter++];
        }

        x = q << k;

        for (int j = k-1; j >= 0; --j) {
//            std::cout << "iter: " << iter << "\t" << std::bitset<1>(buf[iter]) << std::endl;
            x = x | (get_bit(buf) << j);
//            if(rank==rank_ver) std::cout << "x = " << x << std::endl;
        }

        if(neg){
            v[viter] = v[viter-1] - x;
        }else{
            v[viter] = v[viter-1] + x;
        }

        ++viter;

#ifdef __DEBUG1__
        assert(x != INT32_MAX);
        if(verbose_decomp && rank==rank_ver){
//            std::cout << viter << ": v[viter] = " << v[viter-1] << ", diff = " << x << ", q = " << q << " (buf_iter: " << buf_iter << ")\n";
        }
#endif
    }

//    ASSERT(viter == v_sz, "rank " << rank << ": viter: " << viter << ", v_sz: " << v_sz);
//    print_array(v, v_sz, rank_ver, "v decompressed", comm);

    return 0;
}