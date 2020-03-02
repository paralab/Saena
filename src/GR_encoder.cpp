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

int rank_ver = 10;

void GR_encoder::put_bit(uchar *buf, uchar b){

    // TODO: comment out these lines.
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
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

//    if (filled == 8){
//        ++buf_iter;
//        filled = 0;
//    }
//    filled++;
}


// TODO: comment out this.
int wordcode_sz = 6;

index_t GR_encoder::get_bit(uchar *buf){

    // TODO: comment out these lines.
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
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
//        if(!--wordcode_sz){
//            printf("\n");
//            wordcode_sz = 6;
//        }
//    }

    return static_cast<index_t>(tmp);
}


int GR_encoder::compress(index_t *v, index_t v_sz, index_t k, uchar *buf){

    // TODO: comment out these lines.
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
//    int rank_ver = 0;

//    print_array(v, v_sz, rank_ver, "v compress", comm);

    unsigned int M = 1U << k;

    filled    = 7;
    buf_iter  = 0;
    int qiter = 0, diff;
    short q;

    auto r_sz = rem_sz(v_sz, k);
    auto qs   = reinterpret_cast<short*>(&buf[r_sz]);
//    if(rank==rank_ver) std::cout << "rank " << rank << ": compress:   k: " << k << ", M: " << M << ", r_sz: " << r_sz << ", v_sz: " << v_sz << std::endl;

    // ======================================
    // encode rows
    // ======================================
    // Since we want to encode the difference of the values (v[i] - v[i-1]), we first perform the encoding on the
    // first element here.

    diff = v[0];
    put_bit(buf, 0); // the first entry is a positive number, not a difference, so it cannot be negative.

//    if(diff < 0){
//        diff = -diff;
//        put_bit(buf, 1);
//    }else{
//        put_bit(buf, 0);
//    }

    q = diff / M;
//    if(rank==rank_ver) std::cout << "\nv[0]: " << diff << "\nq = " << q << "\t(iter: " << iter << "), r = " << diff - q*M << std::endl;

    if(q){
        qs[qiter++] = q;
        put_bit(buf, 1);
    }else{
        put_bit(buf, 0);
    }

//    if(rank==rank_ver) std::cout << "qs[0]: " << qs[0] << std::endl;

    for(int j = k-1; j >= 0; --j) {
//        std::cout << ((diff >> j) & 1);
        put_bit(buf, (diff >> j) & 1);
    }
//    cout << endl;

//    if(rank==rank_ver){
//        std::cout << 0 << "\tdiff: " << diff << ", v[i]: " << v[0] << ", q: " << q << std::endl;
//    }

    // ======================================
    // perform the encoding on the rest of v
    // ======================================

    for(int i = 1; i < v_sz; ++i){
        assert(buf_iter <= r_sz);

        diff = v[i] - v[i-1];

//        if(rank==rank_ver){
//            std::cout << i << "\t" << v[i] << "\t" << diff << std::endl;
//        }

        if(diff < 0){
            diff = -diff;
            put_bit(buf, 1);
        }else{
            put_bit(buf, 0);
        }

        q = diff / M;

//        if(rank==rank_ver){
//            std::cout << i << "\tdiff: " << diff << ", v[i]: " << v[i] << ", v[i-1]: " << v[i-1] << ", q: " << q << std::endl;
//        }

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
    }

//    print_array(qs, qiter, 0, "qs before", MPI_COMM_WORLD);
//    for(int i = 0; i < qiter; ++i){
//        qs[i] = static_cast<short>(i);
//        if(rank==0) std::cout << i << "\t" << qs[i] << std::endl;
//    }

    return 0;
}


int GR_encoder::decompress(index_t *v, index_t v_sz, index_t k, int q_sz, uchar *buf) {

    // TODO: comment out these lines.
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
//    int rank_ver = 0;

    unsigned int M = 1U << k;

    auto r_sz = rem_sz(v_sz, k);
    auto qs   = reinterpret_cast<short*>(&buf[r_sz]);
//    if(rank==rank_ver) std::cout << "\nrank " << rank << ": decompress: k: " << k << ", M: " << M << ", r_sz: " << r_sz << ", q_sz: " << q_sz << std::endl;

//    print_array(qs, q_sz, 0, "qs after", MPI_COMM_WORLD);

    int qiter = 0, viter = 0;
    index_t x;
    short q;

    // decode the first element
    filled   = 7;
    buf_iter = 0;
//    iter = 0;
    bool neg = false; // flag for negavtive

    // if(get_bit == 1) -> negative
    if(get_bit(buf)) {
        printf("rank %d: decompress: error: the first entry is a row index not difference, so it cannot be negative.\n", rank);
        exit(EXIT_FAILURE);
    }

    q = 0;
    if(get_bit(buf)) {
//        if(rank==rank_ver) std::cout << "q nonzero" << std::endl;
        q = qs[qiter++];
    }

    x = q * M;
//    if(rank==rank_ver) std::cout << "init x = " << x << std::endl;

//    for (index_t j = 0; j < k; ++j) {
    for (int j = k-1; j >= 0; --j) {
//        std::cout << "\niter: " << iter << "\t" << std::bitset<1>(buf[iter]) << "\t" << (buf[iter] << j) << std::endl;
        x = x | (get_bit(buf) << j);
//        if(rank==rank_ver) std::cout << "x = " << x << std::endl;
    }
    v[viter++] = x;
//    if(rank==rank_ver) std::cout << viter << ": v[viter] = " << v[viter-1] << ", diff = " << x << " (buf_iter: " << buf_iter << ")\n";

    // decode the rest
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

        x = q * M;
//        if(rank==rank_ver) std::cout << "q: " << q << ", x = q * M = " << x << std::endl;

        for (int j = k-1; j >= 0; --j) {
//            std::cout << "iter: " << iter << "\t" << std::bitset<1>(buf[iter]) << std::endl;
            x = x | (get_bit(buf) << j);
//            if(rank==rank_ver) std::cout << "x = " << x << std::endl;
        }

        assert(x != INT32_MAX);

        if(neg){
            v[viter] = v[viter-1] - x;
        }else{
            v[viter] = v[viter-1] + x;
        }

//        if(rank==rank_ver) std::cout << viter << ": v[viter] = " << v[viter] << ", diff = " << x << " (buf_iter: " << buf_iter << ")" << std::endl;
        ++viter;
    }

    ASSERT(viter == v_sz, "rank " << rank << ": viter: " << viter << ", v_sz: " << v_sz);

//    print_array(v, v_sz, rank_ver, "v decompressed", comm);

    return 0;
}