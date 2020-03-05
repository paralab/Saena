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

//int rank_ver = 0;

// the version with debug commands
// inline void GR_encoder::put_bit(uint8_t *buf, uint8_t b)
/*
inline void GR_encoder::put_bit(uint8_t *buf, uint8_t b){

//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    int rank_ver = 1;

    buf[buf_iter] = buf[buf_iter] | ((b & 1u) << filled);

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
*/

// the version with debug commands
// inline index_t GR_encoder::get_bit(const uint8_t *buf)
/*
inline index_t GR_encoder::get_bit(const uint8_t *buf){

//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint8_t tmp = (buf[buf_iter] >> filled) & 1u;

//    if(rank == rank_ver) {
//        cout << "bit: " << static_cast<int>(tmp) <<  ", bitset(tmp): " << std::bitset<8>(tmp) << ", bitset(buf): "
//           << std::bitset<8>(buf[buf_iter]) << ", filled: " << filled << ", bit op: " << static_cast<index_t>(buf[buf_iter] >> filled) << endl;
//    }

    if (filled == 0){
        ++buf_iter;
        filled = 8;
    }
    --filled;

    return static_cast<index_t>(tmp);
}
*/


inline void GR_encoder::put_bit(uint8_t *buf, uint8_t b){
    buf[buf_iter] = buf[buf_iter] | ((b & 1u) << filled);
    if (filled == 0){
        ++buf_iter;
        filled = 8;
    }
    --filled;
}


inline index_t GR_encoder::get_bit(const uint8_t *buf){
    uint8_t tmp = (buf[buf_iter] >> filled) & 1u;
    if (filled == 0){
        ++buf_iter;
        filled = 8;
    }
    --filled;
    return static_cast<index_t>(tmp);
}


void GR_encoder::compress(index_t *v, index_t v_sz, index_t k, uchar *buf){

#ifdef __DEBUG1__
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int rank_ver = 0;

//    print_array(v, v_sz, rank_ver, "v compress", comm);
//    unsigned int M = 1U << k;
#endif

    if(v_sz == 0){
        return;
    }

    int i, j;
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

    assert(v[0] >= 0);

    diff = v[0];
//    put_bit(buf, 0); // the first entry is a positive number, not a difference, so it cannot be negative.

//    q = diff / M;
    q = diff >> k;

    if(q != 0){
        qs[qiter++] = q;
        put_bit(buf, 1);
    }else{
        put_bit(buf, 0);
    }

    for(j = k-1; j >= 0; --j) {
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

    for(i = 1; i < v_sz; ++i){

        diff = static_cast<int>(v[i] - v[i-1]);

//        q = diff / M;
        q = diff >> k;

        if(q != 0){
            qs[qiter++] = q; // q can be negative (if v[i] - v[i-1] is negative).
            put_bit(buf, 1);
        }else{
            put_bit(buf, 0);
        }

        for(j = k-1; j >= 0; --j) {
//            if(rank==rank_ver) std::cout << ((diff >> j) & 1);
            put_bit(buf, (diff >> j) & 1);
        }
//        if(rank==rank_ver) std::cout << std::endl;

#ifdef __DEBUG1__
        assert(buf_iter <= r_sz);
        if(verbose_comp && rank==rank_ver){
//            std::cout << i << "\t" << v[i] << "\t" << diff << std::endl;
//            std::cout << i << "\tdiff: " << diff << ", v[i]: " << v[i] << ", v[i-1]: " << v[i-1] << ", q: " << q << "\n";
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
}


void GR_encoder::decompress(index_t *v, index_t v_sz, index_t k, int q_sz, uint8_t *buf) {

#ifdef __DEBUG1__
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int rank_ver = 1;

//    unsigned int M = 1U << k;
#endif

    auto r_sz = rem_sz(v_sz, k);
    auto qs   = reinterpret_cast<short*>(&buf[r_sz]);

#ifdef __DEBUG1__
//    print_array(qs, q_sz, 0, "qs after", comm);
    if(verbose_decomp && rank==rank_ver){
        std::cout << "\nrank " << rank << ": decompress: k: " << k << ", M: " << (1U << k) << ", r_sz: " << r_sz << ", q_sz: " << q_sz << std::endl;
    }
#endif

    int x;
    short q;
    index_t qiter = 0, viter = 0;
    index_t k_1s = (1u << k) - 1;

    // 1- decode v[0]
    // ======================================

    buf_iter = 0;

    q = 0;
    if(buf[buf_iter] >> k){
        q = qs[qiter++];
    }

    x = (q << k) | (buf[buf_iter] & k_1s);
    ++buf_iter;

#if 0
    q = 0;
    if(get_bit(buf)){
        q = qs[qiter++];
    }

//    x = q * M;
    x = q << k;

    for (j = k-1; j >= 0; --j) {
//        std::cout << "\niter: " << iter << "\t" << std::bitset<1>(buf[iter]) << "\t" << (buf[iter] << j) << std::endl;
        x = x | (get_bit(buf) << j);
//        if(rank==rank_ver) std::cout << "x = " << x << std::endl;
    }
#endif

    v[viter++] = x;

#ifdef __DEBUG1__
    if(verbose_decomp && rank==rank_ver){
        std::cout << 0 << ": v[viter] = " << v[viter-1] << ", diff = " << x << ", q = " << q << " (buf_iter: " << buf_iter << ")\n";
//        print_array(qs, q_sz, 0, "qs after", comm);
    }
#endif

    // 2- decode the rest of v
    // ======================================

    while(viter < v_sz){

        q = 0;
        if(buf[buf_iter] >> k){
            q = qs[qiter++];
        }

        x = (q << k) | (buf[buf_iter] & k_1s);
        ++buf_iter;
        v[viter] = v[viter-1] + x;
        ++viter;

#if 0
        q = 0;
        if(get_bit(buf)){
            q = qs[qiter++];
        }

//        x = q * M;
        x = q << k;

        for (j = k-1; j >= 0; --j) {
//            std::cout << "iter: " << iter << "\t" << std::bitset<1>(buf[iter]) << std::endl;
            x = x | (get_bit(buf) << j);
//            if(rank==rank_ver) std::cout << "x = " << x << std::endl;
        }
#endif

#ifdef __DEBUG1__
//        cout << "buf[buf_iter]: " << std::bitset<8>(buf[buf_iter]) << ", buf[buf_iter+1]: " << std::bitset<8>(buf[buf_iter+1])
//                << ", tmp2: " << std::bitset<16>(tmp2) << ", tmp: " << std::bitset<16>(tmp)
//                << ", (tmp & k_1s): " << (tmp & k_1s) << ", filled: " << filled << ", ofst: " << ofst
//                << ", tmp >> k: " << (tmp >> k) << endl;

        assert(buf_iter < buf_sz);
        assert(x != INT32_MAX);
        if(verbose_decomp && rank==rank_ver){
            std::cout << viter-1 << ": v[viter] = " << v[viter-1] << ", diff = " << x << ", q = " << q << " (buf_iter: " << buf_iter << ")\n";
        }
#endif
    }

#ifdef __DEBUG1__
//    ASSERT(viter == v_sz, "rank " << rank << ": viter: " << viter << ", v_sz: " << v_sz);
//    print_array(v, v_sz, rank_ver, "v decompressed", comm);
#endif
}


void GR_encoder::decompress2(index_t *v, index_t v_sz, index_t k, int q_sz, uint8_t *buf) {

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
//    print_array(qs, q_sz, 0, "qs after", comm);
    if(verbose_decomp && rank==rank_ver){
        std::cout << "\nrank " << rank << ": decompress: k: " << k << ", M: " << (1U << k) << ", r_sz: " << r_sz << ", q_sz: " << q_sz << std::endl;
    }
#endif

    index_t k_1s   = (1u << k) - 1;
    index_t kp1    = k + 1;
    index_t kp1_1s = (1u << kp1) - 1;
    index_t ofst   = 16 - kp1;

    int x;
    short q;
    index_t qiter = 0, viter = 0;

    // 1- decode v[0]
    // ======================================

    filled   = 0;
    buf_iter = 0;

    uint16_t tmp2 = (buf[buf_iter] << 8u) | (buf[buf_iter+1]);
    uchar    tmp  = tmp2 >> (ofst - filled);

//    cout << "tmp: " << std::bitset<16>(tmp) << ", (tmp & k_1s): " << (tmp & k_1s) << endl;

    q = 0;
    if(tmp >> k){
        q = qs[qiter++];
    }

    x = (q << k) | (tmp & k_1s);

    filled += kp1;
    if (filled >= 8){
        ++buf_iter;
        filled -= 8;
    }

#if 0
    q = 0;
    if(get_bit(buf)){
        q = qs[qiter++];
    }

//    x = q * M;
    x = q << k;

    for (j = k-1; j >= 0; --j) {
//        std::cout << "\niter: " << iter << "\t" << std::bitset<1>(buf[iter]) << "\t" << (buf[iter] << j) << std::endl;
        x = x | (get_bit(buf) << j);
//        if(rank==rank_ver) std::cout << "x = " << x << std::endl;
    }
#endif

    v[viter++] = x;

#ifdef __DEBUG1__
    if(verbose_decomp && rank==rank_ver){
        std::cout << viter << ": v[viter] = " << v[viter-1] << ", diff = " << x << ", q = " << q << " (buf_iter: " << buf_iter << ")\n";
//        print_array(qs, q_sz, 0, "qs after", comm);
    }
#endif

    // 2- decode the rest of v
    // ======================================

    while(viter < v_sz){

//        tmp2 = (buf[buf_iter] << 8u) | (buf[buf_iter+1]);
//        tmp  = (tmp2 >> (ofst - filled)) & kp1_1s;

        tmp = ( ((buf[buf_iter] << 8u) | (buf[buf_iter+1])) >> (ofst - filled)) & kp1_1s;

        q = 0;
        if(tmp >> k){
            q = qs[qiter++];
        }

        x = (q << k) | (tmp & k_1s);

        filled += kp1;
        if (filled >= 8){
            ++buf_iter;
            filled -= 8;
        }

        v[viter] = v[viter-1] + x;
        ++viter;

#if 0
        q = 0;
        if(get_bit(buf)){
            q = qs[qiter++];
        }

//        x = q * M;
        x = q << k;

        for (j = k-1; j >= 0; --j) {
//            std::cout << "iter: " << iter << "\t" << std::bitset<1>(buf[iter]) << std::endl;
            x = x | (get_bit(buf) << j);
//            if(rank==rank_ver) std::cout << "x = " << x << std::endl;
        }
#endif

#ifdef __DEBUG1__
//        cout << "buf[buf_iter]: " << std::bitset<8>(buf[buf_iter]) << ", buf[buf_iter+1]: " << std::bitset<8>(buf[buf_iter+1])
//                << ", tmp2: " << std::bitset<16>(tmp2) << ", tmp: " << std::bitset<16>(tmp)
//                << ", (tmp & k_1s): " << (tmp & k_1s) << ", filled: " << filled << ", ofst: " << ofst
//                << ", tmp >> k: " << (tmp >> k) << endl;

        assert(buf_iter < buf_sz);
        assert(x != INT32_MAX);
        if(verbose_decomp && rank==rank_ver){
            std::cout << viter << ": v[viter] = " << v[viter-1] << ", diff = " << x << ", q = " << q << " (buf_iter: " << buf_iter << ")\n";
        }
#endif
    }

#ifdef __DEBUG1__
//    ASSERT(viter == v_sz, "rank " << rank << ": viter: " << viter << ", v_sz: " << v_sz);
//    print_array(v, v_sz, rank_ver, "v decompressed", comm);
#endif
}