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


void GR_encoder::compress(index_t *v, index_t v_sz, index_t k, uint8_t *buf){

    if(k == 7){
        compress_1byte(v, v_sz, k, buf);
    }else if(k == 15){
        auto buf2 = reinterpret_cast<uint16_t*>(buf);
        compress_2bytes(v, v_sz, k, buf2);
    }else{
        printf("compress(): compression for when k != 7 or 15 is not supported! k is %d\n", k);
        exit(EXIT_FAILURE);
    }

}


void GR_encoder::compress_1byte(index_t *v, index_t v_sz, index_t k, uint8_t *buf){

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

    short q;
    int i;
    int diff;
    int qiter = 0;
    buf_iter  = 0;

    uint8_t x;
    index_t k_1s = (1u << k) - 1;

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

    x = 0;
    if(q != 0){
        qs[qiter++] = q;
        x = (1u << k);
    }

    buf[buf_iter++] = x | (diff & k_1s);


#ifdef __DEBUG1__
    if(verbose_comp && rank==rank_ver){
        // print the binary representation
        // cout << std::bitset<32>(diff) << "\t" << std::bitset<8>(x) << endl;
        std::cout << "\n" << 0 << "\tdiff: " << diff << ",\tv[i]: " << v[0] << ",\tq: " << q << std::endl;
    }
#endif

    // 2- encode the rest of v
    // ======================================

    for(i = 1; i < v_sz; ++i){

        diff = static_cast<int>(v[i] - v[i-1]);

//        q = diff / M;
        q = diff >> k;

        x = 0;
        if(q != 0){
            qs[qiter++] = q;
            x = (1u << k);
        }

        buf[buf_iter++] = x | (diff & k_1s);

#ifdef __DEBUG1__
        assert(buf_iter <= r_sz);
        if(verbose_comp && rank==rank_ver){
            // print the binary representation
//            std::cout << std::bitset<32>(diff) << "\t" << std::bitset<8>(x) << std::endl;
            std::cout << i << "\tdiff: " << diff << ",\tv[i]: " << v[i] << ",\tv[i-1]: " << v[i-1] << ",\tq: " << q << "\n";
        }
#endif
    }

#ifdef __DEBUG1__
    {
//        if(!rank) std::cout << "buf_iter = " << buf_iter << ", qiter = " << qiter << ", tot = " << buf_iter + (qiter * sizeof(short)) << std::endl;
//        print_array(qs, qiter, 0, "qs before", MPI_COMM_WORLD);
//        for(int i = 0; i < qiter; ++i){
//            qs[i] = static_cast<short>(i);
//            if(rank==0) std::cout << i << "\t" << qs[i] << std::endl;
//        }
    }
#endif
}


void GR_encoder::compress_2bytes(index_t *v, index_t v_sz, index_t k, uint16_t *buf){

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

    short q;
    int i;
    int diff;
    int qiter = 0;
    buf_iter  = 0;

    uint16_t x;
    index_t k_1s = (1u << k) - 1;

    auto r_sz = rem_sz(v_sz, k);
    auto qs   = reinterpret_cast<short*>(&buf[r_sz/2]); // each buf value is 2 bytes, but r_sz is the size of remainder in bytes, so devide by 2.

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

    x = 0;
    if(q != 0){
        qs[qiter++] = q;
        x = (1u << k);
    }

    buf[buf_iter++] = x | (diff & k_1s);


#ifdef __DEBUG1__
    if(verbose_comp && rank==rank_ver){
        // print the binary representation
        // cout << std::bitset<32>(diff) << "\t" << std::bitset<8>(x) << endl;
        std::cout << "\n" << 0 << "\tdiff: " << diff << ",\tv[i]: " << v[0] << ",\tq: " << q << std::endl;
    }
#endif

    // 2- encode the rest of v
    // ======================================

    for(i = 1; i < v_sz; ++i){

        diff = static_cast<int>(v[i] - v[i-1]);

//        q = diff / M;
        q = diff >> k;

        x = 0;
        if(q != 0){
            qs[qiter++] = q;
            x = (1u << k);
        }

        buf[buf_iter++] = x | (diff & k_1s);

#ifdef __DEBUG1__
        assert(2 * buf_iter <= r_sz);
        if(verbose_comp && rank==rank_ver){
            // print the binary representation
//            std::cout << std::bitset<32>(diff) << "\t" << std::bitset<8>(x) << std::endl;
            std::cout << i << "\tdiff: " << diff << ",\tv[i]: " << v[i] << ",\tv[i-1]: " << v[i-1] << ",\tq: " << q << "\n";
        }
#endif

    }

#ifdef __DEBUG1__
    {
        // each buffer is 2 bytes, so mutipl buf_iter by 2 to compute the total size in bytes.
//        if(!rank) std::cout << "buf_iter = " << buf_iter << ", qiter = " << qiter << ", tot = " << 2 * buf_iter + (qiter * sizeof(short)) << std::endl;
//        print_array(qs, qiter, 0, "qs before", MPI_COMM_WORLD);

//        for(int i = 0; i < qiter; ++i){
//            qs[i] = static_cast<short>(i);
//            if(rank==0) std::cout << i << "\t" << qs[i] << std::endl;
//        }
    }
#endif
}



void GR_encoder::decompress(index_t *v, index_t v_sz, index_t k, int q_sz, uint8_t *buf) {

    if(k == 7){
        decompress_1byte(v, v_sz, k, q_sz, buf);
    }else if(k == 15){
        auto buf2 = reinterpret_cast<uint16_t*>(buf);
        decompress_2bytes(v, v_sz, k, q_sz, buf2);
    }else{
        printf("decompress(): decompression for when k != 7 or 15 is not supported! k is %d\n", k);
        exit(EXIT_FAILURE);
    }

}


void GR_encoder::decompress_1byte(index_t *v, index_t v_sz, index_t k, int q_sz, uint8_t *buf) {

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

    int x;
    short q;
    nnz_t qiter = 0;
    nnz_t iter  = 0;
    index_t k_1s  = (1u << k) - 1;

    // 1- decode v[0]
    // ======================================

    q = 0;
    if(buf[iter] >> k){
        q = qs[qiter++];
    }

    x = (q << k) | (buf[iter] & k_1s);

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

    v[iter++] = x;

#ifdef __DEBUG1__
    if(verbose_decomp && rank==rank_ver){
        std::cout << "\n" << 0 << ":\tv[iter] = " << v[iter-1] << ",\tdiff = " << x << ",\tq = " << q << "\n";
//        print_array(qs, q_sz, 0, "qs after", comm);
    }
#endif

    // 2- decode the rest of v
    // ======================================

    while(iter < v_sz){
        q = 0;
        if(buf[iter] >> k){
            q = qs[qiter++];
        }

        x = (q << k) | (buf[iter] & k_1s);
        v[iter] = v[iter-1] + x;
        ++iter;

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

        assert(iter < buf_sz);
        assert(x != INT32_MAX);
        if(verbose_decomp && rank==rank_ver){
            std::cout << iter-1 << ":\tv[iter] = " << v[iter-1] << ",\tdiff = " << x << ",\tq = " << q << "\n";
        }
#endif
    }

#ifdef __DEBUG1__
//    ASSERT(viter == v_sz, "rank " << rank << ": viter: " << viter << ", v_sz: " << v_sz);
//    print_array(v, v_sz, rank_ver, "v decompressed", comm);
#endif
}


void GR_encoder::decompress_2bytes(index_t *v, index_t v_sz, index_t k, int q_sz, uint16_t *buf) {

#ifdef __DEBUG1__
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int rank_ver = 0;

//    unsigned int M = 1U << k;
#endif

    auto r_sz = rem_sz(v_sz, k);
    auto qs   = reinterpret_cast<short*>(&buf[r_sz/2]); // each buf value is 2 bytes, but r_sz is the size of remainder in bytes, so devide by 2.

#ifdef __DEBUG1__
//    print_array(qs, q_sz, 0, "qs after", comm);
    if(verbose_decomp && rank==rank_ver){
        std::cout << "\nrank " << rank << ": decompress: k: " << k << ", M: " << (1U << k) << ", r_sz: " << r_sz << ", q_sz: " << q_sz << std::endl;
    }
#endif

    int x;
    short q;
    nnz_t qiter = 0;
    nnz_t iter  = 0;
    index_t k_1s  = (1u << k) - 1;

    // 1- decode v[0]
    // ======================================

    q = 0;
    if(buf[iter] >> k){
        q = qs[qiter++];
    }

    x = (q << k) | (buf[iter] & k_1s);

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

    v[iter++] = x;

#ifdef __DEBUG1__
    if(verbose_decomp && rank==rank_ver){
//        std::cout << "\n" << 0 << std::bitset<16>(buf[buf_iter]) << ")\n";
        std::cout << "\n" << 0 << ":\tv[iter] = " << v[iter-1] << ",\tdiff = " << x << ",\tq = " << q << "\n";
//        print_array(qs, q_sz, 0, "qs after", comm);
    }
#endif

    // 2- decode the rest of v
    // ======================================

    while(iter < v_sz){
        q = 0;
        if(buf[iter] >> k){
            q = qs[qiter++];
        }

        x = (q << k) | (buf[iter] & k_1s);
        v[iter] = v[iter-1] + x;
        ++iter;

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

        assert(iter < buf_sz);
        assert(x != INT32_MAX);
        if(verbose_decomp && rank==rank_ver){
            std::cout << iter-1 << ":\tv[iter] = " << v[iter-1] << ",\tdiff = " << x << ",\tq = " << q << "\n";
        }
#endif
    }

#ifdef __DEBUG1__
    if(verbose_decomp && rank==rank_ver){
        std::cout << "\niter = " << iter << ", qiter = " << qiter << "\n";
    }
//    ASSERT(viter == v_sz, "rank " << rank << ": viter: " << viter << ", v_sz: " << v_sz);
//    print_array(v, v_sz, rank_ver, "v decompressed", comm);
#endif
}


// compress bit by bit
void GR_encoder::compress2(index_t *v, index_t v_sz, index_t k, uchar *buf){

#ifdef __DEBUG1__
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int rank_ver = 0;

//    print_array(v, v_sz, rank_ver, "v compress", comm);
//    unsigned int M = 1U << k;
#endif

    printf("the compress buffer should be initialized to 0 before using compress2()\n");
    exit(EXIT_FAILURE);

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