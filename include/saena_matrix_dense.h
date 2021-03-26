#ifndef SAENA_MATRIX_DENSE_H
#define SAENA_MATRIX_DENSE_H

#include "data_struct.h"

#ifdef SAENA_USE_ZFP
#include "zfparray1.h"
#endif

class saena_matrix;

class saena_matrix_dense {

private:
    value_t* v_send = nullptr;
    value_t* v_recv = nullptr;
//    std::vector<value_t> v_send;
//    std::vector<value_t> v_recv;
    std::vector<float> v_send_f; // single precision
    std::vector<float> v_recv_f; // single precision

public:

    MPI_Comm comm  = MPI_COMM_WORLD;
    index_t  M     = 0;
    index_t  Nbig  = 0;
    index_t  M_max = 0; // biggest M on all the processors

//    vector<vector<value_t>> entry;
//    vector<value_t> entry;
    value_t *entry = nullptr;
    std::vector<index_t> split; // (row-wise) partition of the matrix between processes

    bool use_double = true; // to determine the precision for matvec
    int MPI_flag = 0;

    saena_matrix_dense();
    saena_matrix_dense(index_t M, index_t Nbig);
    saena_matrix_dense(index_t M, index_t Nbig, MPI_Comm comm);
    saena_matrix_dense(const saena_matrix_dense &B); // copy constructor
//    saena_matrix_dense(char* Aname, MPI_Comm com);
    ~saena_matrix_dense();

    saena_matrix_dense& operator=(const saena_matrix_dense &B);

    int assemble();

    int erase();

    inline value_t get(index_t row, index_t col){
        if(row >= M || col >= Nbig){
            printf("\ndense matrix get out of range!\n");
            exit(EXIT_FAILURE);
        }else{
//            return entry[row][col];
            return entry[row * Nbig + col];
        }
    }

    inline void set(index_t row, index_t col, value_t val){
        if(row >= M || col >= Nbig){
            printf("\ndense matrix set out of range: row = %d, M = %d, col = %d, Nbig = %d\n", row, M, col, Nbig);
            exit(EXIT_FAILURE);
        }else{
//            entry[row][col] = val;
            entry[row * Nbig + col] = val;
        }
    }

    int print(int ran);

    void matvec(std::vector<value_t>& v, std::vector<value_t>& w){
        if(use_double) matvec_dense(v, w);
        else matvec_dense_float(v, w);
    }

    void matvec_dense(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_dense_float(std::vector<value_t>& v, std::vector<value_t>& w);

    int convert_saena_matrix(saena_matrix *A);

    // zfp parameters and functions
    // ***********************************************************
#ifdef SAENA_USE_ZFP
    zfp_type    zfptype = zfp_type_double;

    zfp_field*  send_field;  // array meta data
    zfp_stream* send_zfp;    // compressed stream
    bitstream*  send_stream; // bit stream to write to or read from

    zfp_field*  recv_field;  // array meta data
    zfp_stream* recv_zfp;    // compressed stream
    bitstream*  recv_stream; // bit stream to write to or read from

    bool          use_zfp          = false;
    bool          free_zfp_buff    = false;
    unsigned char *zfp_send_buff   = nullptr, // storage for compressed stream to be sent
    *zfp_recv_buff   = nullptr; // storage for compressed stream to be received
    unsigned      zfp_send_buff_sz = 0,
            zfp_send_comp_sz = 0,
            zfp_recv_buff_sz = 0;

    int    zfp_rate = 32;
    double zfp_prec = 32;

    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;

    int matvec_comp(std::vector<value_t>& v, std::vector<value_t>& w);
    int allocate_zfp();
    int deallocate_zfp();
#endif

    int matvec_test(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_time_init();
    void matvec_time_print() const;
    unsigned long matvec_iter = 0;
    double part1 = 0, part2 = 0, part3 = 0, part4 = 0, part5 = 0, part6 = 0;
};

#endif //SAENA_MATRIX_DENSE_H
