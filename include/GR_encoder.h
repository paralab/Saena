#ifndef SAENA_GR_ENCODER_H
#define SAENA_GR_ENCODER_H

#include "data_struct.h"

typedef unsigned int  index_t;
typedef unsigned long nnz_t;
typedef double        value_t;
typedef unsigned char uchar;


// Golomb-Rice encoder
class GR_encoder {
private:
    nnz_t   buf_sz   = 0; // size of the allocated buffer
    index_t filled   = 0; // to go over 8 bites of each byte.
    nnz_t   buf_iter = 0; // to go over the compression buffer

    bool verbose_comp   = false;
    bool verbose_decomp = false;

public:
    //constructors
    GR_encoder() :  buf_sz(0), filled(0), buf_iter(0) {}
    explicit GR_encoder(nnz_t _buf_sz) :  buf_sz(_buf_sz), filled(0), buf_iter(0) {}

    void    put_bit(uint8_t *buf, uint8_t b);
    index_t get_bit(const uint8_t *buf);
    void    compress(index_t *v, index_t v_sz, index_t k, uint8_t *buf);
    void    decompress(index_t *v, index_t buf_sz, index_t k, int q_sz, uint8_t *buf);
};


#endif //SAENA_GR_ENCODER_H
