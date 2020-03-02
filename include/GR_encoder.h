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
    int     buf_iter = 0;
    index_t filled   = 0;

    bool verbose_comp   = false;
    bool verbose_decomp = true;

public:
    void    put_bit(uchar *buf, uchar b);
    index_t get_bit(uchar *buf);
    int     compress(index_t *v, index_t v_sz, index_t k, uchar *buf);
    int     decompress(index_t *v, index_t buf_sz, index_t k, int q_sz, uchar *buf);
};


#endif //SAENA_GR_ENCODER_H
