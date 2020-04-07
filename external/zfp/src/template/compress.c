#if defined(WITH_IPP)
/*
* This source code file was modified with Intel(R) Integrated Performance Primitives library content
*/
#define REVERSIBLE(zfp) ((zfp)->minexp < ZFP_MIN_EXP) /* reversible mode? */
#endif
/* compress 1d contiguous array */
static void
_t2(compress, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  uint nx = field->nx;
  uint mx = nx & ~3u;
  uint x;

  /* compress array one block of 4 values at a time */
  for (x = 0; x < mx; x += 4, data += 4)
    _t2(zfp_encode_block, Scalar, 1)(stream, data);
  if (x < nx)
    _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, data, nx - x, 1);
}

/* compress 1d strided array */
static void
_t2(compress_strided, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = field->data;
  uint nx = field->nx;
  int sx = field->sx ? field->sx : 1;
  uint x;

  /* compress array one block of 4 values at a time */
  for (x = 0; x < nx; x += 4) {
    const Scalar* p = data + sx * (ptrdiff_t)x;
    if (nx - x < 4)
      _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, p, nx - x, sx);
    else
      _t2(zfp_encode_block_strided, Scalar, 1)(stream, p, sx);
  }
}

/* compress 2d strided array */
static void
_t2(compress_strided, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  uint x, y;

  /* compress array one block of 4x4 values at a time */
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      if (nx - x < 4 || ny - y < 4)
        _t2(zfp_encode_partial_block_strided, Scalar, 2)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        _t2(zfp_encode_block_strided, Scalar, 2)(stream, p, sx, sy);
    }
}

#if defined(WITH_IPP) && !defined(_SET_TMP_BLOCK_FROM_)
#define _SET_TMP_BLOCK_FROM_
static void  CopyFromPartialBlock(const Ipp32f *pSrc, int stepY, int stepZ, int sizeX, int sizeY, int sizeZ, Ipp32f *pTmpBlock)
{
    Ipp32f    *pTmp;
    int       x, y, z, serIdx;
    int       copyX, copyY, copyZ;
    for (serIdx = z = 0; z < 4; z++) {
        copyZ = (z < sizeZ) ? z : sizeZ - 1;
        for (y = 0; y < 4; y++) {
            copyY = (y < sizeY) ? y : sizeY - 1;
            pTmp = (Ipp32f*)pSrc + copyZ * stepZ + copyY * stepY;
            for (x = 0; x < 4; x++) {
                copyX = (x < sizeX) ? x : sizeX - 1;
                pTmpBlock[serIdx++] = pTmp[copyX];
            }
        }
    }
}
#endif
/* compress 3d strided array */
static void
_t2(compress_strided, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  int sz = field->sz ? field->sz : (int)(nx * ny);
  uint x, y, z;

#if defined(IPP_OPTIMIZATION_ENABLED)

  IppEncodeZfpState_32f* pState = NULL;
  int srcStep = nx * sizeof(Ipp32f);
  int srcPlaneStep = srcStep * ny;
  Ipp32f pTmpBlock[64];
  bitstream *pBitStream = NULL;
  uint min_bits, max_bits, max_prec;
  int min_exp;
  int sizeState = 0;
  if (!(REVERSIBLE(stream)))
  {
    ippsEncodeZfpGetStateSize_32f(&sizeState);
    pState = (IppEncodeZfpState_32f *)ippsMalloc_8u(sizeState);
    pBitStream = stream->stream;
    ippsEncodeZfpInitLong_32f((Ipp8u*)stream_data(pBitStream), stream_capacity(pBitStream), pState);
    zfp_stream_params(stream, &min_bits, &max_bits, &max_prec, &min_exp);
    ippsEncodeZfpSet_32f(min_bits, max_bits, max_prec, min_exp, pState);
  }
#endif
  /* compress array one block of 4x4x4 values at a time */
  for (z = 0; z < nz; z += 4)
    for (y = 0; y < ny; y += 4)
      for (x = 0; x < nx; x += 4) {
        const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
        if (nx - x < 4 || ny - y < 4 || nz - z < 4)
        {
        #if !defined(IPP_OPTIMIZATION_ENABLED)
          _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
        #else
          if (!(REVERSIBLE(stream)))
          {
            CopyFromPartialBlock((const Ipp32f *)p, sy, sz, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), pTmpBlock);
            ippsEncodeZfp444_32f(pTmpBlock, 4 * sizeof(Ipp32f), 4 * 4 * sizeof(Ipp32f), pState);
          }
          else
          {
            _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
          }
        #endif
        }
        else
        { 
	      #if !defined(IPP_OPTIMIZATION_ENABLED)
           _t2(zfp_encode_block_strided, Scalar, 3)(stream, p, sx, sy, sz);
        #else
          if (!(REVERSIBLE(stream)))
          {
            ippsEncodeZfp444_32f((const Ipp32f *)p, srcStep, srcPlaneStep, pState);
          }
          else
          {
            _t2(zfp_encode_block_strided, Scalar, 3)(stream, p, sx, sy, sz);
          }
        #endif
        }
      }
#if defined(IPP_OPTIMIZATION_ENABLED)
  if (!(REVERSIBLE(stream)) && pState != NULL)
  {
      Ipp64u comprLen;
      ippsEncodeZfpFlush_32f(pState);
      ippsEncodeZfpGetCompressedSizeLong_32f(pState, &comprLen);
      stream_set_eos(pBitStream, comprLen);
      ippsFree(pState);
  }
#endif
}

/* compress 4d strided array */
static void
_t2(compress_strided, Scalar, 4)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  uint nw = field->nw;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  int sz = field->sz ? field->sz : (int)(nx * ny);
  int sw = field->sw ? field->sw : (int)(nx * ny * nz);
  uint x, y, z, w;

  /* compress array one block of 4x4x4x4 values at a time */
  for (w = 0; w < nw; w += 4)
    for (z = 0; z < nz; z += 4)
      for (y = 0; y < ny; y += 4)
        for (x = 0; x < nx; x += 4) {
          const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
          if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
            _t2(zfp_encode_partial_block_strided, Scalar, 4)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
          else
            _t2(zfp_encode_block_strided, Scalar, 4)(stream, p, sx, sy, sz, sw);
        }
}
