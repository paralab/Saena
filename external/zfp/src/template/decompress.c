#if defined(WITH_IPP)
/*
* This source code file was modified with Intel(R) Integrated Performance Primitives library content
*/
#define REVERSIBLE(zfp) ((zfp)->minexp < ZFP_MIN_EXP) /* reversible mode? */
#endif
/* decompress 1d contiguous array */
static void
_t2(decompress, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  uint nx = field->nx;
  uint mx = nx & ~3u;
  uint x;

  /* decompress array one block of 4 values at a time */
  for (x = 0; x < mx; x += 4, data += 4)
    _t2(zfp_decode_block, Scalar, 1)(stream, data);
  if (x < nx)
    _t2(zfp_decode_partial_block_strided, Scalar, 1)(stream, data, nx - x, 1);
}

/* decompress 1d strided array */
static void
_t2(decompress_strided, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = field->data;
  uint nx = field->nx;
  int sx = field->sx ? field->sx : 1;
  uint x;

  /* decompress array one block of 4 values at a time */
  for (x = 0; x < nx; x += 4) {
    Scalar* p = data + sx * (ptrdiff_t)x;
    if (nx - x < 4)
      _t2(zfp_decode_partial_block_strided, Scalar, 1)(stream, p, nx - x, sx);
    else
      _t2(zfp_decode_block_strided, Scalar, 1)(stream, p, sx);
  }
}

/* decompress 2d strided array */
static void
_t2(decompress_strided, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  uint x, y;

  /* decompress array one block of 4x4 values at a time */
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      if (nx - x < 4 || ny - y < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 2)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        _t2(zfp_decode_block_strided, Scalar, 2)(stream, p, sx, sy);
    }
}

#if defined(IPP_OPTIMIZATION_ENABLED) && !defined(_SET_TMP_BLOCK_TO_)
#define _SET_TMP_BLOCK_TO_
static void CopyToPartialBlock(Ipp32f *pDst, int stepY, int stepZ, int sizeX, int sizeY, int sizeZ, const Ipp32f *pTmpBlock)
{
    int       x, y, z;
    for(z = 0; z < sizeZ; z++)
        for(y = 0; y < sizeY; y++)
            for (x = 0; x < sizeX; x++)
            {
                int idx = x + stepY * y + stepZ * z;
                pDst[idx] = pTmpBlock[x + 4 * y + 4 * 4 * z];
            }
}
#endif
/* decompress 3d strided array */
static void
_t2(decompress_strided, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  int sz = field->sz ? field->sz : (int)(nx * ny);
  uint x, y, z;

#if defined(IPP_OPTIMIZATION_ENABLED)
  IppDecodeZfpState_32f* pState = NULL;
  int stateSize;
  bitstream* pBitStream = NULL;
  uint min_bits, max_bits, max_prec;
  int min_exp;
  int dstStep = nx * sizeof(Ipp32f);
  int dstPlaneStep = dstStep * ny;
  Ipp32f tmpBlock[64];
  if (!(REVERSIBLE(stream)))
  {
    ippsDecodeZfpGetStateSize_32f(&stateSize);
    pState = (IppDecodeZfpState_32f*)ippsMalloc_8u(stateSize);
    pBitStream = stream->stream;
    ippsDecodeZfpInitLong_32f((Ipp8u*)stream_data(pBitStream), stream_capacity(pBitStream), pState);
    zfp_stream_params(stream, &min_bits, &max_bits, &max_prec, &min_exp);
    ippsDecodeZfpSet_32f(min_bits, max_bits, max_prec, min_exp, pState);
  }
#endif
  /* decompress array one block of 4x4x4 values at a time */
  for (z = 0; z < nz; z += 4)
    for (y = 0; y < ny; y += 4)
      for (x = 0; x < nx; x += 4) {
        Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
        if (nx - x < 4 || ny - y < 4 || nz - z < 4)
        {
        #if !defined(IPP_OPTIMIZATION_ENABLED)
          _t2(zfp_decode_partial_block_strided, Scalar, 3)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
        #else
          if (!(REVERSIBLE(stream)))
          {
            ippsDecodeZfp444_32f(pState, (Ipp32f*)tmpBlock, 4 * sizeof(Ipp32f), 4 * 4 * sizeof(Ipp32f));
            CopyToPartialBlock((Ipp32f*)p, sy, sz, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), (const Ipp32f*)tmpBlock);
          }
          else 
          {
            _t2(zfp_decode_partial_block_strided, Scalar, 3)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
          }
        #endif
        }
        else
        {
        #if !defined(IPP_OPTIMIZATION_ENABLED)
          _t2(zfp_decode_block_strided, Scalar, 3)(stream, p, sx, sy, sz);
        #else
          if (!(REVERSIBLE(stream)))
          {
            ippsDecodeZfp444_32f(pState, (Ipp32f*)p, dstStep, dstPlaneStep);
          }
          else
          {
            _t2(zfp_decode_block_strided, Scalar, 3)(stream, p, sx, sy, sz);
          }
        #endif
        }
      }
#if defined(IPP_OPTIMIZATION_ENABLED)
      if (!(REVERSIBLE(stream)) && pState != NULL)
      {
          Ipp64u decompressed_size = 0;
          ippsDecodeZfpGetDecompressedSizeLong_32f(pState, &decompressed_size);
          ippsFree(pState);
          stream_set_eos(pBitStream, decompressed_size);
      }
#endif
}

/* decompress 4d strided array */
static void
_t2(decompress_strided, Scalar, 4)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  uint nw = field->nw;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  int sz = field->sz ? field->sz : (int)(nx * ny);
  int sw = field->sw ? field->sw : (int)(nx * ny * nz);
  uint x, y, z, w;

  /* decompress array one block of 4x4x4x4 values at a time */
  for (w = 0; w < nw; w += 4)
    for (z = 0; z < nz; z += 4)
      for (y = 0; y < ny; y += 4)
        for (x = 0; x < nx; x += 4) {
          Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
          if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
            _t2(zfp_decode_partial_block_strided, Scalar, 4)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
          else
            _t2(zfp_decode_block_strided, Scalar, 4)(stream, p, sx, sy, sz, sw);
        }
}
