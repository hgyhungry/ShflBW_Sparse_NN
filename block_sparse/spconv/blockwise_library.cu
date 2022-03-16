#include "./blockwise_op.h"
#include "block_sparse/common/library_util.h"


// #define NAME_FUN(type, fn, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE)\
// type##_##BM##x##BLOCK_N##x##BLOCK_K##_##WARP_M##x##WARP_N##x##WARP_K##_##MMA_M##x##MMA_N##x##MMA_K##_##NSTAGE##_##fn##Fn

#define INSTANTIATE_FUN(type, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
type##InitFn_t NAME_FUN(type, Init, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##InitFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>; \
type##ExecFn_t NAME_FUN(type, Exec, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##ExecFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>; 

#if GPU_CC >= 80
// block_sz = 16
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 4);
// block_sz = 32
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,64, 32, 32, 64, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 16, 4);
// block_sz = 64
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 5);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 16, 3);
// block_sz = 128
INSTANTIATE_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 16, 4);

#endif
#if GPU_CC >= 75

// block_sz = 16
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 8, 4);
// block_sz = 32
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,64, 32, 32, 64, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 8, 4);
// block_sz = 64
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 5);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 8, 3);
// block_sz = 128
INSTANTIATE_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 8, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 8, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 8, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 8, 4);

#else 
// block_sz = 16
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 4);
// block_sz = 32
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 16, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 16, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 16, 64, 32, 16, 16, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 32, 32, 16, 4);
// block_sz = 64
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,16, 32, 64, 16, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,16, 32, 64, 16, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,16, 32, 64, 16, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 64, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 64, 32, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 32, 64, 32, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,16, 64, 32, 16, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,16, 64, 32, 16, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,16, 64, 32, 16, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 32, 32, 16, 4);
// block_sz = 128
INSTANTIATE_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 32, 32, 16, 3);
INSTANTIATE_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 32, 32, 16, 4);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,64, 32, 64, 64, 32, 32, 16, 2);
INSTANTIATE_FUN(SpconvBlockwise, 128,128,64, 64, 32, 64, 32, 32, 16, 2);

#endif 