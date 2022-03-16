#pragma once
#include "block_sparse/common/library_util.h"

#if GPU_CC >= 80
// block_sz = 16
DECL_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 4);
// block_sz = 32
DECL_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 128,64, 32, 32, 64, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 16, 4);
// block_sz = 64
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 16, 5);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 16, 3);
// block_sz = 128
DECL_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 16, 4);

#endif
#if GPU_CC >= 75

// block_sz = 16
DECL_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 16, 16, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 8, 4);
// block_sz = 32
DECL_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 32, 64, 16, 32, 16, 16, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 32, 64, 32, 32, 16, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 32, 128,64, 32, 32, 64, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 16, 16, 8, 4);
// block_sz = 64
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 16, 16, 8, 5);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 32, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 64, 256,16, 32, 64, 16, 16, 16, 8, 3);
// block_sz = 128
DECL_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 16, 64, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 32, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 8, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 16, 16, 8, 4);
DECL_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 8, 2);
DECL_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 16, 16, 8, 4);

#else 
// block_sz = 16
DECL_FUN(SpconvBlockwise, 16, 64, 16, 16, 16, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 64, 32, 16, 16, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 128,16, 16, 32, 16, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 128,32, 16, 32, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 256,16, 16, 64, 16, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 16, 256,32, 16, 64, 32, 16, 16, 16, 4);
// block_sz = 32
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 128,16, 32, 32, 16, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 128,32, 16, 64, 32, 16, 16, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 128,32, 16, 64, 32, 16, 16, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 128,32, 16, 64, 32, 16, 16, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 128,32, 32, 32, 32, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 256,16, 32, 64, 16, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 32, 256,32, 32, 64, 32, 32, 32, 16, 4);
// block_sz = 64
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 64, 16, 32, 32, 16, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 64, 32, 32, 32, 32, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 128,16, 32, 64, 16, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 128,16, 32, 64, 16, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 128,16, 32, 64, 16, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 64, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 64, 32, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 128,32, 32, 64, 32, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 128,16, 64, 32, 16, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 128,16, 64, 32, 16, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 128,16, 64, 32, 16, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 64, 128,32, 64, 32, 32, 32, 32, 16, 4);
// block_sz = 128
DECL_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 128,32, 32, 32, 32, 32, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 32, 32, 16, 3);
DECL_FUN(SpconvBlockwise, 128,64, 32, 32, 64, 32, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 128,128,32, 32, 64, 32, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 128,128,32, 64, 32, 32, 32, 32, 16, 4);
DECL_FUN(SpconvBlockwise, 128,128,64, 32, 64, 64, 32, 32, 16, 2);
DECL_FUN(SpconvBlockwise, 128,128,64, 64, 32, 64, 32, 32, 16, 2);

#endif 

