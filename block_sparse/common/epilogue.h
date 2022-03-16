#pragma once

#include "vector.h"
#include "memcpy.h"


template<
    int block_M, int block_N, int blockDim, 
    typename CSwizzle, bool GridMappingXYToMN
>
__device__ __forceinline__ 
void epilogue_impl
(const int M, const int N, half *D, 
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
    const int kAccess = 8;
    const int smem_ldc = block_N;
    half *shared_C = shared_mem_workspace;
    int ldc = N;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    bool is_residue = (N % block_N != 0) && 
        (idx_block_N == (GridMappingXYToMN ? gridDim.y : gridDim.x) -1);

    half *D_tile = D + idx_block_M * block_M * ldc + idx_block_N * block_N;
    HalfVector<8> buffer;
    CSwizzle cSwizzle;

    if (beta == 0.0f) {
        for (int i = 0; i < (CEIL(block_M * block_N / kAccess, blockDim)); i++) {
            int idx = kAccess * (threadIdx.x + i * blockDim);
            half *src = shared_C + cSwizzle(idx);
            half *dst = D_tile + (idx / smem_ldc) * ldc + (idx % smem_ldc);
            bool valid = (idx < block_M * block_N);
            if (is_residue)
                valid = valid && ((idx % block_N)< (N-block_N*idx_block_N));
            if (valid) {
                buffer.ld(src);
                buffer.mul(alpha);
                buffer.st(dst);
            }
        }        
    }
    else {
        const half *C_tile = C + idx_block_M * block_M * ldc + idx_block_N * block_N;
        HalfVector<8> buffer2;
        for (int i = 0; i < (CEIL(block_M * block_N / kAccess, blockDim)); i++) {
            int idx = kAccess * (threadIdx.x + i * blockDim);
            half *src = shared_C + cSwizzle(idx);
            int global_offset = (idx / smem_ldc) * ldc + (idx % smem_ldc);
            half *dst  = D_tile + global_offset;
            const half *src2 = C_tile + global_offset;
            bool valid = (idx < block_M * block_N);
            if (is_residue)
                valid = valid && ((idx % block_N)< (N-block_N*idx_block_N));
            if (valid) {
                buffer.ld(src);
                buffer2.ld(src2);
                buffer2.mul(beta);
                buffer.hfma(alpha, buffer2);
                buffer.st(dst);
            }
        }        
    }   
}

// *** with row swizzling **
template<
    int block_M, int block_N, int blockDim, 
    typename CSwizzle, bool GridMappingXYToMN
>
__device__ __forceinline__ 
void epilogue_impl
(const int M, const int N, const int *A_row_indices, half *D, 
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
    const int kAccess = 8;
    const int smem_ldc = block_N;
    half *shared_C = shared_mem_workspace;
    int ldc = N;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;

    int  *I_shared = (int*)(shared_C + block_M * block_N);
    int logical_row_start = idx_block_M * block_M;
    const int  *I_panel  = A_row_indices + logical_row_start;
    my_pipeline::copy_and_sync((unsigned*)I_shared, (const unsigned*)I_panel, 
                                block_M);
    bool is_residue = (N % block_N != 0) && 
        (idx_block_N == (GridMappingXYToMN ? gridDim.y : gridDim.x) -1);

    half *D_tile = D + idx_block_N * block_N;
    HalfVector<8> buffer;
    CSwizzle cSwizzle;

    if (beta == 0.0f) {
        for (int i = 0; i < (CEIL(block_M * block_N / kAccess, blockDim)); i++) {
            int idx = kAccess * (threadIdx.x + i * blockDim);
            half *src = shared_C + cSwizzle(idx);
            bool valid = (idx < block_M * block_N);
            if (is_residue)
                valid = valid && ((idx % block_N)< (N-block_N*idx_block_N));
            if (valid) {
                int row_id = I_shared[idx / smem_ldc];
                half *dst = D_tile + row_id * ldc + (idx % smem_ldc);
                buffer.ld(src);
                buffer.mul(alpha);
                buffer.st(dst);
            }
        }        
    }
    else {
        const half *C_tile = C + idx_block_N * block_N;
        HalfVector<8> buffer2;
        for (int i = 0; i < (CEIL(block_M * block_N / kAccess, blockDim)); i++) {
            int idx = kAccess * (threadIdx.x + i * blockDim);
            half *src = shared_C + cSwizzle(idx);
            bool valid = (idx < block_M * block_N);
            if (is_residue)
                valid = valid && ((idx % block_N)< (N-block_N*idx_block_N));
            if (valid) {
                int row_id = I_shared[idx / smem_ldc];
                int global_offset = row_id * ldc + (idx % smem_ldc);
                half *dst  = D_tile + global_offset;
                const half *src2 = C_tile + global_offset;
                buffer.ld(src);
                buffer2.ld(src2);
                buffer2.mul(beta);
                buffer.hfma(alpha, buffer2);
                buffer.st(dst);
            }
        }        
    }   
}

/*
template<
  int block_M, int block_N, int blockDim, typename ComputeType, 
  typename OutputType, typename CSwizzle, bool GridMappingXYToMN
>
DEVICE_INLINE
void epilogue_impl(const int M, const int N, OutputType *D, 
    const AccumulatorType* shared_mem_workspace, 
    const float alpha, const OutputType* C, const float beta)
{
    const int kAccess = 128/(8*sizeof(ComputeType));
    ValueVector<ComputeType, kAccess> buffer;
    CSwizzle swizzle;
    
    int ldc = N;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    OutputType *D_tile = D + idx_block_M * block_M * ldc + idx_block_N * block_N;


    int iter = CEIL(block_M * block_M / kAccess, blockDim);
    int smem_ldc = block_N;
    if (beta == 0.0f) {
        for (int i = 0; i < iter; i++) {
            int idx = kAccess * (threadIdx.x + i*blockDim);
            const ComputeType* src = shared_mem_workspace + swizzle(idx);
            OutputType* dst = D_tile + (idx / smem_ldc) * ldc + (idx % smem_ldc);
            if (idx < block_M * block_N) {
                buffer.ld(src);
                buffer.mul(alpha);
                buffer.st(dst);
            }
        }
    }
    else {
        const OutputType *C_tile = C + idx_block_M * block_M * ldc
                                     + idx_block_N * block_N;
        ValueVector<OutputType, kAccess> buffer2;
        for (int i = 0; i < iter; i++) {
            int idx = kAccess * (threadIdx.x + i*blockDim);
            const ComputeType* src = shared_mem_workspace + swizzle(idx);
            int global_offset = (idx / smem_ldc) * ldc + (idx % smem_ldc);
            const OutputType* src2 = C_tile + global_offset
            OutputType* dst = D_tile + global_offset;
            if (idx < block_M * block_N) {
                buffer.ld(src);
                buffer2.ld(src2);
                buffer2.mul(beta);
                buffer.fma(alpha, buffer2);
                buffer.st(dst);
            }
        }
    }
}
*/