#pragma once 

#include "block_sparse/common/base.h"
#include "block_sparse/common/mma.h"
#include "block_sparse/common/memcpy.h"
#include "block_sparse/common/swizzle.h"
#include "block_sparse/common/epilogue.h"

template<
    // block-sparse pattern
    int BM_, int BK_,
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // threadblock level pipeline stage
    int      kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle,
    typename BSwizzle,
    typename CSwizzle,
    // pipeline configuration
    bool     UseRegisterDoubleBuffer = false,
    bool     UseMinimumSync = true,
    bool     GridMappingXYToMN_ = false
>
struct SpmmBlockwiseKernel {
    static constexpr int BM      = BM_;
    static constexpr int BK      = BK_;
    static constexpr int block_M = ThreadBlockShape::M;
    static constexpr int block_N = ThreadBlockShape::N;
    static constexpr int block_K = ThreadBlockShape::K;
    static constexpr int warp_M  = WarpShape::M;
    static constexpr int warp_N  = WarpShape::N;
    static constexpr int mma_M   = MmaShape::M;
    static constexpr int mma_N   = MmaShape::N;
    static constexpr int mma_K   = MmaShape::K;
    static_assert(BM == block_M, 
    "Only support threadblock shape M == BM in block-sparse shape.\n");
    
    static_assert(WarpShape::K == ThreadBlockShape::K, 
    "K-dim of warp and threadblock tiling must be the same. "
    "Split-K not supported.\n");
    
    static_assert( block_M % warp_M == 0);
    static_assert( block_N % warp_N == 0);
    static_assert( warp_M  % mma_M  == 0);
    static_assert( warp_N  % mma_N  == 0);
    static_assert( block_K % mma_K  == 0);
    static_assert( block_K % BK     == 0);
    static_assert( kThreadBlockStage > 1);
    
    static_assert( warp_N % 16 == 0, 
    "Only support warp shape M>=16 for performance.\n");
    
    static constexpr int metaPrefetchBlock = 512;
    static_assert(metaPrefetchBlock / (block_K/BK) >= kThreadBlockStage);

    // precompute constants
    static constexpr bool GridMappingXYToMN = GridMappingXYToMN_; 
    static constexpr int blockDim = 32 * (block_M/warp_M) * (block_N/warp_N);
#if GPU_CC >= 80
    // use multi-stage shared-mem buffer (needed by async copy)
    static constexpr size_t input_buffer_size_static = 
        (block_M + block_N) * block_K * kThreadBlockStage * sizeof(half)
        + metaPrefetchBlock * 2 * sizeof(int);
#else
    // use ping-pong buffer and multi-stage regfile buffering
    static constexpr size_t input_buffer_size_static = 
        (block_M + block_N) * block_K * 2 * sizeof(half)
        + metaPrefetchBlock * sizeof(int);
#endif // GPU_CC >= 80

    static constexpr size_t output_buffer_size_static = 
        (block_M * block_N) * sizeof(half);
    
    // mainloop interface
    __device__ __forceinline__ 
    void mainLoop(const int M, const int N, const int K, 
        const int *A_bsr_indptr, const int *A_bsr_indices, 
        const half *A_values, const half *B, half *shared_mem_workspace
    );

    __device__ __forceinline__
    void epilogue(const int M, const int N, half *D, 
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

    // row swizzling
    __device__ __forceinline__
    void epilogue(const int M, const int N, const int* A_row_indices, half *D,
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

};

#if GPU_CC >= 80
// async-copy multi-stage kernel

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape, 
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__ 
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle, 
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::mainLoop
(const int M, const int N, const int K, 
    const int *A_bsr_indptr, const int *A_bsr_indices, 
    const half *A_values, const half *B, half *shared_mem_workspace) 
{
    // compute some ids
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int idx_warp_M  = (threadIdx.x >> 5) % (block_M / warp_M);
    int idx_warp_N  = (threadIdx.x >> 5) / (block_M / warp_M); 

    // compute global offsets
    int bsr_start = A_bsr_indptr[idx_block_M];
    int bsr_end   = A_bsr_indptr[idx_block_M + 1];

    const half* B_panel = &B[idx_block_N * block_N];
    int ldb = N;
    const half* A_panel = &A_values[bsr_start * BM * BK];
    const int*  I_panel = &A_bsr_indices[bsr_start];

    // compute global to shared copy constants
    const int kAccess = 8; 
    const int iter_copy_A = CEIL(block_M * block_K / kAccess, blockDim);
    const int iter_copy_B = CEIL(block_N * block_K / kAccess, blockDim);
    // global N-boudary handling
    bool is_residue = (N % block_N != 0) && 
        (idx_block_N == (GridMappingXYToMN ? gridDim.y : gridDim.x) -1);
   

    // compute shared memory buffer addresses
    const int NStage = kThreadBlockStage;
    const int size_of_tile_A = block_M * block_K;
    const int size_of_tile_B = block_N * block_K;
    half *shared_B = shared_mem_workspace ;
    half *shared_A = shared_B + size_of_tile_B * NStage;
    int  *shared_I = (int*)(shared_A + size_of_tile_A * NStage);    

    // compute shared memory offsets
    int A_warp_panel_offset = idx_warp_M * warp_M; 
    const int smem_lda = block_M;
    int B_warp_panel_offset = idx_warp_N * warp_N;
    const int smem_ldb = block_N;
    ASwizzle aSwizzle;
    BSwizzle bSwizzle;

    // define mma buffers
    typedef typename my_mma::fragment_a_colmajor<MmaShape> FragmentA;
    typedef typename my_mma::fragment_b_rowmajor<MmaShape> FragmentB;
    typedef typename my_mma::fragment_c<MmaShape, AccumulatorType> FragmentC;
    const int iter_mma_M = warp_M / mma_M;
    const int iter_mma_N = warp_N / mma_N;
    const int kWarpStage = (UseRegisterDoubleBuffer ? 2 : 1);
    FragmentA afrag[kWarpStage][iter_mma_M];
    FragmentB bfrag[kWarpStage][iter_mma_N];
    FragmentC cfrag[iter_mma_M][iter_mma_N];

    // load indices first
    int num_block = bsr_end - bsr_start;

    // main loop
    int block_per_tile = block_K / BK;
    int num_tile = CEIL(num_block, block_per_tile) ;
    
    my_pipeline::Pipeline<NStage, UseMinimumSync> pipe;
    int fetch = 0, compute = 0; 

    int num_meta_tile = CEIL(num_block, metaPrefetchBlock);
    int fetch_meta = 0;
    const int prefetchMetaStage = metaPrefetchBlock / (block_K/BK);
    // load the first tile of indices
    my_pipeline::copy_and_sync((unsigned*)shared_I, (const unsigned*)I_panel,
                                min(num_block, metaPrefetchBlock));
    fetch_meta++;

    for(; compute < num_tile; compute++) {
        for (; fetch < compute + NStage; fetch++) {
            pipe.acquire_writer();

            // prefetch metadata
            if (fetch % prefetchMetaStage ==0) {
                if(fetch_meta < num_meta_tile) {
                    int *shared_tile_I = shared_I + (fetch_meta %2)*metaPrefetchBlock;
                    const int *tile_I = I_panel + fetch_meta * metaPrefetchBlock;
                    int meta_num = min(metaPrefetchBlock, 
                                        num_block - fetch_meta * metaPrefetchBlock);
                    // for (int i = threadIdx.x; i < meta_num; i+= blockDim) {
                    //     shared_tile_I[i] = tile_I[i];
                    // }
                    my_pipeline::cp_async_tile<metaPrefetchBlock, blockDim>(
                        (uint*)shared_tile_I, (const uint*)tile_I, meta_num);
                }
                fetch_meta++;
            }

            // fetch data
            if (fetch < num_tile) {
                int *shared_tile_I = shared_I + (fetch_meta %2)*metaPrefetchBlock;
                int *tile_I = shared_tile_I + (fetch % prefetchMetaStage) * block_per_tile;
                int block_this_tile = min(block_per_tile, 
                    num_block - fetch*block_per_tile);
                const half* tile_A = A_panel + fetch * BM * block_K;
                half *shared_tile_B = shared_B + (fetch % NStage) * size_of_tile_B;
                half *shared_tile_A = shared_A + (fetch % NStage) * size_of_tile_A;

                #pragma unroll
                for (int i = 0; i < iter_copy_A; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    bool valid = (idx < size_of_tile_A);
                    bool zfill = (idx >= block_this_tile * BK * BM);
                    const half *src = tile_A + idx;
                          half *dst = shared_tile_A + aSwizzle(idx);
                    my_pipeline::cp_async_pred_zfill<16>(dst, src, valid, zfill);
                }
                #pragma unroll
                for (int i = 0; i < iter_copy_B; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    int nz_block_idx = (idx / (block_N * BK));
                    int k_base = tile_I[ nz_block_idx ];
                    int sub_row = (idx / block_N) % BK;
                    int k = k_base + sub_row;

                    bool valid = (idx < size_of_tile_B);
                    bool zfill = (nz_block_idx >= block_this_tile);
                    // residue handling
                    if (is_residue)
                    {
                        valid = valid && ((idx % block_N) < (N-idx_block_N*block_N));
                        zfill = zfill || ((idx % block_N) >= (N-idx_block_N*block_N));
                    }
                    const half *src = B_panel + k * ldb + (idx % block_N);
                          half *dst = shared_tile_B + bSwizzle(idx);
                    my_pipeline::cp_async_pred_zfill<16>(dst, src, valid, zfill); 
                }
            }
            pipe.commit_stage();
        }
        pipe.acquire_reader();

        half *shared_tile_B = shared_B + (compute % NStage) * size_of_tile_B;
        half *shared_tile_A = shared_A + (compute % NStage) * size_of_tile_A;

        #pragma unroll
        for (int k = 0; k < block_K / mma_K; k++) {
            #pragma unroll
            for (int m = 0; m < iter_mma_M; m++) {
                int offset = A_warp_panel_offset + m * mma_M + k * mma_K * smem_lda;
                my_mma::load_matrix_sync<ASwizzle>(afrag[k % kWarpStage][m], 
                    shared_tile_A, offset, smem_lda);
            }
            #pragma unroll 
            for (int n = 0; n < iter_mma_N; n++) {
                int offset = B_warp_panel_offset + n * mma_N + k * mma_K * smem_ldb;
                my_mma::load_matrix_sync<BSwizzle>(bfrag[k % kWarpStage][n], 
                    shared_tile_B, offset, smem_ldb);
            }
            #pragma unroll
            for (int m = 0; m < iter_mma_M; m++) {
                #pragma unroll
                for (int n = 0; n < iter_mma_N; n++) {
                    my_mma::mma_sync(cfrag[m][n], afrag[k % kWarpStage][m], 
                        bfrag[k % kWarpStage][n], cfrag[m][n]);
                }
            }
        }        
        pipe.release_reader();
    }

    // store C to shared memory
    __syncthreads();
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int C_warp_tile_offset = idx_warp_M * warp_M * smem_ldc + idx_warp_N * warp_N;
    
    #pragma unroll
    for (int m = 0; m < iter_mma_M; m++) {
        #pragma unroll
        for (int n = 0; n < iter_mma_N; n++) {
            int offset = C_warp_tile_offset + m * mma_M * smem_ldc + n * mma_N;
            my_mma::store_matrix_sync<CSwizzle>(cfrag[m][n], shared_C, 
                offset, smem_ldc);
        }
    }
    __syncthreads();
}

#else // GPU_CC >= 80
// shared-memory ping-pong buffering, regfile buffered multi-stage kernel

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape, 
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle, 
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::mainLoop
(const int M, const int N, const int K, 
    const int *A_bsr_indptr, const int *A_bsr_indices, 
    const half *A_values, const half *B, half *shared_mem_workspace) 
{
    // compute some ids
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int idx_warp_M  = (threadIdx.x >> 5) % (block_M / warp_M);
    int idx_warp_N  = (threadIdx.x >> 5) / (block_M / warp_M); 

    // compute global offsets
    int bsr_start = A_bsr_indptr[idx_block_M];
    int bsr_end   = A_bsr_indptr[idx_block_M + 1];

    const half* B_panel = &B[idx_block_N * block_N];
    int ldb = N;
    const half* A_panel = &A_values[bsr_start * BM * BK];
    const int*  I_panel = &A_bsr_indices[bsr_start];

    // compute global to shared copy constants
    const int NStage = kThreadBlockStage;
    const int kAccess = 8; 
    const int iter_copy_A = CEIL(block_M * block_K / kAccess, blockDim);
    const int iter_copy_B = CEIL(block_N * block_K / kAccess, blockDim);
    float4 copy_buffer_A[NStage][iter_copy_A];
    float4 copy_buffer_B[NStage][iter_copy_B];
    // global N-boudary handling
    bool is_residue = (N % block_N != 0) && 
        (idx_block_N == (GridMappingXYToMN ? gridDim.y : gridDim.x) -1);

    // compute shared memory buffer addresses
    const int NBuffer = 2;
    const int size_of_tile_A = block_M * block_K;
    const int size_of_tile_B = block_N * block_K;
    half *shared_B = shared_mem_workspace ;
    half *shared_A = shared_B + size_of_tile_B * NBuffer;
    int  *shared_I = (int*)(shared_A + size_of_tile_A * NBuffer);    

    // compute shared memory offsets
    int A_warp_panel_offset = idx_warp_M * warp_M; 
    const int smem_lda = block_M;
    int B_warp_panel_offset = idx_warp_N * warp_N;
    const int smem_ldb = block_N;
    ASwizzle aSwizzle;
    BSwizzle bSwizzle;

    // define mma buffers
    typedef typename my_mma::fragment_a_colmajor<MmaShape> FragmentA;
    typedef typename my_mma::fragment_b_rowmajor<MmaShape> FragmentB;
    typedef typename my_mma::fragment_c<MmaShape, AccumulatorType> FragmentC;
    const int iter_mma_M = warp_M / mma_M;
    const int iter_mma_N = warp_N / mma_N;
    const int kWarpStage = (UseRegisterDoubleBuffer ? 2 : 1);
    FragmentA afrag[kWarpStage][iter_mma_M];
    FragmentB bfrag[kWarpStage][iter_mma_N];
    FragmentC cfrag[iter_mma_M][iter_mma_N];

    // load indices first
    int num_block = bsr_end - bsr_start;

    int block_per_tile = block_K / BK;
    int num_tile = CEIL(num_block, block_per_tile) ;

    // preload indices

    int num_meta_tile = CEIL(num_block, metaPrefetchBlock);
    int fetch_meta = 0;
    const int prefetchMetaStage = metaPrefetchBlock / (block_K/BK);
    const int iter_copy_I = CEIL(metaPrefetchBlock, blockDim);
    int copy_buffer_I[iter_copy_I];
    // load the first tile of indices
    #pragma unroll
    for (int i = 0; i < iter_copy_I; i++) {
        int idx = threadIdx.x + i*blockDim;
        if (idx < min(num_block, metaPrefetchBlock)) {
            copy_buffer_I[i] = I_panel[idx];
        }
        else {
            copy_buffer_I[i] = 0;
        }
    }
    fetch_meta++;

    
    int fetch = 0, compute = -NStage; 
    while(compute < num_tile) {
        #pragma unroll
        for (int stage = 0; stage < NStage; stage++) {

            if (compute >= 0 && compute < num_tile) {
                // store

                half *shared_tile_B = shared_B + (compute % NBuffer) * size_of_tile_B;
                half *shared_tile_A = shared_A + (compute % NBuffer) * size_of_tile_A;
                #pragma unroll
                for (int i = 0; i < iter_copy_A; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    bool valid = (idx < size_of_tile_A);
                    half *dst = shared_tile_A + aSwizzle(idx);
                    if (valid)
                        *(float4*)dst = copy_buffer_A[stage][i];
                }
                #pragma unroll
                for (int i = 0; i < iter_copy_B; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    bool valid = (idx < size_of_tile_B);
                    // residue handling
                    if (is_residue)
                    {
                        valid = valid && ((idx % block_N) < (N-idx_block_N*block_N));
                    }
                    half *dst = shared_tile_B + bSwizzle(idx);
                    if (valid) 
                        *(float4*)dst = copy_buffer_B[stage][i];
                }
                __syncthreads();
    
                // compute

                #pragma unroll
                for (int k = 0; k < block_K / mma_K; k++) {
                    #pragma unroll
                    for (int m = 0; m < iter_mma_M; m++) {
                        int offset = A_warp_panel_offset + m * mma_M + k * mma_K * smem_lda;
                        my_mma::load_matrix_sync<ASwizzle>(afrag[k % kWarpStage][m], 
                            shared_tile_A, offset, smem_lda);
                    }
                    #pragma unroll 
                    for (int n = 0; n < iter_mma_N; n++) {
                        int offset = B_warp_panel_offset + n * mma_N + k * mma_K * smem_ldb;
                        my_mma::load_matrix_sync<BSwizzle>(bfrag[k % kWarpStage][n], 
                            shared_tile_B, offset, smem_ldb);
                    }
                    #pragma unroll
                    for (int m = 0; m < iter_mma_M; m++) {
                        #pragma unroll
                        for (int n = 0; n < iter_mma_N; n++) {
                            my_mma::mma_sync(cfrag[m][n], afrag[k % kWarpStage][m], 
                                bfrag[k % kWarpStage][n], cfrag[m][n]);
                        }
                    }
                }                    
            }
            compute++;

            // load

            if (fetch % prefetchMetaStage == 0) {
                const int* tile_I = I_panel + fetch_meta * metaPrefetchBlock;
                int *shared_tile_I = shared_I;
                // int *shared_tile_I = shared_I + ((fetch_meta-1)%2)*metaPrefetchBlock;

                int meta_num = min(metaPrefetchBlock, 
                                num_block-metaPrefetchBlock*fetch_meta);
                #pragma unroll
                for(int i = 0; i < iter_copy_I; i++) {
                    int idx = threadIdx.x + i*blockDim;
                    if (idx < metaPrefetchBlock) {
                        shared_tile_I[idx] = copy_buffer_I[i];
                    }
                    if (fetch_meta < num_meta_tile && idx < meta_num) {
                        copy_buffer_I[i] = tile_I[idx];
                    }
                }
                fetch_meta++;
                __syncthreads();
            }

            if (fetch < num_tile) {
                int* shared_tile_I = shared_I;
                // int* shared_tile_I = shared_I + (fetch_meta %2)*metaPrefetchBlock;
                int *tile_I = shared_tile_I + (fetch % prefetchMetaStage) * block_per_tile;
                int block_this_tile = min(block_per_tile, 
                    num_block - fetch*block_per_tile);
                const half* tile_A = A_panel + fetch * BM * block_K;

                #pragma unroll
                for (int i = 0; i < iter_copy_A; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    bool valid = (idx < size_of_tile_A);
                    bool zfill = (idx >= block_this_tile * BK * BM);
                    const half *src = tile_A + idx;
                    if (valid)
                        if (!zfill) 
                            copy_buffer_A[stage][i] = *(const float4*)(src);
                        else 
                            copy_buffer_A[stage][i] = {0,0,0,0};
                }
                #pragma unroll
                for (int i = 0; i < iter_copy_B; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    int nz_block_idx = (idx / (block_N * BK));
                    int k_base = tile_I[ nz_block_idx ];
                    int sub_row = (idx / block_N) % BK;
                    int k = k_base + sub_row;

                    bool valid = (idx < size_of_tile_B);
                    bool zfill = (nz_block_idx >= block_this_tile);
                    // residue handling
                    if (is_residue)
                    {
                        zfill = zfill || ((idx % block_N) >= (N-idx_block_N*block_N));
                    }
                    const half *src = B_panel + k * ldb + (idx % block_N);
                    if (valid)
                        if (!zfill) 
                            copy_buffer_B[stage][i] = *(const float4*)src;
                        else 
                            copy_buffer_B[stage][i] = {0,0,0,0};
                    // if (valid && !zfill) 
                    //     copy_buffer_B[stage][i] = *(const float4*)src;
                }
            }
            fetch++;
        }
    }

    // store C to shared memory
    __syncthreads();
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int C_warp_tile_offset = idx_warp_M * warp_M * smem_ldc + idx_warp_N * warp_N;
    
    #pragma unroll
    for (int m = 0; m < iter_mma_M; m++) {
        #pragma unroll
        for (int n = 0; n < iter_mma_N; n++) {
            int offset = C_warp_tile_offset + m * mma_M * smem_ldc + n * mma_N;
            my_mma::store_matrix_sync<CSwizzle>(cfrag[m][n], shared_C, 
                offset, smem_ldc);
        }
    }
    __syncthreads();    
}

#endif // GPU_CC >= 80

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape, 
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__ 
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle, 
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, half *D, 
    half *shared_mem_workspace, float alpha, const half *C, float beta
) {
    epilogue_impl<BM, block_N, blockDim, CSwizzle, GridMappingXYToMN>(M, N, D, 
        shared_mem_workspace, alpha, C, beta);
}

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape, 
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__ 
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle, 
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, const int *A_row_indices, half *D, 
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
    epilogue_impl<BM, block_N, blockDim, CSwizzle, GridMappingXYToMN>(M, N, 
        A_row_indices, D, shared_mem_workspace, alpha, C, beta);
}
