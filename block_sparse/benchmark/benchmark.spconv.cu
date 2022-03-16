// benchmark for block-sparse spconv

#include "block_sparse/spconv/spconv_op.h"
#include "block_sparse/spconv/spconv_library_decl.h"
#include "block_sparse/cuda_array.h"
#include "argparse_util.h"
#include "timing_util.h"

inline float benchmark(SpconvBlockwiseInitFn_t init_fn, 
    SpconvBlockwiseExecFn_t exec_fn,
    BlockwiseSpFilter<half> &spfilter, int B, int H, int W, int stride, 
    half *IFMap, half *OFMap, cudaStream_t stream = NULL, 
    int warmup=10, int repeat = 100)
{
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return -1;
    }

    GpuTimer gpu_timer;
    
    SpconvBlockwiseOpState state = (*init_fn)(spfilter, B, H, W, stride, IFMap, OFMap);
    if (!state.initSuccess) {
        std::cerr << "return due to unsuccessful initialization. " << std::endl;
        return -1;
    }

    (*exec_fn)(state, stream);
    cudaDeviceSynchronize();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "kernel failed." << std::endl;
        return -1;
    }

    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup) 
            gpu_timer.start();
        
        (*exec_fn)(state, stream);
    }
    gpu_timer.stop();


    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "kernel failed." << std::endl;
        return -1;
    }
    float dur = gpu_timer.elapsed_msecs() / repeat;
    return dur;    
}

int main(int argc, const char** argv)
{
    int B, H, W, C, F, R, S, stride;
    int pattern_code, block_sz;
    float density;
    unsigned seed; 
    bool row_permute;
    bool load_pattern, store_pattern;
    std::string load_path, store_path;
    parseSpconvArgs(argc, argv, B, H, W, C, F, R, S, stride, 
        density, seed, pattern_code, block_sz, row_permute, store_pattern, 
        store_path, load_pattern, load_path,
    /* verbose */true);
    
    // pad shapes
    // const int n_pad_to = pattern_code >= 2 ? 32 : 64;
    // const int n_pad_to = 32 ;
    if (F % block_sz != 0) {
        if (load_pattern) {
            std::cerr << "Loaded matrix shape nrow is not padded to block_sz\n";
            exit(EXIT_FAILURE);
        }
        else {
            F += (block_sz - (F % block_sz));
            std::cerr << "F padded to : " << F << "\n";
        }
    }
    // if ((B*H*W) % n_pad_to != 0) {
    //     // n += (n_pad_to - (n % n_pad_to));
    //     // std::cerr << "n padded to : " << n << " for better alignment.\n";
    //     std::cerr << "Batch * Height * Width = " << B*H*W 
    //               << " not multiples of tiling " << n_pad_to;
    //     return EXIT_FAILURE;
    // }

    CudaRandomArray<half> IFMap;
    CudaZerosArray<half> OFMap;
    IFMap.initialize(B*(H*stride)*(W*stride)*C);
    OFMap.initialize(B*H*W*F);
    IFMap.sync_device();
    OFMap.sync_device();
    
    // branch on pattern_code
    if (pattern_code == 0) { // blockwise

        BlockwiseSpFilter<half> spfilter;
        spfilter.init_random(F, R, S, C, block_sz, 1, density, row_permute, seed);
        
        spfilter.transform_and_sync_device();

        // benchmark
        float gflop_count = (float)B / 1e9 * H*W*C*R*S*F*2;

#define BENCHMARK(BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE) \
{\
    std::cout << GPU_CC << " " << spfilter.config_str << " ";\
    printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d ", B, H, W, stride, \
    BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE);\
    float dur = benchmark( \
            NAME_FUN(SpconvBlockwise, Init, BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE), \
            NAME_FUN(SpconvBlockwise, Exec, BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE), \
            spfilter, B, H, W, stride, IFMap.device_ptr, OFMap.device_ptr); \
    printf("%f (us) %f (effective tflop/s)\n", dur*1e3, gflop_count/dur); \
}

        #if GPU_CC >= 80
        switch (block_sz) {
            case 16:
            BENCHMARK(16, 64, 16, 16, 16, 16, 16, 16, 16, 2);
            BENCHMARK(16, 64, 16, 16, 16, 16, 16, 16, 16, 3);
            BENCHMARK(16, 64, 16, 16, 16, 16, 16, 16, 16, 4);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 16, 2);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 16, 3);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 16, 4);
            BENCHMARK(16, 128,16, 16, 16, 16, 16, 16, 16, 2);
            BENCHMARK(16, 128,16, 16, 16, 16, 16, 16, 16, 3);
            BENCHMARK(16, 128,16, 16, 32, 16, 16, 16, 16, 4);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 16, 2);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 16, 3);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 16, 4);
            BENCHMARK(16, 256,16, 16, 64, 16, 16, 16, 16, 2);
            BENCHMARK(16, 256,16, 16, 64, 16, 16, 16, 16, 3);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 16, 2);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 16, 3);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 16, 4);
            break;
            
            case 32:
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 16, 16, 2);
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 16, 16, 3);
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 16, 16, 4);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 16, 16, 2);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 16, 16, 3);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 16, 16, 4);
            BENCHMARK(32, 128,16, 32, 32, 16, 16, 16, 16, 2);
            BENCHMARK(32, 128,16, 32, 32, 16, 16, 16, 16, 3);
            BENCHMARK(32, 128,16, 32, 32, 16, 16, 16, 16, 4);
            BENCHMARK(32, 128,32, 32, 32, 32, 16, 16, 16, 2);
            BENCHMARK(32, 128,32, 32, 32, 32, 16, 16, 16, 3);
            BENCHMARK(32, 128,32, 32, 32, 32, 16, 16, 16, 4);
            BENCHMARK(32, 128,64, 32, 32, 64, 16, 16, 16, 2);
            BENCHMARK(32, 256,16, 32, 64, 16, 16, 16, 16, 2);
            BENCHMARK(32, 256,16, 32, 64, 16, 16, 16, 16, 3);
            BENCHMARK(32, 256,32, 32, 64, 32, 16, 16, 16, 2);
            BENCHMARK(32, 256,32, 32, 64, 32, 16, 16, 16, 3);
            BENCHMARK(32, 256,32, 32, 64, 32, 16, 16, 16, 4);
            break;

            case 64:
            BENCHMARK(64, 64, 16, 32, 32, 16, 16, 16, 16, 2);
            BENCHMARK(64, 64, 16, 32, 32, 16, 16, 16, 16, 3);
            BENCHMARK(64, 64, 16, 32, 32, 16, 16, 16, 16, 4);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 16, 2);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 16, 3);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 16, 4);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 16, 5);
            BENCHMARK(64, 128,32, 32, 32, 32, 16, 16, 16, 2);
            BENCHMARK(64, 128,32, 32, 32, 32, 16, 16, 16, 3);
            BENCHMARK(64, 128,32, 32, 32, 32, 16, 16, 16, 4);
            BENCHMARK(64, 128,32, 64, 32, 32, 16, 16, 16, 2);
            BENCHMARK(64, 128,32, 64, 32, 32, 16, 16, 16, 3);
            BENCHMARK(64, 128,32, 64, 32, 32, 16, 16, 16, 4);
            BENCHMARK(64, 256,16, 32, 64, 16, 16, 16, 16, 2);
            BENCHMARK(64, 256,16, 32, 64, 16, 16, 16, 16, 3);
            break;
            
            case 128:
            BENCHMARK(128,32, 32, 32, 32, 32, 16, 16, 16, 2);
            BENCHMARK(128,32, 32, 32, 32, 32, 16, 16, 16, 3);
            BENCHMARK(128,64, 32, 16, 64, 32, 16, 16, 16, 2);
            BENCHMARK(128,64, 32, 16, 64, 32, 16, 16, 16, 3);
            BENCHMARK(128,64, 32, 16, 64, 32, 16, 16, 16, 4);
            BENCHMARK(128,64, 32, 32, 32, 32, 16, 16, 16, 2);
            BENCHMARK(128,64, 32, 32, 32, 32, 16, 16, 16, 3);
            BENCHMARK(128,64, 32, 32, 32, 32, 16, 16, 16, 4);
            BENCHMARK(128,64, 32, 32, 64, 32, 16, 16, 16, 2);
            BENCHMARK(128,64, 32, 32, 64, 32, 16, 16, 16, 3);
            BENCHMARK(128,64, 32, 32, 64, 32, 16, 16, 16, 4);
            BENCHMARK(128,128,32, 32, 64, 32, 16, 16, 16, 2);
            BENCHMARK(128,128,32, 32, 64, 32, 16, 16, 16, 4);
            BENCHMARK(128,128,32, 64, 32, 32, 16, 16, 16, 2);
            BENCHMARK(128,128,32, 64, 32, 32, 16, 16, 16, 4);        
            break;
        }
        #else 
        #if GPU_CC >= 75
        switch (block_sz) {
            case 16:
            BENCHMARK(16, 64, 16, 16, 16, 16, 16, 16, 8, 2);
            BENCHMARK(16, 64, 16, 16, 16, 16, 16, 16, 8, 3);
            BENCHMARK(16, 64, 16, 16, 16, 16, 16, 16, 8, 4);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 8, 2);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 8, 3);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 8, 4);
            BENCHMARK(16, 128,16, 16, 16, 16, 16, 16, 8, 2);
            BENCHMARK(16, 128,16, 16, 16, 16, 16, 16, 8, 3);
            BENCHMARK(16, 128,16, 16, 32, 16, 16, 16, 8, 4);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 8, 2);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 8, 3);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 8, 4);
            BENCHMARK(16, 256,16, 16, 64, 16, 16, 16, 8, 2);
            BENCHMARK(16, 256,16, 16, 64, 16, 16, 16, 8, 3);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 8, 2);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 8, 3);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 8, 4);
            break; case 32:
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 16, 8, 2);
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 16, 8, 3);
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 16, 8, 4);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 16, 8, 2);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 16, 8, 3);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 16, 8, 4);
            BENCHMARK(32, 128,16, 32, 32, 16, 16, 16, 8, 2);
            BENCHMARK(32, 128,16, 32, 32, 16, 16, 16, 8, 3);
            BENCHMARK(32, 128,16, 32, 32, 16, 16, 16, 8, 4);
            BENCHMARK(32, 128,32, 32, 32, 32, 16, 16, 8, 2);
            BENCHMARK(32, 128,32, 32, 32, 32, 16, 16, 8, 3);
            BENCHMARK(32, 128,32, 32, 32, 32, 16, 16, 8, 4);
            BENCHMARK(32, 128,64, 32, 32, 64, 16, 16, 8, 2);
            BENCHMARK(32, 256,16, 32, 64, 16, 16, 16, 8, 2);
            BENCHMARK(32, 256,16, 32, 64, 16, 16, 16, 8, 3);
            BENCHMARK(32, 256,32, 32, 64, 32, 16, 16, 8, 2);
            BENCHMARK(32, 256,32, 32, 64, 32, 16, 16, 8, 3);
            BENCHMARK(32, 256,32, 32, 64, 32, 16, 16, 8, 4);
            break; case 64:
            BENCHMARK(64, 64, 16, 32, 32, 16, 16, 16, 8, 2);
            BENCHMARK(64, 64, 16, 32, 32, 16, 16, 16, 8, 3);
            BENCHMARK(64, 64, 16, 32, 32, 16, 16, 16, 8, 4);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 8, 2);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 8, 3);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 8, 4);
            BENCHMARK(64, 64, 32, 32, 32, 32, 16, 16, 8, 5);
            BENCHMARK(64, 128,32, 32, 32, 32, 16, 16, 8, 2);
            BENCHMARK(64, 128,32, 32, 32, 32, 16, 16, 8, 3);
            BENCHMARK(64, 128,32, 32, 32, 32, 16, 16, 8, 4);
            BENCHMARK(64, 128,32, 64, 32, 32, 16, 16, 8, 2);
            BENCHMARK(64, 128,32, 64, 32, 32, 16, 16, 8, 3);
            BENCHMARK(64, 128,32, 64, 32, 32, 16, 16, 8, 4);
            BENCHMARK(64, 256,16, 32, 64, 16, 16, 16, 8, 2);
            BENCHMARK(64, 256,16, 32, 64, 16, 16, 16, 8, 3);
            break; case 128:
            BENCHMARK(128,32, 32, 32, 32, 32, 16, 16, 8, 2);
            BENCHMARK(128,32, 32, 32, 32, 32, 16, 16, 8, 3);
            BENCHMARK(128,64, 32, 16, 64, 32, 16, 16, 8, 2);
            BENCHMARK(128,64, 32, 16, 64, 32, 16, 16, 8, 3);
            BENCHMARK(128,64, 32, 16, 64, 32, 16, 16, 8, 4);
            BENCHMARK(128,64, 32, 32, 32, 32, 16, 16, 8, 2);
            BENCHMARK(128,64, 32, 32, 32, 32, 16, 16, 8, 3);
            BENCHMARK(128,64, 32, 32, 32, 32, 16, 16, 8, 4);
            BENCHMARK(128,64, 32, 32, 64, 32, 16, 16, 8, 2);
            BENCHMARK(128,64, 32, 32, 64, 32, 16, 16, 8, 3);
            BENCHMARK(128,64, 32, 32, 64, 32, 16, 16, 8, 4);
            BENCHMARK(128,128,32, 32, 64, 32, 16, 16, 8, 2);
            BENCHMARK(128,128,32, 32, 64, 32, 16, 16, 8, 4);
            BENCHMARK(128,128,32, 64, 32, 32, 16, 16, 8, 2);
            BENCHMARK(128,128,32, 64, 32, 32, 16, 16, 8, 4);
        }
        #else
        switch(block_sz) {
            case 16:
            BENCHMARK(16, 64, 16, 16, 16, 16, 16, 16, 16, 2);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 16, 2);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 16, 3);
            BENCHMARK(16, 64, 32, 16, 16, 32, 16, 16, 16, 4);
            BENCHMARK(16, 128,16, 16, 32, 16, 16, 16, 16, 2);
            BENCHMARK(16, 128,16, 16, 32, 16, 16, 16, 16, 3);
            BENCHMARK(16, 128,16, 16, 32, 16, 16, 16, 16, 4);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 16, 2);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 16, 3);
            BENCHMARK(16, 128,32, 16, 32, 32, 16, 16, 16, 4);
            BENCHMARK(16, 256,16, 16, 64, 16, 16, 16, 16, 2);
            BENCHMARK(16, 256,16, 16, 64, 16, 16, 16, 16, 3);
            BENCHMARK(16, 256,16, 16, 64, 16, 16, 16, 16, 4);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 16, 2);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 16, 3);
            BENCHMARK(16, 256,32, 16, 64, 32, 16, 16, 16, 4);
            break; case 32:
            BENCHMARK(32, 128,16, 32, 32, 16, 32, 32, 16, 2);
            BENCHMARK(32, 128,16, 32, 32, 16, 32, 32, 16, 3);
            BENCHMARK(32, 128,16, 32, 32, 16, 32, 32, 16, 4);
            BENCHMARK(32, 128,32, 16, 64, 32, 16, 16, 16, 2);
            BENCHMARK(32, 128,32, 16, 64, 32, 16, 16, 16, 3);
            BENCHMARK(32, 128,32, 16, 64, 32, 16, 16, 16, 4);
            BENCHMARK(32, 128,32, 32, 32, 32, 32, 32, 16, 2);
            BENCHMARK(32, 128,32, 32, 32, 32, 32, 32, 16, 3);
            BENCHMARK(32, 128,32, 32, 32, 32, 32, 32, 16, 4);
            BENCHMARK(32, 256,16, 32, 64, 16, 32, 32, 16, 2);
            BENCHMARK(32, 256,16, 32, 64, 16, 32, 32, 16, 3);
            BENCHMARK(32, 256,16, 32, 64, 16, 32, 32, 16, 4);
            BENCHMARK(32, 256,32, 32, 64, 32, 32, 32, 16, 2);
            BENCHMARK(32, 256,32, 32, 64, 32, 32, 32, 16, 3);
            BENCHMARK(32, 256,32, 32, 64, 32, 32, 32, 16, 4);
            break; case 64:
            BENCHMARK(64, 64, 16, 32, 32, 16, 32, 32, 16, 2);
            BENCHMARK(64, 64, 16, 32, 32, 16, 32, 32, 16, 3);
            BENCHMARK(64, 64, 16, 32, 32, 16, 32, 32, 16, 4);
            BENCHMARK(64, 64, 32, 32, 32, 32, 32, 32, 16, 2);
            BENCHMARK(64, 64, 32, 32, 32, 32, 32, 32, 16, 3);
            BENCHMARK(64, 64, 32, 32, 32, 32, 32, 32, 16, 4);
            BENCHMARK(64, 128,16, 32, 64, 16, 32, 32, 16, 2);
            BENCHMARK(64, 128,16, 32, 64, 16, 32, 32, 16, 3);
            BENCHMARK(64, 128,16, 32, 64, 16, 32, 32, 16, 4);
            BENCHMARK(64, 128,32, 32, 64, 32, 32, 32, 16, 2);
            BENCHMARK(64, 128,32, 32, 64, 32, 32, 32, 16, 3);
            BENCHMARK(64, 128,32, 32, 64, 32, 32, 32, 16, 4);
            BENCHMARK(64, 128,16, 64, 32, 16, 32, 32, 16, 2);
            BENCHMARK(64, 128,16, 64, 32, 16, 32, 32, 16, 3);
            BENCHMARK(64, 128,16, 64, 32, 16, 32, 32, 16, 4);
            BENCHMARK(64, 128,32, 64, 32, 32, 32, 32, 16, 2);
            BENCHMARK(64, 128,32, 64, 32, 32, 32, 32, 16, 3);
            BENCHMARK(64, 128,32, 64, 32, 32, 32, 32, 16, 4);
            break; case 128:
            BENCHMARK(128,32, 32, 32, 32, 32, 32, 32, 16, 2);
            BENCHMARK(128,32, 32, 32, 32, 32, 32, 32, 16, 3);
            BENCHMARK(128,64, 32, 32, 64, 32, 32, 32, 16, 2);
            BENCHMARK(128,64, 32, 32, 64, 32, 32, 32, 16, 3);
            BENCHMARK(128,64, 32, 32, 64, 32, 32, 32, 16, 4);
            BENCHMARK(128,128,32, 32, 64, 32, 32, 32, 16, 2);
            BENCHMARK(128,128,32, 32, 64, 32, 32, 32, 16, 4);
            BENCHMARK(128,128,32, 64, 32, 32, 32, 32, 16, 2);
            BENCHMARK(128,128,32, 64, 32, 32, 32, 32, 16, 4);
            BENCHMARK(128,128,64, 32, 64, 64, 32, 32, 16, 2);
            BENCHMARK(128,128,64, 64, 32, 64, 32, 32, 16, 2);        
        }
        #endif  // GPU_CC >= 75
        #endif  // GPU_CC >= 80
    }
    else { // block-2in4
        std::cerr << "only pattern:block is implemented.\n";
        exit(EXIT_FAILURE);
    }
}