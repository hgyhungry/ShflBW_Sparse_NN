#pragma once
#include "block_sparse/spconv/spconv_op.h"
#include "block_sparse/spconv/spconv_library_decl.h"
#include "block_sparse/cuda_array.h"

template<typename T>
inline bool checkEqual(int f, int n, int h, int w, const std::vector<T> &res, 
    const std::vector<float> &ref, bool verbose=false) 
{
    bool passed = true;
    for (int nn = 0; nn < n; nn++) {
        for (int hh = 0; hh < h; hh++) {
            for (int ww = 0; ww < w; ww++) {
                for (int ff = 0; ff < f; ff++) {
                    int idx = nn + (ww + (hh + ff*h)*w)*n;
                    float o = static_cast<float>(res[idx]);
                    float o_ref = ref[idx];
                    if (fabs(o_ref - o) > 1e-3 * fabs(o_ref)) {
                        passed = false;
                        if (verbose) {
                            printf("n = %d, h = %d, w = %d, f = %d, result %f != %f\n", 
                                nn, hh, ww, ff, o, o_ref);
                            break;
                        }
                        else 
                            return passed; // early exit;
                    }
                }
            }
        }
    }
    return passed;
}

inline bool verify(SpconvBlockwiseInitFn_t init_fn,
                   SpconvBlockwiseExecFn_t exec_fn,
                   BlockwiseSpFilter<half> &spfilter, 
                   int N, int H, int W, int stride,
                   CudaRandomArray<half> &IFMap,
                   CudaZerosArray<half>  &OFMap,
                   const std::vector<float> &OFMap_ref)
{
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return false;
    }

    CUDA_CHECK( cudaMemset(OFMap.device_ptr, 0x0, sizeof(half)*spfilter.F*N*H*W));

    SpconvBlockwiseOpState state = (*init_fn)(spfilter, N, H, W, stride, 
        IFMap.device_ptr, OFMap.device_ptr);
    if (!state.initSuccess) {
        std::cerr << "return due to unsuccessful initialization. " << std::endl;
        return false;
    }

    (*exec_fn)(state, NULL);
    cudaDeviceSynchronize();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "kernel failed." << std::endl;
        return false;
    }

    OFMap.sync_host();

    bool passed = checkEqual<half>(spfilter.F, N, H, W, OFMap.host_array, 
        OFMap_ref, true);
    return passed;
}