#pragma once

#include "block_sparse/spmm/spmm_op.h"
#include "block_sparse/spmm/spmm_library_decl.h"
#include "block_sparse/cuda_array.h"


template<typename T>
inline bool checkEqual(int m, int n, const std::vector<T> &res, 
    const std::vector<float> &ref, bool verbose=false) 
{
    bool passed = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i*n + j;
            float d = static_cast<float>(res[idx]);
            float d_ref = ref[idx];
            if (fabs(d_ref - d) > 1e-3 * fabs(d_ref)) {
                passed = false;
                if (verbose){
                    printf("i = %d, j = %d, result %f != %f\n", i, j, d, d_ref);
                    break;
                }
                else 
                    return passed; // early exit
            }
        }
    }
    return passed;
}

inline bool verify(SpmmBlockwiseInitFn_t init_fn, 
    SpmmBlockwiseExecFn_t exec_fn,
    BlockwiseSpMatrix<half> &spmat, int N, CudaRandomArray<half> &B, 
    CudaZerosArray<half> &D, const std::vector<float> &D_ref) 
{
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return false;
    }

    CUDA_CHECK( cudaMemset(D.device_ptr, 0x0, sizeof(half)*spmat.nrow*N));

    SpmmBlockwiseOpState state = (*init_fn)(spmat, N, B.device_ptr, D.device_ptr);
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

    D.sync_host();

    bool passed = checkEqual<half>(spmat.nrow, N, D.host_array, D_ref, true);
    return passed;
}
