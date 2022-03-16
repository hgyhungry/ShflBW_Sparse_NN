// benchmark for block-sparse spmm

#include "block_sparse/spmm/spmm_op.h"
#include "block_sparse/cuda_array.h"

int main(int argc, const char** argv)
{
    int m = 1024, n = 1024, k = 1024;
    float density = 0.2;
    unsigned seed = 2021;
    bool row_permute = false;

    const int block_sz = 32;
    
    // define a blockwise sparse matrix
    BlockwiseSpMatrix<half> spmat;
    spmat.init_random(m, k, block_sz, 1, density, row_permute, seed);
    spmat.transform_and_sync_device();

    CudaRandomArray<half> B;
    CudaOnesArray<half> C;
    CudaZerosArray<half> D;
    B.initialize(k*n);
    C.initialize(m*n);
    D.initialize(m*n);
    B.sync_device();
    C.sync_device();
    D.sync_device();

    SpmmBlockwiseOp<ShapeBase<block_sz, 128, 32>,  // block tile
                    ShapeBase<32, 32, 32>,         // warp tile
#if GPU_CC >= 80
                    ShapeBase<16, 16, 16>,         // mma shape
#else
#if GPU_CC >= 75
                    ShapeBase<16, 16, 8>,         // mma shape
#else
                    ShapeBase<32, 32, 16>,         // mma shape
#endif
#endif
                    3>                             // number of pipeline stage
                    op;
    op.initialize(spmat, n, B.device_ptr, C.device_ptr, D.device_ptr);
    
    op();

    // verify correctness

    D.sync_host();
    
    std::vector<float> D_ref(m * n);    
    get_host_reference<half>(spmat, n, B.host_array, 1.0f, C.host_array, 1.0f,
    D_ref);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i*n + j;
            float d = static_cast<float>(D.host_array[idx]);
            float d_ref = D_ref[idx];
            if (fabs(d_ref - d) > 1e-3 * fabs(d_ref)) {
                printf("i = %d, j = %d, result %f != %f\n", i, j, d, d_ref);
                passed = false;
            }
        }
    }
    if (passed) std::cout << "Passed\n";
    else        std::cout << "Failed\n";

    return passed ? EXIT_SUCCESS: EXIT_FAILURE;
}