// benchmark for block-sparse spmm

#include "block_sparse/spconv/spconv_op.h"
#include "block_sparse/cuda_array.h"

int main(int argc, const char** argv)
{
    int B = 32, H = 14, W = 14, C = 256, F = 256, R = 3, S = 3, stride=1;
    float density = 0.2;
    unsigned seed = 2021;
    bool filter_permute = false;

    const int block_sz = 32;
    
    // define a blockwise sparse filter
    BlockwiseSpFilter<half> spfilter;
    spfilter.init_random(F, R, S, C,    // tensor shape
        block_sz, 1,                    // block pattern shape
        density, 
        filter_permute,                 // boolean, if using filter permutation
        seed);
    spfilter.transform_and_sync_device();

    CudaRandomArray<half> IFMap;
    CudaZerosArray<half> OFMap;
    IFMap.initialize(B*(H*stride)*(W*stride)*C);
    OFMap.initialize(B*H*W*F);
    IFMap.sync_device();
    OFMap.sync_device();

    SpconvBlockwiseOp<ShapeBase<block_sz, 128, 32>, // block tile
                    ShapeBase<32, 32, 32>,          // warp tile
#if GPU_CC >= 80
                    ShapeBase<16, 16, 16>,          // mma shape
#else
#if GPU_CC >= 75
                    ShapeBase<16, 16, 8>,           // mma shape
#else
                    ShapeBase<32, 32, 16>,          // mma shape
#endif
#endif
                    3>                              // number of pipeline stage
                    op;
    op.initialize(spfilter, B, H, W, stride, IFMap.device_ptr, OFMap.device_ptr);
    
    op();

    // verify correctness

    OFMap.sync_host();
    
    LayoutFilter filter_layout = filter_channel_first;
    
    std::vector<float> OFMap_ref(B*H*W*F);    
    get_host_reference<half>(spfilter, B, H, W, stride, IFMap.host_array, 
        OFMap_ref, filter_layout);

    bool passed = true;
    for (int nn = 0; nn < B; nn++) {
        for (int hh = 0; hh < H; hh++) {
            for (int ww = 0; ww < W; ww++) {
                for (int ff = 0; ff < F; ff++) {
                    int idx;
                    // if (fmap_layout==fmap_batch_first) 
                        idx = nn + (ww + (hh + ff*H)*W)*B;
                    // else 
                    //     idx = ww + (hh + (nn + ff*B)*H)*W;

                    float o = static_cast<float>(OFMap.host_array[idx]);
                    float o_ref = OFMap_ref[idx];
                    if (fabs(o_ref - o) > 1e-3 * fabs(o_ref)) {
                        passed = false;
                        printf("n = %d, h = %d, w = %d, f = %d, result %f != %f\n", 
                            nn, hh, ww, ff, o, o_ref);
                    }
                }
            }
        }
    }
    if (passed) std::cout << "Passed\n";
    else        std::cout << "Failed\n";

    return passed ? EXIT_SUCCESS: EXIT_FAILURE;
}