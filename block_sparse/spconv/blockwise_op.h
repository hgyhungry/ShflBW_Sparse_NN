#pragma once
#include "blockwise_format.h"
#include "blockwise_kernel.h"


struct SpconvBlockwiseOpState {
    size_t shared_mem_size;
    dim3 gridDim;
    dim3 blockDim;
    bool initSuccess = false;
    struct Argument_t {
        int B, H, W, C, F, R, S, stride;
        half *A_values, *IFMap, *OFMap;
        int  *A_bsr_indptr, *A_bsr_indices, *A_row_indices;
        float alpha, beta;
    } args;
};

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage,
    bool strided = false
>
class SpconvBlockwiseOp {
    static constexpr int BM = ThreadBlockShape::M;
    static constexpr int BK = 1;
    using AccumulatorType = float;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;
    // launch state
public:
    SpconvBlockwiseOpState _state;
    using KernelImpl = SpconvBlockwiseKernel<BM, BK, ThreadBlockShape, 
                        WarpShape, MmaShape, NStage, AccumulatorType,
                        ASwizzle, BSwizzle, CSwizzle, strided>;
    

    void initialize(BlockwiseSpFilter<half> &spfilter, int B, int H, int W, 
    int stride, half *IFMap, half *OFMap);
    
    void operator()(cudaStream_t stream = NULL);

};


// *** device kernel *** 
template<typename KernelImpl> __global__ 
void _spconvBlockwiseKernel(typename SpconvBlockwiseOpState::Argument_t args) 
{
    extern __shared__ half shared_mem_workspace[];
    KernelImpl k;
    k.mainLoop(args.B, args.H, args.W, args.C, args.F, args.R, args.S, args.stride,
        args.A_bsr_indptr, args.A_bsr_indices, args.A_values, args.IFMap, 
        shared_mem_workspace);
    if (args.A_row_indices==nullptr){
        k.epilogue(args.F, (args.B*args.H*args.W), args.OFMap, 
            shared_mem_workspace, args.alpha, nullptr, args.beta);
    }
    else {
        k.epilogue(args.F, (args.B*args.H*args.W), args.A_row_indices, args.OFMap,
            shared_mem_workspace, args.alpha, nullptr, args.beta);
    }
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage,
    bool strided
>
void SpconvBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage, strided>::initialize(
    BlockwiseSpFilter<half> &spfilter, int B, int H, int W, int stride, 
    half *IFMap, half* OFMap)
{
    assert(spfilter.brow == BM && "sparse matrix pattern and kernel mismatch\n");
    assert(spfilter.bcol == BK && "sparse matrix pattern and kernel mismatch\n");
    assert(spfilter.device_synced && "sparse matrix is not synced to device\n");
    
    // set argument
    BlockwiseSpFilter<half>::DeviceRef &A_ref = spfilter.device_ref;
    this->_state.args = SpconvBlockwiseOpState::Argument_t({B, H, W, 
        spfilter.C, spfilter.F, spfilter.R, spfilter.S, stride, 
        A_ref.csb_values, IFMap, OFMap, A_ref.csb_indptr, 
        A_ref.csb_indices, 
        (spfilter.row_permute ? A_ref.row_permute_ids : nullptr), 
        1.0, 0.0});
    
    // compute shared memory buffer size
    size_t input_buffer_size_dyn = 0;
    size_t input_buffer_size = input_buffer_size_dyn + 
                               KernelImpl::input_buffer_size_static;
    size_t output_buffer_size_dyn = 0;  
    if (spfilter.row_permute) 
        // with row swizzling, need buffer for row indices
        output_buffer_size_dyn = sizeof(int) * BM;
    size_t output_buffer_size = output_buffer_size_dyn + 
                                KernelImpl::output_buffer_size_static;
    
    this->_state.shared_mem_size = max(input_buffer_size, output_buffer_size);
    if (this->_state.shared_mem_size >= 32*1024) {
        // set kernel attribute
        if (cudaSuccess != cudaFuncSetAttribute( 
            _spconvBlockwiseKernel<KernelImpl>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, this->_state.shared_mem_size)
        ||  cudaSuccess != cudaFuncSetAttribute( 
            _spconvBlockwiseKernel<KernelImpl>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, 100)) {
            cudaError_t err = cudaGetLastError();
            std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err);
            this->_state.initSuccess = false;
        }
    }

    // calculate launch configuration
    int gdimX = KernelImpl::GridMappingXYToMN ?
                (spfilter.F / KernelImpl::block_M) : (CEIL((B*H*W), KernelImpl::block_N));
    int gdimY = KernelImpl::GridMappingXYToMN ? 
                (CEIL((B*H*W), KernelImpl::block_N)) : (spfilter.F / KernelImpl::block_M);
    this->_state.gridDim = dim3(gdimX, gdimY, 1);
    this->_state.blockDim = dim3(KernelImpl::blockDim, 1, 1);
    
    this->_state.initSuccess = true;
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage,
    bool strided
>
void SpconvBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage, strided>::operator()(
    cudaStream_t stream)
{
    _spconvBlockwiseKernel<KernelImpl><<<this->_state.gridDim, this->_state.blockDim,
        this->_state.shared_mem_size, stream>>>(this->_state.args);
}


// pure-function version of the original c++-object Op
// function handle easy for benchmarking, testing 

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage
>
SpconvBlockwiseOpState SpconvBlockwiseInitFn(BlockwiseSpFilter<half> &spfilter, 
int B, int H, int W, int stride, half *IFMap, half *OFMap)
{
    SpconvBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage, true> op;
    op.initialize(spfilter, B, H, W, stride, IFMap, OFMap);
    return op._state;
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage
>
void SpconvBlockwiseExecFn(SpconvBlockwiseOpState &state, cudaStream_t stream = NULL)
{
    using KernelImpl = typename SpconvBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::KernelImpl;
    using StrideKernelImpl = typename SpconvBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage, true/*strided*/>::KernelImpl;
    if (state.args.stride==1)
    _spconvBlockwiseKernel<KernelImpl><<<state.gridDim, state.blockDim,
        state.shared_mem_size, stream>>>(state.args);
    else 
    _spconvBlockwiseKernel<StrideKernelImpl><<<state.gridDim, state.blockDim,
        state.shared_mem_size, stream>>>(state.args);
}

// signature of SpconvBlockwiseInitFn(...)
typedef SpconvBlockwiseOpState (*SpconvBlockwiseInitFn_t) (BlockwiseSpFilter<half>&, int, int, int, int, half*, half*);

// signature of SpconvBlockwiseExecFn(...)
typedef void (*SpconvBlockwiseExecFn_t) (SpconvBlockwiseOpState&, cudaStream_t);

