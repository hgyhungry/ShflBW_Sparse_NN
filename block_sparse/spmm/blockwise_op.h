#pragma once
#include "blockwise_format.h"
#include "blockwise_kernel.h"


struct SpmmBlockwiseOpState {
    size_t shared_mem_size;
    dim3 gridDim;
    dim3 blockDim;
    bool initSuccess = false;
    struct Argument_t {
        int M, N, K;
        half *A_values, *B, *C, *D;
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
    int NStage
>
class SpmmBlockwiseOp {
    static constexpr int BM = ThreadBlockShape::M;
    static constexpr int BK = 1;
    using AccumulatorType = float;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;
    // launch state
public:
    SpmmBlockwiseOpState _state;
    using KernelImpl = SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, 
                        WarpShape, MmaShape, NStage, AccumulatorType,
                        ASwizzle, BSwizzle, CSwizzle>;
    

    void initialize(BlockwiseSpMatrix<half> &spmat, int N, half *B, half *D);
    
    void initialize(BlockwiseSpMatrix<half> &spmat, int N, half *B, 
    half *C, half *D);

    void initialize(BlockwiseSpMatrix<half> &spmat, int N, half *B, 
    float alpha, half *C, float beta, half *D);
    
    void operator()(cudaStream_t stream = NULL);

};


// *** device kernel *** 
template<typename KernelImpl> __global__ 
void _spmmBlockwiseKernel(typename SpmmBlockwiseOpState::Argument_t args) 
{
    extern __shared__ half shared_mem_workspace[];
    KernelImpl k;
    k.mainLoop(args.M, args.N, args.K, args.A_bsr_indptr, args.A_bsr_indices, 
        args.A_values, args.B, shared_mem_workspace);
    if (args.A_row_indices==nullptr){
        k.epilogue(args.M, args.N, args.D, 
            shared_mem_workspace, args.alpha, args.C, args.beta);
    }
    else {
        k.epilogue(args.M, args.N, args.A_row_indices, args.D, 
            shared_mem_workspace, args.alpha, args.C, args.beta);
    }
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage
>
void SpmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::initialize(
    BlockwiseSpMatrix<half> &spmat, int N, half *B, float alpha, half *C, 
    float beta, half *D)
{
    assert(spmat.brow == BM && "sparse matrix pattern and kernel mismatch\n");
    assert(spmat.bcol == BK && "sparse matrix pattern and kernel mismatch\n");
    assert(spmat.device_synced && "sparse matrix is not synced to device\n");
    
    // set argument
    BlockwiseSpMatrix<half>::DeviceRef &A_ref = spmat.device_ref;
    this->_state.args = SpmmBlockwiseOpState::Argument_t({spmat.nrow, N, 
        spmat.ncol, A_ref.csb_values, B, C, D, A_ref.csb_indptr, 
        A_ref.csb_indices, 
        (spmat.row_permute ? A_ref.row_permute_ids : nullptr), 
        1.0, beta});
    
    // compute shared memory buffer size
    size_t input_buffer_size_dyn = 0;
    size_t input_buffer_size = input_buffer_size_dyn + 
                               KernelImpl::input_buffer_size_static;
    size_t output_buffer_size_dyn = 0;  
    if (spmat.row_permute) 
        // with row swizzling, need buffer for row indices
        output_buffer_size_dyn = sizeof(int) * BM;
    size_t output_buffer_size = output_buffer_size_dyn + 
                                KernelImpl::output_buffer_size_static;
    
    this->_state.shared_mem_size = max(input_buffer_size, output_buffer_size);
    if (this->_state.shared_mem_size >= 32*1024) {
        // set kernel attribute
        if (cudaSuccess != cudaFuncSetAttribute( 
            _spmmBlockwiseKernel<KernelImpl>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, this->_state.shared_mem_size)
        ||  cudaSuccess != cudaFuncSetAttribute( 
            _spmmBlockwiseKernel<KernelImpl>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, 100)) {
            cudaError_t err = cudaGetLastError();
            std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err);
            this->_state.initSuccess = false;
        }
    }

    // calculate launch configuration
    int gdimX = KernelImpl::GridMappingXYToMN ?
                (spmat.nrow / KernelImpl::block_M) : (CEIL(N, KernelImpl::block_N));
    int gdimY = KernelImpl::GridMappingXYToMN ? 
                (CEIL(N, KernelImpl::block_N)) : (spmat.nrow / KernelImpl::block_M);
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
    int NStage
>
void SpmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::initialize(
    BlockwiseSpMatrix<half> &spmat, int N, half *B, half *C, half *D) {
    this->initialize(spmat, N, B, 1.0f, C, 1.0f, D);
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage
>
void SpmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::initialize(
    BlockwiseSpMatrix<half> &spmat, int N, half *B, half *D) {
    this->initialize(spmat, N, B, 1.0f, nullptr, 0.0f, D);
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape, 
    typename MmaShape,
    // pipeline config
    int NStage
>
void SpmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::operator()(
    cudaStream_t stream)
{
    _spmmBlockwiseKernel<KernelImpl><<<this->_state.gridDim, this->_state.blockDim,
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
SpmmBlockwiseOpState SpmmBlockwiseInitFn(BlockwiseSpMatrix<half> &spmat, int N, half *B, 
half *D)
{
    SpmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage> op;
    op.initialize(spmat, N, B, D);
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
void SpmmBlockwiseExecFn(SpmmBlockwiseOpState &state, cudaStream_t stream = NULL)
{
    using KernelImpl = typename SpmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::KernelImpl;
    _spmmBlockwiseKernel<KernelImpl><<<state.gridDim, state.blockDim,
        state.shared_mem_size, stream>>>(state.args);
}

// signature of blockSpmmInit(...)
typedef SpmmBlockwiseOpState (*SpmmBlockwiseInitFn_t) (BlockwiseSpMatrix<half>&, int, half*, half*);

// signature of blockSpmmRun(...)
typedef void (*SpmmBlockwiseExecFn_t) (SpmmBlockwiseOpState&, cudaStream_t);

