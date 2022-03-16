Block Sparse
===

This project provides GPU kernels for sparse neural network inference on Tensor Cores. Specifically, our kernels assume that activations are dense, and parameters are pruned into a special pattern that can be permuted into block-wise-sparse. The following figure shows this sparsity pattern. For more details, you can refer to our [DAC'22 paper](https://arxiv.org/abs/2203.05016).

![Shfl-BW: a Sparse Weight Pattern](docs/format.pdf)

# Environment
Prerequisite
- NVIDIA GPU equipped with Tensor Core (Volta and later architecture)
- Cuda toolkit (tested on v11)

If you can use docker, we have tested and recommend using `nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04`.

# How to build


```bash
cd this_repo/block_sparse
# for V100
make GPU_CC=70
# For other architecture, change 70 to its compute capability, e.g. 75 for T4 and 80 for A100
```
Building will generate `libblock_sparse.a` under `block_sparse/`. 

# How to use
Folder `example/` contains two examples, for sparse GEMM and sparse convolution respectively. 

Compilation needs to link the generated `libblock_sparse.a`, and you can refer to `example/Makefile` for details.

# How to benchmark
```bash
# Example for testing a GEMM (AxB, A is 4096x4096 10% density, B is 4096x4096 dense, sparsity type is Shfl-BW block-size = 64)
block_sparse/benchmark/benchmark.spmm.out --m 4096 --n 4096 --k 4096 --sparsity-type block --block-size 64 --d 0.1 --row-permute
```
```bash
# Example for testing a Conv2D (Input feature: Batch/H/W/InChannel=16/14/14/512, kernel: OutChannel/H/W = 2048/3/3, stride=1, filter sparsity type is Shfl-BW block-size = 64)
block_sparse/benchmark/benchmark.spconv.out --B 16 --H 14 --W 14 --C 512 --F 2048 --R 3 --S 3 --stride 1 --sparsity-type block --block-size 64 --d 0.1 --row-permute
```

# Reproduce paper results
TBD

# Code design
TBD

# Cite
If you use our code, please cite (for now the arxiv version)
```
@article{huang2022shfl,
  title={Shfl-BW: Accelerating Deep Neural Network Inference with Tensor-Core Aware Weight Pruning},
  author={Huang, Guyue and Li, Haoran and Qin, Minghai and Sun, Fei and Din, Yufei and Xie, Yuan},
  journal={arXiv preprint arXiv:2203.05016},
  year={2022}
}
```
