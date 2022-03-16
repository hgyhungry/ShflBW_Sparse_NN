#! /bin/bash

mkdir -p result
log_file=result/gemm.block_sparse.csv

echo "Benchmark.gemm.block_sparse"

shapes="1024,960,1024 \
4096,960,1024 \
1024,960,4096 \
1024,1056,1024 \
4096,1056,1024 \
1024,1056,4096 \
33712,1056,1024 \
4096,6400,1024 \
4096,6400,2048 \
4096,128,1024 \
4096,640,1024 \
4096,640,2048 \
1024,640,1024 \
1024,32000,1024 \
32320,640,1024 \
4096,3200,1024 \
4096,3200,2048 \
4096,64,1024 \
4096,320,1024 \
4096,320,2048 \
1024,320,1024 \
1024,16000,1024 \
32320,320,1024 \
1024,1024,1024 \
2048,2048,2048 \
4096,4096,4096 \
1024,128,1024 \
2048,128,2048 \
4096,128,4096"

for shape in $shapes; do
IFS=","; set -- $shape
m=$1; n=$2; k=$3
for d in 0.02 0.05 0.1 0.15 0.2 0.25 0.5; do
for block_sz in 32 64 128; do
# block_sparse/benchmark/benchmark.spmm.out --m $m --n $n --k $k --d $d --sparsity-type block --block-size $block_sz >> $log_file
block_sparse/benchmark/benchmark.spmm.out --m $m --n $n --k $k --d $d --sparsity-type block --block-size $block_sz --row-permute >> $log_file
done; done; done