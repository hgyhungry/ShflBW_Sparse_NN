We provide the interface to dump and load BlockwiseSpMatrix.

An initialized BlockwiseSpMatrix can be dumped into this format with `.dump(std::ostream &out)` method.
```c++
BlockwiseSpMatrix spmat;
spmat.init_random(...);

spmat.dump(std::cout);// this will dump the serialized version of spmat
// can be redirected to a file for later loading
```
A BlockwiseSpMatrix can load from a previously dumped matrix 
```c++
// define a blockwise sparse matrix
BlockwiseSpMatrix<half> spmat;

// initialize from input stream
spmat.load(std::cin);
```

A simple example to explain the format

```
# matrix
#
# 1 1 0 0
# 2 2 0 0
# 0 0 3 1
# 0 0 1 2

# serialized format
4 # number of rows
4 # number of columns
2 # number of non-zero *blocks*
2 # number of rows in a *block*
2 # number of columns in a *block*. The pattern is 2x2 block-wise
0 # 0 indicates no row shuffling while '1' indicated row shuffling
0 1 2 # offset array of the CSR encoding of block positions
0 1 # index array of the CSR encoding of block positions 
1 2 1 2 3 1 1 2 # values of the block-CSR. Inside each block, column-first
# if there is row shuffling, this row is left for row-permutation ids 
```
In mtx_dump.txt there is a more complex example dumped from the matrix created in spmm.blockwise.cu