NVCC = nvcc
NVCC_INC = -I..
NVCC_FLAG = -O2 \
  -arch=sm_$(GPU_CC) \
  -DGPU_CC=$(GPU_CC) \
  -std=c++17

TARGETS =\
  spmm.blockwise.out spconv.blockwise.out load_bsr.spmm.blockwise.out

OBJS =\
  spmm.blockwise.o spconv.blockwise.o load_bsr.spmm.blockwise.o

all: $(TARGETS)

spmm.blockwise.out: spmm.blockwise.o ../block_sparse/util/random_mask.cc.o
	$(NVCC) $(NVCC_INC) $^ -o $@

spconv.blockwise.out: spconv.blockwise.o ../block_sparse/util/random_mask.cc.o
	$(NVCC) $(NVCC_INC) $^ -o $@

load_bsr.spmm.blockwise.out: load_bsr.spmm.blockwise.o ../block_sparse/util/random_mask.cc.o
	$(NVCC) $(NVCC_INC) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCC_INC) $(NVCC_FLAG) -c $< -o $@

../block_sparse/util/random_mask.cc.o: ../block_sparse/util/random_mask.cc
	$(MAKE) util/random_mask.cc.o -C ../block_sparse GPU_CC=$(GPU_CC)
	
clean:
	rm -f $(TARGETS) $(OBJS)