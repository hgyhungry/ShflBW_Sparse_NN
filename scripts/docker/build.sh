docker run \
--rm -it --gpus all \
-v /path/to/this_repo:/block_sparse \
nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04 \
bash