# Kernel 1
This is the simplest version, each thread loads data of a row of M and a col of N from global memory.

# Kernel 2
Tiling M and N to use shared memory.

For each block, `TILE_WIDTH*TILE_WIDTH` data will be loaded from global memory to shared memory.
Thus we can reduce the global memory accesses by a factor of `TILE_WIDTH` (compared to kernel 1).

# Kernel 3
Compared to the previous method, only half the number of thread blocks are used.
Each tile loads one tile of M and two tile of N into shared memory.
Therefore reduce the global memory accesses by a factor of 2 (compared to kernel 2).

# Compile and run
`nvcc -o GEMM.out GEMM.cu && ./GEMM.out`
# Terminal output
```
hlh@nscc-gz:~/Cuda_pratice/Basic-cuda-kernel/GEMM$ nvcc -o GEMM.out GEMM.cu && ./GEMM.out 
./GEMM.out Starting...
Using Device 5: NVIDIA A100-PCIE-40GB
 Starting...
Kernel 1 start...
Kernel 1 duration time: 0.633344
Kernel 2 start...
Kernel 2 duration time: 0.431104
Kernel 3 start...
Kernel 3 duration time: 0.358400
Host start...
matrixOnHost done!
Matrix size=1048576
Arrays match.

Matrix size=1048576
Arrays match.

Matrix size=1048576
Arrays match.
```
# Performance Breakdown
I profiled these kernels with nsight compute(ncu). Here are some metrics related to the performance differences.
| kernel | Global Load | Shared Load | 
| ------- | ----------- | ----------- |
| kernel_v1 | 67.11 M | 0| 
| kernel_v2 | 2.10 M | 41.94 M| 
| kernel_v3 | 1.57 M | 37.75 M| 

Compared to v1, v2 effectively decreases the load instruction from global memory (~97%).
And v3 further reduces the load both from global memory and shared memory, because once a tile of matrix M is loaded to shared memory, it can be used to calculate two tile of output matrix.
So about a quarter of global load instructions are reduced. 
