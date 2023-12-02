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
Kernel 1 duration time: 0.644960  
Kernel 2 start...  
Kernel 2 duration time: 0.601088  
Kernel 3 start...  
Kernel 3 duration time: 0.360448  
Host start...  
matrixOnHost done!  
Matrix size=1048576  
Arrays match.  

Matrix size=1048576  
Arrays match.  

Matrix size=1048576  
Arrays match.
```
