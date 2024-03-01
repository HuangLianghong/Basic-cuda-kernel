#include <cuda_runtime.h>
#include <stdio.h>
#include "../common.h"

// Warp reduce kernel
template <typename T, const int kWarpSize = warpSize>
__device__ T warpReduce(T data) {
    // 使用递归将warp中的元素逐步减半
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(0xffffffff, data, offset, kWarpSize);
    }
    return data;
}


template <typename T, const int NUM_THREAD=256>  
__global__ void dotProductKernel(T* A, T* B, T* output, size_t N) {

    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    int bx = blockIdx.x;
    const int warp_size = 32;
    int warpId=tx/warp_size;
    int laneId=tx%warp_size;
    constexpr int NUM_WARP=(NUM_THREAD+warp_size-1)/warp_size;
    __shared__ T shared_mem[NUM_WARP];
    
    T prod = tx<N?A[tx]*B[tx]:0;
    prod = warpReduce<T,warp_size>(prod);
    if(laneId==0) {
        shared_mem[warpId]=prod;
    }
    __syncthreads();

    if(laneId < NUM_WARP){
        prod=shared_mem[laneId];
    }
    else{
        prod=0.0f;
    }
    // prod = (laneId<NUM_WARP)?shared_mem[laneId]:0.0f; // maybe is warpId?
    __syncthreads();
    
    prod = warpReduce<T,NUM_WARP>(prod);
    if(threadIdx.x == 0) atomicAdd(output,prod);



}

int main() {
    // 设置输入数据大小
    const int N = 256;
    // 分配和初始化输入数据
    float* h_A = (float*)malloc(N*sizeof(float));
    float* h_B = (float*)malloc(N*sizeof(float));
    float* h_output = (float*)malloc(sizeof(float));
    h_output[0] = 0;
    
    // Init vector A and B
    for (int i = 0; i <N ; i++) {
        // Randomly generate letters from a to z
		// h_input[i] =((((float)rand() / (float)(RAND_MAX))));
        h_A[i] = i;
        h_B[i] = 1.0;
	}



    // 分配和初始化device数据
    float* d_A, *d_B, *d_output;
    CHECK(cudaMalloc(&d_A, N*sizeof(float)));
    CHECK(cudaMalloc(&d_B, N*sizeof(float)));
    CHECK(cudaMalloc(&d_output,sizeof(float)))

    CHECK(cudaMemcpy(d_A, h_A, N*sizeof(float),cudaMemcpyHostToDevice) );
    CHECK(cudaMemcpy(d_B, h_B, N*sizeof(float),cudaMemcpyHostToDevice) );
    CHECK(cudaMemcpy(d_output, h_output, sizeof(float),cudaMemcpyHostToDevice) );

    // 调用reduce kernel
    const int BLOCK_THREAD=128;
    dotProductKernel<float><<<N/BLOCK_THREAD, BLOCK_THREAD>>>(d_A, d_B, d_output,N);
    // check kernel error
    CHECK(cudaGetLastError());
    // 等待CUDA kernel完成
    CHECK(cudaDeviceSynchronize());

    // 从设备上复制结果到主机
    CHECK(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // 打印最终结果
    printf("Result: %f\n ", h_output[0]);

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);

    return 0;
}