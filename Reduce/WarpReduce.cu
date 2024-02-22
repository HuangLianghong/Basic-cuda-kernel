#include <cuda_runtime.h>
#include <stdio.h>
#include "../common.h"

// Warp reduce kernel
template <typename T, const int kWarpSize = warpSize>
__device__ T warpReduce(T data) {
    // 使用递归将warp中的元素逐步减半
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(0xffffffff, data, offset, warpSize);
    }
    return data;
}
// 蝶形规约
template <typename T, const int kWarpSize = warpSize>
__device__ T warpReduce_xor(T data){
    for(int offset = kWarpSize>>1; offset>=1; offset>>=1){
        data += __shfl_xor_sync(0xffffffff, data, offset);
    }
    return data;
}

// 主要的reduce kernel
template <typename T, const int NUM_THREADS=128>
__global__ void reduceKernel(T* input, T* output, int size) {
    constexpr int NUM_WARPS = (NUM_THREADS + 32 - 1) / 32;
    int warpid = threadIdx.x / warpSize;
    // Threads within a warp are referred to as lanes
    int laneid = threadIdx.x % warpSize;
    int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    static __shared__ float shared[NUM_WARPS];

    T mySum = (tidx<size)? input[tidx]:0.0f;
   
    // 对每个warp进行reduce sum
    mySum = warpReduce_xor<T,32>(mySum);
    // shared memory中每个元素存储着每个warp的结果
    if(laneid == 0) {
        shared[warpid] = mySum;
        printf("After shuffle: Block %d warp %d final value = %f\n", blockIdx.x,warpid, mySum);
    }
    __syncthreads();

    // 对shared memory中所有元素进行reduce sum得到每个block reduce sum的结果
    mySum = (laneid < NUM_WARPS)?shared[laneid]:0.0f;
    mySum = warpReduce_xor<T,NUM_WARPS>(mySum);

    if(laneid == 0 && warpid ==0){
        printf("After shared memory shuffle: Block %d warp %d final value = %f\n", blockIdx.x,warpid, mySum);
    }


}

int main() {
    // 设置输入数据大小
    const int size = 1024;
    // 分配和初始化输入数据
    float* h_input = (float*)malloc(size*sizeof(float));
    
    for (int i = 0; i <size ; i++) {
        // Randomly generate letters from a to z
		// h_input[i] =((((float)rand() / (float)(RAND_MAX))));
        h_input[i] = i;
         
	}


    // 分配和初始化device数据
    float* d_input, *d_output;
    CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, sizeof(float)));

    CHECK(cudaMemcpy(d_input,h_input,size*sizeof(float),cudaMemcpyHostToDevice) );

    // 调用reduce kernel
    const int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    reduceKernel<float,block_size><<<grid_size, block_size>>>(d_input, d_output, size);
    // check kernel error
    CHECK(cudaGetLastError());
    // 等待CUDA kernel完成
    CHECK(cudaDeviceSynchronize());

    // 从设备上复制结果到主机
    float h_output;
    CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // 打印最终结果
    printf("Result: %f\n", h_output);

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}