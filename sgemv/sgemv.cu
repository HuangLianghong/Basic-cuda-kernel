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

// matrix: M*N, vect: N*1, output: M*1
template <typename T>  
__global__ void sgemvKernel_N32(T* matrix, T* vect, T* output, size_t M, size_t N) {
    // N为32的倍数
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    const int warp_size = 32;
    int laneId=tx%warp_size;
    //blockDim.x要等于N时才不会计算错误
    int current_row = blockDim.y*bx+ty;

    if(current_row < M){
        T res = 0;

        int kIteration = N/warp_size;
        if(kIteration == 0) kIteration=1;
        #pragma unroll
        for(int i = 0;i<kIteration;i++){
            int current_col = i*warp_size+laneId;
            res += matrix[current_row*N+current_col]*vect[current_col];
        }
        res = warpReduce<T,warp_size>(res);

        if(laneId==0)output[current_row]=res;
    }



}

int main() {
    // 设置输入数据大小
    const int M = 128;
    const int N = 32;
    // 分配和初始化输入数据
    float* h_matrix = (float*)malloc(M*N*sizeof(float));
    float* h_vect = (float*)malloc(N*sizeof(float));
    float* h_output = (float*)malloc(M*sizeof(float));
    
    // Init matrix
    for (int i = 0; i <M*N ; i++) {
        // Randomly generate letters from a to z
		// h_input[i] =((((float)rand() / (float)(RAND_MAX))));
        h_matrix[i] = i;
	}

    // Init vect
    for (int i = 0; i <N ; i++) {
        // Randomly generate letters from a to z
		// h_input[i] =((((float)rand() / (float)(RAND_MAX))));
        h_vect[i] = 1;
	}


    // 分配和初始化device数据
    float* d_matrix, *d_vect, *d_output;
    CHECK(cudaMalloc(&d_matrix, M*N*sizeof(float)));
    CHECK(cudaMalloc(&d_vect, N*sizeof(float)));
    CHECK(cudaMalloc(&d_output,M*sizeof(float)))

    CHECK(cudaMemcpy(d_matrix, h_matrix, M*N*sizeof(float),cudaMemcpyHostToDevice) );
    CHECK(cudaMemcpy(d_vect, h_vect, N*sizeof(float),cudaMemcpyHostToDevice) );

    // 调用reduce kernel
    dim3 dimBlock(32,4);
    sgemvKernel_N32<float><<<M/4, dimBlock>>>(d_matrix, d_vect, d_output, M, N);
    // check kernel error
    CHECK(cudaGetLastError());
    // 等待CUDA kernel完成
    CHECK(cudaDeviceSynchronize());

    // 从设备上复制结果到主机
    CHECK(cudaMemcpy(h_output, d_output, M*sizeof(float), cudaMemcpyDeviceToHost));

    // 打印最终结果
    for(int i = 0; i<M; ++i)
        printf("Result: %f\n ", h_output[i]);

    // 释放内存
    cudaFree(d_matrix);
    cudaFree(d_vect);
    cudaFree(d_output);

    return 0;
}