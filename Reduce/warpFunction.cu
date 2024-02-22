// This example is used for understanding warp primatives


#include <stdio.h>

__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = laneId;
    printf("Before shuffle: Thread %d final value = %d\n", threadIdx.x, value);
    // Use XOR mode to perform butterfly reduction
    // for (int i= warpSize>>1; i>=1; i>>=1){
    //     // value += __shfl_xor_sync(0xffffffff, value, i, 32);
    //     value += __shfl_down_sync(0xffffffff, value, i);
    // }
    
    value += __shfl_xor_sync(0xffffffff, value, 16);
    // "value" now contains the sum across all threads
    printf("After shuffle: Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    warpReduce<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}