// REFERENCE: https://zhuanlan.zhihu.com/p/426978026
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <string>
#include <cublas_v2.h>
#include "../common.h"

using namespace::std;
const int Length = 32*1024*1024;
const int THREAD_PER_BLOCK=256;
const int NEW_THREAD_PER_BLOCK=128;

class HLH_Timer{
private:
    cudaEvent_t start;
    cudaEvent_t stop;

public:
    HLH_Timer(){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    void record_start(){
        cudaEventRecord(start,0);
    }
    void record_end(){
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
    }
    string print_time(){
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime,start,stop); 
        return to_string(elapsedTime)+"ms";
    }
    ~HLH_Timer(){
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

// Sequentially run on CPU to verify whether the kernel function can obtain the right answer.
void reduceSumOnHost(float* C, float* A)
{
    float sum = 0;
    //printf("Host recduce sum start...\n");
    for(int i = 0; i<Length; ++i){
        sum += A[i];

    }
    printf("reduceSumOnHost result: %f\n",sum);
    *C = sum;
}


__global__ void reduceSumKernel_serial(float *C, float *A){
    if(blockIdx.x == 0 && threadIdx.x ==0){
        float sum = 0;
        for(int i = 0; i<Length;++i){
            sum += A[i];
        }
        
        C[0] = sum;
    }
}

__global__ void reduceSumKernel_baseline(float* C, float*A){
    int idx = threadIdx.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float shm[THREAD_PER_BLOCK];
    shm[idx] = A[i];
    __syncthreads();

    for(int s = 1; s<blockDim.x; s*=2){
        // each thread points to the same shm[i] statically
        if(idx%(2*s) == 0){
            shm[idx] += shm[idx+s];  
        }
        __syncthreads();
    }
    if(idx == 0){
        C[blockIdx.x] = shm[idx];
    }
}

__global__ void reduceSumKernel_optimize_warp_divergence(float* C, float*A){
    int idx = threadIdx.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float shm[THREAD_PER_BLOCK];
    shm[idx] = A[i];
    __syncthreads();

    for(int s = 1; s<blockDim.x; s*=2){
        // Each thread points to a different shm[i] dynamically
        int index = idx*s*2;
        if(index < blockDim.x){
            shm[index] += shm[index+s];
        }
        __syncthreads();
    }
    if(idx == 0){
        C[blockIdx.x] = shm[idx];
    }
}

__global__ void reduceSumKernel_optimize_bank_conflict(float* C, float*A){
    int idx = threadIdx.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float shm[THREAD_PER_BLOCK];
    shm[idx] = A[i];
    __syncthreads();
    // All partial sum are store in the front of the array, memory addresses are contiguous. 
    for(int s = blockDim.x/2; s>0; s/=2){
        // Each thread points to a different shm[i] dynamically
        if(idx < s){
            shm[idx] += shm[idx+s];
        }
        __syncthreads();
    }
    if(idx == 0){
        C[blockIdx.x] = shm[idx];
    }
}

__global__ void reduceSumKernel_optimize_idle_thread(float* C, float*A){
    int idx = threadIdx.x;
    int i = blockIdx.x*blockDim.x*2+threadIdx.x;
    __shared__ float shm2[THREAD_PER_BLOCK];
    shm2[idx] = A[i]+A[i+blockDim.x];
    __syncthreads();
    // All partial sum are store in the front of the array, memory addresses are contiguous. 
    for(int s = THREAD_PER_BLOCK/2; s>0; s/=2){
        // Each thread points to a different shm[i] dynamically
        if(idx < s){
            shm2[idx] += shm2[idx+s];
        }
        __syncthreads();
    }
    if(idx == 0){
        C[blockIdx.x] = shm2[idx];
    }
}

__device__ void unrollReduce(volatile float* shared_data, int tid){
    shared_data[tid] += shared_data[tid+32];
    shared_data[tid] += shared_data[tid+16];
    shared_data[tid] += shared_data[tid+8];
    shared_data[tid] += shared_data[tid+4];
    shared_data[tid] += shared_data[tid+2];
    shared_data[tid] += shared_data[tid+1];
}
__global__ void reduceSumKernel_unroll(float* C, float*A){
    int idx = threadIdx.x;
    int i = blockIdx.x*blockDim.x*2+threadIdx.x;
    __shared__ float shm2[THREAD_PER_BLOCK];
    shm2[idx] = A[i]+A[i+blockDim.x];
    __syncthreads();
    // All partial sum are store in the front of the array, memory addresses are contiguous. 
    for(int s = blockDim.x/2; s>32; s/=2){
        // Each thread points to a different shm[i] dynamically
        if(idx < s){
            shm2[idx] += shm2[idx+s];
        }
        __syncthreads();
    }
    if(idx < 32){
        unrollReduce(shm2, idx);
    }
    if(idx == 0){
        C[blockIdx.x] = shm2[idx];
    }
}

__global__ void reduceSumKernel_optimize_launch_parameter(float* C, float*A){
    int idx = threadIdx.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float shm2[NEW_THREAD_PER_BLOCK];
    shm2[idx] = A[i]+A[i+blockDim.x];

    __syncthreads();
    // All partial sum are store in the front of the array, memory addresses are contiguous. 
    for(int s = blockDim.x/2; s>32; s/=2){
        // Each thread points to a different shm[i] dynamically
        if(idx < s){
            shm2[idx] += shm2[idx+s];
        }
        __syncthreads();
    }
    if(idx < 32){
        unrollReduce(shm2, idx);
    }
    if(idx == 0){
        C[blockIdx.x] = shm2[idx];
    }
}

void checkResult(float *hostRef, float *gpuRef)
{
    double epsilon = 1.0E-2;
    bool match = 1;
    printf("Input array size=%i\n",Length);

    for (int i = 0; i < 1; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("idx: %i\n", i);
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match (Small differences of float between GPU and CPU answers are OK).\n\n");
}

void arraySum(float* array, int n){
    for(int i = n-1;i>0;--i){
        array[0] += array[i];
    }
}

void reduction(float *h_A){
    cudaEvent_t start,stop;
    float elapsedTime;
    float *d_A;

    float *d_C1,*d_C2,*d_C3, *d_C4, *d_C5, *d_C6 ,*d_C7;

    int sizeA = Length * sizeof(float);

    int size = sizeof(float);
    int sizeC2 = sizeof(float) * (Length/THREAD_PER_BLOCK);
    int sizeC3 = sizeC2;
    int sizeC4 = sizeC2;
    int sizeC5 = sizeC2/2;
    int sizeC6 = sizeC2/2;
    int sizeC7 = sizeC2/2;

    float *h_C1 = (float*)malloc(size);
    float *h_C2 = (float*)malloc(sizeC2);
    float *h_C3 = (float*)malloc(sizeC3);
    float *h_C4 = (float*)malloc(sizeC4);
    float *h_C5 = (float*)malloc(sizeC5);
    float *h_C6 = (float*)malloc(sizeC6);
    float *h_C7 = (float*)malloc(sizeC7);
    

    CHECK( cudaMalloc((void**)&d_A,sizeA) );

    CHECK( cudaMalloc((void**)&d_C1,size) );
    CHECK( cudaMalloc((void**)&d_C2,sizeC2) );
    CHECK( cudaMalloc((void**)&d_C3,sizeC3) );
    CHECK( cudaMalloc((void**)&d_C4,sizeC4) );
    CHECK( cudaMalloc((void**)&d_C5,sizeC5) );
    CHECK( cudaMalloc((void**)&d_C6,sizeC6) );
    CHECK( cudaMalloc((void**)&d_C7,sizeC7) );

    CHECK( cudaMemcpy(d_A,h_A,sizeA,cudaMemcpyHostToDevice) );

    HLH_Timer Timer;

    printf("Kernel 1 start... (Serial version)\n");
    Timer.record_start();
    reduceSumKernel_serial<<<1,1>>>(d_C1, d_A);
    Timer.record_end();
    cout<<"Kernel 1 duration time:"<< Timer.print_time()<<endl;
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());
    
    // Too slow
    printf("Kernel 2 start... (Baseline)\n");
    Timer.record_start();
    reduceSumKernel_baseline<<<Length/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_C2, d_A);
    Timer.record_end();
    cout<<"Kernel 2 duration time:"<< Timer.print_time()<<endl;
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    printf("Kernel 3 start... (No warp divergence)\n");
    Timer.record_start();
    reduceSumKernel_optimize_warp_divergence<<<Length/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_C3, d_A);
    Timer.record_end();
    cout<<"Kernel 3 duration time:"<< Timer.print_time()<<endl;
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    printf("Kernel 4 start... (No bank conflict)\n");
    Timer.record_start();
    reduceSumKernel_optimize_bank_conflict<<<Length/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_C4, d_A);
    Timer.record_end();
    cout<<"Kernel 4 duration time:"<< Timer.print_time()<<endl;
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    printf("Kernel 5 start... (No idle threads)\n");
    Timer.record_start();
    reduceSumKernel_optimize_idle_thread<<<Length/(THREAD_PER_BLOCK*2),THREAD_PER_BLOCK>>>(d_C5, d_A);
    Timer.record_end();
    cout<<"Kernel 5 duration time:"<< Timer.print_time()<<endl;
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    printf("Kernel 6 start... (Unroll reduction)\n");
    Timer.record_start();
    reduceSumKernel_unroll<<<Length/(THREAD_PER_BLOCK*2),THREAD_PER_BLOCK>>>(d_C6, d_A);
    Timer.record_end();
    cout<<"Kernel 6 duration time:"<< Timer.print_time()<<endl;
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    printf("Kernel 7 start... (Unroll reduction)\n");
    Timer.record_start();
    reduceSumKernel_optimize_launch_parameter<<<Length/(NEW_THREAD_PER_BLOCK*2),NEW_THREAD_PER_BLOCK>>>(d_C7, d_A);
    Timer.record_end();
    cout<<"Kernel 7 duration time:"<< Timer.print_time()<<endl;
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_C1, d_C1,size,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C2, d_C2,sizeC2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C3, d_C3,sizeC3,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C4, d_C4,sizeC4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C5, d_C5,sizeC5,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C6, d_C6,sizeC6,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C7, d_C7,sizeC7,cudaMemcpyDeviceToHost));
    
    float *h_C = (float*)malloc(size);
    
    reduceSumOnHost(h_C, h_A);
    printf("Check 1...\n");
    checkResult(h_C,h_C1);
    arraySum(h_C2, Length/THREAD_PER_BLOCK);
    printf("Check 2...\n");
    checkResult(h_C,h_C2);
    arraySum(h_C3, Length/THREAD_PER_BLOCK);
    printf("Check 3...\n");
    checkResult(h_C,h_C3);
    arraySum(h_C4, Length/THREAD_PER_BLOCK);
    printf("Check 4...\n");
    checkResult(h_C,h_C4);
    arraySum(h_C5, Length/(THREAD_PER_BLOCK*2));
    printf("Check 5... \n");
    checkResult(h_C,h_C5);
    arraySum(h_C6, Length/(THREAD_PER_BLOCK*2));
    printf("Check 6... \n");
    checkResult(h_C,h_C6);
    arraySum(h_C7, Length/(NEW_THREAD_PER_BLOCK*2));
    printf("Check 7...\n");
    checkResult(h_C,h_C7);


    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
    cudaFree(d_C4);
    cudaFree(d_C5);
    cudaFree(d_C6);
    cudaFree(d_C7);
    cudaFree(d_A);

}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 4;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    float *h_A = (float*)malloc(Length*sizeof(float));
    

    srand(time(NULL));
	for (int i = 0; i < Length ; i++) {
        // Float type may lead to different answer in reduce_sum when using different algorithms.
        h_A[i] = ((((float)rand() / (float)(RAND_MAX))));
        // h_A[i] = 1.0;
		
	}
    
    printf(" Starting...\n");
    reduction(h_A);
    
}