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
const int Length = 4096*1024;
const int THREAD_PER_BLOCK=128;
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
void elementwiseOnHost(float* C, float* A, float* B)
{
    //printf("Host recduce sum start...\n");
    for(int i = 0; i<Length; ++i){
        C[i] = A[i] + B[i];
    }
}


__global__ void elementwiseKernel_normal(float* C, float* A, float* B){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx<Length){
        C[idx]=A[idx]+B[idx];
    }
}

// Here we fetch a float2 each time, and retrive the first float2 from the original address
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
__global__ void elementwiseKernel_float2(float* C, float* A, float* B){
    int idx = (threadIdx.x+blockIdx.x*blockDim.x)*2;
    if(idx<Length){
        float2 reg_a = FETCH_FLOAT2(A[idx]);
        float2 reg_b = FETCH_FLOAT2(B[idx]);
        float2 reg_c;
        reg_c.x = reg_a.x+reg_b.x;
        reg_c.y = reg_a.y+reg_b.y;
        FETCH_FLOAT2(C[idx]) = reg_c;
    }
}

// Here we fetch a float4 each time, and retrive the first float4 from the original address
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
__global__ void elementwiseKernel_float4(float* C, float* A, float* B){
    int idx = (threadIdx.x+blockIdx.x*blockDim.x)*4;
    if(idx<Length){
        float4 reg_a = FETCH_FLOAT4(A[idx]);
        float4 reg_b = FETCH_FLOAT4(B[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x+reg_b.x;
        reg_c.y = reg_a.y+reg_b.y;
        reg_c.z = reg_a.z+reg_b.z;
        reg_c.w = reg_a.w+reg_b.w;
        FETCH_FLOAT4(C[idx]) = reg_c;
    }
}



void checkResult(float *hostRef, float *gpuRef)
{
    double epsilon = 1.0E-2;
    bool match = 1;
    printf("Input array size=%i\n",Length);

    for (int i = 0; i < Length; i++)
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


void elementwise(float *h_A, float *h_B){
    cudaEvent_t start,stop;
    float elapsedTime;
    float *d_A1, *d_B1, *d_A2, *d_B2, *d_A3, *d_B3;

    float *d_C1,*d_C2,*d_C3;


    int size = sizeof(float)*Length;


    float *h_C1 = (float*)malloc(size);
    float *h_C2 = (float*)malloc(size);
    float *h_C3 = (float*)malloc(size);
    

    CHECK( cudaMalloc((void**)&d_A1,size) );
    CHECK( cudaMalloc((void**)&d_B1,size) );
    CHECK( cudaMalloc((void**)&d_A2,size) );
    CHECK( cudaMalloc((void**)&d_B2,size) );
    CHECK( cudaMalloc((void**)&d_A3,size) );
    CHECK( cudaMalloc((void**)&d_B3,size) );

    CHECK( cudaMalloc((void**)&d_C1,size) );
    CHECK( cudaMalloc((void**)&d_C2,size) );
    CHECK( cudaMalloc((void**)&d_C3,size) );

    CHECK( cudaMemcpy(d_A1,h_A,size,cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_B1,h_B,size,cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_A2,h_A,size,cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_B2,h_B,size,cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_A3,h_A,size,cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_B3,h_B,size,cudaMemcpyHostToDevice) );

    HLH_Timer Timer;
    printf("Kernel 1 start... (normal version)\n");
    Timer.record_start();
    elementwiseKernel_normal<<<Length/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_C1, d_A1, d_B1);
    Timer.record_end();
    cout<<"Kernel 1 duration time:"<< Timer.print_time()<<endl;
    cudaFree(d_A1);
    cudaFree(d_B1);
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    // printf("Kernel 1.5 start... (normal version)\n");
    // Timer.record_start();
    // elementwiseKernel_normal<<<Length/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_C1, d_A2, d_B2);
    // Timer.record_end();
    // cout<<"Kernel 1.5 duration time:"<< Timer.print_time()<<endl;
    // CHECK(cudaDeviceSynchronize());z
    //  // check kernel error
    // CHECK(cudaGetLastError());
    
    // Too slow
    printf("Kernel 2 start... (float2 version)\n");
    Timer.record_start();
    elementwiseKernel_float2<<<Length/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_C2, d_A2, d_B2);
    Timer.record_end();
    cout<<"Kernel 2 duration time:"<< Timer.print_time()<<endl;
    cudaFree(d_A2);
    cudaFree(d_B2);
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    printf("Kernel 3 start... (float4 version)\n");
    Timer.record_start();
    elementwiseKernel_float4<<<Length/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_C3, d_A3, d_B3);
    Timer.record_end();
    cout<<"Kernel 3 duration time:"<< Timer.print_time()<<endl;
    cudaFree(d_A3);
    cudaFree(d_B3);
    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_C1, d_C1,size,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C2, d_C2,size,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C3, d_C3,size,cudaMemcpyDeviceToHost));
    
    float *h_C = (float*)malloc(size);
    
    elementwiseOnHost(h_C, h_A, h_B);
    printf("Check 1...\n");
    checkResult(h_C,h_C1);
    printf("Check 2...\n");
    checkResult(h_C,h_C2);
    printf("Check 3...\n");
    checkResult(h_C,h_C3);


    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
    
    
    

}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 1;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    float *h_A = (float*)malloc(Length*sizeof(float));
    float *h_B = (float*)malloc(Length*sizeof(float));
    

    srand(time(NULL));
	for (int i = 0; i < Length ; i++) {
        // Float type may lead to different answer in reduce_sum when using different algorithms.
        h_A[i] = ((((float)rand() / (float)(RAND_MAX))));
        h_B[i] = ((((float)rand() / (float)(RAND_MAX))));
        // h_A[i] = 1.0;
		
	}
    
    printf(" Starting...\n");
    elementwise(h_A,h_B);
    
}