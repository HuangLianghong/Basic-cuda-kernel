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
const int Length = 128*1024*1024;

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
    printf("Host recduce sum start...\n");
    
    for(int i = 0; i<Length; ++i){
        sum += A[i];
    }
    printf("reduceSumOnHost done!\n");
    *C = sum;
}


__global__ void reduceSumKernel_baseline(float *C, float *A){
    float sum = 0;
    for(int i = 0; i<Length;++i){
        sum += A[i];
    }
    
    C[0] = sum;
}

__global__ void reduceSumKernel_Twopass(float* C, float*A){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreadNum = gridDim.x*blockDim.x;

    float sum = 0;
    for(int i = idx; i<Length; i+=totalThreadNum){
        sum += A[i];
    }
    __shared__ float sharedSum[128];
    sharedSum[threadIdx.x] = sum;
    __syncthreads();
    float parSum = 0;
    if(threadIdx.x == 0){
        for(int i = 0; i<blockDim.x;++i){
            parSum += sharedSum[i]; 
        }
        atomicAdd(&C[0],parSum);
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
        printf("Arrays do not match.\n\n");
}

void reduction(float *h_A){
    cudaEvent_t start,stop;
    float elapsedTime;
    float *d_A;

    float *d_C1,*d_C2,*d_C3, *d_C4;

    int sizeA = Length * sizeof(float);

    int size = sizeof(float);

    float *h_C1 = (float*)malloc(size);
    float *h_C2 = (float*)malloc(size);
    float *h_C3 = (float*)malloc(size);
    

    CHECK( cudaMalloc((void**)&d_A,sizeA) );

    CHECK( cudaMalloc((void**)&d_C1,size) );
    CHECK( cudaMalloc((void**)&d_C2,size) );
    CHECK( cudaMalloc((void**)&d_C3,size) );

    CHECK( cudaMemcpy(d_A,h_A,sizeA,cudaMemcpyHostToDevice) );
    h_C2[0] = 0;
    CHECK( cudaMemcpy(d_C2,h_C2,size,cudaMemcpyHostToDevice) );

    HLH_Timer Timer;

    printf("Kernel 1 start... (Serial version)\n");
    Timer.record_start();
    reduceSumKernel_baseline<<<1,1>>>(d_C1, d_A);
    Timer.record_end();

    cout<<"Kernel 1 duration time:"<< Timer.print_time()<<endl;

    

    printf("Kernel 2 start... (Tow pass version)\n");
    Timer.record_start();
    reduceSumKernel_baseline<<<32,128>>>(d_C2, d_A);
    Timer.record_end();

    cout<<"Kernel 2 duration time:"<< Timer.print_time()<<endl;





    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_C1, d_C1,size,cudaMemcpyDeviceToHost));
    printf("sum=%f\n",h_C1[0]);
    CHECK(cudaMemcpy(h_C2, d_C2,size,cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(h_C3, d_C3,size,cudaMemcpyDeviceToHost));
    
    
    float *h_C = (float*)malloc(size);
    
    reduceSumOnHost(h_C, h_A);
    checkResult(h_C,h_C1);
    checkResult(h_C,h_C2);
    

    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
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
        h_A[i] = ((((float)rand() / (float)(RAND_MAX)) * 10));
        // h_A[i] = 2;
		
	}
    
    printf(" Starting...\n");
    reduction(h_A);
    
}