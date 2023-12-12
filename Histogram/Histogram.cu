#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include "../common.h"
// We assume all char in buffer are lowercase letters for simplicity
// Each section of histogram consists of 4 elements, so there are 26/4+1 = 7 sections in total
const int HISTOGRAM_WIDTH = 7;
const int GRID_WIDTH = 128;
const int TILE_WIDTH = 256;
const long WIDTH = GRID_WIDTH*TILE_WIDTH*128; 

// Sequentially run on CPU to verify whether the kernel function can obtain the right answer.
void histogramOnHost(char *buffer, int *histo)
{
    
    printf("Host start...\n");
    char *b = buffer;
    int *h = histo;

    for (int i = 0; i < WIDTH; i++)
    {
        int alphabet_position = b[i]-'a';
        histo[alphabet_position/4]++;
    }
    // for(int i = 0;i< HISTOGRAM_WIDTH;i++){
    //     printf("histo[%d]=%i\n",i,histo[i]);
    // }
    printf("histogramOnHost done!\n");
    return;
}

// Kernel 1: based on Figure 9.3
//
__global__ void histogramKernel_v1(char *buffer, int *histo){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // number of elements of each thread
    int section_size = WIDTH / (gridDim.x * blockDim.x);
    int start_idx = idx*section_size;

    for(int i = 0; i<section_size;++i){
        if(start_idx+i<WIDTH){
            int alphabet_aposition = buffer[start_idx+i]-'a';
            if(0<=alphabet_aposition && alphabet_aposition < 26) atomicAdd(&(histo[alphabet_aposition/4]),1);
        }
    }
}

// Kernel 2: Figure 9.7, memory access optimization
__global__ void histogramKernel_v2(char *buffer, int *histo){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    for(int i = idx;i<WIDTH;i += blockDim.x*gridDim.x){
        int alphabet_aposition = buffer[i]-'a';
        if(0<=alphabet_aposition && alphabet_aposition < 26) atomicAdd(&(histo[alphabet_aposition/4]),1);
    }
}
// Kernel 3: Each block has a private histo_s
// Aggregate these private histo_s to histo in global memory
// Different to  Figure 9.10
__global__ void histogramKernel_v3(char *buffer, int *histo){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int section_size = WIDTH / (gridDim.x * blockDim.x);
    int start_idx = idx*section_size;
    __shared__ int histo_s[HISTOGRAM_WIDTH];

    if(threadIdx.x < HISTOGRAM_WIDTH){
        histo_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int i = 0; i<section_size;++i){
        if(start_idx+i<WIDTH){
            int alphabet_aposition = buffer[start_idx+i]-'a';
            if(0<=alphabet_aposition && alphabet_aposition < 26) atomicAdd(&(histo_s[alphabet_aposition/4]),1);
        }
    }
    __syncthreads();

    if(threadIdx.x < HISTOGRAM_WIDTH){
        atomicAdd(&(histo[threadIdx.x]),histo_s[threadIdx.x]);
    }

}

// Kernel 3: Each block has a private histo_s
// Aggregate these private histo_s to histo in global memory
// Different to  Figure 9.10
__global__ void histogramKernel_v4(char *buffer, int *histo){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ int histo_s[HISTOGRAM_WIDTH];

    if(threadIdx.x < HISTOGRAM_WIDTH){
        histo_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int i = idx; i< WIDTH; i+=blockDim.x*gridDim.x){
        int alphabet_aposition = buffer[i]-'a';
         if(0<=alphabet_aposition && alphabet_aposition < 26) atomicAdd(&(histo_s[alphabet_aposition/4]),1);
    }
    __syncthreads();

    if(threadIdx.x < HISTOGRAM_WIDTH){
        atomicAdd(&(histo[threadIdx.x]),histo_s[threadIdx.x]);
    }

}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
    double epsilon = 1.0E-2;
    bool match = 1;
    printf("Matrix size=%i\n",N);   

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("idx: %i\n", i);
            printf("host %i gpu %i\n", hostRef[i], gpuRef[i]);
            break;
        }
        // else{
        //     printf("idx: %i\n", i);
        //     printf("host %i gpu %i\n", hostRef[i], gpuRef[i]);
        // }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

void histogram(char *h_b,int *h_h){
    cudaEvent_t start,stop;
    float elapsedTime;
    char *d_b;
    int *d_h1,*d_h2,*d_h3,*d_h4;

    int size_b = WIDTH * sizeof(char);
    int size_h = HISTOGRAM_WIDTH * sizeof(int);
    
    // Receive answer from GPU
    int *h_h1 = (int*)malloc(size_h);
    int *h_h2 = (int*)malloc(size_h);
    int *h_h3 = (int*)malloc(size_h);
    int *h_h4 = (int*)malloc(size_h);
    

 
    CHECK( cudaMalloc((void**)&d_b,size_b) );
    CHECK( cudaMalloc((void**)&d_h1,size_h) );
    CHECK( cudaMalloc((void**)&d_h2,size_h) );
    CHECK( cudaMalloc((void**)&d_h3,size_h) );
    CHECK( cudaMalloc((void**)&d_h4,size_h) );

    // Initialize all elements in histogram to 0
    CHECK(cudaMemcpy(d_b, h_b, size_b,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_h1, h_h, size_h,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_h2, h_h, size_h,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_h3, h_h, size_h,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_h4, h_h, size_h,cudaMemcpyHostToDevice));
    

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Kernel 1 start...(discrete memory access)\n");
    cudaEventRecord(start,0);
    histogramKernel_v1<<<GRID_WIDTH,TILE_WIDTH>>>(d_b,d_h1);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 1 duration time: %f ms\n",elapsedTime);


    printf("Kernel 2 start...(consecutive memory access)\n");
    
    cudaEventRecord(start,0);
    histogramKernel_v2<<<GRID_WIDTH,TILE_WIDTH>>>(d_b, d_h2);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 2 duration time: %f ms\n",elapsedTime);



    printf("Kernel 3 start...(privatization to reduce atomAdd contention, discrete memory access)\n");
    cudaEventRecord(start,0);
    histogramKernel_v3<<<GRID_WIDTH,TILE_WIDTH>>>(d_b, d_h3);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 3 duration time: %f ms\n",elapsedTime);

    printf("Kernel 4 start...(privatization to reduce atomAdd contention, consecutive memory access)\n");
    cudaEventRecord(start,0);
    histogramKernel_v4<<<GRID_WIDTH,TILE_WIDTH>>>(d_b, d_h4);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 4 duration time: %f ms\n",elapsedTime);


    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_h1, d_h1,size_h,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_h2, d_h2,size_h,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_h3, d_h3,size_h,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_h4, d_h4,size_h,cudaMemcpyDeviceToHost));
    
    
    
    histogramOnHost(h_b,h_h);
    checkResult(h_h,h_h1, HISTOGRAM_WIDTH);
    checkResult(h_h,h_h2, HISTOGRAM_WIDTH);
    checkResult(h_h,h_h3, HISTOGRAM_WIDTH);
    checkResult(h_h,h_h4, HISTOGRAM_WIDTH);

    cudaFree(d_h1);
    cudaFree(d_h2);
    cudaFree(d_h3);
    cudaFree(d_h4);
    
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 4;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int size_b = WIDTH * sizeof(char);
    int size_h = HISTOGRAM_WIDTH * sizeof(int);


    char *h_b = (char*)malloc(size_b);
    int *h_h = (int*)malloc(size_h);
    
    

    srand(time(NULL));
	for (int i = 0; i <WIDTH ; i++) {
        // Randomly generate letters from a to z
		h_b[i] = (int)rand()%26 + 'a';
        // if(i >WIDTH-20)printf("h_b[%d]=%c\n",i,h_b[i]);   
	}
    for(int i = 0;i<HISTOGRAM_WIDTH;i++){
        h_h[i] = 0;
    }
    
    printf(" Starting...\n");
    histogram(h_b,h_h);
    
}