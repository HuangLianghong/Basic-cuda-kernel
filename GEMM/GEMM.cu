#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include "../common.h"

const int TILE_WIDTH = 32;
struct matrixSize{
    int x;
    int y;
};
// Sequentially run on CPU to verify whether the kernel function can obtain the right answer.
void multiplyMatrixOnHost(float *A, float *B, float *C, const int nx,
                     const int ny)
{
    
    printf("Host start...\n");float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[iy*nx+ix] = 0;
            for(int i = 0; i<nx;++i){
                ic[iy*nx+ix] += ia[iy*nx+i]*ib[ix+i*nx]; 
            }

        }
    }
    printf("matrixOnHost done!\n");
    return;
}

// Kernel 1: Follow the design of Figure 4.9
// Simple version of Gemm, no tiling
__global__ void matrixMultiplyKernel_v1(float *P, float *M, float *N, struct matrixSize mSize){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0;

    for(int i = 0; i < mSize.x; ++i){
        val += M[iy*mSize.x+i] * N[i*mSize.x+ix];
    }
    P[iy*mSize.x+ix] = val;

}

// Kernel 2: Follow the design of Figure 4.17
// Tiling M and N to utilize shared memory
__global__ void matrixMultiplyKernel_v2(float *P, float *M, float *N, struct matrixSize mSize){
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float val = 0;
    for(int ph = 0; ph < mSize.x / TILE_WIDTH;++ph){
        ds_M[threadIdx.y][threadIdx.x] = M[Row * mSize.x + ph*TILE_WIDTH + threadIdx.x];
        ds_N[threadIdx.y][threadIdx.x] = N[(ph*TILE_WIDTH + threadIdx.y)*mSize.x + Col];
        __syncthreads();
        
        for(int i = 0; i < TILE_WIDTH;++i){
            val += ds_M[threadIdx.y][i] * ds_N[i][threadIdx.x];
        }
        __syncthreads();
    }
    P[Row*mSize.x+Col] = val;
}

// Kernel 3: Follow the design of Figure 5.17
// Map one block to one tile of M and two tile of N
__global__ void matrixMultiplyKernel_v3(float *P, float *M, float *N, struct matrixSize mSize){
      
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH*2];

    //Row and Col is index of output matrix P
    int Row = blockIdx.y*TILE_WIDTH+threadIdx.y;
    int Col = blockIdx.x*TILE_WIDTH+threadIdx.x;
    float Val1 = 0,Val2 = 0;
    for(int ph = 0; ph < mSize.x/TILE_WIDTH;++ph){
        // Load tiled data to ds_M and ds_N
        ds_M[threadIdx.y][threadIdx.x] = M[Row*mSize.x+ph*TILE_WIDTH+threadIdx.x];
        ds_N[threadIdx.y][threadIdx.x] = N[(ph*TILE_WIDTH+threadIdx.y)*mSize.x+Col];
        ds_N[threadIdx.y][threadIdx.x+TILE_WIDTH] = N[(ph*TILE_WIDTH+threadIdx.y)*mSize.x+Col+mSize.x/2];
        // Blocked untill all thread in a block finish fetch data from global memory to shared memory
        __syncthreads();
        for(int i = 0; i < TILE_WIDTH;++i){
            Val1 += ds_M[threadIdx.y][i] * ds_N[i][threadIdx.x];
            Val2 += ds_M[threadIdx.y][i] * ds_N[i][threadIdx.x+TILE_WIDTH];
        }
        __syncthreads();
    }
    P[Row*mSize.x +Col] = Val1;
    P[Row*mSize.x +Col+mSize.x/2] = Val2;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
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
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

void matrixMultiply(float *h_C, float *h_A, float *h_B, struct matrixSize mSize){
    cudaEvent_t start,stop;
    float elapsedTime;
    float *d_A;
    float *d_B;
    float *d_C1,*d_C2,*d_C3;
    int size = mSize.x * mSize.y * sizeof(float);
    float *h_C1 = (float*)malloc(size);
    float *h_C2 = (float*)malloc(size);
    float *h_C3 = (float*)malloc(size);
    

    CHECK( cudaMalloc((void**)&d_A,size) );
    CHECK( cudaMalloc((void**)&d_B,size) );
    CHECK( cudaMalloc((void**)&d_C1,size) );
    CHECK( cudaMalloc((void**)&d_C2,size) );
    CHECK( cudaMalloc((void**)&d_C3,size) );

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Kernel 1 start...\n");
    dim3 dimGrid1(ceil(mSize.x/32.0),ceil(mSize.y/32.0));
    dim3 dimBlock1(32.0,32.0);
    cudaEventRecord(start,0);
    matrixMultiplyKernel_v1<<<dimGrid1,dimBlock1>>>(d_C1, d_A, d_B, mSize);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 1 duration time: %f\n",elapsedTime);


    printf("Kernel 2 start...\n");
    dim3 dimGrid2(ceil(mSize.x/32.0),ceil(mSize.y/32.0));
    dim3 dimBlock2(32.0,32.0);
    cudaEventRecord(start,0);
    matrixMultiplyKernel_v1<<<dimGrid2,dimBlock2>>>(d_C2, d_A, d_B, mSize);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 2 duration time: %f\n",elapsedTime);



    printf("Kernel 3 start...\n");
    dim3 dimGrid3(ceil(mSize.x/32.0)/2,ceil(mSize.y/32.0));
    dim3 dimBlock3(32.0,32.0);
    cudaEventRecord(start,0);
    matrixMultiplyKernel_v3<<<dimGrid3,dimBlock3>>>(d_C3, d_A, d_B, mSize);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 3 duration time: %f\n",elapsedTime);


    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_C1, d_C1,size,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C2, d_C2,size,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C3, d_C3,size,cudaMemcpyDeviceToHost));
    
    
    float *h_C_2 = (float*)malloc(size);
    
    multiplyMatrixOnHost(h_A,h_B,h_C,mSize.x,mSize.y);
    checkResult(h_C1,h_C,mSize.x * mSize.y);
    checkResult(h_C2,h_C,mSize.x * mSize.y);
    checkResult(h_C3,h_C,mSize.x * mSize.y);

    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
    cudaFree(d_A);
    cudaFree(d_B);
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 5;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    struct matrixSize mS = {1024,1024};
    float *h_C = (float*)malloc(mS.x*mS.y*sizeof(float));
    float *h_B = (float*)malloc(mS.x*mS.y*sizeof(float));
    float *h_A = (float*)malloc(mS.x*mS.y*sizeof(float));
    

    srand(time(NULL));
	for (int i = 0; i < mS.x ; i++) {
		for (int j = 0; j < mS.y ; j++) {
			h_B[i*mS.x+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
			h_A[i*mS.x+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
		}
	}
    printf(" Starting...\n");
    matrixMultiply(h_C,h_A,h_B,mS);
    
}