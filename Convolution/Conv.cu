#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include "../common.h"
// Consume about 4KB/block shared memory with TILE_WIDTH=32
// Default shared memory configuration size is 16KB/SM (configurable to 164KB)
// If we increase TILE_WIDTH to 64, we need 4x shared memory, which exceed the hardware limitation
const int TILE_WIDTH = 32;
const int MASK_WIDTH = 9;
const int WIDTH = 512;
__constant__ float M[MASK_WIDTH*MASK_WIDTH];
// Sequentially run on CPU to verify whether the kernel function can obtain the right answer.
void convolutionOnHost(float *h_M, float *N, float *P)
{
    
    printf("Host start...\n");
    float *n = N;
    float *m =h_M;
    float *p = P;

    for (int y = 0; y < WIDTH; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            int s_y = y-(MASK_WIDTH)/2;
            int s_x = x-(MASK_WIDTH)/2;
            float value = 0;
            for(int i = 0;i<MASK_WIDTH;++i){
                for(int j = 0; j<MASK_WIDTH;++j){
                    int cur_y = s_y+i;
                    int cur_x = s_x+j;

                    if(cur_y>=0 && cur_y<WIDTH && cur_x>=0 && cur_x<WIDTH){
                        value += N[cur_y*WIDTH+cur_x] * h_M[i*MASK_WIDTH+j];
                    }

                }
            }
            P[y*WIDTH+x] = value;

        }
    }
    printf("convolutionOnHost done!\n");
    return;
}

// Kernel 1: Revise the 1D conv kernel from figure 7.8 to perform 2D conv
//
__global__ void convolutionKernel_v1(float *P,float *N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0;
    int start_x = ix-MASK_WIDTH/2;
    int start_y = iy-MASK_WIDTH/2;
    for(int i = 0;i<MASK_WIDTH;++i){
        for(int j = 0; j<MASK_WIDTH;++j){
            int cur_x = start_x+j;
            int cur_y = start_y+i;
            if(cur_x>=0 && cur_x < WIDTH && cur_y>=0 && cur_y<WIDTH){
                val += N[cur_y*WIDTH+cur_x] * M[i*MASK_WIDTH+j];
            }

        }
    }
    P[iy*WIDTH+ix]= val;

}

// Kernel 2: Revise the 1D conv kernel from figure 7.14 to perform 2D conv

__global__ void convolutionKernel_v2(float *P, float *N){
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    // Index of the output matrix
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    
    // Boundary of this tile
    int this_tile_start_x = blockIdx.x*blockDim.x;
    int next_tile_start_x = (blockIdx.x+1)*blockDim.x;
    int this_tile_start_y = blockIdx.y*blockDim.y;
    int next_tile_start_y = (blockIdx.y+1)*blockDim.y;

    // Start index of this convolution
    int start_x = x-MASK_WIDTH/2;
    int start_y = y-MASK_WIDTH/2;

    ds_N[threadIdx.y][threadIdx.x] = N[y*WIDTH+x];
    __syncthreads(); 
    float val = 0;
    for(int i = 0; i<MASK_WIDTH ;++i){
        for(int j = 0; j<MASK_WIDTH ;++j){
            // Current index of the convolution step in input matrix
            int N_x = start_x+j;
            int N_y = start_y+i;

            if(N_x>=0 && N_x<WIDTH && N_y >=0 && N_y <WIDTH){
                if(N_x >= this_tile_start_x && N_x < next_tile_start_x \
                && N_y >= this_tile_start_y && N_y < next_tile_start_y){
                    // Here we map the thread index of this block to the matrix in shared memory
                    // This step is not very intuitive and requires careful thought
                    val += M[i*MASK_WIDTH+j] * ds_N[threadIdx.y-MASK_WIDTH/2+i][threadIdx.x-MASK_WIDTH/2+j];
                }
                else{
                    val += M[i*MASK_WIDTH+j] * N[N_y*WIDTH+N_x];
                }
            }
        }
    }

    P[y*WIDTH+x] = val; 
}
/*
// Kernel 3: Follow the design of Figure 5.17
// Map one block to one tile of M and two tile of N
__global__ void matrixMultiplyKernel_v3(float *P, float *M, float *N){
      
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
*/
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

void convolution(float *h_P,float *h_M,float *h_N){
    cudaEvent_t start,stop;
    float elapsedTime;
    float *d_N;
    float *d_P1,*d_P2,*d_P3;
    int size_NP = WIDTH * WIDTH * sizeof(float);
    float *h_P1 = (float*)malloc(size_NP);
    float *h_P2 = (float*)malloc(size_NP);
    float *h_P3 = (float*)malloc(size_NP);
    

 
    CHECK( cudaMalloc((void**)&d_N,size_NP) );
    CHECK( cudaMalloc((void**)&d_P1,size_NP) );
    CHECK( cudaMalloc((void**)&d_P2,size_NP) );
    CHECK( cudaMalloc((void**)&d_P3,size_NP) );


    CHECK(cudaMemcpy(d_N, h_N, size_NP,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(M, h_M, MASK_WIDTH*MASK_WIDTH*sizeof(float)));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Kernel 1 start...\n");
    dim3 dimGrid1(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
    dim3 dimBlock1(TILE_WIDTH, TILE_WIDTH);
    cudaEventRecord(start,0);
    convolutionKernel_v1<<<dimGrid1,dimBlock1>>>(d_P1,d_N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 1 duration time: %f\n",elapsedTime);


    printf("Kernel 2 start...\n");
    dim3 dimGrid2(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
    dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH);
    cudaEventRecord(start,0);
    convolutionKernel_v2<<<dimGrid2,dimBlock2>>>(d_P2, d_N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop); 
    printf("Kernel 2 duration time: %f\n",elapsedTime);



    // printf("Kernel 3 start...\n");
    // dim3 dimGrid3(ceil(mSize.x/32.0)/2,ceil(mSize.y/32.0));
    // dim3 dimBlock3(32.0,32.0);
    // cudaEventRecord(start,0);
    // convolutionKernel_v3<<<dimGrid3,dimBlock3>>>(d_P3, d_A, d_B, mSize);
    // cudaEventRecord(stop,0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime,start,stop); 
    // printf("Kernel 3 duration time: %f\n",elapsedTime);


    CHECK(cudaDeviceSynchronize());
     // check kernel error
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_P1, d_P1,size_NP,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_P2, d_P2,size_NP,cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(h_P3, d_P3,size_NP,cudaMemcpyDeviceToHost));
    
    
    
    convolutionOnHost(h_M,h_N,h_P);
    checkResult(h_P,h_P1,WIDTH*WIDTH);
    checkResult(h_P2,h_P,WIDTH*WIDTH);
    // checkResult(h_P3,h_P,mSize.x * mSize.y);

    cudaFree(d_P1);
    cudaFree(d_P2);
    cudaFree(d_P3);
    cudaFree(M);
    cudaFree(d_N);
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 5;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int size_NP = WIDTH * WIDTH * sizeof(float);
    int size_M = MASK_WIDTH * MASK_WIDTH * sizeof(float);


    float *h_P = (float*)malloc(size_NP);
    float *h_M = (float*)malloc(size_M);
    float *h_N = (float*)malloc(size_NP);
    

    srand(time(NULL));
	for (int i = 0; i <WIDTH ; i++) {
		for (int j = 0; j < WIDTH ; j++) {
			h_N[i*WIDTH+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
		}
	}
    for (int i = 0; i <MASK_WIDTH ; i++) {
		for (int j = 0; j < MASK_WIDTH ; j++) {
			h_M[i*MASK_WIDTH+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));

		}
	}
    printf(" Starting...\n");
    convolution(h_P,h_M,h_N);
    
}