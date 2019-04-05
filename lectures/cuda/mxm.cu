#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

// to compile for a 3.5 capable device (like the titan in bodge):
// nvcc -arch=sm_35 -O3  -o mxm mxm.cu -lm
// 
// to run a partial reduction on a vector of length 8192 :
// ./mxm 8192

// assume going forward 32x32 threads in each thread-block
#define BDIM 32

// naive CUDA mxm kernel
__global__ void mxmV1(int N,
		      const float *  __restrict__  A ,
		      const float *  __restrict__  B ,
		      float * __restrict__  C){
	   
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int idy = threadIdx.y + blockDim.y*blockIdx.y;

  float axb = 0.;
  for(int n=0;n<N;++n){
    axb += A[n + idy*N] * B[idx + n*N];
  }
  C[idx + idy*N] = axb; 
}

// smem CUDA matrix-matrix multiply kernel
__global__ void mxmV2(int N,
		      const float * const  __restrict__  A,
		      const float * const  __restrict__  B,
		      float * __restrict__  C){
	   
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int idy = threadIdx.y + blockDim.y*blockIdx.y;

  __shared__ float s_A[BDIM][BDIM];
  __shared__ float s_B[BDIM][BDIM];

  float axb = 0.;
  for(int offset = 0; offset < N; offset += BDIM){

    // load local block into shared memory
    s_A[threadIdx.y][threadIdx.x] = A[idy*N + (offset + threadIdx.x)];
    s_B[threadIdx.y][threadIdx.x] = B[(offset+threadIdx.y)*N + idx];

    __syncthreads(); // make sure both blocks are loaded 

    for(int j = 0; j < BDIM; ++j){
      axb += s_A[threadIdx.y][j] * s_B[j][threadIdx.x]; // col of A, row of B
    }
  }

  C[idx+idy*N] = axb;
}


// smem  CUDA matrix-matrix multiply kernel
__global__ void mxmV3(int N,
		      const float * __restrict__ A ,
		      const float * __restrict__ B ,
		      float * __restrict__ C){
	   
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int idy = threadIdx.y + blockDim.y*blockIdx.y;

  __shared__ float s_A[BDIM][BDIM];
  __shared__ float s_B[BDIM][BDIM+1]; // pad for column accesses

  float axb = 0.;
  for(int offset = 0; offset < N; offset+=BDIM){
    
    s_A[threadIdx.y][threadIdx.x] = A[idy*N + (offset + threadIdx.x)];
    s_B[threadIdx.y][threadIdx.x] = B[(offset+threadIdx.y)*N + idx];

    __syncthreads(); // make sure both blocks are loaded 

    for(int j=0;j<BDIM;++j){
      axb += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
    }
  }

  C[idx+idy*N] = axb;
  
}


int main(int argc, char **argv){
  
  int N = 1024;

  float *A = (float*) calloc(N*N, sizeof(float));
  float *B = (float*) calloc(N*N, sizeof(float));
  float *C = (float*) calloc(N*N, sizeof(float));

  printf("N=%d\n", N);

  for(int i=0;i<N;++i){
    for(int j=0;j<N;++j){
      A[i+j*N] = 0.;
      if (i==j){
	A[i+j*N] = 1.;
      }
      B[i+j*N] = 1.;
    }
  }

  float *c_A, *c_B, *c_C;

  size_t sz = N*N*sizeof(float);
  cudaMalloc(&c_A, sz);
  cudaMalloc(&c_B, sz);
  cudaMalloc(&c_C, sz);

  cudaMemcpy(c_A, A, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(c_B, B, sz, cudaMemcpyHostToDevice);

  int Nb = (N+BDIM-1)/BDIM;

  dim3 threadsPerBlock(BDIM,BDIM,1);
  dim3 blocks(Nb,Nb,1);

  mxmV1 <<< blocks, threadsPerBlock >>> (N, c_A, c_B, c_C);  
  mxmV2 <<< blocks, threadsPerBlock >>> (N, c_A, c_B, c_C);
  mxmV3 <<< blocks, threadsPerBlock >>> (N, c_A, c_B, c_C);

  cudaMemcpy(C, c_C, sz, cudaMemcpyDeviceToHost);

  float maxerr = 0.;
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      float errij = C[i+j*N]-1.0;
      maxerr += errij*errij;
    }
  }
  printf("err = %f\n",maxerr);

  // --------------------------------------------------------------------------------

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA ERROR: %s\n", 
	    cudaGetErrorString(err));
  }
}
