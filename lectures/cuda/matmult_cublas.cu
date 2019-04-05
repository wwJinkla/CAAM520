#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// nvcc -I./cuda_samples/common/inc matmult_cublas.cu -lcublas

#define dfloat float // switch between double/single precision

// C = A*B
__global__ void matmult(int N, dfloat *A, dfloat *Bt, dfloat *C){

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;  
  if (i < N && j < N){
    dfloat cij = 0.;
    for (int k = 0; k < N; ++k){
      // storing each matrix in column major format 
      // guarantees coalesced memory accesses
      cij += A[i + k*N]*Bt[j + k*N]; 
    }
    C[i + j*N] += cij;
  }   
}

int main(void)
{

  int N = 1024;

  dfloat*A = (dfloat*)malloc(N*N*sizeof(dfloat));
  dfloat*Bt = (dfloat*)malloc(N*N*sizeof(dfloat));
  dfloat*C = (dfloat*)malloc(N*N*sizeof(dfloat));
  for (int i = 0; i < N*N; i++) {
    A[i] = 0.;
    Bt[i] = 42.0;
    C[i] = 0.;  
  }
  for (int i = 0; i < N; i++) {
    A[i + i*N] = 1.0; // make A = Identity
  }

  // note the reference of a *pointer*!
  dfloat *c_A, *c_B, *c_C;
  cudaMalloc(&c_A, N*N*sizeof(dfloat)); 
  cudaMalloc(&c_B, N*N*sizeof(dfloat));
  cudaMalloc(&c_C, N*N*sizeof(dfloat));

  // copy host memory over to GPU 
  cudaMemcpy(c_A, A, N*N*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(c_B, Bt, N*N*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(c_C, C, N*N*sizeof(dfloat), cudaMemcpyHostToDevice);

  // add vectors together
  int Nthreads = 32; // good if it's a multiple of 32, can't have more than 1024
  int Nblocks = (N+Nthreads-1)/Nthreads; 
  dim3 threadsPerBlock(Nthreads,Nthreads,1);  
  dim3 blocks(Nblocks,Nblocks,1);

  // time
  matmult <<< blocks, threadsPerBlock >>> (N, c_A, c_B, c_C);
  
  // cublas - reset c_C = 0
  cudaMemcpy(c_C, C, N*N*sizeof(dfloat), cudaMemcpyHostToDevice);

  // arguments: handle, transpose or not x2, sizes, ax+b parameters, 
  dfloat alpha = 1.0;
  dfloat beta = 1.0;
  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle)); // cublas handle
  checkCudaErrors(cublasSgemm(handle, 
			      CUBLAS_OP_N, CUBLAS_OP_N, // transpose or not
			      N, N, N, // rows of A/C, cols of B/C, cols of A/B
			      &alpha, c_A, N, c_B, N, 
			      &beta, c_C, N));

  // copy result c_C back to CPU (in C)
  cudaMemcpy(C, c_C, N*N*sizeof(dfloat), cudaMemcpyDeviceToHost);

  // check result
  dfloat maxError = 0.0f;
  for (int i = 0; i < N*N; i++){
    maxError = max(maxError, abs(C[i]-42.0));
  }
  printf("Max error: %f\n", maxError);

  // free memory on both CPU and GPU
  cudaFree(c_A);
  cudaFree(c_B);
  cudaFree(c_C);
  free(A);
  free(Bt);
  free(C);
}
