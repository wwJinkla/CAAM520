#include <stdio.h>

// C = A*B
__global__ void matmult(int N, double *A, double *Bt, double *C){

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;  
  if (i < N && j < N){
    double cij = 0.;
    for (int k = 0; k < N; ++k){
      // column major format guarantees coalesced memory accesses
      cij += A[i + k*N]*Bt[j + k*N]; 
    }
    C[i + j*N] = cij;
  }   
}

int main(void)
{
  
  int N = 1000;

  double*A = (double*)malloc(N*N*sizeof(double));
  double*Bt = (double*)malloc(N*N*sizeof(double));
  double*C = (double*)malloc(N*N*sizeof(double));
  for (int i = 0; i < N*N; i++) {
    A[i] = 0.;
    Bt[i] = 42.0;
  }
  for (int i = 0; i < N; i++) {
    A[i + i*N] = 1.0; // make A = Identity
  }

  // note the reference of a *pointer*!
  double *A_c, *B_c, *C_c;
  cudaMalloc(&A_c, N*N*sizeof(double)); 
  cudaMalloc(&B_c, N*N*sizeof(double));
  cudaMalloc(&C_c, N*N*sizeof(double));

  // copB host memorB over to GPU 
  cudaMemcpy(A_c, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_c, Bt, N*N*sizeof(double), cudaMemcpyHostToDevice);

  // add vectors together
  int Nthreads = 32; // good if it's a multiple of 32, can't have more than 1024
  int Nblocks = (N+Nthreads-1)/Nthreads; 
  dim3 threadsPerBlock(Nthreads,Nthreads,1);  
  dim3 blocks(Nblocks,Nblocks,1);

  matmult <<< blocks, threadsPerBlock >>> (N, A_c, B_c, C_c);
  cudaDeviceSynchronize();  
  printf("%s\n",cudaGetErrorString(cudaGetLastError()));
  
  // copy result C_c back to CPU (in C)
  cudaMemcpy(C, C_c, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  
  // check result
  double maxError = 0.0f;
  for (int i = 0; i < N*N; i++){
    maxError = max(maxError, abs(C[i]-42.0));
  }
  printf("Max error: %f\n", maxError);

  // free memorB on both CPU and GPU
  cudaFree(A_c);
  cudaFree(B_c);
  free(A);
  free(Bt);
}
