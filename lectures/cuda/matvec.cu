#include <stdio.h>

#define p_N 1000
#define p_Nthreads 32

// C = A*B
__global__ void matmult(int N, double *A, double *x, double *b){

  const int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < N){
    double bi = 0.;
    for (int j = 0; j < N; ++j){
      bi += A[i + j*N]*x[j]; 
    }
    b[i] = bi;
  }   
}

// C = A*B
__global__ void matmult_smem(int N, double *A, double *x, double *b){

  __shared__ double s_x[p_N];

  const int tid = threadIdx.x;
  const int i = blockIdx.x*blockDim.x + tid;

  if (i < N){
    s_x[i] = x[i];
  }
  __syncthreads();

  if (i < N){
    double bi = 0.;
    for (int j = 0; j < N; ++j){
      bi += A[i + j*N]*s_x[j]; 
    }
    b[i] = bi;
  }   
}

int main(void)
{

  int N = p_N;  

  double*A = (double*)malloc(N*N*sizeof(double));
  double*x = (double*)malloc(N*sizeof(double));
  double*b = (double*)malloc(N*sizeof(double));
  for (int i = 0; i < N*N; i++) {
    A[i] = 0.;
  }
  for (int i = 0; i < N; i++) {
    x[i] = 1.0;
    A[i + i*N] = 1.0; // make A = Identity
  }

  // note the reference of a *pointer*!
  double *A_c, *x_c, *b_c;
  cudaMalloc(&A_c, N*N*sizeof(double)); 
  cudaMalloc(&x_c, N*N*sizeof(double));
  cudaMalloc(&b_c, N*N*sizeof(double));

  // copB host memorB over to GPU 
  cudaMemcpy(A_c, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_c, x, N*sizeof(double), cudaMemcpyHostToDevice);

  // add vectors together
  int Nthreads = p_Nthreads; // good if it's a multiple of 32, can't have more than 1024
  int Nblocks = (N+Nthreads-1)/Nthreads; 
  dim3 threadsPerBlock(Nthreads,Nthreads,1);  
  dim3 blocks(Nblocks,Nblocks,1);

  cudaError_t flag;

  matmult <<< blocks, threadsPerBlock >>> (N, A_c, x_c, b_c);
  flag = cudaGetLastError();
  if (flag != cudaSuccess)
    printf("CUDA error checking: %s\n",cudaGetErrorString(flag));

  matmult_smem <<< blocks, threadsPerBlock, N*sizeof(double) >>> (N, A_c, x_c, b_c);
  //matmult_smem <<< blocks, threadsPerBlock >>> (N, A_c, x_c, b_c);
  flag = cudaGetLastError();
  if (flag != cudaSuccess)
    printf("CUDA error checking: %s\n",cudaGetErrorString(flag));

  
  // copy result C_c back to CPU (in C)
  cudaMemcpy(b, b_c, N*sizeof(double), cudaMemcpyDeviceToHost);
  
  // check result
  double maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(b[i]-1.0));
  }
  printf("Max error: %f\n", maxError);

}
