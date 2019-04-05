#include <stdio.h>

#define p_Nthreads 32

// on NOTS: you need "module load CUDA" first
// to compile/run: nvcc stencil.cu -o stencil; ./stencil

__global__ void stencil(int N, double *x, double *y){
  
  const int i = blockIdx.x*blockDim.x + threadIdx.x;  

  if (i > 0 & i < N-1){
    y[i] = (x[i-1] + x[i] + x[i+1])/3.;
  }
}

__global__ void stencil_smem(int N, double *x, double *y){

  __shared__ double s_x[p_Nthreads + 2];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x*blockDim.x;
  const int i = bid + tid;

  int ii = tid;
  while ((bid+ii > 0 & bid+ii < N-1) & (ii < p_Nthreads + 2)){
    s_x[ii] = x[bid + ii - 1];
    ii += p_Nthreads;
  }
  __syncthreads();

  if (i > 0 && i < N-1){
    y[i] = (s_x[tid] + s_x[tid+1] + s_x[tid+2])/3.;
  }

}

int main(void)
{

  int N = 10;

  double*x = (double*)malloc(N*sizeof(double));
  double*y = (double*)malloc(N*sizeof(double));
  for (int i = 0; i < N; i++) {
    x[i] = (double) i;
  }

  // alloc and copy host memory over to GPU 
  double *x_c, *y_c;
  cudaMalloc(&x_c, N*sizeof(double)); 
  cudaMalloc(&y_c, N*sizeof(double));
  cudaMemcpy(x_c, x, N*sizeof(double), cudaMemcpyHostToDevice);

  // run kernel
  int Nthreads = p_Nthreads;
  int Nblocks = (N+Nthreads-1)/Nthreads;
  dim3 blocks(Nblocks,1,1);
  dim3 threadsPerBlock(Nthreads,1,1);

  cudaError_t flag;

  stencil <<< blocks, threadsPerBlock >>> (N, x_c, y_c);
  flag = cudaGetLastError();
  if (flag != cudaSuccess)
    printf("CUDA error checking: %s\n",cudaGetErrorString(flag));

  stencil_smem <<< blocks, threadsPerBlock >>> (N, x_c, y_c);
  flag = cudaGetLastError();
  if (flag != cudaSuccess)
    printf("CUDA error checking: %s\n",cudaGetErrorString(flag));

  // get / check result
  cudaMemcpy(y, y_c, N*sizeof(double), cudaMemcpyDeviceToHost);

  double maxError = 0.0f;
  for (int i = 1; i < N-1; ++i){
    maxError = max(maxError, abs(y[i]-i));
  }
  printf("Max error: %f\n", maxError);
  
  // free memory on both CPU and GPU
  cudaFree(x_c);
  cudaFree(y_c);
  free(x);
  free(y);
}
