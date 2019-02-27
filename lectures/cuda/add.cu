#include <stdio.h>

__global__
void add(int n, float *x, float *y, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    z[i] = x[i] + y[i];
  }
}

int main(void)
{
  int N = 1e3;

  float*x = (float*)malloc(N*sizeof(float));
  float*y = (float*)malloc(N*sizeof(float));
  float*z = (float*)malloc(N*sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = 42-i;
  }

  // note the reference of a *pointer*!
  float *x_c, *y_c, *z_c;
  cudaMalloc(&x_c, N*sizeof(float)); 
  cudaMalloc(&y_c, N*sizeof(float));
  cudaMalloc(&z_c, N*sizeof(float));

  // copy host memory over to GPU 
  cudaMemcpy(x_c, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_c, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // add vectors together
  int Nthreads = 256;
  dim3 blocks((N+Nthreads-1)/Nthreads,1,1);
  dim3 threadsPerBlock(Nthreads,1,1);
  add <<< blocks, threadsPerBlock >>> (N, x_c, y_c, z_c);

  // copy result z_c back to CPU (in z)
  cudaMemcpy(z, z_c, N*sizeof(float), cudaMemcpyDeviceToHost);

  // check result
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(z[i]-42.f));

  printf("Max error: %f\n", maxError);

  // free memory on both CPU and GPU
  cudaFree(x_c);
  cudaFree(y_c);
  free(x);
  free(y);
}
