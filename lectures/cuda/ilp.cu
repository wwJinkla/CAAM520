#include <stdio.h>

#define dfloat float // switch between double/single precision

#define p_iter 50

// C = A*B
__global__ void my_axpy(int N, 
			dfloat * x, 
			dfloat * y, 
			dfloat * z){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  dfloat xval = x[i];
  dfloat yval = y[i];
  dfloat val = 0.f;
  for (int iter = 0; iter < p_iter; ++iter){
    val += xval + yval;
  }
  z[i] = val;
}


#define p_chunk 8

// optimize using const/restrict keywords and instruction level parallelism
__global__ void my_axpy_opt(const int Nchunk, 
			    const dfloat * __restrict__ x, 
			    const dfloat * __restrict__ y, 
			    dfloat * __restrict__ z){
  
  const int i = threadIdx.x + p_chunk * blockIdx.x*blockDim.x;

  dfloat r_x[p_chunk], r_y[p_chunk], r_z[p_chunk]; // local memory
  for (int j = 0; j < p_chunk; ++j){
    const int id = i + j*blockDim.x;
    r_x[j] = x[id];
    r_y[j] = y[id];
    r_z[j] = 0.;
  }

  for (int iter = 0; iter < p_iter; ++iter){
    for (int j = 0; j < p_chunk; ++j){
      r_z[j] += r_x[j] + r_y[j]; 
    }
  }

  for (int j = 0; j < p_chunk; ++j){
    z[i + j*blockDim.x] = r_z[j];
  }
}

int main(void)
{

  int N = 1048576; // 2^20
  dfloat*x = (dfloat*)malloc(N*sizeof(dfloat));
  dfloat*y = (dfloat*)malloc(N*sizeof(dfloat));
  dfloat*z = (dfloat*)malloc(N*sizeof(dfloat));
  dfloat*zero = (dfloat*)malloc(N*sizeof(dfloat));
  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = 42.0 - i;
    z[i] = 0.;
    zero[i] = 0.;
  }

  // note the reference of a *pointer*!
  dfloat *c_x, *c_y, *c_z;
  cudaMalloc(&c_x, N*sizeof(dfloat)); 
  cudaMalloc(&c_y, N*sizeof(dfloat));
  cudaMalloc(&c_z, N*sizeof(dfloat));

  // copy host memory over to GPU 
  cudaMemcpy(c_x, x, N*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(c_y, y, N*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(c_z, zero, N*sizeof(dfloat), cudaMemcpyHostToDevice);

  // add vectors together
  int Nthreads = 32; // good if it's a multiple of 32, can't have more than 1024
  int Nblocks = (N+Nthreads-1)/Nthreads; 
  dim3 threadsPerBlock(Nthreads,1,1);  
  dim3 blocks(Nblocks,1,1);
  dim3 chunk_blocks(Nblocks/p_chunk,1,1);  

  // warmup kernel launches
  my_axpy <<< blocks, threadsPerBlock >>> (N, c_x, c_y, c_z);
  my_axpy_opt <<< chunk_blocks, threadsPerBlock >>> (N/p_chunk, c_x, c_y, c_z);
  cudaMemcpy(c_z, zero, N*sizeof(dfloat), cudaMemcpyHostToDevice); // reset z to zero  

  // check results for simple kernel
  my_axpy <<< blocks, threadsPerBlock >>> (N, c_x, c_y, c_z);
  cudaMemcpy(z, c_z, N*sizeof(dfloat), cudaMemcpyDeviceToHost);
  dfloat maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(z[i]-42.0));
  }
  printf("Naive kernel max error: %f\n", maxError);
  printf("\n");

  // run optimized kernel
  cudaMemcpy(c_z, zero, N*sizeof(dfloat), cudaMemcpyHostToDevice); // reset c_z to zero
  my_axpy_opt <<< chunk_blocks, threadsPerBlock >>> (N/p_chunk, c_x, c_y, c_z);

  // check result
  cudaMemcpy(z, c_z, N*sizeof(dfloat), cudaMemcpyDeviceToHost);
  maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(z[i]-42.0));
  }
  printf("Optimized kernel max error: %f\n", maxError);

  // free memory on both CPU and GPU
  cudaFree(c_x);
  cudaFree(c_y);
  cudaFree(c_z);
  free(x);
  free(y);
  free(z);
}
