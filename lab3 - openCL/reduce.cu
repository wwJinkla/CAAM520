#include <stdio.h>

#define p_Nthreads 128

__global__ void reduce(int N, float *x, float *xout){

  __shared__ float s_x[p_Nthreads];

  const int tid = threadIdx.x;
  const int i = blockIdx.x*blockDim.x + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i];
  }
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2){
    // strided access
    // s = 1 -> [0, 2, 4, 8, ...]
    // s = 2 -> [0, 4, 8, 16, ...]

    if (tid % (2*s)==0){ 
      s_x[tid] += s_x[tid + s];
    }
    __syncthreads();
  }   

  if (tid==0){
    xout[blockIdx.x] = s_x[0];
  }
}


__global__ void reduce1(int N, float *x, float *xout){

  __shared__ float s_x[p_Nthreads];

  const int tid = threadIdx.x;
  const int i = blockIdx.x*blockDim.x + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i];
  }
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2){
    int index = 2*s*tid;
    if (index < blockDim.x){
      s_x[index] += s_x[index+s]; // bank conflicts
    }
    __syncthreads();
  }   

  if (tid==0){
    xout[blockIdx.x] = s_x[0];
  }
}

__global__ void reduce2(int N, float *x, float *xout){

  __shared__ float s_x[p_Nthreads];

  const int tid = threadIdx.x;
  const int i = blockIdx.x*blockDim.x + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i];
  }
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; // fewer bank conflicts
    }
    __syncthreads();
  }   

  if (tid==0){
    xout[blockIdx.x] = s_x[0];
  }
}

// use all threads
__global__ void reduce3(int N, float *x, float *xout){

  __shared__ float s_x[p_Nthreads];

  const int tid = threadIdx.x;
  const int i = blockIdx.x*(2*blockDim.x) + tid;

  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i] + x[i + blockDim.x];
  }
  __syncthreads();
  
  for (unsigned int s = blockDim.x/2; s > 0; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; // no wasted threads on first iteration
    }
    __syncthreads();
  }   

  if (tid==0){
    xout[blockIdx.x] = s_x[0];
  }
}


// use all threads
__global__ void reduce4(int N, float *x, float *xout){

  __shared__ volatile float s_x[p_Nthreads]; // volatile for in-warp smem mods

  const int tid = threadIdx.x;
  const int i = blockIdx.x*(2*blockDim.x) + tid;

  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i] + x[i + blockDim.x];
  }
  __syncthreads();
  
  // stop at s = 64
  for (unsigned int s = blockDim.x/2; s > 32; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; 
    }
    __syncthreads();
  }   

  // manually reduce within a warp
  if (tid < 32){
    s_x[tid] += s_x[tid + 32];
    s_x[tid] += s_x[tid + 16];
    s_x[tid] += s_x[tid + 8];
    s_x[tid] += s_x[tid + 4];
    s_x[tid] += s_x[tid + 2];
    s_x[tid] += s_x[tid + 1];   
  }
  if (tid==0){
    xout[blockIdx.x] = s_x[0];
  }
}



int main(void)
{
  
  int N = 1048576;

  float*x = (float*)malloc(N*sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = 1;
  }

  // alloc GPU mem and copy over
  float *x_c, *xout_c, *xouthalf_c;
  cudaMalloc(&x_c, N*sizeof(float)); 
  cudaMemcpy(x_c, x, N*sizeof(float), cudaMemcpyHostToDevice);

  // run kernel, copy result back to CPU
  int Nthreads = p_Nthreads; // good if it's a multiple of 32, can't have more than 1024
  int Nblocks = (N+Nthreads-1)/Nthreads; 
  dim3 threadsPerBlock(Nthreads,1,1);  
  dim3 blocks(Nblocks,1,1);

  float*xout = (float*)malloc(Nblocks*sizeof(float));
  cudaMalloc(&xout_c, Nblocks*sizeof(float));
  cudaMalloc(&xouthalf_c, Nblocks/2*sizeof(float));

  // version 1: slow
  reduce <<< blocks, threadsPerBlock >>> (N, x_c, xout_c);  
  // version 2: remove % operator
  reduce1 <<< blocks, threadsPerBlock >>> (N, x_c, xout_c);  
  // version 3: reduce bank conflicts
  reduce2 <<< blocks, threadsPerBlock >>> (N, x_c, xout_c);  
  cudaMemcpy(xout, xout_c, Nblocks*sizeof(float), cudaMemcpyDeviceToHost);

  // check result
  int reduction = 0;
  for (int i = 0; i < Nblocks; i++){
    reduction += xout[i];
  }
  printf("error = %d\n",reduction-N);

  // --- the following versions use only 1/2 the number of blocks
  dim3 halfblocks(Nblocks/2,1,1);  
  float*xouthalf = (float*)malloc((Nblocks/2)*sizeof(float));
  cudaMalloc(&xouthalf_c, (Nblocks/2)*sizeof(float));

  // version 4: fewer idle threads
  reduce3 <<< halfblocks, threadsPerBlock >>> (N, x_c, xouthalf_c);
  cudaMemcpy(xouthalf, xouthalf_c, Nblocks/2*sizeof(float), cudaMemcpyDeviceToHost);  

  // version 5: manually unrolled last warp
  reduce4 <<< halfblocks, threadsPerBlock >>> (N, x_c, xouthalf_c);  
  cudaMemcpy(xouthalf, xouthalf_c, Nblocks/2*sizeof(float), cudaMemcpyDeviceToHost);  

  // check result
  reduction = 0;
  for (int i = 0; i < Nblocks/2; i++){
    reduction += xouthalf[i];
  }
  printf("error = %d\n",reduction-N); 
  
}
