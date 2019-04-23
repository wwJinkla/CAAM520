#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// TO compile: 
//    nvcc -o hw04 hw04.c -lm

// TO run with tolerance 1e-4 and 4x4 loop currents
//    ./hw04 4 1e-4

#define PI 3.14159265359
#define MAX(a,b) (((a)>(b))?(a):(b))
#define p_Nthreads 32
#define SERIAL false

// kernel for one iteration of Jacobi
__global__ void Jacobi(float* unew_c, float* u_c, float* f_c, int N){
	float invD = 1./4.0f;  // factor of h cancels out

	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int j = blockIdx.y*blockDim.y + threadIdx.y;
	const int id = i + j*(N+2); // x-index first

	if (i >= 1 && j >= 1 && i <= N && j <= N){
		const float Ru = -u_c[id-(N+2)]-u_c[id+(N+2)]
			-u_c[id-1]-u_c[id+1];
		const float rhs = invD*(f_c[id]-Ru);
		unew_c[id] = rhs;
	}
}


// sequential addressing reduction kernel. Modified from class notes  
__global__ void reduce2(int N, float *x1, float *x2, float *xout){

  __shared__ float s_x[p_Nthreads];

  const int tid = threadIdx.x;
  const int i = blockIdx.x*blockDim.x + tid;

  // load smem
  s_x[tid] = 0;

  if (i < N){
    s_x[tid] = (x1[i] - x2[i])*(x1[i] - x2[i]);
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

// compute square of residual
float residual_square(float* x1, float* x2, int N){
	float res2 = 0.0f; 
	float *xout_c;

	int Nthreads = p_Nthreads; 
	int Nblocks = ((N+2)*(N+2)+Nthreads-1)/Nthreads; 
	dim3 threadsPerBlock(Nthreads,1,1);  
	dim3 blocks(Nblocks,1,1);

	float *xout = (float*)malloc(Nblocks*sizeof(float));
	cudaMalloc(&xout_c, Nblocks*sizeof(float));
	reduce2 <<< blocks, threadsPerBlock >>> ((N+2)*(N+2), x1, x2, xout_c);
	cudaMemcpy(xout, xout_c, Nblocks*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < Nblocks; i++){
		res2 += xout[i];
	}
	cudaFree(xout_c);

	return res2; 
}

// parallely solve for solution vector u
int parallel_solve(const int N, const float tol, float * u, float * f){

  float *unew = (float*)calloc((N+2)*(N+2),sizeof(float));

  float res2 = 1.0f;
  unsigned int iter = 0;

  // allocate CUDA global memory and copy from host to device 
  float *unew_c, *u_c, *f_c;
  cudaMalloc(&unew_c, (N+2)*(N+2)*sizeof(float));
  cudaMalloc(&u_c, (N+2)*(N+2)*sizeof(float));
  cudaMalloc(&f_c, (N+2)*(N+2)*sizeof(float));

  cudaMemcpy(unew_c, unew, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(u_c, u, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(f_c, f, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);

  int Nthreads = p_Nthreads;
  int Nblocks = ((N+2)+Nthreads-1)/Nthreads;
  dim3 threadsPerBlock(Nthreads,Nthreads,1);  
  dim3 blocks(Nblocks,Nblocks,1);

  while(res2>tol*tol){
 	//    update interior nodes using Jacobi 
    Jacobi <<<blocks, threadsPerBlock>>> (unew_c, u_c, f_c, N); 
		
		cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));		
		
    // compute residual using reduction
    res2 = residual_square(unew_c,u_c,N);
    
	// update u on device
    float* pingPong = u_c;
    u_c = unew_c;
    unew_c = pingPong;

    ++iter;
    if(!(iter%500)){
      printf("at iter %d: residual = %g\n", iter, sqrt(res2));
    }
  }

  // copy from device back to host
  cudaError_t cudaStatus = cudaMemcpy(u, u_c, (N+2)*(N+2)*sizeof(float), 
  	cudaMemcpyDeviceToHost);

  cudaFree(unew_c);
  cudaFree(u_c);
  cudaFree(f_c);

  return iter;
}


// serialy solve for solution vector u
int serial_solve(const int N, const float tol, float * u, float * f){

  float *unew = (float*)calloc((N+2)*(N+2),sizeof(float));
  
  //float w = 1.0f; // not used for Pure Jacobi
  float invD = 1./4.f;  // factor of h cancels out

  float res2 = 1.0f;
  unsigned int iter = 0;
  while(res2>tol*tol){

    res2 = 0.0f;

    // update interior nodes using (Pure) Jacobi
    for(int i=1; i<=N; ++i){
      for(int j=1; j<=N; ++j){
	
	const int id = i + j*(N+2); // x-index first
	const float Ru = -u[id-(N+2)]-u[id+(N+2)]-u[id-1]-u[id+1];
	const float rhs = invD*(f[id]-Ru);
	const float oldu = u[id];
	// const float newu = w*rhs + (1.0-w)*oldu;
	const float newu = rhs;

	// compute residual 
	res2 += (newu-oldu)*(newu-oldu);
	unew[id] = newu;
      }
    }

    for (int i = 0; i < (N+2)*(N+2); ++i){
      u[i] = unew[i];
    }
    
    ++iter;
    if(!(iter%500)){
      printf("at iter %d: residual = %g\n", iter, sqrt(res2));
    }
  }

  return iter;
}

int main(int argc, char **argv){
  
  if(argc!=3){
    printf("Usage: ./main N tol\n");
    exit(-1);
  }
  
  int N = atoi(argv[1]);
  float tol = atof(argv[2]);

  float *u = (float*) calloc((N+2)*(N+2), sizeof(float));
  float *f = (float*) calloc((N+2)*(N+2), sizeof(float));
  float h = 2.0/(N+1);
  for (int i = 0; i < N+2; ++i){
    for (int j = 0; j < N+2; ++j){
      const float x = -1.0 + i*h;
      const float y = -1.0 + j*h;
      f[i + j*(N+2)] = sin(PI*x)*sin(PI*y) * h*h;
    }
  }

  int iter;
  if(!SERIAL){
  	iter = parallel_solve(N, tol, u, f);
  }else{
  	iter = serial_solve(N, tol, u, f);
  } 

  float err = 0.0f;
  for (int i = 0; i < (N+2)*(N+2); ++i){
    err = MAX(err,fabs(u[i] - f[i]/(h*h*2.0*PI*PI)));
  }
  
  printf("Iters: %d\n", iter);
  printf("Max error: %lg\n", err);
  
  free(u);
  free(f);  
}
  
