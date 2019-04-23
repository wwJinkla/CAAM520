#include <math.h>

// for file IO
#include <stdio.h>
#include <stdlib.h> 
#include <sys/stat.h> 

#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.h>
#endif

#define PI 3.14159265359f
#define MAX(a,b) (((a)>(b))?(a):(b))
#define BDIM 32

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data){
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

void oclInit(int plat, int dev,
	     cl_context &context,
	     cl_device_id &device,
	     cl_command_queue &queue){

  /* set up CL */
  cl_int            err;
  cl_platform_id    platforms[100];
  cl_uint           platforms_n;
  cl_device_id      devices[100];
  cl_uint           devices_n ;

  /* get list of platform IDs (platform == implementation of OpenCL) */
  clGetPlatformIDs(100, platforms, &platforms_n);
  
  if( plat > platforms_n) {
    printf("ERROR: platform %d unavailable \n", plat);
    exit(-1);
  }
  
  // find all available device IDs on chosen platform (could restrict to CPU or GPU)
  cl_uint dtype = CL_DEVICE_TYPE_ALL;
  clGetDeviceIDs( platforms[plat], dtype, 100, devices, &devices_n);
  
  printf("devices_n = %d\n", devices_n);
  
  if(dev>=devices_n){
    printf("invalid device number for this platform\n");
    exit(0);
  }

  // choose user specified device
  device = devices[dev];
  
  // make compute context on device, pass in function pointer for error messaging
  context = clCreateContext((cl_context_properties *)NULL, 1, &device, &pfn_notify, (void*)NULL, &err); 

  // create command queue
  queue   = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); // synchronized execution
}

void oclBuildKernel(const char *sourceFileName,
		    const char *functionName,
		    cl_context &context,
		    cl_device_id &device,
		    cl_kernel &kernel,
		    const char *flags
		    ){

  cl_int            err;

  // read in text from source file
  FILE *fh = fopen(sourceFileName, "r"); // file handle
  if (fh == 0){
    printf("Failed to open: %s\n", sourceFileName);
    throw 1;
  }

  // C function, get stats for source file (just need total size = statbuf.st_size)
  struct stat statbuf; 
  stat(sourceFileName, &statbuf); 

  // read text from source file and add terminator
  char *source = (char *) malloc(statbuf.st_size + 1); // +1 for "\0" at end
  fread(source, statbuf.st_size, 1, fh); // read in 1 string element of size "st_size" from "fh" into "source"
  source[statbuf.st_size] = '\0'; // terminates the string

  // create program from source 
  cl_program program = clCreateProgramWithSource(context,
						 1, // compile 1 kernel
						 (const char **) & source,
						 (size_t*) NULL, // lengths = number of characters in each string. NULL = \0 terminated.
						 &err); 

  if (!program){
    printf("Error: Failed to create compute program!\n");
    throw 1;
  }
    
  // compile and build program 
  err = clBuildProgram(program, 1, &device, flags, (void (*)(cl_program, void*))  NULL, NULL);

  // check for compilation errors 
  char *build_log;
  size_t ret_val_size;
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size); // get size of build log
  
  build_log = (char*) malloc(ret_val_size+1);
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, (size_t*) NULL); // read build log
  
  // to be careful, terminate the build log string with \0
  // there's no information in the reference whether the string is 0 terminated or not 
  build_log[ret_val_size] = '\0';

  // print out compilation log 
  fprintf(stderr, "%s", build_log );

  // create runnable kernel
  kernel = clCreateKernel(program, functionName, &err);
  if (! kernel || err != CL_SUCCESS){
    printf("Error: Failed to create compute kernel!\n");
    throw 1;
  }
}



int main(int argc, char **argv){
   
  int N = atoi(argv[1]);
  float tol = atof(argv[2]);

  float *u = (float*) calloc((N+2)*(N+2), sizeof(float));
  float *unew = (float*)calloc((N+2)*(N+2),sizeof(float));
  float *f = (float*) calloc((N+2)*(N+2), sizeof(float));
  float h = 2.0/(N+1);
  for (int i = 0; i < N+2; ++i){
    for (int j = 0; j < N+2; ++j){
      const float x = -1.0 + i*h;
      const float y = -1.0 + j*h;
      f[i + j*(N+2)] = sin(PI*x)*sin(PI*y) * h*h;
    }
  } 

  // cuda memory for Jacobi variables
  float *c_u, *c_f, *c_unew;
  cudaMalloc(&c_u, (N+2)*(N+2)*sizeof(float));
  cudaMalloc(&c_f, (N+2)*(N+2)*sizeof(float));
  cudaMalloc(&c_unew, (N+2)*(N+2)*sizeof(float));
  cudaMemcpy(c_u,u, (N+2)*(N+2)*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(c_f,f, (N+2)*(N+2)*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(c_unew,unew,(N+2)*(N+2)*sizeof(float),cudaMemcpyHostToDevice);

  // run kernel, copy result back to CPU
  int Nthreads = p_Nthreads; // good if it's a multiple of 32, can't have more than 1024
  int Nblocks = (N + Nthreads-1)/Nthreads; 
  dim3 threadsPerBlock(Nthreads,Nthreads,1);  
  dim3 blocks(Nblocks,Nblocks,1);

  // for reduce kernel
  int Nthreads1D = p_Nthreads; 
  int Nblocks1D = ((N+2)*(N+2) + Nthreads-1)/Nthreads; 
  int halfNblocks1D = (Nblocks1D + 1)/2; 
  dim3 threadsPerBlock1D(Nthreads1D,1,1);  
  dim3 halfblocks1D(halfNblocks1D,1,1);

  // storage for residual
  float *res = (float*) calloc(halfNblocks1D, sizeof(float));
  float *c_res;
  cudaMalloc(&c_res, halfNblocks1D*sizeof(float));

  int iter = 0;
  float r2 = 1.;
  while (r2 > tol*tol){

    jacobi <<< blocks, threadsPerBlock >>> (N, c_u, c_f, c_unew);
    reduce <<< halfblocks1D, threadsPerBlock1D >>> ((N+2)*(N+2), c_u, c_unew, c_res);

    // finish block reduction on CPU
    cudaMemcpy(res,c_res,halfNblocks1D*sizeof(float),cudaMemcpyDeviceToHost);
    r2 = 0.f;
    for (int j = 0; j < halfNblocks1D; ++j){
      r2 += res[j];
    }

    ++iter;
  }
 
  cudaMemcpy(u,c_unew,(N+2)*(N+2)*sizeof(float),cudaMemcpyDeviceToHost);

  float err = 0.0;
  for (int i = 0; i < (N+2)*(N+2); ++i){
    err = MAX(err,fabs(u[i] - f[i]/(h*h*2.0*PI*PI)));
  }
  
  printf("Max error: %f, r2 = %f, iterations = %d\n", err,r2,iter);

}
  
