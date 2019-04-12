
// for file IO
#include <stdio.h>
#include <stdlib.h> 
#include <sys/stat.h> 

#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.h>
#endif

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

  cl_int            err;

  int plat = 0;
  int dev  = 0;

  cl_context context;
  cl_device_id device;
  cl_command_queue queue;

  cl_kernel kernel;

  oclInit(plat, dev, context, device, queue);

  const char *sourceFileName = "transpose.cl";
  const char *functionName = "transpose";

  int BDIM = 16;
  char flags[BUFSIZ];
  sprintf(flags, "-DBDIM=%d", BDIM);

  oclBuildKernel(sourceFileName, functionName,
		 context, device,
		 kernel, flags);

  // START OF PROBLEM IMPLEMENTATION 
  int N = atoi(argv[argc-1]); // matrix size  

  /* create host array */
  size_t sz = N*N*sizeof(float);

  float *h_A = (float*) malloc(sz);
  float *h_AT = (float*) malloc(sz);
  for(int n=0;n<N*N;++n){
    h_A[n] = n;
    h_AT[n] = -999;
  }

  // create device buffer and copy from host buffer
  cl_mem c_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, h_A, &err);
  cl_mem c_AT = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, h_AT, &err);

  // now set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_A);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_AT);
  
  // set thread array 
  int dim = 2;
  int Nt = BDIM;
  int Ng = Nt*((N+Nt-1)/Nt);
  size_t local_dims[3] = {Nt,Nt,1};
  size_t global_dims[3] = {Ng,Ng,1};

  // queue up kernel 
  clEnqueueNDRangeKernel(queue, kernel, dim, 0, 
			 global_dims, local_dims, 0, (cl_event*)NULL, NULL);

  // blocking read from device to host 
  clFinish(queue);
  
  // blocking read to host 
  clEnqueueReadBuffer(queue, c_AT, CL_TRUE, 0, sz, h_AT, 0, 0, 0);
  
  /* print out results */
  printf("A=\n");
  for(int i=0; i<N; ++i){
    for(int j=0; j < N; ++j){
      printf("%g ", h_A[i+j*N]);
    }
    printf("\n");
  }
  printf("AT=\n");
  for(int i=0; i < N; ++i){
    for(int j=0; j < N; ++j){
      printf("%g ", h_AT[i+j*N]);
    }
    printf("\n");
  }

  exit(0);
  
}
