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
#define BDIM 16

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

  cl_kernel jacobi_kernel;
	cl_kernel reduce_kernel;
	
	oclInit(plat, dev, context, device, queue);
  const char *jacobi_sfn = "jacobi.cl";
	const char *reduce_sfn = "reduce.cl";
  const char *jacobi_functionName = "jacobi";
  const char *reduce_functionName = "reduce2";
	
	char flags[BUFSIZ];
  sprintf(flags, "-DBDIM=%d", BDIM);
	
	oclBuildKernel(jacobi_sfn, jacobi_functionName, 
		 context, device, 
		 jacobi_kernel, flags);
  oclBuildKernel(reduce_sfn, reduce_functionName,
		 context, device,
		 reduce_kernel, flags);
	
	// START OF PROBLEM IMPLEMENTATION
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

	// create device buffer and copy from host buffer
	size_t sz0 = (N+2)*(N+2)*sizeof(float);
	cl_mem c_u = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz0, u, &err);
	cl_mem c_f = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz0, f, &err);
	cl_mem c_unew = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz0, unew, &err);  

  // run kernel, copy result back to CPU
  int Nt = BDIM; // good if it's a multiple of 32, can't have more than 1024
  int Nb = (N + Nt-1)/Nt;
	int Ng = Nt*Nb; 
  size_t local_dims[3] = {Nt,Nt,1};
  size_t global_dims[3] = {Ng,Ng,1};

  // for reduce2 kernel
  int Nt1D = BDIM; 
  int Nb1D = ((N+2)*(N+2) + Nt-1)/Nt; 
	int Ng1D = Nt1D*Nb1D;
  size_t local_dims1D[3] = {Nt1D,1,1};
	size_t global_dims1D[3] = {Ng1D,1,1};

//	int halfNb1D = (Nb1D + 1)/2;
//	int halfNg1D = Nt1D*halfNb1D; 
//	size_t global_dims1D[3] = {halfNg1D,1,1};

  // storage for residual
  float *res = (float*) calloc(Nb1D, sizeof(float));
	size_t sz1 = Nb1D*sizeof(float);
	cl_mem c_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz1, res, &err);

  int iter = 0;
  float r2 = 1.;
	double jacobi_nanoSeconds = 0;
	double reduce_nanoSeconds = 0;
  while (r2 > tol*tol){
		// Run Jacobi 		
		clSetKernelArg(jacobi_kernel, 0, sizeof(int), &N);
		clSetKernelArg(jacobi_kernel, 1, sizeof(cl_mem), &c_u);
		clSetKernelArg(jacobi_kernel, 2, sizeof(cl_mem), &c_f);
		clSetKernelArg(jacobi_kernel, 3, sizeof(cl_mem), &c_unew);
		
  	cl_event event; // cl event for timing
		clEnqueueNDRangeKernel(queue, jacobi_kernel, 2, 0, global_dims, local_dims, 0, (cl_event*)NULL, &event);
		clWaitForEvents(1, &event); // 1 event, pointer to event list  
  	clFinish(queue); // blocking read from device to host
		
  	cl_ulong start,end;
  	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		jacobi_nanoSeconds += end-start;


		//Run reduce
		int N1 = (N+2)*(N+2);
		clSetKernelArg(reduce_kernel, 0, sizeof(int), &N1);
		clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &c_u);
		clSetKernelArg(reduce_kernel, 2, sizeof(cl_mem), &c_unew);
		clSetKernelArg(reduce_kernel, 3, sizeof(cl_mem), &c_res);
		cl_event event1; // cl event for timing
		clEnqueueNDRangeKernel(queue, reduce_kernel, 1, 0, global_dims1D, local_dims1D, 0, (cl_event*)NULL, &event1);		
		clWaitForEvents(1, &event1);
  	clFinish(queue);
		cl_ulong start1,end1;
  	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start1, NULL);
  	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end1, NULL);
		reduce_nanoSeconds += end1-start1;

    // blocking read to host 
		clEnqueueReadBuffer(queue, c_res, CL_TRUE, 0, sz1, res, 0, 0, 0);
    r2 = 0.f;
    for (int j = 0; j < Nb1D; ++j){
      r2 += res[j];
    }

    ++iter;
  }
 
	clEnqueueReadBuffer(queue, c_unew, CL_TRUE, 0, sz0, u, 0, 0, 0);

  float error = 0.0;
  for (int i = 0; i < (N+2)*(N+2); ++i){
    error = MAX(error,fabs(u[i] - f[i]/(h*h*2.0*PI*PI)));
  }
  
  printf("Max error: %g, r2 = %g, iterations = %d\n", error,r2,iter);
  printf("jacobi total execution time is: %0.3f milliseconds \n",jacobi_nanoSeconds / 1e6);
  printf("reduce total execution time is: %0.3f milliseconds \n",reduce_nanoSeconds / 1e6);
}
  
