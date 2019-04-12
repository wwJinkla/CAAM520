#include <stdio.h>

#define ulNULL ((unsigned long int * ) NULL)

#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.h>
#endif



#define CL_CHECK(_expr)							\
  do {									\
    cl_int _err = _expr;						\
    if (_err == CL_SUCCESS)						\
      break;								\
    fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
    abort();								\
  } while (0)


#define CL_CHECK_ERR(_expr)						\
  ({									\
    cl_int _err = CL_INVALID_VALUE;					\
    typeof(_expr) _ret = _expr;					\
    if (_err != CL_SUCCESS) {						\
      fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
      abort();								\
    }									\
    _ret;								\
  })

void cl_list_all_devices() {
  int i;
  cl_platform_id platforms[100];
  cl_uint       platforms_n = 0;
  int plat;

  // get list of platform IDs (platform == implementation of OpenCL)
  CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));

  // print out information strings for each found platform
  printf("====================================\n");
  printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
  printf("====================================\n\n");

  // print out information for each platform
  for (plat=0; plat<platforms_n; ++plat)
    {
      cl_device_id devices[100];
      cl_uint devices_n = 0;
      char buffer[10240];

      printf("-----------------------------------------------\nPLATFORM: %d\n", plat);
      CL_CHECK(clGetPlatformInfo(platforms[plat], CL_PLATFORM_PROFILE, 10240, buffer, ulNULL));
      printf("PROFILE = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[plat], CL_PLATFORM_VERSION, 10240, buffer, ulNULL));
      printf("VERSION = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[plat], CL_PLATFORM_NAME, 10240, buffer, ulNULL));
      printf("NAME = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[plat], CL_PLATFORM_VENDOR, 10240, buffer, ulNULL));
      printf("VENDOR = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[plat], CL_PLATFORM_EXTENSIONS, 10240, buffer, ulNULL));
      printf("EXTENSIONS = %s\n", buffer);

      // take any device type (could specify CPU or GPU instead of ALL)
      cl_uint dtype = CL_DEVICE_TYPE_ALL;
      CL_CHECK(clGetDeviceIDs(platforms[plat], dtype, 100, devices, &devices_n));

      // print out information for each device on this platform
      for (i=0; i<devices_n; i++)
	{
	  char buffer[10240];
	  cl_uint buf_uint;
	  cl_ulong buf_ulong;
	  printf("***********************************************\nPLATFORM: %d DEVICE: %d\n", plat, i);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, ulNULL));
	  printf("DEVICE_NAME = %s\n", buffer);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, ulNULL));
	  printf("DEVICE_VENDOR = %s\n", buffer);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, ulNULL));
	  printf("DEVICE_VERSION = %s\n", buffer);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, ulNULL));
	  printf("DRIVER_VERSION = %s\n", buffer);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, ulNULL));
	  printf("DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, ulNULL));
	  printf("DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, ulNULL));
	  printf("DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, ulNULL));
	  printf("DEVICE_MAX_WORK_GROUP_SIZE = %llu\n", (unsigned long long)buf_ulong);
	  CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(buf_ulong), &buf_ulong, ulNULL));
	  printf("DEVICE_MAX_MEM_ALLOC_SIZE = %llu\n", (unsigned long long)buf_ulong);

	  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS

	  cl_uint workitem_dims;
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, ulNULL);
	  printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", (unsigned int) workitem_dims);

	  size_t workitem_size[3];
	  clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, ulNULL);
	  printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%d / %d / %d \n",
		 (int)workitem_size[0], (int)workitem_size[1], (int)workitem_size[2]);
	}
      printf("-----------------------------------------------\n\n");
    }
}

int main(int argc, char **argv){

  cl_list_all_devices();

  return 0;
  
}
