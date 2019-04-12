// smem CUDA matrix-matrix multiply kernel
__kernel void mxm(int N,
		  __global const float * __restrict__ A ,
		  __global const float * __restrict__ B ,
		  __global float * __restrict__ C){
	   
  const int idx = get_local_id(0) + get_local_size(0)*get_group_id(0);
  const int idy = get_local_id(1) + get_local_size(1)*get_group_id(1);

  __local float s_A[BDIM][BDIM];
  __local float s_B[BDIM][BDIM+1]; // pad for column accesses

  float axb = 0.;
  for(int offset = 0; offset < N; offset+=BDIM){

    // Number amount of memory moved: 3 * N^2 * sizeof(float) (read/write)
    s_A[get_local_id(1)][get_local_id(0)] = A[idy*N + (offset + get_local_id(0))];
    s_B[get_local_id(1)][get_local_id(0)] = B[(offset+get_local_id(1))*N + idx];

    barrier(CLK_LOCAL_MEM_FENCE); // make sure both blocks are loaded 

    for(int j=0;j<BDIM;++j){
      // flops: 2*N^2
      axb += s_A[get_local_id(1)][j] * s_B[j][get_local_id(0)]; 
    }
  }

  C[idx+idy*N] = axb;
  
}

