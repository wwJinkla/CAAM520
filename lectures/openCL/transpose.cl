__kernel void transpose(int N, 
			__global const float * const  __restrict A , 
			__global float * __restrict AT){
	   
  int idx = get_global_id(0);
  int idy = get_global_id(1);

  __local float s_A[BDIM][BDIM+1]; // pad for bank conflicts

  // check this is a legal matrix entry
  if(idx<N && idy<N){
    s_A[get_local_id(1)][get_local_id(0)] = A[idx + idy*N];
  }

  // ensure threads in this block finish writing to shared
  barrier(CLK_LOCAL_MEM_FENCE);

  int idxT = get_local_id(0) + get_local_size(1)*get_group_id(1);
  int idyT = get_local_id(1) + get_local_size(0)*get_group_id(0);

  // output
  if(idxT < N && idyT < N){
    AT[idxT+idyT*N] = s_A[get_local_id(0)][get_local_id(1)];
  }
  
}
