__kernel void reduce2(int N, __global float *u, __global float *unew, __global float *res){

  __local float s_x[BDIM];

  const int tid = get_local_id(0);
  const int i = get_group_id(0)*get_local_size(0) + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
		const float unew1 = unew[i];
		const float diff1 = unew1 - u[i];
    s_x[tid] = diff1*diff1;
		
		// update
		u[i] = unew1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int s = get_local_size(0)/2; s > 0; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; // fewer bank conflicts
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }   

  if (tid==0){
    res[get_group_id(0)] = s_x[0];
  }
}

// use all threads
// Note: It only works for N = BDIM*2^n for n = 0,1,2,...
__kernel void reduce3(int N, __global float *u, __global float *unew, __global float *res){

  __local float s_x[BDIM];

  const int tid = get_local_id(0);
  const int i = get_group_id(0)*(2*get_local_size(0)) + tid;

  s_x[tid] = 0;
  if (i < N){
		const float unew1 = unew[i];
		const float unew2 = unew[i + get_local_size(0)];
		const float diff1 = unew1 - u[i];
		const float diff2 = unew2 - u[i + get_local_size(0)];
    s_x[tid] = diff1*diff1 + diff2*diff2;

		// update u
		u[i] = unew1;
		u[i + get_local_size(0)] = unew2; 
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (unsigned int s = get_local_size(0)/2; s > 0; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; // no wasted threads on first iteration
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }   

  if (tid==0){
    res[get_group_id(0)] = s_x[0];
  }
}


// use all threads
// NOTE: this kernel deos not work on Intel's CPUs. It only works for N = BDIM*2^n for n = 0,1,2,...
__kernel void reduce4(int N, __global float *u, __global float *unew, __global float *res){

  __local volatile float s_x[BDIM]; // volatile for in-warp smem mods

  const int tid = get_local_id(0);
  const int i = get_group_id(0)*(2*get_local_size(0)) + tid;

  s_x[tid] = 0;
  if (i < N){
		const float unew1 = unew[i];
		const float unew2 = unew[i + get_local_size(0)];
		const float diff1 = unew1 - u[i];
		const float diff2 = unew2 - u[i + get_local_size(0)];
    s_x[tid] = diff1*diff1 + diff2*diff2;

		// update u
		u[i] = unew1;
		u[i + get_local_size(0)] = unew2;  
	}
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // stop at s = 64
  for (unsigned int s = get_local_size(0)/2; s > 32; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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
    res[get_group_id(0)] = s_x[0];
  }
}


