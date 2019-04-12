__kernel void reduce(int N, __global float *x, __global float *xout){

  __local float s_x[p_Nthreads];

  const int tid = get_global_id(0);
  const int i = get_group_id(0)*get_local_size(0) + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int s = 1; s < get_local_size(0); s *= 2){
    // strided access
    // s = 1 -> [0, 2, 4, 8, ...]
    // s = 2 -> [0, 4, 8, 16, ...]

    if (tid % (2*s)==0){ 
      s_x[tid] += s_x[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }   

  if (tid==0){
    xout[get_group_id(0)] = s_x[0];
  }
}


__kernel void reduce1(int N, float *x, float *xout){

  __local float s_x[p_Nthreads];

  const int tid = get_global_id(0);
  const int i = get_group_id(0)*get_local_size(0) + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int s = 1; s < get_local_size(0); s *= 2){
    int index = 2*s*tid;
    if (index < get_local_size(0)){
      s_x[index] += s_x[index+s]; // bank conflicts
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }   

  if (tid==0){
    xout[get_group_id(0)] = s_x[0];
  }
}

__kernel void reduce2(int N, float *x, float *xout){

  __local float s_x[p_Nthreads];

  const int tid = get_global_id(0);
  const int i = get_group_id(0)*get_local_size(0) + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int s = get_local_size(0)/2; s > 0; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; // fewer bank conflicts
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }   

  if (tid==0){
    xout[get_group_id(0)] = s_x[0];
  }
}

// use all threads
__kernel void reduce3(int N, float *x, float *xout){

  __local float s_x[p_Nthreads];

  const int tid = get_global_id(0);
  const int i = get_group_id(0)*(2*get_local_size(0)) + tid;

  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i] + x[i + get_local_size(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (unsigned int s = get_local_size(0)/2; s > 0; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; // no wasted threads on first iteration
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }   

  if (tid==0){
    xout[get_group_id(0)] = s_x[0];
  }
}


// use all threads
__kernel void reduce4(int N, float *x, float *xout){

  __local volatile float s_x[p_Nthreads]; // volatile for in-warp smem mods

  const int tid = get_global_id(0);
  const int i = get_group_id(0)*(2*get_local_size(0)) + tid;

  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i] + x[i + get_local_size(0)];
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
    xout[get_group_id(0)] = s_x[0];
  }
}


