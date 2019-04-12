
__kernel void foo(int N, __global float *x){

  int id = get_global_id(0);
  
  if(id<N)
    x[id] *= 2;

}
