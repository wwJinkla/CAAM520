
#include <stdlib.h>
#include <stdio.h>

#include "metis-5.1.0/include/metis.h"

// to compile: gcc -O3 -o metisDriver metisDriver.c -L./metis-5.1.0/build/Linux-x86_64/libmetis -lmetis -lm
// to run: ./metisDriver

int main(int argc, char **argv){

  // set up a grid
  int N = 20, id, sk, nedges;
  idx_t nvtxs = N*N;
  idx_t ncon = 1;
  idx_t *vwgt = NULL;
  idx_t *vsize = NULL;
  idx_t *adjwgt = NULL;
  real_t *tpwgts = NULL;
  real_t *ubvec = NULL;
  idx_t *options = NULL;
  idx_t objval;
  
  idx_t nparts = 5; // will request partitions  

  int n,m;

  // count connections for cartesian network 
  nedges = 0;
  for(n=0;n<N;++n){
    for(m=0;m<N;++m){
      if(m>0) ++nedges;
      if(m<N-1) ++nedges;
      if(n>0) ++nedges;
      if(n<N-1) ++nedges;
    }
  }

  // build connectivity info for metis 
  idx_t * xadj = (idx_t*) calloc(nvtxs+1, sizeof(idx_t));
  idx_t * adjncy = (idx_t*) calloc(nedges, sizeof(idx_t));
  idx_t * part = (idx_t*) calloc(nvtxs, sizeof(idx_t));
  
  // build adjacency for cartesian network 
  nedges = 0;
  id = 0;
  for(n=0;n<N;++n){
    for(m=0;m<N;++m){
      xadj[id] = nedges;
	
      if(m>0)   adjncy[nedges++] = m-1+n*N;
      if(m<N-1) adjncy[nedges++] = m+1+n*N;
      if(n>0)   adjncy[nedges++] = m+(n-1)*N;
      if(n<N-1) adjncy[nedges++] = m+(n+1)*N;

      ++id;
    }
  }
  xadj[id] = nedges;

  // call metis recursive spectral bisection 
  int ret = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, 
				     vwgt, vsize, adjwgt, &nparts, tpwgts,
				     ubvec, options, &objval, part);
  
  // check the output 
  if(ret!=METIS_OK) printf("METIS: NOT OK \n");

  sk = 0;
  for(n=0;n<N;++n){
    for(m=0;m<N;++m){
      printf("%02d ", part[sk++]);
    }
    printf("\n");
  }
}
