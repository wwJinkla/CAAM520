#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// full optimized code: compile using 
// gcc -fopenmp -O3 matmat_omp_opt.c -o matmat; ./matmat 4
// 
int main (int argc, char **argv){

  int size = atoi(argv[argc-1]);
  
  int N = 100;
  
  double *A = (double*) calloc(N*N,sizeof(double*));
  double *Btranspose = (double*) calloc(N*N,sizeof(double*));
  double *C = (double*) calloc(N*N,sizeof(double*));
  for (int i = 0; i < N*N; ++i){ 
    A[i] = drand48(); // row major
    Btranspose[i] = drand48(); // row major storage of *transpose*
    C[i] = 0.0;
  }

  double elapsed, start, end;
  start = omp_get_wtime();

  omp_set_nested(1); // spawn nested threads
  
#pragma omp parallel for num_threads(size)
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      double cij = 0.;
      for (int k = 0; k < N; ++k){	
	cij += A[k + i*N]*Btranspose[k + j*N];
      }
      C[i+j*N] = cij;
    }
  }

  
  end = omp_get_wtime();
  elapsed = end-start;
  
  printf("elapsed time = %g seconds\n",elapsed);


}
