#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main (int argc, char **argv){

  int size = atoi(argv[argc-1]);

  int N = 100;
  double *a = (double*) calloc(N,sizeof(double));
  double *b = (double*) calloc(N,sizeof(double));
  double *c = (double*) calloc(N,sizeof(double));  

  // structured block
#pragma omp parallel for num_threads(size) shared(a,b) 
  for (int i = 0; i < N; ++i){
    a[i] = 10./(i+1.);
    b[i] = (i+1.)/10.;
  }

  // need reduction clause to avoid race conditions
  double d = 0.0;
#pragma omp parallel for num_threads(size) shared(a,b) reduction(+:d)
  for (int i = 0; i < N; ++i){
    d += a[i]*b[i];
  }

  printf("d = %f\n",d);
  free(a);
  free(b);    
}
