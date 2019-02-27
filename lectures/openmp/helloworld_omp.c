#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main (int argc, char **argv){

  int size = atoi(argv[argc-1]);

  // omp_set_num_threads(size);
  
  // structured block
#pragma omp parallel num_threads(2)
  {
    int id = omp_get_thread_num();
    int size = omp_get_num_threads();
    printf("Hello world on thread %d out of %d\n",id,size);
  }
}
