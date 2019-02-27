#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>


int main(int argc, char **argv){

  int rank, size;

  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* find MPI rank and size */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // seed for random numbers
  srand48(183274654 + rank); 

  // store one entry per processor
  double x_local = (double) rank + 1.0;
  //double x_local = drand48();

  // store one row of A per processor
  double * ai = (double*) calloc(size,sizeof(double));  
  for (int j = 0; j < size; ++j){
    //ai[j] = drand48();
    ai[j] = (double) j + rank + 1.0;
  }

  // gather locally stored pieces to assemble global x
  double * x_global = (double*) calloc(size,sizeof(double));  
  MPI_Allgather(&x_local, 1, MPI_DOUBLE, x_global, 1, MPI_DOUBLE, MPI_COMM_WORLD);

  // compute product
  double bi = 0.0;
  for (int j = 0; j < size; ++j){
    bi += ai[j]*x_global[j];
  }

  printf("b[rank = %d] = %f\n", rank,bi);
  
  free(x_global);
  
  MPI_Finalize();
}
