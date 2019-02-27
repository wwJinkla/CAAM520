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

  // assume matrix is N*size
  int N = atoi(argv[1]);
  
  // store one entry per processor
  double* x_local = (double*) calloc(N,sizeof(double));
  for (int i = 0; i < N; ++i){
    x_local[i] = drand48();
  }
  
  // store N rows of A per processor
  double * a_local = (double*) calloc(N*N*size,sizeof(double));  
  for (int j = 0; j < N*N*size; ++j){
    a_local[j] = drand48();
  }
 
  // gather locally stored pieces to assemble global x
  double * x_global = (double*) calloc(N*size,sizeof(double));  
  double * b_local = (double*) calloc(N,sizeof(double));

  // timing part
  double start = MPI_Wtime();

  int maxit = 5;
  for (int iter = 0; iter < maxit; ++iter){

    MPI_Allgather(x_local, N, MPI_DOUBLE, x_global, N, MPI_DOUBLE, MPI_COMM_WORLD);
    
    // compute product
    for (int i = 0; i < N; ++i){ // rows
      for (int j = 0; j < N*size; ++j){ // cols
	b_local[i] += a_local[i*N + j] * x_global[j];
      }
    }
  }
  
  double elapsed = (MPI_Wtime() - start) / (double) maxit;

  double total_elapsed;
  MPI_Reduce(&elapsed, &total_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank==0){
    printf("Matrix size = %d, number of ranks = %d, total elapsed time = %g\n",N*size,size,total_elapsed);
  }

  /*
  printf("on rank %d:\n",rank);
  for (int i = 0; i < N; ++i){
    printf("b_local[%d] = %f\n", i,b_local[i]);
  }
  */
  
  free(x_global);
  free(a_local);
  free(b_local);
  
  MPI_Finalize();
}
