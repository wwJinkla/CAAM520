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


  // intialize x on process 0; distribute x to other ranks  
	int root = 0;	
	double * x_j = (double*) malloc(sizeof(double));	
	double * x = (double*) calloc(size,sizeof(double));
	if (rank == root){
  	for (int i = 0; i < size; i++){
			x[i] = (double) i + 1.0;		
		}  
	}
	MPI_Scatter(x, 1, MPI_DOUBLE, 
				x_j, 1, MPI_DOUBLE,
				0, MPI_COMM_WORLD);	

  // store one coulmn of Ax per processor
  double * xjaj = (double*) calloc(size,sizeof(double));  
  for (int i = 0; i < size; ++i){
    xjaj[i] = (double) i + rank + 1.0;
		xjaj[i] *= (*x_j); 
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// perform transpose
	double * tr_xjaj = (double*) calloc(size,sizeof(double));
	MPI_Alltoall(xjaj, 1, MPI_DOUBLE, tr_xjaj, 1, MPI_DOUBLE, MPI_COMM_WORLD);

	// compute summation inside a row
	double xaj = 0.0;
	for (int i = 0; i < size; ++i){
		xaj += tr_xjaj[i];
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// accumulate and print out
	double * b = (double*) calloc(size, sizeof(double));
	if (rank==root){
    b = (double*) calloc(size, sizeof(double));
  }
	MPI_Gather(&xaj, 1, MPI_DOUBLE, b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank==root){
		for (int i = 0; i < size; ++i){
			printf("b[rank = %d] = %f\n", i,b[i]);		
		}	
	}		
  
	
	free(x_j);
  free(x);
  free(xjaj);	
	free(tr_xjaj);
	free(b);

  MPI_Finalize();
	
	return 0;
}
