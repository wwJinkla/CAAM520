#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
// TO compile: 
//    gcc -O3 -o hw01 hw01.c -lm

// TO run with tolerance 1e-4 and 4x4 loop currents
//    ./hw01 4 1e-4

#define PI 3.14159265359
#define MAX(a,b) (((a)>(b))?(a):(b))

// solve for solution vector u
int solve(const int rank, const int size, const int local_row_size,
	const int N, const double tol, double * u, double * f){

	/*
		Ghost nodes are stored on 
				u[0,...,N+1] 
		and 
				u[(local_row_size-1)*(N+2),...,(local_row_size)*(N+2)-1]
		Note that for the bottom and top, the exterior nodes will be 
		ghost nodes. 
	*/
	double *unew = (double*)calloc((N+2)*(local_row_size),sizeof(double));
	
	double w = 1.0;
	double invD = 1./4.;  // factor of h cancels out

	double global_res2 = 1.0;
	unsigned int iter = 0;
	MPI_Status status;
	while(global_res2>tol*tol){

		double local_res2 = 0.0;
		for (int i=0; i<= N+1; ++i){
			/*Send up unless at the top, then receive from below*/
			if (rank < size - 1)
				MPI_Send(&u[i + (local_row_size-2)*(N+2)], 1, MPI_DOUBLE, 
					rank + 1, 0, MPI_COMM_WORLD);
			if (rank > 0)
				MPI_Recv(&u[i], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
			/*Send down unless at the bottom, then receive from above*/
			if (rank > 0)
				MPI_Send(&u[i + 1*(N+2)], 1, MPI_DOUBLE, 
					rank-1, 1, MPI_COMM_WORLD);
			if (rank < size - 1)
				MPI_Recv(&u[i + (local_row_size-1)*(N+2)], 1, MPI_DOUBLE,
					rank + 1, 1, MPI_COMM_WORLD, &status);
		}

		// update interior nodes using Jacobi; does not touch ghost nodes
		// int global_id;
		for(int i=1; i<=N; ++i){
			for(int j=1; j<=local_row_size-2; ++j){
			const int id = i + j*(N+2); // x-index first
			const double Ru = -u[id-(N+2)]-u[id+(N+2)]-u[id-1]-u[id+1];
			const double rhs = invD*(f[id]-Ru);
			const double oldu = u[id];
			const double newu = w*rhs + (1.0-w)*oldu;
			local_res2 += (newu-oldu)*(newu-oldu);
			unew[id] = newu;
			}
		}

		for (int i = 0; i < (N+2)*(local_row_size); ++i){
			u[i] = unew[i];
		} 
		
		++iter;
		
		// calcualte global residual
		MPI_Allreduce(&local_res2, &global_res2, 1, MPI_DOUBLE, MPI_SUM,
			MPI_COMM_WORLD);
		
		if((rank == 0) & !(iter%500)){
			printf("at iter %d: residual = %g\n", iter, sqrt(global_res2));
		}
	}

	return iter;
}

int main(int argc, char **argv){

	MPI_Init( &argc, &argv);

	if(argc!=3){
		printf("Usage: ./main N tol\n");
		exit(-1);
	}

	// intialize 
	int N = atoi(argv[1]);
	double tol = atof(argv[2]);
	int rank, size;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank);
	MPI_Comm_size( MPI_COMM_WORLD, &size);

	/* 
		Design decision: allocate extra N%size rows to processor 0,
		and N/size + 2 to all other processes. 
	*/  
	int local_row_size = N/size + 2;
	if (rank == 0){
		local_row_size += N%size;
	}

	double *local_u = (double*) calloc((local_row_size)*(N+2),sizeof(double));
	double *f = (double*) calloc((local_row_size)*(N+2), sizeof(double));
	double h = 2.0/(N+1);

	for (int i = 0; i < N+2; ++i){
		for (int j = 0; j <
		  local_row_size; ++j){
			const double x = -1.0 + i*h;
			double y = -1.0 + j*h;
			if (rank > 0)
				y += (rank - 1)*(N/size)*h + (N/size + N%size)*h;
			f[i + j*(N+2)] = sin(PI*x)*sin(PI*y) * h*h;
		}
	}
	
	// Solve the linear system use (parallel) weighted Jacobi. Time
	// the computation time
  double start = MPI_Wtime();
	int iter = solve(rank, size, local_row_size, N, tol, local_u, f);
	double elapsed = (MPI_Wtime() - start)/((double) size);

	// TODO: assemble global_u here on rank 0; 
	// This probebaly involves some pointer arithmetics.
	// Also printf on rank 0;
	double local_err = 0.0;
	for(int i=1; i<=N; ++i){
		for(int j=1; j<=local_row_size-2; ++j){
			const int id = i + j*(N+2);
			local_err = MAX(local_err,
				fabs(local_u[id] - f[id]/(h*h*2.0*PI*PI)));
		}
	}

	double global_err;
	MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 
			0, MPI_COMM_WORLD);

	double total_elapsed;
  MPI_Reduce(&elapsed, &total_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0){
		printf("Iters: %d\n", iter);
		printf("Max error: %lg\n", global_err);
		printf("Total elapsed time = %g\n", total_elapsed );

	}

	// printf("Iters: %d\n", iter);
	// printf("Max error: %lg\n", err);
	// printf("Memory usage: %lg GB\n", (N+2)*(N+2)*sizeof(double)/1.e9);
	
	MPI_Finalize();
	//free(local_u);
	//free(f);  

}
	
