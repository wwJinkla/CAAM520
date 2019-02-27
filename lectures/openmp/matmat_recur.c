#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void mult_leaf(double *A, double *Bt, double *C,
	       int r_A, int c_A, int r_B, int c_B, int Nloc, int N){

  for (int i = 0; i < Nloc; ++i){
    for (int j = 0; j < Nloc; ++j){

      const int rowA = r_A+i;
      const int colB = c_B+j;
      
      double cij = C[rowA + colB*N];
      for (int k = 0; k < Nloc; ++k){
	const int colA = c_A+k;
	const int rowB = r_B+k;
	cij += A[colA + rowA*N]*Bt[rowB + colB*N];
      }
      C[rowA + colB*N] = cij;
    }
  }  
}

void mult_recur(double *A, double *Bt, double *C,
		int r_A, int c_A, int r_B, int c_B,
		int Nloc, int N){

  if (Nloc <= 2){

    mult_leaf(A,Bt,C, r_A, c_A, r_B, c_B, Nloc,N);

  }else{

    // [A11 A12]*[B11 B12] = [A11*B11+A12*B21, A11*B12+A12*B22]
    // [A21 A22] [B21 B22]   [A21*B11+A22*B21, A21*B12+A22*B22]

    // A11*B11
    mult_recur(A,Bt,C, r_A, c_A, r_B, c_B, Nloc/2, N);
    // A12*B21
    mult_recur(A,Bt,C, r_A, c_A+Nloc/2, r_B+Nloc/2, c_B, Nloc/2, N);

    // A11*B12
    mult_recur(A,Bt,C, r_A, c_A, r_B, c_B+Nloc/2, Nloc/2, N);
    // A12*B22
    mult_recur(A,Bt,C, r_A, c_A+Nloc/2, r_B+Nloc/2, c_B+Nloc/2, Nloc/2, N);

    // A21*B11
    mult_recur(A,Bt,C, r_A+Nloc/2, c_A, r_B, c_B, Nloc/2, N);
    // A22*B21
    mult_recur(A,Bt,C, r_A+Nloc/2, c_A+Nloc/2, r_B+Nloc/2, c_B, Nloc/2, N);

    // A21*B12
    mult_recur(A,Bt,C, r_A+Nloc/2, c_A, r_B, c_B+Nloc/2, Nloc/2, N);
    // A22*B22
    mult_recur(A,Bt,C, r_A+Nloc/2, c_A+Nloc/2, r_B+Nloc/2, c_B+Nloc/2, Nloc/2, N);

  }
}

// gcc -fopenmp 
int main (int argc, char **argv){

  int size = atoi(argv[argc-1]);
  
  int N = 4;

  // initialize matrices
  double *A = (double*) calloc(N*N,sizeof(double));
  double *Bt = (double*) calloc(N*N,sizeof(double));
  double *C = (double*) calloc(N*N,sizeof(double));
  double *Crecur = (double*) calloc(N*N,sizeof(double));
  for (int i = 0; i < N*N; ++i){ 
    A[i] = 1.0; //drand48(); // row major
    Bt[i] = (double)i; //drand48(); // row major
  }

  double start = omp_get_wtime();
  
  //#pragma omp parallel for num_threads(size)
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      double cij = 0.;
      for (int k = 0; k < N; ++k){	
	cij += A[k + i*N]*Bt[k + j*N];
      }
      C[i+j*N] = cij;
    }
  }
  double elapsed = omp_get_wtime()-start;

  start = omp_get_wtime();
  mult_recur(A,Bt,Crecur,0,0,0,0,N,N);  
  double elapsed_rec = omp_get_wtime()-start;

  double err2 = 0.;
  for (int i = 0; i < N; ++i){    
    for (int j = 0; j < N; ++j){
      double err = (C[i+j*N]-Crecur[i+j*N]);
      err2+=err*err;      
    }
  }
  printf("err2 = %g\n",err2);
  
  printf("matmat time = %g seconds, recursive time = %g\n",elapsed,elapsed_rec);


}
