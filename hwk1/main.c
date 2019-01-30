/*
 * HW1.c
 * 01-28-2019
 * Author: Wei Wu
 * 
 * A matrix free implementation of the 2D explicit finite difference for 
 * the Laplace Equation -(d^2/dx^2 + d^2/dy^2)u = f(x,y) on [-1,1]^2, with
 * Dirichlet BV = 0. 
 * I used Weighted Jacobi method to solve the resulting sparse linear
 * system.   
 * 
 */
#include <stdio.h>
#include <math.h>
#include <time.h>

/* 
 * Compute the sqrt((Au - b)^2). The computation only considers
 * those points on the interior, since the BV = 0 for our problem.
 *
 * @param: u - array of doubles
 * 				 b - array of doubles
 * 				 N - int, length of u (and b)
 * @return: L_2 norm of Au - b
 * 
 */
double L_2_norm(double u[], double b[], int N);

/* 
 * Compute the max of abs(u - u_true). The computation only considers
 * those points on the interior, since the BV = 0 for our problem.
 * 
 * @param: u - array of doubles
 * 				 u_true - array of doubles
 * 				 N - int, length of u (and u_true)
 * @return: L_inf norm of u - u_true
 * 
 */
double L_inf_norm(double u[], double b[], int N);

int main(void)
{
	/* initilization */
	int N;
	double tol;
	printf ("Enter an integer for N: ");
	scanf ("%d", &N);
	printf ("Enter an double for tol: ");
	scanf ("%lf", &tol);	
	
	const double pi = 3.1415926;
	const double h = 2.0/(N+1);
	double w = 1.0/2;
  int size = (N + 2)*(N + 2);
	
	/* initialize arrays b, u, and u_true */
	double u[size];
	double u_true[size];
  double b[size];
	for(int i = 0; i <= (N + 1); i++){
		for(int j = 0; j <= (N + 1); j++){
			
			/* compute coordiantes and index */
			double x = -1 + i*h;
			double y = -1 + j*h;
			int idx = i*(N + 2) + j;
			
			/* initialize b and u_true*/
			b[idx] = h*h*sin(pi*x)*sin(pi*y); // note that I multiply RHS by h^2
			u_true[idx] = 1.0/(2*pi*pi)*sin(pi*x)*sin(pi*y);
			
			/* initialize u to be 1 on the interior, and 0 otherwise */
			u[idx] = (i != 0)*(i != (N+1))*(j != 0)*(j != (N+1));
		}
	}
	
	// weighted Jacobi method for solving Au = b
	double norm = tol + 1; 
	int k = 0;
	double alpha = w/4;
	double beta = (1-w);
	
	clock_t begin = clock();
	while(norm > tol){
		for(int i = 1; i <= N; i++){
			for(int j = 1; j <= N; j++){
				int idx = i*(N + 2) + j; 
				double sigma = u[idx - (N + 2)] + u[idx + (N + 2)]
							+ u[idx - 1] + u[idx + 1];
				u[idx] = alpha*(b[idx] + sigma) + beta*u[idx];
			}
		}
		k++;
		norm = L_2_norm(u ,b, N);
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	
	/* write out result */
   FILE *fp;

   fp = fopen("u.dat", "w+");
	
	for(int i = 0; i <= N+1; i++){
		for(int j = 0; j <= N+1;j++){
			int idx = i*(N + 2) + j;
			fprintf(fp, "u[%d] = %lf \n", idx, u[idx]);
		}
	}
	fclose(fp);

	/* print out program statistics*/
	double max_error = L_inf_norm(u, u_true, N); 
	printf("For N = %d: \n", N);
	printf("Takes %d many iterations to converge to a residual < 1e-6. \n", k);
	printf("Wall time = %lf seconds. \n", time_spent);
	printf("Max error of u  is %f. \n", max_error);
	
	return 0;
}

double L_2_norm(double u[], double b[], int N)
{
	double norm = 0;
	double temp = 0;
	for(int i = 1; i <= N; i++){
		for(int j = 1; j <= N; j++){
			/* compute |A_i*u_i - b_i| */
			int idx = i*(N + 2) + j; 
			temp =  4*u[idx] 
							- u[idx - (N + 2)] - u[idx + (N + 2)]
							- u[idx - 1] - u[idx + 1] 
							- b[idx]; 
			norm += pow(temp,2.0);	 
		}
	}
	
	return sqrt(norm);
}

double L_inf_norm(double u[], double u_true[], int N)
{
	double norm = 0;
	double temp = 0;
	for(int i = 1; i <= N; i++){
		for(int j = 1; j <= N; j++){
			int idx = i*(N + 2) + j; 
			temp = fabs(u[idx] - u_true[idx]);
			if(temp >= norm) 
				norm = temp;	 
		}
	}
	
	return norm;
}






