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

  int N;

//#if 0
  // 1) MPI_Allreduce: reduce and resend
  int x_send = rank;
  int x_reduce;
  MPI_Allreduce(&x_send, &x_reduce, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  printf("MPI_Allreduce result on rank %d: %d\n", rank, x_reduce);
//#endif

#if 0
  // 2) MPI_Allgather
  N = 2;
  int * message = (int*) calloc(N,sizeof(int));
  for (int i = 0; i < N; ++i){
    message[i] = 2*rank + i;
  }

  int * x_gather = (int *) calloc(N*size, sizeof(int));
  MPI_Allgather(message, N, MPI_INT, x_gather, N, MPI_INT, MPI_COMM_WORLD);
  for (int r = 0; r < size; ++r){
    if (rank==r){
      for (int i = 0; i < N*size; ++i){
	printf("MPI_Allgather result[%d] on rank %d: %d\n",i, rank, x_gather[i]);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
  }
  free(message);
  free(x_gather);
#endif
  
#if 0
  // 3) MPI_Alltoall
  N = 1;
  int * alltoallmsg = (int *) calloc(N*size,sizeof(int));
  int * alltoallrecv = (int *) calloc(N*size,sizeof(int));
  for (int i = 0; i < N*size; ++i){
    alltoallmsg[i] = i + rank*N*size;
  }

  for (int r = 0; r < size; ++r){
    if (rank==r){
      printf("On rank %d: MPI_alltoall message = [",rank);
      for (int i = 0; i < N*size; ++i){
	printf("%d ",alltoallmsg[i]);
      }
      printf("]\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Alltoall(alltoallmsg, N, MPI_INT, alltoallrecv, N, MPI_INT, MPI_COMM_WORLD);

  if (rank==0){
    printf("\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);  
  
  for (int r = 0; r < size; ++r){
    if (rank==r){
      printf("On rank %d: MPI_alltoall received = [",rank);
      for (int i = 0; i < N*size; ++i){
	printf("%d ",alltoallrecv[i]);
      }
      printf("]\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  free(alltoallmsg);
  free(alltoallrecv);
#endif
  
  MPI_Finalize();
}
