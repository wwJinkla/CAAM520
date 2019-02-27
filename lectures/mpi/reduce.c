#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

int main(int argc, char **argv){

  // initializes the MPI environment
  MPI_Init(&argc,&argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Status status;  

  int tag = 999;
  int x = rank;
  printf("On rank %d: x = %d\n",rank,x);

  int active = size;
  while (active > 1){

    int half = active/2;

    // do parallel reduction
    if (rank >= half){

      int dest = rank - half;
      MPI_Send(&x,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
      break; // stop running if rank is done sending
      
    } else if (rank < half){

      int source = rank + half; 
      int x_recv;
      MPI_Recv(&x_recv,1,MPI_INT,source,tag,MPI_COMM_WORLD, &status);

      // sum reduction
      x += x_recv;      
    }

    active = half;
    
  }


  // compare result to MPI_Reduce
  int x_send = rank;
  int x_reduce;
  int root = 0;
  MPI_Reduce(&x_send, &x_reduce, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
  
  if (rank==0){
    printf("Sum of values on all ranks = %d\n",x);
    printf("MPI_Reduce result = %d\n",x_reduce);
  }


  MPI_Finalize();
}
