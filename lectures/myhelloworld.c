
#include <stdio.h>
#include <stdlib.h>

// include headers for MPI definitions
#include <mpi.h>

// purpose: demonstrate 
//          1. sending a message from process 0 to process 2
//          2. receiving a message by process 2 from process 0
// 
//          using MPI functions MPI_Send and 

// main program - we need those input args
int main(int argc, char **argv){

  // This code gets executed by all processes launched by mpiexec.

  // variable to store the MPI rank of this process
  // rank signifies a unique integer identifier (0-indexed)
  int rank;

  // start MPI environment
  MPI_Init(&argc, &argv);

  // find rank (number of this process)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // send the message if you are process 0
  if(rank==0){
    // message
    int N = 1;
    int *message = (int*) calloc(1, sizeof(int));
    int destination = 2;
    int tag = 999;

    message[0] = 42;

    // send message 
    MPI_Send(message, N, MPI_INT, destination, tag, MPI_COMM_WORLD);
  }

  // receive the message if you are process 1
  if(rank==1){
    // message
    int N = 1;
    int *message = (int*) calloc(1, sizeof(int));
    int origin = 0;
    int tag = 999;
    MPI_Status status;

    // receive message 
    MPI_Recv(message, N, MPI_INT, origin, tag, MPI_COMM_WORLD, 
	     &status);

    printf("Process %d got from rank %d the message: %d\n", 
	   rank, origin, message[0]);
  }

  // close down MPI environment
  MPI_Finalize();

}
