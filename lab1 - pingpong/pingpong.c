
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
  int tag = 999;
  
  // start MPI environment
  MPI_Init(&argc, &argv);

  // find rank (number of this process)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ping_pong = 0;
  int partner_rank = (rank + 1) % 2;
  while(ping_pong <= 10){
    if (rank == ping_pong % 2){
      ping_pong += 1;
      MPI_Send(&ping_pong, 1, MPI_INT, partner_rank, tag, MPI_COMM_WORLD);
      printf("rank %d incremented and sent ping_pong=%d to rank %d\n",
	     rank, ping_pong, partner_rank);
    } else {
      MPI_Recv(&ping_pong, 1, MPI_INT, partner_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("rank %d received ping_pong=%d from %d\n",
	     rank, ping_pong, partner_rank);
    }    
  }

  // close down MPI environment
  MPI_Finalize();

}
