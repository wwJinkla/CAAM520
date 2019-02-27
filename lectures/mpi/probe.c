#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

int main(int argc, char **argv){

  int rank, size;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // send data from rank 0
  if(rank==0){
    
    int data = 911;
    int count = 1;
    int tag = 999;
    int dest = 1;
    
    MPI_Send(&data, count, MPI_INT, dest, tag, MPI_COMM_WORLD);
  }


  
  // check for messages
  if(rank==1){

    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);    
    int source = status.MPI_SOURCE;
    int tag    = status.MPI_TAG;
    
    int count;
    MPI_Get_count(&status, MPI_INT, &count);

    printf("Before receiving, rank 1 has determined that the message has count=%d tag=%d source=%d\n",
	   count, tag, source);

    int *indata = (int*) calloc(count, sizeof(int));
    MPI_Recv(indata, count, MPI_INT, source, tag,
	     MPI_COMM_WORLD, &status);

    printf("process 1 received %d\n", indata[0]);
  }


  MPI_Finalize();
  

}
