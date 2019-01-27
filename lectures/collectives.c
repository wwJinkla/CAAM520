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

  // 1) barrier - all ranks must get here before proceeding
  for(int r=0; r<size; ++r){
    if(rank==r){
      printf("On rank %d \n", rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  // 2) broadcast from root rank
  int root = 0;
  int N = 1;
  int message = 42+rank;
  MPI_Bcast(&message, N, MPI_INT, root, MPI_COMM_WORLD);
  printf("rank %d got %d from root %d Bcast\n", rank,message,root);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // 3) scatter - several messages to diff ranks sent from root rank
  N = 2;
  root = 0;
  int *rootmessages;
  int *recvmessages = (int*)calloc(N,sizeof(int));
  if (rank==root){
    rootmessages = (int*) calloc(N*size, sizeof(int)); // send to other ranks
    for (int i = 0; i < size; ++i){
      for (int j = 0; j < N; ++j){
	rootmessages[j + i*N] = 42 + i;
      }
    }
  }
  
  // MPI_scatter = multiple sends 
  MPI_Scatter(rootmessages, N, MPI_INT,
	      recvmessages, N, MPI_INT,
	      root, MPI_COMM_WORLD);  

  //  MPI_Barrier(MPI_COMM_WORLD);  
  for (int r = 0; r < size; ++r){
    if (r==rank){
      for (int i = 0; i < N; ++i){
	printf("rank %d got message[%d] = %d from root %d Scatter\n",
	       rank,i,recvmessages[i],root);
      }      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }


  // 4) gather - opposite of scatter. pull multiple messages to one rank
  message = 42 + rank;
  if (rank==root){
    recvmessages = (int*) calloc(size, sizeof(int));
  }
  MPI_Gather(&message,1, MPI_INT, recvmessages, 1, MPI_INT, root,MPI_COMM_WORLD);

  if (rank==root){
    for (int i = 0; i < size; ++i){
      printf("Root recieved message[%d] %d from Gather\n",i,recvmessages[i]);
    }
  }

  
  MPI_Finalize();
}
