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

  int N = 10; // number of entries
  int *senddata = (int*) calloc(N,sizeof(int));
  int *recvdata = (int*) calloc(N,sizeof(int));
  int tag = 999; 

  for (int i = 0; i < N; ++i){
    senddata[i] = rank;
  }

  int source, dest;  
  if (rank%2==0){ // if rank is even
    dest = rank + 1;
    source = rank + 1;
  }

  if (rank%2==1){ // if rank is odd
    source = rank - 1;
    dest = rank - 1;
  }

  /*
  // blocking version which may deadlock
  MPI_Send(senddata, N, MPI_INT, dest, tag, MPI_COMM_WORLD); 
  MPI_Recv(recvdata, N, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
  */  

  // blocking version which should avoid deadlock
  MPI_Sendrecv(senddata, N, MPI_INT, dest, tag,
	       recvdata, N, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
  
  /*  
  // non-blocking version
  MPI_Request send_request, recv_request;  
  MPI_Isend(senddata, N, MPI_INT, dest, tag, MPI_COMM_WORLD, &send_request); 
  MPI_Irecv(recvdata, N, MPI_INT, source, tag, MPI_COMM_WORLD, &recv_request);
  // MPI_Wait(&send_request, &status); // not necessary here
  MPI_Wait(&recv_request, &status);
  */
  
  char filename[BUFSIZ];
  sprintf(filename, "results%02d.dat", rank);
  FILE *fp = fopen(filename,"w");
  for (int i = 0; i < N; ++i){
    fprintf(fp,"message[%d] = %d\n",i,recvdata[i]);
  }
  fclose(fp);
  

  MPI_Finalize();
}
