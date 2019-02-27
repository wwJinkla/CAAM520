#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

void merge(double * data,double *partnerdata,int N, int rank_high){

  double *tmp = (double*) calloc(2*N,sizeof(double));

  int id1 = 0;
  int id2 = 0;  
  for (int i = 0; i < 2*N; ++i){
    if (id1 < N && id2 < N){
      if (data[id1] < partnerdata[id2]){
	tmp[i] = data[id1];
	++id1;
      }else{
	tmp[i] = partnerdata[id2];
	++id2;
      }
    } else if (id1 < N){
      tmp[i] = data[id1];
      ++id1;
    } else{
      tmp[i] = partnerdata[id2];
      ++id2;
    }
  }

  // pick top N or bottom N elements
  // and put them back into data/partnerdata
  for (int i = 0; i < N; ++i){
    if (rank_high == 0){
      data[i] = tmp[i]; 
    }else{
      data[i] = tmp[i + N];
    }
  }
  free(tmp);
}

int compare(const void *apt, const void *bpt){
  double *a = (double*) apt;
  double *b = (double*) bpt;
  if(*a < *b) return -1;
  if(*a > *b) return +1; 
  return 0;
}

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Status status;
  int tag = 999;
  
  int N = 4;

  /* seed random number to get same result*/
  srand48(183274654*rank);

  /* each process creates a local array */
  double *data = (double*) calloc(N, sizeof(double));
  for(int i =0;i<N;++i){ 
    data[i] = drand48();
  }  

  // use quicksort on first N entries with compare function
  qsort(data, N, sizeof(double), compare);
  
  double *partnerdata = (double*) calloc(N, sizeof(double));
  for (int phase = 0; phase < size; ++phase){

    int partner;
    if (rank %2 == 0){
      if (phase % 2){ // even phase
	partner = rank + 1;
      }else{
	partner = rank - 1;
      }
      
    }else{

      if (phase %2){
	partner = rank - 1;
      }else{
	partner = rank + 1;
      }      
    }

    if (partner != -1 && partner != size){      
      MPI_Sendrecv(data, N, MPI_DOUBLE, partner, tag,
		   partnerdata, N, MPI_DOUBLE, partner, tag,
		   MPI_COMM_WORLD, &status);
      // merge
      merge(data, partnerdata, N, rank > partner);
          
    }      

    
  }

  for (int r = 0; r < size; ++r){
    if (rank == r){
      for (int i = 0; i < N; ++i){
	printf("on rank %d: data[%d] = %f\n",rank,i,data[i]);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
    
  

  free(data);
  free(partnerdata);
  
  MPI_Finalize();
  
  return 0;
}
