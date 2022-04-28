#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdlib.h>

#define DEBUG 1

#define N 1024

int main(int argc, char *argv[] ) {

    int i, j,k,numprocs,rank;
    int rows;
    float matrix[N][N];
    float vector[N];
    float result[N];
    struct timeval  tv1, tv2;

    /* float **localMatrix; */
    /* float *localResult; */

    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    /* Initialize Matrix and Vector */
    if(rank==0){
        if(rank==0){
            for(i=0;i<N;i++) {
                vector[i] = i;
                for(j=0;j<N;j++) {
                    matrix[i][j] = i+j;
                }
            }
        }
    }
    // distribución del vector 
    MPI_Bcast(&vector,N,MPI_FLOAT,0,MPI_COMM_WORLD);

    //Calculo del numero de filas por preceso

    rows = (N+numprocs-1)/numprocs;

    float localMatrix[rows][N];
    /* localMatrix = (float**)malloc (sizeof(float)*rows*N); */
    float localResult[rows];
    /* localResult = (float*)malloc (sizeof(float)*rows); */
    
    //Correcion del tamaño de matrix para el scatter
    /* if((rank == 0) && (N%numprocs)) */
    /*     matrix = (float *) realloc(matrix,sizeof(float)*N*numprocs*rows); */
    
    //Scatter de los datos de matrix
    MPI_Scatter(matrix,rows*N,MPI_FLOAT,localMatrix,rows*N,MPI_FLOAT,0,MPI_COMM_WORLD);

    //Correcion del numero de filas en p-1 (si no es multiplo)
    if(rank == numprocs-1)
        rows =  N-rows*(numprocs-1);

    gettimeofday(&tv1, NULL);
    printf("SOY EL PROCESO %d\n",rank);
    //Lazo computacional
    for(i=0;i<rows;i++) {
        localResult[i]=0;
        for(j=0;j<N;j++){
            localResult[i] += localMatrix[i][j]*vector[j];
            printf("Local result %d %lf\n",rank,localResult[i]);
        }
    }
    printf("local vector%d\n", rank);
    for(i=0;i<rows;i++)
        printf(" %lf", localResult[i]);
    printf("\n");
    //Sobre reserva para el vector x en el proceso 0
    /* if(rank==0) */
    /*     localResult = (float*) malloc(sizeof(float)*N*numprocs*rows); */

    MPI_Gather(localResult,rows,MPI_FLOAT,result,rows,MPI_FLOAT,0,MPI_COMM_WORLD);
    
    gettimeofday(&tv2, NULL);
    
    int microseconds = (tv2.tv_usec - tv1.tv_usec)+ 1000000 * (tv2.tv_sec - tv1.tv_sec);
    
    /*Display result */
    if(rank==0 && DEBUG){
        for(i=0;i<N;i++) {
            printf(" %f \t ",result[i]);
        }
        printf("\n");
    } else {
        printf ("Time (seconds) = %lf\n", (double) microseconds/1E6);
    }    
    MPI_Finalize();
    return 0;
}
