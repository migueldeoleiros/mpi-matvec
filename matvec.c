#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdlib.h>

#define DEBUG 1

#define N 1024

int main(int argc, char *argv[] ) {

    int i,j,k,numprocs,rank;
    int rows;
    int compu_time, transfer_time;
    float vector[N];
    float *matrix;
    float *result;
    struct timeval tv_compu1, tv_compu2, tv_transfer1, tv_transfer2;

    float *localMatrix;
    float *localResult;

    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    int sizes[numprocs], desp[numprocs];

    // Initialize Matrix and Vector
    if(rank==0){
        matrix = malloc(sizeof(float)*N*N);
        result = malloc(sizeof(float)*N);
        for(i=0;i<N;i++) {
            vector[i] = i;
            for(j=0;j<N;j++)
                matrix[i+j*N] = i+j;
        }
    }

    gettimeofday(&tv_transfer1, NULL);
    // distribución del vector 
    MPI_Bcast(&vector,N,MPI_FLOAT,0,MPI_COMM_WORLD);
    gettimeofday(&tv_transfer2, NULL);

    transfer_time = (tv_transfer2.tv_usec - tv_transfer1.tv_usec)+ 1000000
                 * (tv_transfer2.tv_sec - tv_transfer1.tv_sec);

    //Calculo del numero de filas por proceso
    rows = N/numprocs;
    //Correcion del numero de filas en p-1 (si no es multiplo)
    if(rank == numprocs-1)
        rows = rows+(N%numprocs);

    localMatrix = malloc(sizeof(float)*N*rows);
    localResult = malloc(sizeof(float)*N);

    // creamos sentcounts y displ para statterv
    if(rank==0){
        for(i=0;i<numprocs;i++){
            if(i==numprocs-1)
                sizes[i] =  N*(rows+(N%numprocs));
            else
                sizes[i] = rows*N;
        }
        desp[0] = 0;
        for(i=1;i<numprocs;i++)
            desp[i] = desp[i-1] + sizes[i-1];
    }
    
    gettimeofday(&tv_transfer1, NULL);
    //Scatter de los datos de matrix
    MPI_Scatterv(matrix,sizes,desp,MPI_FLOAT,localMatrix,rows*N,MPI_FLOAT,0,MPI_COMM_WORLD);
    gettimeofday(&tv_transfer2, NULL);

    transfer_time += (tv_transfer2.tv_usec - tv_transfer1.tv_usec)+ 1000000
                 * (tv_transfer2.tv_sec - tv_transfer1.tv_sec);

    gettimeofday(&tv_compu1, NULL);
    //Lazo computacional
    for(i=0;i<rows;i++) {
        localResult[i]=0;
        for(j=0;j<N;j++)
            localResult[i] += localMatrix[j+(i*N)] * vector[j];
    }
    gettimeofday(&tv_compu2, NULL);

    // creamos sentcounts y displ para gatherv
    if(rank==0){
        for(i=0;i<numprocs;i++){
            sizes[i] = sizes[i]/N;
        }
        for(i=1;i<numprocs;i++){
            desp[i] = desp[i-1] + sizes[i-1];
        }
    }

    gettimeofday(&tv_transfer1, NULL);
    MPI_Gatherv(localResult,rows,MPI_FLOAT,result,sizes,desp,MPI_FLOAT,0,MPI_COMM_WORLD);
    gettimeofday(&tv_transfer2, NULL);

    transfer_time += (tv_transfer2.tv_usec - tv_transfer1.tv_usec)+ 1000000
                 * (tv_transfer2.tv_sec - tv_transfer1.tv_sec);
    
    compu_time = (tv_compu2.tv_usec - tv_compu1.tv_usec)+ 1000000
                 * (tv_compu2.tv_sec - tv_compu1.tv_sec);

    
    if(DEBUG){
        /*Display result */
        if(rank==0){
            for(i=0;i<N;i++) {
                printf(" %f \t ",result[i]);
            }
            printf("\n");
        }
    }else{
        /*Display times */
        printf ("process %d:\t Computational Time = %lf\t Transfer Time = %lf\n",
                rank, (double) compu_time/1E6, (double) transfer_time/1E6);
    }    

    // free memory
    if(rank==0){
        free(matrix);
        free(result);
    }
    free(localMatrix);
    free(localResult);


    MPI_Finalize();
    return 0;
}
