/*
 * Uso de Open MP para la paralelizacion en la multiplicacion de matrices NxN
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1000
#define chunk 10


void matrixMultCPU(int *A,int *B,int *C){
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			int acc=0;
			for(int k=0;k<N;k++){
				int m = A[i*N+k];
				int n = B[k*N+j];
				acc += m*n;
			}
		C[i*N+j] = acc;
		}
	}
}

int main(){
	double begin, end;
	int *A, *B, *C1, *d_C;
	int size = N*N*sizeof(int);
	
  	A = (int*)malloc(size);
 	B = (int*)malloc(size);
 	C1 = (int*)malloc(size);
 	d_C = (int*)malloc(size);

	for(int i=0;i<N*N;i++){
			A[i]=1;
			B[i]=2;
	}

	//------------CPU----------------
	//clock_t tic = clock();
	begin = omp_get_wtime();
	matrixMultCPU(A,B,C1);
  	//clock_t toc = clock();
	end = omp_get_wtime();
	printf("Tiempo CPU: %.5f segundos\n", (end - begin));
	//-------------------------------
 
	int i,j,k, acc;

	//------------OpenMP-------------
	//tic = clock();
	begin = omp_get_wtime();
	#pragma omp parallel for private(i,j,k) shared(A,B,d_C) schedule(dynamic,chunk) reduction (+:acc)
		for(i=0;i<N;i++){
			for(j=0;j<N;j++){
				acc = 0; 
				for(k=0; k<N; k++)
					acc += A[i*N+k] * B[k*N+j];
			d_C[i*N+j] = acc;
			}
		}

  	//toc = clock();
	end = omp_get_wtime();
	printf("\n\nTiempo GPU: %.5f segundos\n", (end - begin));
	//--------------------------------
  
	free(A);
	free(B);
	free(C1);
	free(d_C);
	return 0;
}


/*

RESULTADOS OBTENIDOS:

			|		CPU		|		GPU		|		
			---------------------------------
|	N		|			t					|	MEJORA (%)		|
----------------------------------------------------------------|
|	100		|	0,007842	|	0,007023	|	10,4437643458	|
|	200		|	0,056081	|	0,041652	|	25,7288564755	|
|	500		|	0,707555	|	0,620655	|	12,2817307488	|
|	1000	|	6,648028	|	6,033606	|	9,2421692568	|
|	1500	|	24,398489	|	22,411754	|	8,142860814		|
|	2000	|	47,579339	|	43,735873	|	8,078014703		|

*/
