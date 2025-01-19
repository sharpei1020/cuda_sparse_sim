# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"

//compile nvcc *.cu -o test

__global__ void global_latency (unsigned int * my_array, int array_length, int iterations,  unsigned int * duration, unsigned int *index);


void parametric_measure_global(int N, int iterations);

void measure_global();


int main(){

	cudaSetDevice(0);

	measure_global();

	cudaDeviceReset();
	return 0;
}


void measure_global() {

	int N, iterations; 
	//stride in element
	iterations = 2;
	
	N = 400*1024*1024;
	// N = 100*1024*1024;
		printf("\n=====%10.4f MB array, Kepler pattern read, read 160 element====\n", sizeof(unsigned int)*(float)N/1024/1024);
		parametric_measure_global(N, iterations);
		printf("===============================================\n\n");
	
}


void parametric_measure_global(int N, int iterations) {
	cudaDeviceReset();

	cudaError_t error_id;
	
	int i;
	unsigned int * h_a;
	/* allocate arrays on CPU */
	h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N+2));
	unsigned int * d_a;
	/* allocate arrays on GPU */
	error_id = cudaMalloc ((void **) &d_a, sizeof(unsigned int) * (N+2));
	if (error_id != cudaSuccess) {
		printf("Error 1.0 is %s\n", cudaGetErrorString(error_id));
	}


	int off = 8;
	int iter = 3;
   	/* initialize array elements*/
	for (i=0; i<N; i++) 
		h_a[i] = 0;
	// 32 MB stride (original)
	for (i=0; i<49; i++){
		for (int j=0; j<(iter+1); j++)
		// h_a[i * 1024 * 1024 * 8] = (i+1)*1024*1024*8;
			h_a[i*1024*1024*8+off*j] = (i+1)*1024*1024*8+off*j;	
		// h_a[i * 1024 * 1024 * 8 + 2*off] = (i+1)*1024*1024*8+2*off;
	}
	// 1568 MB entry
	h_a[392*1024*1024+ iter*off] = 392*1024*1024 + iter*off + 1;
	h_a[392*1024*1024 + iter*off + 1] = 392*1024*1024 + iter*off + 2;
	h_a[392*1024*1024 + iter*off + 2] = 392*1024*1024 + iter*off;	

	// 1MB stride
	for (i=0; i< 31; i++) {
		for (int j=0; j<iter; j++)
		// h_a[(i+1568)*1024*256] = (i + 1569)*1024*256;
			h_a[(i+1568)*1024*256+j*off] = (i + 1569)*1024*256+j*off;
	}

	for (int j=0; j<iter; j++)
	// h_a[1599*1024*256] = off;
		h_a[1599*1024*256+j*off] = off*(j+1);
	// //8MB
	// for (i=0; i<50; i++) {
	// 	h_a[i*1024*1024*2] = (i+1)*1024*1024*2;
	// 	h_a[i*1024*1024*2+1] = (i+1)*1024*1024*2+1;
	// }
	// //256KB
	// for (i=0; i<31; i++)
	// 	h_a[(i+1568)*1024*64] = (i+1569)*1024*64;
	
	// h_a[1599*1024*64] = 1;
	// h_a[98*1024*1024+1] = h_a[98*1024*1024+2];
	// h_a[98*1024*1024+2] = h_a[98*1024*1024+3];
	// h_a[98*1024*1024+3] = h_a[98*1024*1024+1];	

	h_a[N] = 0;
	h_a[N+1] = 0;
	/* copy array elements from CPU to GPU */
        error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
	if (error_id != cudaSuccess) {
		printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
	}


	unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*2048*iterations);
	unsigned int *h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int)*2048*iterations);

	unsigned int *duration;
	error_id = cudaMalloc ((void **) &duration, sizeof(unsigned int)*2048*iterations);
	if (error_id != cudaSuccess) {
		printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
	}


	unsigned int *d_index;
	error_id = cudaMalloc( (void **) &d_index, sizeof(unsigned int)*2048*iterations);
	if (error_id != cudaSuccess) {
		printf("Error 1.3 is %s\n", cudaGetErrorString(error_id));
	}





	cudaThreadSynchronize ();
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(1,1,1);


	global_latency <<<Dg, Db, 32768>>>(d_a, N, iterations,  duration, d_index);

	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error kernel is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();



        error_id = cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int)*2048*iterations, cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
	}
        error_id = cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int)*2048*iterations, cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
	}

	cudaThreadSynchronize ();

	int mean_front = 0, mean_mid = 0, mean_end = 0;
	for(i=0;i<2048*iterations;i++) {
		if (i < 49)
			mean_front += h_timeinfo[i];
		else if (i < 81)
			mean_mid += h_timeinfo[i];
		else if (i < 130)
			mean_end += h_timeinfo[i];
		printf("%d, ", h_timeinfo[i]);
	}
	printf("\nmean_front: %d\n", mean_front/49);
	printf("mean_mid: %d\n", mean_mid/32);
	printf("mean_end: %d\n", mean_end/49);

	/* free memory on GPU */
	cudaFree(d_a);
	cudaFree(d_index);
	cudaFree(duration);


	/*free memory on CPU */
	free(h_a);
	free(h_index);
	free(h_timeinfo);
	
	cudaDeviceReset();	

}



__global__ void global_latency (unsigned int * my_array, int array_length, int iterations, unsigned int * duration, unsigned int *index) {

	unsigned int start_time, end_time;
	unsigned int j = 0; 

	const int loop = 2048;

	__shared__ unsigned int s_tvalue[loop];
	__shared__ unsigned int s_index[loop];
	extern __shared__ unsigned int s[];

	int k;

	// //first round
	// for (k = 0; k < iterations*loop; k++) 
	// 	j = my_array[j];
	// j = 0;
	for (int i = 0; i < iterations; i++) {
		for(k=0; k<loop; k++){
			s_index[k] = 0;
			s_tvalue[k] = 0;
		}
		//second round 
		for (k = 0; k < loop; k++) {
			
				start_time = clock();

				j = my_array[j];
				s_index[k]= j;
				end_time = clock();

				s_tvalue[k] = end_time-start_time;

		}

		// my_array[array_length] = j;
		// my_array[array_length+1] = my_array[j];
		if (threadIdx.x == 0)
			for(k=0; k<loop; k++){
				index[i * loop + k]= s_index[k];
				duration[i * loop + k] = s_tvalue[k];
			}
	}
}



