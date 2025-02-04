# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"

//compile nvcc -arch=sm_35 *.cu -o test

__global__ void global_latency (const unsigned int * __restrict__ my_array, int array_length, int iterations,  unsigned int * duration, unsigned int *index);


void parametric_measure_global(int N, int iterations, int stride);

void measure_global();


int main(){

	cudaSetDevice(0);

	measure_global();

	cudaDeviceReset();
	return 0;
}


void measure_global() {

	int N, iterations, stride; 
	//stride in element
	iterations = 1;
	
	// stride = 32/sizeof(unsigned int); //stride, in element
	stride = 4; //stride, in element
	// N = 120*256; 
	for (N = 2*1024*256; N <= 2*1024*256; N+=stride) {
		printf("\n=====%10.4f KB array, warm TLB, record 1024 element====\n", sizeof(unsigned int)*(float)N/1024);
		printf("Stride = %d element, %d byte\n", stride, stride * sizeof(unsigned int));
		printf("Set: %d \n", (N - 32*256) / stride);
		parametric_measure_global(N, iterations, stride );
		printf("===============================================\n\n");
		break;
	}
}


void parametric_measure_global(int N, int iterations, int stride) {
	cudaDeviceReset();

	cudaError_t error_id;
	
	int i;
	unsigned int * h_a;
	/* allocate arrays on CPU */
	h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N));
	unsigned int * d_a;
	/* allocate arrays on GPU */
	error_id = cudaMalloc ((void **) &d_a, sizeof(unsigned int) * (N));
	if (error_id != cudaSuccess) {
		printf("Error 1.0 is %s\n", cudaGetErrorString(error_id));
	}


   	/* initialize array elements on CPU with pointers into d_a. */
	
	for (i = 0; i < N; i++) {		
	//original:	
		h_a[i] = (i+stride)%N;	
	}

	/* copy array elements from CPU to GPU */
        error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
	if (error_id != cudaSuccess) {
		printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
	}


	unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*1024*iterations);
	unsigned int *h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int)*1024*iterations);

	unsigned int *duration;
	error_id = cudaMalloc ((void **) &duration, sizeof(unsigned int)*1024*iterations);
	if (error_id != cudaSuccess) {
		printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
	}


	unsigned int *d_index;
	error_id = cudaMalloc( (void **) &d_index, sizeof(unsigned int)*1024*iterations );
	if (error_id != cudaSuccess) {
		printf("Error 1.3 is %s\n", cudaGetErrorString(error_id));
	}





	cudaThreadSynchronize ();
	/* launch kernel*/
	dim3 Db = dim3(4);
	dim3 Dg = dim3(1,1,1);


	global_latency <<<Dg, Db>>>(d_a, N, iterations,  duration, d_index);

	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error kernel is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();



        error_id = cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int)*1024*iterations, cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
	}
        error_id = cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int)*1024*iterations, cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
	}

	cudaThreadSynchronize ();

	int miss = 0;
	for(i=0;i<1024*iterations;i++) {
		// if(h_timeinfo[i]>200) {
		printf("%d, ", h_timeinfo[i]);
		if(h_timeinfo[i]>450) miss++;
	}
	printf("Miss number: %d\n", miss);

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



__global__ void global_latency (const unsigned int * __restrict__ my_array, int array_length, int iterations, unsigned int * duration, unsigned int *index) {

	unsigned int start_time, end_time;
	unsigned int j = threadIdx.x; 

	__shared__ unsigned int s_tvalue[1024];
	__shared__ unsigned int s_index[1024];

	int k;
	// first round
	// for (k = 0; k < 16*iterations*1024; k++) 
	// 	j = __ldg(&my_array[j]);
	for (int i = 0; i < iterations; i++) {
		for(k=0; k<1024; k++){
			s_index[k] = 0;
			s_tvalue[k] = 0;
		}

		//second round 
		for (int m = 0; m < 1; m++) {
			j = threadIdx.x;
			for (k = 0; k < 1024; k++) {
				start_time = clock();
				j = __ldcs(&my_array[j]);
				// j = my_array[j];
				s_index[m*1024+k]= j;
				end_time = clock();
				s_tvalue[m*1024+k] = end_time-start_time;
			}
		}

		if (threadIdx.x == 0) {
			for(k=0; k<1024; k++){
				index[k+i*1024]= s_index[k];
				duration[k+i*1024] = s_tvalue[k];
			}
		}
	}
}



