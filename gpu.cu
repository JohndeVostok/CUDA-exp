#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int N = 1024 * 1024 * 64 * 16;

const int grid_size = 112;
const int block_size = 1024;

static long long ustime(void) {
	struct timeval tv;
	long long ust;
	gettimeofday(&tv, NULL);
	ust = ((long)tv.tv_sec)*1000000;
	ust += tv.tv_usec;
	return ust;
}

double getPi(int n) {
	double s = 0;
		for (int i = 0; i < n; i++) {
		double t = (2 * i + 1) / (n * 2.0);
		s += 4 / (1 + t * t);
	}
	return s / n;
}

__global__ void reducePi(double *sum) {
	__shared__ double cache[block_size];
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheIdx = threadIdx.x;

	double x, t = 0;
	while (tid < N) {
		x = (2 * tid + 1) / (N * 2.0);
		t += 4 / (1 + x * x);
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIdx] = t;
	__syncthreads();

	for (int i = (blockDim.x >> 1); i; i >>= 1) {
		if (cacheIdx < i) {
			cache[cacheIdx] += cache[cacheIdx + i];
		}
		__syncthreads();
	}
	if (!cacheIdx) {
		sum[blockIdx.x] = cache[0];
	}
}

__global__ void sumUp(double *sum, double *s) {
	__shared__ double cache[grid_size];
	int tid = threadIdx.x;
	cache[tid] = sum[tid];
	__syncthreads();
	for (int i = (blockDim.x >> 1); i; i >>= 1) {
		if (tid < i) {
			cache[tid] += cache[tid + i];
		}
		__syncthreads();
	}
	if (!tid) {
		*s = cache[0];
	}
}

int main() {
	long long op, ed, tcpu, tgpu;
	double sum, ans;

	double *dev_s_sum;
	double *dev_sum;

	//int grid_size, grid_min_size, block_size;

	//int gs, bs;
	//cudaOccupancyMaxPotentialBlockSize(&gs, &bs, reducePi, 0, N);
	//printf("%d %d\n", gs, bs);
	//grid_size = (N - 1) / block_size + 1;

	cudaMalloc((void **) &dev_s_sum, grid_size * sizeof(double));
	cudaMalloc((void **) &dev_sum, sizeof(double));

	op = ustime();
	reducePi <<<grid_size, block_size>>>(dev_s_sum);
	sumUp <<<1, 64>>>(dev_s_sum, dev_sum);
	ed = ustime();
	tgpu = ed - op;

	cudaMemcpy(&sum, dev_sum, sizeof(double), cudaMemcpyDeviceToHost);
	ans = sum / double(N);
	printf("GPU: Pi: %.12lf Time(ms): %f\n", ans, tgpu / 1000.0);

	cudaFree(dev_s_sum);
	cudaFree(dev_sum);

	op = ustime();
	ans = getPi(N);
	ed = ustime();
	tcpu = ed - op;
	printf("CPU: Pi: %.12f Time(ms): %f\n", ans, tcpu / 1000.0);
	printf("Speed up: %.2f\n", tcpu / (double) tgpu);
}
