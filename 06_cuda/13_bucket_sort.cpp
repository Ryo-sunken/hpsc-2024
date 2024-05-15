#include <cstdio>
#include <cstdlib>
#include <vector>
#include "hip/hip_runtime.h"

__global__ void reset(int *bucket) {
  bucket[threadIdx.x] = 0;
}

__global__ void count(int *key, int *bucket, int n) {
  for (int i = 0; i < n; i++) {
    if (key[i] == threadIdx.x) {
      bucket[threadIdx.x]++;
    }
  }
}

__global__ void sort(int *key, int *bucket) {
  int offset = 0;
  for (int i = 0; i < threadIdx.x; i++) offset += bucket[i];
  for (int i = 0; i < bucket[threadIdx.x]; i++) key[offset + i] = threadIdx.x;
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  hipMallocManaged(&key, n * sizeof(int));
  hipMallocManaged(&bucket, range * sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  
  hipLaunchKernelGGL(reset, dim3(1), dim3(n), bucket);
  hipDeviceSynchronize();

  hipLaunchKernelGGL(count, dim3(1), dim3(range), key, bucket, n);
  hipDeviceSynchronize();

  hipLaunchKernelGGL(sort, dim3(1), dim3(range), key, bucket);
  hipDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  hipFree(key);
  hipFree(bucket);
}