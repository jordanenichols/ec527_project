#include <cstlib>
#include <cstdio>
#include "rtweekend.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
    cudaGetErrorString(code), file, line);
    if(abort)
        exit(code);
    }
}

int main()
{
    
}