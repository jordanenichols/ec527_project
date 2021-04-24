#include <iostream>
#include "vec3.cuh"
#include "ray.cuh"
#include "Timer.h"
#include "sphere.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "camera.cuh"
//Include for FLT_MAX (infinity)
#include <float.h>
// CUDA RANDOM library
#include <curand_kernel.h>
#include "material.cuh"

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

// #define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

// __device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
//     vec3 p;
//     do {
//         p = 2.0f*RANDVEC3 - vec3(1,1,1);
//     } while (p.length_squared() >= 1.0f);
//     return p;
// }

// __device__ vec3 random_unit_vector(curandState* local_rand_state)
// {
//     return unit_vector(random_in_unit_sphere(local_rand_state));
// }

// __device__ vec3 random_in_hemisphere(const vec3& normal, curandState* local_rand_state) {
//     vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
//     if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
//         return in_unit_sphere;
//     else
//         return -in_unit_sphere;
// }

__device__ color ray_color(const ray& r, hittable** world, int max_depth, curandState* local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < max_depth; i++) {
       hit_record rec;
       if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
        ray scattered;
        vec3 attenuation;
        if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
        else {
            return vec3(0.0,0.0,0.0);
        }
    }
       else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
         }
       }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void clean_up(hittable **d_list, hittable **d_world, camera** d_camera) {
    for(int i=0; i < 5; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
 }

 __global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void init_rand_state(int image_width, int image_height, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height)) return;
    int pixel_index = j * image_width + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* frame_buffer, int image_width, int image_height, int samples_per_pixel,
                       camera** d_camera, hittable** d_world, curandState* rand_state, int max_depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= image_width) || (j >= image_height)) return;
    int pixel_index = j * image_width + i;
    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0, 0, 0);
    for(int s = 0; s < samples_per_pixel; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(image_width);
        float v = float(j + curand_uniform(&local_rand_state)) / float(image_height);
        ray r = (*d_camera)->get_ray(u, v, &local_rand_state);
        pixel_color += ray_color(r, d_world, max_depth, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    pixel_color /= float(samples_per_pixel);
    pixel_color[0] = sqrt(pixel_color[0]);
    pixel_color[1] = sqrt(pixel_color[1]);
    pixel_color[2] = sqrt(pixel_color[2]);

    frame_buffer[pixel_index] = pixel_color;
}

int main()
{
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1080;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    int num_pixels = image_width*image_height;
    size_t frame_buffer_size = num_pixels * sizeof(vec3);
    const int samples_per_pixel = 50;
    const int max_depth = 50;


    // allocate random state
    curandState *d_rand_state;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    /***
    This call renders the image on the GPU and has the CUDA runtime divide the work on the GPU
    into blocks of 8×8 threads. You might also try other thread sizes to what works best for your machine. I tried to find a block size that:
    (1) is a small, square region so the work would be similar.  This should help each pixel do a 
    similar amount of work. If some pixels work much longer than other pixels in a block, the efficiency
    of that block is impacted. (2) has a pixel count that is a multiple of 32 in order to fit into warps evenly.
    ***/
    int tx = 8;
    int ty = 8;

    dim3 blocks(image_width/tx+1, image_height/ty+1);
    dim3 threads(tx, ty);

    init_rand_state<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // make our world of hittables & the camera
    hittable **d_list;
    int num_hittables = 22*22+1+3;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_list, num_hittables*sizeof(hittable *)));
    hittable **d_world;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    // allocate Frame buffer
    vec3* frame_buffer;
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&frame_buffer, frame_buffer_size));
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Render our buffer

    
    Timer timer;
    render<<<blocks, threads>>>(frame_buffer, image_width, image_height, samples_per_pixel,
                                d_camera, d_world, d_rand_state, max_depth);
    
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout << "P3\n"
              << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height-1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            int ir = int(255.99 * frame_buffer[pixel_index].r());
            int ig = int(255.99 * frame_buffer[pixel_index].g());
            int ib = int(255.99 * frame_buffer[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    std::cerr << "\nDone.\n";
    CUDA_SAFE_CALL(cudaFree(frame_buffer));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    clean_up<<<1,1>>>(d_list,d_world, d_camera);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaFree(d_list));
    CUDA_SAFE_CALL(cudaFree(d_world));
    // CUDA_SAFE_CALL(cudaFree(frame_buffer));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}