#include "rtweekend.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "Timer.h"
#include <omp.h>

color ray_color(const ray &r, const hittable &world, int depth)
{
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec))
    {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return color(0, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

hittable_list set_scene()
{
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_float();
            point3 center(a + 0.9, 0.2, b + 0.9);

            if ((center - point3(4, 0.2, 0)).length() > 0.9)
            {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8)
                {
                    // diffuse
                    auto albedo = color::random(0.7, 0.7) * color::random(0.7, 0.7);
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                {
                    // metal
                    auto albedo = color::random(0.5, 0.5);
                    auto fuzz = random_float(0, 0);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}
hittable_list random_scene()
{
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_float();
            point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

            if ((center - point3(4, 0.2, 0)).length() > 0.9)
            {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8)
                {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5f);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0f, material1));

    auto material2 = make_shared<lambertian>(color(0.4f, 0.2f, 0.1f));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0f, material2));

    auto material3 = make_shared<metal>(color(0.7f, 0.6f, 0.5f), 0.0f);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0f, material3));

    return world;
}


void pixel_serial_unopt(camera &cam, hittable_list &world, int max_depth, int samples_per_pixel, int image_width, int image_height, int i, int j) {
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s)
    {
        auto u = (i + random_float()) / (image_width - 1);
        auto v = (j + random_float()) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, world, max_depth);
    }
    write_color(std::cout, pixel_color, samples_per_pixel);
}

void pixel_serial_unroll_2(camera &cam, hittable_list &world, int max_depth, int samples_per_pixel, int image_width, int image_height, int i, int j) {
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s += 2)
    {
        auto u1 = (i + random_float()) / (image_width - 1);
        auto u2 = (i + random_float()) / (image_width - 1);
        auto v1 = (j + random_float()) / (image_height - 1);
        auto v2 = (j + random_float()) / (image_height - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        pixel_color += ray_color(r1, world, max_depth);
        pixel_color += ray_color(r2, world, max_depth);
    }
    write_color(std::cout, pixel_color, samples_per_pixel);
}
void pixel_serial_unroll_2(camera &cam, hittable_list &world, color pixel_colors[], int max_depth, int samples_per_pixel, int image_width, int image_height, int i, int j) {
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s += 2)
    {
        auto u1 = (i + random_float()) / (image_width - 1);
        auto u2 = (i + random_float()) / (image_width - 1);
        auto v1 = (j + random_float()) / (image_height - 1);
        auto v2 = (j + random_float()) / (image_height - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        pixel_color += ray_color(r1, world, max_depth);
        pixel_color += ray_color(r2, world, max_depth);
    }
    pixel_colors[j * image_width + i] = color(pixel_color.x(), pixel_color.y(), pixel_color.z());
}

void pixel_serial_unroll_4(camera &cam, hittable_list &world, color pixel_colors[], int max_depth, int samples_per_pixel, int image_width, int image_height, int i, int j) {
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s += 4)
    {
        auto u1 = (i + random_float()) / (image_width - 1);
        auto u2 = (i + random_float()) / (image_width - 1);
        auto u3 = (i + random_float()) / (image_width - 1);
        auto u4 = (i + random_float()) / (image_width - 1);
        auto v1 = (j + random_float()) / (image_height - 1);
        auto v2 = (j + random_float()) / (image_height - 1);
        auto v3 = (j + random_float()) / (image_height - 1);
        auto v4 = (j + random_float()) / (image_height - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        ray r3 = cam.get_ray(u3, v3);
        ray r4 = cam.get_ray(u4, v4);
        pixel_color += ray_color(r1, world, max_depth);
        pixel_color += ray_color(r2, world, max_depth);
        pixel_color += ray_color(r3, world, max_depth);
        pixel_color += ray_color(r4, world, max_depth);
    }
    pixel_colors[((image_height - j - 1) * image_width) + i] = color(pixel_color.x(), pixel_color.y(), pixel_color.z());
}
int main()
{

    // Image

    const auto aspect_ratio = 16.0f / 9.0f;
    const int image_width = 256;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 20;
    const int max_depth = 50;
    color* pixel_colors= new color[image_width * image_height];

    // World

    auto world = set_scene();

    // Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0f;
    auto aperture = 0.1f;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // Render

    std::cout << "P3\n"
              << image_width << ' ' << image_height << "\n255\n";
    {
        Timer timer;
        // Fill output matrix: rows and columns are i and j respectively
        omp_set_dynamic(0);     // Explicitly disable dynamic teams

        omp_set_num_threads(16);
        #pragma omp parallel for
        for (int j = image_height - 1; j >= 0; --j)
        {
            std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
            for (int i = 0; i < image_width; ++i)
            {
                pixel_serial_unroll_4(cam, world, pixel_colors, max_depth, samples_per_pixel, image_width, image_height, i, j);
                // color pixel_color(0, 0, 0);
                // for (int s = 0; s < samples_per_pixel; ++s)
                // {
                //     auto u = (i + random_float()) / (image_width - 1);
                //     auto v = (j + random_float()) / (image_height - 1);
                //     ray r = cam.get_ray(u, v);
                //     pixel_color += ray_color(r, world, max_depth);
                // }
                // write_color(std::cout, pixel_color, samples_per_pixel);
            }
        }
    }
    write_colors(std::cout, pixel_colors, image_width * image_height, samples_per_pixel);
    std::cerr << "\nDone.\n";
}