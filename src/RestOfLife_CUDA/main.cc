#include "common/rtweekend.h"
#include "common/camera.h"
#include "common/rtw_stb_image.h"
#include "common/texture.h"
#include "aarect.h"
#include "box.h"
#include "bvh.h"
#include "hittable_list.h"
#include "material.h"
#include "pdf.h"
#include "moving_sphere.h"
#include "sphere.h"

/// ---
#include "cuda_support.h"
#include "scene_builder.h"
#include "renderer.h"
/// ---


#include <iostream>
#include <cstdlib>
#include <vector>

inline vec3 de_nan(const vec3 &c)
{
  vec3 temp = c;
  // use identity to de-"NaN" a color vector
  if (!(temp[0] == temp[0]))
    temp[0] = 0;
  if (!(temp[1] == temp[1]))
    temp[1] = 0;
  if (!(temp[2] == temp[2]))
    temp[2] = 0;
  return temp;
}

vec3 color(const ray &r, hittable *world, hittable *light_shape, int depth)
{
  hit_record hrec;
  if (world->hit(r, 0.001, MAXFLOAT, hrec))
  {
    scatter_record srec;
    vec3 emitted = hrec.mat_ptr->emitted(r, hrec, hrec.u, hrec.v, hrec.p);
    if (depth < 50 && hrec.mat_ptr->scatter(r, hrec, srec))
    {
      if (srec.is_specular)
      {
        return srec.attenuation * color(srec.specular_ray, world, light_shape, depth + 1);
      }
      else
      {
        hittable_pdf plight(light_shape, hrec.p);
        mixture_pdf p(&plight, srec.pdf_ptr);
        ray scattered = ray(hrec.p, p.generate(), r.time());
        float pdf_val = p.value(scattered.direction());
        // there is a new cosine_pdf in struct that gets leaked
        delete srec.pdf_ptr;
        return emitted + srec.attenuation * hrec.mat_ptr->scattering_pdf(r, hrec, scattered) * color(scattered, world, light_shape, depth + 1) / pdf_val;
      }
    }
    else
    {
      return emitted;
    }
  }
  else
  {
    return vec3(0, 0, 0);
  }
}

void cornell_box(hittable **scene, camera **cam, float aspect)
{
  int i = 0;
  hittable **list = new hittable *[8];
  material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
  material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
  material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
  material *light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));
  list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
  list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
  // light source now has a emitting direction
  list[i++] = new flip_normals(new xz_rect(213, 343, 227, 332, 554, light));
  list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
  list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
  list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
  // replaced with sphere
  // list[i++] = new
  //   translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18), vec3(130,0,65));
  material *glass = new dielectric(1.5);
  list[i++] = new sphere(vec3(190, 90, 190), 90, glass);

  // list[i++] = new
  //   translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white),  15), vec3(265,0,295));
  material *aluminum = new metal(vec3(0.91, 0.92, 0.92), 0.008);
  // material *gold = new metal(vec3(1.00, 0.71, 0.29), 0.0);
  list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), aluminum), 23), vec3(265, 0, 295));
  // translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), gold), 23), vec3(265,0,295));

  *scene = new hittable_list(list, i);
  vec3 lookfrom(278, 278, -800);
  // vec3 lookfrom(278, 412, -800);
  vec3 lookat(278, 278, 0);
  float dist_to_focus = 10.0;
  float aperture = 0.0;
  float vfov = 40.0;
  *cam = new camera(lookfrom, lookat, vec3(0, 1, 0),
                    vfov, aspect, aperture, dist_to_focus, 0.0, 1.0);
}

// ./build/apps/program > output.ppm

int main(int argc, char **argv)
{

  // default values
  bool SUPER_QUALITY_RENDER = !true;
  bool HIGH_QUALITY_RENDER = !true;
  bool MEDIUM_QUALITY_RENDER = !true;

  // handle command line arguments
  if (argc >= 2)
  {
    // first command line argument is "SH"?
    if (std::string(argv[1]) == "SH")
    {
      SUPER_QUALITY_RENDER = true;
    }
    // first command line argument is "HQ"?
    if (std::string(argv[1]) == "HQ")
    {
      HIGH_QUALITY_RENDER = true;
    }
    // first command line argument is "MQ"?
    if (std::string(argv[1]) == "MQ")
    {
      MEDIUM_QUALITY_RENDER = true;
    }
  }

  int nx, ny, ns;

  if (SUPER_QUALITY_RENDER)
  {
    nx = 500;
    ny = 500;
    nx *= 2;
    ny *= 2;
    ns = 1000 / 2;
  }
  else if (HIGH_QUALITY_RENDER)
  {
    nx = 500;
    ny = 500;
    ns = 1000;
  }
  else if (MEDIUM_QUALITY_RENDER)
  {
    nx = 400;
    ny = 400;
    ns = 24;
  }
  else
  {
    nx = 500;
    ny = 500;
    ns = 10;
  }

  std::cerr << "Samples per point: " << ns << std::endl;
  std::cerr << "Total Scanlines: " << ny << std::endl;

  std::cout << "P3\n"
            << nx << " " << ny << "\n255\n";

  hittable *world;
  camera *cam;
  float aspect = float(ny) / float(nx);
  cornell_box(&world, &cam, aspect);

  SceneBuilder builder;
  DeviceScene device_scene = builder.build_from(*world);
  CudaRenderer cuda_renderer;
  ///const char *seed_env = std::getenv("RENDER_SEED");
  ///if (seed_env != nullptr)
  ///{
  ///  cuda_renderer.set_seed(static_cast<uint32_t>(std::strtoul(seed_env, nullptr, 10)));
  ///}

  cuda_renderer.set_seed(6767);


  std::vector<vec3> pixels = cuda_renderer.render(device_scene, *cam, nx, ny, ns);
  for (int j = ny - 1; j >= 0; j--)
  {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < nx; i++)
    {
      vec3 col = de_nan(pixels[j * nx + i]);
      col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

      int ir = int(255.99 * col[0]);
      int ig = int(255.99 * col[1]);
      int ib = int(255.99 * col[2]);

      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  std::cerr << "\nDone.\n";
}
