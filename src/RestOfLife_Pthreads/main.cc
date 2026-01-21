
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

#include <cstdlib>
#include <atomic>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <vector>


inline vec3 de_nan(const vec3& c) {
  vec3 temp = c;
  // use identity to de-"NaN" a color vector
  if (!(temp[0] == temp[0])) temp[0] = 0;
  if (!(temp[1] == temp[1])) temp[1] = 0;
  if (!(temp[2] == temp[2])) temp[2] = 0;
  return temp;
}


vec3 color(const ray& r, hittable *world, hittable *light_shape, int depth) {
  hit_record hrec;
  if (world->hit(r, 0.001, MAXFLOAT, hrec)) {
    scatter_record srec;
    vec3 emitted = hrec.mat_ptr->emitted(r, hrec, hrec.u, hrec.v, hrec.p);
    if (depth < 50 && hrec.mat_ptr->scatter(r, hrec, srec)) {
      if (srec.is_specular) {
        return srec.attenuation
          * color(srec.specular_ray, world, light_shape, depth+1);
      }
      else {
        hittable_pdf plight(light_shape, hrec.p);
        mixture_pdf p(&plight, srec.pdf_ptr);
        ray scattered = ray(hrec.p, p.generate(), r.time());
        float pdf_val = p.value(scattered.direction());
        // there is a new cosine_pdf in struct that gets leaked
        delete srec.pdf_ptr;
        return emitted + srec.attenuation
          * hrec.mat_ptr->scattering_pdf(r, hrec, scattered)
          * color(scattered, world, light_shape, depth+1)
          / pdf_val;
      }
    }
    else {
      return emitted;
    }
  }
  else {
    return vec3(0,0,0);
  }
}

struct RenderTask {
  int nx;
  int ny;
  int ns;
  hittable *world;
  hittable *light_shape;
  camera *cam;
  vec3 *framebuffer;
  std::atomic<int> *next_row;
  std::atomic<int> *rows_done;
};

struct ThreadParams {
  RenderTask *task;
  uint32_t thread_id;
};

void* render_worker(void *data) {
  ThreadParams *params = static_cast<ThreadParams*>(data);
  RenderTask *task = params->task;
  const uint32_t base_seed = 6767u;
  const uint32_t nx_u = static_cast<uint32_t>(task->nx);
  const uint32_t ny_u = static_cast<uint32_t>(task->ny);
  const uint32_t ns_u = static_cast<uint32_t>(task->ns);
  uint32_t pixel_x = 0u;
  uint32_t pixel_y = 0u;
  if (nx_u > 0u && ny_u > 0u) {
    const uint64_t pixel_count = static_cast<uint64_t>(nx_u) * ny_u;
    const uint32_t linear = static_cast<uint32_t>(params->thread_id % pixel_count);
    pixel_x = linear % nx_u;
    pixel_y = linear / nx_u;
  }
  const uint32_t width_u = static_cast<uint32_t>(task->nx);
  const uint32_t pixel_hash = static_cast<uint32_t>(
    static_cast<uint64_t>(pixel_y) * width_u * pixel_x
  ) ^ ns_u;
  const uint32_t thread_seed = base_seed ^ (pixel_hash ^ 9781u);
  set_random_seed(thread_seed);

  while (true) {
    int j = task->next_row->fetch_sub(1) - 1;
    if (j < 0) break;

    for (int i = 0; i < task->nx; i++) {
      vec3 col(0, 0, 0);
      for (int s = 0; s < task->ns; s++) {
        float u = float(i + random_double()) / float(task->nx);
        float v = float(j + random_double()) / float(task->ny);
        ray r = task->cam->get_ray(u, v);
        col += de_nan(color(r, task->world, task->light_shape, 0));
      }

      col /= float(task->ns);
      col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

      task->framebuffer[j * task->nx + i] = col;
    }

    task->rows_done->fetch_add(1);
  }

  return nullptr;
}

void cornell_box(hittable **scene, camera **cam, float aspect) {
  int i = 0;
  hittable **list = new hittable*[8];
  material *red = new lambertian( new constant_texture(vec3(0.65, 0.05, 0.05)) );
  material *white = new lambertian( new constant_texture(vec3(0.73, 0.73, 0.73)) );
  material *green = new lambertian( new constant_texture(vec3(0.12, 0.45, 0.15)) );
  material *light = new diffuse_light( new constant_texture(vec3(15, 15, 15)) );
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
  list[i++] = new sphere(vec3(190, 90, 190),90 , glass);

  // list[i++] = new
  //   translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white),  15), vec3(265,0,295));
  material *aluminum = new metal(vec3(0.91, 0.92, 0.92), 0.008);
  //material *gold = new metal(vec3(1.00, 0.71, 0.29), 0.0);
  list[i++] = new
    translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), aluminum), 23), vec3(265,0,295));
    // translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), gold), 23), vec3(265,0,295));

  *scene = new hittable_list(list,i);
  vec3 lookfrom(278, 278, -800);
  //vec3 lookfrom(278, 412, -800);
  vec3 lookat(278, 278, 0);
  float dist_to_focus = 10.0;
  float aperture = 0.0;
  float vfov = 40.0;
  *cam = new camera(lookfrom, lookat, vec3(0,1,0),
                    vfov, aspect, aperture, dist_to_focus, 0.0, 1.0);
}


// ./build/apps/program > output.ppm

int main (int argc, char** argv) {

  // default values
  bool SUPER_QUALITY_RENDER = !true;
  bool HIGH_QUALITY_RENDER = !true;
  bool MEDIUM_QUALITY_RENDER = !true;

  // handle command line arguments
  if (argc >= 2) {
    // first command line argument is "SH"?
    if (std::string(argv[1]) == "SH") {
      SUPER_QUALITY_RENDER = true;
    }
    // first command line argument is "HQ"?
    if (std::string(argv[1]) == "HQ") {
      HIGH_QUALITY_RENDER = true;
    }
    // first command line argument is "MQ"?
    if (std::string(argv[1]) == "MQ") {
      MEDIUM_QUALITY_RENDER = true;
    }
  }

  unsigned int user_thread_count = 0;
  if (argc >= 3) {
    int parsed = std::atoi(argv[2]);
    if (parsed > 0) {
      user_thread_count = static_cast<unsigned int>(parsed);
    }
  }

  int nx, ny, ns;

  if (SUPER_QUALITY_RENDER) {
    nx = 500;
    ny = 500;
    nx *= 2; ny *= 2;
    ns = 1000/2;
  } else if (HIGH_QUALITY_RENDER) {
    nx = 500;
    ny = 500;
    ns = 1000;
  } else if (MEDIUM_QUALITY_RENDER) {
    nx = 400;
    ny = 400;
    ns = 24;
  } else {
    nx = 500;
    ny = 500;
    ns = 10;
  }

  std::cerr << "Samples per point: " << ns << std::endl;
  std::cerr << "Total Scanlines: " << ny << std::endl;

  std::cout << "P3\n" << nx << " " << ny << "\n255\n";

  hittable *world;
  camera *cam;
  float aspect = float(ny) / float(nx);
  cornell_box(&world, &cam, aspect);

  hittable *a[2];
  hittable *light_shape  = new xz_rect(213, 343, 227, 332, 554, 0);
  hittable *glass_sphere = new sphere(vec3(190, 90, 190), 90, 0);
  a[0] = light_shape;
  a[1] = glass_sphere;
  hittable_list hlist(a,2);

  std::vector<vec3> framebuffer(nx * ny);
  std::atomic<int> next_row(ny);
  std::atomic<int> rows_done(0);

  RenderTask task {
    nx,
    ny,
    ns,
    world,
    &hlist,
    cam,
    framebuffer.data(),
    &next_row,
    &rows_done
  };

  unsigned int thread_count = user_thread_count
    ? user_thread_count
    : std::thread::hardware_concurrency();
  if (thread_count == 0) thread_count = 4;

  std::vector<pthread_t> threads(thread_count);
  std::vector<ThreadParams> thread_params(thread_count);
  for (unsigned int t = 0; t < thread_count; ++t) {
    thread_params[t].task = &task;
    thread_params[t].thread_id = static_cast<uint32_t>(t);
    pthread_create(&threads[t], nullptr, render_worker, &thread_params[t]);
  }

  while (rows_done.load() < ny) {
    int remaining = ny - rows_done.load();
    std::cerr << "\rScanlines remaining: " << remaining << ' ' << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }

  for (unsigned int t = 0; t < thread_count; ++t) {
    pthread_join(threads[t], nullptr);
  }

  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      vec3 col = framebuffer[j * nx + i];

      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  std::cerr << "\nDone.\n";

}
