#include "renderer.h"

#include "onb.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>

namespace {

constexpr int kTileX = 1;
constexpr int kTileY = 1;
constexpr int kSampleLanes = 1;
constexpr float kPi = 3.14159265358979323846f;

struct Rng {
  __device__ explicit Rng(uint32_t seed) : state(seed) {}
  __device__ float next() {
    state = 1664525u * state + 1013904223u;
    return static_cast<float>(state & 0x00FFFFFF) * (1.0f / 16777216.0f);
  }
  uint32_t state;
};

__device__ vec3 random_in_unit_sphere(Rng &rng) {
  while (true) {
    vec3 p(2.0f * rng.next() - 1.0f,
           2.0f * rng.next() - 1.0f,
           2.0f * rng.next() - 1.0f);
    if (p.squared_length() < 1.0f) {
      return p;
    }
  }
}

__device__ vec3 random_in_unit_disk(Rng &rng) {
  while (true) {
    vec3 p(2.0f * rng.next() - 1.0f,
           2.0f * rng.next() - 1.0f,
           0.0f);
    if (dot(p, p) < 1.0f) {
      return p;
    }
  }
}

__device__ vec3 random_cosine_direction(Rng &rng) {
  float r1 = rng.next();
  float r2 = rng.next();
  float z = sqrtf(1.0f - r2);
  float phi = 2.0f * kPi * r1;
  float x = cosf(phi) * sqrtf(r2);
  float y = sinf(phi) * sqrtf(r2);
  return vec3(x, y, z);
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n) {
  return v - 2.0f * dot(v, n) * n;
}

__device__ bool refract(const vec3 &v,
                        const vec3 &n,
                        float ni_over_nt,
                        vec3 &refracted) {
  vec3 uv = unit_vector(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
  if (discriminant > 0.0f) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  }
  return false;
}

__device__ float schlick(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

struct DeviceCamera {
  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  float lens_radius;
  float time0;
  float time1;
};

__device__ ray camera_get_ray(const DeviceCamera &cam,
                              float s,
                              float t,
                              Rng &rng) {
  vec3 rd = cam.lens_radius * random_in_unit_disk(rng);
  vec3 offset = cam.u * rd.x() + cam.v * rd.y();
  float time = cam.time0 + rng.next() * (cam.time1 - cam.time0);
  return ray(cam.origin + offset,
             cam.lower_left_corner + s * cam.horizontal + t * cam.vertical -
                 cam.origin - offset,
             time);
}

struct HitRecord {
  float t;
  float u;
  float v;
  vec3 p;
  vec3 normal;
  int material_index;
};

__device__ vec3 rotate_y(const vec3 &p, float cos_theta, float sin_theta) {
  return vec3(cos_theta * p.x() + sin_theta * p.z(),
              p.y(),
              -sin_theta * p.x() + cos_theta * p.z());
}

__device__ ray transform_ray_to_local(const DeviceTransform &t,
                                      const ray &r) {
  vec3 origin = r.origin() - t.translation;
  vec3 direction = r.direction();
  float cos_theta = t.cos_theta;
  float sin_theta = -t.sin_theta;  // inverse rotation
  origin = rotate_y(origin, cos_theta, sin_theta);
  direction = rotate_y(direction, cos_theta, sin_theta);
  return ray(origin, direction, r.time());
}

__device__ vec3 transform_normal_to_world(const DeviceTransform &t,
                                          const vec3 &n,
                                          bool flip) {
  vec3 rotated = rotate_y(n, t.cos_theta, t.sin_theta);
  vec3 nn = flip ? -rotated : rotated;
  return unit_vector(nn);
}

__device__ bool hit_sphere(const DevicePrimitive &prim,
                           const ray &r,
                           float t_min,
                           float t_max,
                           HitRecord &rec) {
  vec3 oc = r.origin() - prim.sphere.center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - prim.sphere.radius * prim.sphere.radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0.0f) {
    float temp = (-b - sqrtf(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      vec3 outward = (rec.p - prim.sphere.center) / prim.sphere.radius;
      rec.normal = outward;
      float phi = atan2f(outward.z(), outward.x());
      float theta = asinf(outward.y());
      rec.u = 1.0f - (phi + kPi) / (2.0f * kPi);
      rec.v = (theta + kPi * 0.5f) / kPi;
      return true;
    }
    temp = (-b + sqrtf(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      vec3 outward = (rec.p - prim.sphere.center) / prim.sphere.radius;
      rec.normal = outward;
      float phi = atan2f(outward.z(), outward.x());
      float theta = asinf(outward.y());
      rec.u = 1.0f - (phi + kPi) / (2.0f * kPi);
      rec.v = (theta + kPi * 0.5f) / kPi;
      return true;
    }
  }
  return false;
}

__device__ bool hit_rect_xy(const DevicePrimitive &prim,
                            const ray &r,
                            float t_min,
                            float t_max,
                            HitRecord &rec) {
  float t = (prim.rect.k - r.origin().z()) / r.direction().z();
  if (t < t_min || t > t_max) return false;
  float x = r.origin().x() + t * r.direction().x();
  float y = r.origin().y() + t * r.direction().y();
  if (x < prim.rect.x0 || x > prim.rect.x1 ||
      y < prim.rect.y0 || y > prim.rect.y1) {
    return false;
  }
  rec.u = (x - prim.rect.x0) /
          (prim.rect.x1 - prim.rect.x0);
  rec.v = (y - prim.rect.y0) /
          (prim.rect.y1 - prim.rect.y0);
  rec.t = t;
  rec.p = r.point_at_parameter(t);
  rec.normal = vec3(0.0f, 0.0f, 1.0f);
  return true;
}

__device__ bool hit_rect_xz(const DevicePrimitive &prim,
                            const ray &r,
                            float t_min,
                            float t_max,
                            HitRecord &rec) {
  float t = (prim.rect.k - r.origin().y()) / r.direction().y();
  if (t < t_min || t > t_max) return false;
  float x = r.origin().x() + t * r.direction().x();
  float z = r.origin().z() + t * r.direction().z();
  if (x < prim.rect.x0 || x > prim.rect.x1 ||
      z < prim.rect.y0 || z > prim.rect.y1) {
    return false;
  }
  rec.u = (x - prim.rect.x0) /
          (prim.rect.x1 - prim.rect.x0);
  rec.v = (z - prim.rect.y0) /
          (prim.rect.y1 - prim.rect.y0);
  rec.t = t;
  rec.p = r.point_at_parameter(t);
  rec.normal = vec3(0.0f, 1.0f, 0.0f);
  return true;
}

__device__ bool hit_rect_yz(const DevicePrimitive &prim,
                            const ray &r,
                            float t_min,
                            float t_max,
                            HitRecord &rec) {
  float t = (prim.rect.k - r.origin().x()) / r.direction().x();
  if (t < t_min || t > t_max) return false;
  float y = r.origin().y() + t * r.direction().y();
  float z = r.origin().z() + t * r.direction().z();
  if (y < prim.rect.x0 || y > prim.rect.x1 ||
      z < prim.rect.y0 || z > prim.rect.y1) {
    return false;
  }
  rec.u = (y - prim.rect.x0) /
          (prim.rect.x1 - prim.rect.x0);
  rec.v = (z - prim.rect.y0) /
          (prim.rect.y1 - prim.rect.y0);
  rec.t = t;
  rec.p = r.point_at_parameter(t);
  rec.normal = vec3(1.0f, 0.0f, 0.0f);
  return true;
}

__device__ bool hit_primitive(const DevicePrimitive &prim,
                              const ray &world_r,
                              float t_min,
                              float t_max,
                              HitRecord &rec) {
  ray local_r = transform_ray_to_local(prim.transform, world_r);
  HitRecord local_rec{};
  bool hit = false;
  switch (prim.type) {
  case DevicePrimitiveType::kSphere:
    hit = hit_sphere(prim, local_r, t_min, t_max, local_rec);
    break;
  case DevicePrimitiveType::kXYRect:
    hit = hit_rect_xy(prim, local_r, t_min, t_max, local_rec);
    break;
  case DevicePrimitiveType::kXZRect:
    hit = hit_rect_xz(prim, local_r, t_min, t_max, local_rec);
    break;
  case DevicePrimitiveType::kYZRect:
    hit = hit_rect_yz(prim, local_r, t_min, t_max, local_rec);
    break;
  }
  if (hit) {
    rec = local_rec;
    rec.normal = transform_normal_to_world(
        prim.transform, rec.normal, (prim.flags & 1) != 0);
    rec.p = rotate_y(rec.p, prim.transform.cos_theta, prim.transform.sin_theta) +
            prim.transform.translation;
    rec.material_index = prim.material_index;
  }
  return hit;
}

__device__ bool hit_scene(DeviceSceneView scene,
                          const ray &r,
                          float t_min,
                          float t_max,
                          HitRecord &rec) {
  bool hit_anything = false;
  float closest = t_max;
  for (std::size_t i = 0; i < scene.primitive_count; ++i) {
    HitRecord temp{};
    if (hit_primitive(scene.primitives[i], r, t_min, closest, temp)) {
      hit_anything = true;
      closest = temp.t;
      rec = temp;
    }
  }
  return hit_anything;
}

__device__ vec3 emitted(const DeviceMaterial &mat,
                        const vec3 &dir,
                        const vec3 &normal) {
  if (mat.type == DeviceMaterialType::kDiffuseLight &&
      dot(normal, dir) < 0.0f) {
    return mat.emit;
  }
  return vec3(0.0f, 0.0f, 0.0f);
}

__device__ float cosine_pdf(const vec3 &dir, const vec3 &n) {
  float cosine = dot(unit_vector(dir), n);
  return cosine > 0.0f ? cosine / kPi : 0.0f;
}

__device__ vec3 sample_light_direction(DeviceSceneView scene,
                                       int light_index,
                                       const vec3 &origin,
                                       Rng &rng,
                                       float &pdf) {
  pdf = 0.0f;
  const DevicePrimitive &light = scene.primitives[light_index];
  if (light.type == DevicePrimitiveType::kXZRect) {
    float x = light.rect.x0 +
              rng.next() * (light.rect.x1 - light.rect.x0);
    float z = light.rect.y0 +
              rng.next() * (light.rect.y1 - light.rect.y0);
    vec3 local_point(x, light.rect.k, z);
    vec3 world_point =
        rotate_y(local_point, light.transform.cos_theta, light.transform.sin_theta) +
        light.transform.translation;
    vec3 dir = world_point - origin;
    float distance_squared = dir.squared_length();
    dir = unit_vector(dir);
    float area = (light.rect.x1 - light.rect.x0) *
                 (light.rect.y1 - light.rect.y0);
    float cosine = fabsf(dot(dir, vec3(0.0f, 1.0f, 0.0f)));
    pdf = distance_squared / (cosine * area);
    return dir;
  }
  // Fallback: uniform hemisphere
  vec3 d = random_in_unit_sphere(rng);
  pdf = 1.0f / (4.0f * M_PI);
  return unit_vector(d);
}

__device__ float light_pdf_value(DeviceSceneView scene,
                                 int light_index,
                                 const vec3 &origin,
                                 const vec3 &dir) {
  if (light_index < 0) {
    return 0.0f;
  }
  const DevicePrimitive &light = scene.primitives[light_index];
  if (light.type == DevicePrimitiveType::kXZRect) {
    // Reuse CPU logic
    HitRecord rec{};
    ray r(origin, dir);
    if (hit_rect_xz(light, transform_ray_to_local(light.transform, r),
                    0.001f, FLT_MAX, rec)) {
      float area = (light.rect.x1 - light.rect.x0) *
                   (light.rect.y1 - light.rect.y0);
      float distance_squared = rec.t * rec.t * dir.squared_length();
      float cosine = fabsf(dot(dir, vec3(0.0f, 1.0f, 0.0f)) / dir.length());
      return distance_squared / (cosine * area);
    }
  }
  return 0.0f;
}

__device__ bool valid_material(DeviceSceneView scene, int index) {
  return index >= 0 &&
         index < static_cast<int>(scene.material_count) &&
         scene.materials != nullptr;
}

__device__ int valid_light_index(DeviceSceneView scene) {
  if (scene.light_primitive < 0 ||
      scene.light_primitive >= static_cast<int>(scene.primitive_count)) {
    return -1;
  }
  return scene.light_primitive;
}

__device__ vec3 shade(const ray &r_in,
                      DeviceSceneView scene,
                      Rng &rng,
                      int depth) {
  if (depth > 50) {
    return vec3(0.0f, 0.0f, 0.0f);
  }
  HitRecord rec{};
  if (hit_scene(scene, r_in, 0.001f, FLT_MAX, rec)) {
    if (!valid_material(scene, rec.material_index)) {
      return vec3(0.0f, 0.0f, 0.0f);
    }
    const DeviceMaterial &mat = scene.materials[rec.material_index];
    vec3 emit = emitted(mat, r_in.direction(), rec.normal);

    if (mat.type == DeviceMaterialType::kDiffuseLight) {
      return emit;
    }

    if (mat.type == DeviceMaterialType::kMetal ||
        mat.type == DeviceMaterialType::kDielectric) {
      vec3 attenuation(1.0f, 1.0f, 1.0f);
      ray scattered;
      if (mat.type == DeviceMaterialType::kMetal) {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p,
                        reflected + mat.fuzz * random_in_unit_sphere(rng),
                        r_in.time());
        attenuation = mat.albedo;
        if (dot(scattered.direction(), rec.normal) <= 0.0f) {
          return emit;
        }
      } else { // dielectric
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
          outward_normal = -rec.normal;
          ni_over_nt = mat.ref_idx;
          cosine = mat.ref_idx * dot(r_in.direction(), rec.normal) /
                   r_in.direction().length();
        } else {
          outward_normal = rec.normal;
          ni_over_nt = 1.0f / mat.ref_idx;
          cosine = -dot(r_in.direction(), rec.normal) /
                   r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
          reflect_prob = schlick(cosine, mat.ref_idx);
        } else {
          reflect_prob = 1.0f;
        }
        if (rng.next() < reflect_prob) {
          scattered = ray(rec.p, reflected, r_in.time());
        } else {
          scattered = ray(rec.p, refracted, r_in.time());
        }
      }
      return emit + attenuation * shade(scattered, scene, rng, depth + 1);
    }

    // Lambertian with mixture PDF (light + cosine)
    vec3 attenuation = mat.albedo;
    vec3 direction;

    int light_index = valid_light_index(scene);
    if (light_index >= 0 && rng.next() < 0.5f) {
      float light_pdf = 0.0f;
      direction =
          sample_light_direction(scene, light_index, rec.p, rng, light_pdf);
    } else {
      onb uvw;
      uvw.build_from_w(rec.normal);
      direction = uvw.local(random_cosine_direction(rng));
    }

    float cosine_pdf_val = cosine_pdf(direction, rec.normal);
    float pdf = cosine_pdf_val;
    if (light_index >= 0) {
      float light_pdf =
          light_pdf_value(scene, light_index, rec.p, direction);
      pdf = 0.5f * light_pdf + 0.5f * cosine_pdf_val;
    }

    if (pdf <= 0.0f) {
      return emit;
    }

    ray scattered(rec.p, unit_vector(direction), r_in.time());
    float scattering_pdf = cosine_pdf(scattered.direction(), rec.normal);
    vec3 incoming =
        shade(scattered, scene, rng, depth + 1);
    return emit +
           attenuation * scattering_pdf * incoming / pdf;
  }
  return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void render_kernel(float3 *pixels,
                              DeviceSceneView scene,
                              DeviceCamera cam,
                              int width,
                              int height,
                              int samples_per_pixel,
                              uint32_t base_seed) {
  int px = blockIdx.x * kTileX + threadIdx.x;
  int py = blockIdx.y * kTileY + threadIdx.y;
  bool active = (px < width && py < height);

  extern __shared__ float tile_accum[];
  int tile_idx = threadIdx.y * kTileX + threadIdx.x;
  float *accum = tile_accum + tile_idx * 3;

  if (threadIdx.z == 0) {
    accum[0] = accum[1] = accum[2] = 0.0f;
  }
  __syncthreads();

  uint32_t pixel_hash =
      static_cast<uint32_t>(py * width + px) ^ static_cast<uint32_t>(samples_per_pixel);
  uint32_t seed = base_seed ^ (pixel_hash * 9781u);
  Rng rng(seed);

  vec3 local_sum(0.0f, 0.0f, 0.0f);
  if (active) {
    for (int s = threadIdx.z; s < samples_per_pixel; s += blockDim.z) {
      float u =
          (static_cast<float>(px) + rng.next()) / static_cast<float>(width);
      float v =
          (static_cast<float>(py) + rng.next()) / static_cast<float>(height);
      ray r = camera_get_ray(cam, u, v, rng);
      local_sum += shade(r, scene, rng, 0);
    }
  }

  atomicAdd(accum + 0, local_sum.x());
  atomicAdd(accum + 1, local_sum.y());
  atomicAdd(accum + 2, local_sum.z());

  __syncthreads();

  if (threadIdx.z == 0 && active) {
    vec3 col(accum[0], accum[1], accum[2]);
    col /= static_cast<float>(samples_per_pixel);
    int idx = py * width + px;
    pixels[idx] = make_float3(col.x(), col.y(), col.z());
  }
}

DeviceCamera make_device_camera(const camera &cam) {
  DeviceCamera dc;
  dc.origin = cam.origin;
  dc.lower_left_corner = cam.lower_left_corner;
  dc.horizontal = cam.horizontal;
  dc.vertical = cam.vertical;
  dc.u = cam.u;
  dc.v = cam.v;
  dc.w = cam.w;
  dc.lens_radius = cam.lens_radius;
  dc.time0 = cam.time0;
  dc.time1 = cam.time1;
  return dc;
}

} // namespace

CudaRenderer::CudaRenderer()
    : block_dim_(kTileX, kTileY, kSampleLanes),
      max_threads_per_sm_(128),
      base_seed_(5489u) {}

CudaRenderer::LaunchDimensions CudaRenderer::launch_dimensions(int width,
                                                               int height) const {
  dim3 block = block_dim_;
  dim3 grid((width + kTileX - 1) / kTileX,
            (height + kTileY - 1) / kTileY,
            1);
  return {grid, block};
}

std::vector<vec3> CudaRenderer::render(const DeviceScene &scene,
                                       const camera &cam,
                                       int width,
                                       int height,
                                       int samples_per_pixel) const {
  std::vector<vec3> host_pixels(static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height),
                                vec3(0.0f, 0.0f, 0.0f));
  if (width == 0 || height == 0 || scene.empty()) {
    return host_pixels;
  }

  DeviceBuffer pixel_buffer;
  pixel_buffer.allocate(host_pixels.size() * sizeof(float3));

  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 65536));

  LaunchDimensions dims = launch_dimensions(width, height);
  std::size_t shared_bytes =
      static_cast<std::size_t>(kTileX * kTileY * 3) * sizeof(float);

  DeviceSceneView view = scene.view();
  DeviceCamera dcam = make_device_camera(cam);

  render_kernel<<<dims.grid, dims.block, shared_bytes>>>(
      reinterpret_cast<float3 *>(pixel_buffer.data()),
      view,
      dcam,
      width,
      height,
      samples_per_pixel,
      base_seed_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_pixels.data(),
                        pixel_buffer.data(),
                        host_pixels.size() * sizeof(float3),
                        cudaMemcpyDeviceToHost));

  return host_pixels;
}
