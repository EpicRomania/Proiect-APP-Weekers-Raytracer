#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "cuda_support.h"
#include "device_scene.h"

#include "common/camera.h"

#include <cstdint>
#include <vector>

class CudaRenderer {
 public:
  struct LaunchDimensions {
    dim3 grid;
    dim3 block;
  };

  CudaRenderer();

  void set_seed(uint32_t seed) { base_seed_ = seed; }

  std::vector<vec3> render(const DeviceScene& scene,
                           const camera& cam,
                           int width,
                           int height,
                           int samples_per_pixel) const;

  LaunchDimensions launch_dimensions(int width, int height) const;

 private:
  dim3 block_dim_;
  int max_threads_per_sm_;
  uint32_t base_seed_;
};

#endif  // CUDA_RENDERER_H
