#ifndef DEVICE_SCENE_H
#define DEVICE_SCENE_H

#include "cuda_support.h"

#include "common/ray.h"
#include "common/vec3.h"

#include <cstddef>

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

  void allocate(std::size_t bytes);
  void upload(const void *data, std::size_t bytes);
  void reset();

  void *data() { return data_; }
  const void *data() const { return data_; }
  std::size_t size() const { return size_; }

private:
  void *data_ = nullptr;
  std::size_t size_ = 0;
};

enum class DeviceMaterialType : int {
  kLambertian = 0,
  kMetal = 1,
  kDielectric = 2,
  kDiffuseLight = 3,
};

struct DeviceMaterial {
  DeviceMaterialType type = DeviceMaterialType::kLambertian;
  vec3 albedo = vec3(0.0f, 0.0f, 0.0f);
  vec3 emit = vec3(0.0f, 0.0f, 0.0f);
  float fuzz = 0.0f;      // only for metal
  float ref_idx = 1.0f;   // only for dielectric
};

enum class DevicePrimitiveType : int {
  kSphere = 0,
  kXYRect = 1,
  kXZRect = 2,
  kYZRect = 3,
};

// Only rotation around Y plus translation are needed for the current scenes.
struct DeviceTransform {
  float cos_theta = 1.0f;
  float sin_theta = 0.0f;
  vec3 translation = vec3(0.0f, 0.0f, 0.0f);
};

struct DeviceSphere {
  vec3 center = vec3(0.0f, 0.0f, 0.0f);
  float radius = 0.0f;
};

struct DeviceRect {
  float x0 = 0.0f;
  float x1 = 0.0f;
  float y0 = 0.0f;
  float y1 = 0.0f;
  float k = 0.0f;
};

struct DevicePrimitive {
  DevicePrimitiveType type = DevicePrimitiveType::kSphere;
  int material_index = -1;
  int flags = 0;  // bit0: flip normal
  DeviceTransform transform{};
  DeviceSphere sphere{};
  DeviceRect rect{};
};

struct DeviceSceneView {
  const DevicePrimitive *primitives = nullptr;
  const DeviceMaterial *materials = nullptr;
  std::size_t primitive_count = 0;
  std::size_t material_count = 0;
  int light_primitive = -1;
};

class DeviceScene {
public:
  struct Metadata {
    std::size_t primitive_count = 0;
    std::size_t material_count = 0;
    int light_primitive = -1;
  };

  DeviceScene() = default;
  DeviceScene(DeviceScene &&) noexcept = default;
  DeviceScene &operator=(DeviceScene &&) noexcept = default;

  void adopt(DeviceBuffer primitives,
             DeviceBuffer materials,
             Metadata metadata);

  const DeviceBuffer &primitive_buffer() const { return primitives_; }
  const DeviceBuffer &material_buffer() const { return materials_; }
  const Metadata &metadata() const { return metadata_; }
  bool empty() const { return primitives_.size() == 0; }

  DeviceSceneView view() const;

private:
  DeviceBuffer primitives_;
  DeviceBuffer materials_;
  Metadata metadata_;
};

#endif // DEVICE_SCENE_H
