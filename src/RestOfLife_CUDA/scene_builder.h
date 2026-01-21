#ifndef SCENE_BUILDER_H
#define SCENE_BUILDER_H

#include "device_scene.h"

#include <unordered_map>
#include <vector>

class hittable;
class material;
class flip_normals;
class translate;
class rotate_y;
class hittable_list;
class sphere;
class xy_rect;
class xz_rect;
class yz_rect;
class box;

// converts the CPU hittable world into flat buffers ready for GPU upload
class SceneBuilder {
 public:
  SceneBuilder() = default;

  DeviceScene build_from(const hittable& world);

 private:
  struct BuildCtx {
    DeviceTransform transform;
    bool flip = false;
  };

  void serialize_world(const hittable& world);
  void append_hittable(const hittable& obj, BuildCtx ctx);
  void add_primitive(DevicePrimitive primitive, const material* mat_ptr);

  int material_index(const material* mat_ptr);

  std::vector<DevicePrimitive> primitives_;
  std::vector<DeviceMaterial> materials_;
  std::unordered_map<const material*, int> material_lut_;
  int light_primitive_ = -1;
};

#endif  // SCENE_BUILDER_H
