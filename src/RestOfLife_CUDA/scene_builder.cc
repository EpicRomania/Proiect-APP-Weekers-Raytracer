#include "scene_builder.h"

#include "aarect.h"
#include "box.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#include "common/texture.h"

#include <utility>

namespace {

vec3 as_constant(const texture *tex) {
  if (const auto *ct = dynamic_cast<const constant_texture *>(tex)) {
    return ct->color;
  }
  return vec3(0.0f, 0.0f, 0.0f);
}

DeviceMaterial to_device_material(const material *mat_ptr) {
  DeviceMaterial mat;
  if (const auto *lam = dynamic_cast<const lambertian *>(mat_ptr)) {
    mat.type = DeviceMaterialType::kLambertian;
    mat.albedo = as_constant(lam->albedo);
  } else if (const auto *m = dynamic_cast<const metal *>(mat_ptr)) {
    mat.type = DeviceMaterialType::kMetal;
    mat.albedo = m->albedo;
    mat.fuzz = m->fuzz;
  } else if (const auto *d = dynamic_cast<const dielectric *>(mat_ptr)) {
    mat.type = DeviceMaterialType::kDielectric;
    mat.ref_idx = d->ref_idx;
    mat.albedo = vec3(1.0f, 1.0f, 1.0f);
  } else if (const auto *l = dynamic_cast<const diffuse_light *>(mat_ptr)) {
    mat.type = DeviceMaterialType::kDiffuseLight;
    mat.emit = as_constant(l->emit);
    mat.albedo = vec3(0.0f, 0.0f, 0.0f);
  }
  return mat;
}

DeviceRect make_rect(float a0, float a1, float b0, float b1, float k) {
  DeviceRect r;
  r.x0 = a0;
  r.x1 = a1;
  r.y0 = b0;
  r.y1 = b1;
  r.k = k;
  return r;
}

DeviceTransform combine_rotation_y(const DeviceTransform &base,
                                   const rotate_y &rot) {
  DeviceTransform t = base;
  // Combine rotations around Y: R(a) then R(b) = R(a + b).
  float c = base.cos_theta * rot.cos_theta - base.sin_theta * rot.sin_theta;
  float s = base.sin_theta * rot.cos_theta + base.cos_theta * rot.sin_theta;
  t.cos_theta = c;
  t.sin_theta = s;
  return t;
}

} // namespace

DeviceScene SceneBuilder::build_from(const hittable &world) {
  primitives_.clear();
  materials_.clear();
  material_lut_.clear();
  light_primitive_ = -1;

  serialize_world(world);

  DeviceBuffer prim_buffer;
  DeviceBuffer mat_buffer;

  if (!primitives_.empty()) {
    prim_buffer.upload(primitives_.data(),
                       primitives_.size() * sizeof(DevicePrimitive));
  }

  if (!materials_.empty()) {
    mat_buffer.upload(materials_.data(),
                      materials_.size() * sizeof(DeviceMaterial));
  }

  DeviceScene::Metadata meta;
  meta.primitive_count = primitives_.size();
  meta.material_count = materials_.size();
  meta.light_primitive = light_primitive_;

  DeviceScene scene;
  scene.adopt(std::move(prim_buffer), std::move(mat_buffer), meta);
  return scene;
}

void SceneBuilder::serialize_world(const hittable &world) {
  BuildCtx ctx;
  ctx.transform.cos_theta = 1.0f;
  ctx.transform.sin_theta = 0.0f;
  ctx.transform.translation = vec3(0.0f, 0.0f, 0.0f);
  append_hittable(world, ctx);
}

void SceneBuilder::append_hittable(const hittable &obj, BuildCtx ctx) {
  if (const auto *list = dynamic_cast<const hittable_list *>(&obj)) {
    for (int i = 0; i < list->list_size; ++i) {
      append_hittable(*list->list[i], ctx);
    }
    return;
  }

  if (const auto *flip = dynamic_cast<const flip_normals *>(&obj)) {
    BuildCtx next = ctx;
    next.flip = !next.flip;
    append_hittable(*flip->ptr, next);
    return;
  }

  if (const auto *tr = dynamic_cast<const translate *>(&obj)) {
    BuildCtx next = ctx;
    next.transform.translation += tr->offset;
    append_hittable(*tr->ptr, next);
    return;
  }

  if (const auto *rot = dynamic_cast<const rotate_y *>(&obj)) {
    BuildCtx next = ctx;
    next.transform = combine_rotation_y(ctx.transform, *rot);
    append_hittable(*rot->ptr, next);
    return;
  }

  if (const auto *bx = dynamic_cast<const box *>(&obj)) {
    // box owns a hittable_list of faces
    if (bx->list_ptr != nullptr) {
      append_hittable(*bx->list_ptr, ctx);
    }
    return;
  }

  if (const auto *s = dynamic_cast<const sphere *>(&obj)) {
    DevicePrimitive prim;
    prim.type = DevicePrimitiveType::kSphere;
    prim.material_index = material_index(s->mat_ptr);
    prim.flags = ctx.flip ? 1 : 0;
    prim.transform = ctx.transform;
    prim.sphere.center = s->center;
    prim.sphere.radius = s->radius;
    add_primitive(prim, s->mat_ptr);
    return;
  }

  if (const auto *xy = dynamic_cast<const xy_rect *>(&obj)) {
    DevicePrimitive prim;
    prim.type = DevicePrimitiveType::kXYRect;
    prim.material_index = material_index(xy->mp);
    prim.flags = ctx.flip ? 1 : 0;
    prim.transform = ctx.transform;
    prim.rect = make_rect(xy->x0, xy->x1, xy->y0, xy->y1, xy->k);
    add_primitive(prim, xy->mp);
    return;
  }

  if (const auto *xz = dynamic_cast<const xz_rect *>(&obj)) {
    DevicePrimitive prim;
    prim.type = DevicePrimitiveType::kXZRect;
    prim.material_index = material_index(xz->mp);
    prim.flags = ctx.flip ? 1 : 0;
    prim.transform = ctx.transform;
    prim.rect = make_rect(xz->x0, xz->x1, xz->z0, xz->z1, xz->k);
    add_primitive(prim, xz->mp);
    return;
  }

  if (const auto *yz = dynamic_cast<const yz_rect *>(&obj)) {
    DevicePrimitive prim;
    prim.type = DevicePrimitiveType::kYZRect;
    prim.material_index = material_index(yz->mp);
    prim.flags = ctx.flip ? 1 : 0;
    prim.transform = ctx.transform;
    prim.rect = make_rect(yz->y0, yz->y1, yz->z0, yz->z1, yz->k);
    add_primitive(prim, yz->mp);
    return;
  }
}

void SceneBuilder::add_primitive(DevicePrimitive primitive,
                                 const material *mat_ptr) {
  if (light_primitive_ == -1) {
    const auto *light = dynamic_cast<const diffuse_light *>(mat_ptr);
    if (light != nullptr) {
      light_primitive_ = static_cast<int>(primitives_.size());
    }
  }
  primitives_.push_back(primitive);
}

int SceneBuilder::material_index(const material *mat_ptr) {
  auto it = material_lut_.find(mat_ptr);
  if (it != material_lut_.end()) {
    return it->second;
  }
  int idx = static_cast<int>(materials_.size());
  materials_.push_back(to_device_material(mat_ptr));
  material_lut_.emplace(mat_ptr, idx);
  return idx;
}
