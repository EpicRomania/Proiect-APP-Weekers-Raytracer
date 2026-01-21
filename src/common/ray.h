#ifndef RAY_H
#define RAY_H

#include "common/vec3.h"

class ray
{
public:
  CUDA_HD ray() {}
  CUDA_HD ray(const vec3& a, const vec3& b, float ti = 0.0) { A = a; B = b; _time = ti; }
  CUDA_HD vec3 origin() const       { return A; }
  CUDA_HD vec3 direction() const    { return B; }
  CUDA_HD float time() const { return _time ; }
  CUDA_HD vec3 point_at_parameter(float t) const { return A + t*B; }

  vec3 A;
  vec3 B;
  float _time;
};

#endif
