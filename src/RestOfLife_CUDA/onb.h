#ifndef ONB_H
#define ONB_H

#include "common/rtweekend.h"

class onb {
 public:
  CUDA_HD onb() {}
  CUDA_HD inline vec3 operator[](int i) const { return axis[i]; }
  CUDA_HD vec3 u() const       { return axis[0]; }
  CUDA_HD vec3 v() const       { return axis[1]; }
  CUDA_HD vec3 w() const       { return axis[2]; }
  CUDA_HD vec3 local(float a, float b, float c) const { return a*u() + b*v() + c*w(); }
  CUDA_HD vec3 local(const vec3& a) const { return a.x()*u() + a.y()*v() + a.z()*w(); }
  CUDA_HD void build_from_w(const vec3&);
  vec3 axis[3];
};


CUDA_HD inline void onb::build_from_w(const vec3& n) {
  axis[2] = unit_vector(n);
  vec3 a;
  // snap to a cartesion axis
  if (fabs(w().x()) > 0.9)
    a = vec3(0, 1, 0);
  else
    a = vec3(1, 0, 0);
  axis[1] = unit_vector(cross(w(), a));
  axis[0] = cross(w(), v());
}

#endif
