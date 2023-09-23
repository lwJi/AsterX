#ifndef ASTERX_INTERP_HXX
#define ASTERX_INTERP_HXX

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <mat.hxx>
#include <vec.hxx>
#include <sum.hxx>
#include <simd.hxx>

#include <algorithm>
#include <array>
#include <cmath>

namespace AsterX {
using namespace std;
using namespace Loop;
using namespace Arith;

// Second-order average of vertex-centered grid functions to cell center
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_v2c(const GF3D2<const T> &gf, const PointDesc &p) {
  T gf_avg = 0.0;

  for (int dk = 0; dk < 2; ++dk) {
    for (int dj = 0; dj < 2; ++dj) {
      for (int di = 0; di < 2; ++di) {
        gf_avg += gf(p.I + p.DI[0] * di + p.DI[1] * dj + p.DI[2] * dk);
      }
    }
  }
  return gf_avg / 8.0;
}

// Second-order average of cell-centered grid functions to face-centered
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_c2f(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  T gf_avg = 0.0;
  const vect<T, 4> wt = {-1.0/16, 9.0/16, 9.0/16, -1.0/16};

  for (int di = 0; di < 4; ++di) {
    gf_avg += gf(p.I + p.DI[dir] * (di - 2)) * wt[di];
  }
  return gf_avg;
}

// Second-order average of edge-centered grid functions to vertex-centered
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_e2v(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  T gf_avg = 0.0;

  for (int di = 0; di < 2; ++di) {
    gf_avg += gf(p.I - p.DI[dir] * di);
  }
  return gf_avg / 2.0;
}

// Second-order average of edge-centered grid functions (along dir) to cell
// center
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_e2c(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  T gf_avg = 0.0;
  const int j = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  const int k = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);
  for (int dk = 0; dk < 2; ++dk) {
    for (int dj = 0; dj < 2; ++dj) {
      gf_avg += gf(p.I + p.DI[j] * dj + p.DI[k] * dk);
    }
  }
  return gf_avg / 4.0;
}

// Second-order average of vertex-centered grid functionsto face
// center (perp to dir)
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_v2f(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  T gf_avg = 0.0;
  const int j = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  const int k = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);
  for (int dk = 0; dk < 2; ++dk) {
    for (int dj = 0; dj < 2; ++dj) {
      gf_avg += gf(p.I + p.DI[j] * dj + p.DI[k] * dk);
    }
  }
  return gf_avg / 4.0;
}

// Second-order average of cell-centered grid functions to edge center
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_c2e(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  T gf_avg = 0.0;
  const int j = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  const int k = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);
  for (int dk = 0; dk < 2; ++dk) {
    for (int dj = 0; dj < 2; ++dj) {
      gf_avg += gf(p.I - p.DI[j] * dj - p.DI[k] * dk);
    }
  }
  return gf_avg / 4.0;
}

template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_c2v(const GF3D2<const T> &gf, const PointDesc &p) {
  T gf_avg = 0.0;

  for (int dk = 0; dk < 2; ++dk) {
    for (int dj = 0; dj < 2; ++dj) {
      for (int di = 0; di < 2; ++di) {
        gf_avg += gf(p.I - p.DI[0] * di - p.DI[1] * dj - p.DI[2] * dk);
      }
    }
  }
  return gf_avg / 8.0;
}

} // namespace AsterX

#endif // ASTERX_INTERP_HXX
