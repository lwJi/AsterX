#ifndef ASTER_FD_HXX
#define ASTER_FD_HXX

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <mat.hxx>
#include <simd.hxx>
#include <sum.hxx>
#include <vec.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

namespace AsterUtils {
using namespace std;
using namespace Loop;
using namespace Arith;

// FD2: vertex centered input, vertex centered output, oneside stencil
template <int Sign, typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_fd2_v2v_oneside(const GF3D2<const T> &gf, const PointDesc &p,
                     const int dir) {
  static_assert(Sign == +1 || Sign == -1, "Sign must be +1 or -1");
  constexpr int s = Sign;

  const auto i0 = p.I;
  const auto i1 = i0 + s * p.DI[dir];
  const auto i2 = i0 + 2 * s * p.DI[dir];

  const T f0 = gf(i0);
  const T f1 = gf(i1);
  const T f2 = gf(i2);

  const T num = T(s) * (T(-3) * f0 + T(4) * f1 - f2);
  return num * (T(0.5) / p.DX[dir]);
}

// FD2: vertex centered input, edge centered output
template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_fd2_v2e(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  return (gf(p.I + p.DI[dir]) - gf(p.I)) / p.DX[dir];
}

template <typename T>
CCTK_DEVICE CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_fd2_e2v(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  return (gf(p.I) - gf(p.I - p.DI[dir])) / p.DX[dir];
}

// FD2: vertex centered input, cell centered output
template <int FDORDER, typename T>
CCTK_DEVICE CCTK_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline std::enable_if_t<FDORDER == 2, T>
    calc_fd_v2c(const GF3D2<const T> &gf, const PointDesc &p, int dir) {
  T dgf1, dgf2, dgf3, dgf4;
  const int dir1 = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  const int dir2 = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);

  dgf1 = (gf(p.I + p.DI[dir]) - gf(p.I)) / p.DX[dir];
  dgf2 = (gf(p.I + p.DI[dir1] + p.DI[dir]) - gf(p.I + p.DI[dir1])) / p.DX[dir];
  dgf3 = (gf(p.I + p.DI[dir2] + p.DI[dir]) - gf(p.I + p.DI[dir2])) / p.DX[dir];
  dgf4 = (gf(p.I + p.DI[dir1] + p.DI[dir2] + p.DI[dir]) -
          gf(p.I + p.DI[dir1] + p.DI[dir2])) /
         p.DX[dir];
  return 0.25 * (dgf1 + dgf2 + dgf3 + dgf4);
}

// FD4: vertex centered input, cell centered output
// Interpolation from edges to cell centers is 2nd order
template <int FDORDER, typename T>
CCTK_DEVICE CCTK_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline std::enable_if_t<FDORDER == 4, T>
    calc_fd_v2c(const GF3D2<const T> &gf, const PointDesc &p, int dir) {
  T dgf1, dgf2, dgf3, dgf4;

  const int dir1 = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  const int dir2 = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);

  dgf1 = ((1.0 / 24.0) * gf(p.I - p.DI[dir]) - (27.0 / 24.0) * gf(p.I) +
          (27.0 / 24.0) * gf(p.I + p.DI[dir]) -
          (1.0 / 24.0) * gf(p.I + 2 * p.DI[dir])) /
         p.DX[dir];

  dgf2 = ((1.0 / 24.0) * gf(p.I + p.DI[dir1] - p.DI[dir]) -
          (27.0 / 24.0) * gf(p.I + p.DI[dir1]) +
          (27.0 / 24.0) * gf(p.I + p.DI[dir1] + p.DI[dir]) -
          (1.0 / 24.0) * gf(p.I + p.DI[dir1] + 2 * p.DI[dir])) /
         p.DX[dir];

  dgf3 = ((1.0 / 24.0) * gf(p.I + p.DI[dir2] - p.DI[dir]) -
          (27.0 / 24.0) * gf(p.I + p.DI[dir2]) +
          (27.0 / 24.0) * gf(p.I + p.DI[dir2] + p.DI[dir]) -
          (1.0 / 24.0) * gf(p.I + p.DI[dir2] + 2 * p.DI[dir])) /
         p.DX[dir];

  dgf4 = ((1.0 / 24.0) * gf(p.I + p.DI[dir1] + p.DI[dir2] - p.DI[dir]) -
          (27.0 / 24.0) * gf(p.I + p.DI[dir1] + p.DI[dir2]) +
          (27.0 / 24.0) * gf(p.I + p.DI[dir1] + p.DI[dir2] + p.DI[dir]) -
          (1.0 / 24.0) * gf(p.I + p.DI[dir1] + p.DI[dir2] + 2 * p.DI[dir])) /
         p.DX[dir];

  return 0.25 * (dgf1 + dgf2 + dgf3 + dgf4);
}

// FD2: cell centered input, cell centered output
template <int FDORDER, typename T>
CCTK_DEVICE CCTK_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline std::enable_if_t<FDORDER == 2, T>
    calc_fd_c2c(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  return (0.5 / p.DX[dir]) * (gf(p.I + p.DI[dir]) - gf(p.I - p.DI[dir]));
}

// FD4: cell centered input, cell centered output
template <int FDORDER, typename T>
CCTK_DEVICE CCTK_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline std::enable_if_t<FDORDER == 4, T>
    calc_fd_c2c(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  return (1.0 / (12.0 * p.DX[dir])) *
         (-gf(p.I + 2 * p.DI[dir]) + 8.0 * gf(p.I + p.DI[dir]) -
          8.0 * gf(p.I - p.DI[dir]) + gf(p.I - 2 * p.DI[dir]));
}

} // namespace AsterUtils

#endif // ASTER_FD_HXX
