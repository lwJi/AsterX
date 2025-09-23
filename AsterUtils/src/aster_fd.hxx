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
  const int j = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  const int k = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);

  const auto I = p.I;
  const auto dI = p.DI[dir];
  const auto dJ = p.DI[j];
  const auto dK = p.DI[k];
  const auto dJdK = dJ + dK;

  const auto line_fd = [&](const auto &B) CCTK_ATTRIBUTE_ALWAYS_INLINE {
    return gf(B + dI) - gf(B);
  };

  const T num =
      line_fd(I) + line_fd(I + dJ) + line_fd(I + dK) + line_fd(I + dJdK);

  const T inv4dx = T(0.25) / p.DX[dir];
  return num * inv4dx;
}

// FD4: vertex centered input, cell centered output
// Interpolation from edges to cell centers is 2nd order
template <int FDORDER, typename T>
CCTK_DEVICE CCTK_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline std::enable_if_t<FDORDER == 4, T>
    calc_fd_v2c(const GF3D2<const T> &gf, const PointDesc &p, int dir) {
  T dgf1, dgf2, dgf3, dgf4;
  const int j = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  const int k = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);

  const auto I = p.I;
  const auto dI = p.DI[dir];
  const auto dJ = p.DI[j];
  const auto dK = p.DI[k];
  const auto dJdK = dJ + dK;

  const auto line_fd4 = [&](const auto &B) CCTK_ATTRIBUTE_ALWAYS_INLINE {
    const T f_m1 = gf(B - dI);
    const T f_0 = gf(B);
    const T f_p1 = gf(B + dI);
    const T f_p2 = gf(B + dI + dI);
    return (T(1) * f_m1 + T(-27) * f_0 + T(27) * f_p1 + T(-1) * f_p2);
  };

  const T num =
      line_fd4(I) + line_fd4(I + dJ) + line_fd4(I + dK) + line_fd4(I + dJdK);

  const T inv96dx = T(1) / (T(96) * p.DX[dir]);
  return num * inv96dx;
}

// FD2: cell centered input, cell centered output
template <int FDORDER, typename T>
CCTK_DEVICE CCTK_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline std::enable_if_t<FDORDER == 2, T>
    calc_fd_c2c(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  const auto I = p.I;
  const auto dI = p.DI[dir];

  const T fp = gf(I + dI);
  const T fm = gf(I - dI);

  const T inv2dx = T(0.5) / p.DX[dir];
  return (fp - fm) * inv2dx;
}

// FD4: cell centered input, cell centered output
template <int FDORDER, typename T>
CCTK_DEVICE CCTK_HOST
    CCTK_ATTRIBUTE_ALWAYS_INLINE inline std::enable_if_t<FDORDER == 4, T>
    calc_fd_c2c(const GF3D2<const T> &gf, const PointDesc &p, const int dir) {
  const auto I = p.I;
  const auto dI = p.DI[dir];

  const T f_p1 = gf(I + dI);
  const T f_m1 = gf(I - dI);
  const T f_p2 = gf(I + dI + dI);
  const T f_m2 = gf(I - dI - dI);

  const T inv12dx = T(1) / (T(12) * p.DX[dir]);
  return ((T(-1) * f_p2) + (T(8) * f_p1) + (T(-8) * f_m1) + (T(1) * f_m2)) *
         inv12dx;
}

} // namespace AsterUtils

#endif // ASTER_FD_HXX
