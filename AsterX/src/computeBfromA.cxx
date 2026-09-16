#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <array>

#include "aster_utils.hxx"

struct metric {
  CCTK_REAL gxx, gxy, gxz, gyy, gyz, gzz;
};

namespace AsterX {
using namespace Loop;
using namespace Arith;
using namespace AsterUtils;

// dB*_stag = curl(Avec) on the faces. One pass over the whole component
// (loop_all_device: the union over tiles is the full box). At every point the
// derivative along each direction uses the highest order in
// {mag_correction_order, ..., 2} whose forward-midpoint stencil fits inside
// the Avec component being differentiated, so the order is reduced only along
// the direction and only at the points where the full stencil would leave the
// component (the outermost ghost or outer-boundary layers).
template <int dir> void ComputeStaggeredB(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_ComputedBstagFromA;
  DECLARE_CCTK_PARAMETERS;

  static_assert(dir >= 0 && dir < 3, "");

  constexpr array<int, dim> face_centred = {!(dir == 0), !(dir == 1),
                                            !(dir == 2)};

  // Extents of the edge-centred Avec components: each Avec_d is cell-centred
  // along d (one point fewer than lsh) and vertex-centred along the other two
  // (lsh points). The curl differentiates Avec only along its vertex-centred
  // directions. Computed from grid.lsh (component-wide, not tile bounds) and
  // captured by value so the kernel's order choice is tile-independent.
  const vect<int, dim> ext_Avec_x = grid.lsh - vect<int, dim>{1, 0, 0};
  const vect<int, dim> ext_Avec_y = grid.lsh - vect<int, dim>{0, 1, 0};
  const vect<int, dim> ext_Avec_z = grid.lsh - vect<int, dim>{0, 0, 1};
  const int max_order = mag_correction_order;

  grid.loop_all_device<face_centred[0], face_centred[1], face_centred[2]>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        if (dir == 0) {
          /* dBx is curl(A) at (i-1/2,j,k): d_y Avec_z - d_z Avec_y */
          const int oy = fitting_order(max_order, p.I[1],
                                       ext_Avec_z[1] - 1 - p.I[1]);
          const int oz = fitting_order(max_order, p.I[2],
                                       ext_Avec_y[2] - 1 - p.I[2]);
          dBx_stag(p.I) = calc_fd_forward_midpoint<1>(Avec_z, p, oy) -
                          calc_fd_forward_midpoint<2>(Avec_y, p, oz);
        } else if (dir == 1) {
          /* dBy is curl(A) at (i,j-1/2,k): d_z Avec_x - d_x Avec_z */
          const int oz = fitting_order(max_order, p.I[2],
                                       ext_Avec_x[2] - 1 - p.I[2]);
          const int ox = fitting_order(max_order, p.I[0],
                                       ext_Avec_z[0] - 1 - p.I[0]);
          dBy_stag(p.I) = calc_fd_forward_midpoint<2>(Avec_x, p, oz) -
                          calc_fd_forward_midpoint<0>(Avec_z, p, ox);
        } else {
          /* dBz is curl(A) at (i,j,k-1/2): d_x Avec_y - d_y Avec_x */
          const int ox = fitting_order(max_order, p.I[0],
                                       ext_Avec_y[0] - 1 - p.I[0]);
          const int oy = fitting_order(max_order, p.I[1],
                                       ext_Avec_x[1] - 1 - p.I[1]);
          dBz_stag(p.I) = calc_fd_forward_midpoint<0>(Avec_y, p, ox) -
                          calc_fd_forward_midpoint<1>(Avec_x, p, oy);
        }
      });
}

extern "C" void AsterX_ComputedBstagFromA(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_ComputedBstagFromA;
  DECLARE_CCTK_PARAMETERS;

  ComputeStaggeredB<0>(cctkGH);
  ComputeStaggeredB<1>(cctkGH);
  ComputeStaggeredB<2>(cctkGH);
}

// dB = face-to-cell average of dB*_stag, one pass over the whole component
// with the same per-point, per-direction order rule as ComputeStaggeredB.
extern "C" void AsterX_ComputedBFromdBstag(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_ComputedBFromdBstag;
  DECLARE_CCTK_PARAMETERS;

  // Extents of the face-centred dB*_stag components: lsh points along their
  // own (vertex-centred) axis, lsh - 1 along the other two. The average to the
  // cell centre reads along the own axis only.
  const vect<int, dim> ext_dBx_stag = grid.lsh - vect<int, dim>{0, 1, 1};
  const vect<int, dim> ext_dBy_stag = grid.lsh - vect<int, dim>{1, 0, 1};
  const vect<int, dim> ext_dBz_stag = grid.lsh - vect<int, dim>{1, 1, 0};
  const int max_order = mag_correction_order;

  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        /* Interpolation of staggered B components to cell center */
        const int ox = fitting_order(max_order, p.I[0],
                                     ext_dBx_stag[0] - 1 - p.I[0]);
        const int oy = fitting_order(max_order, p.I[1],
                                     ext_dBy_stag[1] - 1 - p.I[1]);
        const int oz = fitting_order(max_order, p.I[2],
                                     ext_dBz_stag[2] - 1 - p.I[2]);
        dBx(p.I) = calc_avg_f2c(dBx_stag, p, 0, ox);
        dBy(p.I) = calc_avg_f2c(dBy_stag, p, 1, oy);
        dBz(p.I) = calc_avg_f2c(dBz_stag, p, 2, oz);
      });
}

extern "C" void AsterX_ComputeBFromdB(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_ComputeBFromdB;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        /* Interpolate metric terms from vertices to center */
        metric g;
        g.gxx = calc_avg_v2c(gxx, p);
        g.gxy = calc_avg_v2c(gxy, p);
        g.gxz = calc_avg_v2c(gxz, p);
        g.gyy = calc_avg_v2c(gyy, p);
        g.gyz = calc_avg_v2c(gyz, p);
        g.gzz = calc_avg_v2c(gzz, p);

        /* Determinant of spatial metric */
        const smat<CCTK_REAL, 3> gmat{g.gxx, g.gxy, g.gxz, g.gyy, g.gyz, g.gzz};
        const CCTK_REAL sqrt_detg = sqrt(calc_det(gmat));

        /* Second order interpolation of staggered B components to cell center
         */
        Bvecx(p.I) = dBx(p.I) / sqrt_detg;
        Bvecy(p.I) = dBy(p.I) / sqrt_detg;
        Bvecz(p.I) = dBz(p.I) / sqrt_detg;
      });
}

} // namespace AsterX
