#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include "aster_fd.hxx"
#include "gauge_register.hxx"
#include "sync.hxx"

#include <vector>

namespace AsterX {
using namespace Loop;
using namespace Arith;
using namespace AsterUtils;

////////////////////////////////////////////////////////////////////////////////
// Initialization

extern "C" void AsterX_GaugeRegisterInit(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeRegisterInit;

  // IG = 0 is the state at the end of a full cascade (all levels agree), so
  // the first step of a run is reconciled normally.
  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p)
                            CCTK_ATTRIBUTE_ALWAYS_INLINE { IG(p.I) = 0.0; });
}

////////////////////////////////////////////////////////////////////////////////
// The reconcile, AsterX_GaugeRegisterGroup IN CarpetX_PreRestrict
//
// CarpetX traverses CarpetX_PreRestrict once per time-aligned pass, with
// active_levels widened to every level that is at the current time,
// immediately before it restricts the evolved variables fine-to-coarse. The
// group applies the gauge shift on the coarse levels there, and the driver's
// own transfers that follow do the rest: its Restrict resets every fine-owned
// coarse edge from the fine level, and its ProlongateRestrictedGFs refills
// the fine Avec_* ghost halo from the corrected coarse interior. Nothing in
// this group moves Avec_* across levels. ODESolvers_PostStep in
// CCTK_POSTRESTRICT runs after those transfers, so everything downstream on
// this pass (B from A, con2prim, G, E, output) sees the corrected edges.
//
// With IGr the ledger restricted from the aligned child (and IGr = IG on
// every other point), per aligned pair on the coarse level:
//   A_i -= D_i IGr - D_i IG   on the interior edges whose stencil touches a
//                             restricted node   a pure gauge transformation
//   IG := IGr                 on the restricted nodes   the coarse ledger
//                             adopts the fine
// and on a full cascade (window reaches level 0) IG := 0 on every level.
//
// Band. Everywhere else IGr == IG bit for bit, so the shift is exactly zero
// and the adopt would rewrite the value already there; those points are
// simply not visited. The set of points that can differ is exact geometry:
// the nodal restriction writes the coarse nodes of the child's boxes
// coarsened by 2 (faces included), and the forward-midpoint stencil of an
// edge reaches reach = mag_correction_order/2 - 1 vertices past its
// endpoints. AsterX_GaugeRestrictIGr has sync.cxx compute that band once per
// pass from the child's current AMReX box array (a few local box
// intersections per component, no communication), and the two band kernels
// loop the stored boxes with loop_box_device. On a level without an aligned
// child the band is empty and both kernels do nothing. The copy IGr = IG and
// the full-cascade IG = 0 stay whole-level: the copy is what makes IGr == IG
// off the band (and IGr, being non-checkpointed scratch, must be written
// everywhere it is declared), and the zero is what keeps the coarse ledger
// bounded on the nodes a regrid would prolongate.
//
// Transfers. The one transfer the correction depends on is the nodal
// restriction of IGr from each aligned child in AsterX_GaugeRestrictIGr. At
// mag_correction_order 2 the edge stencil reads only the two endpoints of
// each interior edge, which are interior vertices, so nothing reads a ghost
// vertex of either ledger and the register issues no other communication.
// Above order 2 the stencil reads one (order 4) or two (order 6) ghost
// vertices of IGr and IG; AsterX_GaugeRestrictIGr then also issues one
// same-level ghost exchange of both ledgers, right after the restriction, and
// that in-pass sync is the only source of ledger ghosts during the
// evolution: IG is in no per-stage or post-regrid SYNC list (see
// schedule.ccl; it is synced once on checkpoint recovery, whose checkpoint
// holds only the interior), nothing between the end of CCTK_EVOL and this
// group writes IG, and AsterX_GaugeCopyIGr ran before the restriction so
// IGr's uncovered nodes and ghosts start as that same IG.
//
// The ledger's ghosts stay finite without communication because IG_rhs is
// written on every point (ghosts and outer boundary included, rhs.cxx) and
// the RK update writes the whole allocated box; ODESolvers flags every
// evolved group valid everywhere after each stage under subcycling, so the
// driver's whole-grid validity check at the top of each iteration is
// satisfied with no sync of IG. Without subcycling the register is inert and
// AsterX_GaugeZeroIG (below) keeps IG = 0 and valid everywhere instead.
//
// The local kernels do not test for an aligned child themselves: on a level
// without one the band table has no entry, so the edge update and the
// partial-cascade adopt loop nothing (before the band, they ran on every
// point and were no-ops by arithmetic cancellation).
//
// Regrids need no special handling. CarpetX regrids at the top of an Evolve
// iteration, before the batch loop, so on a two-level run every regrid that
// modifies a level happens at a time-aligned point, right after a full
// cascade zeroed IG on every level: the prolongated ledger on the new fine
// points (IG = 0) is exactly the honest record and the next cascade corrects
// as usual. With three or more levels a regrid can fall on a partial cascade
// where the middle level's IG is nonzero; the new points then carry its
// smooth interpolant instead of the exact record, a one-time mismatch of
// interpolation-error size that is accepted.

namespace {

template <int i>
void GaugeCorrectAvec_impl(CCTK_ARGUMENTS, const int order) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCorrectAvec;

  const vec<GF3D2<CCTK_REAL>, dim> gf_Avec{Avec_x, Avec_y, Avec_z};

  // Same loop centering and same operator as CalcRHSofAvec_impl uses for
  // -d_i G, so the correction is exact at any mag_correction_order; only the
  // edges visited differ: the band boxes of this component, each clipped to
  // this tile (the clip box_int applies), with the boundary box loop_int
  // would derive, so PointDesc is what loop_int would hand the body. At
  // order 2 the stencil reads only the edge's two endpoints, both interior
  // vertices. The order-4 and order-6 stencils read ghost vertices of IGr
  // and IG; those come from the ghost sync of both ledgers that
  // AsterX_GaugeRestrictIGr issues above order 2 (the ledgers' only ghost
  // exchange), and AsterX_GaugeCopyIGr ran before AsterX_GaugeRestrictIGr so
  // IGr's uncovered nodes and ghosts are that same IG.
  const gauge_band_t &band = gauge_band(grid.patch, grid.level, grid.component);
  vect<int, dim> bnd_min, bnd_max;
  grid.boundary_box<i == 0, i == 1, i == 2>(grid.nghostzones, bnd_min, bnd_max);
  using std::max, std::min;
  for (const box_t &b : band.edges[i]) {
    const vect<int, dim> lo = max(b.lo, grid.tmin);
    const vect<int, dim> hi = min(b.hi, grid.tmax);
    grid.loop_box_device<i == 0, i == 1, i == 2>(
        bnd_min, bnd_max, lo, hi,
        [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          gf_Avec(i)(p.I) -= calc_fd_forward_midpoint<i>(IGr, p, order) -
                             calc_fd_forward_midpoint<i>(IG, p, order);
        });
  }
}

} // namespace

// IGr = IG on every point of every active level.
extern "C" void AsterX_GaugeCopyIGr(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCopyIGr;

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p)
                            CCTK_ATTRIBUTE_ALWAYS_INLINE { IGr(p.I) = IG(p.I); });
}

// Compute the coarse-fine bands for this pass, then IGr := restrict(IG_fine)
// on every level with an aligned child (the one transfer the correction
// depends on). Above mag_correction_order 2 the edge stencil reads one or two
// ghost vertices of both ledgers, so a same-level ghost exchange of IG and
// IGr follows; at order 2 nothing reads a ledger ghost and the register
// issues no other communication. The band computation is local (box
// geometry only) and adds none.
extern "C" void AsterX_GaugeRestrictIGr(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  static const std::vector<int> scratch = {CCTK_GroupIndex("AsterX::IGr")};
  static const std::vector<int> ledgers = {CCTK_GroupIndex("AsterX::IG"),
                                           CCTK_GroupIndex("AsterX::IGr")};

  // reach: vertices the forward-midpoint stencil reads past an edge's
  // endpoints on each side (0 at order 2, 1 at order 4, 2 at order 6)
  ComputeGaugeBands(cctkGH, mag_correction_order / 2 - 1);
  RestrictFromAlignedChildren(cctkGH, scratch);
  if (mag_correction_order > 2)
    SyncGhostsOnly(cctkGH, ledgers);
}

// A_i -= D_i (IGr - IG) on the interior edges within a stencil of each
// refinement boundary (the band of this component; nothing on a level
// without an aligned child). This is a gauge transformation (B unchanged);
// the driver's restriction that follows this group turns it into a physical
// correction by undoing it on every fine-owned edge.
extern "C" void AsterX_GaugeCorrectAvec(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCorrectAvec;
  DECLARE_CCTK_PARAMETERS;

  GaugeCorrectAvec_impl<0>(CCTK_PASS_CTOC, mag_correction_order);
  GaugeCorrectAvec_impl<1>(CCTK_PASS_CTOC, mag_correction_order);
  GaugeCorrectAvec_impl<2>(CCTK_PASS_CTOC, mag_correction_order);
}

// IG := 0 on every point on a full cascade (all levels agree and are
// re-zeroed together so |IG| never exceeds one coarse step's worth of
// int G dt), otherwise IG := IGr on the restricted nodes (the band of this
// component), i.e. the coarse ledger adopts the fine one on covered and
// interface nodes; everywhere else IGr == IG already.
extern "C" void AsterX_GaugeAdoptIG(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeAdoptIG;

  if (full_cascade()) {
    grid.loop_all_device<0, 0, 0>(
        grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p)
                              CCTK_ATTRIBUTE_ALWAYS_INLINE { IG(p.I) = 0.0; });
    return;
  }

  const gauge_band_t &band = gauge_band(grid.patch, grid.level, grid.component);
  vect<int, dim> bnd_min, bnd_max;
  grid.boundary_box<0, 0, 0>(grid.nghostzones, bnd_min, bnd_max);
  using std::max, std::min;
  for (const box_t &b : band.nodes) {
    const vect<int, dim> lo = max(b.lo, grid.tmin);
    const vect<int, dim> hi = min(b.hi, grid.tmax);
    grid.loop_box_device<0, 0, 0>(
        bnd_min, bnd_max, lo, hi,
        [=] CCTK_DEVICE(const PointDesc &p)
            CCTK_ATTRIBUTE_ALWAYS_INLINE { IG(p.I) = IGr(p.I); });
  }
}

////////////////////////////////////////////////////////////////////////////////
// Inert runs without subcycling

// Without subcycling the register is inert (IG_rhs = 0, no reconcile) and IG
// is identically zero, but the driver still requires every evolved group to
// be valid everywhere at the top of each iteration, and the non-subcycled
// ODESolvers flags only the interior after its update. This local
// write-only pass, scheduled in ODESolvers_PostStep only when
// !use_subcycling (which also covers postinitial, postregrid and
// post_recover_variables), keeps the flags honest without a sync, so an
// inert run never communicates the ledger.
extern "C" void AsterX_GaugeZeroIG(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeZeroIG;

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p)
                            CCTK_ATTRIBUTE_ALWAYS_INLINE { IG(p.I) = 0.0; });
}

} // namespace AsterX
