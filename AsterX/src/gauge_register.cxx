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
//   A_i -= D_i IGr - D_i IG   on every interior edge   a pure gauge transformation
//   IG := IGr                                          the coarse ledger adopts the fine
// and on a full cascade (window reaches level 0) IG := 0 on every level.
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
// The local kernels do not test for an aligned child: on a level without one
// IGr is an untouched copy of IG, so both the edge update and the adopt are
// exact no-ops (the two stencils cancel bit for bit).
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

  // Same loop and same operator as CalcRHSofAvec_impl uses for -d_i G, so
  // the correction is exact at any mag_correction_order. At order 2 the
  // stencil reads only the edge's two endpoints, both interior vertices. The
  // order-4 and order-6 stencils read ghost vertices of IGr and IG; those
  // come from the ghost sync of both ledgers that AsterX_GaugeRestrictIGr
  // issues above order 2 (the ledgers' only ghost exchange), and
  // AsterX_GaugeCopyIGr ran before AsterX_GaugeRestrictIGr so IGr's
  // uncovered nodes and ghosts are that same IG.
  grid.loop_int_device<i == 0, i == 1, i == 2>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        gf_Avec(i)(p.I) -= calc_fd_forward_midpoint<i>(IGr, p, order) -
                           calc_fd_forward_midpoint<i>(IG, p, order);
      });
}

} // namespace

// IGr = IG on every point of every active level.
extern "C" void AsterX_GaugeCopyIGr(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCopyIGr;

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p)
                            CCTK_ATTRIBUTE_ALWAYS_INLINE { IGr(p.I) = IG(p.I); });
}

// IGr := restrict(IG_fine) on every level with an aligned child (the one
// transfer the correction depends on). Above mag_correction_order 2 the edge
// stencil reads one or two ghost vertices of both ledgers, so a same-level
// ghost exchange of IG and IGr follows; at order 2 nothing reads a ledger
// ghost and the register issues no other communication.
extern "C" void AsterX_GaugeRestrictIGr(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  static const std::vector<int> scratch = {CCTK_GroupIndex("AsterX::IGr")};
  static const std::vector<int> ledgers = {CCTK_GroupIndex("AsterX::IG"),
                                           CCTK_GroupIndex("AsterX::IGr")};

  RestrictFromAlignedChildren(cctkGH, scratch);
  if (mag_correction_order > 2)
    SyncGhostsOnly(cctkGH, ledgers);
}

// A_i -= D_i (IGr - IG) on every interior edge of every active level. This is
// a gauge transformation (B unchanged); the driver's restriction that follows
// this group turns it into a physical correction by undoing it on every
// fine-owned edge.
extern "C" void AsterX_GaugeCorrectAvec(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCorrectAvec;
  DECLARE_CCTK_PARAMETERS;

  GaugeCorrectAvec_impl<0>(CCTK_PASS_CTOC, mag_correction_order);
  GaugeCorrectAvec_impl<1>(CCTK_PASS_CTOC, mag_correction_order);
  GaugeCorrectAvec_impl<2>(CCTK_PASS_CTOC, mag_correction_order);
}

// IG := 0 on a full cascade (all levels agree and are re-zeroed together so
// |IG| never exceeds one coarse step's worth of int G dt), otherwise
// IG := IGr, i.e. the coarse ledger adopts the fine one on covered and
// interface nodes and is unchanged elsewhere.
extern "C" void AsterX_GaugeAdoptIG(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeAdoptIG;

  const bool zero = full_cascade();

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        IG(p.I) = zero ? 0.0 : IGr(p.I);
      });
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
