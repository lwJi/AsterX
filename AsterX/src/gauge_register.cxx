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
// Parameter check and initialization

extern "C" void AsterX_GaugeRegisterParamCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  // The register is requested and would matter (Lorenz gauge, subcycling),
  // but the driver's sync mode never traverses CCTK_POSTRESTRICT, so the
  // reconcile cannot run. Warn rather than abort. Without subcycling the
  // ledger mismatch is identically zero (AsterX_RestrictAuxTermsForAvecPsiRHS
  // injects the fine G into the coarse shared nodes at every stage), and
  // restrict_during_sync = yes is the normal non-subcycled configuration, so
  // no warning is issued there.
  if (gauge_register &&
      CCTK_EQUALS(vector_potential_gauge, "generalized Lorenz") &&
      use_subcycling && carpetx_restrict_during_sync()) {
    CCTK_VINFO("The nodal gauge register (AsterX::gauge_register = yes) is "
               "inactive because CarpetX::restrict_during_sync = yes.");
    CCTK_VWARN(CCTK_WARN_ALERT,
               "AsterX::gauge_register = yes was requested with the "
               "generalized Lorenz gauge, but CarpetX::restrict_during_sync = "
               "yes. The nodal gauge register needs the CCTK_POSTRESTRICT "
               "bin, which CarpetX only traverses with restrict_during_sync = "
               "no; the register is inactive (IG_rhs = 0, no reconcile) and "
               "the coarse-fine gauge mismatch is not corrected. Set "
               "CarpetX::restrict_during_sync = no to activate it.");
  }
}

extern "C" void AsterX_GaugeRegisterInit(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeRegisterInit;

  // IG = 0 is the state at the end of a full cascade (all levels agree), so
  // the first step of a run is reconciled normally.
  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p)
                            CCTK_ATTRIBUTE_ALWAYS_INLINE { IG(p.I) = 0.0; });
}

////////////////////////////////////////////////////////////////////////////////
// The reconcile, AsterX_GaugeRegisterGroup AT postrestrict
//
// CarpetX traverses CCTK_POSTRESTRICT once per time-aligned pass, right after
// it has restricted the evolved variables fine-to-coarse and prolongated the
// restricted groups' fine ghost halos, with active_levels widened to every
// level that is at the current time. The group runs BEFORE
// ODESolvers_PostStep, so everything downstream on this pass (B from A,
// con2prim, G, E, output) sees the corrected edges.
//
// With IGr the ledger restricted from the aligned child (and IGr = IG on
// every other point), per aligned pair on the coarse level:
//   A_i -= D_i IGr - D_i IG   on every edge   a pure gauge transformation
//   restrict Avec_* from the fine             resets fine-owned + interface edges
//   prolongate the fine Avec_* halo           from the corrected coarse level
//   IG := IGr                                 the coarse ledger adopts the fine
// and on a full cascade (window reaches level 0) IG := 0 on every level.
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
  // the correction is exact at any mag_correction_order. The order-4 stencil
  // reads one ghost vertex of IGr and IG: IGr was ghost-synced after the
  // restriction, IG in the previous pass's ODESolvers_PostStep.
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

// IGr := restrict(IG_fine) on every level with an aligned child, then a
// same-level ghost exchange of IGr so the order-4 stencil can read one ghost
// vertex of it.
extern "C" void AsterX_GaugeRestrictIGr(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::IGr")};

  RestrictFromAlignedChildren(cctkGH, groups);
  SyncGhostsOnly(cctkGH, groups);
}

// A_i -= D_i (IGr - IG) on every interior edge of every active level. This is
// a gauge transformation (B unchanged); the restriction below turns it into a
// physical correction by undoing it on every fine-owned edge.
extern "C" void AsterX_GaugeCorrectAvec(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCorrectAvec;
  DECLARE_CCTK_PARAMETERS;

  GaugeCorrectAvec_impl<0>(CCTK_PASS_CTOC, mag_correction_order);
  GaugeCorrectAvec_impl<1>(CCTK_PASS_CTOC, mag_correction_order);
  GaugeCorrectAvec_impl<2>(CCTK_PASS_CTOC, mag_correction_order);
}

// Restrict Avec_* from every aligned child (fine to coarse, so a middle
// level's corrected outer edges flow into its parent), then re-prolongate
// the aligned children's Avec_* ghost halos from the corrected coarse
// levels. The driver's own ProlongateRestrictedGFs ran before this group
// with the pre-correction coarse edges, and under subcycling the SYNC in
// ODESolvers_PostStep never prolongates an evolved group, so without this
// the first stage of the next fine step would read a stale halo. Only the
// current timelevel is touched: it is the one the correction modified.
extern "C" void AsterX_GaugeRestrictAvec(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::Avec_x"),
                                          CCTK_GroupIndex("AsterX::Avec_y"),
                                          CCTK_GroupIndex("AsterX::Avec_z")};

  RestrictFromAlignedChildren(cctkGH, groups);
  ProlongateHaloFromAlignedParents(groups, 0);
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

} // namespace AsterX
