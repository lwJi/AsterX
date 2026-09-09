#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

// CarpetX internals, reached the same way sync.cxx does: active_levels,
// RestrictNoPoison, SyncGroupsByDirIGhostOnly, FillPatch_ProlongateOnly,
// ghext, task_manager.
#include "../../../CarpetX/CarpetX/src/fillpatch.hxx"
#include "../../../CarpetX/CarpetX/src/schedule.hxx"
#include "../../../CarpetX/CarpetX/src/task_manager.hxx"

#include "aster_fd.hxx"
#include "gauge_register.hxx"

#include <cassert>
#include <vector>

namespace AsterX {
using namespace Loop;
using namespace Arith;
using namespace AsterUtils;

////////////////////////////////////////////////////////////////////////////////
// Phase 1: parameter check and initialization

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
// Phase 2: the reconcile, AsterX_GaugeRegisterGroup AT postrestrict
//
// CarpetX traverses CCTK_POSTRESTRICT once per time-aligned pass, right after
// it has restricted the evolved variables fine-to-coarse and prolongated the
// restricted groups' fine ghost halos, with active_levels widened to every
// level that is at the current time. The group runs BEFORE
// ODESolvers_PostStep, so everything downstream on this pass (B from A,
// con2prim, G, E, output) sees the corrected edges.
//
// Per aligned pair (fine -> coarse), on the coarse level:
//   IGr := restrict(IG_fine)          nodal injection; IGr = IG elsewhere
//   lam := IGr - IG                   != 0 only on covered + interface nodes
//   A_i -= D_i lam   on every edge    a pure gauge transformation
//   restrict Avec_* from the fine     resets fine-owned + interface edges
//   prolongate the fine Avec_* halo   from the corrected coarse level
//   IG  := IG + lam  (= IGr)          the coarse ledger adopts the fine one
// and on a full cascade (window reaches level 0) IG := 0 on every level.
//
// The local kernels do not test for an aligned child: on a level without one
// IGr stays an untouched copy of IG, lam = 0, and both the edge update and
// the adopt are exact no-ops.

namespace {

// A level has a time-aligned child iff that child is inside the active
// window [min_level, max_level). Same guard as AsterX_RestrictFluxes.
inline bool has_aligned_child(const int level) {
  assert(CarpetX::active_levels);
  return level + 1 < CarpetX::active_levels->max_level;
}

// The aligned window reaches level 0: every level has just been reconciled
// and the ledgers can be re-zeroed together.
inline bool full_cascade() {
  assert(CarpetX::active_levels);
  return CarpetX::active_levels->min_level == 0;
}

// Values of the checkpointed grid scalar gauge_register_state.
constexpr CCTK_INT GAUGE_REGISTER_CORRECT = 0;    // correct on the next full cascade
constexpr CCTK_INT GAUGE_REGISTER_ADOPT_ONLY = 1; // set by postregrid

template <int i>
void GaugeCorrectAvec_impl(CCTK_ARGUMENTS, const int order) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCorrectAvec;

  const vec<GF3D2<CCTK_REAL>, dim> gf_Avec{Avec_x, Avec_y, Avec_z};

  // Same loop and same operator as CalcRHSofAvec_impl uses for -d_i G, with
  // lam (held in IGr) in place of G, so the correction is exact at any
  // mag_correction_order.
  grid.loop_int_device<i == 0, i == 1, i == 2>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        gf_Avec(i)(p.I) -= calc_fd_forward_midpoint<i>(IGr, p, order);
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
// vertex of lam.
extern "C" void AsterX_GaugeRestrictIGr(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::IGr")};

  CarpetX::active_levels->loop_fine_to_coarse([&](const auto &leveldata) {
    if (has_aligned_child(leveldata.level))
      CarpetX::RestrictNoPoison(cctkGH, leveldata.level, groups);
  });

  CarpetX::SyncGroupsByDirIGhostOnly(cctkGH, groups.size(), groups.data(),
                                     nullptr);
}

// IGr := IGr - IG  (= lam), materialised in place so the edge kernel can
// hand it to calc_fd_forward_midpoint verbatim.
extern "C" void AsterX_GaugeFormLambda(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeFormLambda;

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones, [=] CCTK_DEVICE(const PointDesc &p)
                            CCTK_ATTRIBUTE_ALWAYS_INLINE { IGr(p.I) -= IG(p.I); });
}

// A_i -= D_i lam on every interior edge of every active level. This is a
// gauge transformation (B unchanged); the second restriction below turns it
// into a physical correction by undoing it on every fine-owned edge.
extern "C" void AsterX_GaugeCorrectAvec(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeCorrectAvec;
  DECLARE_CCTK_PARAMETERS;

  // First full cascade after a regrid: the fine ledger on new points is a
  // prolongation of the coarse one, not a record of what the fine level
  // applied, so the edges are left alone (adopt and re-zero still happen).
  // Partial cascades are never skipped. AsterX_GaugeFinish clears the state
  // and logs the skipped cascade once.
  if (full_cascade() && *gauge_register_state != GAUGE_REGISTER_CORRECT)
    return;

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
// the first stage of the next fine step would read a stale halo.
extern "C" void AsterX_GaugeRestrictAvec(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::Avec_x"),
                                          CCTK_GroupIndex("AsterX::Avec_y"),
                                          CCTK_GroupIndex("AsterX::Avec_z")};

  CarpetX::active_levels->loop_fine_to_coarse([&](const auto &leveldata) {
    if (has_aligned_child(leveldata.level))
      CarpetX::RestrictNoPoison(cctkGH, leveldata.level, groups);
  });

  // Halo-only prolongation, per aligned pair, following ApplyOuterBC's
  // task_manager pattern and SyncGroupsByDirIProlongateOnly_impl's choice of
  // interpolator and bcrecs (both from the fine GroupData). Only the current
  // timelevel is touched: it is the one the correction and the restriction
  // above modified. The coarse patch is gathered from the coarse valid
  // region only, so the coarse level's same-level ghosts (still stale after
  // the correction until the SYNC in ODESolvers_PostStep) do not enter.
  CarpetX::task_manager tasks1;
  CarpetX::task_manager tasks2;
  CarpetX::task_manager tasks3;

  for (const int gi : groups) {
    CarpetX::active_levels->loop_coarse_to_fine([&](auto &restrict leveldata) {
      if (!has_aligned_child(leveldata.level))
        return;
      const int level = leveldata.level;
      auto &restrict patchdata = CarpetX::ghext->patchdata.at(leveldata.patch);
      auto &restrict fineleveldata = patchdata.leveldata.at(level + 1);
      auto &restrict coarsegroupdata = *leveldata.groupdata.at(gi);
      auto &restrict finegroupdata = *fineleveldata.groupdata.at(gi);
      assert(!coarsegroupdata.mfab.empty());
      assert(!finegroupdata.mfab.empty());
      assert(coarsegroupdata.numvars == finegroupdata.numvars);
      const int tl = 0;
      tasks1.submit_serially([&tasks2, &tasks3, &patchdata, &finegroupdata,
                              &coarsegroupdata, level, tl]() {
        CarpetX::FillPatch_ProlongateOnly(
            tasks2, tasks3, finegroupdata, coarsegroupdata,
            *finegroupdata.mfab.at(tl), *coarsegroupdata.mfab.at(tl),
            patchdata.amrcore->Geom(level + 1), patchdata.amrcore->Geom(level),
            finegroupdata.interpolator, finegroupdata.bcrecs);
      });
    });
  } // for gi

  tasks1.run_tasks_serially();
  CarpetX::synchronize();
  tasks2.run_tasks_serially();
  CarpetX::synchronize();
  tasks3.run_tasks_serially();
  CarpetX::synchronize();

  assert(CarpetX::ghext->num_patches() == 1);
}

// IG := 0 on a full cascade (all levels agree and are re-zeroed together so
// |IG| never exceeds one coarse step's worth of int G dt), otherwise
// IG := IG + lam, i.e. the coarse ledger adopts the fine one on covered and
// interface nodes and is unchanged elsewhere.
extern "C" void AsterX_GaugeAdoptIG(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeAdoptIG;

  const bool zero = full_cascade();

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        IG(p.I) = zero ? 0.0 : IG(p.I) + IGr(p.I);
      });
}

////////////////////////////////////////////////////////////////////////////////
// Phase 4: regrid adopt-only path. The state lives in a checkpointed grid
// scalar, so a checkpoint taken between a regrid and the next full cascade
// recovers into the same adopt-only path.

// AT initial (global): the run starts in the correct state. CCTK_INITIAL is
// traversed once per level during initialisation, after that level's
// postregrid, so the last traversal leaves the state at 0.
extern "C" void AsterX_GaugeRegisterInitState(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeRegisterInitState;
  *gauge_register_state = GAUGE_REGISTER_CORRECT;
}

// AT postregrid (global): a level changed during evolution, so the next full
// cascade must adopt the ledger without correcting edges. Postregrid
// traversals while the hierarchy is being built (iteration 0) are ignored:
// the ledger is identically zero there and the postrestrict traversal at the
// end of initialisation resets the state anyway.
extern "C" void AsterX_GaugeRegisterMarkRegrid(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeRegisterMarkRegrid;

  if (cctk_iteration <= 0)
    return;
  *gauge_register_state = GAUGE_REGISTER_ADOPT_ONLY;
  CCTK_VINFO("AsterX gauge register: regrid at iteration %d; the next full "
             "cascade will adopt the ledger without correcting edges",
             cctk_iteration);
}

// Last member of AsterX_GaugeRegisterGroup (global): on a full cascade the
// ledgers have just been reconciled and re-zeroed, so the state returns to
// correct. The skipped (adopt-only) cascade is logged exactly once so runs
// are auditable.
extern "C" void AsterX_GaugeFinish(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_GaugeFinish;

  if (!full_cascade())
    return;
  if (*gauge_register_state != GAUGE_REGISTER_CORRECT) {
    CCTK_VINFO("AsterX gauge register: adopt-only full cascade at iteration "
               "%d (first after a regrid); coarse edge correction skipped, "
               "ledger adopted and re-zeroed",
               cctk_iteration);
    *gauge_register_state = GAUGE_REGISTER_CORRECT;
  }
}

} // namespace AsterX
