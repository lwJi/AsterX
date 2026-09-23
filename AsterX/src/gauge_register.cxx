#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include "aster_fd.hxx"
#include "gauge_register.hxx"
#include "sync.hxx"

#include <cassert>
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
// Ghost vertices read by the order-4/6 stencil: IGr's come from the ghost
// sync at the end of AsterX_GaugeRestrictIGr; IG's from the AsterX_Sync in
// the ODESolvers_PostStep of each level's last RK stage, and nothing between
// the end of CCTK_EVOL and this group writes IG.
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
  // the correction is exact at any mag_correction_order. The order-4 and
  // order-6 stencils read ghost vertices of IGr and IG. Neither is synced
  // inside this routine: IGr's ghosts come from the ghost sync at the end of
  // AsterX_GaugeRestrictIGr, IG's from the unconditional AsterX_Sync in the
  // ODESolvers_PostStep of the last RK stage, and AsterX_GaugeCopyIGr ran
  // before AsterX_GaugeRestrictIGr so IGr's uncovered nodes and ghosts are
  // that same IG.
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

////////////////////////////////////////////////////////////////////////////////
// TEMPORARY timing instrumentation (throwaway branch gauge_register_timers).
//
// CarpetX already wraps every scheduled function in a flesh timer named
// "CallFunction AsterX_GaugeRegisterGroup: AsterX::<routine>", so the split among
// the four reconcile routines is free. What it does not separate is the two
// halves of AsterX_GaugeRestrictIGr (the fine-to-coarse restrict and the
// same-level ghost sync), nor does it count how many passes and aligned pairs
// the totals are spread over. Both are added here. The counters are printed
// at CCTK_TERMINATE; the timers land in the TimerReport output like every
// other flesh timer.

namespace {

// Minimal RAII around a flesh timer. CarpetX::Timer does the same (plus nvtx
// on CUDA), but pulling CarpetX headers into this file is avoided on purpose:
// sync.cxx is the one AsterX file that includes driver internals.
class FleshTimer {
  int handle;

public:
  explicit FleshTimer(const char *const name)
      : handle(CCTK_TimerCreate(name)) {
    assert(handle >= 0);
  }
  void start() { CCTK_TimerStartI(handle); }
  void stop() { CCTK_TimerStopI(handle); }
};

class FleshInterval {
  FleshTimer &timer;

public:
  explicit FleshInterval(FleshTimer &timer) : timer(timer) { timer.start(); }
  ~FleshInterval() { timer.stop(); }
  FleshInterval(const FleshInterval &) = delete;
  FleshInterval &operator=(const FleshInterval &) = delete;
};

// Pass statistics, accumulated in AsterX_GaugeRestrictIGr (a global
// function, called exactly once per CarpetX_PreRestrict traversal).
long long stat_passes = 0;        // PreRestrict traversals with the group
long long stat_full_cascades = 0; // ... of which the window reached level 0
long long stat_aligned_pairs = 0; // aligned (coarse, fine) pairs restricted

} // namespace

// IGr := restrict(IG_fine) on every level with an aligned child, then a
// same-level ghost exchange of IGr so the order-4 stencil can read one ghost
// vertex of it.
extern "C" void AsterX_GaugeRestrictIGr(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::IGr")};

  static FleshTimer timer_restrict("AsterX_GaugeRestrictIGr::restrict");
  static FleshTimer timer_sync("AsterX_GaugeRestrictIGr::ghost_sync");

  int npairs;
  {
    FleshInterval interval(timer_restrict);
    npairs = RestrictFromAlignedChildren(cctkGH, groups);
    // Restrict_impl launches its average_down kernels asynchronously and
    // does not wait for them; the host timer would otherwise stop at launch.
    DeviceSynchronize();
  }
  {
    FleshInterval interval(timer_sync);
    // SyncGroupsByDirIGhostOnly synchronizes the device itself before it
    // returns, so no extra DeviceSynchronize is needed here.
    SyncGhostsOnly(cctkGH, groups);
  }

  ++stat_passes;
  if (full_cascade())
    ++stat_full_cascades;
  stat_aligned_pairs += npairs;
}

// TEMPORARY: print the pass statistics so the TimerReport totals can be
// normalized per pass and per aligned pair. Scheduled in CCTK_TERMINATE.
extern "C" void AsterX_GaugeRegisterTimingReport(CCTK_ARGUMENTS) {
  CCTK_VINFO("Gauge register timing: %lld PreRestrict passes (%lld full "
             "cascades, %lld partial), %lld aligned level pairs restricted "
             "(%.3f per pass)",
             stat_passes, stat_full_cascades, stat_passes - stat_full_cascades,
             stat_aligned_pairs,
             stat_passes > 0 ? double(stat_aligned_pairs) / double(stat_passes)
                             : 0.0);
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

} // namespace AsterX
