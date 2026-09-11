#ifndef ASTERX_SYNC_HXX
#define ASTERX_SYNC_HXX

#include <cctk.h>

#include <vector>

// Thin wrappers around the CarpetX driver's level window and its
// restriction / prolongation entry points. They are defined in sync.cxx, the
// one AsterX file that includes CarpetX internals, so other files (in
// particular gauge_register.cxx) can express "every time-aligned level pair"
// without reaching into the driver themselves.
//
// Under subcycling CarpetX calls global functions with active_levels set to
// the window [min_level, max_level) of levels that are at the current time.
// Without subcycling every level is in the window.

namespace AsterX {

// `level` has a child that is in the active window, i.e. is time-aligned
// with it.
bool has_aligned_child(int level);

// The active window reaches level 0: every level is at the same time.
bool full_cascade();

// Restrict `groups` fine-to-coarse onto every level that has an aligned
// child, finest pair first, without poisoning the restricted region.
void RestrictFromAlignedChildren(const cGH *cctkGH,
                                 const std::vector<int> &groups);

// Same-level ghost exchange (plus outer boundary conditions) of `groups` on
// every active level. No prolongation.
void SyncGhostsOnly(const cGH *cctkGH, const std::vector<int> &groups);

// Refill the ghost halo of timelevel `tl` of `groups` on every level whose
// parent is in the active window, by prolongation from that parent. Under
// subcycling only time-aligned pairs are touched (CarpetX's
// SyncGroupsByDirIProlongateOnlyAligned); without subcycling every pair is.
void ProlongateHaloFromAlignedParents(const cGH *cctkGH,
                                      const std::vector<int> &groups, int tl);

} // namespace AsterX

#endif // #ifndef ASTERX_SYNC_HXX
