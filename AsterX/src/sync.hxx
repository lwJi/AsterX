#ifndef ASTERX_SYNC_HXX
#define ASTERX_SYNC_HXX

#include <cctk.h>
#include <loop_device.hxx>

#include <array>
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

////////////////////////////////////////////////////////////////////////////////
// Coarse-fine bands for the nodal gauge register

// A half-open box [lo, hi) in the local index space of one grid function
// component (the indices PointDesc::I carries and loop_box_device takes).
struct box_t {
  Arith::vect<int, Loop::dim> lo, hi;
};

// The band of one component of a coarse level that has an aligned child:
// the coarse points at which the register's edge shift can be nonzero.
//   nodes     vertex indices of the coarsened fine footprint, i.e. exactly
//             the coarse nodes the nodal restriction of the scratch ledger
//             writes; disjoint boxes
//   edges[i]  indices of the direction-i edges whose forward-midpoint stencil
//             reads at least one footprint node (the footprint nodes grown by
//             reach+1 below and reach above along i); disjoint boxes
// Every box is clipped to the component's interior, so looping it never
// touches a ghost or outer-boundary point. The boxes are not clipped to a
// tile: a kernel clips them to its own [tmin, tmax) (the same clip box_int
// applies), because CarpetX hands every tile of a component the same
// cctk_component.
struct gauge_band_t {
  std::vector<box_t> nodes;
  std::array<std::vector<box_t>, Loop::dim> edges;
};

// Global: rebuild the band table for every level of the active window that
// has an aligned child, from the child's current AMReX box array. A pure
// local computation (a few box intersections per component), no
// communication. `reach` is the number of vertices the edge stencil reads
// beyond the edge's endpoints on each side: mag_correction_order / 2 - 1.
void ComputeGaugeBands(const cGH *cctkGH, int reach);

// The band stored for (patch, level, component); an empty band on a level
// without an aligned child or before ComputeGaugeBands ran. Read-only after
// ComputeGaugeBands, so local kernels may call it concurrently.
const gauge_band_t &gauge_band(int patch, int level, int component);

} // namespace AsterX

#endif // #ifndef ASTERX_SYNC_HXX
