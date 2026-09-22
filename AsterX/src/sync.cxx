#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include "../../../CarpetX/CarpetX/src/fillpatch.hxx"
#include "../../../CarpetX/CarpetX/src/schedule.hxx"
#include "../../../CarpetX/CarpetX/src/task_manager.hxx"

#include "sync.hxx"

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_IntVect.H>

#include <cassert>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace AsterX {
using namespace CarpetX;
using Arith::vect;
using Loop::dim;

////////////////////////////////////////////////////////////////////////////////
// Level-window helpers (declared in sync.hxx)

bool has_aligned_child(const int level) {
  assert(active_levels);
  return level + 1 < active_levels->max_level;
}

bool full_cascade() {
  assert(active_levels);
  return active_levels->min_level == 0;
}

void RestrictFromAlignedChildren(const cGH *const cctkGH,
                                 const std::vector<int> &groups) {
  assert(active_levels);
  active_levels->loop_fine_to_coarse([&](const auto &leveldata) {
    // Only restrict from a child level that is inside the active window
    // [min_level, max_level), i.e. one that is time-aligned with this level.
    // Under subcycling a coarse-only batch has no active child, so nothing
    // is restricted; without subcycling every level is active, so this is
    // the same as restricting from every level but the finest.
    if (has_aligned_child(leveldata.level))
      RestrictNoPoison(cctkGH, leveldata.level, groups);
  });
}

void SyncGhostsOnly(const cGH *const cctkGH, const std::vector<int> &groups) {
  SyncGroupsByDirIGhostOnly(cctkGH, groups.size(), groups.data(), nullptr);
}

////////////////////////////////////////////////////////////////////////////////
// Coarse-fine bands for the nodal gauge register (declared in sync.hxx)
//
// The band of a coarse level is derived from the geometry of its aligned
// child: the nodal restriction of the scratch ledger (average_down_nodal)
// writes exactly the coarse nodes of the child's box array converted to nodes
// and coarsened by 2, and the register's edge shift D_i IGr - D_i IG is
// nonzero only on the edges whose stencil reads one of those nodes. Both sets
// are built once per pass as plain integer boxes in the coarse level's global
// AMReX index space, made disjoint (abutting fine boxes share face nodes, and
// the grown edge ranges of neighbouring boxes overlap; a point visited twice
// would be corrected twice), then clipped to each component's interior and
// shifted to that component's local index space. With AsterX::debug_mode the
// stored boxes are printed on every pass.

namespace {

// Global integer boxes, one set per (patch, level).
struct footprint_t {
  amrex::BoxList nodes;
  std::array<amrex::BoxList, dim> edges;
};

using band_key_t = std::tuple<int, int, int>; // patch, level, component

std::map<band_key_t, gauge_band_t> gauge_band_table;
const gauge_band_t empty_gauge_band{};

// Treat a box as a set of integer indices, whatever centering it was built
// with: the AMReX box algebra used below (removeOverlap) supports cell-typed
// boxes only.
amrex::Box as_index_box(const amrex::Box &b) {
  return amrex::Box(b.smallEnd(), b.bigEnd(), amrex::IndexType::TheCellType());
}

// Nodal footprint of the aligned child of `level` on `level`, and the edge
// bands grown from it.
footprint_t make_footprint(const int patch, const int level, const int reach) {
  const auto &patchdata = ghext->patchdata.at(patch);
  // The child's cell boxes; the fine IGr MultiFab lives on their nodal
  // conversion, and average_down_nodal writes their coarsening by 2. For
  // ratio 2 that is the same as [lo, hi+1] of the coarsened cell box.
  const amrex::BoxArray &fine_boxes = patchdata.amrcore->boxArray(level + 1);
  const amrex::IntVect ratio(2, 2, 2);

  amrex::BoxList nodes(amrex::IndexType::TheCellType());
  for (int i = 0; i < fine_boxes.size(); ++i) {
    amrex::Box b = fine_boxes[i];
    b.surroundingNodes();
    b.coarsen(ratio);
    nodes.push_back(as_index_box(b));
  }
  footprint_t fp;
  fp.nodes = amrex::removeOverlap(nodes);

  for (int d = 0; d < dim; ++d) {
    // An edge with cell index e along d spans the nodes e and e+1 and its
    // stencil reads the nodes [e - reach, e + 1 + reach]; it touches the
    // footprint node range [nlo, nhi] iff e is in [nlo - 1 - reach,
    // nhi + reach]. Transverse to d an edge sits at node indices.
    amrex::BoxList edges(amrex::IndexType::TheCellType());
    for (const amrex::Box &n : fp.nodes) {
      amrex::Box e(n);
      e.growLo(d, reach + 1);
      e.growHi(d, reach);
      edges.push_back(e);
    }
    fp.edges[d] = amrex::removeOverlap(edges);
  }
  return fp;
}

// Clip a set of global boxes to the interior [int_lo, int_hi) of one
// component and shift them to its local indices (global = local + shift).
std::vector<box_t> clip_to_component(const amrex::BoxList &boxes,
                                     const vect<int, dim> &shift,
                                     const vect<int, dim> &int_lo,
                                     const vect<int, dim> &int_hi) {
  using std::max, std::min;
  std::vector<box_t> result;
  for (const amrex::Box &b : boxes) {
    vect<int, dim> lo, hi;
    for (int d = 0; d < dim; ++d) {
      lo[d] = b.smallEnd(d) - shift[d];
      hi[d] = b.bigEnd(d) + 1 - shift[d];
    }
    lo = max(lo, int_lo);
    hi = min(hi, int_hi);
    if (all(lo < hi))
      result.push_back(box_t{lo, hi});
  }
  return result;
}

// Every stored box lies inside [int_lo, int_hi) and no two boxes of the set
// intersect, so a kernel looping the set writes each point at most once and
// never touches a ghost or outer-boundary point. Cheap (a handful of boxes
// per component per pass), so it is always on.
void check_band_boxes(const char *const what, const std::vector<box_t> &boxes,
                      const vect<int, dim> &int_lo,
                      const vect<int, dim> &int_hi, const int patch,
                      const int level, const int component) {
  for (std::size_t a = 0; a < boxes.size(); ++a) {
    const box_t &ba = boxes[a];
    if (!(all(ba.lo >= int_lo) && all(ba.hi <= int_hi) && all(ba.lo < ba.hi)))
      CCTK_VERROR("Gauge band %s box [%d,%d,%d)-[%d,%d,%d) of patch %d level "
                  "%d component %d is not inside the interior "
                  "[%d,%d,%d)-[%d,%d,%d)",
                  what, ba.lo[0], ba.lo[1], ba.lo[2], ba.hi[0], ba.hi[1],
                  ba.hi[2], patch, level, component, int_lo[0], int_lo[1],
                  int_lo[2], int_hi[0], int_hi[1], int_hi[2]);
    for (std::size_t c = a + 1; c < boxes.size(); ++c) {
      const box_t &bc = boxes[c];
      if (all(ba.lo < bc.hi) && all(bc.lo < ba.hi))
        CCTK_VERROR("Gauge band %s boxes of patch %d level %d component %d "
                    "overlap: [%d,%d,%d)-[%d,%d,%d) and [%d,%d,%d)-[%d,%d,%d)",
                    what, patch, level, component, ba.lo[0], ba.lo[1], ba.lo[2],
                    ba.hi[0], ba.hi[1], ba.hi[2], bc.lo[0], bc.lo[1], bc.lo[2],
                    bc.hi[0], bc.hi[1], bc.hi[2]);
    }
  }
}

} // namespace

void ComputeGaugeBands(const cGH *const cctkGH, const int reach) {
  DECLARE_CCTK_PARAMETERS;
  assert(active_levels);
  assert(reach >= 0);
  gauge_band_table.clear();

  std::map<std::pair<int, int>, footprint_t> footprints;

  // Walk the components of the active window with the local cGH each one
  // is called with, so the index conversion reads the same lbnd, lsh and
  // nghostzones a kernel's GridDesc sees. CarpetX enumerates tiles here and
  // gives every tile of a component the same cctk_component (the AMReX fab
  // index), so the band is stored once per component and clipped to the
  // component's interior, not to a tile.
  active_levels->loop_serially([&](const int patch, const int level,
                                   const int index, const int /*tile*/,
                                   const cGH *const local_cctkGH) {
    if (!has_aligned_child(level))
      return;

    const Loop::GridDescBase grid(local_cctkGH);
    assert(grid.patch == patch);
    assert(grid.level == level);
    assert(grid.component == index);
    const band_key_t key{patch, level, grid.component};
    if (gauge_band_table.count(key))
      return; // another tile of this component

    auto fp = footprints.find({patch, level});
    if (fp == footprints.end())
      fp = footprints.emplace(std::make_pair(patch, level),
                              make_footprint(patch, level, reach))
               .first;

    // Global AMReX index of local index 0 (schedule.cxx: lbnd is the
    // valid box's low corner, and the allocated box starts nghostzones
    // below it).
    const vect<int, dim> shift = grid.lbnd - grid.nghostzones;

    gauge_band_t band;
    {
      // Vertices: interior [nghostzones, lsh - nghostzones)
      const vect<int, dim> int_lo = grid.nghostzones;
      const vect<int, dim> int_hi = grid.lsh - grid.nghostzones;
      band.nodes = clip_to_component(fp->second.nodes, shift, int_lo, int_hi);
      check_band_boxes("node", band.nodes, int_lo, int_hi, patch, level,
                       grid.component);
    }
    for (int d = 0; d < dim; ++d) {
      // Direction-d edges: cell-centred along d, so one fewer point there
      // (the offset box_int applies for <CI,CJ,CK> = e_d).
      vect<int, dim> offset{0, 0, 0};
      offset[d] = 1;
      const vect<int, dim> int_lo = grid.nghostzones;
      const vect<int, dim> int_hi = grid.lsh - offset - grid.nghostzones;
      band.edges[d] =
          clip_to_component(fp->second.edges[d], shift, int_lo, int_hi);
      check_band_boxes("edge", band.edges[d], int_lo, int_hi, patch, level,
                       grid.component);
    }

    if (debug_mode) {
      const auto show = [&](const char *const what,
                            const std::vector<box_t> &boxes) {
        for (const box_t &b : boxes)
          CCTK_VINFO("Gauge band iteration %d patch %d level %d component %d "
                     "%s: local [%d,%d,%d)-[%d,%d,%d) global [%d,%d,%d]-[%d,%d,%d]",
                     cctkGH->cctk_iteration, patch, level, grid.component, what,
                     b.lo[0], b.lo[1], b.lo[2], b.hi[0], b.hi[1], b.hi[2],
                     b.lo[0] + shift[0], b.lo[1] + shift[1], b.lo[2] + shift[2],
                     b.hi[0] - 1 + shift[0], b.hi[1] - 1 + shift[1],
                     b.hi[2] - 1 + shift[2]);
      };
      show("nodes", band.nodes);
      show("edges_x", band.edges[0]);
      show("edges_y", band.edges[1]);
      show("edges_z", band.edges[2]);
    }

    gauge_band_table.emplace(key, std::move(band));
  });
}

const gauge_band_t &gauge_band(const int patch, const int level,
                               const int component) {
  const auto it = gauge_band_table.find(band_key_t{patch, level, component});
  return it == gauge_band_table.end() ? empty_gauge_band : it->second;
}

////////////////////////////////////////////////////////////////////////////////

extern "C" void AsterX_Sync(CCTK_ARGUMENTS) {
  // do nothing
}

void ApplyOuterBC(CCTK_ARGUMENTS, const std::vector<int> &groups) {
  task_manager tasks1;
  task_manager tasks2;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);

      const int ntls = groupdata.mfab.size();
      const int sync_tl = ntls > 1 ? ntls - 1 : ntls;

      // Copy from adjacent boxes on same level and apply boundary conditions
      // Even though this introduces additional communication, it makes the code
      // compatible with symmetries.
      for (int tl = 0; tl < sync_tl; ++tl) {
        tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
          FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                         ghext->patchdata.at(leveldata.patch)
                             .amrcore->Geom(leveldata.level));
        });
      } // for tl
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();

  assert(ghext->num_patches() == 1);
}

extern "C" void AsterX_ApplyOuterBCOnPrim(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {
      CCTK_GroupIndex("HydroBaseX::rho"),
      CCTK_GroupIndex("HydroBaseX::vel"),
      CCTK_GroupIndex("HydroBaseX::eps"),
      CCTK_GroupIndex("HydroBaseX::press"),
      CCTK_GroupIndex("HydroBaseX::Bvec"),
      CCTK_GroupIndex("HydroBaseX::temperature"),
      CCTK_GroupIndex("HydroBaseX::entropy"),
      CCTK_GroupIndex("HydroBaseX::Ye"),
      CCTK_GroupIndex("AsterX::zvec"),
      CCTK_GroupIndex("AsterX::svec"),
      CCTK_GroupIndex("AsterX::dBx_stag"),
      CCTK_GroupIndex("AsterX::dBy_stag"),
      CCTK_GroupIndex("AsterX::dBz_stag")};

  ApplyOuterBC(CCTK_PASS_CTOC, groups);
}

extern "C" void AsterX_RestrictFluxes(CCTK_ARGUMENTS) {
  static const std::vector<int> restrict_groups = {
      CCTK_GroupIndex("AsterX::flux_x"), CCTK_GroupIndex("AsterX::flux_y"),
      CCTK_GroupIndex("AsterX::flux_z")};

  RestrictFromAlignedChildren(cctkGH, restrict_groups);
}

extern "C" void AsterX_RestrictAuxTermsForAvecPsiRHS(CCTK_ARGUMENTS) {
  static const std::vector<int> restrict_groups = {
      CCTK_GroupIndex("AsterX::G"), CCTK_GroupIndex("AsterX::Ex"),
      CCTK_GroupIndex("AsterX::Ey"), CCTK_GroupIndex("AsterX::Ez")};

  RestrictFromAlignedChildren(cctkGH, restrict_groups);
}

extern "C" void AsterX_ProlongatedBstag(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::dBx_stag"),
                                          CCTK_GroupIndex("AsterX::dBy_stag"),
                                          CCTK_GroupIndex("AsterX::dBz_stag")};

  SyncGroupsByDirIProlongateOnly(cctkGH, groups.size(), groups.data(), nullptr);
}

extern "C" void AsterX_CommdBstag(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::dBx_stag"),
                                          CCTK_GroupIndex("AsterX::dBy_stag"),
                                          CCTK_GroupIndex("AsterX::dBz_stag")};

  SyncGroupsByDirIGhostOnly(cctkGH, groups.size(), groups.data(), nullptr);
}

extern "C" void AsterX_CommdB(CCTK_ARGUMENTS) {
  static const std::vector<int> groups = {CCTK_GroupIndex("AsterX::dB")};

  SyncGroupsByDirIGhostOnly(cctkGH, groups.size(), groups.data(), nullptr);
}

} // namespace AsterX
