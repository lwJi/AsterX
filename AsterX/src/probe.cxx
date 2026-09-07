// Scratch probe for task debug-subcycling-refinement-boundary (Phase 2).
//
// After every ODESolvers_PostStep, print the max norm of the face-centred
// densitised magnetic field dB*_stag on refinement-boundary strips of every
// active level.  Read-only: it never writes a grid function and never
// communicates except for the final max/sum reductions.  Enabled by
// AsterX::boundary_probe; this file, the parameter and the schedule entry are
// removed before the fix is submitted.
//
// Strips (faces of the three dB*_stag MultiFabs, classified from the level's
// own BoxArray, the coarsened BoxArray of the next finer level, and the domain
// box; nothing depends on the coarse-fine mask or on subcycling):
//
//   levels with a parent
//     ghost   faces outside the level's valid region but inside the domain
//             (the coarse-fine ghost halo)
//     inner3  valid faces within 3 of the level's edge (p +- 3 e_d leaves the
//             valid region into the domain for some axis d)
//   levels with a child
//     under3  faces covered by the child within 3 of the child's edge
//     outer3  faces not covered by the child with p +- 3 e_d covered for some d
//   every level
//     phys    faces inside the domain within 3 of the physical boundary
//             (adjacent finding: the physical-boundary mode seen in Phase 1)
//
// Faces outside the physical domain are skipped.  Same-level halo overlaps
// are counted once per face (amrex::OwnerMask over the grown boxes) so the
// first-call face counts are a pure function of the level geometry; the max
// norm is unaffected by duplicates.
//
// stdout, one line per (level, call):
//   probe level=L call=n it=I t=T lit=N/D ghost=.. inner3=.. under3=.. outer3=.. phys=.. nonfinite=..
// and whenever a level's strip face counts change (first call, and again
// once a child level has been created or after a regrid):
//   probe-geom level=L call=n nboxes=B ghost=.. inner3=.. under3=.. outer3=.. phys=..
// `lit` is the level's own (rational) iteration, so the sub-step is
// unambiguous even where cctk_time coincides between stages.

#include <cctk.h>
#include <cctk_Arguments.h>

#include "../../../CarpetX/CarpetX/src/driver.hxx"
#include "../../../CarpetX/CarpetX/src/schedule.hxx"

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace AsterX {
using namespace CarpetX;

namespace {

enum strip_t { S_GHOST = 0, S_INNER3, S_UNDER3, S_OUTER3, S_PHYS, S_NSTRIPS };
constexpr const char *strip_names[S_NSTRIPS] = {"ghost", "inner3", "under3",
                                                "outer3", "phys"};
constexpr int strip_width = 3;

// A set of boxes in one index type, with cheap point containment.
struct boxset_t {
  std::vector<amrex::Box> boxes;

  boxset_t() = default;
  boxset_t(const amrex::BoxArray &ba, const amrex::IndexType typ) {
    const amrex::BoxArray cba = amrex::convert(ba, typ);
    boxes.reserve(cba.size());
    for (amrex::Long b = 0; b < cba.size(); ++b)
      boxes.push_back(cba[b]);
  }
  boxset_t(const amrex::Box &bx, const amrex::IndexType typ) {
    boxes.push_back(amrex::convert(bx, typ));
  }

  bool contains(const amrex::IntVect &p) const {
    for (const auto &b : boxes)
      if (b.contains(p))
        return true;
    return false;
  }
};

// True if, for some axis d and sign s, the point q = p + s*strip_width*e_d
// satisfies set.contains(q) == want_in (and lies inside `within`, if given).
// Only single-axis offsets are tested, so corner regions do not count.
bool offset_test(const amrex::IntVect &p, const boxset_t &set,
                 const bool want_in, const boxset_t *within) {
  for (int d = 0; d < dim; ++d) {
    for (int s = -1; s <= 1; s += 2) {
      amrex::IntVect q = p;
      q[d] += s * strip_width;
      if (set.contains(q) != want_in)
        continue;
      if (within && !within->contains(q))
        continue;
      return true;
    }
  }
  return false;
}

} // namespace

extern "C" void AsterX_BoundaryProbe(CCTK_ARGUMENTS) {
  static const std::array<int, dim> gis = {
      CCTK_GroupIndex("AsterX::dBx_stag"), CCTK_GroupIndex("AsterX::dBy_stag"),
      CCTK_GroupIndex("AsterX::dBz_stag")};
  for (const int gi : gis)
    assert(gi >= 0);

  static std::map<int, int> ncalls;                                // per level
  static std::map<int, std::array<amrex::Long, S_NSTRIPS> > lastcnt; // per level

  active_levels->loop_serially([&](const auto &leveldata) {
    const int level = leveldata.level;
    const int patch = leveldata.patch;
    const auto &patchdata = ghext->patchdata.at(patch);
    const amrex::Geometry &geom = patchdata.amrcore->Geom(level);
    const amrex::Box dom_cc = geom.Domain();
    const amrex::BoxArray &own_cc = leveldata.fab->boxArray();

    const bool has_parent = level > 0;
    const bool has_child = level + 1 < int(patchdata.leveldata.size());
    amrex::BoxArray child_cc;
    if (has_child) {
      child_cc = patchdata.leveldata.at(level + 1).fab->boxArray();
      child_cc.coarsen(2);
    }

    std::array<CCTK_REAL, S_NSTRIPS> vmax;
    std::array<amrex::Long, S_NSTRIPS> cnt;
    vmax.fill(0);
    cnt.fill(0);
    amrex::Long nonfinite = 0;

    for (int dir = 0; dir < dim; ++dir) {
      const auto &groupdata = *leveldata.groupdata.at(gis[dir]);
      assert(!groupdata.mfab.empty());
      const amrex::MultiFab &mf = *groupdata.mfab.at(0);
      const amrex::IndexType typ = mf.ixType();

      const boxset_t own(own_cc, typ);
      const boxset_t dom(dom_cc, typ);
      const boxset_t child = has_child ? boxset_t(child_cc, typ) : boxset_t();

      // Owner mask over the grown boxes: each face of the union of all fab
      // boxes (interior + halo) is owned by exactly one fab.
      const auto omask =
          amrex::OwnerMask(mf, geom.periodicity(), mf.nGrowVect());

      for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box &bx = mf[mfi].box(); // fab box including halo
        const auto arr = mf.const_array(mfi);
        const auto oarr = omask->const_array(mfi);
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);

        for (int k = lo.z; k <= hi.z; ++k) {
          for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
              const amrex::IntVect p(i, j, k);
              if (!dom.contains(p))
                continue; // physical-boundary halo

              std::array<bool, S_NSTRIPS> in;
              in.fill(false);
              if (has_parent) {
                if (!own.contains(p))
                  in[S_GHOST] = true;
                else if (offset_test(p, own, false, &dom))
                  in[S_INNER3] = true;
              }
              if (has_child) {
                if (child.contains(p)) {
                  if (offset_test(p, child, false, &dom))
                    in[S_UNDER3] = true;
                } else if (offset_test(p, child, true, nullptr)) {
                  in[S_OUTER3] = true;
                }
              }
              if (offset_test(p, dom, false, nullptr))
                in[S_PHYS] = true;

              bool any = false;
              for (int s = 0; s < S_NSTRIPS; ++s)
                any |= in[s];
              if (!any)
                continue;

              const CCTK_REAL v = std::fabs(arr(i, j, k, 0));
              const bool finite = std::isfinite(v);
              const bool owner = oarr(i, j, k) != 0;
              if (!finite && owner)
                ++nonfinite;
              for (int s = 0; s < S_NSTRIPS; ++s) {
                if (!in[s])
                  continue;
                if (owner)
                  ++cnt[s];
                if (finite && v > vmax[s])
                  vmax[s] = v;
              }
            }
          }
        }
      } // for mfi
    } // for dir

    amrex::ParallelDescriptor::ReduceRealMax(vmax.data(), S_NSTRIPS);
    amrex::ParallelDescriptor::ReduceLongSum(cnt.data(), S_NSTRIPS);
    amrex::ParallelDescriptor::ReduceLongSum(nonfinite);

    const int call = ++ncalls[level];
    if (amrex::ParallelDescriptor::IOProcessor()) {
      const auto it = lastcnt.find(level);
      if (it == lastcnt.end() || it->second != cnt) {
        lastcnt[level] = cnt;
        std::printf("probe-geom level=%d call=%d nboxes=%lld", level, call,
                    (long long)own_cc.size());
        for (int s = 0; s < S_NSTRIPS; ++s)
          std::printf(" %s=%lld", strip_names[s], (long long)cnt[s]);
        std::printf("\n");
      }
      std::printf("probe level=%d call=%d it=%d t=%.17g lit=%lld/%lld", level,
                  call, cctkGH->cctk_iteration, double(cctkGH->cctk_time),
                  (long long)leveldata.iteration.num,
                  (long long)leveldata.iteration.den);
      for (int s = 0; s < S_NSTRIPS; ++s)
        std::printf(" %s=%.17e", strip_names[s], double(vmax[s]));
      std::printf(" nonfinite=%lld\n", (long long)nonfinite);
      std::fflush(stdout);
    }
  });
}

} // namespace AsterX
