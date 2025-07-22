#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include "../../../CarpetX/CarpetX/src/schedule.hxx"
#include "../../../CarpetX/CarpetX/src/task_manager.hxx"
#include "../../../CarpetX/CarpetX/src/fillpatch.hxx"

#include <AMReX_MultiFabUtil.H>

namespace AsterX {
using namespace CarpetX;

////////////////////////////////////////////////////////////////////////////////

void Restrict(const cGH *cctkGH, int level, const vector<int> &groups) {
  DECLARE_CCTK_PARAMETERS;

  for (const auto &patchdata : ghext->patchdata) {
    const int patch = patchdata.patch;
    if (level + 1 < int(patchdata.leveldata.size())) {
      auto &leveldata = patchdata.leveldata.at(level);
      const auto &fineleveldata = patchdata.leveldata.at(level + 1);
      const active_levels_t active_levels(level, level + 1, patch, patch + 1);
      const active_levels_t active_fine_levels(level + 1, level + 2, patch,
                                               patch + 1);

      for (const int gi : groups) {
        cGroup group;
        int ierr = CCTK_GroupData(gi, &group);
        assert(!ierr);

        assert(group.grouptype == CCTK_GF);

        auto &groupdata = *leveldata.groupdata.at(gi);
        const auto &finegroupdata = *fineleveldata.groupdata.at(gi);
        const amrex::IntVect reffact{2, 2, 2};

        int ntls = groupdata.mfab.size();
        int restrict_tl = ntls > 1 ? ntls - 1 : ntls;
        for (int tl = 0; tl < restrict_tl; ++tl) {

          // rank: 0: vertex, 1: edge, 2: face, 3: volume
          int rank = 0;
          for (int d = 0; d < dim; ++d)
            rank += groupdata.indextype.at(d);
          switch (rank) {
          case 0:
            average_down_nodal(*finegroupdata.mfab.at(tl),
                               *groupdata.mfab.at(tl), reffact);
            break;
          case 1:
            average_down_edges(*finegroupdata.mfab.at(tl),
                               *groupdata.mfab.at(tl), reffact);
            break;
          case 2:
            average_down_faces(*finegroupdata.mfab.at(tl),
                               *groupdata.mfab.at(tl), reffact);
            break;
          case 3:
            average_down(*finegroupdata.mfab.at(tl), *groupdata.mfab.at(tl), 0,
                         groupdata.numvars, reffact);
            break;
          default:
            assert(0);
          }

        } // for tl
      } // for gi
    } // if level exists
  } // for patchdata
}

extern "C" void AsterX_Sync(CCTK_ARGUMENTS) {
  // do nothing
}

void ApplyOuterBC(CCTK_ARGUMENTS, std::vector<int> &groups) {
  task_manager tasks1;
  task_manager tasks2;

  for (const int gi : groups) {
    active_levels->loop_serially([&](auto &restrict leveldata) {
      auto &restrict groupdata = *leveldata.groupdata.at(gi);

      const int ntls = groupdata.mfab.size();
      const int sync_tl = ntls > 1 ? ntls - 1 : ntls;

      if (leveldata.level == 0) {
        // Copy from adjacent boxes on same level and apply boundary conditions
        for (int tl = 0; tl < sync_tl; ++tl) {
          tasks1.submit_serially([&tasks2, &leveldata, &groupdata, tl]() {
            FillPatch_Sync(tasks2, groupdata, *groupdata.mfab.at(tl),
                           ghext->patchdata.at(leveldata.patch)
                               .amrcore->Geom(leveldata.level));
          });
        } // for tl
      }
    });
  } // for gi

  tasks1.run_tasks_serially();
  synchronize();
  tasks2.run_tasks_serially();
  synchronize();

  assert(ghext->num_patches() == 1);
}

extern "C" void AsterX_ApplyOuterBCOnPrim(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  std::vector<int> groups;

  groups.push_back(CCTK_GroupIndex("HydroBaseX::rho"));
  groups.push_back(CCTK_GroupIndex("HydroBaseX::vel"));
  groups.push_back(CCTK_GroupIndex("HydroBaseX::eps"));
  groups.push_back(CCTK_GroupIndex("HydroBaseX::press"));
  groups.push_back(CCTK_GroupIndex("HydroBaseX::Bvec"));
  groups.push_back(CCTK_GroupIndex("HydroBaseX::temperature"));
  groups.push_back(CCTK_GroupIndex("HydroBaseX::entropy"));
  groups.push_back(CCTK_GroupIndex("HydroBaseX::Ye"));

  groups.push_back(CCTK_GroupIndex("AsterX::zvec"));
  groups.push_back(CCTK_GroupIndex("AsterX::svec"));

  groups.push_back(CCTK_GroupIndex("AsterX::dBx_stag"));
  groups.push_back(CCTK_GroupIndex("AsterX::dBy_stag"));
  groups.push_back(CCTK_GroupIndex("AsterX::dBz_stag"));

  ApplyOuterBC(CCTK_PASS_CTOC, groups);
}

extern "C" void AsterX_ApplyOuterBCOnFluxes(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  std::vector<int> groups;

  groups.push_back(CCTK_GroupIndex("AsterX::flux_x"));
  groups.push_back(CCTK_GroupIndex("AsterX::flux_y"));
  groups.push_back(CCTK_GroupIndex("AsterX::flux_z"));

  groups.push_back(CCTK_GroupIndex("AsterX::vtilde_xface"));
  groups.push_back(CCTK_GroupIndex("AsterX::vtilde_yface"));
  groups.push_back(CCTK_GroupIndex("AsterX::vtilde_zface"));
  groups.push_back(CCTK_GroupIndex("AsterX::a_xface"));
  groups.push_back(CCTK_GroupIndex("AsterX::a_yface"));
  groups.push_back(CCTK_GroupIndex("AsterX::a_zface"));

  ApplyOuterBC(CCTK_PASS_CTOC, groups);
}

extern "C" void AsterX_RestrictFluxes(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  std::vector<int> groups;

  groups.push_back(CCTK_GroupIndex("AsterX::flux_x"));
  groups.push_back(CCTK_GroupIndex("AsterX::flux_y"));
  groups.push_back(CCTK_GroupIndex("AsterX::flux_z"));

  groups.push_back(CCTK_GroupIndex("AsterX::vtilde_xface"));
  groups.push_back(CCTK_GroupIndex("AsterX::vtilde_yface"));
  groups.push_back(CCTK_GroupIndex("AsterX::vtilde_zface"));
  groups.push_back(CCTK_GroupIndex("AsterX::a_xface"));
  groups.push_back(CCTK_GroupIndex("AsterX::a_yface"));
  groups.push_back(CCTK_GroupIndex("AsterX::a_zface"));

  active_levels->loop_fine_to_coarse([&](const auto &leveldata) {
    if (leveldata.level < ghext->num_levels() - 1)
      Restrict(cctkGH, leveldata.level, groups);
  });

  assert(ghext->num_patches() == 1);
}

} // namespace AsterX
