#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include "../../../CarpetX/CarpetX/src/schedule.hxx"
#include "../../../CarpetX/CarpetX/src/task_manager.hxx"
#include "../../../CarpetX/CarpetX/src/fillpatch.hxx"

namespace AsterX {
using namespace CarpetX;

////////////////////////////////////////////////////////////////////////////////

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

} // namespace AsterX
