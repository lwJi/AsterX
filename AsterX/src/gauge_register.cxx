#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include "gauge_register.hxx"

namespace AsterX {
using namespace Loop;

////////////////////////////////////////////////////////////////////////////////

extern "C" void AsterX_GaugeRegisterParamCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  // The register is requested and would matter (Lorenz gauge), but the
  // driver's sync mode never traverses CCTK_POSTRESTRICT, so the reconcile
  // cannot run. Warn rather than abort: existing non-subcycled parfiles with
  // the CarpetX default restrict_during_sync = yes must keep running
  // bit-identically.
  if (gauge_register &&
      CCTK_EQUALS(vector_potential_gauge, "generalized Lorenz") &&
      carpetx_restrict_during_sync()) {
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

} // namespace AsterX
