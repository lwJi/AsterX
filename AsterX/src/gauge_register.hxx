#ifndef ASTERX_GAUGE_REGISTER_HXX
#define ASTERX_GAUGE_REGISTER_HXX

#include <cctk.h>
#include <cctk_Parameter.h>
#include <cctk_Parameters.h>

#include <cassert>

namespace AsterX {

// CarpetX::restrict_during_sync is declared PRIVATE in CarpetX, so it cannot
// be shared through param.ccl (the CST rejects USE of a non-restricted
// parameter). Read it through the flesh's generic accessor instead. It is
// not steerable, so the value is cached after the first lookup.
inline bool carpetx_restrict_during_sync() {
  static const bool value = [] {
    int type = -1;
    const void *const ptr =
        CCTK_ParameterGet("restrict_during_sync", "CarpetX", &type);
    assert(ptr);
    assert(type == PARAMETER_BOOLEAN);
    return bool(*static_cast<const CCTK_INT *>(ptr));
  }();
  return value;
}

// The nodal gauge register is active only when all three hold:
//   - AsterX::gauge_register = yes,
//   - the generalized Lorenz gauge is used (otherwise G == 0), and
//   - CarpetX::restrict_during_sync = no, because the reconcile hangs on the
//     CCTK_POSTRESTRICT bin, which CarpetX traverses only in that mode.
// When inactive, IG_rhs = 0 and no reconcile is scheduled, so the run is
// bit-identical to one without the register.
inline bool gauge_register_active() {
  DECLARE_CCTK_PARAMETERS;
  return gauge_register &&
         CCTK_EQUALS(vector_potential_gauge, "generalized Lorenz") &&
         !carpetx_restrict_during_sync();
}

} // namespace AsterX

#endif // #ifndef ASTERX_GAUGE_REGISTER_HXX
