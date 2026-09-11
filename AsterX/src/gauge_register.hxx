#ifndef ASTERX_GAUGE_REGISTER_HXX
#define ASTERX_GAUGE_REGISTER_HXX

#include <cctk.h>
#include <cctk_Parameters.h>

namespace AsterX {

// The nodal gauge register is active only when all three hold:
//   - AsterX::gauge_register = yes,
//   - the generalized Lorenz gauge is used (otherwise G == 0), and
//   - CarpetX::use_subcycling = yes, the only configuration in which the
//     coarse and fine levels accumulate different time integrals of G at
//     the nodes they share.
// Under subcycling CarpetX rejects its restrict-during-sync mode, so the
// CCTK_POSTRESTRICT bin the reconcile hangs on is always traversed when the
// register is active. The schedule.ccl block that schedules the reconcile
// tests the same three conditions, so when inactive IG_rhs = 0, no
// reconcile is scheduled, and the run is bit-identical to one without the
// register.
inline bool gauge_register_active() {
  DECLARE_CCTK_PARAMETERS;
  return gauge_register &&
         CCTK_EQUALS(vector_potential_gauge, "generalized Lorenz") &&
         use_subcycling;
}

} // namespace AsterX

#endif // #ifndef ASTERX_GAUGE_REGISTER_HXX
