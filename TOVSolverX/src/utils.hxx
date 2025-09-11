#include <loop.hxx>

namespace TOVSolverX {

using namespace Loop;
/* - utility routine
   - fills an real-array 'var' of size 'i' with value 'r' */
void TOVX_C_fill(CCTK_REAL *var, CCTK_INT i, CCTK_REAL r) {
  for (i--; i >= 0; i--)
    var[i] = r;
}

void TOVX_Copy(CCTK_INT size, CCTK_REAL *var_p, CCTK_REAL *var) {
#pragma omp parallel for
  for (int i = 0; i < size; i++)
    var_p[i] = var[i];
}

} // namespace TOVSolverX
