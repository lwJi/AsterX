#ifndef IO_HXX
#define IO_HXX

#include <cctk.h>
#undef copysign
#undef fpclassify
#undef isfinite
#undef isinf
#undef isnan
#undef isnormal
#undef signbit

namespace AMReX {

int OutputGH(const cGH *cctkGH);
} // namespace AMReX

#endif // #ifndef IO_HXX
