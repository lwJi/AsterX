#include <AMReX.hxx>
using namespace AMReX;

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_PlotFileUtil.H>
using namespace amrex;

#include <omp.h>

#include <cmath>
using namespace std;

namespace WaveToyAMReX {

// Linear interpolation between (i0, x0) and (i1, x1)
template <typename Y, typename X> Y linterp(Y y0, Y y1, X x0, X x1, X x) {
  return Y(x - x0) / Y(x1 - x0) * y0 + Y(x - x1) / Y(x0 - x1) * y1;
}

// Square
template <typename T> T sqr(T x) { return x * x; }

// Standing wave
template <typename T> T standing(T t, T x, T y, T z) {
  T Lx = 4, Ly = 4, Lz = 4;
  T kx = 2 * M_PI / Lx, ky = 2 * M_PI / Ly, kz = 2 * M_PI / Lz;
  T omega = sqrt(sqr(kx) + sqr(ky) + sqr(kz));
  return cos(omega * t) * cos(kx * x) * cos(ky * y) * cos(kz * z);
}

////////////////////////////////////////////////////////////////////////////////

extern "C" void WaveToyAMReX_Setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const CCTK_REAL *restrict const x0 = ghext->geom.ProbLo();
  const CCTK_REAL *restrict const x1 = ghext->geom.ProbHi();
  for (int d = 0; d < 3; ++d) {
    CCTK_REAL dx = (x1[d] - x0[d]) / ghext->ncells;
    cctkGH->cctk_origin_space[d] = x0[d] + 0.5 * dx;
    cctkGH->cctk_delta_space[d] = dx;
  }

  cctkGH->cctk_time = 0.0;
  cctkGH->cctk_delta_time = 0.5 / ghext->ncells;
}

extern "C" void WaveToyAMReX_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const CCTK_REAL t0 = cctk_time;
  const CCTK_REAL dt = cctk_delta_time;
  const CCTK_REAL *restrict const x0 = ghext->geom.ProbLo();
  const CCTK_REAL *restrict const x1 = ghext->geom.ProbHi();

  // Initialize phi
  const MFIter &mfi = *mfis.at(omp_get_thread_num());
  const Box &fbx = mfi.fabbox();
  const Box &bx = mfi.growntilebox();

  const Dim3 amin = lbound(fbx);
  const Dim3 amax = ubound(fbx);
  const Dim3 imin = lbound(bx);
  const Dim3 imax = ubound(bx);

  constexpr int di = 1;
  const int dj = di * (amax.x - amin.x + 1);
  const int dk = dj * (amax.y - amin.y + 1);

  const Array4<CCTK_REAL> &vars = ghext->mfab.array(mfi);
  CCTK_REAL *restrict const phi = vars.ptr(0, 0, 0, 0);
  CCTK_REAL *restrict const phi_p = vars.ptr(0, 0, 0, 1);

  for (int k = imin.z; k <= imax.z; ++k)
    for (int j = imin.y; j <= imax.y; ++j)
#pragma omp simd
      for (int i = imin.x; i <= imax.x; ++i) {
        const int idx = i * di + j * dj + k * dk;
        CCTK_REAL x = linterp(x0[0], x1[0], -1, 2 * ghext->ncells - 1, 2 * i);
        CCTK_REAL y = linterp(x0[1], x1[1], -1, 2 * ghext->ncells - 1, 2 * j);
        CCTK_REAL z = linterp(x0[2], x1[2], -1, 2 * ghext->ncells - 1, 2 * k);
        phi[idx] = standing(t0, x, y, z);
        phi_p[idx] = standing(t0 - dt, x, y, z);
      }
}

extern "C" void WaveToyAMReX_Cycle(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Cycle time levels
  MultiFab::Copy(ghext->mfab, ghext->mfab, 1, 2, 1, ghext->nghostzones);
  MultiFab::Copy(ghext->mfab, ghext->mfab, 0, 1, 1, ghext->nghostzones);

  // Step time
  cctkGH->cctk_time += cctkGH->cctk_delta_time;
}

extern "C" void WaveToyAMReX_Evolve(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Cycle time levels
  MultiFab::Copy(ghext->mfab, ghext->mfab, 1, 2, 1, ghext->nghostzones);
  MultiFab::Copy(ghext->mfab, ghext->mfab, 0, 1, 1, ghext->nghostzones);

  const CCTK_REAL *restrict const dx = cctk_delta_space;
  const CCTK_REAL dt = cctk_delta_time;

  // Evolve phi
  const MFIter &mfi = *mfis.at(omp_get_thread_num());
  const Box &fbx = mfi.fabbox();
  const Box &bx = mfi.tilebox();

  const Dim3 amin = lbound(fbx);
  const Dim3 amax = ubound(fbx);
  const Dim3 imin = lbound(bx);
  const Dim3 imax = ubound(bx);

  constexpr int di = 1;
  const int dj = di * (amax.x - amin.x + 1);
  const int dk = dj * (amax.y - amin.y + 1);

  const Array4<CCTK_REAL> &vars = ghext->mfab.array(mfi);
  CCTK_REAL *restrict const phi = vars.ptr(0, 0, 0, 0);
  const CCTK_REAL *restrict const phi_p = vars.ptr(0, 0, 0, 1);
  const CCTK_REAL *restrict const phi_p_p = vars.ptr(0, 0, 0, 2);

  for (int k = imin.z; k <= imax.z; ++k)
    for (int j = imin.y; j <= imax.y; ++j)
#pragma omp simd
      for (int i = imin.x; i <= imax.x; ++i) {
        const int idx = i * di + j * dj + k * dk;
        CCTK_REAL ddx_phi =
            (phi_p[idx - di] - 2 * phi_p[idx] + phi_p[idx + di]) / sqr(dx[0]);
        CCTK_REAL ddy_phi =
            (phi_p[idx - dj] - 2 * phi_p[idx] + phi_p[idx + dj]) / sqr(dx[1]);
        CCTK_REAL ddz_phi =
            (phi_p[idx - dk] - 2 * phi_p[idx] + phi_p[idx + dk]) / sqr(dx[2]);
        phi[idx] = -phi_p_p[idx] + 2 * phi_p[idx] +
                   sqr(dt) * (ddx_phi + ddy_phi + ddz_phi);
      }
}

extern "C" void WaveToyAMReX_Sync(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Synchronize
  ghext->mfab.FillBoundary(ghext->geom.periodicity());
}

extern "C" void WaveToyAMReX_Error(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const CCTK_REAL t0 = cctk_time;
  const CCTK_REAL *restrict const x0 = ghext->geom.ProbLo();
  const CCTK_REAL *restrict const x1 = ghext->geom.ProbHi();

  const MFIter &mfi = *mfis.at(omp_get_thread_num());
  const Box &fbx = mfi.fabbox();
  const Box &bx = mfi.growntilebox();

  const Dim3 amin = lbound(fbx);
  const Dim3 amax = ubound(fbx);
  const Dim3 imin = lbound(bx);
  const Dim3 imax = ubound(bx);

  constexpr int di = 1;
  const int dj = di * (amax.x - amin.x + 1);
  const int dk = dj * (amax.y - amin.y + 1);

  const Array4<CCTK_REAL> &vars = ghext->mfab.array(mfi);
  CCTK_REAL *restrict const err = vars.ptr(0, 0, 0, 3);
  const CCTK_REAL *restrict const phi = vars.ptr(0, 0, 0, 0);

  for (int k = imin.z; k <= imax.z; ++k)
    for (int j = imin.y; j <= imax.y; ++j)
#pragma omp simd
      for (int i = imin.x; i <= imax.x; ++i) {
        const int idx = i * di + j * dj + k * dk;
        CCTK_REAL x = linterp(x0[0], x1[0], -1, 2 * ghext->ncells - 1, 2 * i);
        CCTK_REAL y = linterp(x0[1], x1[1], -1, 2 * ghext->ncells - 1, 2 * j);
        CCTK_REAL z = linterp(x0[2], x1[2], -1, 2 * ghext->ncells - 1, 2 * k);
        err[idx] = phi[idx] - standing(t0, x, y, z);
      }
}

extern "C" void WaveToyAMReX_Output(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // Output phi
  string filename = amrex::Concatenate("wavetoy/phi", cctk_iteration, 6);
  WriteSingleLevelPlotfile(filename, ghext->mfab,
                           {"phi", "phi_p", "phi_p_p", "error"}, ghext->geom,
                           cctk_time, cctk_iteration);
}

} // namespace WaveToyAMReX
