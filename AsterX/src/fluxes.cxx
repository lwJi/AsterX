#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <mat.hxx>
#include <simd.hxx>
#include <sum.hxx>
#include <vec.hxx>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

#include "aster_utils.hxx"
#include "eigenvalues.hxx"
#include "fluxes.hxx"
#include "reconstruct.hxx"
#include "setup_eos.hxx"

namespace AsterX {
using namespace std;
using namespace Loop;
using namespace Arith;
using namespace EOSX;
using namespace ReconX;
using namespace AsterUtils;

enum class flux_t { LxF, HLLE };
enum class eos_3param { IdealGas, Hybrid, Tabulated };
enum class rec_var_t { v_vec, z_vec, s_vec };

// Calculate the fluxes in direction `dir`. This function is more
// complex because it has to handle any direction, but as reward,
// there is only one function, not three.
template <int dir_i, typename EOSType>
void CalcFlux(CCTK_ARGUMENTS, EOSType *eos_3p, const rec_var_t rec_var,
              const reconstruction_t reconstruction,
              const reconstruction_t reconstruction_LO,
              const reconstruct_params_t reconstruct_params,
              const flux_t fluxtype) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_Fluxes;
  DECLARE_CCTK_PARAMETERS;

  switch (reconstruction) {
  case reconstruction_t::Godunov:
    assert(cctk_nghostzones[dir_i] >= 1);
    break;
  case reconstruction_t::minmod:
    assert(cctk_nghostzones[dir_i] >= 2);
    break;
  case reconstruction_t::monocentral:
    assert(cctk_nghostzones[dir_i] >= 2);
    break;
  case reconstruction_t::ppm:
    assert(cctk_nghostzones[dir_i] >= 3);
    break;
  case reconstruction_t::eppm:
    assert(cctk_nghostzones[dir_i] >= 3);
    break;
  case reconstruction_t::wenoz:
    assert(cctk_nghostzones[dir_i] >= 3);
  case reconstruction_t::mp5:
    assert(cctk_nghostzones[dir_i] >= 3);
    break;
  }

  /* grid functions for fluxes */
  const vec<GF3D2<CCTK_REAL>, dim> fluxdenss{fxdens, fydens, fzdens};
  const vec<GF3D2<CCTK_REAL>, dim> fluxDEnts{fxDEnt, fyDEnt, fzDEnt};
  const vec<GF3D2<CCTK_REAL>, dim> fluxmomxs{fxmomx, fymomx, fzmomx};
  const vec<GF3D2<CCTK_REAL>, dim> fluxmomys{fxmomy, fymomy, fzmomy};
  const vec<GF3D2<CCTK_REAL>, dim> fluxmomzs{fxmomz, fymomz, fzmomz};
  const vec<GF3D2<CCTK_REAL>, dim> fluxtaus{fxtau, fytau, fztau};
  const vec<GF3D2<CCTK_REAL>, dim> fluxDYes{fxDYe, fyDYe, fzDYe};
  const vec<GF3D2<CCTK_REAL>, dim> fluxBxs{fxBx, fyBx, fzBx};
  const vec<GF3D2<CCTK_REAL>, dim> fluxBys{fxBy, fyBy, fzBy};
  const vec<GF3D2<CCTK_REAL>, dim> fluxBzs{fxBz, fyBz, fzBz};

  /* grid functions */
  const vec<GF3D2<const CCTK_REAL>, dim> gf_vels{velx, vely, velz};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_zvec{zvec_x, zvec_y, zvec_z};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_svec{svec_x, svec_y, svec_z};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_Bvecs{Bvecx, Bvecy, Bvecz};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_dBstags{dBx_stag, dBy_stag,
                                                    dBz_stag};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_beta{betax, betay, betaz};
  const smat<GF3D2<const CCTK_REAL>, dim> gf_g{gxx, gxy, gxz, gyy, gyz, gzz};

  /* grid functions for Upwind CT */
  const vec<GF3D2<CCTK_REAL>, dim> vbar_j{vbar_y_xface, vbar_z_yface,
                                          vbar_x_zface};
  const vec<GF3D2<CCTK_REAL>, dim> vbar_k{vbar_z_xface, vbar_x_yface,
                                          vbar_y_zface};
  const vec<GF3D2<CCTK_REAL>, dim> ap_face{amax_xface, amax_yface, amax_zface};
  const vec<GF3D2<CCTK_REAL>, dim> am_face{amin_xface, amin_yface, amin_zface};

  static_assert(dir_i >= 0 && dir_i < 3, "");

  // Prebind the velocity slice for this direction once
  const auto gf_vel_dir_i = gf_vels(dir_i);

  const auto reconstruct_pt =
      [=] CCTK_DEVICE(const GF3D2<const CCTK_REAL> &var, const PointDesc &p,
                      bool gf_is_rho, bool gf_is_press) {
        return reconstruct<vec<CCTK_REAL, 2>>(var, p, reconstruction, dir_i,
                                              gf_is_rho, gf_is_press, press,
                                              gf_vel_dir_i, reconstruct_params);
      };
  const auto reconstruct_loworder =
      [=] CCTK_DEVICE(const GF3D2<const CCTK_REAL> &var, const PointDesc &p,
                      bool gf_is_rho, bool gf_is_press) {
        return reconstruct<vec<CCTK_REAL, 2>>(var, p, reconstruction_LO, dir_i,
                                              gf_is_rho, gf_is_press, press,
                                              gf_vel_dir_i, reconstruct_params);
      };

  const auto calcflux =
      [=] CCTK_DEVICE(vec<vec<CCTK_REAL, 4>, 2> lam, vec<CCTK_REAL, 2> var,
                      vec<CCTK_REAL, 2> flux) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        CCTK_REAL flx;
        switch (fluxtype) {
        case flux_t::LxF: {
          flx = laxf(lam, var, flux);
          break;
        }
        case flux_t::HLLE: {
          flx = hlle(lam, var, flux);
          break;
        }
        default:
          assert(0);
        }
        return flx;
      };

  // Face-centred grid functions (in direction `dir_i`)
  constexpr array<int, dim> face_centred = {!(dir_i == 0), !(dir_i == 1),
                                            !(dir_i == 2)};

  constexpr int dir_j = (dir_i == 0) ? 1 : ((dir_i == 1) ? 2 : 0);
  constexpr int dir_k = (dir_i == 0) ? 2 : ((dir_i == 1) ? 0 : 1);

  // initialize to zero
  grid.loop_all_device<face_centred[0], face_centred[1], face_centred[2]>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        fluxdenss(dir_i)(p.I) = 0;
        fluxDEnts(dir_i)(p.I) = 0;
        fluxmomxs(dir_i)(p.I) = 0;
        fluxmomys(dir_i)(p.I) = 0;
        fluxmomzs(dir_i)(p.I) = 0;
        fluxtaus(dir_i)(p.I) = 0;
        fluxDYes(dir_i)(p.I) = 0;
        fluxBxs(dir_i)(p.I) = 0;
        fluxBys(dir_i)(p.I) = 0;
        fluxBzs(dir_i)(p.I) = 0;

        ap_face(dir_i)(p.I) = 0;
        am_face(dir_i)(p.I) = 0;
        vbar_j(dir_i)(p.I) = 0;
        vbar_k(dir_i)(p.I) = 0;
      });

  grid.loop_mix_device<face_centred[0], face_centred[1],
                       face_centred[2]>(grid.nghostzones, [=] CCTK_DEVICE(
                                                              const PointDesc
                                                                  &p) {
    /* Reconstruct primitives from the cells on left (indice 0) and right
     * (indice 1) side of this face rc = reconstructed variables or
     * computed from reconstructed variables */

    /* Interpolate metric components from vertices to faces */
    const CCTK_REAL alp_avg = calc_avg_v2f<dir_i>(alp, p);
    const vec<CCTK_REAL, 3> betas_avg(
        [&](int i) ARITH_INLINE { return calc_avg_v2f<dir_i>(gf_beta(i), p); });
    const smat<CCTK_REAL, 3> g_avg([&](int i, int j) ARITH_INLINE {
      return calc_avg_v2f<dir_i>(gf_g(i, j), p);
    });

    /* determinant of spatial metric */
    const CCTK_REAL detg_avg = calc_det(g_avg);
    const CCTK_REAL sqrtg = sqrt(detg_avg);

    // Boolean to decide whether to use low order
    // reconstruction
    bool useLO = false;

    // Reconstruct density
    auto rho_rc = reconstruct_pt(rho, p, true, true);

    // Reconstruct entropy
    auto entropy_rc = reconstruct_pt(entropy, p, false, false);

    // Reconstruct Ye
    auto Ye_rc = reconstruct_pt(Ye, p, false, false);

    // Initialize variables for eps, pressure, and temperature
    vec<CCTK_REAL, 2> eps_rc;
    vec<CCTK_REAL, 2> press_rc;
    vec<CCTK_REAL, 2> temp_rc;

    if (reconstruct_with_temperature) {

      // Reconstruct temperature
      temp_rc = reconstruct_pt(temperature, p, false, false);

      // Use lower-order if reconstructed rho, entropy, Ye or T is <= 0
      if ((rho_rc(0) <= 0.0) || (entropy_rc(0) <= 0.0) || (Ye_rc(0) <= 0.0) ||
          (temp_rc(0) <= 0.0) || (rho_rc(1) <= 0.0) || (entropy_rc(1) <= 0.0) ||
          (Ye_rc(1) <= 0.0) || (temp_rc(1) <= 0.0)) {

        useLO = true;

        rho_rc = reconstruct_loworder(rho, p, true, true);
        entropy_rc = reconstruct_loworder(entropy, p, false, false);
        Ye_rc = reconstruct_loworder(Ye, p, false, false);
        temp_rc = reconstruct_loworder(temperature, p, false, false);
      }
      // End lower-order

      // Compute eps_rc and press_rc using lambdas
      for (int f = 0; f < 2; ++f) {
        eps_rc(f) =
            eos_3p->eps_from_valid_rho_temp_ye(rho_rc(f), temp_rc(f), Ye_rc(f));
        press_rc(f) = eos_3p->press_from_valid_rho_temp_ye(
            rho_rc(f), temp_rc(f), Ye_rc(f));
      }

    } else {

      // Reconstruct pressure
      press_rc = reconstruct_pt(press, p, false, true);

      // Use lower-order if reconstructed rho, entropy, Ye or pressure is <= 0
      if ((rho_rc(0) <= 0.0) || (entropy_rc(0) <= 0.0) || (Ye_rc(0) <= 0.0) ||
          (press_rc(0) <= 0.0) || (rho_rc(1) <= 0.0) ||
          (entropy_rc(1) <= 0.0) || (Ye_rc(1) <= 0.0) || (press_rc(1) <= 0.0)) {

        useLO = true;

        rho_rc = reconstruct_loworder(rho, p, true, true);
        entropy_rc = reconstruct_loworder(entropy, p, false, false);
        Ye_rc = reconstruct_loworder(Ye, p, false, false);
        press_rc = reconstruct_loworder(press, p, false, true);
      }
      // End lower-order

      // Compute eps_rc and temp_rc using lambdas
      for (int f = 0; f < 2; ++f) {
        eps_rc(f) = eos_3p->eps_from_valid_rho_press_ye(rho_rc(f), press_rc(f),
                                                        Ye_rc(f));
        temp_rc(f) =
            eos_3p->temp_from_valid_rho_eps_ye(rho_rc(f), eps_rc(f), Ye_rc(f));
      }
    }

    const vec<CCTK_REAL, 2> rhoh_rc([&](int f) ARITH_INLINE {
      return rho_rc(f) + rho_rc(f) * eps_rc(f) + press_rc(f);
    });

    // Introduce reconstructed Bs
    // Use staggered dB for i == dir_i
    vec<vec<CCTK_REAL, 2>, 3> Bs_rc;

    // Assign the value for the primary direction
    const CCTK_REAL val = gf_dBstags(dir_i)(p.I) / sqrtg;
    Bs_rc(dir_i)(0) = val;
    Bs_rc(dir_i)(1) = val;

    // Lambda to assign the reconstructed values
    auto assign_reconstructed = [&](int d) {
      auto tmp = useLO ? reconstruct_loworder(gf_Bvecs(d), p, false, false)
                       : reconstruct_pt(gf_Bvecs(d), p, false, false);
      Bs_rc(d)(0) = tmp(0);
      Bs_rc(d)(1) = tmp(1);
    };

    // Assign reconstructed values for the two perpendicular directions
    assign_reconstructed(dir_j);
    assign_reconstructed(dir_k);
    // End of setting Bs

    vec<vec<CCTK_REAL, 2>, 3> vels_rc;
    vec<vec<CCTK_REAL, 2>, 3> vlows_rc;
    vec<CCTK_REAL, 2> w_lorentz_rc;
    switch (rec_var) {
    case rec_var_t::v_vec: {

      if (useLO) {

        for (int i = 0; i <= 2; ++i) { // loop over components
          vels_rc(i) = reconstruct_loworder(gf_vels(i), p, false, false);
        }
      } else {

        for (int i = 0; i <= 2; ++i) { // loop over components
          vels_rc(i) = reconstruct_pt(gf_vels(i), p, false, false);
        }
      }

      /* co-velocity measured by Eulerian observer: v_j */
      vlows_rc = calc_contraction(g_avg, vels_rc);
      const auto v2_rc = calc_contraction(vlows_rc, vels_rc);

      /* Lorentz factor: W = 1 / sqrt(1 - v^2) */
      w_lorentz_rc(0) = 1 / sqrt(1 - v2_rc(0));
      w_lorentz_rc(1) = 1 / sqrt(1 - v2_rc(1));
      break;
    };
    case rec_var_t::z_vec: {

      vec<vec<CCTK_REAL, 2>, 3> zvec_rc([&](int i) ARITH_INLINE {
        return reconstruct_pt(gf_zvec(i), p, false, false);
      });

      // Lower-order
      if (useLO) {

        for (int i = 0; i <= 2; ++i) { // loop over components
          zvec_rc(i) = reconstruct_loworder(gf_zvec(i), p, false, false);
        }
      }
      // End lower-order

      const vec<vec<CCTK_REAL, 2>, 3> zveclow_rc =
          calc_contraction(g_avg, zvec_rc);
      const auto z2_rc = calc_contraction(zveclow_rc, zvec_rc);

      w_lorentz_rc(0) = sqrt(1 + z2_rc(0));
      w_lorentz_rc(1) = sqrt(1 + z2_rc(1));

      for (int i = 0; i <= 2; ++i) {   // loop over components
        for (int j = 0; j <= 1; ++j) { // loop over left and right state
          vels_rc(i)(j) = zvec_rc(i)(j) / w_lorentz_rc(j);
          vlows_rc(i)(j) = zveclow_rc(i)(j) / w_lorentz_rc(j);
        }
      }
      break;
    };
    case rec_var_t::s_vec: {

      vec<vec<CCTK_REAL, 2>, 3> svec_rc([&](int i) ARITH_INLINE {
        return reconstruct_pt(gf_svec(i), p, false, false);
      });

      // Lower-order
      if (useLO) {

        for (int i = 0; i <= 2; ++i) { // loop over components
          svec_rc(i) = reconstruct_loworder(gf_svec(i), p, false, false);
        }
      }
      // End lower-order

      const vec<vec<CCTK_REAL, 2>, 3> sveclow_rc =
          calc_contraction(g_avg, svec_rc);
      const auto s2_rc = calc_contraction(sveclow_rc, svec_rc);

      w_lorentz_rc(0) =
          sqrt(0.5 + sqrt(0.25 + s2_rc(0) / rhoh_rc(0) / rhoh_rc(0)));
      w_lorentz_rc(1) =
          sqrt(0.5 + sqrt(0.25 + s2_rc(1) / rhoh_rc(1) / rhoh_rc(1)));

      for (int i = 0; i <= 2; ++i) {   // loop over components
        for (int j = 0; j <= 1; ++j) { // loop over left and right state
          vels_rc(i)(j) =
              svec_rc(i)(j) / w_lorentz_rc(j) / w_lorentz_rc(j) / rhoh_rc(j);
          vlows_rc(i)(j) =
              sveclow_rc(i)(j) / w_lorentz_rc(j) / w_lorentz_rc(j) / rhoh_rc(j);
        }
      }
      break;
    };
    }

    /* vtilde^i = alpha * v^i - beta^i */
    const vec<vec<CCTK_REAL, 2>, 3> vtildes_rc([&](int i) ARITH_INLINE {
      return vec<CCTK_REAL, 2>([&](int f) ARITH_INLINE {
        return alp_avg * vels_rc(i)(f) - betas_avg(i);
      });
    });

    /* alpha * b0 = W * B^i * v_i */
    const vec<CCTK_REAL, 2> alp_b0_rc([&](int f) ARITH_INLINE {
      return w_lorentz_rc(f) * calc_contraction(Bs_rc, vlows_rc)(f);
    });
    /* covariant magnetic field measured by the Eulerian observer */
    const vec<vec<CCTK_REAL, 2>, 3> Blows_rc = calc_contraction(g_avg, Bs_rc);
    /* B^2 = B^i * B_i */
    const vec<CCTK_REAL, 2> B2_rc = calc_contraction(Bs_rc, Blows_rc);
    /* covariant magnetic field measured by the comoving observer:
     *  b_i = B_i/W + alpha*b^0*v_i */
    const vec<vec<CCTK_REAL, 2>, 3> blows_rc([&](int i) ARITH_INLINE {
      return vec<CCTK_REAL, 2>([&](int f) ARITH_INLINE {
        return Blows_rc(i)(f) / w_lorentz_rc(f) + alp_b0_rc(f) * vlows_rc(i)(f);
      });
    });
    /* b^2 = b^{\mu} * b_{\mu} */
    const vec<CCTK_REAL, 2> bsq_rc([&](int f) ARITH_INLINE {
      return (B2_rc(f) + pow2(alp_b0_rc(f))) / pow2(w_lorentz_rc(f));
    });

    /* componets correspond to the dir_i we are considering */
    const CCTK_REAL beta_avg = betas_avg(dir_i);
    const vec<CCTK_REAL, 2> vel_rc{vels_rc(dir_i)};
    const vec<CCTK_REAL, 2> B_rc{Bs_rc(dir_i)};
    const vec<CCTK_REAL, 2> vtilde_rc{vtildes_rc(dir_i)};

    // TODO: Compute pressure based on user-specified EOS.
    // Currently, computing press for classical ideal gas from reconstructed
    // vars

    const vec<CCTK_REAL, 2> cs2_rc([&](int f) ARITH_INLINE {
      return eos_3p->csnd_from_valid_rho_eps_ye(rho_rc(f), eps_rc(f),
                                                Ye_rc(f)) *
             eos_3p->csnd_from_valid_rho_eps_ye(rho_rc(f), eps_rc(f), Ye_rc(f));
    });

    const vec<CCTK_REAL, 2> h_rc([&](int f) ARITH_INLINE {
      return 1 + eps_rc(f) + press_rc(f) / rho_rc(f);
    });

    /* Computing conservatives from primitives: */

    /* dens = sqrt(g) * D = sqrt(g) * (rho * W) */
    const vec<CCTK_REAL, 2> dens_rc([&](int f) ARITH_INLINE {
      return sqrtg * rho_rc(f) * w_lorentz_rc(f);
    });

    /* DEnt = sqrt(g) * D * s  = sqrt(g) * (rho * W) * s */
    /*    s = entropy */
    const vec<CCTK_REAL, 2> DEnt_rc([&](int f) ARITH_INLINE {
      return sqrtg * rho_rc(f) * w_lorentz_rc(f) * entropy_rc(f);
    });

    /* auxiliary: dens * h * W = sqrt(g) * rho * h * W^2 */
    const vec<CCTK_REAL, 2> dens_h_W_rc([&](int f) ARITH_INLINE {
      return dens_rc(f) * h_rc(f) * w_lorentz_rc(f);
    });
    /* auxiliary: sqrt(g) * (rho*h + b^2)*W^2 */
    const vec<CCTK_REAL, 2> dens_h_W_plus_sqrtg_W2b2_rc =
        dens_h_W_rc + sqrtg * (pow2(alp_b0_rc) + B2_rc);
    /* auxiliary: (pgas + pmag) */
    const vec<CCTK_REAL, 2> press_plus_pmag_rc = press_rc + 0.5 * bsq_rc;

    /* mom_i = sqrt(g)*S_i = sqrt(g)((rho*h+b^2)*W^2*v_i - alpha*b^0*b_i) */
    const vec<vec<CCTK_REAL, 2>, 3> moms_rc([&](int i) ARITH_INLINE {
      return vec<CCTK_REAL, 2>([&](int f) ARITH_INLINE {
        return dens_h_W_plus_sqrtg_W2b2_rc(f) * vlows_rc(i)(f) -
               sqrtg * alp_b0_rc(f) * blows_rc(i)(f);
      });
    });

    /* tau = sqrt(g)*t =
     *  sqrt(g)((rho*h + b^2)*W^2 - (pgas+pmag) - (alpha*b^0)^2 - D) */
    const vec<CCTK_REAL, 2> tau_rc =
        dens_h_W_rc - dens_rc + sqrtg * (B2_rc - press_plus_pmag_rc);

    /* Btildes^i = sqrt(g) * B^i */
    const vec<vec<CCTK_REAL, 2>, 3> Btildes_rc(
        [&](int i) ARITH_INLINE { return sqrtg * Bs_rc(i); });

    /* Computing fluxes of conserved variables: */

    /* auxiliary: unit in 'dir_i' */
    const vec<CCTK_REAL, 3> unit_dir_i{vec<int, 3>::unit(dir_i)};
    /* auxiliary: alpha * sqrt(g) */
    const CCTK_REAL alp_sqrtg = alp_avg * sqrtg;
    /* auxiliary: B^i / W */
    const vec<CCTK_REAL, 2> B_over_w_lorentz_rc(
        [&](int f) ARITH_INLINE { return B_rc(f) / w_lorentz_rc(f); });

    /* flux(dens) = sqrt(g) * D * vtilde^i = sqrt(g) * rho * W * vtilde^i */
    const vec<CCTK_REAL, 2> flux_dens(
        [&](int f) ARITH_INLINE { return dens_rc(f) * vtilde_rc(f); });

    /* flux(DEnt) = sqrt(g) * D * s * vtilde^i = sqrt(g) * rho * W * s *
     * vtilde^i */
    const vec<CCTK_REAL, 2> flux_DEnt(
        [&](int f) ARITH_INLINE { return DEnt_rc(f) * vtilde_rc(f); });

    /* flux(mom_j)^i = sqrt(g)*(
     *  S_j*vtilde^i + alpha*((pgas+pmag)*delta^i_j - b_jB^i/W) ) */
    const vec<vec<CCTK_REAL, 2>, 3> flux_moms([&](int j) ARITH_INLINE {
      return vec<CCTK_REAL, 2>([&](int f) ARITH_INLINE {
        return moms_rc(j)(f) * vtilde_rc(f) +
               alp_sqrtg * (press_plus_pmag_rc(f) * unit_dir_i(j) -
                            blows_rc(j)(f) * B_over_w_lorentz_rc(f));
      });
    });

    /* flux(tau) = sqrt(g)*(
     *  t*vtilde^i + alpha*((pgas+pmag)*v^i-alpha*b0*B^i/W) ) */
    const vec<CCTK_REAL, 2> flux_tau([&](int f) ARITH_INLINE {
      return tau_rc(f) * vtilde_rc(f) +
             alp_sqrtg * (press_plus_pmag_rc(f) * vel_rc(f) -
                          alp_b0_rc(f) * B_over_w_lorentz_rc(f));
    });

    /* flux(DYe) = sqrt(g) * (D * Ye * vtilde^i) */
    const vec<CCTK_REAL, 2> DYe_rc(
        [&](int f) ARITH_INLINE { return dens_rc(f) * Ye_rc(f); });
    const vec<CCTK_REAL, 2> flux_DYe(
        [&](int f) ARITH_INLINE { return DYe_rc(f) * vtilde_rc(f); });

    /* electric field E_i = \tilde\epsilon_{ijk} Btilde_j * vtilde_k */
    const vec<vec<CCTK_REAL, 2>, 3> Es_rc =
        calc_cross_product(Btildes_rc, vtildes_rc);
    /* flux(Btildes) = {{0, -Ez, Ey}, {Ez, 0, -Ex}, {-Ey, Ex, 0}} */
    const vec<vec<CCTK_REAL, 2>, 3> flux_Btildes =
        calc_cross_product(unit_dir_i, Es_rc);

    /* Calculate eigenvalues: */

    /* variable for either g^xx, g^yy or g^zz depending on the direction */
    const CCTK_REAL u_avg = calc_inv(g_avg, detg_avg)(dir_i, dir_i);
    /* eigenvalues */
    vec<vec<CCTK_REAL, 4>, 2> lambda =
        eigenvalues(alp_avg, beta_avg, u_avg, vel_rc, rho_rc, cs2_rc,
                    w_lorentz_rc, h_rc, bsq_rc);

    /* Calculate numerical fluxes */
    fluxdenss(dir_i)(p.I) = calcflux(lambda, dens_rc, flux_dens);
    fluxDEnts(dir_i)(p.I) = calcflux(lambda, DEnt_rc, flux_DEnt);
    fluxmomxs(dir_i)(p.I) = calcflux(lambda, moms_rc(0), flux_moms(0));
    fluxmomys(dir_i)(p.I) = calcflux(lambda, moms_rc(1), flux_moms(1));
    fluxmomzs(dir_i)(p.I) = calcflux(lambda, moms_rc(2), flux_moms(2));
    fluxtaus(dir_i)(p.I) = calcflux(lambda, tau_rc, flux_tau);
    fluxDYes(dir_i)(p.I) = calcflux(lambda, DYe_rc, flux_DYe);
    fluxBxs(dir_i)(p.I) =
        (dir_i != 0) * calcflux(lambda, Btildes_rc(0), flux_Btildes(0));
    fluxBys(dir_i)(p.I) =
        (dir_i != 1) * calcflux(lambda, Btildes_rc(1), flux_Btildes(1));
    fluxBzs(dir_i)(p.I) =
        (dir_i != 2) * calcflux(lambda, Btildes_rc(2), flux_Btildes(2));

#ifdef CCTK_DEBUG
    if (isnan(dens_rc(0)) || isnan(dens_rc(1)) || isnan(moms_rc(0)(0)) ||
        isnan(moms_rc(0)(1)) || isnan(moms_rc(1)(0)) || isnan(moms_rc(1)(1)) ||
        isnan(moms_rc(2)(0)) || isnan(moms_rc(2)(1)) || isnan(tau_rc(0)) ||
        isnan(tau_rc(1)) || isnan(Btildes_rc(0)(0)) ||
        isnan(Btildes_rc(0)(1)) || isnan(Btildes_rc(1)(0)) ||
        isnan(Btildes_rc(1)(1)) || isnan(Btildes_rc(2)(0)) ||
        isnan(Btildes_rc(2)(1)) || isnan(DYe_rc(0)) || isnan(DYe_rc(1)) ||
        isnan(flux_dens(0)) || isnan(flux_dens(1)) || isnan(flux_moms(0)(0)) ||
        isnan(flux_moms(0)(1)) || isnan(flux_moms(1)(0)) ||
        isnan(flux_moms(1)(1)) || isnan(flux_moms(2)(0)) ||
        isnan(flux_moms(2)(1)) || isnan(flux_DYe(0)) || isnan(flux_DYe(1)) ||
        isnan(flux_tau(0)) || isnan(flux_tau(1)) || isnan(flux_Btildes(0)(0)) ||
        isnan(flux_Btildes(0)(1)) || isnan(flux_Btildes(1)(0)) ||
        isnan(flux_Btildes(1)(1)) || isnan(flux_Btildes(2)(0)) ||
        isnan(flux_Btildes(2)(1)) || isnan(fluxdenss(dir_i)(p.I)) ||
        isnan(fluxmomxs(dir_i)(p.I)) || isnan(fluxmomys(dir_i)(p.I)) ||
        isnan(fluxmomzs(dir_i)(p.I)) || isnan(fluxtaus(dir_i)(p.I)) ||
        isnan(fluxBxs(dir_i)(p.I)) || isnan(fluxBys(dir_i)(p.I)) ||
        isnan(fluxBzs(dir_i)(p.I)) || rho_rc(0) < 0.0 || rho_rc(1) < 0.0 ||
        press_rc(0) < 0.0 || press_rc(1) < 0.0) {
      printf("cctk_iteration = %i,  dir_i = %i,  ijk = %i, %i, %i, "
             "x, y, z = %16.8e, %16.8e, %16.8e.\n",
             cctk_iteration, dir_i, p.i, p.j, p.k, p.x, p.y, p.z);
      printf("  fluxdenss = %16.8e,\n", fluxdenss(dir_i)(p.I));
      printf("  fluxmoms  = %16.8e, %16.8e, %16.8e,\n", fluxmomxs(dir_i)(p.I),
             fluxmomys(dir_i)(p.I), fluxmomzs(dir_i)(p.I));
      printf("  fluxtaus  = %16.8e,\n", fluxtaus(dir_i)(p.I));
      printf("  fluxBs    = %16.8e, %16.8e, %16.8e\n", fluxBxs(dir_i)(p.I),
             fluxBys(dir_i)(p.I), fluxBzs(dir_i)(p.I));
      printf("  flux_denss = %16.8e, %16.8e,\n", flux_dens(0), flux_dens(1));
      printf("  flux_moms  = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e,\n",
             flux_moms(0)(0), flux_moms(0)(1), flux_moms(1)(0), flux_moms(1)(1),
             flux_moms(2)(0), flux_moms(2)(1));
      printf("  flux_taus  = %16.8e, %16.8e,\n", flux_tau(0), flux_tau(1));
      printf("  flux_DYes  = %16.8e, %16.8e,\n", flux_DYe(0), flux_DYe(1));
      printf("  flux_Bts   = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e,\n",
             flux_Btildes(0)(0), flux_Btildes(0)(1), flux_Btildes(1)(0),
             flux_Btildes(1)(1), flux_Btildes(2)(0), flux_Btildes(2)(1));
      printf("  dens_rc = %16.8e, %16.8e,\n", dens_rc(0), dens_rc(1));
      printf("  moms_rc = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e,\n",
             moms_rc(0)(0), moms_rc(0)(1), moms_rc(1)(0), moms_rc(1)(1),
             moms_rc(2)(0), moms_rc(2)(1));
      printf("  tau_rc  = %16.8e, %16.8e,\n", tau_rc(0), tau_rc(1));
      printf("  DYe_rc  = %16.8e, %16.8e,\n", DYe_rc(0), DYe_rc(1));
      printf("  Bs_rc  = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e,\n",
             Bs_rc(0)(0), Bs_rc(0)(1), Bs_rc(1)(0), Bs_rc(1)(1), Bs_rc(2)(0),
             Bs_rc(2)(1));
      printf("  Bts_rc  = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e,\n",
             Btildes_rc(0)(0), Btildes_rc(0)(1), Btildes_rc(1)(0),
             Btildes_rc(1)(1), Btildes_rc(2)(0), Btildes_rc(2)(1));
      printf("  lam = %16.8e, %16.8e, %16.8e, %16.8e,\n"
             "        %16.8e, %16.8e, %16.8e, %16.8e.\n",
             lambda(0)(0), lambda(0)(1), lambda(0)(2), lambda(0)(3),
             lambda(1)(0), lambda(1)(1), lambda(1)(2), lambda(1)(3));
      printf("  alp_avg = %16.8e, beta_avg = %16.8e, u_avg = %16.8e \n",
             alp_avg, beta_avg, u_avg);
      printf("  vel_rc  = %16.8e, %16.8e \n", vel_rc(0), vel_rc(1));
      printf("  rho_rc  = %16.8e, %16.8e \n", rho_rc(0), rho_rc(1));
      printf("  cs2_rc  = %16.8e, %16.8e \n", cs2_rc(0), cs2_rc(1));
      printf("  wlor_rc = %16.8e, %16.8e \n", w_lorentz_rc(0), w_lorentz_rc(1));
      printf("  h_rc    = %16.8e, %16.8e \n", h_rc(0), h_rc(1));
      printf("  bsq_rc  = %16.8e, %16.8e \n", bsq_rc(0), bsq_rc(1));
      printf("  press_rc = %16.8e, %16.8e \n", press_rc(0), press_rc(1));
      printf("  eps_rc   = %16.8e, %16.8e \n", eps_rc(0), eps_rc(1));
      printf("  rho = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e;\n",
             rho(p.I - p.DI[dir_i] * 3), rho(p.I - p.DI[dir_i] * 2),
             rho(p.I - p.DI[dir_i]), rho(p.I), rho(p.I + p.DI[dir_i]),
             rho(p.I + p.DI[dir_i] * 2));
      printf("  press = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e;\n",
             press(p.I - p.DI[dir_i] * 3), press(p.I - p.DI[dir_i] * 2),
             press(p.I - p.DI[dir_i]), press(p.I), press(p.I + p.DI[dir_i]),
             press(p.I + p.DI[dir_i] * 2));
      printf("  eps   = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e;\n",
             eps(p.I - p.DI[dir_i] * 3), eps(p.I - p.DI[dir_i] * 2),
             eps(p.I - p.DI[dir_i]), eps(p.I), eps(p.I + p.DI[dir_i]),
             eps(p.I + p.DI[dir_i] * 2));
      printf("  alp_avg, beta_avg = %16.8e, %16.8e, %16.8e, %16.8e,\n", alp_avg,
             betas_avg(0), betas_avg(1), betas_avg(2));
      printf("  g_avg = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e.\n",
             g_avg(0, 0), g_avg(0, 1), g_avg(0, 2), g_avg(1, 1), g_avg(1, 2),
             g_avg(2, 2));
      printf("  sqrtg = %16.8e,\n", sqrtg);
      printf("  vlows_rc  = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e.\n",
             vlows_rc(0)(0), vlows_rc(0)(1), vlows_rc(1)(0), vlows_rc(1)(1),
             vlows_rc(2)(0), vlows_rc(2)(1));
      printf("  vups_rc   = %16.8e, %16.8e, %16.8e, %16.8e, %16.8e, %16.8e.\n",
             vels_rc(0)(0), vels_rc(0)(1), vels_rc(1)(0), vels_rc(1)(1),
             vels_rc(2)(0), vels_rc(2)(1));
      printf("  vtilde_rc = %16.8e, %16.8e.\n", vtilde_rc(0), vtilde_rc(1));
      assert(0);
    }
#endif

    /* Begin code for upwindCT */

    CCTK_REAL ap, am;
    maxspeeds_from_lambdas(lambda, ap, am);

    ap_face(dir_i)(p.I) = ap;
    am_face(dir_i)(p.I) = am;

    const CCTK_REAL vjL = vtildes_rc(dir_j)(0);
    const CCTK_REAL vjR = vtildes_rc(dir_j)(1);
    const CCTK_REAL vkL = vtildes_rc(dir_k)(0);
    const CCTK_REAL vkR = vtildes_rc(dir_k)(1);
    const CCTK_REAL vj_face = avg_upwind(vjL, vjR, ap, am);
    const CCTK_REAL vk_face = avg_upwind(vkL, vkR, ap, am);

    vbar_j(dir_i)(p.I) = vj_face;
    vbar_k(dir_i)(p.I) = vk_face;

    /* End code for upwindCT */
  });
}

extern "C" void AsterX_Fluxes(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_AsterX_Fluxes;
  DECLARE_CCTK_PARAMETERS;

  eos_3param eos_3p_type;

  if (CCTK_EQUALS(evolution_eos, "IdealGas")) {
    eos_3p_type = eos_3param::IdealGas;
  } else if (CCTK_EQUALS(evolution_eos, "Hybrid")) {
    eos_3p_type = eos_3param::Hybrid;
  } else if (CCTK_EQUALS(evolution_eos, "Tabulated3d")) {
    eos_3p_type = eos_3param::Tabulated;
  } else {
    CCTK_ERROR("Unknown value for parameter \"evolution_eos\"");
  }

  rec_var_t rec_var;
  if (CCTK_EQUALS(recon_type, "v_vec")) {
    rec_var = rec_var_t::v_vec;
  } else if (CCTK_EQUALS(recon_type, "z_vec")) {
    rec_var = rec_var_t::z_vec;
  } else if (CCTK_EQUALS(recon_type, "s_vec")) {
    rec_var = rec_var_t::s_vec;
  } else {
    CCTK_ERROR("Unknown value for parameter \"recon_type\"");
  }

  // Primary reconstruction method
  reconstruction_t reconstruction;
  if (CCTK_EQUALS(reconstruction_method, "Godunov"))
    reconstruction = reconstruction_t::Godunov;
  else if (CCTK_EQUALS(reconstruction_method, "minmod"))
    reconstruction = reconstruction_t::minmod;
  else if (CCTK_EQUALS(reconstruction_method, "monocentral"))
    reconstruction = reconstruction_t::monocentral;
  else if (CCTK_EQUALS(reconstruction_method, "ppm"))
    reconstruction = reconstruction_t::ppm;
  else if (CCTK_EQUALS(reconstruction_method, "eppm"))
    reconstruction = reconstruction_t::eppm;
  else if (CCTK_EQUALS(reconstruction_method, "wenoz"))
    reconstruction = reconstruction_t::wenoz;
  else if (CCTK_EQUALS(reconstruction_method, "mp5"))
    reconstruction = reconstruction_t::mp5;
  else
    CCTK_ERROR("Unknown value for parameter \"reconstruction_method\"");

  // Lower-order fallback for negative values
  reconstruction_t reconstruction_LO;
  if (CCTK_EQUALS(loworder_method, "Godunov"))
    reconstruction_LO = reconstruction_t::Godunov;
  else if (CCTK_EQUALS(loworder_method, "minmod"))
    reconstruction_LO = reconstruction_t::minmod;
  else if (CCTK_EQUALS(loworder_method, "monocentral"))
    reconstruction_LO = reconstruction_t::monocentral;
  else
    CCTK_ERROR("Unknown value for parameter \"loworder_method\"");

  // reconstruction parameters struct
  reconstruct_params_t reconstruct_params;

  // ppm parameters
  reconstruct_params.ppm_shock_detection = ppm_shock_detection;
  reconstruct_params.ppm_zone_flattening = ppm_zone_flattening;
  reconstruct_params.poly_k = poly_k;
  reconstruct_params.poly_gamma = poly_gamma;
  reconstruct_params.ppm_eta1 = ppm_eta1;
  reconstruct_params.ppm_eta2 = ppm_eta2;
  reconstruct_params.ppm_eps = ppm_eps;
  reconstruct_params.ppm_eps_shock = ppm_eps_shock;
  reconstruct_params.ppm_small = ppm_small;
  reconstruct_params.ppm_omega1 = ppm_omega1;
  reconstruct_params.ppm_omega2 = ppm_omega2;
  reconstruct_params.enhanced_ppm_C2 = enhanced_ppm_C2;
  // wenoz parameters
  reconstruct_params.weno_eps = weno_eps;
  // mp5 parameters
  reconstruct_params.mp5_alpha = mp5_alpha;

  flux_t fluxtype;
  if (CCTK_EQUALS(flux_type, "LxF")) {
    fluxtype = flux_t::LxF;
  } else if (CCTK_EQUALS(flux_type, "HLLE")) {
    fluxtype = flux_t::HLLE;
  } else {
    CCTK_ERROR("Unknown value for parameter \"flux_type\"");
  }

  switch (eos_3p_type) {
  case eos_3param::IdealGas: {
    // Get local eos object
    auto eos_3p_ig = global_eos_3p_ig;

    CalcFlux<0>(cctkGH, eos_3p_ig, rec_var, reconstruction, reconstruction_LO,
                reconstruct_params, fluxtype);
    CalcFlux<1>(cctkGH, eos_3p_ig, rec_var, reconstruction, reconstruction_LO,
                reconstruct_params, fluxtype);
    CalcFlux<2>(cctkGH, eos_3p_ig, rec_var, reconstruction, reconstruction_LO,
                reconstruct_params, fluxtype);
    break;
  }
  case eos_3param::Hybrid: {
    // Get local eos object
    auto eos_3p_hyb = global_eos_3p_hyb;

    CalcFlux<0>(cctkGH, eos_3p_hyb, rec_var, reconstruction, reconstruction_LO,
                reconstruct_params, fluxtype);
    CalcFlux<1>(cctkGH, eos_3p_hyb, rec_var, reconstruction, reconstruction_LO,
                reconstruct_params, fluxtype);
    CalcFlux<2>(cctkGH, eos_3p_hyb, rec_var, reconstruction, reconstruction_LO,
                reconstruct_params, fluxtype);
    break;
  }
  case eos_3param::Tabulated: {
    // Get local eos object
    auto eos_3p_tab3d = global_eos_3p_tab3d;

    CalcFlux<0>(cctkGH, eos_3p_tab3d, rec_var, reconstruction,
                reconstruction_LO, reconstruct_params, fluxtype);
    CalcFlux<1>(cctkGH, eos_3p_tab3d, rec_var, reconstruction,
                reconstruction_LO, reconstruct_params, fluxtype);
    CalcFlux<2>(cctkGH, eos_3p_tab3d, rec_var, reconstruction,
                reconstruction_LO, reconstruct_params, fluxtype);
    break;
  }
  default:
    assert(0);
  }
}

template <int i, bool use_uct>
void CalcE_impl(CCTK_ARGUMENTS, const reconstruction_t reconstruction,
                const reconstruct_params_t reconstruct_params) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_CalcAuxTermsForAvecPsiRHS;

  // the other two directions
  constexpr int j = (i == 0) ? 1 : ((i == 1) ? 2 : 0);
  constexpr int k = (i == 0) ? 2 : ((i == 1) ? 0 : 1);

  // flux-CT
  const vec<vec<GF3D2<const CCTK_REAL>, dim>, dim> gf_fBs{
      {fxBx, fyBx, fzBx}, {fxBy, fyBy, fzBy}, {fxBz, fyBz, fzBz}};

  // upwind-CT
  const vec<GF3D2<const CCTK_REAL>, dim> gf_vels{velx, vely, velz};
  const vec<GF3D2<const CCTK_REAL>, dim> dB_stag{dBx_stag, dBy_stag, dBz_stag};
  const vec<GF3D2<const CCTK_REAL>, dim> ap_face{amax_xface, amax_yface,
                                                 amax_zface};
  const vec<GF3D2<const CCTK_REAL>, dim> am_face{amin_xface, amin_yface,
                                                 amin_zface};
  const vec<vec<GF3D2<const CCTK_REAL>, 2>, dim> vbars{
      {vbar_x_yface, vbar_x_zface},
      {vbar_y_zface, vbar_y_xface},
      {vbar_z_xface, vbar_z_yface}};
  // mapping from 3d to 2d, since we don't need iface
  constexpr int jface = 1;
  constexpr int kface = 0;

  const vec<GF3D2<CCTK_REAL>, dim> gf_E{Ex, Ey, Ez};

  // edge centered loop
  if constexpr (use_uct) { // upwind-CT
    grid.loop_int_device<i == 0, i == 1, i == 2>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          // reconstruct in k-dir
          const vec<CCTK_REAL, 2> dBstag_jface_krc{
              reconstruct(dB_stag(j), p, reconstruction, k, false, false, press,
                          gf_vels(k), reconstruct_params)};
          const vec<CCTK_REAL, 2> vbar_k_jface_krc{
              reconstruct(vbars(k)(jface), p, reconstruction, k, false, false,
                          press, gf_vels(k), reconstruct_params)};
          // reconstruct in j-dir
          const vec<CCTK_REAL, 2> dBstag_kface_jrc{
              reconstruct(dB_stag(k), p, reconstruction, j, false, false, press,
                          gf_vels(j), reconstruct_params)};
          const vec<CCTK_REAL, 2> vbar_j_kface_jrc{
              reconstruct(vbars(j)(kface), p, reconstruction, j, false, false,
                          press, gf_vels(j), reconstruct_params)};

          const CCTK_REAL BjL = dBstag_jface_krc(0);
          const CCTK_REAL BjR = dBstag_jface_krc(1);
          const CCTK_REAL vkL = vbar_k_jface_krc(0);
          const CCTK_REAL vkR = vbar_k_jface_krc(1);

          const CCTK_REAL BkL = dBstag_kface_jrc(0);
          const CCTK_REAL BkR = dBstag_kface_jrc(1);
          const CCTK_REAL vjL = vbar_j_kface_jrc(0);
          const CCTK_REAL vjR = vbar_j_kface_jrc(1);

          const CCTK_REAL ap_k = ap_face(k)(p.I);
          const CCTK_REAL am_k = am_face(k)(p.I);
          const CCTK_REAL ap_j = ap_face(j)(p.I);
          const CCTK_REAL am_j = am_face(j)(p.I);

          gf_E(i)(p.I) =
              hll_upwind(BjL, BjR, vkL * BjL, vkR * BjR, ap_k, am_k) -
              hll_upwind(BkL, BkR, vjL * BkL, vjR * BkR, ap_j, am_j);
        });
  } else { // flux-CT
    grid.loop_int_device<i == 0, i == 1, i == 2>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const CCTK_REAL Fjk = gf_fBs(j)(k)(p.I);
          const CCTK_REAL Fjk_m = gf_fBs(j)(k)(p.I - p.DI[j]);
          const CCTK_REAL Fkj = gf_fBs(k)(j)(p.I);
          const CCTK_REAL Fkj_m = gf_fBs(k)(j)(p.I - p.DI[k]);
          gf_E(i)(p.I) = CCTK_REAL(0.25) * ((Fjk + Fjk_m) - (Fkj + Fkj_m));
        });
  }
}

template <int i>
void CalcE(CCTK_ARGUMENTS, const bool use_uct,
           const reconstruction_t reconstruction,
           const reconstruct_params_t reconstruct_params) {
  if (use_uct) {
    CalcE_impl<i, true>(CCTK_PASS_CTOC, reconstruction, reconstruct_params);
  } else {
    CalcE_impl<i, false>(CCTK_PASS_CTOC, reconstruction, reconstruct_params);
  }
}

template <int i> void CalcFstag(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_CalcAuxTermsForAvecPsiRHS;
  DECLARE_CCTK_PARAMETERS;

  // the other two directions
  constexpr int j = (i == 0) ? 1 : ((i == 1) ? 2 : 0);
  constexpr int k = (i == 0) ? 2 : ((i == 1) ? 0 : 1);

  const vec<GF3D2<CCTK_REAL>, dim> gf_Fstag{Fx_stag, Fy_stag, Fz_stag};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_Avecs{Avec_x, Avec_y, Avec_z};
  const smat<GF3D2<const CCTK_REAL>, dim> gf_g{gxx, gxy, gxz, gyy, gyz, gzz};

  grid.loop_mix_device<i == 0, i == 1, i == 2>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        const CCTK_REAL alp_e = calc_avg_v2e<i>(alp, p);
        const smat<CCTK_REAL, 3> g_e([&](int m, int n) ARITH_INLINE {
          return calc_avg_v2e<i>(gf_g(m, n), p);
        });
        const CCTK_REAL detg_e = calc_det(g_e);
        const CCTK_REAL sqrtg_e = sqrt(detg_e);
        const smat<CCTK_REAL, 3> ug_e = calc_inv(g_e, detg_e);

        vec<CCTK_REAL, 3> A_e;
        A_e(i) = gf_Avecs(i)(p.I);
        A_e(j) = calc_avg_e2e<i, j>(gf_Avecs(j), p);
        A_e(k) = calc_avg_e2e<i, k>(gf_Avecs(k), p);

        const vec<CCTK_REAL, 3> Aup_e = calc_contraction(ug_e, A_e);

        gf_Fstag(i)(p.I) = alp_e * sqrtg_e * Aup_e(i);
      });
}

extern "C" void AsterX_CalcAuxTermsForAvecPsiRHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_CalcAuxTermsForAvecPsiRHS;
  DECLARE_CCTK_PARAMETERS;

  reconstruction_t reconstruction;
  if (CCTK_EQUALS(reconstruction_method, "Godunov"))
    reconstruction = reconstruction_t::Godunov;
  else if (CCTK_EQUALS(reconstruction_method, "minmod"))
    reconstruction = reconstruction_t::minmod;
  else if (CCTK_EQUALS(reconstruction_method, "monocentral"))
    reconstruction = reconstruction_t::monocentral;
  else if (CCTK_EQUALS(reconstruction_method, "ppm"))
    reconstruction = reconstruction_t::ppm;
  else if (CCTK_EQUALS(reconstruction_method, "eppm"))
    reconstruction = reconstruction_t::eppm;
  else if (CCTK_EQUALS(reconstruction_method, "wenoz"))
    reconstruction = reconstruction_t::wenoz;
  else if (CCTK_EQUALS(reconstruction_method, "mp5"))
    reconstruction = reconstruction_t::mp5;
  else
    CCTK_ERROR("Unknown value for parameter \"reconstruction_method\"");

  // reconstruction parameters struct
  reconstruct_params_t reconstruct_params;

  // ppm parameters
  reconstruct_params.ppm_shock_detection = ppm_shock_detection;
  reconstruct_params.ppm_zone_flattening = ppm_zone_flattening;
  reconstruct_params.poly_k = poly_k;
  reconstruct_params.poly_gamma = poly_gamma;
  reconstruct_params.ppm_eta1 = ppm_eta1;
  reconstruct_params.ppm_eta2 = ppm_eta2;
  reconstruct_params.ppm_eps = ppm_eps;
  reconstruct_params.ppm_eps_shock = ppm_eps_shock;
  reconstruct_params.ppm_small = ppm_small;
  reconstruct_params.ppm_omega1 = ppm_omega1;
  reconstruct_params.ppm_omega2 = ppm_omega2;
  reconstruct_params.enhanced_ppm_C2 = enhanced_ppm_C2;
  // wenoz parameters
  reconstruct_params.weno_eps = weno_eps;
  // mp5 parameters
  reconstruct_params.mp5_alpha = mp5_alpha;

  const vec<GF3D2<const CCTK_REAL>, dim> gf_Avecs{Avec_x, Avec_y, Avec_z};
  const smat<GF3D2<const CCTK_REAL>, dim> gf_g{gxx, gxy, gxz, gyy, gyz, gzz};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_beta{betax, betay, betaz};

  CalcE<0>(CCTK_PASS_CTOC, use_uct, reconstruction, reconstruct_params);
  CalcE<1>(CCTK_PASS_CTOC, use_uct, reconstruction, reconstruct_params);
  CalcE<2>(CCTK_PASS_CTOC, use_uct, reconstruction, reconstruct_params);

  if (CCTK_EQUALS(interp_method_Avert, "average")) {
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const vec<CCTK_REAL, 3> A_vert([&](int i) ARITH_INLINE {
            return calc_avg_e2v(gf_Avecs(i), p, i);
          });
          const smat<CCTK_REAL, 3> g(
              [&](int i, int j) ARITH_INLINE { return gf_g(i, j)(p.I); });
          const vec<CCTK_REAL, 3> betas(
              [&](int i) ARITH_INLINE { return gf_beta(i)(p.I); });
          const CCTK_REAL detg = calc_det(g);
          const CCTK_REAL sqrtg = sqrt(detg);

          G(p.I) =
              alp(p.I) * Psi(p.I) / sqrtg - calc_contraction(betas, A_vert);
        });
  } else if (CCTK_EQUALS(interp_method_Avert, "hermite")) {
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const vec<CCTK_REAL, 3> A_vert([&](int i) ARITH_INLINE {
            return calc_avg_e2v_hermite(gf_Avecs(i), p, i);
          });
          const smat<CCTK_REAL, 3> g(
              [&](int i, int j) ARITH_INLINE { return gf_g(i, j)(p.I); });
          const vec<CCTK_REAL, 3> betas(
              [&](int i) ARITH_INLINE { return gf_beta(i)(p.I); });
          const CCTK_REAL detg = calc_det(g);
          const CCTK_REAL sqrtg = sqrt(detg);

          G(p.I) =
              alp(p.I) * Psi(p.I) / sqrtg - calc_contraction(betas, A_vert);
        });
  } else {
    CCTK_ERROR("Unknown value for parameter \"interp_method_Avert\"");
  }

  grid.loop_all_device<0, 0, 0>(grid.nghostzones,
                                [=] CCTK_DEVICE(const PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      Fbetax(p.I) = betax(p.I) * Psi(p.I);
                                      Fbetay(p.I) = betay(p.I) * Psi(p.I);
                                      Fbetaz(p.I) = betaz(p.I) * Psi(p.I);
                                    });

  CalcFstag<0>(CCTK_PASS_CTOC);
  CalcFstag<1>(CCTK_PASS_CTOC);
  CalcFstag<2>(CCTK_PASS_CTOC);
}

} // namespace AsterX
