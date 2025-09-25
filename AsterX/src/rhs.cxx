#include <array>
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include "aster_utils.hxx"
#include "reconstruct.hxx"

namespace AsterX {
using namespace Loop;
using namespace Arith;
using namespace ReconX;
using namespace AsterUtils;

enum class vector_potential_gauge_t { algebraic, generalized_lorentz };

template <typename T>
CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE static inline T
hll_upwind(T uL, T uR, T fL, T fR, T ap, T am) {
  const T s = ap + am;
  if (s <= T(1e-14)) {
    return T(0.5) * (fL + fR);
  }
  return (ap * fL + am * fR - ap * am * (uR - uL)) / s;
}

template <int i, vector_potential_gauge_t gauge, bool use_uct>
void CalcRHSofAvec_impl(CCTK_ARGUMENTS, const reconstruction_t reconstruction,
                        const reconstruct_params_t reconstruct_params) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_RHS;

  // the other two directions
  constexpr int j = (i == 0) ? 1 : ((i == 1) ? 2 : 0);
  constexpr int k = (i == 0) ? 2 : ((i == 1) ? 0 : 1);

  constexpr array<int, dim> edge_centred = {(i == 0), (i == 1), (i == 2)};

  // flux-CT
  const vec<vec<GF3D2<const CCTK_REAL>, dim>, dim> gf_fBs{
      {fxBx, fyBx, fzBx}, {fxBy, fyBy, fzBy}, {fxBz, fyBz, fzBz}};

  // upwind-CT
  const vec<GF3D2<const CCTK_REAL>, dim> gf_vels{velx, vely, velz};
  const vec<GF3D2<const CCTK_REAL>, dim> dBstag{dBx_stag, dBy_stag, dBz_stag};
  // j-comp reconstructed in k-dir
  const vec<GF3D2<const CCTK_REAL>, dim> vbars_j_reck{
      vbar_y_zface, vbar_z_xface, vbar_x_yface};
  // k-comp reconstructed in j-dir
  const vec<GF3D2<const CCTK_REAL>, dim> vbars_k_recj{
      vbar_z_yface, vbar_x_zface, vbar_y_xface};

  const vec<GF3D2<const CCTK_REAL>, dim> amax_reck{amax_zface, amax_xface,
                                                   amax_yface};
  const vec<GF3D2<const CCTK_REAL>, dim> amax_recj{amax_yface, amax_zface,
                                                   amax_xface};
  const vec<GF3D2<const CCTK_REAL>, dim> amin_reck{amin_zface, amin_xface,
                                                   amin_yface};
  const vec<GF3D2<const CCTK_REAL>, dim> amin_recj{amin_yface, amin_zface,
                                                   amin_xface};

  const vec<GF3D2<CCTK_REAL>, dim> gf_Avec_rhs{Avec_x_rhs, Avec_y_rhs,
                                               Avec_z_rhs};

  grid.loop_int_device<edge_centred[0], edge_centred[1], edge_centred[2]>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        CCTK_REAL E;

        if constexpr (use_uct) { // upwind-CT
          // reconstruct in k-dir
          const vec<CCTK_REAL, 2> dBstag_j_reck_rc{
              reconstruct(dBstag(j), p, reconstruction, k, false, false, press,
                          gf_vels(k), reconstruct_params)};
          const vec<CCTK_REAL, 2> vbars_k_recj_reck_rc{
              reconstruct(vbars_k_recj(i), p, reconstruction, k, false, false,
                          press, gf_vels(k), reconstruct_params)};
          // reconstruct in j-dir
          const vec<CCTK_REAL, 2> dBstag_k_recj_rc{
              reconstruct(dBstag(k), p, reconstruction, j, false, false, press,
                          gf_vels(j), reconstruct_params)};
          const vec<CCTK_REAL, 2> vbars_j_reck_recj_rc{
              reconstruct(vbars_j_reck(i), p, reconstruction, j, false, false,
                          press, gf_vels(j), reconstruct_params)};

          const CCTK_REAL ap_k = amax_reck(i)(p.I);
          const CCTK_REAL am_k = amin_reck(i)(p.I);
          const CCTK_REAL ap_j = amax_recj(i)(p.I);
          const CCTK_REAL am_j = amin_recj(i)(p.I);

          const CCTK_REAL BjL = dBstag_j_reck_rc(0);
          const CCTK_REAL BjR = dBstag_j_reck_rc(1);
          const CCTK_REAL vkL = vbars_k_recj_reck_rc(0);
          const CCTK_REAL vkR = vbars_k_recj_reck_rc(1);

          const CCTK_REAL BkL = dBstag_k_recj_rc(0);
          const CCTK_REAL BkR = dBstag_k_recj_rc(1);
          const CCTK_REAL vjL = vbars_j_reck_recj_rc(0);
          const CCTK_REAL vjR = vbars_j_reck_recj_rc(1);

          E = hll_upwind(BjL, BjR, vkL * BjL, vkR * BjR, ap_k, am_k) -
              hll_upwind(BkL, BkR, vjL * BkL, vjR * BkR, ap_j, am_j);
        } else { // flux-CT
          E = 0.25 * ((gf_fBs(j)(k)(p.I) + gf_fBs(j)(k)(p.I - p.DI[j])) -
                      (gf_fBs(k)(j)(p.I) + gf_fBs(k)(j)(p.I - p.DI[k])));
        }

        if constexpr (gauge == vector_potential_gauge_t::algebraic) {
          gf_Avec_rhs(i)(p.I) = -E;
        } else if constexpr (gauge ==
                             vector_potential_gauge_t::generalized_lorentz) {
          gf_Avec_rhs(i)(p.I) = -E - calc_fd2_v2e<i>(G, p);
        }
      });
}

template <vector_potential_gauge_t gauge>
void CalcRHSofPsi_impl(CCTK_ARGUMENTS, const CCTK_REAL damp_fac) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_RHS;

  const vec<GF3D2<const CCTK_REAL>, dim> gf_Fstag{Fx_stag, Fy_stag, Fz_stag};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_beta{betax, betay, betaz};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_Fbeta{Fbetax, Fbetay, Fbetaz};

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        if constexpr (gauge == vector_potential_gauge_t::algebraic) {
          Psi_rhs(p.I) = 0.0;
        } else if constexpr (gauge ==
                             vector_potential_gauge_t::generalized_lorentz) {
          CCTK_REAL dF = 0.0;
          for (int i = 0; i < dim; i++) {
            dF += calc_fd2_e2v(gf_Fstag(i), p, i) -
                  (gf_beta(i)(p.I) < 0
                       ? calc_fd2_v2v_oneside<-1>(gf_Fbeta(i), p, i)
                       : calc_fd2_v2v_oneside<+1>(gf_Fbeta(i), p, i));
          }
          Psi_rhs(p.I) = -dF - damp_fac * alp(p.I) * Psi(p.I);
        }
      });
}

template <int i>
void CalcRHSofAvec(CCTK_ARGUMENTS, const vector_potential_gauge_t gauge,
                   const bool use_uct, const reconstruction_t reconstruction,
                   const reconstruct_params_t reconstruct_params) {
  switch (gauge) {
  case vector_potential_gauge_t::algebraic: {
    if (use_uct) {
      CalcRHSofAvec_impl<i, vector_potential_gauge_t::algebraic, true>(
          CCTK_PASS_CTOC, reconstruction, reconstruct_params);
    } else {
      CalcRHSofAvec_impl<i, vector_potential_gauge_t::algebraic, false>(
          CCTK_PASS_CTOC, reconstruction, reconstruct_params);
    }
    break;
  }
  case vector_potential_gauge_t::generalized_lorentz: {
    if (use_uct) {
      CalcRHSofAvec_impl<i, vector_potential_gauge_t::generalized_lorentz,
                         true>(CCTK_PASS_CTOC, reconstruction,
                               reconstruct_params);
    } else {
      CalcRHSofAvec_impl<i, vector_potential_gauge_t::generalized_lorentz,
                         false>(CCTK_PASS_CTOC, reconstruction,
                                reconstruct_params);
    }
    break;
  }
  default:
    assert(0);
  }
}

void CalcRHSofPsi(CCTK_ARGUMENTS, const vector_potential_gauge_t gauge,
                  const CCTK_REAL damp_fac) {
  switch (gauge) {
  case vector_potential_gauge_t::algebraic: {
    CalcRHSofPsi_impl<vector_potential_gauge_t::algebraic>(CCTK_PASS_CTOC,
                                                           damp_fac);
    break;
  }
  case vector_potential_gauge_t::generalized_lorentz: {
    CalcRHSofPsi_impl<vector_potential_gauge_t::generalized_lorentz>(
        CCTK_PASS_CTOC, damp_fac);
  } break;
  default:
    assert(0);
  }
}

extern "C" void AsterX_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_RHS;
  DECLARE_CCTK_PARAMETERS;

  vector_potential_gauge_t gauge;
  if (CCTK_EQUALS(vector_potential_gauge, "algebraic"))
    gauge = vector_potential_gauge_t::algebraic;
  else if (CCTK_EQUALS(vector_potential_gauge, "generalized Lorentz"))
    gauge = vector_potential_gauge_t::generalized_lorentz;
  else
    CCTK_ERROR("Unknown value for parameter \"vector_potential_gauge\"");

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

  const vec<CCTK_REAL, dim> idx{1 / CCTK_DELTA_SPACE(0),
                                1 / CCTK_DELTA_SPACE(1),
                                1 / CCTK_DELTA_SPACE(2)};

  const vec<GF3D2<const CCTK_REAL>, dim> gf_fdens{fxdens, fydens, fzdens};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_fDEnt{fxDEnt, fyDEnt, fzDEnt};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_fmomx{fxmomx, fymomx, fzmomx};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_fmomy{fxmomy, fymomy, fzmomy};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_fmomz{fxmomz, fymomz, fzmomz};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_ftau{fxtau, fytau, fztau};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_fDYe{fxDYe, fyDYe, fzDYe};

  const auto calcupdate_hydro =
      [=] CCTK_DEVICE(const vec<GF3D2<const CCTK_REAL>, dim> &gf_fluxes,
                      const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        vec<CCTK_REAL, 3> dfluxes([&](int i) ARITH_INLINE {
          return gf_fluxes(i)(p.I + p.DI[i]) - gf_fluxes(i)(p.I);
        });
        return -calc_contraction(idx, dfluxes);
      };

  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        densrhs(p.I) += calcupdate_hydro(gf_fdens, p);
        DEntrhs(p.I) += calcupdate_hydro(gf_fDEnt, p);
        momxrhs(p.I) += calcupdate_hydro(gf_fmomx, p);
        momyrhs(p.I) += calcupdate_hydro(gf_fmomy, p);
        momzrhs(p.I) += calcupdate_hydro(gf_fmomz, p);
        taurhs(p.I) += calcupdate_hydro(gf_ftau, p);
        DYe_rhs(p.I) += calcupdate_hydro(gf_fDYe, p);

#ifdef CCTK_DEBUG
        if (isnan(densrhs(p.I))) {
          printf("calcupdate = %f, ", calcupdate_hydro(gf_fdens, p));
          printf("densrhs = %f, gf_fdens = %f, %f, %f, %f, %f, %f \n",
                 densrhs(p.I), gf_fdens(0)(p.I), gf_fdens(1)(p.I),
                 gf_fdens(2)(p.I), gf_fdens(0)(p.I + p.DI[0]),
                 gf_fdens(1)(p.I + p.DI[1]), gf_fdens(2)(p.I + p.DI[2]));
        }
        assert(!isnan(densrhs(p.I)));
#endif
      });

  CalcRHSofAvec<0>(CCTK_PASS_CTOC, gauge, use_uct, reconstruction,
                   reconstruct_params);
  CalcRHSofAvec<1>(CCTK_PASS_CTOC, gauge, use_uct, reconstruction,
                   reconstruct_params);
  CalcRHSofAvec<2>(CCTK_PASS_CTOC, gauge, use_uct, reconstruction,
                   reconstruct_params);

  CalcRHSofPsi(CCTK_PASS_CTOC, gauge, lorenz_damp_fac);
}

} // namespace AsterX
