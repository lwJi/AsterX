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

template <int dir>
void CalcRHSofAvec(CCTK_ARGUMENTS, const vector_potential_gauge_t gauge) {
  DECLARE_CCTK_ARGUMENTSX_AsterX_RHS;
  DECLARE_CCTK_PARAMETERS;

  // flux-CT
  const vec<vec<GF3D2<const CCTK_REAL>, dim>, dim> gf_fBs{
      {fxBx, fyBx, fzBx}, {fxBy, fyBy, fzBy}, {fxBz, fyBz, fzBz}};

  // upwind-CT
  const vec<GF3D2<const CCTK_REAL>, dim> gf_vels{velx, vely, velz};
  const vec<GF3D2<const CCTK_REAL>, dim> dBstag_one{dBy_stag, dBz_stag,
                                                    dBx_stag};
  const vec<GF3D2<const CCTK_REAL>, dim> dBstag_two{dBz_stag, dBx_stag,
                                                    dBy_stag};

  const vec<GF3D2<const CCTK_REAL>, dim> vtildes_one{
      vtilde_z_yface, vtilde_x_zface, vtilde_y_xface};
  const vec<GF3D2<const CCTK_REAL>, dim> vtildes_two{
      vtilde_y_zface, vtilde_z_xface, vtilde_x_yface};

  const vec<GF3D2<const CCTK_REAL>, dim> amax_one{amax_zface, amax_xface,
                                                  amax_yface};
  const vec<GF3D2<const CCTK_REAL>, dim> amax_two{amax_yface, amax_zface,
                                                  amax_xface};

  const vec<GF3D2<const CCTK_REAL>, dim> amin_one{amin_zface, amin_xface,
                                                  amin_yface};
  const vec<GF3D2<const CCTK_REAL>, dim> amin_two{amin_yface, amin_zface,
                                                  amin_xface};

  const vec<GF3D2<CCTK_REAL>, dim> gf_Avec_rhs{Avec_x_rhs, Avec_y_rhs,
                                               Avec_z_rhs};

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

  constexpr array<int, dim> edge_centred = {(dir == 0), (dir == 1), (dir == 2)};

  constexpr int j = (dir == 0) ? 1 : ((dir == 1) ? 2 : 0);
  constexpr int k = (dir == 0) ? 2 : ((dir == 1) ? 0 : 1);

  grid.loop_int_device<edge_centred[0], edge_centred[1], edge_centred[2]>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        CCTK_REAL E;

        if (use_uct) { // upwind-CT

          const vec<vec<CCTK_REAL, 2>, 3> dBstag_one_rc(
              [&](int m) ARITH_INLINE {
                return vec<CCTK_REAL, 2>{
                    reconstruct(dBstag_one(m), p, reconstruction, k, false,
                                false, press, gf_vels(k), reconstruct_params)};
              });

          const vec<vec<CCTK_REAL, 2>, 3> dBstag_two_rc(
              [&](int m) ARITH_INLINE {
                return vec<CCTK_REAL, 2>{
                    reconstruct(dBstag_two(m), p, reconstruction, j, false,
                                false, press, gf_vels(j), reconstruct_params)};
              });

          const vec<vec<CCTK_REAL, 2>, 3> vtildes_one_rc(
              [&](int m) ARITH_INLINE {
                return vec<CCTK_REAL, 2>{
                    reconstruct(vtildes_one(m), p, reconstruction, k, false,
                                false, press, gf_vels(k), reconstruct_params)};
              });

          const vec<vec<CCTK_REAL, 2>, 3> vtildes_two_rc(
              [&](int m) ARITH_INLINE {
                return vec<CCTK_REAL, 2>{
                    reconstruct(vtildes_two(m), p, reconstruction, j, false,
                                false, press, gf_vels(j), reconstruct_params)};
              });

          // first term
          CCTK_REAL denom_one = amax_one(dir)(p.I) + amin_one(dir)(p.I);
          E = (amax_one(dir)(p.I) * vtildes_one_rc(dir)(0) *
                   dBstag_one_rc(dir)(0) +
               amin_one(dir)(p.I) * vtildes_one_rc(dir)(1) *
                   dBstag_one_rc(dir)(1) -
               amax_one(dir)(p.I) * amin_one(dir)(p.I) *
                   (dBstag_one_rc(dir)(1) - dBstag_one_rc(dir)(0))) /
              denom_one;

          // second term
          CCTK_REAL denom_two = amax_two(dir)(p.I) + amin_two(dir)(p.I);
          E -= (amax_two(dir)(p.I) * vtildes_two_rc(dir)(0) *
                    dBstag_two_rc(dir)(0) +
                amin_two(dir)(p.I) * vtildes_two_rc(dir)(1) *
                    dBstag_two_rc(dir)(1) -
                amax_two(dir)(p.I) * amin_two(dir)(p.I) *
                    (dBstag_two_rc(dir)(1) - dBstag_two_rc(dir)(0))) /
               denom_two;

        } else { // flux-CT

          E = 0.25 * ((gf_fBs(j)(k)(p.I) + gf_fBs(j)(k)(p.I - p.DI[j])) -
                      (gf_fBs(k)(j)(p.I) + gf_fBs(k)(j)(p.I - p.DI[k])));
        }

        switch (gauge) {
        case vector_potential_gauge_t::algebraic: {
          gf_Avec_rhs(dir)(p.I) = -E;
          break;
        }

        case vector_potential_gauge_t::generalized_lorentz: {
          gf_Avec_rhs(dir)(p.I) = -E - calc_fd2_v2e<dir>(G, p);
          break;
        }

        default:
          assert(0);
        }
      });
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

  const vec<GF3D2<const CCTK_REAL>, dim> gf_Fstag{Fx_stag, Fy_stag, Fz_stag};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_beta{betax, betay, betaz};
  const vec<GF3D2<const CCTK_REAL>, dim> gf_Fbeta{Fbetax, Fbetay, Fbetaz};

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

  CalcRHSofAvec<0>(CCTK_PASS_CTOC, gauge);
  CalcRHSofAvec<1>(CCTK_PASS_CTOC, gauge);
  CalcRHSofAvec<2>(CCTK_PASS_CTOC, gauge);

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        switch (gauge) {
        case vector_potential_gauge_t::algebraic: {
          Psi_rhs(p.I) = 0.0;
          break;
        }

        case vector_potential_gauge_t::generalized_lorentz: {
          CCTK_REAL dF = 0.0;
          for (int i = 0; i < dim; i++) {
            dF += calc_fd2_e2v(gf_Fstag(i), p, i) -
                  (gf_beta(i)(p.I) < 0
                       ? calc_fd2_v2v_oneside<-1>(gf_Fbeta(i), p, i)
                       : calc_fd2_v2v_oneside<+1>(gf_Fbeta(i), p, i));
          }
          Psi_rhs(p.I) = -dF - lorenz_damp_fac * alp(p.I) * Psi(p.I);
          break;
        }

        default:
          assert(0);
        }
      });
}

} // namespace AsterX
