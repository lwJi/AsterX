/*! \file c2p.hxx
\brief Defines a c2p
\author Jay Kalinani

c2p is effectively an interface to be used by different c2p implementations.

*/

#ifndef C2P_HXX
#define C2P_HXX

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <math.h>

#include "atmo.hxx"
#include "c2p_report.hxx"
#include "c2p_utils.hxx"
#include "cons.hxx"
#include "prims.hxx"
#include "setup_eos.hxx"

namespace Con2PrimFactory {

using namespace AsterUtils;

constexpr CCTK_INT X = 0;
constexpr CCTK_INT Y = 1;
constexpr CCTK_INT Z = 2;

/* Abstract class c2p */
class c2p {
protected:
  /* The constructor must initialize the following variables */

  atmosphere atmo;
  CCTK_INT maxIterations;
  CCTK_REAL tolerance;
  CCTK_REAL alp_thresh;
  CCTK_REAL cons_error;
  CCTK_REAL vw_lim;
  CCTK_REAL w_lim;
  CCTK_REAL v_lim;
  CCTK_REAL Bsq_lim;
  CCTK_REAL rho_BH;
  CCTK_REAL eps_BH;
  CCTK_REAL vwlim_BH;
  bool ye_lenient;
  bool use_zprim;
  bool use_temp;
  bool use_press_atmo;

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  get_Ssq_Exact(const vec<CCTK_REAL, 3> &mom,
                const smat<CCTK_REAL, 3> &gup) const;
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  get_Bsq_Exact(const vec<CCTK_REAL, 3> &B_up,
                const smat<CCTK_REAL, 3> &glo) const;
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  get_BiSi_Exact(const vec<CCTK_REAL, 3> &Bvec,
                 const vec<CCTK_REAL, 3> &mom) const;
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline vec<CCTK_REAL, 2>
  get_WLorentz_bsq_Seeds(const vec<CCTK_REAL, 3> &B_up,
                         const vec<CCTK_REAL, 3> &v_up,
                         const smat<CCTK_REAL, 3> &glo) const;

  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  prims_floors_and_ceilings(const EOSType *eos_3p, prim_vars &pv,
                            const cons_vars &cv, const smat<CCTK_REAL, 3> &glo,
                            c2p_report &rep) const;

public:
  template <typename EOSType, bool limiting>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  bh_interior(const EOSType *eos_3p, prim_vars &pv, cons_vars &cv,
              const smat<CCTK_REAL, 3> &glo) const;

  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  cons_floors_and_ceilings(const EOSType *eos_3p, cons_vars &cv,
                           const smat<CCTK_REAL, 3> &glo,
                           const CCTK_REAL &tauFluid_atm) const;
};

template <typename EOSType>
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
c2p::prims_floors_and_ceilings(const EOSType *eos_3p, prim_vars &pv,
                               const cons_vars &cv,
                               const smat<CCTK_REAL, 3> &glo,
                               c2p_report &rep) const {

  bool recomp_eps_press_entropy = false;

  // ----------
  // Floor and ceiling for Ye
  // ----------

  if (pv.Ye < eos_3p->rgye.min) {

    pv.Ye = eos_3p->rgye.min;
    rep.adjust_cons = true;
    recomp_eps_press_entropy = true;
  }

  if (pv.Ye > eos_3p->rgye.max) {

    pv.Ye = eos_3p->rgye.max;
    rep.adjust_cons = true;
    recomp_eps_press_entropy = true;
  }

  // ----------
  // Floor and ceiling for rho and velocity
  // ----------

  // check if computed velocities are within the specified limit
  const vec<CCTK_REAL, 3> v_low = calc_contraction(glo, pv.vel);
  CCTK_REAL vsq_Sol = calc_contraction(v_low, pv.vel);
  CCTK_REAL sol_v = sqrt(vsq_Sol);

  if (sol_v > v_lim) {
    // add mass, keeps conserved density D
    pv.rho = cv.dens / w_lim;
    pv.vel *= v_lim / sol_v;
    pv.w_lor = w_lim;
    rep.adjust_cons = true;

    if (use_temp) {
      // changes pressure
      recomp_eps_press_entropy = true;
    } else {
      // keeps pressure, changes eps
      recomp_eps_press_entropy = false;
      pv.eps = eos_3p->eps_from_valid_rho_press_ye(pv.rho, pv.press, pv.Ye);
      pv.temperature =
          eos_3p->temp_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
      pv.entropy = eos_3p->kappa_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
    }
  }

  if (pv.rho > eos_3p->rgrho.max) {
    // remove mass, changes conserved density D
    pv.rho = eos_3p->rgrho.max;
    rep.adjust_cons = true;

    if (use_temp) {
      // changes pressure
      recomp_eps_press_entropy = true;
    } else {
      // keeps pressure, changes eps
      recomp_eps_press_entropy = false;
      pv.eps = eos_3p->eps_from_valid_rho_press_ye(pv.rho, pv.press, pv.Ye);
      pv.temperature =
          eos_3p->temp_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
      pv.entropy = eos_3p->kappa_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
    }
  }

  // ----------
  // Ceiling for temperature
  // Keeps rho the same and changes press
  // ----------

  if (pv.temperature > eos_3p->rgtemp.max) {

    pv.temperature = eos_3p->rgtemp.max;
    recomp_eps_press_entropy = true;
    rep.adjust_cons = true;
  }

  // ----------
  // Floors
  // ----------

  if (use_press_atmo) {

    // ----------
    // Pressure floor
    // Keeps rho the same and changes temperature
    // ----------

    if (pv.press < atmo.press_atmo) {

      pv.press = atmo.press_atmo;
      pv.eps = eos_3p->eps_from_valid_rho_press_ye(pv.rho, pv.press, pv.Ye);
      pv.temperature =
          eos_3p->temp_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
      pv.entropy = eos_3p->kappa_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
      recomp_eps_press_entropy = false;
      rep.adjust_cons = true;
    }

  } else {

    // ----------
    // Temperature floor
    // Keeps rho the same and changes press
    // ----------

    if (pv.temperature < atmo.temp_atmo) {

      pv.temperature = atmo.temp_atmo;
      recomp_eps_press_entropy = true;
      rep.adjust_cons = true;
    }
  }

  if (recomp_eps_press_entropy) {
    pv.eps = eos_3p->eps_from_valid_rho_temp_ye(pv.rho, pv.temperature, pv.Ye);
    pv.press =
        eos_3p->press_from_valid_rho_temp_ye(pv.rho, pv.temperature, pv.Ye);
    pv.entropy = eos_3p->kappa_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
  }
}

template <typename EOSType, bool limiting>
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
c2p::bh_interior(const EOSType *eos_3p, prim_vars &pv, cons_vars &cv,
                 const smat<CCTK_REAL, 3> &glo) const {

  // Treatment for BH interiors after C2P failures
  // NOTE: By default, alp_thresh=0 so the if condition below is never
  // triggered. One must be very careful when using this functionality and
  // must correctly set alp_thresh, rho_BH, eps_BH and vwlim_BH in the
  // parfile

  const CCTK_REAL wlim_BH = sqrt(1.0 + vwlim_BH * vwlim_BH);
  const CCTK_REAL vlim_BH = vwlim_BH / wlim_BH;

  bool recomp_flag = false;

  if constexpr (limiting) {

    if (pv.rho > rho_BH) {
      pv.rho = rho_BH; // typically set to 0.01% to 1% of rho_max of initial
                       // NS or disk
      recomp_flag = true;
    };

    if (pv.eps > eps_BH) {
      pv.eps = eps_BH;
      recomp_flag = true;
    };

    const CCTK_REAL sol_v = sqrt((pv.w_lor * pv.w_lor - 1.0)) / pv.w_lor;
    if (sol_v > vlim_BH) {
      pv.vel *= vlim_BH / sol_v;
      pv.w_lor = wlim_BH;
      recomp_flag = true;
    };

    if (recomp_flag) {

      pv.temperature =
          eos_3p->temp_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
      pv.press = eos_3p->press_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
      pv.entropy = eos_3p->kappa_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);

      cv.from_prim(pv, glo);
    };

  } else {

    pv.rho = rho_BH; // typically set to 0.01% to 1% of rho_max of initial
                     // NS or disk
    pv.eps = eps_BH;
    pv.Ye = atmo.ye_atmo;

    pv.temperature = eos_3p->temp_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
    pv.press = eos_3p->press_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);
    pv.entropy = eos_3p->kappa_from_valid_rho_eps_ye(pv.rho, pv.eps, pv.Ye);

    // Set velocity such that new conserved momentum has same
    // direction as before

    // Inverse metric
    const CCTK_REAL spatial_detg = calc_det(glo);
    const smat<CCTK_REAL, 3> gup = calc_inv(glo, spatial_detg);

    // Compute Z = rho * h * W * W
    const CCTK_REAL Z_loc =
        (pv.rho * (1.0 + pv.eps) + pv.press) * wlim_BH * wlim_BH;

    // Get Bsq
    const vec<CCTK_REAL, 3> B_low = calc_contraction(glo, pv.Bvec);
    const CCTK_REAL Bsq = calc_contraction(B_low, pv.Bvec);

    // Norm of conserved momentum, undensitize here
    vec<CCTK_REAL, 3> mom_low = cv.mom / sqrt(spatial_detg);
    vec<CCTK_REAL, 3> mom_up = calc_contraction(gup, mom_low);
    const CCTK_REAL Ssq_old = calc_contraction(mom_low, mom_up);
    const CCTK_REAL S_old = sqrt(Ssq_old) + 1e-50;

    // Get BiSi = S_iB^i
    const CCTK_REAL BiSi_old = calc_contraction(mom_low, pv.Bvec);

    // Normalize S_iB^i by S = sqrt(S_iS^i)
    const CCTK_REAL BiEsi = BiSi_old / S_old;

    // Compute magnitude of new conserved momentum
    const CCTK_REAL Ssq_new =
        ((Z_loc + Bsq) * (Z_loc + Bsq) * vlim_BH * vlim_BH) /
        (1.0 + BiEsi * BiEsi * (2.0 * Z_loc + Bsq) / (Z_loc * Z_loc));
    const CCTK_REAL S_new = sqrt(Ssq_new);

    // Rescale momenta
    mom_low *= S_new / S_old;
    mom_up *= S_new / S_old;

    // Finally, compute velocity
    // This is (24) from https://arxiv.org/pdf/1712.07538
    pv.vel(X) = mom_up(X) / (Z_loc + Bsq);
    pv.vel(X) += BiEsi * S_new * pv.Bvec(X) / (Z_loc * (Z_loc + Bsq));

    pv.vel(Y) = mom_up(Y) / (Z_loc + Bsq);
    pv.vel(Y) += BiEsi * S_new * pv.Bvec(Y) / (Z_loc * (Z_loc + Bsq));

    pv.vel(Z) = mom_up(Z) / (Z_loc + Bsq);
    pv.vel(Z) += BiEsi * S_new * pv.Bvec(Z) / (Z_loc * (Z_loc + Bsq));

    pv.w_lor = wlim_BH;

    cv.from_prim(pv, glo);
  };
};

template <typename EOSType>
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
c2p::cons_floors_and_ceilings(const EOSType *eos_3p, cons_vars &cv,
                              const smat<CCTK_REAL, 3> &glo,
                              const CCTK_REAL &tauFluid_atm) const {

  // Limit conservative variables
  // Note that conservatives are densitized

  const CCTK_REAL spatial_detg = calc_det(glo);
  const CCTK_REAL sqrt_detg = sqrt(spatial_detg);

  const smat<CCTK_REAL, 3> gup = calc_inv(glo, spatial_detg);

  // Lower limit on tau/conserved internal energy
  // Based on Appendix A of https://arxiv.org/pdf/1112.0568

  // Compute Bsq
  vec<CCTK_REAL, 3> B_low = calc_contraction(glo, cv.dBvec);
  const CCTK_REAL BsqL = calc_contraction(B_low, cv.dBvec);
  const CCTK_REAL tauF_atmo =
      std::max(cv.dens * atmo.eps_atmo, sqrt_detg * tauFluid_atm);
  const CCTK_REAL tau_lim = 0.5 * BsqL / sqrt_detg;

  if (cv.tau <= tau_lim) {
    cv.tau = tau_lim + tauF_atmo;
  }

  // Dominant energy condition
  // (A5) from https://arxiv.org/pdf/1505.01607

  vec<CCTK_REAL, 3> mom_up = calc_contraction(gup, cv.mom);
  const CCTK_REAL mom2L = calc_contraction(cv.mom, mom_up);

  const CCTK_REAL slim = cv.dens + cv.tau;
  const CCTK_REAL slim2 = slim * slim;

  if (mom2L > slim2) {
    // (A51) from https://arxiv.org/pdf/1112.0568
    cv.mom = cv.mom * sqrt(slim2 / mom2L);
  }
};

} // namespace Con2PrimFactory

#endif
