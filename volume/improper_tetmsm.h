/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Gabriel Alkuino (Syracuse University)

   improper_style tetmsm

   optional keyword:
     scale value = scale factor (default 1.0) for the volume stiffness kappa_V

   Tetrahedral volume penalty for mass-spring models on tet meshes.
   Reference volume V0 is auto-computed from initial geometry and
   stored in binary restart files.

   Energy:   U = (1/2) kappa_V * V0 * (1 - V/V0)^2
   Force:    f_i = -kappa_V * (V - V0) / V0 * dV/dx_i

    where kappa_V = scale * G * (4*nu - 1) / (1 - 2*nu) is computed internally
   from the user-supplied shear modulus G and Poisson's ratio nu.

   Coeffs:   improper_coeff TYPE G nu
------------------------------------------------------------------------- */

#ifdef IMPROPER_CLASS
// clang-format off
ImproperStyle(tetmsm,ImproperTetMSM);
// clang-format on
#else

#ifndef LMP_IMPROPER_TETMSM_H
#define LMP_IMPROPER_TETMSM_H

#include "improper.h"

#include <unordered_map>
#include <cstdint>

namespace LAMMPS_NS {

class ImproperTetMSM : public Improper {
 public:
  ImproperTetMSM(class LAMMPS *);
  ~ImproperTetMSM() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void *extract(const char *, int &) override;

 protected:
  double *G_coeff;
  double *nu_coeff;
  double *kappa_v;

  double scale;
  bool scale_set;

  struct ImpKey {
    int64_t t1, t2, t3, t4;
    bool operator==(const ImpKey &o) const {
      return t1 == o.t1 && t2 == o.t2 && t3 == o.t3 && t4 == o.t4;
    }
  };
  struct ImpKeyHash {
    size_t operator()(const ImpKey &k) const {
      size_t h = std::hash<int64_t>()(k.t1);
      h ^= std::hash<int64_t>()(k.t2) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int64_t>()(k.t3) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int64_t>()(k.t4) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };
  std::unordered_map<ImpKey, double, ImpKeyHash> v0_map;
  bool built;    // true after V0 map replication or restart load

  double compute_tet_volume(double **x, int i1, int i2, int i3, int i4);
  double get_v0(int64_t t1, int64_t t2, int64_t t3, int64_t t4,
                double **x, int i1, int i2, int i3, int i4);

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif