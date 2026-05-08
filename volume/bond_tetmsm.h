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

   bond_style tetmsm

   Harmonic spring for tetrahedral mass-spring models.
   No user coefficients. Shear modulus G is read from improper_style
   tetmsm, and per-bond stiffness kappa_E is computed from the
   tetrahedral mesh topology on the first compute() call.

   Energy:   U = (1/2) kappa_E * r0 * (1 - r/r0)^2
   Force:    F = -kappa_E * (r - r0) / r0

   Requires improper_style tetmsm.

   Usage:
     bond_style tetmsm
     bond_coeff *
------------------------------------------------------------------------- */

#ifdef BOND_CLASS
// clang-format off
BondStyle(tetmsm,BondTetMSM);
// clang-format on
#else

#ifndef LMP_BOND_TETMSM_H
#define LMP_BOND_TETMSM_H

#include "bond.h"

#include <unordered_map>
#include <cstdint>

namespace LAMMPS_NS {

class BondTetMSM : public Bond {
 public:
  BondTetMSM(class LAMMPS *);
  ~BondTetMSM() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, double, int, int, double &) override;
  void *extract(const char *, int &) override;

 protected:
  struct BondKey {
    int64_t lo, hi;
    bool operator==(const BondKey &o) const {
      return lo == o.lo && hi == o.hi;
    }
  };
  struct BondKeyHash {
    size_t operator()(const BondKey &k) const {
      size_t h = std::hash<int64_t>()(k.lo);
      h ^= std::hash<int64_t>()(k.hi) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  std::unordered_map<BondKey, double, BondKeyHash> r0_map;
  std::unordered_map<BondKey, double, BondKeyHash> kappa_E_map;
  double max_r0;
  bool built;    // true after topology scan or restart load

  BondKey make_key(int64_t t1, int64_t t2);
  void build_from_improperlist();
  static double tet_volume(double **x, int a1, int a2, int a3, int a4);

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif