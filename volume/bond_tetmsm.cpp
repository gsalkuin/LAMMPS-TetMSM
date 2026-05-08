/* ----------------------------------------------------------------------
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
------------------------------------------------------------------------- */

#include "bond_tetmsm.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "memory.h"
#include "neighbor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondTetMSM::BondTetMSM(LAMMPS *_lmp) : Bond(_lmp)
{
  writedata = 0;
  reinitflag = 0;
  max_r0 = 0.0;
  built = false;
}

/* ---------------------------------------------------------------------- */

BondTetMSM::~BondTetMSM()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
  }
}

/* ---------------------------------------------------------------------- */

BondTetMSM::BondKey BondTetMSM::make_key(int64_t t1, int64_t t2)
{
  return {std::min(t1, t2), std::max(t1, t2)};
}

/* ---------------------------------------------------------------------- */

double BondTetMSM::tet_volume(double **x, int a1, int a2, int a3, int a4)
{
  double ax = x[a2][0] - x[a1][0];
  double ay = x[a2][1] - x[a1][1];
  double az = x[a2][2] - x[a1][2];

  double bx = x[a3][0] - x[a1][0];
  double by = x[a3][1] - x[a1][1];
  double bz = x[a3][2] - x[a1][2];

  double cx = x[a4][0] - x[a1][0];
  double cy = x[a4][1] - x[a1][1];
  double cz = x[a4][2] - x[a1][2];

  return (ax * (by * cz - bz * cy) +
          ay * (bz * cx - bx * cz) +
          az * (bx * cy - by * cx)) / 6.0;
}

/* ----------------------------------------------------------------------
   Scan neighbor->improperlist to compute per-bond kappa_E and r0.
   Must be called after neighbor lists are built (ghost atoms present).
   Each tet contributes G_Delta * l_eff to its 6 edges, supporting
   per-improper-type G for multi-material meshes.
------------------------------------------------------------------------- */

void BondTetMSM::build_from_improperlist()
{
  double **x = atom->x;
  tagint *tag = atom->tag;

  // Extract per-type G array from improper_style tetmsm
  int dim;
  double *G_ptr = (double *) force->improper->extract("G", dim);
  if (!G_ptr)
    error->all(FLERR, "Bond tetmsm: cannot extract G from improper_style tetmsm");

  int **improperlist = neighbor->improperlist;
  int nimproperlist = neighbor->nimproperlist;

  // Phase 1: accumulate G * l_eff per edge across local adjacent tets

  std::unordered_map<BondKey, double, BondKeyHash> Gleff_sum_local;

  for (int n = 0; n < nimproperlist; n++) {
    int i1 = improperlist[n][0];
    int i2 = improperlist[n][1];
    int i3 = improperlist[n][2];
    int i4 = improperlist[n][3];
    int type = improperlist[n][4];

    double G_tet = G_ptr[type];
    double vol = tet_volume(x, i1, i2, i3, i4);
    double l_eff = cbrt(12.0 * fabs(vol) / sqrt(2.0));
    double Gl = G_tet * l_eff;

    // Push G * l_eff into all 6 edges of this tet
    tagint tags[4] = {tag[i1], tag[i2], tag[i3], tag[i4]};
    for (int e1 = 0; e1 < 3; e1++)
      for (int e2 = e1 + 1; e2 < 4; e2++)
        Gleff_sum_local[make_key(tags[e1], tags[e2])] += Gl;
  }

  // Replicate full per-edge sum across all ranks
  // Entry layout: (lo, hi, Gleff_sum) = 2 int64_t + 1 double

  const int nprocs = comm->nprocs;
  const int entry_bytes = 2 * sizeof(int64_t) + sizeof(double);

  int local_n = Gleff_sum_local.size();
  std::vector<char> local_buf(local_n * entry_bytes);
  char *ptr = local_buf.data();
  for (auto &kv : Gleff_sum_local) {
    const BondKey &key = kv.first;
    const double val = kv.second;
    memcpy(ptr, &key.lo, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(ptr, &key.hi, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(ptr, &val, sizeof(double));     ptr += sizeof(double);
  }

  std::vector<int> counts(nprocs), displs(nprocs), byte_counts(nprocs);
  MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

  displs[0] = 0;
  for (int i = 1; i < nprocs; i++)
    displs[i] = displs[i - 1] + counts[i - 1] * entry_bytes;
  for (int i = 0; i < nprocs; i++)
    byte_counts[i] = counts[i] * entry_bytes;

  const int total_bytes = displs[nprocs - 1] + byte_counts[nprocs - 1];
  std::vector<char> global_buf(total_bytes);
  MPI_Allgatherv(local_buf.data(), local_n * entry_bytes, MPI_CHAR,
                 global_buf.data(), byte_counts.data(), displs.data(),
                 MPI_CHAR, world);

  std::unordered_map<BondKey, double, BondKeyHash> Gleff_sum;
  Gleff_sum.reserve(global_buf.size() / entry_bytes);

  ptr = global_buf.data();
  const int total_entries = total_bytes / entry_bytes;
  for (int i = 0; i < total_entries; i++) {
    int64_t lo, hi;
    double val;
    memcpy(&lo, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&hi, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&val, ptr, sizeof(double));  ptr += sizeof(double);
    Gleff_sum[{lo, hi}] += val;
  }

  // Phase 2: compute r0 where endpoints are locally resolvable, then replicate

  std::unordered_map<BondKey, double, BondKeyHash> r0_local;
  r0_local.reserve(Gleff_sum.size());

  for (auto &[key, Gl_sum] : Gleff_sum) {
    int a1 = atom->map(key.lo);
    int a2 = atom->map(key.hi);
    if (a1 < 0 || a2 < 0) continue;

    double dx = x[a1][0] - x[a2][0];
    double dy = x[a1][1] - x[a2][1];
    double dz = x[a1][2] - x[a2][2];
    domain->minimum_image(FLERR, dx, dy, dz);
    double r0 = sqrt(dx * dx + dy * dy + dz * dz);

    if (r0 <= 0.0)
      error->one(FLERR, "Bond tetmsm: zero initial bond length");

    r0_local[key] = r0;
  }

  int local_r0_n = r0_local.size();
  std::vector<char> r0_buf(local_r0_n * entry_bytes);
  ptr = r0_buf.data();
  for (auto &kv : r0_local) {
    const BondKey &key = kv.first;
    const double val = kv.second;
    memcpy(ptr, &key.lo, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(ptr, &key.hi, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(ptr, &val, sizeof(double));     ptr += sizeof(double);
  }

  MPI_Allgather(&local_r0_n, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

  displs[0] = 0;
  for (int i = 1; i < nprocs; i++)
    displs[i] = displs[i - 1] + counts[i - 1] * entry_bytes;
  for (int i = 0; i < nprocs; i++)
    byte_counts[i] = counts[i] * entry_bytes;

  const int total_r0_bytes = displs[nprocs - 1] + byte_counts[nprocs - 1];
  std::vector<char> global_r0_buf(total_r0_bytes);
  MPI_Allgatherv(r0_buf.data(), local_r0_n * entry_bytes, MPI_CHAR,
                 global_r0_buf.data(), byte_counts.data(), displs.data(),
                 MPI_CHAR, world);

  r0_map.clear();
  r0_map.reserve(global_r0_buf.size() / entry_bytes);

  ptr = global_r0_buf.data();
  const int total_r0_entries = total_r0_bytes / entry_bytes;
  for (int i = 0; i < total_r0_entries; i++) {
    int64_t lo, hi;
    double val;
    memcpy(&lo, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&hi, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&val, ptr, sizeof(double));  ptr += sizeof(double);
    r0_map[{lo, hi}] = val;
  }

  for (auto &[key, Gl_sum] : Gleff_sum) {
    if (r0_map.find(key) == r0_map.end())
      error->all(FLERR, "Bond tetmsm: some edges could not be resolved; increase communication cutoff");
  }

  const double prefactor = sqrt(2.0) / 5.0;

  kappa_E_map.clear();
  kappa_E_map.reserve(Gleff_sum.size());
  max_r0 = 0.0;

  for (auto &[key, Gl_sum] : Gleff_sum) {
    const double r0 = r0_map[key];
    kappa_E_map[key] = prefactor * r0 * Gl_sum;
    if (r0 > max_r0) max_r0 = r0;
  }

  built = true;
}

/* ---------------------------------------------------------------------- */

void BondTetMSM::init_style()
{
  // Validate that improper_style tetmsm is defined
  if (!force->improper)
    error->all(FLERR, "Bond tetmsm requires improper_style tetmsm");

  int dim;
  if (!force->improper->extract("G", dim))
    error->all(FLERR, "Bond tetmsm requires improper_style tetmsm "
               "(could not extract G)");

  // Do NOT build here — neighbor lists don't exist yet.
  // build_from_improperlist() is called on the first compute().
}

/* ---------------------------------------------------------------------- */

void BondTetMSM::compute(int eflag, int vflag)
{
  int i1, i2, n, type;
  double delx, dely, delz, ebond, fbond;
  double rsq, r, r0, dr, kappa_E;

  // Deferred build: neighbor lists (with ghost atoms) now exist
  if (!built)
    build_from_improperlist();

  ebond = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx * delx + dely * dely + delz * delz;
    r = sqrt(rsq);

    BondKey key = make_key(tag[i1], tag[i2]);

    auto it_r0 = r0_map.find(key);
    auto it_kE = kappa_E_map.find(key);
    if (it_r0 == r0_map.end() || it_kE == kappa_E_map.end())
      error->one(FLERR, "Bond tetmsm: bond not found in internal maps");

    r0 = it_r0->second;
    kappa_E = it_kE->second;

    // U = (1/2) kappa_E * r0 * (1 - r/r0)^2
    //   = kappa_E * (r - r0)^2 / (2 * r0)
    // F = -kappa_E * (r - r0) / r0

    dr = r - r0;

    if (eflag) ebond = kappa_E * dr * dr / (2.0 * r0);

    if (r > 0.0)
      fbond = -kappa_E * dr / (r0 * r);
    else
      fbond = 0.0;

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += fbond * delx;
      f[i1][1] += fbond * dely;
      f[i1][2] += fbond * delz;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= fbond * delx;
      f[i2][1] -= fbond * dely;
      f[i2][2] -= fbond * delz;
    }

    if (evflag) ev_tally(i1, i2, nlocal, newton_bond, ebond, fbond,
                         delx, dely, delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondTetMSM::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   bond_coeff TYPE   (no additional arguments)
------------------------------------------------------------------------- */

void BondTetMSM::coeff(int narg, char **arg)
{
  if (narg != 1)
    error->all(FLERR, "Incorrect args for bond coefficients: "
               "expected 'bond_coeff TYPE' with no additional arguments");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  for (int i = ilo; i <= ihi; i++)
    setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */

double BondTetMSM::equilibrium_distance(int /*type*/)
{
  if (max_r0 > 0.0) return max_r0;

  // Before first run: scan per-atom bond arrays for max bond length
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;

  double maxlen = 0.0;
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < num_bond[i]; j++) {
      int k = atom->map(bond_atom[i][j]);
      if (k < 0) continue;
      double dx = x[i][0] - x[k][0];
      double dy = x[i][1] - x[k][1];
      double dz = x[i][2] - x[k][2];
      maxlen = fmax(maxlen, sqrt(dx * dx + dy * dy + dz * dz));
    }
  }

  double maxall;
  MPI_Allreduce(&maxlen, &maxall, 1, MPI_DOUBLE, MPI_MAX, world);
  return maxall;
}

/* ---------------------------------------------------------------------- */

double BondTetMSM::single(int type, double rsq, int i, int j, double &fforce)
{
  double r = sqrt(rsq);
  BondKey key = make_key(atom->tag[i], atom->tag[j]);

  auto it_r0 = r0_map.find(key);
  auto it_kE = kappa_E_map.find(key);
  if (it_r0 == r0_map.end() || it_kE == kappa_E_map.end()) {
    fforce = 0.0;
    return 0.0;
  }

  double r0 = it_r0->second;
  double kappa_E = it_kE->second;
  double dr = r - r0;

  fforce = 0.0;
  if (r > 0.0) fforce = -kappa_E * dr / (r0 * r);

  return kappa_E * dr * dr / (2.0 * r0);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart:
     1. r0 map  (n, then [lo, hi, r0] triples)
     2. kappa_E map  (n, then [lo, hi, kappa_E] triples)
------------------------------------------------------------------------- */

void BondTetMSM::write_restart(FILE *fp)
{
  // r0 map
  int n = r0_map.size();
  fwrite(&n, sizeof(int), 1, fp);
  for (auto &[key, val] : r0_map) {
    fwrite(&key.lo, sizeof(int64_t), 1, fp);
    fwrite(&key.hi, sizeof(int64_t), 1, fp);
    fwrite(&val, sizeof(double), 1, fp);
  }

  // kappa_E map
  n = kappa_E_map.size();
  fwrite(&n, sizeof(int), 1, fp);
  for (auto &[key, val] : kappa_E_map) {
    fwrite(&key.lo, sizeof(int64_t), 1, fp);
    fwrite(&key.hi, sizeof(int64_t), 1, fp);
    fwrite(&val, sizeof(double), 1, fp);
  }
}

/* ---------------------------------------------------------------------- */

void BondTetMSM::read_restart(FILE *fp)
{
  allocate();
  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;

  const int entry_bytes = 2 * sizeof(int64_t) + sizeof(double);

  auto read_map = [&](std::unordered_map<BondKey, double, BondKeyHash> &map) {
    int n = 0;
    if (comm->me == 0)
      utils::sfread(FLERR, &n, sizeof(int), 1, fp, nullptr, error);
    MPI_Bcast(&n, 1, MPI_INT, 0, world);

    std::vector<char> buf(n * entry_bytes);
    if (comm->me == 0 && n > 0)
      utils::sfread(FLERR, buf.data(), entry_bytes, n, fp, nullptr, error);
    if (n > 0)
      MPI_Bcast(buf.data(), n * entry_bytes, MPI_CHAR, 0, world);

    map.clear();
    map.reserve(n);
    char *ptr = buf.data();
    for (int i = 0; i < n; i++) {
      int64_t lo, hi;
      double val;
      memcpy(&lo, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(&hi, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(&val, ptr, sizeof(double));  ptr += sizeof(double);
      map[{lo, hi}] = val;
    }
  };

  read_map(r0_map);
  read_map(kappa_E_map);

  max_r0 = 0.0;
  for (auto &[key, val] : r0_map)
    if (val > max_r0) max_r0 = val;

  built = true;
}

/* ---------------------------------------------------------------------- */

void BondTetMSM::write_data(FILE * /*fp*/)
{
  // no per-type coefficients
}

/* ---------------------------------------------------------------------- */

void *BondTetMSM::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "max_r0") == 0) return (void *) &max_r0;
  return nullptr;
}