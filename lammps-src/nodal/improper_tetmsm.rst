.. index:: improper_style tetmsm

improper_style tetmsm command
=============================

Syntax
""""""

.. code-block:: LAMMPS

   improper_style tetmsm

Examples
""""""""

.. code-block:: LAMMPS

   improper_style tetmsm
   improper_coeff 1 1000.0 0.4

   improper_style tetmsm
   improper_coeff * 500.0 0.25

Description
"""""""""""

The *tetmsm* improper style computes a volumetric penalty for a
tetrahedral mass-spring model (MSM).  Each improper interaction is
defined by four atoms that form one tetrahedron.
A nodal-averaged volume penalty is used to avoid volumetric locking
since the number of tetrahedra typically exceed the nodal degrees
of freedom (:ref:`Bonet1998 <Bonet1998b>`). 

A potential energy that is quadratic in the volumetric strain
is applied to each atom :math:`i`:

.. math::

   U_i = \frac{1}{2}\,\kappa_V\,V_{0,i}\left(\frac{V_i - V_{0,i}}{V_{0,i}}\right)^2,

where :math:`V_i` is the nodal-averaged volume of atom :math:`i`,
:math:`V_{0,i}` is the reference nodal volume computed from the initial geometry,
and :math:`\kappa_V` is the volume stiffness (energy/volume).
More details are given below.

This command should be used in conjunction with :doc:`bond_style tetmsm <bond_tetmsm>`
to model an isotropic linear-elastic solid with arbitrary Poisson's ratio :math:`\nu`.
The volume stiffness :math:`\kappa_V` is given by (:ref:`Golec2020 <Golec2020b>`)

.. math::

   \kappa_V = \frac{G\,(4\nu - 1)}{1 - 2\nu} .

For a MSM with only central-force springs, the Cauchy relations require :math:`\nu = 1/4`;
a volumetric term is added to achieve other values of :math:`\nu`
(:ref:`Golec2020 <Golec2020b>`, :ref:`Clemmer2024 <Clemmer2024b>`).
If :math:`\nu = 1/4`, :math:`\kappa_V = 0` and the MSM will reduce to Lloyd's MSM (:ref:`Lloyd2007 <Lloyd2007b>`).

The four atoms in each improper must be ordered such that the signed
tetrahedron volume :math:`V > 0` in the reference configuration.  The standard
convention is to list nodes :math:`(1,2,3)` in counter-clockwise
order when viewed from node :math:`4`.  If the reference volume is non-positive,
the code will abort with an error.

The following coefficients must be defined for each improper type via the
:doc:`improper_coeff <improper_coeff>` command as in the example above,
or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands:

* :math:`G` (pressure units) = shear modulus
* :math:`\nu` (dimensionless) = Poisson's ratio

The volume stiffness :math:`\kappa_V` is computed internally from these
values.  The shear modulus :math:`G` is also used by
:doc:`bond_style tetmsm <bond_tetmsm>` to compute the spring stiffness.

The reference nodal volumes are computed from the initial atom positions
at the beginning of the first :doc:`run <run>` or :doc:`minimize <minimize>`
command after the improper style has been defined.

.. note::

   All improper types must have the same :math:`G` and :math:`\nu`.  
   Different :math:`\kappa_V`'s per tetrahedron would require a nontrivial
   modification to the current implementation.  The code will abort at
   the start of a run if mismatched coefficients are detected.

.. note::

   The reference nodal volumes :math:`V_{0,i}` are determined from the atom
   positions at the beginning of the first :doc:`run <run>` or
   :doc:`minimize <minimize>` command after the improper style has been defined.
   Subsequent ``run`` commands in the same input script reuse the cached values.
   To reset :math:`V_{0,i}`, re-issue the ``improper_style tetmsm`` command before
   the next ``run``.

.. note::

   This style creates an internal ``fix property/atom`` to store the
   per-node reference volumes (``d_nodalvol0``) and a communication
   helper fix (``fix tetmsm/comm``) for MPI exchanges.  These are
   managed automatically and should not be deleted or modified by the
   user.

----------

Implementation details
""""""""""""""""""""""

First, define the (signed) volume of a tetrahedron :math:`\Delta` with
nodes :math:`(1,2,3,4)` as

.. math::

   V = \frac{1}{6}\,\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c}),
   \qquad
   \mathbf{a} = \mathbf{x}_2 - \mathbf{x}_1,\;
   \mathbf{b} = \mathbf{x}_3 - \mathbf{x}_1,\;
   \mathbf{c} = \mathbf{x}_4 - \mathbf{x}_1.

Then define a nodal-averaged volume for each atom :math:`i` as

.. math::

   V_i = \frac{1}{4}\sum_{\Delta \in \mathcal{T}_i} V_{\Delta}.

where :math:`\mathcal{T}_i` is the set of tetrahedra adjacent to node :math:`i`.

Let :math:`V_{0,i}` be the nodal reference volume computed from the
initial geometry.  Define the nodal volumetric strain

.. math::

   \varepsilon_i = \frac{V_i - V_{0,i}}{V_{0,i}}.

The volumetric penalty energy of atom :math:`i` is

.. math::

   U_i = \frac{1}{2}\,\kappa_V\,V_{0,i}\,\varepsilon_i^2 ,

and the total volumetric energy of the system is :math:`U = \sum_j U_j`.
The force on node :math:`i` follows from differentiating
:math:`U` with respect to :math:`\mathbf{x}_i`:

.. math::

   \mathbf{f}_i = -\frac{\partial U}{\partial \mathbf{x}_i}
               = -\sum_j \kappa_V\,\varepsilon_j\,
                  \frac{\partial V_j}{\partial \mathbf{x}_i},

where the sum runs over all nodes in :math:`\mathcal{T}_i`.
From the nodal averaging definition of :math:`V_j`, it follows that

.. math::

   \frac{\partial V_j}{\partial \mathbf{x}_i}
   = \frac{1}{4}\sum_{\Delta \in \mathcal{T}_j \cap \mathcal{T}_i}
     \frac{\partial V_\Delta}{\partial \mathbf{x}_i}.

Therefore,

.. math::

   \mathbf{f}_i
   = -\frac{1}{4}\sum_j \kappa_V\,\varepsilon_j
     \sum_{\Delta \in \mathcal{T}_j \cap \mathcal{T}_i}
     \frac{\partial V_\Delta}{\partial \mathbf{x}_i}.

Let :math:`\chi_{j\Delta}` be the mesh incidence indicator
(:math:`\chi_{j\Delta} = 1` if node :math:`j` is a node of
:math:`\Delta`, and :math:`0` otherwise).  Writing the restricted
sums as unrestricted sums filtered by :math:`\chi` and defining
:math:`\mathbf{g}^\Delta_i = \partial V_\Delta / \partial \mathbf{x}_i`,

.. math::

   \sum_j \varepsilon_j
   \sum_{\Delta \in \mathcal{T}_j \cap \mathcal{T}_i}
   \mathbf{g}^\Delta_i
   \;=\;
   \sum_j \sum_\Delta
   \varepsilon_j\,\mathbf{g}^\Delta_i\,
   \chi_{j\Delta}\,\chi_{i\Delta}
   \;=\;
   \sum_\Delta \chi_{i\Delta}\,\mathbf{g}^\Delta_i
   \sum_j \chi_{j\Delta}\,\varepsilon_j
   \;=\;
   \sum_{\Delta \in \mathcal{T}_i}
   \mathbf{g}^\Delta_i
   \sum_{v \in \mathrm{verts}(\Delta)} \varepsilon_v.

The swap is valid because :math:`\chi_{i\Delta}` does not depend
on :math:`j` and :math:`\varepsilon_j` does not depend on
:math:`\Delta`.  The result is

.. math::

   \mathbf{f}_i
   = -\frac{1}{4}
     \sum_{\Delta \in \mathcal{T}_i}
     \frac{\partial V_\Delta}{\partial \mathbf{x}_i}
     \sum_{v \in \mathrm{verts}(\Delta)} \kappa_V\,\varepsilon_v.

For each tetrahedron :math:`\Delta` with nodes :math:`\{1,2,3,4\}`,
the force contribution on node :math:`k` is

.. math::

   \mathbf{f}_k^{(\Delta)}
   = -\frac{1}{4}\,\frac{\partial V_\Delta}{\partial \mathbf{x}_k}
     \sum_{v \in \{1,2,3,4\}} \kappa_V\,\varepsilon_v.

The tetrahedron volume gradients (face-area vectors) are:

.. math::

   \frac{\partial V_\Delta}{\partial \mathbf{x}_2} &= \frac{1}{6}(\mathbf{b}\times\mathbf{c}), \qquad
   \frac{\partial V_\Delta}{\partial \mathbf{x}_3} = \frac{1}{6}(\mathbf{c}\times\mathbf{a}), \qquad
   \frac{\partial V_\Delta}{\partial \mathbf{x}_4} = \frac{1}{6}(\mathbf{a}\times\mathbf{b}), \\
   \frac{\partial V_\Delta}{\partial \mathbf{x}_1} &=
   -\left(\frac{\partial V_\Delta}{\partial \mathbf{x}_2}
   + \frac{\partial V_\Delta}{\partial \mathbf{x}_3}
   + \frac{\partial V_\Delta}{\partial \mathbf{x}_4}\right).

Define the per-tetrahedron shorthand

.. math::

   S_{\Delta} = \kappa_V\,(\varepsilon_1 + \varepsilon_2 + \varepsilon_3 + \varepsilon_4).

The resulting forces on each node of tetrahedron :math:`\Delta` are:

.. math::

   \mathbf{f}_2 &= -\frac{S_{\Delta}}{24}(\mathbf{b}\times\mathbf{c}), \qquad
   \mathbf{f}_3 = -\frac{S_{\Delta}}{24}(\mathbf{c}\times\mathbf{a}), \qquad
   \mathbf{f}_4 = -\frac{S_{\Delta}}{24}(\mathbf{a}\times\mathbf{b}), \\
   \mathbf{f}_1 &= -(\mathbf{f}_2 + \mathbf{f}_3 + \mathbf{f}_4).

The implementation requires two passes over the improper list:
first to compute the per-atom volume and strain, and then to 
compute the forces as described above.

----------

Restart info
""""""""""""

This improper style supports the :doc:`write_restart <write_restart>` and
:doc:`read_restart <read_restart>` commands.

The :math:`G` and :math:`\nu` coefficients for each improper type are stored
by this improper style. However, all improper types must share
the same :math:`G` and :math:`\nu`.

The nodal reference volumes :math:`V_{0,i}` are stored as a per-atom custom
vector created internally via :doc:`fix property/atom <fix_property_atom>`, and
are included in binary restart files.

Restrictions
""""""""""""

This improper style requires binary restart files to continue from a deformed
state with the original nodal reference volumes :math:`V_{0,i}`.
When using :doc:`read_data <read_data>`, the reference nodal volumes will be
re-initialized from the current geometry.

This improper style requires that atoms have tags (``atom_modify id yes``,
which is the default).


Related commands
""""""""""""""""

:doc:`improper_coeff <improper_coeff>`,
:doc:`bond_style tetmsm <bond_tetmsm>`

Default
"""""""

none

----------

.. _Bonet1998b:

**(Bonet1998)** Bonet, Burton, Comput Methods Appl Mech Eng, 154(1-2),
73-81 (1998).

.. _Lloyd2007b:

**(Lloyd2007)** Lloyd, Szekely, Harders, IEEE Trans Vis Comput Graph, 13(5),
1081-1094 (2007).

.. _Golec2020b:

**(Golec2020)** Golec, Palierne, Zara, Nicolle, Damiand, Vis Comput, 36,
809-825 (2020).

.. _Clemmer2024b:

**(Clemmer2024)** Clemmer, Monti, Lechman, Soft Matter, 20, 1702-1718 (2024).