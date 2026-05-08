.. index:: bond_style tetmsm

bond_style tetmsm command
=========================

Syntax
""""""

.. code-block:: LAMMPS

   bond_style tetmsm

Examples
""""""""

.. code-block:: LAMMPS

   bond_style tetmsm
   bond_coeff *

   improper_style tetmsm
   improper_coeff * 1000.0 0.4   # G nu

Description
"""""""""""

The *tetmsm* bond style computes a harmonic spring interaction designed
for tetrahedral mass-spring models (MSM) (:ref:`Lloyd2007 <Lloyd2007>`).
The energy and force are given by:

.. math::

   U      &= \frac{1}{2}\,\kappa_E\,r_0 \left(1 - \frac{r}{r_0}\right)^2 \\
   F      &= -\kappa_E\,\frac{r - r_0}{r_0}

where :math:`r` is the current bond length, :math:`r_0` is
the reference length, and :math:`\kappa_E` is the spring
stiffness (energy/length). The conventional spring constant is
:math:`k_E = \kappa_E / r_0` (force/length).

This command should be used with :doc:`improper_style tetmsm <improper_tetmsm>`
to model an isotropic linear-elastic solid. The spring stiffness
:math:`\kappa_E` is computed automatically from the shear modulus :math:`G`
(read from the improper style) and the tetrahedral mesh topology
(:ref:`Lloyd2007 <Lloyd2007>`, :ref:`Golec2020 <Golec2020>`):

.. math::

   k_E = \frac{\sqrt{2}}{5} \sum_\Delta^\text{adj. tets}
   G_\Delta \left(\frac{12 V_\Delta}{\sqrt{2}} \right)^{1/3}

where the summand is an effective edge length of an irregular tetrahedron
:math:`\Delta` with volume :math:`V_\Delta`. For multi-material meshes
with multiple improper types, each tetrahedron contributes its own
:math:`G_\Delta` according to its improper type. The sum over adjacent
tetrahedra is evaluated by scanning the improper topology at the beginning
of the first :doc:`run <run>` or :doc:`minimize <minimize>` command.
Interior edges, shared by more tetrahedra, naturally accumulate higher
stiffness than boundary edges.

This bond style accepts no coefficients. The :doc:`bond_coeff <bond_coeff>`
command must still be issued for all bond types, but with no arguments:

.. code-block:: LAMMPS

   bond_coeff *

.. note::

   The equilibrium bond length :math:`r_0` and stiffness :math:`\kappa_E`
   are determined from the atom positions and the improper (tetrahedral)
   topology at the beginning of the first :doc:`run <run>` or
   :doc:`minimize <minimize>` command after the bond style has been defined.
   Subsequent ``run`` commands in the same input script reuse the cached values.
   To reset, re-issue the ``bond_style tetmsm`` command before the next ``run``.

----------

Restart info
""""""""""""

This bond style supports the :doc:`write_restart <write_restart>` and
:doc:`read_restart <read_restart>` commands. The per-bond reference lengths
:math:`r_0` and stiffness :math:`\kappa_E` are stored.

Restrictions
""""""""""""

This bond style requires :doc:`improper_style tetmsm <improper_tetmsm>` to be
defined. The improper topology provides the tetrahedral connectivity needed
to compute the per-bond stiffness.

This bond style maintains internal data (reference lengths :math:`r_0` and
stiffness :math:`\kappa_E`).  This information will be written to binary
restart files but not to data files.  Thus, continuing a simulation from a
deformed state is only possible with :doc:`read_restart <read_restart>`.
When using :doc:`read_data <read_data>`, the internal data will be
re-initialized from the current geometry.

This bond style requires that atoms have tags (``atom_modify id yes``,
which is the default).

Related commands
""""""""""""""""

:doc:`bond_coeff <bond_coeff>`,
:doc:`bond_style harmonic/restrain <bond_harmonic_restrain>`,
:doc:`improper_style tetmsm <improper_tetmsm>`

Default
"""""""

none

----------

.. _Lloyd2007:

**(Lloyd2007)** Lloyd, Szekely, Harders, IEEE Trans Vis Comput Graph, 13(5),
1081-1094 (2007).

.. _Golec2020:

**(Golec2020)** Golec, Palierne, Zara, Nicolle, Damiand, Vis Comput, 36,
809-825 (2020).