.. index:: improper_style tetmsm

improper_style tetmsm command
=============================

Syntax
""""""

.. code-block:: LAMMPS

   improper_style tetmsm args

* args = optional arguments

.. parsed-literal::

   args = *scale* value
     *scale* value = scale factor (default 1.0) for the volume stiffness :math:`\kappa_V`

Examples
""""""""

.. code-block:: LAMMPS

   improper_style tetmsm
   improper_coeff 1 1000.0 0.4

   improper_style tetmsm scale 2.0
   improper_coeff 1 1000.0 0.4

   improper_style tetmsm
   improper_coeff * 500.0 0.25

Description
"""""""""""

The *tetmsm* improper style computes a volumetric penalty on each
tetrahedron in a tetrahedral mass-spring model (MSM).  Each improper
interaction is defined by four atoms that form one tetrahedron.
The energy and forces are given by:

.. math::

   U      &= \frac{1}{2}\,\kappa_V\,V_0 \left(1 - \frac{V}{V_0}\right)^2 \\
   \mathbf{f}_i &= -\kappa_V\,\frac{V - V_0}{V_0}\,\frac{\partial V}{\partial \mathbf{x}_i}

where :math:`V` is the volume of the tetrahedron:

.. math::

   V = \frac{1}{6}\,\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c}),
   \qquad
   \mathbf{a} = \mathbf{x}_2 - \mathbf{x}_1,\;
   \mathbf{b} = \mathbf{x}_3 - \mathbf{x}_1,\;
   \mathbf{c} = \mathbf{x}_4 - \mathbf{x}_1

and the volume gradients are:

.. math::

   \frac{\partial V}{\partial \mathbf{x}_2} &= \frac{1}{6}(\mathbf{b}\times\mathbf{c}), \qquad
   \frac{\partial V}{\partial \mathbf{x}_3} = \frac{1}{6}(\mathbf{c}\times\mathbf{a}), \qquad
   \frac{\partial V}{\partial \mathbf{x}_4} = \frac{1}{6}(\mathbf{a}\times\mathbf{b}) \\
   \frac{\partial V}{\partial \mathbf{x}_1} &=
   -\left(\frac{\partial V}{\partial \mathbf{x}_2}
   + \frac{\partial V}{\partial \mathbf{x}_3}
   + \frac{\partial V}{\partial \mathbf{x}_4}\right)

The coefficient :math:`\kappa_V` is the volume stiffness
(energy/volume).  The conventional volume penalty constant is
:math:`k_V = \kappa_V \cdot V_0` (energy).

The optional *scale* keyword multiplies the volume stiffness :math:`\kappa_V`
by the value provided.  This scales the improper energy, forces, and virial
contributions from this style, but does not change the coefficients used by
:doc:`bond_style tetmsm <bond_tetmsm>`.

This command should be used in conjunction with :doc:`bond_style tetmsm <bond_tetmsm>`
to model an isotropic linear-elastic solid with arbitrary Poisson's ratio :math:`\nu`.
The volume stiffness :math:`\kappa_V` is computed internally from the coefficients:

.. math::

   \kappa_V = \text{scale} \cdot \frac{G\,(4\nu - 1)}{1 - 2\nu}

For a MSM with only central-force springs, the Cauchy relations require :math:`\nu = 1/4`;
a volumetric term is added to achieve other values of :math:`\nu`
(:ref:`Golec2020 <Golec2020b>`, :ref:`Clemmer2024 <Clemmer2024b>`).
If :math:`\nu = 1/4`, :math:`\kappa_V = 0` and the MSM will reduce to Lloyd's MSM (:ref:`Lloyd2007 <Lloyd2007b>`).

The four atoms in each improper must be ordered such that the signed
volume :math:`V_0 > 0` in the reference configuration.  The standard
convention is to list vertices :math:`(1,2,3)` in counter-clockwise
order when viewed from vertex 4.  If the reference volume is non-positive
(e.g. degenerate tetrahedron or incorrect vertex ordering), the code will
abort with an error.

The following coefficients must be defined for each improper type via the
:doc:`improper_coeff <improper_coeff>` command as in the example above,
or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands:

* :math:`G` (pressure) = shear modulus
* :math:`\nu` (dimensionless) = Poisson's ratio

The volume stiffness :math:`\kappa_V` is computed internally from these
values.  The shear modulus :math:`G` is also used by
:doc:`bond_style tetmsm <bond_tetmsm>` to compute the spring stiffness.

The reference volume :math:`V_0` is **not** specified as a coefficient;
instead, it is computed from the initial atom positions on the first
timestep and cached internally (keyed by global atom tags).

.. note::

   The equilibrium volume :math:`V_0` for each tetrahedron is determined from
   the atom positions at the beginning of the first :doc:`run <run>` or
   :doc:`minimize <minimize>` command after the improper style has been defined.
   Subsequent ``run`` commands in the same input script reuse the cached values.
   To reset :math:`V_0`, re-issue the ``improper_style tetmsm`` command before
   the next ``run``.

----------

Restart info
""""""""""""

This improper style supports the :doc:`write_restart <write_restart>` and
:doc:`read_restart <read_restart>` commands. The :math:`G` and :math:`\nu`
coefficients for each improper type and the per-tet reference volumes
:math:`V_0` are stored.  The *scale* setting is also stored.

Restrictions
""""""""""""

This improper style maintains internal data to determine the original tetrahedron
volumes :math:`V_0`.  This information will be written to binary restart
files but not to data files.  Thus, continuing a simulation from a
deformed state is only possible with :doc:`read_restart <read_restart>`.
When using :doc:`read_data <read_data>`, the reference volumes will be
re-initialized from the current geometry.

This improper style requires that atoms have tags (``atom_modify id yes``,
which is the default).

Related commands
""""""""""""""""

:doc:`improper_coeff <improper_coeff>`,
:doc:`improper_style harmonic <improper_harmonic>`,
:doc:`bond_style tetmsm <bond_tetmsm>`

Default
"""""""

The default for *scale* is 1.0.

----------

.. _Lloyd2007b:

**(Lloyd2007)** Lloyd, Szekely, Harders, IEEE Trans Vis Comput Graph, 13(5),
1081-1094 (2007).

.. _Golec2020b:

**(Golec2020)** Golec, Palierne, Zara, Nicolle, Damiand, Vis Comput, 36,
809-825 (2020).

.. _Clemmer2024b:

**(Clemmer2024)** Clemmer, Monti, Lechman, Soft Matter, 20, 1702-1718 (2024).