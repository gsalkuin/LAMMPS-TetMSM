#!/usr/bin/env bash

nus=(-1 -0.5 0 0.25 0.333 0.4 0.45 0.49 0.499)

for nu in "${nus[@]}"
do
	rsync -av --delete template/ nu-$nu
	
	cd nu-$nu

	sed -i "s/NUVAL/$nu/g" in.deform
	sed -i "s/NUVAL/$nu/g" lammps-slurm.sb

	sbatch lammps-slurm.sb

	cd ..	
done