#!/usr/bin/env bash

nus=(0.25 0.30 0.35 0.40 0.45 0.46 0.47 0.48 0.49)


for nu in "${nus[@]}"
do
	rsync -av --delete template/ nu-$nu
	
	cd nu-$nu

	sed -i "s/NUVAL/$nu/g" in.deform
	sed -i "s/NUVAL/$nu/g" lammps-slurm.sb

	sbatch lammps-slurm.sb

	cd ..	
done