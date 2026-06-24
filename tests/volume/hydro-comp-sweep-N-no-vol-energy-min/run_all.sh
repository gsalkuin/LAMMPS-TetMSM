#!/usr/bin/env bash

layers=(4 8 16 32 64)
ntasks=(1 2 8 16 64)

for i in "${!layers[@]}"
do
	n=${layers[i]}
	ntsk=${ntasks[i]}
	
	rsync -av --delete template/ $n
	cp ./lam/cube-N$n.lam $n/
	
	cd $n

	sed -i "s/NLAYERS/$n/g" in.deform
	sed -i "s/NLAYERS/$n/g" lammps-slurm.sb
	sed -i "s/NTASKS/$ntsk/g" lammps-slurm.sb

	sbatch lammps-slurm.sb

	cd ..	
done