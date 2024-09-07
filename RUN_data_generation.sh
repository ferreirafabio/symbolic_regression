









sbatch -p bosch_cpu-cascadelake -nodes=1 -time=1:00:00
srun -p bosch_cpu-cascadelake --nodes=1 --time=1:00:00  --bosch --pty bash