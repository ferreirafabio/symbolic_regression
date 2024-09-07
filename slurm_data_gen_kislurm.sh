#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH -p bosch_cpu-cascadelake
#SBATCH --cpus-per-task=32
#SBATCH --job-name=gpr_d
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/frankej/experiments/symreg/output/mpi-out.%j
#SBATCH --error=/home/frankej/experiments/symreg/error/mpi-err.%j

cuda12.1

echo "KISLURM Job"
source /home/frankej/workspace/ScalingSymbolicRegression/venv/bin/activate

PYTHON_SCRIPT=/home/frankej/workspace/ScalingSymbolicRegression/gpr/data/data_creator.py

srun --bosch python $PYTHON_SCRIPT -f $@

