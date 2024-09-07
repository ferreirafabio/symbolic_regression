




pip install antlr4-python3-runtime==4.11




sbatch -p bosch_cpu-cascadelake -nodes=1 -time=4:00:00
srun -p bosch_cpu-cascadelake --nodes=1 --time=1:00:00 --cpus-per-task=32 --bosch --pty bash

/home/frankej/workspace/ScalingSymbolicRegression/venv/bin/python gpr/data/data_creator.py -f -c feynman_arc_config data_dir="/mhome/frankej/workspace/data"

/home/frankej/workspace/ScalingSymbolicRegression/venv/bin/python gpr/data/data_creator.py -f -c feynman_arc_config data_dir="/mhome/frankej/workspace/data"



sbatch --time=1:00:00 slurm_data_gen_kislurm.sh -c feynman_arc_config data_dir="/mhome/frankej/workspace/test_data"